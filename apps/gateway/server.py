"""LLM coding-router server.

Exposes one OpenAI-compatible model ("auto") on explicit Chat Completions
and Responses endpoints. Requests are routed by the Supra-Router-51M complexity
gate.
Responses requests are restricted to OpenAI models via OpenRouter;
Chat Completions behavior remains independent.

Config via env:
  ROUTELLM_HOST=127.0.0.1
  ROUTELLM_PORT=5500
  ROUTELLM_KEY=sk-route-local          # bearer token clients must present
  EXPENSIVE_BASE=https://openrouter.ai/api/v1
  EXPENSIVE_KEY=...
  CHEAP_BASE=https://opencode.ai/zen/go/v1
  CHEAP_KEY=...
  EXPENSIVE_MODEL=openai/gpt-5.6-sol
  CHEAP_MODEL=deepseek-v4-flash
  LOG_FILE=~/.local/share/mantis/router/decisions.log
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
import tomllib
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Defaults mirror the router launcher (canonical source, calibrated there) —
# keep in sync so bare `python server.py` behaves identically to the launcher.
_DEFAULT_AI_ROUTING_CONFIG = Path.home() / ".config" / "ai-routing" / "catalog.toml"
# A legacy raw environment value must never block an explicit catalog/JSON
# source from becoming authoritative. Detect that condition before any legacy
# environment parsing so malformed legacy values degrade to defaults instead of
# failing startup for an independent source deployment.
_EXPLICIT_SOURCE_PRESENT = (
    "ROUTELLM_TARGETS_JSON" in os.environ
    or "AI_ROUTING_CONFIG" in os.environ
    or _DEFAULT_AI_ROUTING_CONFIG.exists()
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        if _EXPLICIT_SOURCE_PRESENT:
            return default
        raise


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or not value:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if _EXPLICIT_SOURCE_PRESENT:
            return default
        raise


HOST = os.environ.get("ROUTELLM_HOST", "127.0.0.1")
PORT = _env_int("ROUTELLM_PORT", 5500)
SERVER_KEY = os.environ.get("ROUTELLM_KEY", "sk-route-local")
ROUTELLM_CONTEXT_WINDOW = os.environ.get("ROUTELLM_CONTEXT_WINDOW", "auto")
# Floor for targets without an explicit catalog max_tokens; the catalog
# carries exact per-model output caps (models.dev): azure gpt-5.6 = 128000,
# deepseek v4 flash = 384000.
ROUTELLM_MAX_TOKENS = _env_int("ROUTELLM_MAX_TOKENS", 131072)
MODEL_ID = "auto"

def _base(name: str, direct_default: str) -> str:
    return os.environ.get(name) or direct_default


def _key(name: str) -> str:
    return os.environ.get(name, "")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if not value:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if _EXPLICIT_SOURCE_PRESENT:
            return None
        raise


_TARGET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The shared provider catalog includes every protocol supported by a catalog
# consumer. RouteLLM targets use a smaller subset; see
# _routellm_target_protocols below.
_CATALOG_PROTOCOLS = frozenset({"chat_completions", "responses", "anthropic_messages"})
# Adapters are part of the transport contract.  They are deliberately explicit
# rather than inferred from a hostname: a provider endpoint can move without
# changing the provider behavior it expects from this router.
_ADAPTER_PROTOCOLS = {
    "openrouter": frozenset({"chat_completions", "responses"}),
    "opencode-go": frozenset({"chat_completions"}),
    "modal": frozenset({"chat_completions"}),
    "openai-compatible": frozenset({"chat_completions", "responses"}),
    # Mantis may use Bifrost's native Anthropic endpoint. RouteLLM does not
    # route this protocol, but must accept its provider in the shared catalog.
    "anthropic": frozenset({"anthropic_messages"}),
}
_LITERAL_CREDENTIAL_FIELDS = frozenset({
    "key", "api_key", "token", "credential", "credential_value", "secret", "password",
})
_AI_ROUTING_CONFIG_EXPLICIT = "AI_ROUTING_CONFIG" in os.environ
AI_ROUTING_CONFIG = Path(os.environ.get("AI_ROUTING_CONFIG", str(_DEFAULT_AI_ROUTING_CONFIG))).expanduser()


def _config_error(message: str) -> ValueError:
    """Build a configuration error without ever including secret values."""
    return ValueError(f"invalid AI routing configuration: {message}")


def _catalog_mapping(value, field: str) -> dict:
    if not isinstance(value, dict):
        raise _config_error(f"{field} must be a table/object")
    return value


def _ensure_fields(spec: dict, allowed: set[str] | frozenset[str], field: str) -> None:
    unknown = sorted(set(spec) - set(allowed))
    if unknown:
        raise _config_error(f"{field} has unexpected field {unknown[0]}")


def _reject_literal_credentials(spec: dict, field: str) -> None:
    for name in _LITERAL_CREDENTIAL_FIELDS:
        if name in spec:
            raise _config_error(f"{field} must use credential_env, not {name}")
    if "model" in spec:
        raise _config_error(f"{field} must use upstream_model, not model")


def _is_catalog_version_one(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 1


def _valid_target_id(value, field: str) -> str:
    if not isinstance(value, str) or not _TARGET_ID_RE.fullmatch(value):
        raise _config_error(f"{field} must be an identifier")
    return value


def _valid_credential_env(value, field: str) -> str:
    if not isinstance(value, str) or not _ENV_NAME_RE.fullmatch(value):
        raise _config_error(f"{field} must name an environment variable")
    return value


def _valid_text(value, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _config_error(f"{field} must be a non-empty trimmed string")
    return value


def _valid_base_url(value, field: str) -> str:
    raw = _valid_text(value, field)
    try:
        parsed = urlsplit(raw)
        # Accessing .port makes urllib reject malformed and out-of-range ports
        # during configuration, rather than much later in HTTPX. Never chain
        # the parser exception: it can echo the offending value.
        _ = parsed.port
    except ValueError:
        raise _config_error(f"{field} must be a valid URL") from None
    if (parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise _config_error(f"{field} must be an http(s) base URL without credentials")
    return raw.rstrip("/")


def _valid_optional_positive_int(value, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _config_error(f"{field} must be a positive integer")
    return value


def _valid_bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise _config_error(f"{field} must be a boolean")
    return value


def _catalog_protocols(value, field: str, *, required: bool = True) -> tuple[str, ...]:
    """Validate protocols that may occur anywhere in the shared catalog."""
    if value is None and not required:
        return ()
    if not isinstance(value, list) or not value:
        raise _config_error(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or item not in _CATALOG_PROTOCOLS for item in value):
        raise _config_error(f"{field} contains an unsupported protocol")
    if len(set(value)) != len(value):
        raise _config_error(f"{field} must not contain duplicates")
    return tuple(value)


def _routellm_target_protocols(value, field: str, *, required: bool = True) -> tuple[str, ...]:
    """Validate RouteLLM target protocols, excluding Anthropic Messages."""
    protocols = _catalog_protocols(value, field, required=required)
    if "anthropic_messages" in protocols:
        raise _config_error(f"{field} cannot declare anthropic_messages for a RouteLLM target")
    return protocols


def _valid_adapter(value, field: str) -> str:
    if not isinstance(value, str) or value not in _ADAPTER_PROTOCOLS:
        raise _config_error(f"{field} is unsupported")
    return value


def _valid_developer_role(value, field: str) -> str:
    if not isinstance(value, str) or value not in {"native", "system"}:
        raise _config_error(f"{field} must be 'native' or 'system'")
    return value


def _target_fallbacks(value, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _config_error(f"{field} must be a list")
    fallbacks = tuple(_valid_target_id(item, f"{field} entry") for item in value)
    if len(set(fallbacks)) != len(fallbacks):
        raise _config_error(f"{field} must not contain duplicates")
    return fallbacks


def _valid_target_rank(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _config_error(f"{field} must be a non-negative integer")
    return value


def _resolve_credential(credential_env: str, environ: dict[str, str] | None = None) -> str:
    # Keep the resolved value only in the in-memory backend dict. It is never
    # returned by config helpers or logged.
    environ = os.environ if environ is None else environ
    value = environ.get(credential_env)
    if not isinstance(value, str) or not value:
        raise _config_error(f"credential environment variable {credential_env} is not set")
    return value


def _backend(name: str, *, base: str, key: str, model: str, effort: str,
             max_tokens: int | None = None, usage_include: bool | None = None,
             provider: str | None = None, credential_env: str | None = None,
             adapter: str | None = None, protocols: tuple[str, ...] | None = None,
             fallbacks: tuple[str, ...] = (), rank: int | None = None,
             force_reasoning_effort: bool = False, developer_role: str = "system") -> dict:
    # `usage.include` is an OpenRouter wire extension.  Adapter identity is
    # therefore operationally meaningful (and participates in revisioning),
    # rather than merely descriptive metadata.
    if adapter == "openrouter":
        default_usage = True
    elif adapter is not None:
        default_usage = False
    else:
        # Preserve legacy endpoint-derived behavior while no catalog is active.
        default_usage = "openrouter.ai" in base
    usage_name = re.sub(r"[^A-Z0-9]", "_", name.upper())
    usage_name = f"ROUTELLM_{usage_name}_USAGE_INCLUDE"
    return {
        # `tier` remains for old telemetry/tests; target is the stable generic
        # identifier exposed in headers and logs for catalog-backed routing.
        "tier": name, "target": name, "base": base, "key": key, "model": model,
        "effort": effort, "max_tokens": max_tokens, "provider": provider,
        "credential_env": credential_env, "adapter": adapter, "protocols": protocols,
        "fallbacks": fallbacks, "rank": rank,
        "force_reasoning_effort": force_reasoning_effort,
        "developer_role": developer_role,
        "usage_include": (_env_bool(usage_name, default_usage)
                           if usage_include is None else usage_include),
    }


# The shared catalog has one root namespace for provider bindings plus its two
# approved consumers.  Keeping this closed makes an accidental top-level table
# fail at startup instead of silently falling back to legacy router behavior.
_CATALOG_ROOT_FIELDS = frozenset({"version", "providers", "routellm", "mantis"})
_PROVIDER_FIELDS = frozenset({"adapter", "base_url", "credential_env", "developer_role", "protocols"})
_TARGET_FIELDS = frozenset({
    "provider", "adapter", "base_url", "credential_env", "developer_role",
    "upstream_model", "reasoning_effort", "max_tokens", "protocols", "fallbacks", "rank",
    "force_reasoning_effort", "usage_include",
})
_ROUTELLM_FIELDS = frozenset({"active_policy", "revision", "invalid_complexity_target", "targets", "policies"})
_POLICY_FIELDS = frozenset({"complexity_targets", "revision", "invalid_complexity_target"})


def _parse_provider_specs(value, *, field: str) -> dict[str, dict]:
    if value is None:
        return {}
    providers = _catalog_mapping(value, field)
    result = {}
    for provider_id, raw in providers.items():
        provider_id = _valid_target_id(provider_id, f"{field} provider id")
        spec = _catalog_mapping(raw, f"{field}.{provider_id}")
        # Detect prohibited credential aliases before generic closed-schema
        # reporting so neither a field nor its value can be mistaken for a
        # supported secret-bearing configuration path.
        _reject_literal_credentials(spec, f"{field}.{provider_id}")
        _ensure_fields(spec, _PROVIDER_FIELDS, f"{field}.{provider_id}")
        adapter = _valid_adapter(spec.get("adapter"), f"{field}.{provider_id}.adapter")
        provider_protocols = _catalog_protocols(
            spec.get("protocols"), f"{field}.{provider_id}.protocols", required=False)
        if provider_protocols and not set(provider_protocols) <= _ADAPTER_PROTOCOLS[adapter]:
            raise _config_error(f"{field}.{provider_id}.protocols exceeds adapter capabilities")
        result[provider_id] = {
            "adapter": adapter,
            "base_url": _valid_base_url(spec.get("base_url"), f"{field}.{provider_id}.base_url"),
            "credential_env": _valid_credential_env(
                spec.get("credential_env"), f"{field}.{provider_id}.credential_env"),
            "developer_role": _valid_developer_role(
                spec.get("developer_role", "system"), f"{field}.{provider_id}.developer_role"),
            "protocols": provider_protocols,
        }
    return result


def _build_target_registry(target_specs, providers: dict[str, dict], *, require_provider: bool,
                           environ: dict[str, str] | None = None,
                           field: str = "routellm.targets") -> dict[str, dict]:
    """Validate non-secret target specs and resolve credential_env at startup."""
    target_specs = _catalog_mapping(target_specs, field)
    if not target_specs:
        raise _config_error(f"{field} must define at least one target")
    targets = {}
    for target_id, raw in target_specs.items():
        target_id = _valid_target_id(target_id, f"{field} target id")
        spec = _catalog_mapping(raw, f"{field}.{target_id}")
        _reject_literal_credentials(spec, f"{field}.{target_id}")
        _ensure_fields(spec, _TARGET_FIELDS, f"{field}.{target_id}")
        provider_id = spec.get("provider")
        direct_fields = {"adapter", "base_url", "credential_env", "developer_role"} & set(spec)
        if provider_id is not None:
            if not isinstance(provider_id, str) or provider_id not in providers:
                raise _config_error(f"{field}.{target_id}.provider is unknown")
            if direct_fields:
                raise _config_error(
                    f"{field}.{target_id} must use provider or direct binding fields, not both")
            binding = providers[provider_id]
        else:
            if require_provider:
                raise _config_error(f"{field}.{target_id}.provider is required")
            if not direct_fields:
                raise _config_error(f"{field}.{target_id} needs provider or direct binding fields")
            adapter = _valid_adapter(spec.get("adapter"), f"{field}.{target_id}.adapter")
            binding = {
                "adapter": adapter,
                "base_url": _valid_base_url(spec.get("base_url"), f"{field}.{target_id}.base_url"),
                "credential_env": _valid_credential_env(
                    spec.get("credential_env"), f"{field}.{target_id}.credential_env"),
                "developer_role": _valid_developer_role(
                    spec.get("developer_role", "system"), f"{field}.{target_id}.developer_role"),
                "protocols": (),
            }
            provider_id = None
        model = _valid_text(spec.get("upstream_model"), f"{field}.{target_id}.upstream_model")
        effort = spec.get("reasoning_effort")
        if effort is None:
            effort = ""
        else:
            effort = _valid_text(effort, f"{field}.{target_id}.reasoning_effort")
        max_tokens = _valid_optional_positive_int(
            spec.get("max_tokens"), f"{field}.{target_id}.max_tokens")
        protocols = _routellm_target_protocols(
            spec.get("protocols"), f"{field}.{target_id}.protocols")
        if not set(protocols) <= _ADAPTER_PROTOCOLS[binding["adapter"]]:
            raise _config_error(f"{field}.{target_id}.protocols exceeds adapter capabilities")
        if binding["protocols"] and not set(protocols) <= set(binding["protocols"]):
            raise _config_error(f"{field}.{target_id}.protocols exceeds provider protocols")
        fallbacks = _target_fallbacks(spec.get("fallbacks"), f"{field}.{target_id}.fallbacks")
        if target_id in fallbacks:
            raise _config_error(f"{field}.{target_id}.fallbacks must not contain itself")
        # Rank is optional and defaults to stable declaration order via
        # _configured_rank; validate only when explicitly supplied.
        rank = _valid_target_rank(spec["rank"], f"{field}.{target_id}.rank") if "rank" in spec else None
        force_reasoning_effort = _valid_bool(
            spec.get("force_reasoning_effort", False),
            f"{field}.{target_id}.force_reasoning_effort",
        )
        if force_reasoning_effort and not effort:
            raise _config_error(f"{field}.{target_id}.force_reasoning_effort requires reasoning_effort")
        usage_include = spec.get("usage_include")
        if usage_include is not None:
            usage_include = _valid_bool(usage_include, f"{field}.{target_id}.usage_include")
        credential_env = binding["credential_env"]
        targets[target_id] = _backend(
            target_id, base=binding["base_url"], key=_resolve_credential(credential_env, environ),
            model=model, effort=effort, max_tokens=max_tokens, provider=provider_id,
            credential_env=credential_env, adapter=binding["adapter"], protocols=protocols,
            fallbacks=fallbacks, rank=rank, force_reasoning_effort=force_reasoning_effort,
            developer_role=binding["developer_role"], usage_include=usage_include,
        )
    for target_id, backend in targets.items():
        for fallback in backend["fallbacks"]:
            if fallback not in targets:
                raise _config_error(f"{field}.{target_id}.fallbacks references unknown target {fallback}")
    return targets


def _read_catalog(path: Path, *, explicit: bool) -> dict:
    if not path.exists():
        if explicit:
            raise _config_error("AI_ROUTING_CONFIG does not exist")
        return {}
    if not path.is_file():
        raise _config_error("AI_ROUTING_CONFIG must name a regular file")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        # A catalog is private configuration. Do not surface parser/decoder
        # detail (which can echo offending source) through startup diagnostics.
        raise _config_error("AI_ROUTING_CONFIG could not be read") from None
    if not isinstance(data, dict):
        raise _config_error("AI_ROUTING_CONFIG must contain a TOML table")
    _ensure_fields(data, _CATALOG_ROOT_FIELDS, "catalog")
    return data


def _catalog_routellm_spec(catalog: dict) -> dict | None:
    value = catalog.get("routellm")
    if value is None:
        return None
    if not _is_catalog_version_one(catalog.get("version")):
        raise _config_error("catalog version must be 1 when routellm is configured")
    routellm = _catalog_mapping(value, "routellm")
    _ensure_fields(routellm, _ROUTELLM_FIELDS, "routellm")
    targets = _catalog_mapping(routellm.get("targets"), "routellm.targets")
    policies = _catalog_mapping(routellm.get("policies"), "routellm.policies")
    active_policy = routellm.get("active_policy")
    if not isinstance(active_policy, str) or active_policy not in policies:
        raise _config_error("routellm.active_policy must name a configured policy")
    # Every policy table is closed-schema validated, not only the active one,
    # so a dormant policy cannot smuggle typos or invalid revision types.
    for policy_id, raw_policy in policies.items():
        policy = _catalog_mapping(raw_policy, f"routellm.policies.{policy_id}")
        _ensure_fields(policy, _POLICY_FIELDS, f"routellm.policies.{policy_id}")
        if "revision" in policy:
            _explicit_revision(policy["revision"], f"routellm.policies.{policy_id}.revision")
    policy = _catalog_mapping(policies[active_policy], f"routellm.policies.{active_policy}")
    return {
        "targets": targets,
        "policy_targets": policy.get("complexity_targets"),
        "revision": policy.get("revision", routellm.get("revision")),
        "invalid_target": policy.get("invalid_complexity_target", routellm.get("invalid_complexity_target")),
    }


def _reject_duplicate_json_keys(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _read_targets_json(raw: str | None) -> dict | None:
    if raw is None:
        return None
    try:
        data = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
    except (TypeError, ValueError):
        # Covers JSONDecodeError and duplicate-key rejection. Never surface
        # parser detail, which could echo source values.
        raise _config_error("ROUTELLM_TARGETS_JSON must be valid JSON") from None
    if not isinstance(data, dict) or not data:
        raise _config_error("ROUTELLM_TARGETS_JSON must be a non-empty object")
    wrapper_fields = {
        "version", "targets", "providers", "complexity_targets", "revision",
        "invalid_complexity_target",
    }
    if "targets" in data:
        # The wrapper form is unambiguous and versioned exactly like a catalog.
        _ensure_fields(data, wrapper_fields, "ROUTELLM_TARGETS_JSON wrapper")
        if not _is_catalog_version_one(data.get("version")):
            raise _config_error("ROUTELLM_TARGETS_JSON.version must be 1")
        return {
            "targets": data["targets"],
            "providers": data.get("providers", {}),
            "policy_targets": data.get("complexity_targets"),
            "revision": data.get("revision"),
            "invalid_target": data.get("invalid_complexity_target"),
        }
    # A plain target-id -> target-spec mapping is convenient for a renderer
    # that supplies ROUTELLM_SUPRA_TARGETS separately. It is atomic too: it
    # cannot inherit provider definitions from an unrelated catalog. Wrapper
    # keys are reserved so no plain map can be misread as a wrapper.
    reserved = sorted(wrapper_fields - {"targets"} & set(data))
    if reserved:
        raise _config_error(f"ROUTELLM_TARGETS_JSON target id {reserved[0]} is reserved")
    return {"targets": data, "providers": {}, "policy_targets": None,
            "revision": None, "invalid_target": None}


def _parse_complexity_targets(value, known: dict[str, dict], field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",")]
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        raise _config_error(f"{field} must contain exactly five target IDs")
    targets = tuple(_valid_target_id(item, f"{field} entry") for item in value)
    unknown = [target for target in targets if target not in known]
    if unknown:
        raise _config_error(f"{field} references unknown target {unknown[0]}")
    if any(not str(known[target].get("base", "")) for target in targets):
        raise _config_error(f"{field} references an unconfigured target")
    return targets


def _explicit_revision(value, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _TARGET_ID_RE.fullmatch(value):
        raise _config_error(f"{field} must be an identifier")
    return value


def _configured_rank(backend: dict, index: int) -> int:
    rank = backend.get("rank")
    return rank if isinstance(rank, int) and not isinstance(rank, bool) else index


def _configured_safe_target(targets: dict[str, dict]) -> str:
    # Stable equal-rank policy: earliest declaration wins.  TOML and JSON
    # preserve mapping order, so changing an ID does not accidentally alter a
    # tie unless the author intentionally changes declaration order.
    return max(targets, key=lambda target: (
        _configured_rank(targets[target], list(targets).index(target)),
        -list(targets).index(target),
    ))


def _target_config_fingerprint(source: str, targets: dict[str, dict],
                               complexity_targets: tuple[str, ...], invalid_target: str) -> str:
    # Never include backend["key"].  Every routing/wire-affecting, non-secret
    # binding and policy field is included so a label cannot conceal a stale
    # session, learned pin, or replay cache entry.
    target_order = list(targets)
    safe_targets = {
        target_id: {
            "base": backend.get("base"), "provider": backend.get("provider"),
            "credential_env": backend.get("credential_env"), "adapter": backend.get("adapter"),
            "developer_role": backend.get("developer_role"), "model": backend.get("model"),
            "effort": backend.get("effort"), "protocols": backend.get("protocols"),
            "fallbacks": backend.get("fallbacks"), "rank": backend.get("rank"),
            "max_tokens": backend.get("max_tokens"),
            "force_reasoning_effort": backend.get("force_reasoning_effort"),
            "usage_include": backend.get("usage_include"),
        }
        for target_id, backend in sorted(targets.items())
    }
    payload = json.dumps({
        "source": source, "target_order": target_order, "targets": safe_targets,
        "complexity_targets": complexity_targets, "invalid_target": invalid_target,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _target_config_revision(source: str, explicit: str | None, fingerprint: str) -> str:
    label = _explicit_revision(explicit, "target revision") or source
    return f"{label}-{fingerprint}"


EXPENSIVE_BASE = _base("EXPENSIVE_BASE", "https://openrouter.ai/api/v1")
CHEAP_BASE = _base("CHEAP_BASE", "https://opencode.ai/zen/go/v1")
MIDDLE_BASE = _base("MIDDLE_BASE", "")
EXPENSIVE = _backend(
    "expensive", base=EXPENSIVE_BASE, key=_key("EXPENSIVE_KEY"),
    model=os.environ.get("EXPENSIVE_MODEL", "openai/gpt-5.6-sol"),
    effort=os.environ.get("EXPENSIVE_REASONING_EFFORT", "medium"),
    max_tokens=_optional_int("EXPENSIVE_MAX_TOKENS"), rank=2,
)
CHEAP = _backend(
    "cheap", base=CHEAP_BASE, key=_key("CHEAP_KEY"),
    model=os.environ.get("CHEAP_MODEL", "deepseek-v4-flash"),
    effort=os.environ.get("CHEAP_REASONING_EFFORT", "none"),
    max_tokens=_env_int("CHEAP_MAX_TOKENS", ROUTELLM_MAX_TOKENS), rank=0,
)
MIDDLE = _backend(
    "middle", base=MIDDLE_BASE, key=_key("MIDDLE_KEY"),
    model=os.environ.get("MIDDLE_MODEL", "openai/gpt-5.6-terra"),
    effort=os.environ.get("MIDDLE_REASONING_EFFORT", "max"),
    max_tokens=_env_int("MIDDLE_MAX_TOKENS", ROUTELLM_MAX_TOKENS), rank=1,
    force_reasoning_effort=True,
)
_LEGACY_BACKENDS = {"cheap": CHEAP, "middle": MIDDLE, "expensive": EXPENSIVE}
MIDDLE_MIN_COMPLEXITY = _env_int("ROUTELLM_MIDDLE_MIN_COMPLEXITY", 3)
MIDDLE_CONFIGURED = bool(MIDDLE["base"])
EXPENSIVE_MIN_COMPLEXITY = _env_int(
    "ROUTELLM_EXPENSIVE_MIN_COMPLEXITY",
    # With Terra enabled, reserve Sol for Supra's highest complexity level.
    # Direct two-tier mode retains the historical threshold immediately above
    # the Supra cutoff.
    5 if MIDDLE_CONFIGURED else 3,
)


def _legacy_complexity_targets() -> tuple[str, ...]:
    """Translate threshold-era config to a stable five-level target map."""
    return tuple(
        "expensive" if level >= EXPENSIVE_MIN_COMPLEXITY else
        "middle" if MIDDLE_CONFIGURED and level >= MIDDLE_MIN_COMPLEXITY else
        "cheap"
        for level in range(1, 6)
    )


# JSON is a complete renderer/source override.  Parse it before touching the
# optional catalog so an independent JSON deployment cannot inherit, depend on,
# or be rejected by an unrelated catalog binding.
_json_routellm = _read_targets_json(os.environ.get("ROUTELLM_TARGETS_JSON"))
if _json_routellm is None:
    _catalog = _read_catalog(AI_ROUTING_CONFIG, explicit=_AI_ROUTING_CONFIG_EXPLICIT)
    _catalog_routellm = _catalog_routellm_spec(_catalog)
else:
    _catalog = {}
    _catalog_routellm = None
_TARGETS_ARE_EXPLICIT = _json_routellm is not None or _catalog_routellm is not None
if _json_routellm is not None:
    # JSON is an atomic renderer/source override.  In particular, it must not
    # silently reuse a catalog provider with a similarly named ID.
    _json_providers = _parse_provider_specs(
        _json_routellm["providers"], field="ROUTELLM_TARGETS_JSON.providers")
    BACKENDS = _build_target_registry(
        _json_routellm["targets"], _json_providers, require_provider=False,
        field="ROUTELLM_TARGETS_JSON.targets")
    _policy_targets = _json_routellm["policy_targets"]
    _config_revision = _json_routellm["revision"]
    _invalid_target = _json_routellm["invalid_target"]
    TARGET_CONFIG_SOURCE = "json"
elif _catalog_routellm is not None:
    _catalog_providers = _parse_provider_specs(_catalog.get("providers"), field="providers")
    BACKENDS = _build_target_registry(
        _catalog_routellm["targets"], _catalog_providers, require_provider=True,
        field="routellm.targets")
    _policy_targets = _catalog_routellm["policy_targets"]
    _config_revision = _catalog_routellm["revision"]
    _invalid_target = _catalog_routellm["invalid_target"]
    TARGET_CONFIG_SOURCE = "catalog"
else:
    BACKENDS = _LEGACY_BACKENDS
    _policy_targets = _legacy_complexity_targets()
    _config_revision = "legacy"
    _invalid_target = None
    TARGET_CONFIG_SOURCE = "legacy"

# Legacy raw policy overrides remain useful for inactive deployments and a
# plain JSON target map, but must never silently supersede an active catalog or
# an explicit JSON policy. A full ROUTELLM_TARGETS_JSON source is itself an
# intentional source replacement; its declared policy is authoritative too.
_POLICY_IS_AUTHORITATIVE = (
    _catalog_routellm is not None
    or (_json_routellm is not None and _json_routellm["policy_targets"] is not None)
)
if "ROUTELLM_SUPRA_TARGETS" in os.environ:
    if _POLICY_IS_AUTHORITATIVE:
        raise _config_error("ROUTELLM_SUPRA_TARGETS cannot override an active target policy")
    SUPRA_TARGETS = _parse_complexity_targets(
        os.environ["ROUTELLM_SUPRA_TARGETS"], BACKENDS, "ROUTELLM_SUPRA_TARGETS")
    _SUPRA_TARGETS_FROM_ENV = True
else:
    SUPRA_TARGETS = _parse_complexity_targets(_policy_targets, BACKENDS, "complexity_targets")
    _SUPRA_TARGETS_FROM_ENV = False
if "ROUTELLM_SUPRA_INVALID_TARGET" in os.environ:
    if _POLICY_IS_AUTHORITATIVE:
        raise _config_error("ROUTELLM_SUPRA_INVALID_TARGET cannot override an active target policy")
    _invalid_target = os.environ["ROUTELLM_SUPRA_INVALID_TARGET"]
if _invalid_target is not None:
    SUPRA_INVALID_TARGET = _valid_target_id(_invalid_target, "invalid complexity target")
    if SUPRA_INVALID_TARGET not in BACKENDS:
        raise _config_error("invalid complexity target is unknown")
else:
    SUPRA_INVALID_TARGET = _configured_safe_target(BACKENDS)

# Public compatibility aliases: BACKENDS is now the generic target registry.
TARGETS = BACKENDS
TIER_ORDER = {
    target: _configured_rank(backend, index)
    for index, (target, backend) in enumerate(BACKENDS.items())
}
TARGET_CONFIG_FINGERPRINT = _target_config_fingerprint(
    TARGET_CONFIG_SOURCE, BACKENDS, SUPRA_TARGETS, SUPRA_INVALID_TARGET)
TARGET_CONFIG_REVISION = _target_config_revision(
    TARGET_CONFIG_SOURCE,
    os.environ.get("ROUTELLM_TARGETS_REVISION", _config_revision),
    TARGET_CONFIG_FINGERPRINT,
)


# One async pool is created and closed by the ASGI lifespan.
TIMEOUT_S = _env_float("ROUTELLM_TIMEOUT_S", 600.0)
_client: httpx.AsyncClient | None = None
RETRY_STATUSES = {429, 500, 502, 503, 504}

# Reject oversized bodies before they are buffered into memory (413).
MAX_BODY_BYTES = _env_int("ROUTELLM_MAX_BODY_BYTES", 50 * 1024 * 1024)

DATA_DIR = Path(os.environ.get("MANTIS_DATA_DIR", str(Path.home()/".local/share/mantis")))
LOG_PATH = Path(os.environ.get("LOG_FILE", str(DATA_DIR/"router/decisions.log")))
TRAINING_LOG_ENABLED = os.environ.get("ROUTELLM_TRAINING_LOG", "0").lower() in {"1", "true", "yes", "on"}
TRAINING_LOG_PATH = Path(os.environ.get("TRAINING_LOG_FILE", str(DATA_DIR/"router/training.jsonl")))
OUTCOME_LOG_PATH = Path(os.environ.get("OUTCOME_LOG_FILE", str(DATA_DIR/"router/outcomes.jsonl")))
# Same prompt re-sent within this window usually means the previous route failed.
RETRY_WINDOW_S = _env_float("ROUTELLM_RETRY_WINDOW_S", 900.0)
LOG_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
LOG_PATH.parent.chmod(0o700)
if TRAINING_LOG_ENABLED:
    TRAINING_LOG_PATH.touch(mode=0o600, exist_ok=True)
    TRAINING_LOG_PATH.chmod(0o600)

_cached_context_window = None


async def _fetch_model_context_length(base: str, key: str, model_id: str) -> int | None:
    """Discover capability without blocking the event loop."""
    if _client is None:
        return None
    try:
        url = ("https://openrouter.ai/api/v1/models" if "openrouter.ai" in base
               else base.rstrip("/") + "/models")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        resp = await _client.get(url, headers=headers, timeout=5.0)
        if resp.status_code != 200:
            return None
        for item in resp.json().get("data", []):
            item_id = item.get("id") or item.get("model_name")
            # Providers can return either a bare name or provider/name.
            if item_id == model_id or str(item_id).rsplit("/", 1)[-1] == model_id.rsplit("/", 1)[-1]:
                value = item.get("context_window") or item.get("context_length")
                if isinstance(value, int) and value > 0:
                    return value
    except (httpx.HTTPError, ValueError, TypeError):
        return None
    return None


async def _get_context_window() -> int:
    global _cached_context_window
    if _cached_context_window is not None:
        return _cached_context_window
    env_val = os.environ.get("ROUTELLM_CONTEXT_WINDOW")
    if env_val and env_val.isdigit() and int(env_val) > 0:
        _cached_context_window = int(env_val)
        return _cached_context_window
    # Generic targets may share a binding. Query each distinct endpoint/model
    # once, rather than assuming the old expensive/cheap/middle names exist.
    backends = []
    seen = set()
    for backend in BACKENDS.values():
        identity = (backend.get("base"), backend.get("model"))
        if identity not in seen:
            seen.add(identity)
            backends.append(backend)
    values = await asyncio.gather(*(
        _fetch_model_context_length(b["base"], b["key"], b["model"])
        for b in backends
    ))
    valid = [value for value in values if isinstance(value, int) and value > 0]
    _cached_context_window = min(valid) if valid else 1_000_000
    return _cached_context_window


def _extract_prompt(body: dict) -> str:
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                # text parts only
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
            return str(c)
    return ""


def _extract_responses_prompt(body: dict) -> str:
    value = body.get("input")
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""

    def item_text(item: dict) -> list[str]:
        content = item.get("content")
        if isinstance(content, str):
            return [content]
        if not isinstance(content, list):
            return []
        return [
            part["text"] for part in content
            if isinstance(part, dict) and part.get("type") in {"input_text", "text"}
            and isinstance(part.get("text"), str)
        ]

    for item in reversed(value):
        if isinstance(item, dict) and item.get("role") == "user":
            user_texts = item_text(item)
            if user_texts:
                return " ".join(user_texts)
    return " ".join(text for item in value if isinstance(item, dict) for text in item_text(item))


_supra_model = None
_supra_tokenizer = None


def _load_supra():
    global _supra_model, _supra_tokenizer
    if _supra_model is not None:
        return _supra_model, _supra_tokenizer
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _supra_tokenizer = AutoTokenizer.from_pretrained("SupraLabs/Supra-Router-51M")
    _supra_model = AutoModelForCausalLM.from_pretrained(
        "SupraLabs/Supra-Router-51M", dtype=torch.float32,
    )
    _supra_model.eval()
    return _supra_model, _supra_tokenizer


def _parse_supra_complexity(text: str) -> int:
    for part in text.split("|"):
        part = part.strip()
        if part.lower().startswith("complexity:"):
            try:
                return int(part.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                return 0
    return 0


def _supra_complexity(prompt: str) -> tuple[int, int]:
    model, tokenizer = _load_supra()
    from transformers import StoppingCriteria

    class _ComplexitySeen(StoppingCriteria):
        """Stop generation as soon as the 'Complexity:' field is emitted.

        Supra emits 'Domain: ... | Complexity: N | ...' and the complexity
        digit appears within the first ~10 generated tokens. Greedy decode is
        deterministic, so stopping early yields the exact same parsed value
        while cutting median inference from ~480ms to ~160ms (3x)."""

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            text = tokenizer.decode(input_ids[0][-24:], skip_special_tokens=True)
            # Stop only once the complexity digit itself has been emitted;
            # stopping at the bare "Complexity:" prefix would parse as 0.
            return re.search(r"Complexity:\s*\d", text) is not None

    fmt = f"Task: {prompt}\nAnalysis: "
    inputs = tokenizer(fmt, return_tensors="pt", truncation=True,
                       max_length=tokenizer.model_max_length)
    import torch
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=128, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[_ComplexitySeen()],
        )
    supra_ms = int((time.time() - t0) * 1000)
    gen = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()
    return _parse_supra_complexity(gen), supra_ms


def _target_order(target: str) -> int:
    try:
        return list(BACKENDS).index(target)
    except ValueError:
        return len(BACKENDS)


def _target_rank(target: str) -> int:
    backend = BACKENDS.get(target)
    if backend is None:
        return -1
    rank = backend.get("rank")
    return rank if isinstance(rank, int) and not isinstance(rank, bool) else TIER_ORDER.get(target, -1)


def _safe_target() -> str:
    """Return the highest-ranked target with deterministic equal-rank handling."""
    return max(BACKENDS, key=lambda target: (_target_rank(target), -_target_order(target)))


def _default_target() -> str:
    """Use the configured level-one policy for no-text/non-routable prompts."""
    targets = _effective_supra_targets()
    return targets[0] if targets else _safe_target()


def _effective_supra_targets() -> tuple[str, ...]:
    if not _TARGETS_ARE_EXPLICIT and not _SUPRA_TARGETS_FROM_ENV:
        return _legacy_complexity_targets()
    return SUPRA_TARGETS


def _target_for_complexity(complexity: int | None) -> tuple[str, str]:
    """Map only Supra's valid 1..5 values; malformed output takes safe target."""
    if isinstance(complexity, int) and not isinstance(complexity, bool) and 1 <= complexity <= 5:
        return _effective_supra_targets()[complexity - 1], "supra_complexity"
    return SUPRA_INVALID_TARGET, "supra_invalid_complexity"


def _supra_reason(complexity: int | None) -> str | None:
    return _target_for_complexity(complexity)[1] if complexity is not None else None


def _fallback_routes(decision: str) -> tuple[str, ...]:
    """Return at most two deduplicated graph attempts for legacy callers."""
    if decision not in BACKENDS:
        decision = _safe_target()
    if not _TARGETS_ARE_EXPLICIT:
        # Preserve the established adjacent legacy graph exactly.
        if not MIDDLE_CONFIGURED:
            return (decision, "cheap" if decision == "expensive" else "expensive")
        if decision == "cheap":
            return ("cheap", "middle")
        if decision == "middle":
            return ("middle", "expensive")
        return ("expensive", "middle")
    routes = [decision]
    for fallback in BACKENDS[decision].get("fallbacks") or ():
        if fallback in BACKENDS and fallback not in routes:
            routes.append(fallback)
        if len(routes) == 2:
            break
    return tuple(routes)


def _declared_fallbacks(target: str) -> tuple[str, ...]:
    """All ordered direct graph edges; actual attempts remain bounded elsewhere."""
    if target not in BACKENDS:
        return ()
    if not _TARGETS_ARE_EXPLICIT:
        return _fallback_routes(target)[1:]
    return tuple(
        fallback for fallback in BACKENDS[target].get("fallbacks") or ()
        if fallback in BACKENDS and fallback != target
    )


def _supports_api(backend: dict, api_format: str) -> bool:
    required = "responses" if api_format == "responses" else "chat_completions"
    adapter = backend.get("adapter")
    if adapter is not None:
        capabilities = _ADAPTER_PROTOCOLS.get(adapter)
        if capabilities is None or required not in capabilities:
            return False
    protocols = backend.get("protocols")
    if protocols is not None:
        return required in protocols
    return api_format != "responses" or _supports_responses(backend)


def _ranked_compatible_targets(decision: str, api_format: str) -> tuple[str, ...]:
    """Quality-preserving candidates, with stable declaration-order ties."""
    if decision not in BACKENDS:
        decision = _safe_target()
    floor = _target_rank(decision)
    return tuple(
        target for target in sorted(BACKENDS, key=lambda item: (_target_rank(item), _target_order(item)))
        if _target_rank(target) >= floor and _supports_api(_backend_for(target), api_format)
    )


def _api_routes(decision: str, api_format: str) -> tuple[str, ...]:
    """Return at most two protocol-compatible attempts.

    A protocol-incompatible decision is promoted to the first compatible target
    at the same-or-higher configured rank.  For the bounded retry, prefer a
    declared compatible fallback.  If no direct edge is usable, look beyond
    that first promotion through the stable rank/declaration-order candidate
    list; this is important when the promoted target is already the only
    compatible direct fallback of the original decision.
    """
    if decision not in BACKENDS:
        decision = _safe_target()
    backend = _backend_for(decision)
    ranked = _ranked_compatible_targets(decision, api_format)
    if _supports_api(backend, api_format):
        primary = decision
    elif ranked:
        primary = ranked[0]
    else:
        return ()

    routes = [primary]
    # Try declared edges from the primary first; they encode the configured
    # bounded failover graph and are intentionally ordered.
    for fallback in _declared_fallbacks(primary):
        if fallback not in routes and _supports_api(_backend_for(fallback), api_format):
            routes.append(fallback)
            return tuple(routes)
    # If promotion skipped an incompatible original target, its declared edges
    # are still meaningful candidates.  This preserves a policy such as
    # chat-only -> responses-a -> responses-b without ever POSTing wrong API.
    if primary != decision:
        for fallback in _declared_fallbacks(decision):
            if fallback not in routes and _supports_api(_backend_for(fallback), api_format):
                routes.append(fallback)
                return tuple(routes)
    # Then use deterministic rank/order candidates, including routes beyond
    # the first compatible promotion.  It can only move equal/higher rank.
    for candidate in ranked:
        if candidate not in routes:
            routes.append(candidate)
            break
    return tuple(routes)


def _chat_routes(decision: str) -> tuple[str, ...]:
    return _api_routes(decision, "chat")


def _responses_routes(decision: str) -> tuple[str, ...]:
    return _api_routes(decision, "responses")


def _responses_tier(decision: str) -> str | None:
    routes = _responses_routes(decision)
    return routes[0] if routes else None


def _chat_tier(decision: str) -> str | None:
    routes = _chat_routes(decision)
    return routes[0] if routes else None


def _decide_uncached(trimmed_prompt: str) -> tuple[str, None, int | None, int | None]:
    global SUPRA_FALLBACK_COUNT
    try:
        complexity, elapsed_ms = _supra_complexity(trimmed_prompt)
        target, _reason = _target_for_complexity(complexity)
        return target, None, complexity, elapsed_ms
    except Exception as err:
        SUPRA_FALLBACK_COUNT += 1
        print(f"Supra scoring failed ({err}); defaulting to safe target", flush=True)
        return _safe_target(), None, None, None


@lru_cache(maxsize=256)
def _decide_cached(trimmed_prompt: str) -> tuple[str, float | None, int | None, int | None]:
    return _decide_uncached(trimmed_prompt)


def _session_target(state: dict, api_format: str) -> str | None:
    routes = state.get("routes")
    if isinstance(routes, dict):
        target = routes.get(api_format)
    else:
        # Compatibility for pre-catalog in-memory/test state.  It remains
        # readable, but never supplies cross-API affinity or an incompatible
        # target to a native endpoint.
        target = state.get("tier")
    if (isinstance(target, str) and target in BACKENDS
            and _supports_api(_backend_for(target), api_format)):
        return target
    # A new-format state deliberately has no cross-protocol affinity.
    return None


def _session_affinity(session_id: str | None, api_format: str) -> str:
    state = _session_get(session_id)
    return "warm" if state is not None and _session_target(state, api_format) is not None else "unknown"


def _store_key(prompt_hash: str, api_format: str) -> str:
    # Keep legacy Chat journal keys unchanged.  Responses must not create or
    # consume Chat pins, whose replay/quality semantics are distinct.
    return prompt_hash if api_format == "chat" else f"{api_format}:{prompt_hash}"


def _decide(prompt: str, session_id: str | None = None, api_format: str = "chat") -> tuple[str, float | None, int | None, int | None]:
    # Session traffic does not consult the global prompt store: short turns
    # like "Proceed" are not transferable across coding sessions. Once a live
    # session exists, continuation turns also skip Supra/MF scoring entirely.
    if session_id is not None and _is_continuation(prompt):
        state = _session_get(session_id)
        target = _session_target(state, api_format) if state is not None else None
        if target is not None:
            return target, None, state.get("last_complexity"), None
    if session_id is None:
        pinned = _store_pinned(_prompt_hash(prompt), api_format=api_format)
        if pinned is not None:
            target = pinned.get("decision")
            if isinstance(target, str) and target in BACKENDS:
                return target, pinned.get("score"), None, None
    trimmed_prompt = prompt[-15000:] if len(prompt) > 15000 else prompt
    return _decide_cached(trimmed_prompt)


def _backend_for(decision: str) -> dict:
    # Preserve the direct two-tier compatibility contract. Explicit target
    # registries never reinterpret a configured target based on legacy names.
    if not _TARGETS_ARE_EXPLICIT and decision == "middle" and not MIDDLE_CONFIGURED:
        return EXPENSIVE if EXPENSIVE["base"] else CHEAP
    return BACKENDS.get(decision, BACKENDS[_safe_target()])


def _supports_responses(backend: dict) -> bool:
    """Legacy Responses capability detection for non-catalog backends."""
    base = str(backend.get("base", ""))
    if not base or not str(backend.get("model", "")).lower().startswith("openai/"):
        return False
    try:
        host = (urlsplit(base).hostname or "").lower()
    except ValueError:
        return False
    return host == "openrouter.ai"


_REFUSAL_RE = re.compile(
    r"cannot (assist|help|comply)|i('| a)?m sorry|not (able|allowed) to (assist|help)|can('| no)t (assist|help)",
    re.I,
)


def _safe_json(content: bytes) -> dict:
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else {"error": {"message": content.decode(errors="replace")[:500]}}
    except Exception:
        return {"error": {"message": content.decode(errors="replace")[:500]}}


def _is_refusal(status: int, data: dict) -> bool:
    """True when an upstream Chat or native Responses object explicitly refuses.

    Responses represents refusals as ``output[].content[].type == "refusal"``;
    that is semantically stronger than matching prose and must remain visible to
    bounded failover even when the refusal text is empty.
    """
    text_parts: list[str] = []
    err = data.get("error")
    if isinstance(err, dict):
        text_parts.extend(str(err.get(name, "")) for name in ("message", "code", "type"))
    elif err:
        text_parts.append(str(err))

    def native_refusal(value) -> bool:
        """Inspect native response objects and SSE item envelopes recursively."""
        if not isinstance(value, dict):
            return False
        kind = value.get("type")
        # Complete Responses objects use content.type == refusal. Native SSE
        # can carry the same item under `item`, or use a refusal event type.
        if kind == "refusal" or (isinstance(kind, str) and ".refusal" in kind):
            return True
        for field in ("refusal", "text"):
            text = value.get(field)
            if isinstance(text, str):
                text_parts.append(text)
        for field in ("output", "content"):
            children = value.get(field)
            if isinstance(children, list) and any(native_refusal(child) for child in children):
                return True
        for field in ("response", "item"):
            if native_refusal(value.get(field)):
                return True
        return False

    if native_refusal(data):
        return True

    choices = data.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") == "content_filter":
                return True
            # Chat uses message.refusal in complete objects and delta.refusal in
            # streamed chunks.  The field itself is structured evidence, including
            # an empty refusal string, so it must not rely on prose matching.
            # Any choice may refuse; a second normal choice must not mask it.
            for message in (choice.get("message"), choice.get("delta")):
                if not isinstance(message, dict):
                    continue
                if isinstance(message.get("refusal"), str):
                    return True
                for field in ("content", "reasoning_content"):
                    value = message.get(field)
                    if isinstance(value, str):
                        text_parts.append(value)
    text = " ".join(text_parts)
    if status != 200 and re.search(r"content[\s_-]?filter", text, re.I):
        return True
    return bool(_REFUSAL_RE.search(text))


def _completion_token_cap(backend: dict) -> int:
    # Catalog targets omit max_tokens entirely (None = uncapped), so the
    # router-wide default is the only clamp that protects them.
    cap = backend.get("max_tokens") or ROUTELLM_MAX_TOKENS
    return min(cap, ROUTELLM_MAX_TOKENS)


def _build_outgoing_body(body: dict, backend: dict) -> dict:
    out_body = dict(body)
    if isinstance(out_body.get("messages"), list):
        out_body["messages"] = _normalize_messages_for_backend(
            out_body["messages"], developer_role=backend.get("developer_role", "system"))
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_tokens"), int):
        out_body["max_completion_tokens"] = out_body.pop("max_tokens")
    if isinstance(out_body.get("max_completion_tokens"), int):
        out_body["max_completion_tokens"] = min(out_body["max_completion_tokens"],
                                                _completion_token_cap(backend))
    out_body.pop("stop", None)
    if backend["model"].rsplit("/", 1)[-1].startswith("gpt-5.6-") and out_body.get("temperature") not in (None, 1):
        out_body.pop("temperature")
    if backend["effort"]:
        out_body["reasoning_effort"] = backend["effort"]
    if backend.get("usage_include") and "usage" not in out_body:
        out_body["usage"] = {"include": True}
    return out_body


def _build_responses_body(body: dict, backend: dict) -> dict:
    out_body = dict(body)
    out_body["model"] = backend["model"]
    if isinstance(out_body.get("max_output_tokens"), int):
        out_body["max_output_tokens"] = min(out_body["max_output_tokens"],
                                            _completion_token_cap(backend))
    # A target can declare its backend reasoning policy authoritative without
    # coupling this protocol behavior to a particular target ID.
    reasoning = out_body.get("reasoning")
    if backend.get("force_reasoning_effort") and backend.get("effort"):
        reasoning = dict(reasoning) if isinstance(reasoning, dict) else {}
        reasoning["effort"] = backend["effort"]
        out_body["reasoning"] = reasoning
    elif backend.get("effort") and "reasoning" not in out_body:
        out_body["reasoning"] = {"effort": backend["effort"]}
    return out_body


def _extract_cost(data: dict) -> float | None:
    try:
        cost = (data.get("usage") or {}).get("cost")
        return float(cost) if isinstance(cost, (int, float)) else None
    except Exception:
        return None


def _normalize_messages_for_backend(messages, *, developer_role: str = "system"):
    out = []
    seen_calls: set = set()
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        role = message.get("role")
        if role == "assistant":
            calls = message.get("tool_calls")
            if isinstance(calls, list):
                for call in calls:
                    if isinstance(call, dict) and isinstance(call.get("id"), str):
                        seen_calls.add(call["id"])
            if message.get("function_call"):
                seen_calls.add(None)
        elif role == "tool" and message.get("tool_call_id") not in seen_calls:
            # Providers hard-400 on results whose call was cut from history
            # (mid-conversation resume, compaction/truncation); drop the orphan.
            changed = True
            continue
        elif role == "function" and None not in seen_calls:
            changed = True
            continue
        elif role == "developer" and developer_role != "native":
            message = {**message, "role": "system"}
            changed = True
        out.append(message)
    return out if changed else messages


def _authorize(authorization: str | None) -> bool:
    if not authorization:
        return False
    if not authorization.startswith("Bearer "):
        return False
    return secrets.compare_digest(authorization[7:].encode(), SERVER_KEY.encode())


_LOG_MAX_BYTES = int(os.environ.get("ROUTER_LOG_MAX_BYTES", str(50 * 1024 * 1024)))


def _secure_append(path: Path, row: dict) -> None:
    try:
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path.parent.chmod(0o700)
        rotated = path.with_name(path.name + ".1")
        if path.exists() and path.stat().st_size > _LOG_MAX_BYTES:
            os.replace(path, rotated)  # keep one generation; bound disk growth
            rotated.chmod(0o600)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        path.chmod(0o600)
    except OSError:
        # Telemetry must never fail a user request.
        return


def _log(
    decision: str, score: float, backend_model: str, prompt: str,
    ttfb_ms: int | None, supra_complexity: int | None = None,
    supra_ms: int | None = None, cost_usd: float | None = None,
    usage: dict | None = None, request_id: str | None = None,
    occurrence_id: str | None = None, api_format: str = "chat", **detail,
):
    row = {
        "ts": time.time(), "request_id": request_id,
        "occurrence_id": occurrence_id or uuid.uuid4().hex,
        "router": "supra", "api_format": api_format,
        "score": round(score, 4) if isinstance(score, (int, float)) else None,
        "supra_complexity": supra_complexity, "supra_ms": supra_ms,
        "decision": decision, "model": backend_model, "ttfb_ms": ttfb_ms,
        "prompt_hash": _prompt_hash(prompt), **detail,
    }
    if cost_usd is not None:
        row["cost_usd"] = cost_usd
    if isinstance(usage, dict) and usage:
        row["usage"] = usage
        prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cached_tokens = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        if isinstance(prompt_tokens, (int, float)):
            row["prompt_tokens"] = prompt_tokens
        if isinstance(cached_tokens, (int, float)):
            row["cached_tokens"] = cached_tokens
            row["fresh_tokens"] = max(0, prompt_tokens - cached_tokens) if isinstance(prompt_tokens, (int, float)) else None
            row["cache_hit_ratio"] = (cached_tokens / prompt_tokens if prompt_tokens else None)
    _secure_append(LOG_PATH, row)
    if TRAINING_LOG_ENABLED:
        _secure_append(TRAINING_LOG_PATH, {**row, "prompt": prompt})


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:24]


def _log_outcome(prompt_hash: str, outcome: str, **detail) -> None:
    row = {
        "ts": time.time(), "occurrence_id": uuid.uuid4().hex,
        "prompt_hash": prompt_hash, "outcome": outcome, **detail,
    }
    _secure_append(OUTCOME_LOG_PATH, row)


_recent_prompts: dict[str, tuple[str, str, float, str]] = {}

# --- Session affinity -------------------------------------------------------
# Session state is intentionally in-memory. A restart must not carry a model
# choice into a new context, while the learned prompt store remains persistent.
_CONTINUATION_RE = re.compile(
    r"^(?:ok(?:ay)?[,. ]*)?(?:proceed|continue|go ahead|do (?:it|that)|yes|yep|sure|"
    r"run (?:it|them|the tests)(?: again)?|try again|fix (?:it|that)|next)(?:[.! ]*)$", re.I,
)
_NEW_TASK_RE = re.compile(
    r"^(?:new task|different task|unrelated|switching topics?|on another topic)\b", re.I,
)
SESSION_TTL_S = _env_float("ROUTELLM_SESSION_TTL_S", 3600.0)
SESSION_STATE_MAX = _env_int("ROUTELLM_SESSION_STATE_MAX", 4096)
_session_state: OrderedDict[str, dict] = OrderedDict()
_session_lock = threading.Lock()
_STICKY_REASONS = frozenset({
    "continuation_sticky", "same_tier", "upgrade_hysteresis", "downgrade_hysteresis",
})


def _is_continuation(prompt: str) -> bool:
    value = prompt.strip()
    return len(value) <= 200 and "```" not in value and bool(_CONTINUATION_RE.fullmatch(value))


def _session_id(body: dict, request: Request) -> tuple[str | None, str | None]:
    """Extract an opaque, source-namespaced session identity."""
    raw, source = request.headers.get("x-route-session"), "header"
    if not raw and isinstance(body.get("metadata"), dict):
        raw, source = body["metadata"].get("session_id"), "metadata"
    if (not raw and os.environ.get("ROUTELLM_SESSION_FROM_USER", "0").lower()
            in {"1", "true", "yes", "on"} and isinstance(body.get("user"), str)):
        raw, source = body["user"], "user"
    if not isinstance(raw, str) or not raw or len(raw.encode()) > 256:
        return None, None
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        return None, None
    secret = SERVER_KEY.encode("utf-8", errors="replace")
    digest = hmac.new(secret, (source + "\0" + raw).encode(), hashlib.sha256).hexdigest()[:24]
    return digest, source


def _session_get(session_id: str | None) -> dict | None:
    if not session_id:
        return None
    with _session_lock:
        state = _session_state.get(session_id)
        if state is None:
            return None
        # State is in-memory, but a process/test may replace the active target
        # registry.  New-format state is revision-bound; legacy in-memory state
        # without a revision remains readable only for compatibility.
        if (time.time() - state["last_seen"] > SESSION_TTL_S
                or (state.get("target_revision") is not None
                    and state.get("target_revision") != TARGET_CONFIG_REVISION)):
            _session_state.pop(session_id, None)
            return None
        _session_state.move_to_end(session_id)
        return dict(state)


def _session_route(session_id: str | None, prompt: str, proposed: str,
                   complexity: int | None, score: float | None = None, *,
                   new_task: bool = False, api_format: str = "chat") -> tuple[str, str]:
    proposed = proposed if proposed in BACKENDS else _safe_target()
    state = _session_get(session_id)
    if not state:
        return proposed, "new_session"
    if new_task or _NEW_TASK_RE.match(prompt.strip()):
        return proposed, "new_task"
    current = _session_target(state, api_format)
    if current is None:
        return proposed, "protocol_unpinned"
    if _is_continuation(prompt):
        return current, "continuation_sticky"
    if proposed == current:
        return current, "same_tier"
    current_rank, proposed_rank = _target_rank(current), _target_rank(proposed)
    if proposed_rank < current_rank:
        return current, "downgrade_hysteresis"
    mapped_target, _ = _target_for_complexity(complexity)
    # Exact catalog policies deliberately expose intermediate ranks.  A normal
    # session turn must be able to climb to any higher mapped target; otherwise
    # a low session would never reach a configured middle target until the
    # highest rank appeared.  Keep the historical strongest-only hysteresis for
    # legacy threshold deployments.
    if (_TARGETS_ARE_EXPLICIT and proposed_rank > current_rank) or (
            complexity is not None and _target_rank(mapped_target) >= _target_rank(_safe_target())):
        return proposed, "strong_upgrade"
    return current, "upgrade_hysteresis"


def _session_note(session_id: str | None, tier: str, complexity: int | None,
                  usage: dict | None = None, *, api_format: str = "chat") -> None:
    if not session_id or tier not in BACKENDS or not _supports_api(_backend_for(tier), api_format):
        return
    now = time.time()
    with _session_lock:
        prior = _session_state.get(session_id, {})
        previous_routes = prior.get("routes")
        routes = dict(previous_routes) if isinstance(previous_routes, dict) else {}
        routes[api_format] = tier
        # `tier` preserves the old in-memory shape for existing observability
        # and tests. New readers use routes so no protocol can inherit another
        # protocol's affinity.
        state = {
            "tier": routes.get("chat", tier), "routes": routes,
            "target_revision": TARGET_CONFIG_REVISION,
            "last_seen": now, "last_complexity": complexity,
            "turns": int(prior.get("turns", 0)) + 1,
        }
        if isinstance(usage, dict) and usage:
            details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
            state["prompt_tokens"] = usage.get("prompt_tokens", usage.get("input_tokens"))
            state["cached_tokens"] = details.get("cached_tokens", usage.get("prompt_cache_hit_tokens"))
        _session_state[session_id] = state
        _session_state.move_to_end(session_id)
        while len(_session_state) > SESSION_STATE_MAX:
            _session_state.popitem(last=False)


# --- Native Responses opaque-continuation affinity -------------------------
# Encrypted reasoning, compaction items, provider-side response IDs, and
# conversations are model/provider-origin-bound. They cannot safely follow the
# normal quality routing graph or bounded cross-target failover. The index below
# keeps only HMAC-derived identifiers and non-secret origin bindings in memory;
# opaque provider state itself is never logged, persisted, or returned.
RESPONSES_AFFINITY_TTL_S = _env_float("ROUTELLM_RESPONSES_AFFINITY_TTL_S", 3600.0)
RESPONSES_AFFINITY_MAX = _env_int("ROUTELLM_RESPONSES_AFFINITY_MAX", 4096)
_RESPONSES_AFFINITY_HEADER = "x-route-responses-affinity"
_responses_affinity: OrderedDict[str, dict] = OrderedDict()
_responses_affinity_lock = threading.Lock()


def _responses_affinity_key(kind: str, value: str) -> str:
    """Return a keyed, non-reversible in-memory index key for opaque state."""
    digest = hmac.new(
        SERVER_KEY.encode("utf-8", errors="replace"),
        (kind + "\0" + value).encode("utf-8", errors="replace"), hashlib.sha256,
    ).hexdigest()[:32]
    return f"{kind}:{digest}"


def _responses_opaque_values(value) -> list[tuple[str, str]]:
    """Collect identifiers for reasoning/compaction state without retaining it.

    A reasoning item remains affinity-bearing even when a client omits its
    encrypted content: its item ID still represents provider/model state. When
    ciphertext is present, its keyed digest lets stateless replay work without
    storing or exposing the ciphertext.
    """
    values: list[tuple[str, str]] = []

    def walk(item) -> None:
        if isinstance(item, list):
            for child in item:
                walk(child)
            return
        if not isinstance(item, dict):
            return
        item_type = item.get("type")
        if item_type in {"reasoning", "compaction"}:
            label = str(item_type)
            item_id = item.get("id")
            if isinstance(item_id, str) and item_id:
                values.append(("opaque-id", label + "\0" + item_id))
            encrypted = item.get("encrypted_content")
            if isinstance(encrypted, str) and encrypted:
                values.append(("opaque-content", label + "\0" + encrypted))
        # Native output/item envelopes can nest the same object in these
        # structured fields. Do not recurse through arbitrary extension fields.
        for field in ("content", "output", "item", "response"):
            walk(item.get(field))

    walk(value)
    return values


def _responses_conversation_value(value) -> str | None:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and isinstance(value.get("id"), str) and value["id"]:
        return value["id"]
    return None


def _responses_continuation_values(body: dict) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    previous = body.get("previous_response_id")
    if isinstance(previous, str) and previous:
        values.append(("response", previous))
    conversation = _responses_conversation_value(body.get("conversation"))
    if conversation is not None:
        values.append(("conversation", conversation))
    values.extend(_responses_opaque_values(body.get("input")))
    return values


def _responses_needs_affinity(body: dict) -> bool:
    """Return true for native state that must stay on its origin binding."""
    return bool(body.get("previous_response_id") or body.get("conversation")
                or _responses_opaque_values(body.get("input")))


def _responses_binding(backend: dict) -> tuple[str, str, str]:
    """Non-secret exact origin identity for opaque Responses continuation state."""
    return (str(backend.get("target", backend.get("tier", ""))),
            str(backend.get("base", "")).rstrip("/"), str(backend.get("model", "")))


def _responses_affinity_record(backend: dict) -> dict:
    target, base, model = _responses_binding(backend)
    return {"target": target, "base": base, "model": model,
            "target_revision": TARGET_CONFIG_REVISION, "last_seen": time.time()}


def _responses_affinity_put(key: str, record: dict) -> None:
    with _responses_affinity_lock:
        _responses_affinity[key] = dict(record)
        _responses_affinity.move_to_end(key)
        while len(_responses_affinity) > RESPONSES_AFFINITY_MAX:
            _responses_affinity.popitem(last=False)


def _responses_affinity_get(key: str) -> dict | None:
    now = time.time()
    with _responses_affinity_lock:
        record = _responses_affinity.get(key)
        if (record is None or now - record.get("last_seen", 0) > RESPONSES_AFFINITY_TTL_S
                or record.get("target_revision") != TARGET_CONFIG_REVISION):
            _responses_affinity.pop(key, None)
            return None
        record = {**record, "last_seen": now}
        _responses_affinity[key] = record
        _responses_affinity.move_to_end(key)
        return dict(record)


def _responses_affinity_token() -> str:
    nonce = secrets.token_urlsafe(24)
    signature = hmac.new(
        SERVER_KEY.encode("utf-8", errors="replace"), nonce.encode(), hashlib.sha256,
    ).hexdigest()[:24]
    return f"{nonce}.{signature}"


def _responses_affinity_token_record(token: str | None) -> dict | None:
    if not isinstance(token, str) or len(token) > 256 or token.count(".") != 1:
        return None
    nonce, signature = token.rsplit(".", 1)
    expected = hmac.new(
        SERVER_KEY.encode("utf-8", errors="replace"), nonce.encode(), hashlib.sha256,
    ).hexdigest()[:24]
    if not secrets.compare_digest(signature, expected):
        return None
    return _responses_affinity_get("token:" + token)


def _responses_payload(event: bytes) -> dict | None:
    data = _sse_data(event)
    if data is None or data.strip() == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _responses_affinity_record_payload(backend: dict, payload: dict | None) -> None:
    """Index opaque identifiers observed in a native Responses event/object."""
    if not isinstance(payload, dict):
        return
    record = _responses_affinity_record(backend)
    response = payload.get("response")
    response = response if isinstance(response, dict) else {}
    response_id = payload.get("id") if isinstance(payload.get("id"), str) else None
    if response_id is None:
        response_id = response.get("id") if isinstance(response.get("id"), str) else None
    if response_id:
        _responses_affinity_put(_responses_affinity_key("response", response_id), record)
    conversation = _responses_conversation_value(payload.get("conversation"))
    if conversation is None:
        conversation = _responses_conversation_value(response.get("conversation"))
    if conversation is not None:
        _responses_affinity_put(_responses_affinity_key("conversation", conversation), record)
    for kind, value in _responses_opaque_values(payload):
        _responses_affinity_put(_responses_affinity_key(kind, value), record)


def _responses_affinity_issue(backend: dict, payload: dict | None = None) -> str:
    """Issue an origin capability and index any opaque state in ``payload``."""
    token = _responses_affinity_token()
    _responses_affinity_put("token:" + token, _responses_affinity_record(backend))
    _responses_affinity_record_payload(backend, payload)
    return token


def _responses_affinity_backend(body: dict, token: str | None) -> dict | None:
    """Resolve a continuation to one verified exact Responses origin.

    Strictly fail closed: every affinity-bearing identifier/content digest in the
    request must resolve to the same recorded binding. An unindexed or
    disagreeing item is rejected even when a valid router-issued token is
    present, because a token proves only possession, never that arbitrary opaque
    state belongs to that origin.
    """
    records = []
    if token is not None:
        token_record = _responses_affinity_token_record(token)
        if token_record is None:
            return None
        records.append(token_record)
    for kind, value in _responses_continuation_values(body):
        record = _responses_affinity_get(_responses_affinity_key(kind, value))
        if record is None:
            return None
        records.append(record)
    if not records:
        return None
    binding = tuple(records[0].get(name) for name in ("target", "base", "model", "target_revision"))
    if any(tuple(record.get(name) for name in ("target", "base", "model", "target_revision")) != binding
           for record in records[1:]):
        return None
    target = records[0].get("target")
    if not isinstance(target, str) or target not in BACKENDS:
        return None
    backend = _backend_for(target)
    if (_responses_binding(backend) != tuple(records[0].get(name) for name in ("target", "base", "model"))
            or not _supports_api(backend, "responses")):
        return None
    return backend


# --- Persistent per-prompt decision store -----------------------------------
# This workload is dominated by repeated prompts (top 25 prompts = ~38% of
# calls). Pins are intentionally API-scoped: a Responses result cannot cause a
# later Chat request to select a Responses-only target, and vice versa.
DECISION_STORE_PATH = Path(os.environ.get("DECISION_STORE_FILE", str(DATA_DIR / "router/decision-state.jsonl")))
PIN_CHEAP_AFTER = _env_int("ROUTELLM_PIN_CHEAP_AFTER", 5)
PIN_EXPENSIVE_AFTER = _env_int("ROUTELLM_PIN_EXPENSIVE_AFTER", 2)
PIN_TTL_S = _env_float("ROUTELLM_PIN_TTL_S", float(7 * 86400))
DECISION_STORE_MAX = _env_int("ROUTELLM_DECISION_STORE_MAX", 4096)
_decision_store: dict[str, dict] = {}
_decision_store_lock = threading.Lock()
SUPRA_FALLBACK_COUNT = 0


def _store_load() -> None:
    """Load only current-target-revision entries from the decision journal."""
    if not DECISION_STORE_PATH.exists():
        return
    try:
        with open(DECISION_STORE_PATH, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_hash = entry.get("prompt_hash")
                api_format = entry.get("api_format", "chat")
                if (isinstance(prompt_hash, str) and prompt_hash
                        and api_format in {"chat", "responses"}
                        and entry.get("target_revision") == TARGET_CONFIG_REVISION):
                    _decision_store[_store_key(prompt_hash, api_format)] = entry
    except OSError:
        return
    if len(_decision_store) > DECISION_STORE_MAX:
        for key in sorted(_decision_store, key=lambda item: _decision_store[item].get("ts", 0))[
                : len(_decision_store) - DECISION_STORE_MAX]:
            del _decision_store[key]


def _store_pinned(prompt_hash: str, *, api_format: str = "chat") -> dict | None:
    """Return a current-revision, protocol-compatible stored pin."""
    if not prompt_hash or api_format not in {"chat", "responses"}:
        return None
    entry = _decision_store.get(_store_key(prompt_hash, api_format))
    if not entry or entry.get("target_revision") != TARGET_CONFIG_REVISION:
        return None
    pin_until = entry.get("pin_until")
    if not isinstance(pin_until, (int, float)) or pin_until < time.time():
        return None
    decision = entry.get("decision")
    if not isinstance(decision, str) or decision not in BACKENDS:
        return None
    if not _supports_api(_backend_for(decision), api_format):
        return None
    return entry


def _refusal_target(decision: str, *, api_format: str = "chat") -> str:
    """Promote a refused target to a compatible bounded graph/rank route."""
    if decision not in BACKENDS:
        return _safe_target()
    if not _TARGETS_ARE_EXPLICIT:
        # Preserve historic cheap -> expensive learned escalation whenever it
        # can serve this API. If it cannot, select the best compatible route.
        legacy = "expensive" if "expensive" in BACKENDS else _safe_target()
        if _supports_api(_backend_for(legacy), api_format):
            return legacy
    for target in _api_routes(decision, api_format):
        if target != decision:
            return target
    compatible = _ranked_compatible_targets(decision, api_format)
    return next((target for target in compatible if target != decision), decision)


def _lowest_compatible_rank(api_format: str) -> int | None:
    ranks = [
        _target_rank(target) for target in BACKENDS
        if _supports_api(_backend_for(target), api_format)
    ]
    return min(ranks) if ranks else None


def _record_refusal_learning(prompt_hash: str, attempts, session_id: str | None,
                             complexity: int | None, *, api_format: str) -> None:
    """Learn only an explicit refusal from this API's lowest viable target.

    A lower-ranked target for the *other* native API must not suppress the
    learning signal here.  This keeps a Chat-only or Responses-only target from
    contaminating protocol-scoped pins and session affinity.
    """
    low_rank = _lowest_compatible_rank(api_format)
    if low_rank is None:
        return
    for attempted, _backend, _status, error in attempts:
        if (error not in {"refusal", "refusal_fallback"} or attempted not in BACKENDS
                or _target_rank(attempted) != low_rank):
            continue
        if session_id is None:
            _store_note(prompt_hash, attempted, ok=False, api_format=api_format)
        else:
            _session_note(session_id, _refusal_target(attempted, api_format=api_format),
                          complexity, api_format=api_format)


def _store_note(prompt_hash: str, decision: str, *, ok: bool = False, score=None,
                api_format: str = "chat") -> None:
    """Record one API-scoped outcome and pin only this target revision."""
    if (not prompt_hash or decision not in BACKENDS or api_format not in {"chat", "responses"}
            or not _supports_api(_backend_for(decision), api_format)):
        return
    now = time.time()
    key = _store_key(prompt_hash, api_format)
    with _decision_store_lock:
        stored = _decision_store.get(key)
        entry = (dict(stored) if stored and stored.get("target_revision") == TARGET_CONFIG_REVISION
                 else {"prompt_hash": prompt_hash, "api_format": api_format, "decision": decision,
                       "ok": 0, "fail": 0, "ts": 0, "target_revision": TARGET_CONFIG_REVISION})
        if now - float(entry.get("ts", 0)) > 86400:
            entry["ok"], entry["fail"] = 0, 0
        entry["decision"] = decision
        entry["api_format"] = api_format
        entry["target_revision"] = TARGET_CONFIG_REVISION
        entry["ts"] = now
        if score is not None:
            entry["score"] = score
        entry["ok"] = int(entry.get("ok", 0)) + (1 if ok else 0)
        entry["fail"] = int(entry.get("fail", 0)) + (0 if ok else 1)
        compatible = [
            target for target in BACKENDS
            if _supports_api(_backend_for(target), api_format)
        ]
        low_rank = min((_target_rank(target) for target in compatible), default=None)
        if (not ok and low_rank is not None and _target_rank(decision) == low_rank
                and entry["fail"] >= PIN_EXPENSIVE_AFTER):
            entry["decision"] = _refusal_target(decision, api_format=api_format)
            entry["pin_until"] = now + PIN_TTL_S
        elif (ok and low_rank is not None and _target_rank(decision) == low_rank
              and entry["ok"] >= PIN_CHEAP_AFTER and entry["fail"] == 0):
            entry["pin_until"] = now + PIN_TTL_S
        elif entry.get("pin_until") and float(entry.get("pin_until", 0)) < now:
            entry.pop("pin_until", None)
        _decision_store[key] = entry
        if len(_decision_store) > DECISION_STORE_MAX:
            for old_key in sorted(_decision_store, key=lambda item: _decision_store[item].get("ts", 0))[
                    : len(_decision_store) - DECISION_STORE_MAX]:
                del _decision_store[old_key]
        _secure_append(DECISION_STORE_PATH, entry)


RESP_CACHE_TTL_S = _env_float("ROUTELLM_RESP_CACHE_TTL_S", 120.0)
RESP_CACHE_MAX_ENTRIES = _env_int("ROUTELLM_RESP_CACHE_MAX_ENTRIES", 128)
RESP_CACHE_MAX_BYTES = _env_int("ROUTELLM_RESP_CACHE_MAX_BYTES", 8 * 1024 * 1024)
_resp_cache: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
_inflight: dict[str, asyncio.Future] = {}
_inflight_lock = asyncio.Lock()
_cache_bytes = 0
_cache_metrics = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}


def _request_body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _cache_key(body: dict, idempotency_key: str | None, *, session_id: str | None = None) -> str | None:
    if not idempotency_key or len(idempotency_key) > 200:
        return None
    # Tool/function traffic can have external side effects and is never
    # replay-safe, including the legacy functions/function_call shapes.
    if (body.get("tools") or body.get("functions") or body.get("function_call")
            or any(isinstance(m, dict) and m.get("role") in {"tool", "function"}
                   or isinstance(m, dict) and (m.get("tool_calls") or m.get("function_call"))
                   for m in body.get("messages", []))):
        return None
    # Session-affine requests must not replay/coalesce across opaque session
    # identities: their selected target and preceding context can legitimately
    # differ even when a caller repeats body + idempotency key. The HMAC-derived
    # ID is safe to hash, and neither raw header nor metadata is retained.
    scope = session_id or "sessionless"
    return hashlib.sha256(
        (TARGET_CONFIG_REVISION + ":" + scope + ":" + idempotency_key + ":" + _request_body_hash(body)).encode()
    ).hexdigest()


def _cache_get(key: str | None) -> bytes | None:
    global _cache_bytes
    if key is None:
        return None
    hit = _resp_cache.get(key)
    if hit is None:
        _cache_metrics["misses"] += 1
        return None
    ts, content = hit
    if time.monotonic() - ts > RESP_CACHE_TTL_S:
        _cache_bytes -= len(content)
        del _resp_cache[key]
        _cache_metrics["misses"] += 1
        return None
    _resp_cache.move_to_end(key)
    _cache_metrics["hits"] += 1
    return content


def _response_replay_safe(body: dict, content: bytes) -> bool:
    # Multi-choice responses are not replay/coalescing candidates: their
    # per-choice ordering and truncation semantics are not reconstructible.
    if body.get("n", 1) != 1:
        return False
    if body.get("stream"):
        text = content.decode("utf-8", errors="replace")
        return (content.rstrip().endswith(b"data: [DONE]")
                and '"tool_calls"' not in text
                and '"function_call"' not in text
                and not re.search(r'"finish_reason"\s*:\s*"(?:content_filter|length)"', text)
                and not _REFUSAL_RE.search(text)
                and not _sse_contains_refusal(content))
    data = _safe_json(content)
    choices = data.get("choices") or []
    return (bool(choices) and not _is_refusal(200, data)
            and all(choice.get("finish_reason") not in {"content_filter", "length"}
                    and not (choice.get("message") or {}).get("tool_calls")
                    and not (choice.get("message") or {}).get("function_call") for choice in choices))


def _cache_put(key: str | None, body: dict, content: bytes) -> None:
    global _cache_bytes
    if key is None or len(content) > RESP_CACHE_MAX_BYTES or not _response_replay_safe(body, content):
        return
    existing = _resp_cache.pop(key, None)
    if existing is not None:
        _cache_bytes -= len(existing[1])
    while _resp_cache and (len(_resp_cache) >= RESP_CACHE_MAX_ENTRIES or _cache_bytes + len(content) > RESP_CACHE_MAX_BYTES):
        _, (_, old) = _resp_cache.popitem(last=False)
        _cache_bytes -= len(old)
        _cache_metrics["evictions"] += 1
    if _cache_bytes + len(content) <= RESP_CACHE_MAX_BYTES:
        _resp_cache[key] = (time.monotonic(), content)
        _cache_bytes += len(content)
        _cache_metrics["stores"] += 1


async def _claim_inflight(key: str | None):
    if key is None:
        return True, None
    async with _inflight_lock:
        future = _inflight.get(key)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            _inflight[key] = future
            return True, future
        return False, future


async def _finish_inflight(key: str | None, future, result) -> None:
    if key is None or future is None:
        return
    async with _inflight_lock:
        if _inflight.get(key) is future:
            _inflight.pop(key, None)
        if not future.done():
            future.set_result(result)


def _finish_inflight_nowait(key: str | None, future, result) -> None:
    """Release ownership synchronously from cancellation cleanup."""
    if key is None or future is None:
        return
    if _inflight.get(key) is future:
        _inflight.pop(key, None)
    if not future.done():
        future.set_result(result)


def _replayed_response(result, request_id: str):
    content, status, media, headers = result
    replay_headers = {**headers, "x-route-coalesced": "true", "x-request-id": request_id}
    if media == "text/event-stream":
        return StreamingResponse(iter([content]), status_code=status, media_type=media, headers=replay_headers)
    return Response(content=content, status_code=status, media_type=media, headers=replay_headers)


def _request_hash(body: dict) -> str:
    """Hash of the tail of the conversation. Agent loops append tool results
    between calls (hash changes); a true retry resends an identical body
    (hash stable) — hashing just the last user message misfires on loops."""
    msgs = body.get("messages") or []
    return hashlib.sha1(
        json.dumps(msgs[-6:], sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()[:16]


def _record_and_detect_retry(req_hash: str, decision: str, model: str, prompt_hash: str,
                             request_id: str, occurrence_id: str) -> None:
    now = time.time()
    for key, value in list(_recent_prompts.items()):
        if now - value[2] > RETRY_WINDOW_S:
            del _recent_prompts[key]
    prev = _recent_prompts.pop(req_hash, None)
    if prev and now - prev[2] <= RETRY_WINDOW_S:
        _log_outcome(prompt_hash, "retried", decision=prev[0], model=prev[1],
                     request_hash=req_hash, request_id=request_id,
                     decision_occurrence_id=prev[3], retry_after_s=round(now - prev[2], 1))
        # Repeated identical bodies are useful outcome telemetry, but are not
        # a safe escalation signal: agent loops legitimately repeat prompts
        # such as "Proceed" and polling instructions. Only an explicit cheap
        # refusal (recorded from the upstream response) may pin expensive.
    _recent_prompts[req_hash] = (decision, model, now, occurrence_id)


def _length_truncated(data: dict) -> bool:
    ch = data.get("choices")
    return bool(isinstance(ch, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in ch))


_READY = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _READY, _client
    _client = httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=min(10.0, TIMEOUT_S)))
    try:
        # Model initialization is CPU/blocking work and must not block the loop.
        await asyncio.to_thread(_load_supra)
        _store_load()
        _READY = True
        yield
    finally:
        _READY = False
        if _client is not None:
            await _client.aclose()
        _client = None


app = FastAPI(title="RouteLLM coding-router", lifespan=lifespan)


def _openai_error(message: str, status: int, *, error_type: str = "invalid_request_error", param=None, code=None):
    return JSONResponse(
        {"error": {"message": message, "type": error_type, "param": param, "code": code}},
        status_code=status,
    )


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if _body_too_large(length):
        return _openai_error("Request body is too large", 413, code="request_too_large")
    return await call_next(request)


def _body_too_large(content_length: str | None) -> bool:
    return bool(content_length and content_length.isdigit() and int(content_length) > MAX_BODY_BYTES)


async def _read_json_body(request: Request) -> dict:
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise OverflowError
        chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid JSON body") from exc
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _validate_request(body: dict) -> tuple[str, str | None] | None:
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return "'messages' must be a non-empty array", "messages"
    roles = {"system", "developer", "user", "assistant", "tool", "function"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in roles:
            return "Each message must contain a supported role", f"messages.{index}.role"
        content = message.get("content")
        if content is not None and not isinstance(content, (str, list)):
            return "Message content must be text, an array, or null", f"messages.{index}.content"
    checks = {
        "stream": bool, "temperature": (int, float), "top_p": (int, float),
        "n": int, "max_tokens": int, "max_completion_tokens": int,
        "presence_penalty": (int, float), "frequency_penalty": (int, float),
        "tools": list, "tool_choice": (str, dict), "response_format": dict,
        "stream_options": dict, "seed": int, "stop": (str, list),
    }
    for name, expected in checks.items():
        if name in body and (isinstance(body[name], bool) and expected is not bool or not isinstance(body[name], expected)):
            return f"'{name}' has an invalid type", name
    for name in ("n", "max_tokens", "max_completion_tokens"):
        if name in body and body[name] <= 0:
            return f"'{name}' must be greater than zero", name
    # Unknown keys are intentionally preserved as reviewed provider extensions.
    return None


def _validate_responses_request(body: dict) -> tuple[str, str | None] | None:
    if body.get("model") != MODEL_ID:
        return "Only model 'auto' is supported", "model"
    if "input" not in body or body.get("input") is None:
        return "'input' is required", "input"
    input_value = body.get("input")
    if not isinstance(input_value, (str, list)):
        return "'input' must be text or an array", "input"
    if not input_value:
        return "'input' must not be empty", "input"
    if "stream" in body and not isinstance(body["stream"], bool):
        return "'stream' has an invalid type", "stream"
    if "max_output_tokens" in body:
        value = body["max_output_tokens"]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            return "'max_output_tokens' must be a positive integer", "max_output_tokens"
    stable_types = {
        "instructions": (str, type(None)), "tools": list, "metadata": dict,
        "reasoning": dict, "tool_choice": (str, dict),
    }
    for name, expected in stable_types.items():
        if name in body and not isinstance(body[name], expected):
            return f"'{name}' has an invalid type", name
    for name in ("temperature", "top_p"):
        if name in body and (isinstance(body[name], bool) or not isinstance(body[name], (int, float))):
            return f"'{name}' has an invalid type", name
    # Responses extensions are preserved rather than rejected.
    return None


def _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                   *, pinned=False, api_format="chat", supra_reason: str | None = None):
    upstream_path = "/responses" if api_format == "responses" else "/chat/completions"
    headers = {
        "x-request-id": request_id, "x-route-decision": decision,
        "x-route-target": str(backend.get("target", decision)),
        "x-route-target-revision": TARGET_CONFIG_REVISION,
        "x-route-score": f"{score:.4f}" if isinstance(score, (int, float)) else "n/a",
        "x-route-model": backend["model"],
        "x-route-router": "supra", "x-route-fallback": "false",
        "x-route-attempts": "1", "x-route-api": api_format,
        "x-route-upstream-path": upstream_path,
    }
    if pinned:
        headers["x-route-pinned"] = "true"
    if supra_complexity is not None:
        headers["x-route-supra-complexity"] = str(supra_complexity)
    if supra_ms is not None:
        headers["x-route-supra-ms"] = str(supra_ms)
    if supra_reason:
        headers["x-route-supra-reason"] = supra_reason
    return headers


def _upstream_request(backend: dict, body: dict, *, api_format: str = "chat") -> httpx.Request:
    assert _client is not None
    if api_format == "responses":
        path, outgoing = "/responses", _build_responses_body(body, backend)
    else:
        path, outgoing = "/chat/completions", _build_outgoing_body(body, backend)
    return _client.build_request(
        "POST", backend["base"].rstrip("/") + path,
        json=outgoing,
        headers={"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"},
    )


async def _send(backend: dict, body: dict, *, stream: bool, api_format: str = "chat") -> httpx.Response:
    if _client is None:
        raise RuntimeError("router is not ready")
    return await _client.send(_upstream_request(backend, body, api_format=api_format), stream=stream)


def _retryable(status: int) -> bool:
    return status in RETRY_STATUSES


def _failover_routes(decision: str) -> tuple[str, ...]:
    """Compatibility alias for Chat's protocol-filtered route graph."""
    return _chat_routes(decision)


async def _open_with_failover(body: dict, decision: str, deadline: float, *, stream: bool,
                              api_format: str = "chat"):
    attempts = []
    routes = _responses_routes(decision) if api_format == "responses" else _chat_routes(decision)
    if not routes:
        return decision, _backend_for(decision), None, attempts
    for index, current in enumerate(routes):
        backend = _backend_for(current)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await _send(backend, body, stream=stream, api_format=api_format)
                data = None
                if not stream or response.status_code != 200:
                    data = _safe_json(await response.aread())
            refusal = data is not None and _is_refusal(response.status_code, data)
            attempts.append((current, backend, response.status_code, "refusal" if refusal else None))
            retry = refusal or _retryable(response.status_code)
            if not retry or index == len(routes) - 1:
                return current, backend, response, attempts
            await response.aclose()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as exc:
            attempts.append((current, backend, None, type(exc).__name__))
    return attempts[-1][0], attempts[-1][1], None, attempts


async def _open_responses_stream_with_failover(body: dict, decision: str, deadline: float):
    """Buffer a small native Responses prefix before committing its SSE bytes.

    Unlike Chat, the Responses endpoint cannot translate a stream.  Before any
    frame is exposed, inspect a bounded prefix for an explicit native refusal;
    if found, discard it and make the one configured compatible retry.  Once a
    non-refusal frame is committed, normal streaming semantics apply.
    """
    attempts = []
    routes = _responses_routes(decision)
    if not routes:
        return decision, _backend_for(decision), None, attempts, []
    for index, current in enumerate(routes):
        backend = _backend_for(current)
        response = None
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            async with asyncio.timeout(remaining):
                response = await _send(backend, body, stream=True, api_format="responses")
            if response.status_code != 200:
                data = _safe_json(await response.aread())
                refusal = _is_refusal(response.status_code, data)
                attempts.append((current, backend, response.status_code, "refusal" if refusal else None))
                if not (refusal or _retryable(response.status_code)) or index == len(routes) - 1:
                    return current, backend, response, attempts, []
                await response.aclose()
                continue
            events = _iter_sse_events(response)
            prefix, refusal = await _responses_prefetch_sse(events, deadline)
            attempts.append((current, backend, response.status_code, "refusal" if refusal else None))
            if not refusal or index == len(routes) - 1:
                return current, backend, response, attempts, (prefix, events)
            await response.aclose()
        except asyncio.CancelledError:
            if response is not None:
                await asyncio.shield(response.aclose())
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
            if response is not None:
                await response.aclose()
            attempts.append((current, backend, None, type(exc).__name__))
    return attempts[-1][0], attempts[-1][1], None, attempts, []


async def _responses_send_bound(body: dict, backend: dict, deadline: float, *, stream: bool):
    """Send a Responses request to exactly one origin; no cross-target failover.

    Used for opaque continuation state that is model/provider-origin-bound and
    cannot be retried on a different target. Transport errors are returned rather
    than retried; the caller surfaces a clear error to the client.
    """
    attempts = []
    response = None
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            response = await _send(backend, body, stream=stream, api_format="responses")
        if stream and response.status_code == 200:
            events = _iter_sse_events(response)
            prefix, refusal = await _responses_prefetch_sse(events, deadline)
            attempts.append((backend.get("target", backend.get("tier")), backend, 200,
                             "refusal" if refusal else None))
            return backend, response, attempts, (prefix, events)
        data = None
        if not stream or response.status_code != 200:
            data = _safe_json(await response.aread())
        refusal = data is not None and _is_refusal(response.status_code, data)
        attempts.append((backend.get("target", backend.get("tier")), backend, response.status_code,
                         "refusal" if refusal else None))
        return backend, response, attempts, None
    except asyncio.CancelledError:
        if response is not None:
            await asyncio.shield(response.aclose())
        raise
    except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
        if response is not None:
            await asyncio.shield(response.aclose())
        attempts.append((backend.get("target", backend.get("tier")), backend, None, type(exc).__name__))
        return None, None, attempts, None


def _log_attempts(attempts, prompt: str, score: float, request_id: str, occurrence_id: str,
                  *, api_format: str = "chat") -> None:
    for index, (decision, backend, status, error) in enumerate(attempts, 1):
        _log(decision, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="attempt", attempt=index,
             tier=backend.get("tier"), status=status, error=error, api_format=api_format)


@app.get("/healthz")
async def healthz():
    return {
        "ok": _READY, "router": "supra",
        "backend": "direct", "tiers": list(BACKENDS),
        "targets": {
            target: {
                "model": backend["model"], "provider": backend.get("provider"),
                "adapter": backend.get("adapter"),
                "protocols": list(backend.get("protocols") or ("chat_completions",)),
                "fallbacks": list(backend.get("fallbacks") or ()), "rank": _target_rank(target),
            }
            for target, backend in BACKENDS.items()
        },
        "supra_targets": list(SUPRA_TARGETS), "supra_invalid_target": SUPRA_INVALID_TARGET,
        "target_config_source": TARGET_CONFIG_SOURCE, "target_config_revision": TARGET_CONFIG_REVISION,
        "target_config_fingerprint": TARGET_CONFIG_FINGERPRINT,
        "supra_fallback_count": SUPRA_FALLBACK_COUNT,
        "ready": _READY, "cache": {**_cache_metrics, "entries": len(_resp_cache), "bytes": _cache_bytes},
    }


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{
        "id": MODEL_ID, "object": "model", "owned_by": "routellm",
        "context_window": await _get_context_window(), "max_tokens": ROUTELLM_MAX_TOKENS,
    }]}


MAX_SSE_EVENT_BYTES = _env_int("ROUTELLM_MAX_SSE_EVENT_BYTES", 1024 * 1024)


def _sse_boundary(buffer: bytearray) -> int | None:
    """Return the end of the first blank SSE line for CR, LF, or CRLF."""
    line_start = 0
    index = 0
    while index < len(buffer):
        if buffer[index] not in (10, 13):
            index += 1
            continue
        if buffer[index] == 13 and index + 1 == len(buffer):
            return None  # wait to distinguish CR from a split CRLF
        end = index + (2 if buffer[index:index + 2] == b"\r\n" else 1)
        if index == line_start:
            return end
        line_start = end
        index = end
    return None


async def _iter_sse_events(response: httpx.Response):
    """Yield exact SSE event frames while bounding one provider event."""
    buffer = bytearray()
    async for chunk in response.aiter_bytes():
        buffer.extend(chunk)
        while (end := _sse_boundary(buffer)) is not None:
            yield bytes(buffer[:end])
            del buffer[:end]
        if len(buffer) > MAX_SSE_EVENT_BYTES:
            raise ValueError("upstream SSE event is too large")
    if buffer:
        if len(buffer) > MAX_SSE_EVENT_BYTES:
            raise ValueError("upstream SSE event is too large")
        yield bytes(buffer)


def _sse_data(event: bytes) -> str | None:
    values = []
    for raw_line in event.splitlines():
        if raw_line.startswith(b"data:"):
            value = raw_line[5:]
            if value.startswith(b" "):
                value = value[1:]
            values.append(value.decode("utf-8", errors="replace"))
    return "\n".join(values) if values else None


def _sse_contains_refusal(content: bytes) -> bool:
    """Inspect complete cached SSE frames for structured or textual refusals."""
    buffer = bytearray(content)
    while (end := _sse_boundary(buffer)) is not None:
        event = bytes(buffer[:end])
        del buffer[:end]
        data = _sse_data(event)
        if data is None or data.strip() == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and _is_refusal(200, payload):
            return True
    # Replay candidates end in a complete DONE event, but preserve conservative
    # safety if a provider supplied a final unframed JSON refusal payload.
    data = _sse_data(bytes(buffer))
    if data and data.strip() != "[DONE]":
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return False
        return isinstance(payload, dict) and _is_refusal(200, payload)
    return False


def _possible_refusal_prefix(text: str) -> bool:
    value = text.strip().lower()
    leads = ("i", "i'", "i’m", "i am", "i cannot", "i can't", "i’m sorry",
             "i'm sorry", "sorry", "unfortunately", "as an ai", "cannot")
    return any(lead.startswith(value) or value.startswith(lead) for lead in leads)


async def _prefetch_sse(events, deadline: float):
    """Inspect a small prefix for standard delta-based refusals before release."""
    prefix, text_parts = [], []
    prefix_bytes = data_events = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        try:
            async with asyncio.timeout(remaining):
                event = await anext(events)
        except StopAsyncIteration:
            return prefix, False
        prefix.append(event)
        prefix_bytes += len(event)
        if prefix_bytes >= MAX_SSE_EVENT_BYTES or data_events >= 8:
            return prefix, False
        data = _sse_data(event)
        if data is None:
            # Comment/keepalive frames carry no decision signal. Commit so a
            # ping-only stream is forwarded instead of being held for prose.
            return prefix, False
        data_events += 1
        if data.strip() == "[DONE]":
            return prefix, False
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return prefix, False
        if _is_refusal(200, payload):
            return prefix, True
        choices = payload.get("choices") or []
        finish = None
        event_content = []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") == "content_filter":
                return prefix, True
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            for obj in (delta, message):
                if not isinstance(obj, dict):
                    continue
                if isinstance(obj.get("refusal"), str):
                    return prefix, True
                for field in ("content", "reasoning_content"):
                    value = obj.get(field)
                    if isinstance(value, str):
                        text_parts.append(value)
                        event_content.append(value)
            if finish is None and choice.get("finish_reason") is not None:
                finish = choice.get("finish_reason")
        combined = "".join(text_parts)
        if _REFUSAL_RE.search(combined):
            return prefix, True
        if finish is not None:
            return prefix, False
        if event_content and not _possible_refusal_prefix(combined):
            return prefix, False
        if not event_content:
            # Metadata-only chunk (for example the common role delta). Holding
            # the live stream here for prose would starve normal/sparse streams
            # until the request deadline.
            return prefix, False


async def _responses_prefetch_sse(events, deadline: float):
    """Native Responses preflight: commit on the first parseable event.

    Chat ``_prefetch_sse`` accumulates choices/delta prose, which starves sparse
    native Responses streams (a single output delta followed by silence would
    wait for the full deadline). Native Responses streams instead commit on the
    first event so ``response.created`` / the first output delta is forwarded
    promptly. A structured refusal carried by that same event is detected first
    and discarded so bounded failover can retry before bytes become visible; a
    refusal that arrives only after committed bytes is forwarded natively with
    the usual no-success-learning semantics.
    """
    prefix = []
    prefix_bytes = data_events = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        try:
            async with asyncio.timeout(remaining):
                event = await anext(events)
        except StopAsyncIteration:
            return prefix, False
        prefix.append(event)
        prefix_bytes += len(event)
        if prefix_bytes >= MAX_SSE_EVENT_BYTES or data_events >= 16:
            return prefix, False
        data_events += 1
        completed, failure, _usage, _response_id = _responses_event_state(event)
        if failure == "refusal":
            return prefix, True
        return prefix, False


def _stream_error(message: str, code: str, request_id: str) -> bytes:
    return ("data: " + json.dumps({"error": {"message": message, "type": "upstream_error", "code": code},
                                   "request_id": request_id}) + "\n\n").encode()


def _responses_event_state(event: bytes) -> tuple[bool, str | None, dict | None, str | None]:
    """Return (completed, terminal_failure, usage, response_id) for native SSE.

    ``terminal_failure`` is ``"refusal"`` for an explicit native refusal so
    callers can retain the original frame but avoid treating it as a successful
    quality signal once stream bytes have become visible. The response ID is
    used only as the opaque client-facing key for a router-owned affinity token;
    it is not persisted or logged.
    """
    event_name = None
    for line in event.splitlines():
        if line.startswith(b"event:"):
            event_name = line[6:].strip().decode("utf-8", errors="replace")
            break
    data_text = _sse_data(event)
    if data_text is None:
        failure = "upstream_error" if event_name in {"error", "response.failed", "response.incomplete"} else None
        return False, failure, None, None
    if data_text.strip() == "[DONE]":
        # A bare terminal SSE marker does not prove a native response.completed
        # object arrived. Keep forwarding it, but never treat it as completion.
        return False, None, None, None
    try:
        payload = json.loads(data_text)
    except json.JSONDecodeError:
        return False, None, None, None
    event_type = payload.get("type") or event_name
    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
    usage = response.get("usage") or payload.get("usage")
    response_id = response.get("id") if isinstance(response.get("id"), str) else None
    # A refusal discovered after the bounded preflight cannot be retried once
    # frames are visible, but it must never be marked as a completed success or
    # seed a positive pin/session affinity.
    if _is_refusal(200, payload):
        return False, "refusal", usage if isinstance(usage, dict) else None, None
    if event_type == "response.completed":
        # The event type alone is not proof of success: an explicit or absent
        # non-completed status must not seed positive learning.
        if response.get("status") == "completed":
            return True, None, usage if isinstance(usage, dict) else None, response_id
        return False, "upstream_error", usage if isinstance(usage, dict) else None, response_id
    if event_type in {"error", "response.failed", "response.incomplete"} or "error" in payload:
        return False, "upstream_error", usage if isinstance(usage, dict) else None, None
    return False, None, usage if isinstance(usage, dict) else None, None


def _responses_stream_error(message: str, code: str, request_id: str) -> bytes:
    payload = {"type": "error", "error": {"message": message, "type": "upstream_error", "code": code},
               "request_id": request_id}
    return ("event: error\ndata: " + json.dumps(payload) + "\n\n").encode()


def _responses_usage(data: dict) -> dict | None:
    usage = data.get("usage")
    if isinstance(usage, dict):
        return usage
    response = data.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return None


@app.post("/v1/responses")
async def responses(request: Request, authorization: str | None = Header(default=None)):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    if not _authorize(authorization):
        return _openai_error("Invalid API key", 401, error_type="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_error("Request body is too large", 413, code="request_too_large")
    except ValueError as exc:
        return _openai_error(str(exc), 400, code="invalid_json")
    invalid = _validate_responses_request(body)
    if invalid:
        return _openai_error(invalid[0], 400, param=invalid[1], code="invalid_request")

    prompt = _extract_responses_prompt(body)
    prompt_hash = _prompt_hash(prompt)
    session_id, session_source = _session_id(body, request)
    affinity_token = request.headers.get(_RESPONSES_AFFINITY_HEADER.lower())
    affinity_bound = _responses_needs_affinity(body)
    bound_backend = None
    if affinity_bound:
        bound_backend = _responses_affinity_backend(body, affinity_token)
        if bound_backend is None:
            return _openai_error(
                "Responses continuation requires a valid route affinity for its "
                "origin model; the bound target is unavailable, stale, or not "
                "configured. Start a new conversation without encrypted reasoning "
                "or compaction items, or send the x-route-responses-affinity header "
                "issued by the originating response.",
                409, error_type="configuration_error",
                code="responses_continuation_affinity_required",
            )
        decision = bound_backend.get("target", bound_backend.get("tier"))
        backend = bound_backend
        route_reason = "responses_affinity"
        protocol_upgraded = False
        pinned = False
        score = None
        supra_complexity = None
        supra_ms = None
    else:
        if prompt.strip():
            proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(
                _decide, prompt, session_id, "responses")
        else:
            proposed, score, supra_complexity, supra_ms = _default_target(), 0.0, None, None
        decision, route_reason = _session_route(
            session_id, prompt, proposed, supra_complexity, score,
            new_task=request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"},
            api_format="responses",
        )
        compatible = _responses_tier(decision)
        protocol_upgraded = compatible is not None and compatible != decision
        if compatible is None:
            return _openai_error(
                "Responses API has no configured compatible target",
                503, error_type="configuration_error", code="responses_backend_unavailable",
            )
        if protocol_upgraded:
            decision, route_reason = compatible, "responses_protocol_upgrade"
        backend = _backend_for(decision)
        # Protocol-scoped pins are resolved by _decide. Re-check only for honest
        # telemetry, and never claim a pin after a protocol promotion.
        pinned = (not protocol_upgraded and session_id is None
                  and _store_pinned(prompt_hash, api_format="responses") is not None)
    occurrence_id = uuid.uuid4().hex
    _record_and_detect_retry(_request_hash({"messages": body.get("input")}), decision,
                             backend["model"], prompt_hash, request_id, occurrence_id)
    headers = _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                             pinned=pinned, api_format="responses", supra_reason=_supra_reason(supra_complexity))
    headers["x-route-reason"] = route_reason
    headers["x-route-sticky"] = str(route_reason in _STICKY_REASONS).lower()
    if session_id:
        headers["x-route-session"] = session_id

    deadline = time.monotonic() + TIMEOUT_S
    upstream = None
    response_stream = None
    try:
        if affinity_bound:
            bound, upstream, attempts, response_stream = await _responses_send_bound(
                body, backend, deadline, stream=bool(body.get("stream")))
            if bound is not None:
                selected = bound.get("target", bound.get("tier"))
                backend = bound
            else:
                selected = decision
        elif body.get("stream"):
            selected, backend, upstream, attempts, response_stream = await _open_responses_stream_with_failover(
                body, decision, deadline)
        else:
            selected, backend, upstream, attempts = await _open_with_failover(
                body, decision, deadline, stream=False, api_format="responses",
            )
    except BaseException:
        if upstream is not None:
            await asyncio.shield(upstream.aclose())
        raise
    if not affinity_bound:
        _record_refusal_learning(prompt_hash, attempts, session_id, supra_complexity,
                                 api_format="responses")
    headers.update({
        "x-route-decision": selected, "x-route-target": str(backend.get("target", selected)),
        "x-route-model": backend["model"],
        "x-route-fallback": str(selected != decision).lower(), "x-route-attempts": str(len(attempts)),
        "x-route-switch": str(protocol_upgraded or selected != decision).lower(),
        "x-route-affinity": _session_affinity(session_id, "responses"),
    })
    _log_attempts(attempts, prompt, score, request_id, occurrence_id, api_format="responses")
    if upstream is None:
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=None,
             error="upstream_unavailable", attempts=len(attempts), pinned=pinned,
             tier=backend.get("tier"), route_reason=route_reason, session_id=session_id,
             session_source=session_source, api_format="responses")
        return _openai_error("Responses upstream was unavailable", 502,
                             error_type="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                content = await upstream.aread()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError):
            await upstream.aclose()
            return _openai_error("Upstream response timed out", 504,
                                 error_type="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        usage = _responses_usage(data)
        refused = _is_refusal(upstream.status_code, data)
        clean_success = (upstream.status_code == 200 and data.get("status") == "completed"
                         and not refused)
        if upstream.status_code != 200:
            _log_outcome(prompt_hash, "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"],
                         status=upstream.status_code)
        elif refused:
            _log_outcome(prompt_hash, "refusal", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        elif not clean_success:
            _log_outcome(prompt_hash, "truncated" if data.get("status") == "incomplete" else "upstream_error",
                         request_id=request_id, decision_occurrence_id=occurrence_id,
                         model=backend["model"], responses_status=data.get("status"))
        # _open_with_failover already recorded the explicit refusal in its
        # attempt graph before this wire object was returned.  Do not count it
        # a second time merely because a final fallback was unavailable.
        affinity_token = None
        if clean_success:
            # A completed native response may be continued later via
            # previous_response_id, conversation, or replayed opaque items.
            # Index its origin and, when the response is complete (non-stream),
            # return a router-issued affinity capability header.
            affinity_token = _responses_affinity_issue(backend, data)
            if not affinity_bound:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score, api_format="responses")
                else:
                    _session_note(session_id, selected, supra_complexity, usage, api_format="responses")
        _log(selected, score, backend["model"], prompt, None, supra_complexity, supra_ms,
             cost_usd=_extract_cost(data), usage=usage, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", attempts=len(attempts),
             status=upstream.status_code, pinned=pinned, tier=backend.get("tier"),
             route_reason=route_reason, session_id=session_id, session_source=session_source,
             api_format="responses")
        response_headers = {**headers, "content-type": upstream.headers.get("content-type", "application/json")}
        if affinity_token:
            response_headers[_RESPONSES_AFFINITY_HEADER] = affinity_token
        return Response(content=content, status_code=upstream.status_code,
                        media_type=None, headers=response_headers)

    if upstream.status_code != 200:
        content = await upstream.aread()
        response_headers = {**headers, "content-type": upstream.headers.get("content-type", "application/json")}
        await upstream.aclose()
        return Response(content=content, status_code=upstream.status_code,
                        media_type=None, headers=response_headers)

    async def response_events():
        completed = failed = False
        terminal_failure: str | None = None
        usage: dict = {}
        started = time.monotonic()
        try:
            async with asyncio.timeout(max(0, deadline - time.monotonic())):
                if response_stream is None:
                    prefix, events = [], _iter_sse_events(upstream)
                else:
                    prefix, events = response_stream

                async def all_events():
                    for event in prefix:
                        yield event
                    async for event in events:
                        yield event

                async for event in all_events():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    event_completed, event_failure, event_usage, _event_response_id = _responses_event_state(event)
                    if event_usage:
                        usage.update(event_usage)
                    # Index native continuation state as it streams so a later
                    # previous_response_id / conversation / replayed opaque item
                    # resolves to this exact origin without any custom header.
                    stream_payload = _responses_payload(event)
                    if stream_payload is not None:
                        _responses_affinity_record_payload(backend, stream_payload)
                    yield event
                    completed = completed or event_completed
                    if event_failure:
                        terminal_failure = event_failure
                        failed = True
                    # A native refusal is visible output rather than a protocol
                    # conversion opportunity at this point. Preserve later
                    # provider terminal frames (for example response.completed)
                    # while withholding positive learning; transport/errors and
                    # incomplete frames retain their immediate termination.
                    if completed or (failed and terminal_failure != "refusal"):
                        break
            if failed:
                _log_outcome(prompt_hash, "refusal" if terminal_failure == "refusal" else "upstream_error",
                             request_id=request_id, decision_occurrence_id=occurrence_id,
                             model=backend["model"], responses_terminal_error=True)
            elif not completed:
                _log_outcome(prompt_hash, "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"], abrupt_eof=True)
                yield _responses_stream_error("Upstream Responses stream ended before completion",
                                              "upstream_truncated", request_id)
        except asyncio.CancelledError:
            _log_outcome(prompt_hash, "disconnected", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            _log_outcome(prompt_hash, "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            yield _responses_stream_error("Upstream Responses stream failed",
                                          "upstream_transport_error", request_id)
        finally:
            await asyncio.shield(upstream.aclose())
            _log(selected, score, backend["model"], prompt,
                 int((time.monotonic() - started) * 1000), supra_complexity, supra_ms,
                 usage=usage or None, request_id=request_id, occurrence_id=occurrence_id,
                 record_type="decision", attempts=len(attempts), completed=completed,
                 pinned=pinned, tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source, api_format="responses")
            if terminal_failure == "refusal":
                # Prefix refusals have already been recorded by the bounded
                # preflight. A later refusal after non-refusal bytes is the
                # only case that needs a new learning signal here.
                if not affinity_bound and not any(error == "refusal" for _target, _backend, _status, error in attempts):
                    _record_refusal_learning(
                        prompt_hash, [(selected, backend, upstream.status_code, "refusal")],
                        session_id, supra_complexity, api_format="responses")
            elif completed:
                if not affinity_bound:
                    if session_id is None:
                        _store_note(prompt_hash, selected, ok=True, score=score, api_format="responses")
                    else:
                        _session_note(session_id, selected, supra_complexity, usage or None, api_format="responses")

    # Streamed Responses cannot add headers once frames are committed, so issue
    # the origin capability now; the generator also indexes provider response
    # IDs and opaque items as they stream. The token is an extra agreeing
    # record and never authorizes unknown state on its own.
    stream_affinity_token = _responses_affinity_issue(backend)
    if stream_affinity_token:
        headers[_RESPONSES_AFFINITY_HEADER] = stream_affinity_token
    return StreamingResponse(response_events(), media_type="text/event-stream", headers=headers)


async def _chat_prefetch_with_failover(body: dict, decision: str, deadline: float,
                                        upstream, backend, selected, attempts):
    """Bounded Chat SSE preflight with one compatible fallback.

    Both a structured/prose refusal and a transport failure before any frame
    escapes are failover signals. The fallback is always prefetched before its
    bytes are exposed, and the final attempt's frames are what reach the client.
    """
    prefix, events = [], None
    tried = [item[0] for item in attempts]
    while True:
        try:
            if upstream is None or upstream.status_code != 200:
                break
            try:
                events = _iter_sse_events(upstream)
                prefix, refusal = await _prefetch_sse(events, deadline)
            except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError) as exc:
                if upstream is not None:
                    await upstream.aclose()
                    upstream = None
                attempts.append((selected, backend, None, type(exc).__name__))
                events = None
                refusal = False
            else:
                if refusal:
                    # The buffered prefix belongs to the current target. Mark it
                    # explicitly so API-scoped refusal learning is attributed to
                    # the refusal rather than its retry route.
                    refused, refused_backend, refused_status, _ = attempts[-1]
                    attempts[-1] = (refused, refused_backend, refused_status, "refusal")
            failure = "refusal" if refusal else None
        except asyncio.CancelledError:
            # Cancellation can arrive while the fallback prefetch is blocked
            # before the helper returns it to the caller. Close the current
            # upstream so no response leaks, then propagate.
            if upstream is not None:
                await asyncio.shield(upstream.aclose())
            raise
        if failure is None:
            # Transport failure is visible through a non-refusal attempt error
            # while upstream was closed by the handler above.
            last = attempts[-1][3] if attempts else None
            if last not in (None, "refusal"):
                failure = "transport"
            else:
                return selected, backend, upstream, prefix, events
        # The bound is two distinct targets. A prefetch transport failure appends
        # an extra attempt entry for the same route, so count unique targets.
        if len(tried) >= 2:
            return selected, backend, upstream, prefix, events
        fallback = next((target for target in _chat_routes(decision) if target not in tried), None)
        if fallback is None:
            return selected, backend, upstream, prefix, events
        if upstream is not None:
            await upstream.aclose()
            upstream = None
        selected, backend = fallback, _backend_for(fallback)
        tried.append(fallback)
        try:
            remaining = deadline - time.monotonic()
            async with asyncio.timeout(max(0, remaining)):
                upstream = await _send(backend, body, stream=True, api_format="chat")
            attempts.append((selected, backend, upstream.status_code, None))
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError) as exc2:
            if upstream is not None:
                await upstream.aclose()
                upstream = None
            attempts.append((selected, backend, None, type(exc2).__name__))
            return selected, backend, upstream, [], None
    return selected, backend, upstream, prefix, events


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request, authorization: str | None = Header(default=None),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    request_id = request.headers.get("x-request-id") or f"req_{uuid.uuid4().hex}"
    if not _authorize(authorization):
        return _openai_error("Invalid API key", 401, error_type="authentication_error", code="invalid_api_key")
    try:
        body = await _read_json_body(request)
    except OverflowError:
        return _openai_error("Request body is too large", 413, code="request_too_large")
    except ValueError as exc:
        return _openai_error(str(exc), 400, code="invalid_json")
    invalid = _validate_request(body)
    if invalid:
        return _openai_error(invalid[0], 400, param=invalid[1], code="invalid_request")

    # Extract the opaque session before cache/coalescing so the cache scope
    # cannot cross an affinity boundary. It is reused below for routing.
    cache_session_id, cache_session_source = _session_id(body, request)
    new_task_requested = request.headers.get("x-route-new-task", "").lower() in {"1", "true", "yes"}
    cache_key = _cache_key(body, idempotency_key, session_id=cache_session_id)
    if new_task_requested:
        # An explicit new-task boundary must re-route and re-generate even when
        # body and idempotency key match a prior turn in this session.
        cache_key = None
    cached = _cache_get(cache_key)
    if cached is not None:
        media = "text/event-stream" if body.get("stream") else "application/json"
        result = (cached, 200, media, {"x-route-cache": "hit", "x-route-api": "chat",
                                      "x-route-upstream-path": "/chat/completions"})
        return _replayed_response(result, request_id)
    leader, inflight = await _claim_inflight(cache_key)
    if not leader:
        result = await asyncio.shield(inflight)
        if result is not None:
            return _replayed_response(result, request_id)
        leader, inflight = await _claim_inflight(cache_key)

    upstream = None
    try:
        occurrence_id = uuid.uuid4().hex
        prompt = _extract_prompt(body)
        prompt_hash = _prompt_hash(prompt)
        session_id, session_source = cache_session_id, cache_session_source
        if prompt.strip():
            # Keep the historic call shapes for the Chat default so in-process
            # integrations which replace _decide(prompt) remain compatible.
            # The default api_format is Chat; a session is the only extra
            # argument the legacy path ever received.
            if session_id is None:
                proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(_decide, prompt)
            else:
                proposed, score, supra_complexity, supra_ms = await asyncio.to_thread(
                    _decide, prompt, session_id)
        else:
            proposed, score, supra_complexity, supra_ms = _default_target(), 0.0, None, None
        decision, route_reason = _session_route(
            session_id, prompt, proposed, supra_complexity, score,
            new_task=new_task_requested,
            api_format="chat",
        )
        compatible = _chat_tier(decision)
        protocol_upgraded = compatible is not None and compatible != decision
        if compatible is None:
            _finish_inflight_nowait(cache_key, inflight, None)
            return _openai_error("Chat Completions has no configured compatible target", 503,
                                 error_type="configuration_error", code="chat_backend_unavailable")
        if protocol_upgraded:
            decision, route_reason = compatible, "chat_protocol_upgrade"
        pinned = (not protocol_upgraded and session_id is None
                  and _store_pinned(prompt_hash, api_format="chat") is not None)
        backend = _backend_for(decision)
        _record_and_detect_retry(_request_hash(body), decision, backend["model"], prompt_hash,
                                 request_id, occurrence_id)
        headers = _route_headers(decision, score, backend, request_id, supra_complexity, supra_ms,
                                 pinned=pinned, api_format="chat", supra_reason=_supra_reason(supra_complexity))
        headers["x-route-reason"] = route_reason
        headers["x-route-sticky"] = str(route_reason in _STICKY_REASONS).lower()
        if session_id:
            headers["x-route-session"] = session_id
        deadline = time.monotonic() + TIMEOUT_S
        selected, backend, upstream, attempts = await _open_with_failover(
            body, decision, deadline, stream=bool(body.get("stream")), api_format="chat")

        prefix = []
        events = None
        if upstream is not None and body.get("stream") and upstream.status_code == 200:
            # A refusal OR a transport failure during preflight (before any
            # frame escapes) is a bounded failover signal: close and try one
            # unused compatible route inside the shared deadline.
            selected, backend, upstream, prefix, events = await _chat_prefetch_with_failover(
                body, decision, deadline, upstream, backend, selected, attempts)

        _record_refusal_learning(prompt_hash, attempts, session_id, supra_complexity,
                                 api_format="chat")

    except BaseException:
        _finish_inflight_nowait(cache_key, inflight, None)
        if upstream is not None:
            await asyncio.shield(upstream.aclose())
        raise

    headers.update({"x-route-decision": selected, "x-route-target": str(backend.get("target", selected)),
                    "x-route-model": backend["model"],
                    "x-route-fallback": str(selected != decision).lower(), "x-route-attempts": str(len(attempts)),
                    "x-route-switch": str(protocol_upgraded or selected != decision).lower(),
                    "x-route-affinity": _session_affinity(session_id, "chat")})
    _log_attempts(attempts, prompt, score, request_id, occurrence_id, api_format="chat")
    if upstream is None:
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=None,
             error="upstream_unavailable", attempts=len(attempts), pinned=pinned,
             tier=backend.get("tier"), route_reason=route_reason,
             session_id=session_id, session_source=session_source)
        _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                     decision_occurrence_id=occurrence_id, attempts=len(attempts))
        await _finish_inflight(cache_key, inflight, None)
        return _openai_error("Upstream providers were unavailable", 502, error_type="upstream_error", code="upstream_unavailable")

    if not body.get("stream"):
        try:
            remaining = deadline - time.monotonic()
            async with asyncio.timeout(max(0, remaining)):
                content = await upstream.aread()
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError):
            await upstream.aclose()
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            _log(selected, score, backend["model"], prompt, None, request_id=request_id,
                 occurrence_id=occurrence_id, record_type="decision", status=504,
                 error="upstream_timeout", attempts=len(attempts), pinned=pinned,
                 tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source)
            await _finish_inflight(cache_key, inflight, None)
            return _openai_error("Upstream response timed out", 504, error_type="upstream_error", code="upstream_timeout")
        finally:
            await upstream.aclose()
        data = _safe_json(content)
        refused = _is_refusal(upstream.status_code, data)
        if upstream.status_code != 200:
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"], status=upstream.status_code)
        elif refused:
            _log_outcome(_prompt_hash(prompt), "refusal", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        truncated = _length_truncated(data)
        if truncated:
            _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        # A complete non-stream result must contain every requested choice with
        # a terminal finish reason; a lone choice 0 with n=2 is not a success.
        choices_data = data.get("choices")
        n_expected = int(body.get("n", 1) or 1)
        choice_indexes = [choice.get("index", 0) for choice in choices_data
                          if isinstance(choice, dict)] if isinstance(choices_data, list) else []
        complete = (len(choice_indexes) == len(set(choice_indexes))
                    and set(choice_indexes) == set(range(n_expected))
                    and all(isinstance(choice, dict) and choice.get("finish_reason")
                            for choice in choices_data if isinstance(choice, dict)))
        if upstream.status_code == 200 and not refused and not complete and not truncated:
            _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
        if upstream.status_code == 200 and not refused:
            _cache_put(cache_key, body, content)
            if not truncated and complete:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score, api_format="chat")
                else:
                    _session_note(session_id, selected, supra_complexity, data.get("usage"), api_format="chat")
        _log(selected, score, backend["model"], prompt, None, supra_complexity, supra_ms,
             cost_usd=_extract_cost(data), usage=data.get("usage"), request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", attempts=len(attempts),
             status=upstream.status_code, pinned=pinned, tier=backend.get("tier"),
             route_reason=route_reason, session_id=session_id, session_source=session_source)
        result = ((content, upstream.status_code, "application/json", headers)
                  if (cache_key and upstream.status_code == 200 and len(content) <= RESP_CACHE_MAX_BYTES
                      and _response_replay_safe(body, content)) else None)
        await _finish_inflight(cache_key, inflight, result)
        return Response(content=content, status_code=upstream.status_code, media_type="application/json", headers=headers)

    if upstream.status_code != 200 or events is None:
        content = await upstream.aread()
        await upstream.aclose()
        _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                     decision_occurrence_id=occurrence_id, model=backend["model"], status=upstream.status_code)
        _log(selected, score, backend["model"], prompt, None, request_id=request_id,
             occurrence_id=occurrence_id, record_type="decision", status=upstream.status_code,
             pinned=pinned, tier=backend.get("tier"), route_reason=route_reason,
             session_id=session_id, session_source=session_source)
        await _finish_inflight(cache_key, inflight, None)
        return Response(content=content, status_code=upstream.status_code, media_type="application/json", headers=headers)

    async def event_stream():
        cache_parts: list[bytes] | None = [] if cache_key is not None else None
        cache_size = 0
        saw_done = saw_finish = saw_length = saw_refusal = False
        emitted_error = False
        usage: dict = {}
        started = time.monotonic()

        def remember(event: bytes) -> None:
            nonlocal cache_parts, cache_size
            if cache_parts is None:
                return
            cache_size += len(event)
            if cache_size > RESP_CACHE_MAX_BYTES:
                cache_parts = None
            else:
                cache_parts.append(event)

        finished_indexes: set[int] = set()
        saw_explicit_index = False
        n_expected = int(body.get("n", 1) or 1)

        def track(event: bytes) -> bool:
            nonlocal saw_done, saw_finish, saw_length, saw_refusal, saw_explicit_index
            data_text = _sse_data(event)
            if data_text is None:
                return False
            if data_text.strip() == "[DONE]":
                saw_done = True
                return True
            try:
                payload = json.loads(data_text)
            except json.JSONDecodeError:
                return False
            # Any choice can carry a structured refusal or content filter; a
            # normal first choice must not mask it.
            if _is_refusal(200, payload):
                saw_refusal = True
            choices = payload.get("choices") or []
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                if choice.get("finish_reason") is not None:
                    index = choice.get("index")
                    if isinstance(index, int):
                        saw_explicit_index = True
                    finished_indexes.add(index if isinstance(index, int) else 0)
                    if choice.get("finish_reason") == "length":
                        saw_length = True
            # Completion requires every expected choice to finish with a valid,
            # in-range index. A lone/duplicate choice 0 with n=2 must not be
            # treated as a complete success; without explicit indexes, n>1
            # completion cannot be verified and is withheld.
            if n_expected == 1:
                saw_finish = 0 in finished_indexes
            else:
                saw_finish = saw_explicit_index and finished_indexes == set(range(n_expected))
            if isinstance(payload.get("usage"), dict):
                usage.update(payload["usage"])
            return False

        try:
            remaining = deadline - time.monotonic()
            async with asyncio.timeout(max(0, remaining)):
                async def all_events():
                    for event in prefix:
                        yield event
                    async for event in events:
                        yield event
                async for event in all_events():
                    if await request.is_disconnected():
                        raise asyncio.CancelledError
                    done = track(event)
                    if done and not saw_finish:
                        saw_done = False
                        emitted_error = True
                        yield _stream_error("Upstream stream ended without a finish reason", "upstream_truncated", request_id)
                        break
                    remember(event)
                    yield event
                    if done:
                        break
            if saw_refusal:
                _log_outcome(_prompt_hash(prompt), "refusal", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"])
            elif not saw_done or not saw_finish:
                _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"], abrupt_eof=True)
                if not saw_done and not emitted_error:
                    yield _stream_error("Upstream stream ended before completion", "upstream_truncated", request_id)
            elif saw_length:
                _log_outcome(_prompt_hash(prompt), "truncated", request_id=request_id,
                             decision_occurrence_id=occurrence_id, model=backend["model"])
            elif cache_parts is not None:
                content = b"".join(cache_parts)
                _cache_put(cache_key, body, content)
        except asyncio.CancelledError:
            _log_outcome(_prompt_hash(prompt), "disconnected", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            raise
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError, ValueError):
            _log_outcome(_prompt_hash(prompt), "upstream_error", request_id=request_id,
                         decision_occurrence_id=occurrence_id, model=backend["model"])
            yield _stream_error("Upstream stream failed", "upstream_transport_error", request_id)
        finally:
            await upstream.aclose()
            _log(selected, score, backend["model"], prompt, int((time.monotonic() - started) * 1000),
                 supra_complexity, supra_ms, usage=usage or None, request_id=request_id,
                 occurrence_id=occurrence_id, record_type="decision",
                 attempts=len(attempts), completed=saw_done and saw_finish and not saw_refusal, pinned=pinned,
                 tier=backend.get("tier"), route_reason=route_reason,
                 session_id=session_id, session_source=session_source)
            if saw_refusal:
                # Prefix refusals were recorded before the stream escaped. A
                # refusal that appears only later still needs an explicit
                # API-scoped failure signal, never a positive success note.
                if not any(error == "refusal" for _target, _backend, _status, error in attempts):
                    _record_refusal_learning(
                        prompt_hash, [(selected, backend, upstream.status_code, "refusal")],
                        session_id, supra_complexity, api_format="chat")
            elif saw_done and saw_finish and not saw_length:
                if session_id is None:
                    _store_note(prompt_hash, selected, ok=True, score=score, api_format="chat")
                else:
                    _session_note(session_id, selected, supra_complexity, usage or None, api_format="chat")
            result = None
            if cache_parts is not None and saw_done and saw_finish and not saw_refusal and not saw_length:
                content = b"".join(cache_parts)
                if _response_replay_safe(body, content):
                    result = (content, 200, "text/event-stream", headers)
            await _finish_inflight(cache_key, inflight, result)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


def _bind_is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "::1", "localhost"}


def _validate_bind_security() -> None:
    if not _bind_is_loopback(HOST) and (not os.environ.get("ROUTELLM_KEY") or SERVER_KEY == "sk-route-local"):
        raise RuntimeError("non-loopback binding requires an externally supplied, non-default ROUTELLM_KEY")


if __name__ == "__main__":
    import uvicorn
    _validate_bind_security()
    print(
        "effective config: "
        "router=supra "
        f"expensive={EXPENSIVE['model']} middle={MIDDLE['model']} "
        f"middle_effort={MIDDLE['effort']} cheap={CHEAP['model']} port={PORT}",
        flush=True,
    )
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
