"""Shared, secret-free catalog schema for the Mantis and router consumers.

The identifier grammar and the adapter/protocol table are identical to the
Mantis router consumer so one catalog can feed both servers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

# Identifier grammar shared with the router consumer.
TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONTRACT = re.compile(r"[0-9a-f]{64}\Z")
PROTOCOLS = frozenset({"chat_completions", "responses", "anthropic_messages"})
EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
# Adapter -> supported wire protocols. The Anthropic adapter retains signed
# thinking blocks across tool continuations for Claude on Bedrock.
ADAPTER_PROTOCOLS = {
    "openrouter": frozenset({"chat_completions", "responses"}),
    "opencode-go": frozenset({"chat_completions"}),
    "modal": frozenset({"chat_completions"}),
    "openai-compatible": frozenset({"chat_completions", "responses"}),
    "anthropic": frozenset({"anthropic_messages"}),
}
ADAPTERS = frozenset(ADAPTER_PROTOCOLS)
SWITCHYARD_FORMATS = frozenset({"openai_chat", "openai_responses", "anthropic_messages"})
ADAPTER_SWITCHYARD_FORMAT = {
    "openrouter": "openai_chat",
    "opencode-go": "openai_chat",
    "modal": "openai_chat",
    "openai-compatible": "openai_chat",
    "anthropic": "anthropic_messages",
}
BASE_TARGET_ROLES = ("efficient", "capable")
BASE_SECTION_KEYS = frozenset(
    {
        "revision",
        "route_id",
        "algorithm",
        "picker",
        "confidence_threshold",
        "recent_turn_window",
        "confirmations",
        "targets",
    }
)
BASE_TARGET_KEYS = frozenset({"efficient", "capable", "judge"})


class CatalogError(ValueError):
    """Raised when the shared catalog cannot safely configure Mantis."""


@dataclass(frozen=True)
class ProviderBinding:
    adapter: str
    base_url: str
    credential_env: str
    protocols: tuple[str, ...]


@dataclass(frozen=True)
class WorkerBinding:
    provider: str
    upstream_model: str
    model_identity: str
    reasoning_effort: str | None
    protocols: tuple[str, ...]
    max_tokens: int | None = None


@dataclass(frozen=True)
class RuntimeBindings:
    providers: dict[str, ProviderBinding]
    workers: dict[str, WorkerBinding]


@dataclass(frozen=True)
class BaseTarget:
    role: Literal["efficient", "capable", "judge"]
    provider: str
    upstream_model: str
    protocols: tuple[str, ...]
    reasoning_effort: str | None
    max_tokens: int | None
    wire_format: str


@dataclass(frozen=True)
class BaseRoute:
    revision: str
    route_id: str
    algorithm: Literal["stage_router", "escalation"]
    picker: Literal["efficient_first", "capable_first"]
    confidence_threshold: float
    recent_turn_window: int
    confirmations: int
    efficient: BaseTarget
    capable: BaseTarget
    judge: BaseTarget | None
    providers: dict[str, ProviderBinding]


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be a table")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not TARGET_ID_RE.fullmatch(value):
        raise CatalogError(f"{label} must be an identifier")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CatalogError(f"{label} must be a non-empty trimmed string")
    return value


def _url(value: Any, label: str) -> str:
    raw = _string(value, label)
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as error:
        raise CatalogError(f"{label} has an invalid port") from error
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise CatalogError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise CatalogError(f"{label} must not contain user information")
    if parsed.query or parsed.fragment:
        raise CatalogError(f"{label} must not contain a query or fragment")
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = f"{host}:{port}" if port is not None and not default_port else host
    return urlunsplit((parsed.scheme.lower(), netloc, parsed.path.rstrip("/"), "", ""))


def _protocols(value: Any, label: str, required: bool = False) -> tuple[str, ...]:
    if value is None:
        if required:
            raise CatalogError(f"{label} must be an explicit non-empty protocol list")
        return ()
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise CatalogError(f"{label} must be a non-empty protocol list")
    result = tuple(value)
    if len(set(result)) != len(result) or not set(result) <= PROTOCOLS:
        raise CatalogError(f"{label} contains an unsupported protocol")
    return result


def _provider(value: Any, label: str) -> ProviderBinding:
    table = _mapping(value, label)
    adapter = _string(table.get("adapter"), f"{label}.adapter")
    if adapter not in ADAPTERS:
        raise CatalogError(f"{label}.adapter is unsupported")
    credential_env = _string(table.get("credential_env"), f"{label}.credential_env")
    if not _ENV_NAME.fullmatch(credential_env):
        raise CatalogError(f"{label}.credential_env must name an environment variable")
    protocols = _protocols(table.get("protocols"), f"{label}.protocols")
    if not set(protocols) <= ADAPTER_PROTOCOLS[adapter]:
        raise CatalogError(f"{label}.protocols exceeds adapter capabilities")
    return ProviderBinding(
        adapter,
        _url(table.get("base_url"), f"{label}.base_url"),
        credential_env,
        protocols,
    )


def _worker(value: Any, label: str) -> WorkerBinding:
    table = _mapping(value, label)
    upstream_model = _string(table.get("upstream_model"), f"{label}.upstream_model")
    if any(char in upstream_model for char in ",|\r\n"):
        raise CatalogError(f"{label}.upstream_model contains a reserved character")
    model_identity = table.get("model_identity", upstream_model)
    model_identity = _string(model_identity, f"{label}.model_identity")
    if any(char in model_identity for char in ",|\r\n"):
        raise CatalogError(f"{label}.model_identity contains a reserved character")
    effort = table.get("reasoning_effort")
    if effort is not None:
        effort = _string(effort, f"{label}.reasoning_effort")
        if effort not in EFFORTS:
            raise CatalogError(f"{label}.reasoning_effort is unsupported")
        if effort == "none":
            effort = None
    max_tokens = table.get("max_tokens")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise CatalogError(f"{label}.max_tokens must be a positive integer")
    return WorkerBinding(
        _identifier(table.get("provider"), f"{label}.provider"),
        upstream_model,
        model_identity,
        effort,
        _protocols(table.get("protocols"), f"{label}.protocols", required=True),
        max_tokens,
    )


def _runtime_bindings(providers_raw: Any, workers_raw: Any) -> RuntimeBindings:
    provider_table = _mapping(providers_raw, "providers")
    worker_table = _mapping(workers_raw, "mantis.workers")
    workers = {
        _identifier(name, "mantis.workers key"): _worker(value, f"mantis.workers.{name}")
        for name, value in worker_table.items()
    }
    provider_names = {worker.provider for worker in workers.values()}
    providers: dict[str, ProviderBinding] = {}
    for name in provider_names:
        if name not in provider_table:
            raise CatalogError(f"mantis worker references unknown provider {name}")
        providers[name] = _provider(provider_table[name], f"providers.{name}")
    for name, worker in workers.items():
        provider_protocols = providers[worker.provider].protocols
        if provider_protocols and not set(worker.protocols) <= set(provider_protocols):
            raise CatalogError(f"mantis.workers.{name}.protocols exceeds provider protocols")
        if not set(worker.protocols) <= ADAPTER_PROTOCOLS[providers[worker.provider].adapter]:
            raise CatalogError(f"mantis.workers.{name}.protocols exceeds adapter capabilities")
    return RuntimeBindings(providers, workers)


def _slot_order(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != 7:
        raise CatalogError("mantis.slot_order must contain exactly seven stable slot IDs")
    slots = tuple(_identifier(item, "mantis.slot_order item") for item in value)
    if len(set(slots)) != len(slots):
        raise CatalogError("mantis.slot_order must not contain duplicates")
    return slots


def _positive_int(value: Any, label: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CatalogError(f"{label} must be a positive integer")
    return value


def _unit_interval(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CatalogError(f"{label} must be a number in [0, 1]")
    number = float(value)
    if number < 0.0 or number > 1.0:
        raise CatalogError(f"{label} must be a number in [0, 1]")
    return number


def _switchyard_format(provider: ProviderBinding, raw: Any, label: str) -> str:
    default = ADAPTER_SWITCHYARD_FORMAT[provider.adapter]
    if raw is None:
        return default
    fmt = _string(raw, label)
    if fmt not in SWITCHYARD_FORMATS:
        raise CatalogError(f"{label} is unsupported")
    if fmt == "anthropic_messages" and provider.adapter != "anthropic":
        raise CatalogError(f"{label} requires the anthropic adapter")
    if fmt != "anthropic_messages" and provider.adapter == "anthropic":
        raise CatalogError(f"{label} must be anthropic_messages for the anthropic adapter")
    return fmt


def _base_target(
    value: Any,
    role: Literal["efficient", "capable", "judge"],
    providers: Mapping[str, ProviderBinding],
) -> BaseTarget:
    label = f"base.targets.{role}"
    table = _mapping(value, label)
    provider_name = _identifier(table.get("provider"), f"{label}.provider")
    if provider_name not in providers:
        raise CatalogError(f"{label} references unknown provider {provider_name}")
    provider = providers[provider_name]
    worker = _worker(value, label)
    if worker.provider != provider_name:
        raise CatalogError(f"{label}.provider is inconsistent")
    if provider.protocols and not set(worker.protocols) <= set(provider.protocols):
        raise CatalogError(f"{label}.protocols exceeds provider protocols")
    if not set(worker.protocols) <= ADAPTER_PROTOCOLS[provider.adapter]:
        raise CatalogError(f"{label}.protocols exceeds adapter capabilities")
    return BaseTarget(
        role,
        provider_name,
        worker.upstream_model,
        worker.protocols,
        worker.reasoning_effort,
        worker.max_tokens,
        _switchyard_format(provider, table.get("format"), f"{label}.format"),
    )


def _load_named_providers(providers_raw: Any, names: set[str]) -> dict[str, ProviderBinding]:
    provider_table = _mapping(providers_raw, "providers")
    providers: dict[str, ProviderBinding] = {}
    for name in sorted(names):
        if name not in provider_table:
            raise CatalogError(f"base target references unknown provider {name}")
        providers[name] = _provider(provider_table[name], f"providers.{name}")
    return providers


def load_base_route(root: Mapping[str, Any]) -> BaseRoute:
    """Parse the catalog [base] route used to generate Switchyard config."""
    if root.get("version") != 1:
        raise CatalogError("catalog version must be 1 when base is configured")
    section = _mapping(root.get("base"), "base")
    unknown_section = sorted(set(section) - BASE_SECTION_KEYS)
    if unknown_section:
        raise CatalogError(f"base contains unknown keys: {', '.join(unknown_section)}")
    algorithm = _string(section.get("algorithm", "stage_router"), "base.algorithm")
    picker = _string(section.get("picker", "efficient_first"), "base.picker")
    targets = _mapping(section.get("targets"), "base.targets")
    unknown_targets = sorted(set(targets) - BASE_TARGET_KEYS)
    if unknown_targets:
        raise CatalogError(
            f"base.targets contains unknown roles: {', '.join(unknown_targets)}"
        )
    missing = [role for role in BASE_TARGET_ROLES if role not in targets]
    if missing:
        raise CatalogError("base.targets must define efficient and capable")
    provider_names = {
        _identifier(
            _mapping(targets[role], f"base.targets.{role}").get("provider"),
            f"base.targets.{role}.provider",
        )
        for role in BASE_TARGET_ROLES
    }
    if "judge" in targets:
        provider_names.add(
            _identifier(
                _mapping(targets["judge"], "base.targets.judge").get("provider"),
                "base.targets.judge.provider",
            )
        )
    providers = _load_named_providers(root.get("providers"), provider_names)
    efficient = _base_target(targets["efficient"], "efficient", providers)
    capable = _base_target(targets["capable"], "capable", providers)
    judge = _base_target(targets["judge"], "judge", providers) if "judge" in targets else None
    if (
        efficient.upstream_model == capable.upstream_model
        and efficient.provider == capable.provider
    ):
        raise CatalogError("base.targets.efficient and capable must be distinct models")
    typed_algorithm: Literal["stage_router", "escalation"]
    match algorithm:
        case "stage_router":
            typed_algorithm = "stage_router"
        case "escalation":
            typed_algorithm = "escalation"
        case _:
            raise CatalogError("base.algorithm must be stage_router or escalation")
    typed_picker: Literal["efficient_first", "capable_first"]
    match picker:
        case "efficient_first":
            typed_picker = "efficient_first"
        case "capable_first":
            typed_picker = "capable_first"
        case _:
            raise CatalogError("base.picker must be efficient_first or capable_first")
    if typed_algorithm == "escalation" and typed_picker != "efficient_first":
        raise CatalogError("escalation routes must use picker efficient_first")
    if typed_algorithm == "stage_router" and judge is not None:
        raise CatalogError("stage_router does not use base.targets.judge")
    return BaseRoute(
        _string(section.get("revision", "unspecified"), "base.revision"),
        _string(section.get("route_id", "mantis-base"), "base.route_id"),
        typed_algorithm,
        typed_picker,
        _unit_interval(section.get("confidence_threshold", 0.5), "base.confidence_threshold"),
        _positive_int(section.get("recent_turn_window", 3), "base.recent_turn_window", 3),
        _positive_int(section.get("confirmations", 2), "base.confirmations", 2),
        efficient,
        capable,
        judge,
        providers,
    )
