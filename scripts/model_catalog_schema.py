"""Shared, secret-free catalog schema for Mantis workers and the Base route.

Trinity/Ultra/Fusion read ``[mantis.workers]``. Switchyard Base reads ``[base]``.
Both share the provider/adapter table; Base does not reuse the worker ABI.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

# Identifier grammar for catalog table keys and provider names.
TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONTRACT = re.compile(r"[0-9a-f]{64}\Z")
PROTOCOLS = frozenset({"chat_completions", "responses", "anthropic_messages"})
EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
# Ordered weakest to strongest. Reasoning tokens bill as output tokens, so this
# is a cost ordering when the upstream model is the same for both targets.
_EFFORT_RANK = ("none", "low", "medium", "high", "xhigh", "max")
# Adapter -> supported wire protocols. The Anthropic adapter retains signed
# thinking blocks across tool continuations for Claude on Bedrock.
ADAPTER_PROTOCOLS = {
    "openrouter": frozenset({"chat_completions", "responses"}),
    "opencode-go": frozenset({"chat_completions"}),
    "modal": frozenset({"chat_completions"}),
    "openai-compatible": frozenset({"chat_completions", "responses"}),
    "anthropic": frozenset({"anthropic_messages"}),
    "bedrock": frozenset({"chat_completions", "responses", "anthropic_messages"}),
    "vertex": frozenset({"chat_completions", "responses"}),
    "azure_ai": frozenset({"responses"}),
}
ADAPTERS = frozenset(ADAPTER_PROTOCOLS)
SWITCHYARD_FORMATS = frozenset({"openai_chat", "openai_responses", "anthropic_messages"})
ADAPTER_SWITCHYARD_FORMAT = {
    "openrouter": "openai_chat",
    "opencode-go": "openai_chat",
    "modal": "openai_chat",
    "openai-compatible": "openai_chat",
    "anthropic": "anthropic_messages",
    "bedrock": "openai_chat",
    "vertex": "openai_chat",
    "azure_ai": "openai_responses",
}
BASE_TARGET_ROLES = ("efficient", "capable")
BASE_SECTION_KEYS = frozenset(
    {
        "revision",
        "picker",
        "confidence_threshold",
        "recent_turn_window",
        "targets",
    }
)
BASE_TARGET_FIELDS = frozenset(
    {
        "provider",
        "upstream_model",
        "reasoning_effort",
        "max_tokens",
        "format",
    }
)


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
    context_window: int | None = None


@dataclass(frozen=True)
class RuntimeBindings:
    providers: dict[str, ProviderBinding]
    workers: dict[str, WorkerBinding]


@dataclass(frozen=True)
class BaseTarget:
    role: Literal["efficient", "capable"]
    provider: str
    upstream_model: str
    reasoning_effort: str | None
    max_tokens: int | None
    wire_format: str


@dataclass(frozen=True)
class BaseRoute:
    revision: str
    picker: Literal["efficient_first", "capable_first"]
    confidence_threshold: float
    recent_turn_window: int
    efficient: BaseTarget
    capable: BaseTarget
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


def _model_name(value: Any, label: str) -> str:
    name = _string(value, label)
    if any(char in name for char in ",|\r\n"):
        raise CatalogError(f"{label} contains a reserved character")
    return name


def _reasoning_effort(value: Any, label: str) -> str | None:
    if value is None:
        return None
    effort = _string(value, label)
    if effort not in EFFORTS:
        raise CatalogError(f"{label} is unsupported")
    return None if effort == "none" else effort


def _max_tokens(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CatalogError(f"{label} must be a positive integer")
    return value


def _validate_effort_ordering(efficient: BaseTarget, capable: BaseTarget) -> None:
    """Reject an efficient target that reasons harder than the capable one.

    Only applies when both targets route to the same upstream model. Across
    different models, effort is not a reliable cost signal (a cheap model at
    high effort can cost less per turn than a strong model at medium), so the
    guard is skipped there.
    """
    if (
        efficient.upstream_model != capable.upstream_model
        or efficient.provider != capable.provider
    ):
        return
    if efficient.reasoning_effort is None or capable.reasoning_effort is None:
        return
    try:
        low = _EFFORT_RANK.index(efficient.reasoning_effort)
        high = _EFFORT_RANK.index(capable.reasoning_effort)
    except ValueError:  # pragma: no cover - EFFORTS already validated upstream
        return
    if low > high:
        raise CatalogError(
            "base.targets.efficient.reasoning_effort "
            f"({efficient.reasoning_effort}) must not exceed "
            f"base.targets.capable.reasoning_effort ({capable.reasoning_effort}); "
            "reasoning tokens bill as output, so this inverts the cost tiers"
        )


def _validate_token_limits(
    max_tokens: int | None, context_window: int | None, label: str
) -> None:
    """Reject an output cap larger than the model's own context window."""
    if max_tokens is not None and context_window is not None and max_tokens > context_window:
        raise CatalogError(
            f"{label}.max_tokens ({max_tokens}) exceeds {label}.context_window "
            f"({context_window}); max_tokens is the output cap, not the context window"
        )


def _worker(value: Any, label: str) -> WorkerBinding:
    table = _mapping(value, label)
    upstream_model = _model_name(table.get("upstream_model"), f"{label}.upstream_model")
    model_identity = _model_name(
        table.get("model_identity", upstream_model), f"{label}.model_identity"
    )
    max_tokens = _max_tokens(table.get("max_tokens"), f"{label}.max_tokens")
    context_window = _max_tokens(table.get("context_window"), f"{label}.context_window")
    _validate_token_limits(max_tokens, context_window, label)
    return WorkerBinding(
        _identifier(table.get("provider"), f"{label}.provider"),
        upstream_model,
        model_identity,
        _reasoning_effort(table.get("reasoning_effort"), f"{label}.reasoning_effort"),
        _protocols(table.get("protocols"), f"{label}.protocols", required=True),
        max_tokens,
        context_window,
    )


def _runtime_bindings(
    providers_raw: Any,
    workers_raw: Any,
    extra_provider_names: set[str] | None = None,
) -> RuntimeBindings:
    provider_table = _mapping(providers_raw, "providers")
    worker_table = _mapping(workers_raw, "mantis.workers")
    workers = {
        _identifier(name, "mantis.workers key"): _worker(value, f"mantis.workers.{name}")
        for name, value in worker_table.items()
    }
    provider_names = {worker.provider for worker in workers.values()}
    provider_names |= extra_provider_names or set()
    provider_names |= set(provider_table)
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
    role: Literal["efficient", "capable"],
    providers: Mapping[str, ProviderBinding],
) -> BaseTarget:
    label = f"base.targets.{role}"
    table = _mapping(value, label)
    unknown = sorted(set(table) - BASE_TARGET_FIELDS)
    if unknown:
        raise CatalogError(f"{label} contains unknown keys: {', '.join(unknown)}")
    provider_name = _identifier(table.get("provider"), f"{label}.provider")
    if provider_name not in providers:
        raise CatalogError(f"{label} references unknown provider {provider_name}")
    provider = providers[provider_name]
    return BaseTarget(
        role,
        provider_name,
        _model_name(table.get("upstream_model"), f"{label}.upstream_model"),
        _reasoning_effort(table.get("reasoning_effort"), f"{label}.reasoning_effort"),
        _max_tokens(table.get("max_tokens"), f"{label}.max_tokens"),
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
    """Parse the catalog [base] stage-router used to generate Switchyard config."""
    if root.get("version") != 1:
        raise CatalogError("catalog version must be 1 when base is configured")
    section = _mapping(root.get("base"), "base")
    unknown_section = sorted(set(section) - BASE_SECTION_KEYS)
    if unknown_section:
        raise CatalogError(f"base contains unknown keys: {', '.join(unknown_section)}")
    picker = _string(section.get("picker", "efficient_first"), "base.picker")
    targets = _mapping(section.get("targets"), "base.targets")
    unknown_targets = sorted(set(targets) - set(BASE_TARGET_ROLES))
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
    providers = _load_named_providers(root.get("providers"), provider_names)
    efficient = _base_target(targets["efficient"], "efficient", providers)
    capable = _base_target(targets["capable"], "capable", providers)
    if (
        efficient.provider == capable.provider
        and efficient.upstream_model == capable.upstream_model
        and efficient.reasoning_effort == capable.reasoning_effort
        and efficient.max_tokens == capable.max_tokens
        and efficient.wire_format == capable.wire_format
    ):
        raise CatalogError("base.targets.efficient and capable must be distinct targets")
    _validate_effort_ordering(efficient, capable)
    typed_picker: Literal["efficient_first", "capable_first"]
    match picker:
        case "efficient_first":
            typed_picker = "efficient_first"
        case "capable_first":
            typed_picker = "capable_first"
        case _:
            raise CatalogError("base.picker must be efficient_first or capable_first")
    return BaseRoute(
        _string(section.get("revision", "unspecified"), "base.revision"),
        typed_picker,
        _unit_interval(section.get("confidence_threshold", 0.5), "base.confidence_threshold"),
        _positive_int(section.get("recent_turn_window", 3), "base.recent_turn_window", 3),
        efficient,
        capable,
        providers,
    )
