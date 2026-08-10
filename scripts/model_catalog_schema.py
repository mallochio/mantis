"""Shared, secret-free catalog schema for the Mantis and llm-router consumers.

The identifier grammar and the adapter/protocol table are identical to the
llm-router (RouteLLM) consumer so one catalog can feed both servers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# Identifier grammar shared with the llm-router consumer.
TARGET_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_CONTRACT = re.compile(r"[0-9a-f]{64}\Z")
PROTOCOLS = frozenset({"chat_completions", "responses"})
EFFORTS = frozenset({"none", "low", "medium", "high", "xhigh", "max"})
# Adapter -> supported wire protocols; identical to the router's schema.
ADAPTER_PROTOCOLS = {
    "openrouter": frozenset({"chat_completions", "responses"}),
    "opencode-go": frozenset({"chat_completions"}),
    "modal": frozenset({"chat_completions"}),
    "openai-compatible": frozenset({"chat_completions", "responses"}),
}
ADAPTERS = frozenset(ADAPTER_PROTOCOLS)


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


@dataclass(frozen=True)
class RuntimeBindings:
    providers: dict[str, ProviderBinding]
    workers: dict[str, WorkerBinding]


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
    return WorkerBinding(
        _identifier(table.get("provider"), f"{label}.provider"),
        upstream_model,
        model_identity,
        effort,
        _protocols(table.get("protocols"), f"{label}.protocols", required=True),
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
