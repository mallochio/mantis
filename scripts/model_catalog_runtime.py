"""Rendered catalog runtime environment: bindings, credentials, fingerprints.

The catalog file is rendered once at launch into ``MANTIS_*`` environment
variables.  This module validates that rendered surface at serving time so a
stale or mutated runtime cannot silently diverge from the trained ABI.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

from model_catalog_schema import (
    CatalogError,
    RuntimeBindings,
    _identifier,
    _json,
    _runtime_bindings,
    _slot_order,
)


def _runtime_abi(
    bindings: RuntimeBindings, source: Mapping[str, str]
) -> tuple[tuple[str, ...], str]:
    slots = _slot_order(source.get("MANTIS_WORKER_MODELS", "").split(","))
    conductor = _identifier(
        source.get("MANTIS_CONDUCTOR_SLOT"),
        "MANTIS_CONDUCTOR_SLOT",
    )
    if set(bindings.workers) != set(slots):
        raise CatalogError("runtime worker bindings must contain exactly the seven stable slots")
    if conductor not in bindings.workers:
        raise CatalogError("runtime conductor must name a stable slot")
    return slots, conductor


def _runtime_contract_hash(
    bindings: RuntimeBindings, slots: tuple[str, ...], conductor: str
) -> str:
    contract = {
        "slot_order": list(slots),
        "conductor": conductor,
        "workers": [
            {
                "slot": slot,
                "model_identity": bindings.workers[slot].model_identity,
                "reasoning_effort": bindings.workers[slot].reasoning_effort,
            }
            for slot in slots
        ],
    }
    return hashlib.sha256(_json(contract).encode()).hexdigest()


def _contract_fingerprint(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise CatalogError("MANTIS_IDENTITY_CONTRACT must be a SHA-256 fingerprint")
    return value


def load_runtime_bindings(env: Mapping[str, str] | None = None) -> RuntimeBindings | None:
    """Load rendered runtime bindings; the seven-slot ABI is always enforced."""
    source = os.environ if env is None else env
    providers = source.get("MANTIS_PROVIDER_BINDINGS", "").strip()
    workers = source.get("MANTIS_WORKER_BINDINGS", "").strip()
    if not providers and not workers:
        return None
    if not providers or not workers:
        raise CatalogError(
            "MANTIS_PROVIDER_BINDINGS and MANTIS_WORKER_BINDINGS must be set together"
        )
    try:
        bindings = _runtime_bindings(json.loads(providers), json.loads(workers))
    except json.JSONDecodeError as error:
        raise CatalogError("Mantis runtime bindings must be JSON objects") from error
    slots, conductor = _runtime_abi(bindings, source)
    expected = source.get("MANTIS_IDENTITY_CONTRACT")
    if expected:
        _contract_fingerprint(expected)
        if _runtime_contract_hash(bindings, slots, conductor) != expected:
            raise CatalogError("Mantis runtime identity contract differs; retraining is required")
    return bindings


def resolve_runtime_credentials(
    bindings: RuntimeBindings, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Resolve provider credentials without exposing their values.

    Launch-injected ``MANTIS_PROVIDER_KEYS`` take priority; the named
    ``credential_env`` variables are the native fallback.
    """
    source = os.environ if env is None else env
    raw = source.get("MANTIS_PROVIDER_KEYS", "").strip()
    keys: dict[str, str] = {}
    if raw:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise CatalogError("MANTIS_PROVIDER_KEYS must be a JSON object") from error
        if not isinstance(value, dict) or not all(
            isinstance(name, str) and isinstance(key, str) and key
            for name, key in value.items()
        ):
            raise CatalogError("MANTIS_PROVIDER_KEYS must map provider ids to non-empty strings")
        keys = value
    missing = [
        f"{name} requires {binding.credential_env}"
        for name, binding in bindings.providers.items()
        if not keys.get(name) and not source.get(binding.credential_env)
    ]
    if missing:
        raise CatalogError("; ".join(missing))
    for name, binding in bindings.providers.items():
        keys[name] = keys.get(name) or source.get(binding.credential_env) or ""
    return keys


def runtime_binding_fingerprint(env: Mapping[str, str] | None = None) -> str:
    """Non-secret fingerprint of the full runtime binding surface.

    Covers slot order, conductor, and every provider/worker binding field.
    Credential values are never part of the fingerprint; readiness compares
    it so a mutable same-host binding change is detected on an existing
    container instead of silently kept.
    """
    source = os.environ if env is None else env
    bindings = load_runtime_bindings(source)
    if bindings is None:
        raise CatalogError("catalog runtime bindings are not configured")
    slots, conductor = _runtime_abi(bindings, source)
    providers = {
        name: {
            "adapter": binding.adapter,
            "base_url": binding.base_url,
            "credential_env": binding.credential_env,
            "protocols": list(binding.protocols),
        }
        for name, binding in sorted(bindings.providers.items())
    }
    workers = {
        name: {
            "provider": binding.provider,
            "upstream_model": binding.upstream_model,
            "model_identity": binding.model_identity,
            "reasoning_effort": binding.reasoning_effort,
            **({"max_tokens": binding.max_tokens} if binding.max_tokens else {}),
            "protocols": list(binding.protocols),
        }
        for name, binding in sorted(bindings.workers.items())
    }
    payload = {
        "slot_order": list(slots),
        "conductor": conductor,
        "providers": providers,
        "workers": workers,
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()
