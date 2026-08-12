"""Load Mantis bindings from the shared, secret-free routing catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_catalog_abi import abi_contract, abi_mismatch, load_abi_manifest
from model_catalog_runtime import (
    load_runtime_bindings,
    resolve_runtime_credentials,
    runtime_binding_fingerprint,
)
from model_catalog_schema import (
    CatalogError,
    RuntimeBindings,
    _identifier,
    _json,
    _mapping,
    _runtime_bindings,
    _slot_order,
    _string,
)

__all__ = [
    "CatalogError",
    "MantisCatalog",
    "RuntimeBindings",
    "abi_contract",
    "abi_mismatch",
    "catalog_path",
    "identity_fingerprint",
    "load_abi_manifest",
    "load_mantis_catalog",
    "load_runtime_bindings",
    "render_mantis_environment",
    "resolve_provider_keys",
    "resolve_runtime_credentials",
    "runtime_binding_fingerprint",
    "shell_exports",
]

DEFAULT_CATALOG_RELATIVE = Path(".config/ai-routing/catalog.toml")


@dataclass(frozen=True)
class MantisCatalog:
    path: Path
    bindings: RuntimeBindings
    slot_order: tuple[str, ...]
    conductor: str
    conductor_model: str = ""
    trained_slot_contract: str | None = None


def identity_contract(catalog: MantisCatalog) -> dict[str, Any]:
    """Return only the immutable trained-router slot contract.

    Provider, adapter, endpoint, credential, wire protocol, and the transport
    model name are serving bindings; they may change without retraining.
    ``model_identity`` pins the checkpoint-facing identity.
    """
    return {
        "slot_order": list(catalog.slot_order),
        "conductor": catalog.conductor,
        "workers": [
            {
                "slot": slot,
                "model_identity": catalog.bindings.workers[slot].model_identity,
                "reasoning_effort": catalog.bindings.workers[slot].reasoning_effort,
            }
            for slot in catalog.slot_order
        ],
    }


def identity_fingerprint(catalog: MantisCatalog) -> str:
    return hashlib.sha256(_json(identity_contract(catalog)).encode()).hexdigest()


def _expected_contract(value: Any, required: bool) -> str | None:
    if value is None and not required:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CatalogError("mantis.trained_slot_contract must be a SHA-256 fingerprint")
    return value


def catalog_path(env: Mapping[str, str] | None = None) -> tuple[Path, bool]:
    source = os.environ if env is None else env
    configured = source.get("AI_ROUTING_CONFIG") or source.get("MANTIS_CATALOG_PATH")
    if configured:
        return Path(configured).expanduser(), True
    root = Path(source.get("HOME", str(Path.home()))).expanduser()
    return root / DEFAULT_CATALOG_RELATIVE, False


def load_mantis_catalog(
    path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    require_contract: bool = True,
    manifest: str | Path | None = None,
) -> MantisCatalog | None:
    """Load a catalog and anchor it to the trained ABI manifest, or None."""
    selected, explicit = (Path(path).expanduser(), True) if path is not None else catalog_path(env)
    if not selected.exists():
        if explicit:
            raise CatalogError(f"Mantis catalog does not exist: {selected}")
        return None
    if not selected.is_file():
        raise CatalogError(f"Mantis catalog is not a file: {selected}")
    try:
        with selected.open("rb") as handle:
            root = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"invalid TOML in Mantis catalog: {error}") from error
    if root.get("mantis") is None:
        return None
    if root.get("version") != 1:
        raise CatalogError("catalog version must be 1 when mantis is configured")
    section = _mapping(root["mantis"], "mantis")
    slots = _slot_order(section.get("slot_order"))
    conductor = _identifier(section.get("conductor"), "mantis.conductor")
    conductor_model = _string(section.get("conductor_model", conductor), "mantis.conductor_model")
    bindings = _runtime_bindings(root.get("providers"), section.get("workers"))
    if set(bindings.workers) != set(slots):
        raise CatalogError("mantis.workers must contain exactly the slot_order IDs")
    if conductor not in bindings.workers:
        raise CatalogError("mantis.conductor must name a stable slot ID")
    expected = _expected_contract(section.get("trained_slot_contract"), require_contract)
    catalog = MantisCatalog(selected, bindings, slots, conductor, conductor_model, expected)
    abi = load_abi_manifest(manifest)
    mismatch = abi_mismatch(abi, catalog.slot_order, catalog.conductor, catalog.bindings.workers)
    if mismatch is not None:
        raise CatalogError(f"{mismatch}; retraining is required")
    if expected is not None and expected != abi_contract(abi):
        raise CatalogError(
            "Mantis identity contract differs from the trained ABI manifest; retraining is required"
        )
    return catalog


def render_mantis_environment(catalog: MantisCatalog) -> dict[str, str]:
    providers = {
        name: {
            **{
                "adapter": binding.adapter,
                "base_url": binding.base_url,
                "credential_env": binding.credential_env,
            },
            **({"protocols": list(binding.protocols)} if binding.protocols else {}),
        }
        for name, binding in sorted(catalog.bindings.providers.items())
    }
    workers = {
        name: {
            "provider": binding.provider,
            "upstream_model": binding.upstream_model,
            "model_identity": binding.model_identity,
            **({"reasoning_effort": binding.reasoning_effort} if binding.reasoning_effort else {}),
            **({"max_tokens": binding.max_tokens} if binding.max_tokens else {}),
            "protocols": list(binding.protocols),
        }
        for name, binding in sorted(catalog.bindings.workers.items())
    }
    return {
        "MANTIS_WORKER_MODELS": ",".join(catalog.slot_order),
        "MANTIS_CONDUCTOR_SLOT": catalog.conductor,
        "MANTIS_CONDUCTOR_MODEL": catalog.conductor_model or catalog.conductor,
        "MANTIS_PROVIDER_BINDINGS": _json(providers),
        "MANTIS_WORKER_BINDINGS": _json(workers),
        "MANTIS_IDENTITY_CONTRACT": identity_fingerprint(catalog),
    }


def resolve_provider_keys(
    catalog: MantisCatalog, env: Mapping[str, str] | None = None
) -> dict[str, str]:
    source = os.environ if env is None else env
    keys: dict[str, str] = {}
    for name, binding in catalog.bindings.providers.items():
        value = source.get(binding.credential_env)
        if not value:
            raise CatalogError(f"catalog provider {name} requires {binding.credential_env}")
        keys[name] = value
    return keys


def shell_exports(environment: Mapping[str, str]) -> str:
    return "\n".join(
        f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fingerprint", "validate", "render"))
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--format", choices=("shell", "json"), default="shell")
    args = parser.parse_args()
    if args.command == "fingerprint":
        # The anchored contract comes from the trained ABI manifest; a catalog
        # is only validated when one is given, so bootstrap works from scratch.
        if args.catalog is not None:
            load_mantis_catalog(args.catalog, require_contract=False)
        print(abi_contract(load_abi_manifest()))
        return
    catalog = load_mantis_catalog(args.catalog)
    if catalog is None:
        return
    if args.command == "validate":
        print(f"valid Mantis catalog: {catalog.path} ({len(catalog.slot_order)} slots)")
    else:
        rendered = render_mantis_environment(catalog)
        print(shell_exports(rendered) if args.format == "shell" else json.dumps(rendered, indent=2))


if __name__ == "__main__":
    main()
