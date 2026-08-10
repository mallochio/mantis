"""Trained-router ABI manifest: the immutable, checkpoint-anchored slot contract.

The seven worker slots, their order, the conductor slot, and every slot's
model identity and reasoning effort are training artifacts.  Operators must
not change them without retraining.  ``abi_manifest.json`` records that ABI
together with a SHA-256 fingerprint of the trained router head, so a catalog
cannot silently accept semantic slot changes by regenerating its contract.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from model_catalog_schema import _CONTRACT, EFFORTS, CatalogError, _identifier, _json, _string

_REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AbiManifest:
    path: Path
    slot_order: tuple[str, ...]
    conductor: str
    workers: dict[str, tuple[str, str | None]]
    training_artifact: str
    training_artifact_sha256: str


def _effort(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in EFFORTS:
        raise CatalogError(f"{label} is unsupported")
    return None if value == "none" else value


def _worker_abi(value: Any, label: str) -> tuple[str, str | None]:
    if not isinstance(value, dict):
        raise CatalogError(f"{label} must be a table")
    identity = _string(value.get("model_identity"), f"{label}.model_identity")
    return identity, _effort(value.get("reasoning_effort"), f"{label}.reasoning_effort")


def load_abi_manifest(path: str | Path | None = None) -> AbiManifest:
    """Load the trained ABI manifest and anchor it to its router head."""
    selected = (
        Path(path).expanduser()
        if path is not None
        else _REPO_ROOT / "artifacts" / "abi_manifest.json"
    )
    if not selected.is_file():
        raise CatalogError(f"trained ABI manifest does not exist: {selected}")
    try:
        root = json.loads(selected.read_text())
    except (OSError, ValueError) as error:
        raise CatalogError(f"invalid trained ABI manifest: {selected}") from error
    if not isinstance(root, dict) or root.get("version") != 1:
        raise CatalogError("trained ABI manifest version must be 1")
    slot_order = root.get("slot_order")
    if not isinstance(slot_order, list) or len(slot_order) != 7:
        raise CatalogError("trained ABI manifest must contain exactly seven slot IDs")
    slots = tuple(_identifier(item, "abi slot_order item") for item in slot_order)
    if len(set(slots)) != len(slots):
        raise CatalogError("trained ABI manifest slot_order must not contain duplicates")
    conductor = _identifier(root.get("conductor"), "abi conductor")
    workers_raw = root.get("workers")
    if not isinstance(workers_raw, dict):
        raise CatalogError("abi workers must be a table")
    workers = {
        _identifier(name, "abi workers key"): _worker_abi(value, f"abi workers.{name}")
        for name, value in workers_raw.items()
    }
    if set(workers) != set(slots):
        raise CatalogError("abi workers must contain exactly the slot_order IDs")
    if conductor not in workers:
        raise CatalogError("abi conductor must name a stable slot ID")
    artifact = root.get("training_artifact")
    if not isinstance(artifact, str) or not artifact or artifact != artifact.strip():
        raise CatalogError("abi training_artifact must be a non-empty trimmed path")
    declared = root.get("training_artifact_sha256")
    if not isinstance(declared, str) or not _CONTRACT.fullmatch(declared):
        raise CatalogError("abi training_artifact_sha256 must be a SHA-256 fingerprint")
    artifact_path = Path(artifact).expanduser()
    if not artifact_path.is_absolute():
        artifact_path = _REPO_ROOT / artifact_path
    if not artifact_path.is_file():
        raise CatalogError(f"trained router artifact does not exist: {artifact_path}")
    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual != declared:
        raise CatalogError("trained router artifact fingerprint differs; retraining is required")
    return AbiManifest(selected.resolve(), slots, conductor, workers, artifact, declared)


def abi_contract(manifest: AbiManifest) -> str:
    """Immutable contract value for the anchored slot ABI."""
    payload = {
        "version": 1,
        "training_artifact": manifest.training_artifact,
        "training_artifact_sha256": manifest.training_artifact_sha256,
        "slot_order": list(manifest.slot_order),
        "conductor": manifest.conductor,
        "workers": {
            slot: {
                "model_identity": identity,
                **({"reasoning_effort": effort} if effort is not None else {}),
            }
            for slot, (identity, effort) in manifest.workers.items()
        },
    }
    return hashlib.sha256(_json(payload).encode()).hexdigest()


def abi_mismatch(
    manifest: AbiManifest,
    slot_order: tuple[str, ...],
    conductor: str,
    workers: dict[str, Any],
) -> str | None:
    """First semantic ABI difference, or None when the catalog matches."""
    if slot_order != manifest.slot_order:
        return "slot_order differs from the trained ABI manifest"
    if conductor != manifest.conductor:
        return "conductor differs from the trained ABI manifest"
    for slot in manifest.slot_order:
        binding = workers[slot]
        identity, effort = manifest.workers[slot]
        if binding.model_identity != identity:
            return f"{slot}.model_identity differs from the trained ABI manifest"
        if binding.reasoning_effort != effort:
            return f"{slot}.reasoning_effort differs from the trained ABI manifest"
    return None
