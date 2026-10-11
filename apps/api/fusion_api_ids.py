"""Public Fusion tool-call IDs resolved from persisted run state."""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any

import fusion

_FUSION_TOOL_ID_PREFIX = "f"
_FUSION_TOOL_ID_SEP = "~"
_RUN_ID_LENGTH = 32
_RUN_TOKEN_LENGTH = 22
_CALL_TOKEN_LENGTH = 16
_PUBLIC_ID_LENGTH = 40


def _encode_fusion_tool_call_id(run_id: str, call_id: str) -> str:
    """Build a bounded ID without reading or mutating the producer's run snapshot."""
    if not _is_run_id(run_id):
        return f"{_FUSION_TOOL_ID_PREFIX}{run_id}{_FUSION_TOOL_ID_SEP}{call_id}"
    run_token = base64.urlsafe_b64encode(bytes.fromhex(run_id)).decode("ascii").rstrip("=")
    return f"{_FUSION_TOOL_ID_PREFIX}{run_token}-{_call_token(call_id)}"


def _decode_fusion_tool_call_id(encoded: str) -> tuple[str, str]:
    """Resolve compressed IDs from pending calls and retain previously issued ID formats."""
    if not encoded.startswith(_FUSION_TOOL_ID_PREFIX):
        raise ValueError("invalid fusion tool_call_id")
    if _FUSION_TOOL_ID_SEP in encoded:
        run_id, call_id = encoded[1:].split(_FUSION_TOOL_ID_SEP, 1)
        return run_id, call_id
    if _is_legacy_index_id(encoded):
        return _decode_legacy_index_id(encoded)
    run_id, token = _split_compressed_id(encoded)
    run = _fusion_tool_id_run(run_id)
    if run is None:
        raise ValueError("unknown fusion tool_call_id run")
    matches = _matching_pending_ids(run, token)
    if len(matches) != 1:
        reason = "ambiguous" if matches else "unknown"
        raise ValueError(f"{reason} fusion tool_call_id")
    return run_id, matches[0]


def _is_legacy_index_id(encoded: str) -> bool:
    """Recognize the old full-UUID-plus-short-index format."""
    return len(encoded) >= 35 and _is_run_id(encoded[1:33]) and encoded[33] == "-"


def _decode_legacy_index_id(encoded: str) -> tuple[str, str]:
    """Resolve an old short index only through the map persisted on its run."""
    run_id, token = encoded[1:33], encoded[34:]
    run = _fusion_tool_id_run(run_id)
    mapping = getattr(run, "_fusion_public_tool_call_ids", {}) if run is not None else {}
    call_id = mapping.get(token) if isinstance(mapping, dict) else None
    if not isinstance(call_id, str):
        raise TypeError("invalid legacy fusion tool_call_id mapping")
    return run_id, call_id


def _split_compressed_id(encoded: str) -> tuple[str, str]:
    """Decode the fixed-size compressed UUID and call fingerprint fields."""
    if len(encoded) != _PUBLIC_ID_LENGTH or encoded[23] != "-":
        raise ValueError("invalid fusion tool_call_id")
    run_token, call_token = encoded[1:23], encoded[24:]
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    if any(char not in allowed for char in encoded[1:]):
        raise ValueError("invalid fusion tool_call_id")
    try:
        raw = base64.urlsafe_b64decode(run_token + "==")
    except (ValueError, binascii.Error) as error:
        raise ValueError("invalid fusion tool_call_id") from error
    if len(raw) != 16 or len(call_token) != _CALL_TOKEN_LENGTH:
        raise ValueError("invalid fusion tool_call_id")
    return raw.hex(), call_token


def _matching_pending_ids(run: Any, token: str) -> list[str]:
    """Return distinct pending call IDs matching a new public fingerprint."""
    matches = {
        str(call.get("id"))
        for call in run.pending_tool_calls or []
        if isinstance(call, dict) and call.get("id") and _call_token(str(call["id"])) == token
    }
    return sorted(matches)


def _fusion_tool_id_run(run_id: str) -> Any | None:
    """Return persisted fusion state for the ID's UUID run."""
    try:
        run = fusion.get_run(run_id)
    except KeyError:
        return None
    return run if getattr(run, "kind", None) == "fusion" else None


def _is_run_id(value: str) -> bool:
    """Recognize the lowercase hexadecimal UUID form used by Fusion runs."""
    return len(value) == _RUN_ID_LENGTH and all(char in "0123456789abcdef" for char in value)


def _call_token(call_id: str) -> str:
    """Return a compact URL-safe fingerprint for a full internal call ID."""
    digest = hashlib.sha256(call_id.encode()).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:_CALL_TOKEN_LENGTH]
