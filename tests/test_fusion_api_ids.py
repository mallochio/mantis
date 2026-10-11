"""Streaming-safe public ID persistence tests for Fusion tool calls."""

from __future__ import annotations

import base64
import copy
import re
from types import SimpleNamespace

import fusion_api_ids
import pytest
import runs
import serve_config


def _run(run_id: str):
    return SimpleNamespace(
        kind="fusion",
        run_id=run_id,
        pending_tool_calls=[],
        main_messages=[],
        sidekick_messages=[],
        main_lane=None,
        sidekick_lanes=[],
        _fusion_public_tool_call_ids={},
    )


def _fake_redis(monkeypatch):
    values = {}

    def load(run_id):
        stored = values.get(run_id)
        return copy.deepcopy(stored) if stored is not None else None

    def store(run):
        values[run.run_id] = copy.deepcopy(run)

    monkeypatch.setattr(runs, "_redis_get", load)
    monkeypatch.setattr(runs, "_redis_put", store)
    return load, store


@pytest.mark.parametrize("store_kind", ["file", "redis"])
def test_streaming_id_survives_stale_producer_snapshot(store_kind, monkeypatch, tmp_path):
    run_id = "1234567890abcdef1234567890abcdef"
    call_id = "lead:provider-call-λ"
    monkeypatch.setattr(runs, "RUN_STORE", store_kind)
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path))
    if store_kind == "redis":
        load, store = _fake_redis(monkeypatch)
        store(_run(run_id))
        stale_producer = load(run_id)
    else:
        runs._file_put(_run(run_id))
        stale_producer = runs._file_get(run_id)
        path = runs._file_path(run_id)
        initial_bytes = path.read_bytes()

    public_id = fusion_api_ids._encode_fusion_tool_call_id(run_id, call_id)

    if store_kind == "redis":
        assert load(run_id).pending_tool_calls == []
        stale_producer.pending_tool_calls = [{"id": call_id}]
        store(stale_producer)
    else:
        assert path.read_bytes() == initial_bytes
        stale_producer.pending_tool_calls = [{"id": call_id}]
        runs._file_put(stale_producer)
    assert len(public_id) == 40
    assert re.fullmatch(r"[A-Za-z0-9_-]{40}", public_id)
    assert fusion_api_ids._decode_fusion_tool_call_id(public_id) == (run_id, call_id)


def test_legacy_ids_decode_with_legacy_and_persisted_index_formats(monkeypatch):
    run_id = "abcdef0123456789abcdef0123456789"
    run = _run(run_id)
    run._fusion_public_tool_call_ids = {"0": "lead:old-call"}
    monkeypatch.setattr(runs, "RUN_STORE", "memory")
    serve_config._runs[run_id] = run

    assert fusion_api_ids._decode_fusion_tool_call_id(f"f{run_id}-0") == (
        run_id,
        "lead:old-call",
    )
    assert fusion_api_ids._decode_fusion_tool_call_id(f"f{run_id}~lead:older-call") == (
        run_id,
        "lead:older-call",
    )
    serve_config._runs.pop(run_id)


def test_new_fingerprint_collision_is_rejected(monkeypatch):
    run_id = "fedcba9876543210fedcba9876543210"
    run = _run(run_id)
    run.pending_tool_calls = [{"id": "lead:first"}, {"id": "lead:second"}]
    monkeypatch.setattr(runs, "RUN_STORE", "memory")
    monkeypatch.setattr(fusion_api_ids, "_call_token", lambda _call_id: "A" * 16)
    serve_config._runs[run_id] = run
    run_token = base64.urlsafe_b64encode(bytes.fromhex(run_id)).decode().rstrip("=")

    with pytest.raises(ValueError, match="ambiguous"):
        fusion_api_ids._decode_fusion_tool_call_id(f"f{run_token}-{'A' * 16}")
    serve_config._runs.pop(run_id)
