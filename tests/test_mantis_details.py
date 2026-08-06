"""Endpoint DX: opt-in X-Mantis-Details orchestration metadata and headers."""

from __future__ import annotations

import time
from types import SimpleNamespace

import api
import pytest
import serve
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    return TestClient(api.app)


def _headers(**extra):
    auth = {"Authorization": "Bearer test-key"}
    return {**auth, **{key.replace("_", "-"): value for key, value in extra.items()}}


def _run():
    return SimpleNamespace(
        run_id="a" * 32,
        kind="trinity",
        terminated_by="verifier_accept",
        turns=[
            {"role": "Worker", "agent_id": 0, "model_name": "openrouter/acme/worker"},
        ],
        tool_observations=[{"name": "bash", "is_error": False, "is_test": True}],
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        usage_models={"openrouter/acme/worker": {"prompt_tokens": 5, "completion_tokens": 7}},
        _activity=[
            {
                "type": "step",
                "role": "Worker",
                "model": "openrouter/acme/worker",
                "status": "completed",
                "summary": "Drafted the answer",
                "duration_ms": 12.3,
            }
        ],
        _started_monotonic=time.monotonic() - 0.02,
        response_metadata={},
        validate_output=lambda _text: None,
    )


def _bind(client_args, monkeypatch, run, event=None):
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: event or {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)


def test_default_response_has_no_mantis_metadata(client, monkeypatch):
    _bind(None, monkeypatch, _run())
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert "mantis" not in response.json()
    assert "x-mantis-run-id" not in response.headers


def test_summary_header_returns_metadata_and_headers(client, monkeypatch):
    _bind(None, monkeypatch, _run())
    monkeypatch.setattr(
        serve, "_price_cache", {"openrouter/acme/worker": (0.000001, 0.000002)}
    )
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(X_Mantis_Details="summary"),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers["x-mantis-run-id"] == "a" * 32
    assert response.headers["x-mantis-mode"] == "trinity"
    assert response.headers["x-mantis-outcome"] == "verifier_accept"
    assert "x-mantis-duration-ms" in response.headers
    assert response.headers["x-mantis-cost-usd"] == "0.000019"
    body = response.json()
    mantis = body["mantis"]
    assert mantis["mode"] == "trinity"
    assert mantis["outcome"] == "verifier_accept"
    assert [step["summary"] for step in mantis["activity"]] == [
        "Drafted the answer",
        "Run completed",
    ]
    assert "duration_ms" not in mantis["activity"][0]  # summary strips timing detail
    usage = mantis["usage"]
    assert usage["known"] is True
    assert usage["total"] == 0.000019
    assert usage["models"][0]["model"] == "openrouter/acme/worker"


def test_unknown_price_reports_known_false(client, monkeypatch):
    _bind(None, monkeypatch, _run())
    monkeypatch.setattr(serve, "_price_cache", {})
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(X_Mantis_Details="summary"),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    usage = response.json()["mantis"]["usage"]
    assert usage["known"] is False
    assert usage["total"] is None
    assert usage["models"][0]["cost"] is None
    assert "x-mantis-cost-usd" not in response.headers


def test_debug_requires_opt_in_environment(client, monkeypatch):
    _bind(None, monkeypatch, _run())
    monkeypatch.delenv("MANTIS_ALLOW_DEBUG_TRACE", raising=False)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(X_Mantis_Details="debug"),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert "duration_ms" not in response.json()["mantis"]["activity"][0]
    monkeypatch.setenv("MANTIS_ALLOW_DEBUG_TRACE", "1")
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(X_Mantis_Details="debug"),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.json()["mantis"]["activity"][0]["duration_ms"] == 12.3


def test_stream_includes_mantis_frame_when_requested(client, monkeypatch):
    _bind(None, monkeypatch, _run())
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(X_Mantis_Details="summary"),
        json={
            "model": "mantis",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200
    assert '"mantis"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")
