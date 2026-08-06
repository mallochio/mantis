"""Provider-facing HTTP contract tests."""

from types import SimpleNamespace

import api
import pytest
import serve
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    return TestClient(api.app)


def _headers():
    return {"Authorization": "Bearer test-key"}


def _run():
    return SimpleNamespace(
        run_id="a" * 32,
        kind="trinity",
        terminated_by="verifier_accept",
        turns=[],
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    )


def test_auth_health_and_validation(client, monkeypatch):
    assert client.get("/health").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=_headers()).status_code == 200
    monkeypatch.delenv("MANTIS_API_KEY")
    assert client.get("/v1/models", headers=_headers()).status_code == 503
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    bad = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis", "messages": [], "unknown": True},
    )
    assert bad.status_code == 422
    no_tools = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        },
    )
    assert no_tools.status_code == 422
    invalid_role_fields = [
        {"role": "tool", "content": "x"},
        {"role": "user", "content": "x", "tool_calls": []},
    ]
    for message in invalid_role_fields:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": "mantis", "messages": [message]},
        )
        assert response.status_code == 422
    assert (
        client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "mantis",
                "messages": [{"role": "user", "content": "hi"}],
                "stream_options": {"include_usage": True},
            },
        ).status_code
        == 422
    )


def test_completion_uses_aggregate_usage_and_hides_trace(client, monkeypatch):
    run = _run()
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "answer"
    assert body["usage"] == run.usage
    assert "mantis" not in body


def test_tool_choice_reaches_run(client, monkeypatch):
    captured = []
    run = _run()

    def create(_mode, body):
        captured.append(body["tool_choice"])
        return run

    monkeypatch.setattr(serve, "create_run", create)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
        },
    )
    assert response.status_code == 200
    assert captured == [{"type": "function", "function": {"name": "read"}}]


def test_buffered_stream_includes_usage(client, monkeypatch):
    run = _run()
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200
    assert response.headers["x-mantis-streaming"] == "buffered"
    assert '"choices": []' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_api_maps_run_errors(client, monkeypatch):
    request = api.ChatRequest(model="mantis", messages=[api.Message(role="user", content="hi")])
    cases = ((KeyError("expired"), 409), (ValueError("bad"), 400), (RuntimeError("upstream"), 502))
    for error, status in cases:
        monkeypatch.setattr(api, "_advance", lambda *_a, error=error: (_ for _ in ()).throw(error))
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=request.model_dump(),
        )
        assert response.status_code == status


def test_usage_accumulator():
    run = serve.NativeRun("usage")
    run.add_usage(None)
    run.add_usage(
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    )
    run.add_usage(
        {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 9,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    )
    assert run.usage == {
        "prompt_tokens": 6,
        "completion_tokens": 8,
        "total_tokens": 14,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }


def test_capacity_returns_429(client, monkeypatch):
    monkeypatch.setattr(api, "_capacity", SimpleNamespace(acquire=lambda **_k: False))
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"
