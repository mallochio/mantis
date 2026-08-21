"""Integration tests for lean_server.py / lean.routes using httpx's ASGI transport.

These tests never trigger the app lifespan (so the real Supra model is never
loaded) and never import the legacy ``server.py`` module.
"""

from __future__ import annotations

import json
import os
import sys

import httpx
import pytest


def _ensure_lean_catalog_env():
    """See ``test_lean_routing.py`` for why this is needed: the shared
    ``conftest.py`` fixture targets the legacy flat-schema ``server.py``
    catalog, while ``lean.config`` requires a ``policies``/``active_policy``
    wrapper. This only touches the environment for the one-time import of
    ``lean.config`` (a no-op if it was already imported by another test
    module), then restores the previous value.
    """
    if "lean.config" in sys.modules:
        return
    payload = {
        "version": 1,
        "providers": {
            "zen": {
                "base_url": "https://opencode.ai/zen/go/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions"],
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions", "responses"],
            },
        },
        "targets": {
            "cheap": {
                "provider": "zen",
                "upstream_model": "deepseek-v4-flash",
                "rank": 0,
                "protocols": ["chat_completions"],
                "fallbacks": ["middle"],
            },
            "middle": {
                "provider": "openrouter",
                "upstream_model": "openai/gpt-5.6-terra",
                "rank": 1,
                "protocols": ["chat_completions", "responses"],
                "fallbacks": ["expensive"],
            },
            "expensive": {
                "provider": "openrouter",
                "upstream_model": "openai/gpt-5.6-sol",
                "rank": 2,
                "protocols": ["chat_completions", "responses"],
                "fallbacks": ["middle"],
            },
        },
        "policies": {
            "default": {"complexity_targets": ["cheap", "cheap", "middle", "middle", "expensive"]},
        },
        "active_policy": "default",
    }
    os.environ.setdefault("MANTIS_ROUTER_TEST_CRED", "test-only")
    os.environ.setdefault("MANTIS_ROUTER_KEY", "sk-route-local")
    prev_targets = os.environ.get("MANTIS_ROUTER_TARGETS_JSON")
    prev_catalog = os.environ.pop("AI_ROUTING_CONFIG", None)
    os.environ["MANTIS_ROUTER_TARGETS_JSON"] = json.dumps(payload)
    try:
        import lean.config  # noqa: F401
    finally:
        if prev_targets is None:
            os.environ.pop("MANTIS_ROUTER_TARGETS_JSON", None)
        else:
            os.environ["MANTIS_ROUTER_TARGETS_JSON"] = prev_targets
        if prev_catalog is not None:
            os.environ["AI_ROUTING_CONFIG"] = prev_catalog


_ensure_lean_catalog_env()

import lean.cache as cache  # noqa: E402
import lean.decision as decision  # noqa: E402
import lean.proxy as proxy  # noqa: E402
import lean.routes as routes  # noqa: E402
import lean.session as session  # noqa: E402
import lean.state as state  # noqa: E402
import lean_server  # noqa: E402

AUTH = {"Authorization": "Bearer sk-route-local"}

pytestmark = pytest.mark.anyio


class FakeUpstreamResponse:
    """Minimal stand-in for an httpx.Response, enough for _open_with_failover callers."""

    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self._content = content
        self.headers = headers or {}

    async def aread(self):
        return self._content

    async def aclose(self):
        return None


@pytest.fixture(autouse=True)
def reset_lean_state():
    cache._resp_cache.clear()
    cache._inflight.clear()
    decision._decision_cache.clear()
    session._session_cache.clear()
    yield
    cache._resp_cache.clear()
    cache._inflight.clear()
    decision._decision_cache.clear()
    session._session_cache.clear()


@pytest.fixture
def fixed_decision(monkeypatch):
    """Avoid loading the real Supra model; always route to the cheap tier."""

    def fake_decide(prompt, sid=None, api="chat"):
        return "cheap", 0.1, 1, 10

    monkeypatch.setattr(routes, "_decide", fake_decide)
    return fake_decide


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=lean_server.app), base_url="http://router"
    ) as value:
        yield value


def upstream(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1)


async def test_healthz_returns_ready_field(client):
    response = await client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert "ready" in data
    # The lifespan never ran under the plain ASGI transport, so the app
    # should honestly report itself as not-ready.
    assert data["ready"] is False


async def test_healthz_reports_ready_true_when_state_flips(client, monkeypatch):
    monkeypatch.setattr(state, "_READY", True)
    response = await client.get("/healthz")
    assert response.json()["ready"] is True


async def test_chat_completions_nonstream_success_with_mocked_failover(client, fixed_decision, monkeypatch):
    async def fake_open_with_failover(body, decision_, deadline, *, stream, api, effort=None, allow_failover=True):
        backend = routes._backend_for(decision_)
        content = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
        return decision_, backend, FakeUpstreamResponse(200, content), [], None

    monkeypatch.setattr(routes, "_open_with_failover", fake_open_with_failover)
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert response.headers["x-route-decision"] == "cheap"


async def test_chat_completions_falls_back_to_second_backend_on_429(client, fixed_decision, monkeypatch):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    mock = upstream(handler)
    monkeypatch.setattr(proxy, "_client", mock)
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert len(calls) == 2
    assert response.headers["x-route-fallback"] == "true"
    assert response.headers["x-route-attempts"] == "2"
    await mock.aclose()


async def test_chat_completions_stream_returns_sse_chunks_and_done(client, fixed_decision, monkeypatch):
    payload = (
        b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    mock = upstream(
        lambda request: httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"})
    )
    monkeypatch.setattr(proxy, "_client", mock)
    response = await client.post(
        "/v1/chat/completions",
        headers=AUTH,
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hello"}]},
    )
    assert response.status_code == 200
    assert "hi" in response.text
    assert response.text.count("data: [DONE]") == 1
    await mock.aclose()


async def test_responses_returns_mocked_response_object(client, monkeypatch):
    def fake_decide(prompt, sid=None, api="chat"):
        return "middle", None, 1, 10

    monkeypatch.setattr(routes, "_decide", fake_decide)

    async def fake_open_with_failover(body, decision_, deadline, *, stream, api, effort=None, allow_failover=True):
        backend = routes._backend_for(decision_)
        content = json.dumps({"id": "resp_1", "status": "completed", "output": []}).encode()
        return decision_, backend, FakeUpstreamResponse(200, content, {"content-type": "application/json"}), [], None

    monkeypatch.setattr(routes, "_open_with_failover", fake_open_with_failover)
    response = await client.post(
        "/v1/responses",
        headers=AUTH,
        json={"model": "auto", "input": "hello there"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["id"] == "resp_1"
