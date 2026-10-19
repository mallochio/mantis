import json
from types import SimpleNamespace

import base_effort
import base_proxy
import httpx
import pytest


@pytest.fixture(autouse=True)
def _tier_effort_on(monkeypatch):
    monkeypatch.setenv("MANTIS_TIER_EFFORT", "1")


def _route():
    return SimpleNamespace(
        efficient=SimpleNamespace(reasoning_effort="medium", wire_format="openai_chat"),
        capable=SimpleNamespace(reasoning_effort="high", wire_format="openai_chat"),
    )


def _capable_decision():
    return {
        "selected": {
            "target": "capable",
            "model": "~openai/gpt-sol-latest:capable",
            "extra_body": {"reasoning_effort": "high"},
        },
        "fallbacks": [],
    }


def _efficient_decision():
    return {
        "selected": {
            "target": "efficient",
            "model": "~openai/gpt-sol-latest",
            "extra_body": {"reasoning_effort": "medium"},
        },
        "fallbacks": [],
    }


def _forward_request(messages, session):
    class Request:
        def model_dump(self, *, exclude_none, exclude):
            body = {"model": "mantis/base", "messages": messages}
            return {k: v for k, v in body.items() if k not in exclude}

    request = Request()
    request.messages = messages
    request.tools = None
    request.metadata = {"session_id": session}
    request.user = None
    request.stream = False
    return request


def _routing_handler(seen, decision):
    def handler(request):
        seen.append((request.url.path, json.loads(request.content.decode())))
        if request.url.path.endswith("/decision"):
            return httpx.Response(200, json=decision)
        return httpx.Response(
            200,
            json={"id": "c", "choices": [{"message": {"content": "ok"}}]},
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "~openai/gpt-sol-latest:capable"},
        )

    return handler


def test_forward_pins_baseline_and_injects_capable_update(monkeypatch):
    monkeypatch.setattr(base_effort, "_load_base_route", lambda: _route())
    seen = []
    handler = _routing_handler(seen, _capable_decision())
    messages = [{"role": "user", "content": "hard turn"}]
    response, _ = base_proxy.forward(
        _forward_request(messages, "sess-effort"),
        {},
        "req-effort",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.status_code == 200
    decision_path, decision_body = seen[0]
    assert decision_path.endswith("/decision")
    assert decision_body["input_format"] == "openai_chat"
    assert decision_body["request"]["session_id"] == "sess-effort"
    assert "reasoning_effort" not in decision_body["request"]
    completion_path, completion_body = seen[1]
    assert completion_path.endswith("/chat/completions")
    assert completion_body["reasoning_effort"] == "medium"
    assert completion_body["session_id"] == "sess-effort"
    assert completion_body["messages"][0] == {
        "role": "system",
        "content": "",
        "configuration_update": {"reasoning": {"effort": "high"}},
    }
    assert completion_body["messages"][1] == {"role": "user", "content": "hard turn"}


def test_dispatch_routed_pins_baseline_for_efficient(monkeypatch):
    monkeypatch.setattr(base_effort, "_load_base_route", lambda: _route())
    seen = []

    async def handler(request):
        seen.append((request.url.path, json.loads(request.content.decode())))
        if request.url.path.endswith("/decision"):
            return httpx.Response(200, json=_efficient_decision())
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = base_proxy.dispatch_routed(
        {"model": "mantis/base", "messages": [{"role": "user", "content": "hi"}]},
        "sess-async",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        timeout_s=5.0,
    )

    assert result.body["choices"][0]["message"]["content"] == "ok"
    assert seen[0][0].endswith("/decision")
    assert seen[1][0].endswith("/chat/completions")
    assert seen[1][1]["reasoning_effort"] == "medium"
    assert seen[1][1]["messages"] == [{"role": "user", "content": "hi"}]


def test_dispatch_routed_fails_open_when_decision_is_unavailable(monkeypatch):
    monkeypatch.setattr(base_effort, "_load_base_route", lambda: _route())
    seen = []

    async def handler(request):
        seen.append((request.url.path, json.loads(request.content.decode())))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    result = base_proxy.dispatch_routed(
        {"model": "mantis/base", "messages": [{"role": "user", "content": "hi"}]},
        None,
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    assert result.body["choices"][0]["message"]["content"] == "ok"
    assert len(seen) == 2
    assert "reasoning_effort" not in seen[1][1]
    assert seen[1][1]["messages"] == [{"role": "user", "content": "hi"}]


def test_env_gate_leaves_forward_on_the_legacy_shape(monkeypatch):
    monkeypatch.setattr(base_effort, "_load_base_route", lambda: _route())
    monkeypatch.setenv("MANTIS_TIER_EFFORT", "0")
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"id": "c", "choices": [{"message": {"content": "ok"}}]})

    messages = [{"role": "user", "content": "hi"}]
    response, _ = base_proxy.forward(
        _forward_request(messages, "sess-off"),
        {},
        "req-off",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert response.status_code == 200
    assert seen == ["/v1/chat/completions"]
