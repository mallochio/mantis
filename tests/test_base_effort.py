import asyncio
import json
from types import SimpleNamespace

import base_effort
import httpx
import pytest


@pytest.fixture(autouse=True)
def _tier_effort_on(monkeypatch):
    monkeypatch.setenv("MANTIS_TIER_EFFORT", "1")


def _route(effort="medium", capable="high", wire="openai_chat"):
    return SimpleNamespace(
        efficient=SimpleNamespace(reasoning_effort=effort, wire_format=wire),
        capable=SimpleNamespace(reasoning_effort=capable, wire_format=wire),
    )


def _decision(target="capable", effort="high"):
    extra = {} if effort is None else {"reasoning_effort": effort}
    return {
        "selected": {"target": target, "model": "~openai/gpt-sol-latest", "extra_body": extra},
        "fallbacks": [],
    }


def _body(role="user", content="hi"):
    return {"model": "mantis/base", "messages": [{"role": role, "content": content}]}


def test_pins_baseline_without_update_for_efficient():
    out = base_effort.apply_router_effort(_body(), _route(), _decision("efficient", "medium"))
    assert out["reasoning_effort"] == "medium"
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_injects_capable_update_before_last_user_message():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]
    original = json.loads(json.dumps(messages))
    out = base_effort.apply_router_effort(
        {"model": "mantis/base", "messages": messages}, _route(), _decision()
    )
    assert out["reasoning_effort"] == "medium"
    assert out["messages"][2] == {
        "role": "system",
        "content": "",
        "configuration_update": {"reasoning": {"effort": "high"}},
    }
    assert out["messages"][3] == {"role": "user", "content": "three"}
    assert out["messages"][:2] == original[:2]
    assert messages == original


def test_appends_update_after_tool_results():
    messages = [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "out"},
    ]
    out = base_effort.apply_router_effort(
        {"model": "mantis/base", "messages": messages}, _route(), _decision()
    )
    assert out["messages"][-1]["configuration_update"] == {"reasoning": {"effort": "high"}}
    assert out["messages"][-2] == messages[-1]


def test_skips_update_when_tier_effort_matches_baseline():
    messages = [{"role": "user", "content": "hi"}]
    out = base_effort.apply_router_effort(
        {"model": "mantis/base", "messages": messages},
        _route(capable="medium"),
        _decision("capable", "medium"),
    )
    assert out["reasoning_effort"] == "medium"
    assert out["messages"] == messages


def test_falls_back_to_catalog_effort_when_decision_lacks_it():
    out = base_effort.apply_router_effort(_body(), _route(), _decision("capable", None))
    assert out["messages"][0]["configuration_update"] == {"reasoning": {"effort": "high"}}


def test_ignores_unusable_decision_and_leaves_body_untouched():
    for decision in (None, {}, {"choices": []}, {"selected": {"target": "other"}}):
        body = _body()
        assert base_effort.apply_router_effort(body, _route(), decision) == body


def test_skips_entirely_for_non_chat_wire():
    body = _body()
    assert (
        base_effort.apply_router_effort(body, _route(wire="openai_responses"), _decision()) == body
    )


def test_skips_injection_when_history_already_carries_update():
    messages = [
        {"role": "user", "content": "one"},
        {"role": "system", "content": "", "configuration_update": {"reasoning": {"effort": "low"}}},
    ]
    out = base_effort.apply_router_effort(
        {"model": "mantis/base", "messages": messages}, _route(), _decision()
    )
    assert out["reasoning_effort"] == "medium"
    assert out["messages"] == messages


def test_skips_when_efficient_tier_has_no_effort():
    body = _body()
    assert base_effort.apply_router_effort(body, _route(effort=None), _decision()) == body


def test_env_gate_disables_the_mechanism(monkeypatch):
    monkeypatch.setenv("MANTIS_TIER_EFFORT", "0")
    assert base_effort.effort_baseline(_route()) is None
    assert base_effort.apply_router_effort(_body(), _route(), _decision()) == _body()


def test_attach_sticky_session_pins_and_caps_the_session():
    body = base_effort.attach_sticky_session({}, "s" * 300)
    assert body["session_id"] == "s" * 256
    assert base_effort.attach_sticky_session({}, None) == {}
    existing = {"session_id": "keep"}
    assert base_effort.attach_sticky_session(existing, "other")["session_id"] == "keep"


def test_decision_payload_wraps_body_with_wire_format():
    assert base_effort.decision_payload({"model": "mantis/base"}) == {
        "input_format": "openai_chat",
        "request": {"model": "mantis/base"},
    }


def test_decision_url_respects_router_env(monkeypatch):
    monkeypatch.setenv("MANTIS_ROUTER_URL", "http://router.test/v1")
    assert base_effort.decision_url() == "http://router.test/v1/decision"
    monkeypatch.delenv("MANTIS_ROUTER_URL")
    assert base_effort.decision_url() == "http://127.0.0.1:5500/v1/decision"


def _sync_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_fetch_decision_sync_parses_decision_json():
    client = _sync_client(lambda request: httpx.Response(200, json=_decision()))
    data = base_effort.fetch_decision_sync(
        client, "http://router.test/v1/decision", {"request": {}}, {}
    )
    assert data["selected"]["target"] == "capable"


def test_fetch_decision_sync_returns_none_on_http_error():
    client = _sync_client(lambda request: httpx.Response(503, json={"error": "down"}))
    assert base_effort.fetch_decision_sync(client, "http://router.test/v1/decision", {}, {}) is None


def test_fetch_decision_sync_returns_none_on_non_dict_json():
    client = _sync_client(lambda request: httpx.Response(200, json=[1, 2]))
    assert base_effort.fetch_decision_sync(client, "http://router.test/v1/decision", {}, {}) is None


def test_fetch_decision_async_propagates_deadline_cancellation():
    async def slow_handler(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, json=_decision())

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler))
        try:
            async with asyncio.timeout(0.02):
                return await base_effort.fetch_decision_async(
                    client, "http://router.test/v1/decision", {}, {}, 5.0
                )
        finally:
            await client.aclose()

    with pytest.raises(TimeoutError):
        asyncio.run(scenario())


def test_fetch_decision_async_returns_none_on_http_error():
    async def handler(request):
        return httpx.Response(503, json={"error": "down"})

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await base_effort.fetch_decision_async(
                client, "http://router.test/v1/decision", {}, {}, 5.0
            )
        finally:
            await client.aclose()

    assert asyncio.run(scenario()) is None


def test_fetch_decision_async_parses_decision_json():
    async def handler(request):
        return httpx.Response(200, json=_decision("efficient", "medium"))

    async def scenario():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            return await base_effort.fetch_decision_async(
                client, "http://router.test/v1/decision", {}, {}, 5.0
            )
        finally:
            await client.aclose()

    assert asyncio.run(scenario())["selected"]["target"] == "efficient"
