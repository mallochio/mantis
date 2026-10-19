"""Focused contracts for routed calls through the internal provider seam."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import base_proxy
import direct_completion
import httpx
import providers
import pytest
import serve_config


@pytest.fixture
def direct_provider_guard(monkeypatch):
    guard = Mock(side_effect=AssertionError("virtual routes must not enter provider resolution"))
    for name in ("_runtime_bindings", "_failover_attempts", "_litellm_completion"):
        monkeypatch.setattr(providers, name, guard)
    return guard


@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "read"}}]])
@pytest.mark.parametrize("timeout", [None, 1.25])
def test_provider_seam_reuses_existing_routed_helper(
    monkeypatch, direct_provider_guard, tools, timeout
):
    payload = {"choices": [{"message": {"content": "answer"}}]}
    routed = Mock(return_value=payload)
    monkeypatch.setattr(direct_completion, "_routed_response", routed)
    messages = [{"role": "user", "content": "read"}]

    assert providers._provider_response("mantis/base", messages, 4, 0.2, tools, timeout) is payload

    routed.assert_called_once_with(messages, 0.2, timeout, tools)
    direct_provider_guard.assert_not_called()


@pytest.fixture
def routed_transport(monkeypatch):
    seen = []
    tools = [{"type": "function", "function": {"name": "read"}}]
    messages = [{"role": "user", "content": "read", "_mantis_model": "old-provider"}]
    payload = {
        "choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1"}]}}],
        "usage": {"prompt_tokens": 20},
    }
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())

    async def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=payload,
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/chosen"},
        )

    _bind_routed_transport(monkeypatch, run, handler)
    return seen, run, payload, messages, tools


def _bind_routed_transport(monkeypatch, run, handler):
    monkeypatch.setenv("MANTIS_ROUTER_URL", "http://switchyard.test/v1")
    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "run-1:worker", raising=False
    )


def test_provider_seam_uses_shared_transport_and_usage(routed_transport, direct_provider_guard):
    seen, run, payload, messages, tools = routed_transport

    assert providers._provider_response("mantis/base", messages, 4, 0.2, tools, 1.25) == payload

    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "http://switchyard.test/v1/chat/completions"
    assert request.headers[base_proxy.SWITCHYARD_SESSION_HEADER] == "run-1:worker"
    assert request.headers["authorization"] == "Bearer router-key"
    assert json.loads(request.content) == {
        "model": "mantis/base",
        "messages": [{"role": "user", "content": "read"}],
        "temperature": 0.2,
        "tools": tools,
        "session_id": "run-1:worker",
    }
    assert request.extensions["timeout"] == dict.fromkeys(
        ["connect", "read", "write", "pool"], 1.25
    )
    assert messages[0]["_mantis_model"] == "old-provider"
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="provider/chosen")
    direct_provider_guard.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError("deadline"), ValueError("non-chat payload")])
def test_provider_seam_propagates_route_errors_without_failover(
    monkeypatch, direct_provider_guard, error
):
    dispatch = Mock(side_effect=error)
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", None, raising=False
    )

    with pytest.raises(type(error), match=str(error)):
        providers._provider_response("mantis/base", [], 4, 0.2)

    dispatch.assert_called_once_with(
        {"model": "mantis/base", "messages": [], "temperature": 0.2},
        "run-1",
        timeout_s=None,
    )
    run.add_usage.assert_not_called()
    direct_provider_guard.assert_not_called()


@pytest.mark.parametrize("spec", ["provider/model", "mantis/other", "mantis/base-extra"])
def test_concrete_provider_does_not_use_routed_helper(monkeypatch, spec):
    routed = Mock(side_effect=AssertionError("concrete models must not be routed"))
    bindings = Mock(side_effect=LookupError("direct provider resolution"))
    monkeypatch.setattr(direct_completion, "_routed_response", routed)
    monkeypatch.setattr(providers, "_runtime_bindings", bindings)

    with pytest.raises(LookupError, match="direct provider resolution"):
        providers._provider_response(spec, [], 4, 0.2)

    bindings.assert_called_once()
    routed.assert_not_called()
