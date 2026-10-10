"""Contracts for tool-capable internal calls through the shared route."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import base_proxy
import httpx
import providers
import pytest
import serve_config
import utils


def _tool_response(content, arguments, **metadata):
    return {
        "choices": [
            {
                "message": {
                    "content": content,
                    "tool_calls": [
                        {"id": "call-1", "function": {"name": "read", "arguments": arguments}}
                    ],
                    **metadata,
                }
            }
        ],
        "usage": {"prompt_tokens": 20},
    }


@pytest.mark.parametrize("tools", [None, [], [{"type": "function", "function": {"name": "read"}}]])
def test_model_completion_reuses_route_with_tools(monkeypatch, tools):
    payload = _tool_response(None, '{"path":"README.md"}')
    dispatch = Mock(return_value=base_proxy.RoutedResponse(payload, "provider/chosen", "lane"))
    provider = Mock(side_effect=AssertionError("virtual routes must not use direct providers"))
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "lane", raising=False
    )
    messages = [{"role": "user", "content": "read the README"}]
    assert utils._model_completion("mantis/base", messages, tools) == (
        "",
        [{"name": "read", "arguments": {"path": "README.md"}}],
    )
    body = {"model": "mantis/base", "messages": messages, "temperature": 0.7}
    if tools:
        body["tools"] = tools
    dispatch.assert_called_once_with(body, "lane", timeout_s=None)
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="provider/chosen")
    provider.assert_not_called()


@pytest.mark.parametrize("run_id", [None, "run-1"])
def test_model_completion_without_lane_uses_shared_session_fallback(monkeypatch, run_id):
    payload = {"choices": [{"message": {"content": "answer"}}]}
    dispatch = Mock(return_value=base_proxy.RoutedResponse(payload, None, run_id))
    run = SimpleNamespace(run_id=run_id, add_usage=Mock()) if run_id else None
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", None, raising=False
    )
    assert utils._model_completion("mantis/base", [], None) == ("answer", [])
    dispatch.assert_called_once_with(
        {"model": "mantis/base", "messages": [], "temperature": 0.7}, run_id, timeout_s=None
    )
    if run:
        run.add_usage.assert_called_once_with({}, model="mantis/base")


def test_model_completion_preserves_direct_provider_and_tool_metadata(monkeypatch):
    metadata = {"reasoning_details": [{"text": "checked"}], "_anthropic_content": []}
    payload = _tool_response(
        "answer", "{}", _anthropic_tool_ids={"call-1": "native-id"}, **metadata
    )
    provider = Mock(return_value=payload)
    dispatch = Mock(side_effect=AssertionError("direct providers must not enter Switchyard"))
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    messages = [{"role": "user", "content": "read"}]
    tools = [{"type": "function", "function": {"name": "read"}}]
    assert utils._model_completion("gpt-5_6-luna", messages, tools) == (
        "answer",
        [
            {
                "name": "read",
                "arguments": {},
                "_anthropic_tool_id": "native-id",
                "_assistant_metadata": metadata,
            }
        ],
    )
    provider.assert_called_once_with("gpt-5_6-luna", messages, 4096, 0.7, tools)
    dispatch.assert_not_called()


@pytest.mark.parametrize("error", [TimeoutError("deadline"), ValueError("non-chat payload")])
def test_model_completion_propagates_route_errors_without_usage_or_failover(monkeypatch, error):
    provider = Mock(side_effect=AssertionError("no direct fallback for a virtual route"))
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", Mock(side_effect=error))
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    with pytest.raises(type(error), match=str(error)):
        utils._model_completion("mantis/base", [], None)
    run.add_usage.assert_not_called()
    provider.assert_not_called()


def _tool_trajectory():
    assistant = _tool_response("checking", '{"path":"README.md"}')["choices"][0]["message"]
    return [
        {"role": "user", "content": "read the README"},
        {
            "role": "assistant",
            **assistant,
            "_mantis_model": "old-provider",
            "reasoning_details": [
                {"type": "reasoning", "text": "portable"},
                {"type": "encrypted", "data": "provider-bound"},
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "README contents"},
    ]


@pytest.fixture
def routed_model_transport(monkeypatch):
    seen = []
    payload = _tool_response("checked", '{"path":"next.md"}')
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    provider = Mock(side_effect=AssertionError("virtual routes must not use direct providers"))

    async def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=payload,
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/chosen"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_URL", "http://switchyard.test/v1")
    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "run-1:worker", raising=False
    )
    return seen, run, provider


def test_model_completion_replays_tool_history_through_shared_transport(routed_model_transport):
    seen, run, provider = routed_model_transport
    messages = _tool_trajectory()
    original = json.loads(json.dumps(messages))
    tools = [{"type": "function", "function": {"name": "read"}}]
    assert utils._model_completion("mantis/base", messages, tools) == (
        "checked",
        [{"name": "read", "arguments": {"path": "next.md"}}],
    )
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "http://switchyard.test/v1/chat/completions"
    assert request.headers[base_proxy.SWITCHYARD_SESSION_HEADER] == "run-1:worker"
    assert request.headers["authorization"] == "Bearer router-key"
    expected_messages = json.loads(json.dumps(messages))
    expected_messages[1].pop("_mantis_model")
    expected_messages[1]["reasoning_details"] = [{"type": "reasoning", "text": "portable"}]
    assert json.loads(request.content) == {
        "model": "mantis/base",
        "messages": expected_messages,
        "temperature": 0.7,
        "tools": tools,
    }
    assert messages == original
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="provider/chosen")
    provider.assert_not_called()
