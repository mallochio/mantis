"""Focused contracts for text-only internal routed worker calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import base_proxy
import httpx
import providers
import pytest
import serve_config
import trinity


def _response(content="answer", selected="provider/chosen"):
    return base_proxy.RoutedResponse(
        {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 20}},
        selected,
        "run-1",
    )


@pytest.mark.parametrize(
    "worker_type", [trinity.DirectTrinityWorker, trinity.DirectConductorWorker]
)
def test_internal_workers_reuse_routed_dispatch(monkeypatch, worker_type):
    dispatch = Mock(return_value=_response())
    provider = Mock(side_effect=AssertionError("virtual routes must not use direct providers"))
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", None, raising=False
    )
    worker = worker_type(["mantis/base"], max_tokens=4, temperature=0.2, timeout=7.5)
    messages = [{"role": "user", "content": "hi"}]
    text = (
        worker.conduct("mantis/base", messages)
        if hasattr(worker, "conduct")
        else worker("Worker", messages, 0)
    )
    assert text == "answer"
    dispatch.assert_called_once_with(
        {"model": "mantis/base", "messages": messages, "temperature": 0.2},
        "run-1",
        timeout_s=7.5,
    )
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="provider/chosen")
    provider.assert_not_called()


@pytest.mark.parametrize("session", [None, "run-1:planner"])
def test_routed_completion_without_active_run(monkeypatch, session):
    dispatch = Mock(return_value=_response(content=None, selected=None))
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(serve_config._history_context, "active_run", None, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", session, raising=False
    )
    assert providers._direct_completion("mantis/base", [], 4, 0.7, 2.0) == ""
    assert dispatch.call_args.args[1] == session


def test_routed_completion_falls_back_to_virtual_usage_model(monkeypatch):
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", Mock(return_value=_response(selected=None)))
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "lane", raising=False
    )
    assert providers._direct_completion("mantis/base", [], 4, 0.7, 2.0) == "answer"
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="mantis/base")


@pytest.mark.parametrize(
    "status,payload", [(503, {"error": "unavailable"}), (200, {"not": "chat"})]
)
def test_routed_errors_propagate_without_direct_failover(monkeypatch, status, payload):
    async def handler(request):
        return httpx.Response(status, json=payload)

    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(serve_config._history_context, "active_run", None, raising=False)
    provider = Mock(side_effect=AssertionError("no direct fallback for a virtual route"))
    monkeypatch.setattr(providers, "_provider_response", provider)
    error = httpx.HTTPStatusError if status == 503 else ValueError
    with pytest.raises(error):
        providers._direct_completion("mantis/base", [], 4, 0.7, 2.0)
    provider.assert_not_called()


def test_routed_completion_uses_shared_transport_hygiene(monkeypatch):
    seen = []
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    messages = [{"role": "user", "content": "hi", "_mantis_model": "old-provider"}]

    async def handler(request):
        seen.append((json.loads(request.content), dict(request.headers)))
        return httpx.Response(
            200,
            json=_response().body,
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/chosen"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "run-1:planner", raising=False
    )
    assert providers._direct_completion("mantis/base", messages, 4, 0.7, 2.0) == "answer"
    body, headers = seen[0]
    assert body == {
        "model": "mantis/base",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
    }
    assert headers[base_proxy.SWITCHYARD_SESSION_HEADER] == "run-1:planner"
    assert headers["authorization"] == "Bearer router-key"
    assert messages[0]["_mantis_model"] == "old-provider"
    run.add_usage.assert_called_once_with({"prompt_tokens": 20}, model="provider/chosen")


def test_non_routed_completion_keeps_existing_provider_call(monkeypatch):
    provider = Mock(return_value=_response().body)
    dispatch = Mock(side_effect=AssertionError("direct providers must not enter Switchyard"))
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    messages = [{"role": "user", "content": "hi"}]
    assert providers._direct_completion("gpt-5_6-luna", messages, 64, 0.5, 2.0) == "answer"
    provider.assert_called_once_with("gpt-5_6-luna", messages, 64, 0.5)
    dispatch.assert_not_called()


def test_routed_deadline_error_propagates_without_usage_or_failover(monkeypatch):
    dispatch = Mock(side_effect=TimeoutError("router deadline expired"))
    provider = Mock(side_effect=AssertionError("no direct fallback for a virtual route"))
    run = SimpleNamespace(run_id="run-1", add_usage=Mock())
    monkeypatch.setattr(base_proxy, "dispatch_routed", dispatch)
    monkeypatch.setattr(providers, "_provider_response", provider)
    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_route_session_id", "run-1:planner", raising=False
    )

    with pytest.raises(TimeoutError, match="router deadline expired"):
        providers._direct_completion("mantis/base", [], 4, 0.7, 0.05)

    dispatch.assert_called_once_with(
        {"model": "mantis/base", "messages": [], "temperature": 0.7},
        "run-1:planner",
        timeout_s=0.05,
    )
    run.add_usage.assert_not_called()
    provider.assert_not_called()
