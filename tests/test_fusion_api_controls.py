"""Request control storage across Fusion API starts and continuations."""

from __future__ import annotations

from types import SimpleNamespace

import api
import fusion
import fusion_api_context
import providers
import pytest
import serve_config

from tests.test_fusion_session import _api_client
from tests.test_fusion_tool_mode import BASH_TOOL, call, reply

CHOICE = {"type": "function", "function": {"name": "bash"}}


@pytest.fixture
def captured_api(monkeypatch, tmp_path):
    client = _api_client(monkeypatch, tmp_path)
    captured = []

    def advance(run_id, **kwargs):
        run = fusion.get_run(run_id)
        captured.append((run_id, run.tool_choice, dict(run.controls)))
        return {"status": "completed", "report": "done"}

    monkeypatch.setattr(fusion, "advance_fusion_run", advance)
    return client, captured


def post(client, messages=None, headers=None, **controls):
    return client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key", **(headers or {})},
        json={
            "model": "mantis/fusion",
            "messages": messages or [{"role": "user", "content": "work"}],
            "tools": [BASH_TOOL],
            "tool_options": {},
            **controls,
        },
    )


@pytest.mark.parametrize("choice", ["auto", "none", "required", CHOICE])
def test_new_api_run_stores_wire_choice_and_parallel_false(captured_api, choice):
    client, captured = captured_api
    response = post(client, tool_choice=choice, parallel_tool_calls=False)

    assert response.status_code == 200
    run_id, stored_choice, controls = captured[-1]
    assert stored_choice == choice
    assert controls == {"parallel_tool_calls": False}
    assert fusion.get_run(run_id).tool_choice == choice


@pytest.mark.parametrize("continuation", ["header", "tool"])
def test_resumed_api_turn_replaces_request_controls(captured_api, continuation):
    client, captured = captured_api
    assert post(client, tool_choice=CHOICE, parallel_tool_calls=False).status_code == 200
    run_id = captured[-1][0]
    messages = [{"role": "user", "content": "continue"}]
    headers = {"X-Mantis-Run-Id": run_id} if continuation == "header" else {}
    if continuation == "tool":
        messages.append({"role": "tool", "tool_call_id": f"f{run_id}~call", "content": "ok"})

    response = post(client, messages, headers, tool_choice="none", parallel_tool_calls=True)

    assert response.status_code == 200
    assert captured[-1] == (run_id, "none", {"parallel_tool_calls": True})
    response = post(client, headers={"X-Mantis-Run-Id": run_id})
    assert response.status_code == 200
    assert captured[-1] == (run_id, None, {})


@pytest.fixture
def provider_api(monkeypatch, tmp_path):
    client = _api_client(monkeypatch, tmp_path)
    resolved = providers.ResolvedModelSpec(
        "mock", "wire-model", None, "https://local.invalid", "TEST_API_KEY", None, None, "mock"
    )
    monkeypatch.setenv("TEST_API_KEY", "test-key")
    monkeypatch.setattr(providers, "_runtime_bindings", lambda: None)
    monkeypatch.setattr(providers, "_failover_attempts", lambda *_args: ["mock"])
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda *_args: resolved)
    monkeypatch.setattr(fusion.FusionCoordinator, "_context_window_for", lambda *_args: 16384)
    monkeypatch.setattr(fusion.FusionCoordinator, "_output_tokens_for", lambda *_args: 1024)
    captured = []
    responses = []

    def completion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=responses.pop(0), finish_reason="stop")], usage={}
        )

    monkeypatch.setattr(providers, "_litellm_completion", completion)
    return client, captured, responses


@pytest.mark.parametrize("choice", ["auto", "none", "required", CHOICE])
def test_api_controls_reach_litellm_provider_kwargs(provider_api, choice):
    client, captured, responses = provider_api
    responses.append(reply("done") if choice == "none" else reply("", call("c1", "bash")))

    response = post(client, tool_choice=choice, parallel_tool_calls=False)

    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0]["tool_choice"] == choice
    assert captured[0]["parallel_tool_calls"] is False
    names = [tool["function"]["name"] for tool in captured[0].get("tools", [])]
    if choice == "none":
        assert names == []
        assert response.json()["choices"][0]["finish_reason"] == "stop"
    else:
        assert "bash" in names
        message = response.json()["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "bash"


@pytest.mark.parametrize("choice", ["required", CHOICE])
def test_sidekick_does_not_inherit_forced_client_tool_choice(provider_api, choice):
    client, captured, responses = provider_api
    responses.append(reply("report"))
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "work"}],
        tools=[BASH_TOOL],
        tool_options={},
        tool_choice=choice,
        parallel_tool_calls=False,
    )
    run = fusion_api_context.create_chat_run(request, "work", None)
    coordinator = fusion.FusionCoordinator()
    run.sidekick_messages.append({"role": "user", "content": "subtask"})
    try:
        serve_config._history_context.active_run = run
        run._call_lane(coordinator, "sidekick", tools=run.tools)
    finally:
        serve_config._history_context.active_run = None

    assert len(captured) == 1
    assert "tool_choice" not in captured[0]
    assert captured[0]["parallel_tool_calls"] is False
