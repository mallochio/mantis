"""Refresh live Fusion delegation names when a client changes tool schemas."""

from __future__ import annotations

import json
from types import SimpleNamespace

import fusion
import fusion_api_context
import fusion_background
import fusion_lead
import fusion_stream


def _prepared_run():
    first = {"type": "function", "function": {"name": "sidekick", "parameters": {}}}
    updated = {"type": "function", "function": {"name": "fusion_sidekick", "parameters": {}}}
    run = fusion.FusionRun(
        "abcdef0123456789abcdef0123456789",
        "task",
        tools=[first],
        tool_options=fusion.FusionToolOptions(delegation="tool"),
    )
    fusion_lead.prepare_run(run)
    return run, run.handoffs, updated, run.main_messages[0]["content"]


def test_refresh_keeps_new_client_tool_and_updates_the_prompt():
    run, manager, updated, initial_prompt = _prepared_run()
    fusion.refresh_fusion_run_tools(run, [updated])
    tools = fusion_background.lead_tools(manager, run)

    assert manager.delegation_names == {"sidekick": "sidekick", "read_subagent": "read_subagent"}
    assert "Use fusion_sidekick to delegate work" in initial_prompt
    assert "fusion_sidekick" not in run.main_messages[0]["content"]
    assert [tool["function"]["name"] for tool in tools] == [
        updated["function"]["name"],
        "sidekick",
        "read_subagent",
    ]


def test_stream_uses_refreshed_names_and_keeps_client_call_out_of_handoff(monkeypatch):
    run, manager, updated, initial_prompt = _prepared_run()
    monkeypatch.setattr(fusion, "try_get_fusion_run", lambda _run_id: run)
    request = SimpleNamespace(tools=[SimpleNamespace(model_dump=lambda: updated)])
    hidden = fusion_api_context.hidden_tool_names(request, run.run_id)
    client_call = {
        "id": "client-call",
        "function": {"name": "fusion_sidekick", "arguments": json.dumps({"message": "private"})},
    }
    encoder = fusion_stream.DeltaEncoder(hidden)
    public = encoder.encode(run.run_id, {"tool_calls": [client_call]}, lambda *_: "public")
    fusion_background.open_lead_batch(manager, [client_call])

    assert hidden == {"sidekick", "read_subagent"}
    assert manager.delegation_names == {"sidekick": "sidekick", "read_subagent": "read_subagent"}
    assert "Use fusion_sidekick to delegate work" in initial_prompt
    assert "fusion_sidekick" not in run.main_messages[0]["content"]
    assert client_call["id"] == "lead:client-call"
    assert public[0]["tool_calls"][0]["function"]["name"] == "fusion_sidekick"
    assert run.sidekick_messages[1:] == []
