"""Harness-neutral Fusion request and workspace compatibility tests."""

from __future__ import annotations

import logging
import re

import api
import fusion
import fusion_harness
import fusion_types
import pytest
import runs
import serve_config

from tests.test_fusion_tool_mode import call, install, reply, tool_run


def test_workspace_parser_accepts_generic_harness_prompt_shapes(tmp_path):
    expected = str(tmp_path.resolve())
    prompts = (
        f"Working directory: {expected}",
        f"cwd: {expected}",
        f"Current working directory: {expected}",
        f"<cwd>{expected}</cwd>",
        f"<env>\nPWD={expected}\n</env>",
    )

    assert [fusion_harness.workspace_from_prompt(prompt) for prompt in prompts] == [
        expected,
        expected,
        expected,
        expected,
        expected,
    ]


def test_workspace_candidate_priority_and_default_precedence(monkeypatch, tmp_path):
    prompt_path = str(tmp_path / "from-prompt")
    monkeypatch.setenv("MANTIS_FUSION_DEFAULT_WORKSPACE", "/environment/default")

    assert fusion_harness.workspace_candidate(
        "/explicit", "/header", f"cwd: {prompt_path}", "/catalog/default"
    ) == ("/explicit", "explicit")
    assert fusion_harness.workspace_candidate(
        None, "/header", f"cwd: {prompt_path}", "/catalog/default"
    ) == ("/header", "header")
    assert fusion_harness.workspace_candidate(
        None, None, f"cwd: {prompt_path}", "/catalog/default"
    ) == (prompt_path, "system prompt")
    assert fusion_harness.workspace_candidate(None, None, "", "/catalog/default", {}) == (
        "/catalog/default",
        "default",
    )
    assert fusion_harness.workspace_candidate(
        None,
        None,
        "",
        "/catalog/default",
        {"MANTIS_FUSION_DEFAULT_WORKSPACE": "/environment/default"},
    ) == ("/environment/default", "default")


def test_workspace_resolution_validates_explicit_and_ignores_discovered_paths(
    tmp_path, monkeypatch, caplog
):
    allowed = tmp_path / "allowed"
    workspace = allowed / "project"
    workspace.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setenv(fusion_types.WORKSPACE_ROOTS_ENV, str(allowed))
    caplog.set_level(logging.INFO, logger="fusion_harness")

    assert fusion_harness.resolve_workspace(None, str(workspace), "", None) == str(
        workspace.resolve()
    )
    assert fusion_harness.resolve_workspace(None, str(outside), "", None) is None
    assert "Ignoring discovered Fusion workspace" in caplog.text
    with pytest.raises(ValueError, match=fusion_types.WORKSPACE_ROOTS_ENV):
        fusion_harness.resolve_workspace(str(outside), None, "", None)


def test_api_discovers_workspace_from_header_then_system_prompt(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    header_path = allowed / "header"
    prompt_path = allowed / "prompt"
    header_path.mkdir(parents=True)
    prompt_path.mkdir()
    monkeypatch.setenv(fusion_types.WORKSPACE_ROOTS_ENV, str(allowed))
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[
            {"role": "system", "content": f"Working directory: {prompt_path}"},
            {"role": "developer", "content": "Additional rules."},
            {"role": "user", "content": "inspect"},
        ],
        tool_options={},
    )

    selected = api._resolve_fusion_workspace(request, {"X-Mantis-Workspace": str(header_path)})
    assert selected.tool_options is not None
    assert selected.tool_options.workspace == str(header_path.resolve())
    prompt_selected = api._resolve_fusion_workspace(request, None)
    assert prompt_selected.tool_options is not None
    assert prompt_selected.tool_options.workspace == str(prompt_path.resolve())
    outside = tmp_path / "outside"
    outside.mkdir()
    sandboxed = api._resolve_fusion_workspace(request, {"x-mantis-workspace": str(outside)})
    assert sandboxed.tool_options is not None
    assert sandboxed.tool_options.workspace is None


def test_session_identity_uses_only_explicit_ids():
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "hello"}],
        tool_options={},
    )
    assert api._fusion_session_id({"X-Mantis-Session-Id": "session-1"}, request) == "session-1"
    assert api._fusion_session_id(None, request) is None
    identified = request.model_copy(update={"metadata": {"session_id": "body-session"}})
    assert api._fusion_session_id(None, identified) == "body-session"
    user_only = request.model_copy(update={"user": "account-id"})
    assert api._fusion_session_id(None, user_only) is None


def test_empty_client_tools_still_give_the_lead_delegation_tools(monkeypatch):
    worker = install(monkeypatch, [reply("No delegation was needed.")], [])
    run = tool_run(tools=[])

    assert run.advance()["status"] == "completed"
    assert [tool["function"]["name"] for tool in worker.seen["lead"][0][1]] == [
        "sidekick",
        "read_subagent",
    ]


def test_tool_names_are_renamed_deterministically_without_mutating_client_tools():
    client_tools = [
        {"type": "function", "function": {"name": "sidekick"}},
        {"type": "function", "function": {"name": "read_subagent"}},
        {"type": "function", "function": {"name": "fusion_sidekick"}},
    ]
    names = fusion_harness.effective_delegation_names(client_tools)
    schemas = [
        {"type": "function", "function": {"name": "sidekick"}},
        {"type": "function", "function": {"name": "read_subagent"}},
    ]
    renamed = fusion_harness.rename_schemas(schemas, names)

    assert names == {
        "sidekick": "fusion_sidekick_2",
        "read_subagent": "fusion_read_subagent",
    }
    assert [schema["function"]["name"] for schema in renamed] == list(names.values())
    assert [schema["function"]["name"] for schema in schemas] == ["sidekick", "read_subagent"]
    assert fusion_harness.canonicalize("fusion_sidekick_2", names) == "sidekick"
    assert "fusion_sidekick_2" in fusion_harness.delegation_name_policy(names)


def test_effective_delegation_aliases_dispatch_and_update_the_lead_prompt(monkeypatch):
    tools = [
        {"type": "function", "function": {"name": "sidekick"}},
        {"type": "function", "function": {"name": "read_subagent"}},
        {"type": "function", "function": {"name": "fusion_sidekick"}},
    ]
    worker = install(
        monkeypatch,
        [
            reply("", call("c1", "fusion_sidekick_2", message="do the work")),
            reply("", call("c2", "fusion_read_subagent", block=True)),
            reply("done"),
        ],
        [reply("report")],
    )
    run = tool_run(tools=tools)

    assert run.advance()["status"] == "completed"
    lead_messages, lead_tools = worker.seen["lead"][0]
    assert "fusion_sidekick_2" in lead_messages[0]["content"]
    assert "fusion_read_subagent" in lead_messages[0]["content"]
    assert [item["function"]["name"] for item in lead_tools] == [
        "fusion_sidekick",
        "read_subagent",
        "sidekick",
        "fusion_sidekick_2",
        "fusion_read_subagent",
    ]
    assert len(worker.seen["sidekick"]) == 1


def test_client_tool_named_sidekick_is_not_shadowed(monkeypatch):
    client_tool = {"type": "function", "function": {"name": "sidekick"}}
    worker = install(
        monkeypatch,
        [reply("", call("client-call", "sidekick", value="client tool"))],
        [],
    )
    run = tool_run(tools=[client_tool])

    event = run.advance()

    assert event["status"] == "awaiting_tools"
    assert event["pending_tool_calls"][0]["function"]["name"] == "sidekick"
    assert event["pending_tool_calls"][0]["id"] == "lead:client-call"
    assert not worker.seen["sidekick"]


def test_required_and_named_tool_choice_accepts_fusion_delegation_tools():
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "delegate"}],
        tool_choice="required",
        tool_options={},
    )
    named = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "look up"}],
        tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
        tool_choice={"type": "function", "function": {"name": "lookup"}},
        tool_options={},
    )

    assert request.tools is None
    assert named.tool_choice.function.name == "lookup"


def test_large_tool_schemas_and_empty_client_tool_lists_are_accepted():
    properties = {f"field_{index}": {"type": "string"} for index in range(1500)}
    large_tool = {
        "type": "function",
        "function": {"name": "large", "parameters": {"type": "object", "properties": properties}},
    }
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "use available tools"}],
        tools=[large_tool],
        tool_options={},
    )
    assert len(request.tools[0].function.parameters["properties"]) == 1500
    run = tool_run(tools=[request.tools[0].model_dump()])
    assert len(run.tools[0]["function"]["parameters"]["properties"]) == 1500


def test_public_tool_ids_survive_common_sanitizing_and_decode_legacy_ids(monkeypatch):
    monkeypatch.setattr(fusion, "RUN_STORE", "memory")
    monkeypatch.setattr(runs, "RUN_STORE", "memory")
    serve_config._runs.clear()
    run_id = "0123456789abcdef0123456789abcdef"
    run = fusion.FusionRun(run_id, "test", tool_options=fusion.FusionToolOptions())
    run.pending_tool_calls = [{"id": "sidekick:provider-call:!@#"}]
    fusion._put_run(run)

    encoded = api._encode_fusion_tool_call_id(run_id, "sidekick:provider-call:!@#")
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "", encoded)[:40]

    assert len(encoded) <= 40
    assert encoded == sanitized
    assert api._decode_fusion_tool_call_id(sanitized) == (run_id, "sidekick:provider-call:!@#")
    assert api._decode_fusion_tool_call_id("frun~lead:old-call") == ("run", "lead:old-call")
    serve_config._runs.clear()
