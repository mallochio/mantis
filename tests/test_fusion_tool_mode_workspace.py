"""Tool delegation: shared workspace, lead-only command denial, catalog and API plumbing."""

from __future__ import annotations

import json
from typing import Any

import api
import fusion
import fusion_prompts
import fusion_types
import pydantic
import pytest
import serve_config
import tool_exec
from fastapi.testclient import TestClient

from tests.test_fusion_tool_mode import (
    BASH_TOOL,
    call,
    install,
    reply,
    tool_messages,
    tool_run,
    use_tool_catalog,
)

HIDDEN_BRIEF = "brief text for the helper"
HIDDEN_REPORT = "report text from the helper"
SIDEKICK_THOUGHT = "hidden sidekick reasoning"
LEAD_SCRIPT = [
    reply(
        "",
        call("c1", "sidekick", message=HIDDEN_BRIEF),
        call("c2", "bash", command="ls"),
        reasoning="lead reasoning",
    ),
    reply("Final answer."),
]


@pytest.fixture(autouse=True)
def tool_catalog(monkeypatch, tmp_path):
    """Use the hermetic catalog of the core tool-mode tests and a clean run store."""
    serve_config._runs.clear()
    yield use_tool_catalog(monkeypatch, tmp_path)
    serve_config._runs.clear()


@pytest.fixture
def workspace(monkeypatch, tmp_path):
    """Return an allowlisted client workspace directory."""
    root = tmp_path / "allowed"
    repo = root / "repo"
    repo.mkdir(parents=True)
    monkeypatch.setenv(fusion_types.WORKSPACE_ROOTS_ENV, str(root))
    return repo


@pytest.fixture
def client(monkeypatch):
    """Return an authenticated API client backed by the memory run store."""
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_RUN_STORE", "memory")
    return TestClient(api.app)


@pytest.mark.parametrize(
    "command",
    [
        "git push origin main",
        "git commit -m wip",
        "cd repo && git -C . commit -am x",
        "git --no-pager -c user.name=x push",
        "gh pr create --fill",
        "gh release create v1",
    ],
)
def test_lead_only_commands_are_denied(command):
    error = tool_exec.lead_only_command_error({"command": command})
    assert error is not None
    assert "report back to the lead" in error


@pytest.mark.parametrize(
    "command", ["git status", "git log --grep push", "git stash push", "git diff", "ls -la"]
)
def test_ordinary_commands_are_allowed(command):
    assert tool_exec.lead_only_command_error({"command": command}) is None
    assert tool_exec.lead_only_command_error(json.dumps({"command": command})) is None


def test_workspace_allowlist(workspace, tmp_path, monkeypatch):
    options = fusion.FusionToolOptions(workspace=str(workspace))
    assert options.workspace == str(workspace.resolve())
    for bad in (str(tmp_path), "relative/path", str(workspace / "missing")):
        with pytest.raises(pydantic.ValidationError):
            fusion.FusionToolOptions(workspace=bad)
    monkeypatch.delenv(fusion_types.WORKSPACE_ROOTS_ENV)
    with pytest.raises(pydantic.ValidationError, match="MANTIS_FUSION_WORKSPACE_ROOTS"):
        fusion.FusionToolOptions(workspace=str(workspace))


def test_api_rejects_workspace_outside_allowlist(client, workspace, tmp_path):
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_options": {"delegation": "tool", "workspace": str(tmp_path)},
        },
    )
    assert response.status_code == 400
    assert "MANTIS_FUSION_WORKSPACE_ROOTS" in response.json()["error"]["message"]


def test_sidekick_works_in_shared_workspace_and_cannot_push(monkeypatch, workspace):
    worker = install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="write and ship")), reply("Shipped.")],
        [
            reply(
                "",
                call("s1", "write_file", path="notes.txt", content="hello"),
                call("s2", "bash", command="git push origin main"),
            ),
            reply("wrote notes.txt; push was denied"),
        ],
    )
    run = tool_run(tools=[BASH_TOOL], workspace=str(workspace))

    event = run.advance()

    assert event["status"] == "completed"
    assert (workspace / "notes.txt").read_text() == "hello"
    _, sidekick_tools = worker.seen["sidekick"][0]
    assert "write_file" in {tool["function"]["name"] for tool in sidekick_tools}
    results = {m["tool_call_id"]: m for m in run.sidekick_messages if m.get("role") == "tool"}
    assert results["sidekick:s1"]["content"] == "ok"
    assert results["sidekick:s2"]["is_error"] is True
    assert "reserved for the lead" in results["sidekick:s2"]["content"]


def _chat(client, messages: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test-key"},
        json={
            "model": "mantis/fusion",
            "messages": messages,
            "tools": [BASH_TOOL],
            "tool_options": {"delegation": "tool"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_chat_hides_delegation_from_client_and_default_trace(client, monkeypatch):
    install(monkeypatch, LEAD_SCRIPT, [reply(HIDDEN_REPORT, reasoning=SIDEKICK_THOUGHT)])
    user = {"role": "user", "content": "do the task"}

    first = _chat(client, [user])
    message = first["choices"][0]["message"]
    assert [c["function"]["name"] for c in message["tool_calls"]] == ["bash"]
    tool_call = message["tool_calls"][0]
    follow_up = [
        user,
        {"role": "assistant", "content": "", "tool_calls": [tool_call]},
        {"role": "tool", "tool_call_id": tool_call["id"], "content": "a.txt"},
    ]
    second = _chat(client, follow_up)

    assert second["choices"][0]["message"]["content"] == "Final answer."
    traces = message.get("reasoning", "") + second["choices"][0]["message"].get("reasoning", "")
    assert "lead reasoning" in traces
    assert "Tool call: bash" in traces
    for hidden in (HIDDEN_BRIEF, HIDDEN_REPORT, SIDEKICK_THOUGHT, "sidekick"):
        assert hidden not in traces
    event = second["mantis_event"]
    assert event["trace_scope"] == "main"
    assert any(entry["kind"] == "handoff" for entry in event["handoff_history"])


def test_catalog_delegation_selects_mode(monkeypatch, tool_catalog):
    tool_catalog.write_text(tool_catalog.read_text() + 'delegation = "tool"\n')
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", fusion.FusionConfig(tool_catalog))
    install(monkeypatch, [reply("tool answer")], [])

    run = fusion.FusionRun("catalog", "hello")
    assert run.advance()["report"] == "tool answer"
    assert run.handoffs is not None

    marker = fusion.FusionRun(
        "override", "hello", tool_options=fusion.FusionToolOptions(delegation="marker")
    )
    assert not fusion.fusion_lead.uses_tool_delegation(marker, fusion.FusionCoordinator())
    assert marker.handoffs is None


def test_invalid_delegation_values_are_rejected(monkeypatch, tool_catalog):
    with pytest.raises(pydantic.ValidationError):
        fusion.FusionToolOptions(delegation="markers")
    tool_catalog.write_text(tool_catalog.read_text() + 'delegation = "bogus"\n')
    with pytest.raises(ValueError, match="'marker' or 'tool'"):
        fusion.FusionConfig(tool_catalog).delegation()


def test_marker_mode_is_the_default(monkeypatch):
    def legacy(_coordinator, slot, messages, tools):
        assert messages[0]["content"].startswith(fusion.MAIN_PREAMBLE)
        assert not any(t["function"]["name"] == "sidekick" for t in tools or [])
        return reply("ANSWER: direct reply"), {}

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", legacy)
    run = fusion.FusionRun("marker", "hello", tools=[BASH_TOOL])

    assert fusion.FusionConfig(fusion._FUSION_CONFIG.path).delegation() == "marker"
    assert run.advance()["report"] == "direct reply"
    assert run.handoffs is None
    assert fusion_prompts.LEAD_DELEGATION_POLICY not in str(run.main_messages)


def test_visible_lead_messages_drop_only_delegation_calls():
    messages = [
        reply("", call("delegate:a", "sidekick", message="x"), call("lead:b", "bash")),
        {"role": "tool", "tool_call_id": "delegate:a", "content": "report"},
        {"role": "tool", "tool_call_id": "lead:b", "content": "out"},
    ]
    visible = fusion.fusion_lead.visible_lead_messages(messages)
    assert [c["id"] for c in visible[0]["tool_calls"]] == ["lead:b"]
    assert tool_messages(visible) == [("lead:b", "out")]
    assert len(messages[0]["tool_calls"]) == 2
