"""Background handoff behavior for tool-based fusion delegation."""

from __future__ import annotations

import json
import threading
from typing import Any

import fusion
import fusion_background
import fusion_prompts
import pytest
import runs
import serve_config

USAGE = {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def reply(content: str = "", *calls: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return message


class ScriptedWorker:
    def __init__(self, lead: list[dict[str, Any]], sidekick: list[dict[str, Any]]) -> None:
        self.script = {"lead": list(lead), "sidekick": list(sidekick)}
        self.seen: dict[str, list[list[dict[str, Any]]]] = {"lead": [], "sidekick": []}
        self.started = threading.Event()
        self.release = threading.Event()
        self.release_on_lead_call: int | None = None
        self.wait_for_sidekick_before_lead_call: int | None = None

    def __call__(self, slot, messages, tools):
        """Return the next scripted message for the active lane."""
        role = "sidekick" if messages[0]["content"] == fusion_prompts.SIDEKICK_POLICY else "lead"
        self.seen[role].append([dict(message) for message in messages])
        if role == "sidekick" and len(self.seen[role]) == 1:
            self.started.set()
            self.release.wait(5)
        if role == "lead" and len(self.seen[role]) == self.wait_for_sidekick_before_lead_call:
            self.started.wait(1)
        if role == "lead" and len(self.seen[role]) == self.release_on_lead_call:
            self.release.set()
        return self.script[role].pop(0), USAGE


@pytest.fixture(autouse=True)
def catalog(monkeypatch, tmp_path):
    """Use an isolated catalog, workspace, and in-memory run store."""
    path = tmp_path / "catalog.toml"
    path.write_text(
        '[fusion]\nmain = "lead-slot"\nsidekick = "sidekick-slot"\n'
        "handoff_wait_timeout_s = 0.02\nrun_max_turns = 32\n"
    )
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", fusion.FusionConfig(path))
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOT", str(tmp_path / "sandboxes"))
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOTS", str(shared))
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path / "runs"))
    serve_config._runs.clear()
    yield
    serve_config._runs.clear()


def install(monkeypatch, lead, sidekick) -> ScriptedWorker:
    """Patch the provider boundary with a deterministic worker."""
    worker = ScriptedWorker(lead, sidekick)
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    return worker


def tool_run(run_id: str, *, server_execution: bool = True, tools=None, workspace=None):
    """Create a run with tool delegation and selected sidekick execution options."""
    return fusion.FusionRun(
        run_id,
        "do the task",
        tools=tools,
        tool_options=fusion.FusionToolOptions(
            server_execution=server_execution, workspace=workspace
        ),
    )


def tool_results(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return tool-call results from one transcript."""
    return [(m["tool_call_id"], m["content"]) for m in messages if m.get("role") == "tool"]


def completion_count(messages: list[dict[str, Any]]) -> int:
    """Count completion notifications in a lead transcript."""
    return sum("<handoff_completion>" in str(message.get("content", "")) for message in messages)


def contains_text(messages: list[dict[str, Any]], text: str) -> bool:
    """Return whether one transcript message contains the requested text."""
    return any(text in str(message.get("content", "")) for message in messages)


def test_async_ack_read_wait_and_single_completion_notification(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("a", "sidekick", message="work", block=False)),
            reply("", call("b", "read_subagent", block=True, timeout=0.5)),
            reply("done"),
            reply("follow-up done"),
        ],
        [reply("the report")],
    )
    worker.release_on_lead_call = 2
    run = tool_run("notification")

    event = run.advance()

    assert event["status"] == "completed"
    assert tool_results(run.main_messages)[0] == (
        "delegate:a",
        "The sidekick started in the background.",
    )
    assert completion_count(worker.seen["lead"][2]) == 1
    assert completion_count(run.main_messages) == 1
    follow_up = run.advance(message="continue")
    assert follow_up["status"] == "completed"
    assert completion_count(worker.seen["lead"][3]) == 1


def test_read_subagent_timeout_then_blocking_wait(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("a", "sidekick", message="work", block=False)),
            reply("", call("b", "read_subagent", block=True, timeout=0.001)),
            reply("", call("c", "read_subagent", block=True, timeout=0.5)),
            reply("finished"),
        ],
        [reply("the report")],
    )
    worker.release_on_lead_call = 3
    run = tool_run("read-wait")

    event = run.advance()

    results = tool_results(worker.seen["lead"][2])
    assert any("timed out" in content for call_id, content in results if call_id == "delegate:b")
    assert event["status"] == "completed"
    assert run.handoffs is not None and run.handoffs.last is not None
    assert run.handoffs.last.report == "the report"


def test_second_sidekick_call_interrupts_without_starting_another_worker(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("a", "sidekick", message="initial", block=False)),
            reply("", call("b", "sidekick", message="change direction")),
            reply("", call("c", "read_subagent", block=True, timeout=0.5)),
            reply("finished"),
        ],
        [reply("working"), reply("updated report")],
    )
    worker.release_on_lead_call = 3
    worker.wait_for_sidekick_before_lead_call = 2
    run = tool_run("interrupt")

    assert run.advance()["status"] == "completed"

    assert len(worker.seen["sidekick"]) == 2
    assert any(
        message.get("role") == "user" and "<sidekick_interrupt>" in str(message.get("content"))
        for message in worker.seen["sidekick"][1]
    )
    assert tool_results(run.main_messages)[1] == (
        "delegate:b",
        "The sidekick received the interrupt and will use it at its next turn.",
    )


def test_cancel_stops_background_worker_and_pickle_has_only_plain_run_state(monkeypatch, tmp_path):
    worker = install(
        monkeypatch,
        [
            reply("", call("a", "sidekick", message="work", block=False)),
            reply("", call("b", "bash", command="ls")),
        ],
        [reply("report")],
    )
    run = tool_run("cancel", tools=[BASH_TOOL], workspace=str(tmp_path / "shared"))

    monkeypatch.setattr(runs, "RUN_STORE", "file")
    assert run.advance()["status"] == "awaiting_tools"
    assert worker.started.wait(1)
    runs._file_put(run)
    restored = runs._file_get(run.run_id)
    assert restored is not None and restored.handoffs is not None
    assert restored.handoffs.current is not None
    run.close()
    worker.release.set()
    assert fusion_background.wait(run.run_id, 1)
    fusion_background.sync_run(run)
    assert run.handoffs is not None and run.handoffs.last is not None
    assert run.handoffs.last.status == "cancelled"
    assert not fusion_background.running(run.run_id)


def test_user_message_stays_with_lead_until_lead_relays_it(monkeypatch, tmp_path):
    """Keep a user message out of the sidekick until the lead forwards it."""
    worker = install(
        monkeypatch,
        [
            reply(
                "",
                call("a", "sidekick", message="initial", block=False),
                call("b", "bash", command="ls"),
            ),
            reply("", call("c", "sidekick", message="clarify the user's latest ask")),
            reply("", call("d", "read_subagent", block=True, timeout=0.5)),
            reply("finished"),
        ],
        [reply("working"), reply("updated report")],
    )
    worker.release_on_lead_call = 3
    worker.wait_for_sidekick_before_lead_call = 2
    run = tool_run("user-message", tools=[BASH_TOOL], workspace=str(tmp_path / "shared"))

    assert run.advance()["status"] == "awaiting_tools"
    assert worker.started.wait(1)
    assert run.advance(message="clarify the user's latest ask")["status"] == "awaiting_tools"
    assert not contains_text(worker.seen["sidekick"][0], "clarify the user's latest ask")
    event = run.advance(tool_results=[{"tool_call_id": "lead:b", "content": "files"}])
    assert event["status"] == "completed"

    assert contains_text(worker.seen["sidekick"][1], "<sidekick_interrupt>")
    assert contains_text(worker.seen["sidekick"][1], "clarify the user's latest ask")


def test_file_store_receives_completed_background_state(monkeypatch, tmp_path):
    """Persist worker-owned transcript and report without serializing its thread."""
    monkeypatch.setattr(runs, "RUN_STORE", "file")
    worker = install(
        monkeypatch,
        [
            reply("", call("a", "sidekick", message="work", block=False)),
            reply("", call("b", "bash", command="ls")),
        ],
        [reply("persisted report")],
    )
    run = tool_run("file-store", tools=[BASH_TOOL], workspace=str(tmp_path / "shared"))

    assert run.advance()["status"] == "awaiting_tools"
    assert worker.started.wait(1)
    task = fusion_background._TASKS[run.run_id]
    runs._file_put(run)
    worker.release.set()
    assert fusion_background.wait(run.run_id, 1)
    task.thread.join(1)

    restored = runs._file_get(run.run_id)
    assert restored is not None
    assert restored.handoffs is not None and restored.handoffs.last is not None
    assert restored.handoffs.last.report == "persisted report"


def test_sidekick_client_tools_fall_back_and_never_reach_pending_calls(monkeypatch):
    install(
        monkeypatch,
        [reply("", call("a", "sidekick", message="inspect", block=False)), reply("done")],
        [reply("", call("s", "bash", command="ls")), reply("found files")],
    )
    run = tool_run("client-tools", server_execution=False, tools=[BASH_TOOL])

    event = run.advance()

    assert event["status"] == "awaiting_tools"
    assert [call["id"] for call in event["pending_tool_calls"]] == ["sidekick:s"]
    tool_names = {call["function"]["name"] for call in event["pending_tool_calls"]}
    assert tool_names.isdisjoint({"sidekick", "read_subagent"})
    event = run.advance(tool_results=[{"tool_call_id": "sidekick:s", "content": "files"}])
    assert event["status"] == "completed"
    assert "waited for report" in tool_results(run.main_messages)[0][1]
