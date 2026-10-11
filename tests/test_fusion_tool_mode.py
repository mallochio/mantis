"""Tool-based fusion delegation: lead loop, persistent sidekick, and handoff results."""

from __future__ import annotations

import json
from typing import Any

import fusion
import fusion_limits
import fusion_prompts
import pytest
import runs
import serve_config

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
}


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    """Build one OpenAI-style tool call."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def reply(content: str = "", *calls: dict[str, Any], reasoning: str = "") -> dict[str, Any]:
    """Build one scripted assistant message."""
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [json.loads(json.dumps(c)) for c in calls]
    if reasoning:
        message["reasoning"] = reasoning
    return message


class ScriptedWorker:
    """Return scripted lead and sidekick replies and snapshot every request."""

    def __init__(self, lead: list[dict[str, Any]], sidekick: list[dict[str, Any]]) -> None:
        self.script = {"lead": list(lead), "sidekick": list(sidekick)}
        self.seen: dict[str, list[tuple[list[dict[str, Any]], list[dict[str, Any]]]]] = {
            "lead": [],
            "sidekick": [],
        }

    def __call__(self, slot, messages, tools):
        role = "sidekick" if messages[0]["content"] == fusion_prompts.SIDEKICK_POLICY else "lead"
        self.seen[role].append(([dict(m) for m in messages], list(tools or [])))
        return self.script[role].pop(0), USAGE


def use_tool_catalog(monkeypatch, tmp_path):
    """Point fusion at a hermetic catalog and per-test run directories."""
    path = tmp_path / "catalog.toml"
    path.write_text(
        '[fusion]\nmain = "gpt-5_6-sol"\nsidekick = "deepseek-v4-flash"\n'
        "sidekick_max_tool_rounds = 8\nrun_max_turns = 32\n"
    )
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", fusion.FusionConfig(path))
    monkeypatch.setattr(
        fusion_limits, "resolve_model_limits", lambda spec: fusion_limits.ModelLimits(200000, 8000)
    )
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path / "runs"))
    return path


@pytest.fixture(autouse=True)
def tool_catalog(monkeypatch, tmp_path):
    """Use a hermetic catalog and a clean in-memory run store."""
    serve_config._runs.clear()
    yield use_tool_catalog(monkeypatch, tmp_path)
    serve_config._runs.clear()


def tool_run(run_id: str = "tool", tools: list[dict[str, Any]] | None = None, **options: Any):
    """Create a fusion run that uses tool delegation."""
    return fusion.FusionRun(
        run_id,
        "build the feature",
        tools=tools,
        tool_options=fusion.FusionToolOptions(**options),
    )


def install(monkeypatch, lead, sidekick) -> ScriptedWorker:
    """Patch the worker boundary with a scripted worker."""
    worker = ScriptedWorker(lead, sidekick)
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    return worker


def tool_messages(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return (tool_call_id, content) for every tool message."""
    return [(m["tool_call_id"], m["content"]) for m in messages if m.get("role") == "tool"]


def test_lead_calls_sidekick_then_answers(monkeypatch):
    worker = install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="implement X")), reply("X is done.")],
        [reply("Report: implemented X, tests pass.")],
    )
    run = tool_run(tools=[BASH_TOOL])

    event = run.advance()

    assert event["status"] == "completed"
    assert event["report"] == "X is done."
    first_messages, first_tools = worker.seen["lead"][0]
    assert first_messages[0]["content"] == fusion_prompts.lead_system_prompt(None)
    assert first_messages[-1]["content"] == "build the feature"
    names = [tool["function"]["name"] for tool in first_tools]
    assert names == ["bash", "sidekick", "read_subagent"]
    second_messages, _ = worker.seen["lead"][1]
    assert tool_messages(second_messages) == [("delegate:c1", "Report: implemented X, tests pass.")]
    sidekick_messages, _ = worker.seen["sidekick"][0]
    assert [m["role"] for m in sidekick_messages] == ["system", "user"]
    assert sidekick_messages[1]["content"] == "implement X"
    assert second_messages[: len(first_messages)] == first_messages
    handoff = next(e for e in run.handoff_history if e["kind"] == "handoff")
    assert handoff["message"] == "implement X"
    assert handoff["report"] == "Report: implemented X, tests pass."
    assert handoff["status"] == "completed"
    assert run.completed_via == "answer"


def test_client_system_prompt_becomes_lead_base(monkeypatch):
    worker = install(monkeypatch, [reply("hi")], [])
    run = fusion.FusionRun(
        "base",
        "hello",
        messages=[
            {"role": "system", "content": "Client rules."},
            {"role": "user", "content": "hi"},
        ],
        tool_options=fusion.FusionToolOptions(),
    )

    assert run.advance()["report"] == "hi"
    messages, _ = worker.seen["lead"][0]
    assert messages[0]["content"] == fusion_prompts.lead_system_prompt("Client rules.")
    assert [m["role"] for m in messages] == ["system", "user"]


def test_mixed_client_tool_and_sidekick_merge_on_resume(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("c1", "bash", command="ls"), call("c2", "sidekick", message="go")),
            reply("Both done."),
        ],
        [reply("sidekick report")],
    )
    run = tool_run(tools=[BASH_TOOL])

    event = run.advance()

    assert event["status"] == "awaiting_tools"
    pending = event["pending_tool_calls"]
    assert [(c["id"], c["function"]["name"]) for c in pending] == [("lead:c1", "bash")]
    assert len(worker.seen["sidekick"]) == 1

    event = run.advance(tool_results=[{"tool_call_id": "lead:c1", "content": "a.txt"}])

    assert event["status"] == "completed"
    assert event["report"] == "Both done."
    messages, _ = worker.seen["lead"][1]
    assert tool_messages(messages) == [("lead:c1", "a.txt"), ("delegate:c2", "sidekick report")]


def test_sidekick_context_persists_across_handoffs(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("c1", "sidekick", message="first task")),
            reply("", call("c2", "sidekick", message="second task")),
            reply("All done."),
        ],
        [reply("first report"), reply("second report")],
    )
    run = tool_run()

    assert run.advance()["status"] == "completed"

    second, _ = worker.seen["sidekick"][1]
    assert [(m["role"], m["content"]) for m in second[1:]] == [
        ("user", "first task"),
        ("assistant", "first report"),
        ("user", "second task"),
    ]
    kinds = [e["kind"] for e in run.handoff_history]
    assert kinds.count("handoff") == 2


def test_escalation_reaches_lead_as_prefixed_result(monkeypatch):
    worker = install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="decide")), reply("I decided.")],
        [reply("ESCALATE_TO_MAIN: needs a design decision")],
    )
    run = tool_run()

    run.advance()

    messages, _ = worker.seen["lead"][1]
    assert tool_messages(messages) == [("delegate:c1", "ESCALATE_TO_MAIN: needs a design decision")]
    assert run.handoff_history[-2]["status"] == "escalated"


def test_read_subagent_returns_latest_report_and_block_false_still_blocks(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("c1", "sidekick", message="work", block=False)),
            reply("", call("c2", "read_subagent", block=True)),
            reply("done"),
        ],
        [reply("the report")],
    )
    run = tool_run()

    run.advance()

    messages, _ = worker.seen["lead"][2]
    assert tool_messages(messages) == [("delegate:c1", "the report"), ("delegate:c2", "the report")]
    assert run.handoffs is not None and run.handoffs.last is not None
    assert run.handoffs.last.block is False


def test_sidekick_client_tool_falls_back_to_awaiting_tools(monkeypatch):
    worker = install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="inspect")), reply("Inspected.")],
        [reply("", call("s1", "bash", command="ls")), reply("found a.txt")],
    )
    run = tool_run(tools=[BASH_TOOL])

    event = run.advance()

    assert event["status"] == "awaiting_tools"
    assert [c["id"] for c in event["pending_tool_calls"]] == ["sidekick:s1"]
    runs._file_put(run)
    restored = runs._file_get(run.run_id)
    assert isinstance(restored, fusion.FusionRun)
    event = restored.advance(tool_results=[{"tool_call_id": "sidekick:s1", "content": "a.txt"}])

    assert event["status"] == "completed"
    assert event["report"] == "Inspected."
    assert tool_messages(restored.sidekick_messages) == [("sidekick:s1", "a.txt")]
    assert len(worker.seen["sidekick"]) == 2


def test_usage_is_recorded_per_role(monkeypatch):
    install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="work")), reply("done")],
        [reply("report")],
    )
    run = tool_run()

    run.advance()

    assert set(run.role_usage) == {"main", "sidekick"}
    assert run.role_usage["sidekick"]["deepseek-v4-flash"]["prompt_tokens"] == 10
    assert run.role_usage["main"]["gpt-5_6-sol"]["prompt_tokens"] == 20


def test_budget_exhaustion_synthesizes_and_pairs_calls(monkeypatch):
    install(
        monkeypatch,
        [reply("", call(f"c{i}", "sidekick", message="loop")) for i in range(3)]
        + [reply("partial result")],
        [reply("report")] * 3,
    )
    run = tool_run()
    run.budget = fusion.FusionBudgetGuard(max_turns=3)

    event = run.advance()

    assert event["status"] == "completed"
    assert event["report"] == "partial result"
    assert run.completed_via == "budget"
    call_ids = {c["id"] for m in run.main_messages for c in m.get("tool_calls") or []}
    assert call_ids <= {m.get("tool_call_id") for m in run.main_messages}
