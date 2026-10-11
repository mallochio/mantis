"""Regression tests for tool-only delegation limits and sidekick authority."""

from __future__ import annotations

import fusion
import fusion_background
import fusion_lead
import tool_exec

from tests.test_fusion import BASH_TOOL, DEFAULT_USAGE, SequenceWorker
from tests.test_fusion_tool_mode import call, install, reply, use_tool_catalog


def test_fusion_records_tool_observations(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    install(
        monkeypatch,
        [reply("", call("a", "sidekick", message="verify")), reply("done")],
        [reply("", call("s", "bash", command="echo hello")), reply("verified")],
    )
    run = fusion.FusionRun("tool-observation", "goal", tools=[BASH_TOOL])
    assert run.advance()["status"] == "awaiting_tools"
    event = run.advance(tool_results=[{"tool_call_id": "sidekick:s", "content": "hello"}])
    assert event["status"] == "completed"
    assert run.tool_observations == [
        {"name": "bash", "is_error": False, "is_test": False},
        {"name": "sidekick", "is_error": False, "is_test": False},
    ]


def test_fusion_unlimited_sidekick_exceeds_previous_16_round_cap(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    install(
        monkeypatch,
        [reply("", call("a", "sidekick", message="verify")), reply("done")],
        [
            *[reply("", call(str(i), "bash", command="echo hello")) for i in range(17)],
            reply("verified"),
        ],
    )
    run = fusion.FusionRun("unlimited-rounds", "goal", tools=[BASH_TOOL])
    run.sidekick_max_tool_rounds = 0
    run.run_max_turns = 64
    assert run.advance()["status"] == "awaiting_tools"
    for i in range(17):
        event = run.advance(tool_results=[{"tool_call_id": f"sidekick:{i}", "content": "ok"}])
    assert event["status"] == "completed"
    assert run.handoffs.tool_rounds == 17


def test_fusion_run_guard_stops_unproductive_unlimited_sidekick(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = SequenceWorker(
        [
            ("", [call("a", "sidekick", message="verify")], DEFAULT_USAGE),
            ("partial", None, DEFAULT_USAGE),
        ],
        [("", [call(str(i), "bash", command="echo hello")], DEFAULT_USAGE) for i in range(10)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("run-guard", "goal", tools=[BASH_TOOL])
    run.sidekick_max_tool_rounds = 0
    run.budget = fusion.FusionBudgetGuard(max_turns=5)
    event = run.advance()
    for i in range(10):
        if event["status"] == "completed":
            break
        event = run.advance(tool_results=[{"tool_call_id": f"sidekick:{i}", "content": "same"}])
    assert event["status"] == "completed"
    assert run.completed_via == "budget"
    assert run.handoffs.tool_rounds < 10


def test_fusion_uses_slots_frozen_on_run(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    worker = SequenceWorker(
        [
            ("", [call("a", "sidekick", message="verify")], DEFAULT_USAGE),
            ("done", None, DEFAULT_USAGE),
        ],
        [("verified", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("frozen-slots", "goal")
    run.main_router = run.sidekick_router = None
    run.main_slot, run.sidekick_slot = "run-main", "run-sidekick"
    coordinator = fusion.FusionCoordinator()
    coordinator.main_slot, coordinator.sidekick_slot = "other-main", "other-sidekick"
    assert run.advance(coordinator=coordinator)["status"] == "completed"
    assert [slot for slot, _, _ in worker.calls] == ["run-main", "run-sidekick", "run-main"]


def test_fusion_lane_cache_namespaces(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    namespaces = []

    def run_once(run_id):
        worker = SequenceWorker(
            [
                ("", [call("a", "sidekick", message="verify")], DEFAULT_USAGE),
                ("done", None, DEFAULT_USAGE),
            ],
            [("verified", None, DEFAULT_USAGE)],
        )
        run = fusion.FusionRun(run_id, "goal", tools=[BASH_TOOL])

        def capture(_coordinator, slot, messages, tools):
            namespaces.append(run.cache_namespace)
            return worker(slot, messages, tools)

        monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", capture)
        assert run.advance()["status"] == "completed"

    run_once("cache-a")
    run_once("cache-b")
    assert namespaces[0] == namespaces[2]
    assert namespaces[0] != namespaces[1]
    assert namespaces[:3] == namespaces[3:]


def test_server_execution_does_not_hijack_a_client_declared_tool(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    install(
        monkeypatch,
        [reply("", call("a", "sidekick", message="verify")), reply("done")],
        [reply("", call("s", "bash", command="echo hello")), reply("verified")],
    )
    run = fusion.FusionRun(
        "client-authority",
        "goal",
        tools=[BASH_TOOL],
        tool_options=fusion.FusionToolOptions(enabled=["shell"], server_execution=True),
    )

    def forbidden(*args):
        raise AssertionError("client-declared tool must not execute on the server")

    monkeypatch.setattr(tool_exec, "execute_calls", forbidden)
    event = run.advance()
    assert event["status"] == "awaiting_tools"
    assert [call["id"] for call in event["pending_tool_calls"]] == ["sidekick:s"]


def test_server_execution_still_runs_tools_fusion_injected(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    install(
        monkeypatch,
        [reply("", call("a", "sidekick", message="verify")), reply("done")],
        [reply("", call("s", "bash", command="echo hello")), reply("verified")],
    )
    executed = []

    def execute(calls, workspace):
        executed.extend(calls)
        return [{"tool_call_id": call["id"], "content": "ok"} for call in calls]

    monkeypatch.setattr(tool_exec, "execute_calls", execute)
    run = fusion.FusionRun(
        "server-authority",
        "goal",
        tool_options=fusion.FusionToolOptions(enabled=["shell"], server_execution=True),
    )
    assert run.advance()["status"] == "completed"
    assert [call["id"] for call in executed] == ["sidekick:s"]


def test_server_execution_off_claims_nothing(monkeypatch, tmp_path):
    use_tool_catalog(monkeypatch, tmp_path)
    run = fusion.FusionRun(
        "server-off",
        "goal",
        tools=[BASH_TOOL],
        tool_options=fusion.FusionToolOptions(enabled=["shell"]),
    )
    fusion_lead.prepare_run(run)
    assert fusion_background.server_schemas(run.handoffs, run) == []
    assert fusion_background.sidekick_tools(run.handoffs, run) == [BASH_TOOL]
