"""Learning telemetry parity tests: test isolation, harness tool detection,
abandoned-run records, and error detail persistence."""

from __future__ import annotations

import json
from typing import Any

import serve


def _learn_env(monkeypatch, tmp_path) -> None:
    """Enable learning writes into a scratch directory."""
    monkeypatch.setenv("MANTIS_LEARNING", "1")
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_LEARNING_INSTANCE", "test/host")


def _records(tmp_path) -> list[dict[str, Any]]:
    path = tmp_path / "runs-test_host.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _insert(run: serve.NativeRun) -> None:
    with serve._runs_lock:
        serve._runs[run.run_id] = run


def test_conftest_disables_learning_by_default():
    # The dev shell sets MANTIS_LEARNING=1; the autouse conftest guard must
    # keep test runs out of the production learning file.
    assert serve._learning_enabled() is False


def test_is_test_matches_harness_ipython_tool():
    pending = {
        "asst": {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "ipython",
                        "arguments": json.dumps({"code": "uv run pytest tests -q"}),
                    },
                }
            ]
        }
    }
    run = serve.NativeRun("t1")
    run.record_tool_results(pending, [{"tool_call_id": "c1", "is_error": False}])
    assert run.tool_observations[0]["is_test"] is True


def test_is_test_matches_bash_command_tool():
    pending = {
        "asst": {
            "tool_calls": [
                {
                    "id": "c1",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": "python3 -m unittest -v"}),
                    },
                }
            ]
        }
    }
    run = serve.NativeRun("t2")
    run.record_tool_results(pending, [{"tool_call_id": "c1", "is_error": True}])
    assert run.tool_observations[0]["is_test"] is True
    assert run.tool_observations[0]["is_error"] is True


def test_is_test_rejects_non_test_tool_payloads():
    def observe(name: str, args: dict[str, Any]) -> bool:
        pending = {
            "asst": {
                "tool_calls": [
                    {"id": "c1", "function": {"name": name, "arguments": json.dumps(args)}}
                ]
            }
        }
        run = serve.NativeRun("t3")
        run.record_tool_results(pending, [{"tool_call_id": "c1", "is_error": False}])
        return run.tool_observations[0]["is_test"]

    assert observe("ipython", {"code": "print('hello')"}) is False
    assert observe("read", {"path": "tests/test_api.py"}) is False  # unknown tool name
    assert observe("bash", {"command": "ls -la"}) is False


def test_record_activity_persists_error_detail():
    run = serve.NativeRun("t4")
    run.record_activity("failover", model="openrouter/x", status="failed", detail="HTTP 500: boom")
    assert run._activity[-1]["error"] == "HTTP 500: boom"
    run.record_activity("step", role="Worker", model="openrouter/x")
    assert "error" not in run._activity[-1]


def test_delete_run_records_error_detail(monkeypatch, tmp_path):
    _learn_env(monkeypatch, tmp_path)
    run = serve.TrinityRun("b" * 32, [{"role": "user", "content": "hi"}], [])
    _insert(run)
    assert serve.delete_run(run.run_id, error="openrouter/x returned HTTP 400: bad request")
    [record] = _records(tmp_path)
    assert record["terminated_by"] == "deleted"
    assert record["error"] == "openrouter/x returned HTTP 400: bad request"


def test_sweeper_records_abandoned_progressed_run(monkeypatch, tmp_path):
    _learn_env(monkeypatch, tmp_path)
    progressed = serve.TrinityRun("a" * 32, [{"role": "user", "content": "hi"}], [])
    progressed.turns.append({"role": "Worker", "agent_id": 0, "model_name": "m1", "reply": "x"})
    progressed.last_active = 0
    _insert(progressed)
    untouched = serve.NativeRun("c" * 32)  # never advanced; must not be recorded
    untouched.last_active = 0
    _insert(untouched)
    serve._sweep_runs()
    assert progressed.run_id not in serve._runs
    assert untouched.run_id not in serve._runs
    [record] = _records(tmp_path)
    assert record["terminated_by"] == "abandoned"
    assert record["turn_count"] == 1
    assert record["steps"] == [{"role": "Worker", "model": "m1", "agent_id": 0}]
    assert record["error"] is None


def test_learning_record_includes_steps_and_error():
    run = serve.TrinityRun("d" * 32, [{"role": "user", "content": "hi"}], [], slot_models=["m0"])
    run.turns = [
        {"role": "Worker", "agent_id": 1, "model_name": "m1", "reply": "x"},
        {"role": "Verifier", "agent_id": 2, "model_name": "m2", "reply": "ACCEPT"},
    ]
    record = serve._learning_record(
        run, {"type": "final", "terminated_by": "verifier_accept", "error": None}
    )
    assert record["steps"] == [
        {"role": "Worker", "model": "m1", "agent_id": 1},
        {"role": "Verifier", "model": "m2", "agent_id": 2},
    ]
    assert record["error"] is None
    record_with_error = serve._learning_record(
        run, {"type": "error", "terminated_by": "deleted", "error": "boom"}
    )
    assert record_with_error["error"] == "boom"
