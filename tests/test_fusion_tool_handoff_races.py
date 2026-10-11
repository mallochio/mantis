"""Regression coverage for background handoff races."""

from __future__ import annotations

import copy
import threading

import fusion
import fusion_background
import fusion_background_store
import pytest
import runs
import serve_config
from fusion_budget import FusionBudgetGuard
from fusion_handoff import HandoffManager

from tests.test_fusion_tool_handoffs import (
    BASH_TOOL,
    call,
    install,
    reply,
    tool_results,
    tool_run,
)

_BASELINE_USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 4,
    "total_tokens": 14,
    "prompt_tokens_details": {"cached_tokens": 2},
}
_MODEL_IDENTITY = "catalog-gateway/team/model-v1"
_BASELINE_MODELS = {
    _MODEL_IDENTITY: {"prompt_tokens": 10, "completion_tokens": 4, "cached_tokens": 2}
}


@pytest.fixture(autouse=True)
def race_catalog(monkeypatch, tmp_path):
    """Use a local catalog and workspace for concurrency regressions."""
    path = tmp_path / "catalog.toml"
    path.write_text(
        '[fusion]\nmain = "lead-slot"\nsidekick = "sidekick-slot"\n'
        "handoff_wait_timeout_s = 1\nrun_max_turns = 32\n"
    )
    shared = tmp_path / "shared"
    shared.mkdir()
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", fusion.FusionConfig(path))
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOT", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOTS", str(shared))
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path / "runs"))
    serve_config._runs.clear()
    yield shared
    serve_config._runs.clear()


class Transcript:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_role = "main"
        self.pending_tool_calls = []
        self.sidekick_entered = threading.Event()
        self.release_sidekick = threading.Event()
        self.main_entered = threading.Event()
        self.records = []

    def _append_tool_results(self, results):
        if self.active_role == "sidekick":
            self.sidekick_entered.set()
            self.release_sidekick.wait(1)
        call_id = self.pending_tool_calls[0]["id"]
        self.records.append((self.active_role, call_id, results[0]["tool_call_id"]))
        if self.active_role == "main":
            self.main_entered.set()


def _interleave_append(run):
    sidekick = threading.Thread(
        target=fusion_background.append_results,
        args=(
            run,
            "sidekick",
            [{"id": "sidekick-call"}],
            [{"tool_call_id": "sidekick-call", "content": "s"}],
        ),
    )
    main = threading.Thread(
        target=fusion_background.append_results,
        args=(run, "main", [{"id": "main-call"}], [{"tool_call_id": "main-call", "content": "m"}]),
    )
    sidekick.start()
    assert run.sidekick_entered.wait(1)
    main.start()
    overlapped = run.main_entered.wait(0.001)
    run.release_sidekick.set()
    sidekick.join(1)
    main.join(1)
    return overlapped


def test_role_result_append_is_atomic():
    run = Transcript()
    run.active_role = "outer"
    run.pending_tool_calls = [{"id": "existing-call"}]
    overlapped = _interleave_append(run)
    assert not overlapped
    assert set(run.records) == {
        ("sidekick", "sidekick-call", "sidekick-call"),
        ("main", "main-call", "main-call"),
    }
    assert run.active_role == "outer"
    assert run.pending_tool_calls == [{"id": "existing-call"}]


def test_interrupt_is_added_to_sidekick_transcript_once(monkeypatch):
    run = tool_run("interrupt-once")
    manager = HandoffManager(None)
    run.handoffs = manager
    queued = []
    monkeypatch.setattr(
        fusion_background, "interrupt", lambda run_id, text: queued.append(text) or True
    )

    def take_interrupts(run_id):
        messages = list(queued)
        queued.clear()
        return messages

    monkeypatch.setattr(fusion_background, "take_interrupts", take_interrupts)
    manager.interrupt(run, "delegate-call", "change direction")
    manager._inject_interrupts(run)

    tagged = [
        message
        for message in run.sidekick_messages
        if "<sidekick_interrupt>" in str(message.get("content"))
    ]
    assert len(tagged) == 1
    assert "change direction" in tagged[0]["content"]


def _resume_file_run(run_id):
    with runs._file_run_lock(run_id):
        loaded = runs._file_get(run_id)
        assert loaded is not None
        event = loaded.advance(tool_results=[{"tool_call_id": "lead:b", "content": "file list"}])
        runs._file_put(loaded)
    return event, loaded


def _usage_run(run_id, prompt, completion, total, cached):
    run = tool_run(run_id)
    run.handoffs = HandoffManager(None)
    run.handoffs.background_usage = {
        "token": "usage-token",
        "baseline_role": {"shared-slot": copy.deepcopy(_BASELINE_MODELS[_MODEL_IDENTITY])},
        "models": {},
        "budget": {"tokens": 0, "turns": 0},
        "applied": {},
    }
    run.usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    run.usage_models = {
        _MODEL_IDENTITY: {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached,
        }
    }
    return run


def _continue_file_handoff(run_id):
    task = fusion_background._TASKS[run_id]
    thread = task.thread
    event, loaded = _resume_file_run(run_id)
    thread.join(1)
    return event, loaded


def test_loaded_continuation_waits_for_and_persists_completion(monkeypatch, race_catalog):
    monkeypatch.setattr(runs, "RUN_STORE", "file")
    worker = install(
        monkeypatch,
        [
            reply(
                "",
                call("a", "sidekick", message="work", block=False),
                call("b", "bash", command="ls"),
            ),
            reply("", call("c", "read_subagent", block=True, timeout=1)),
            reply("lead answer"),
        ],
        [reply("completed background report")],
    )
    worker.release_on_lead_call = 2
    run = tool_run("loaded-wait", tools=[BASH_TOOL], workspace=str(race_catalog))

    assert run.advance()["status"] == "awaiting_tools"
    assert worker.started.wait(1)
    runs._file_put(run)
    _, loaded = _continue_file_handoff(run.run_id)
    assert ("delegate:c", "completed background report") in tool_results(loaded.main_messages)

    stored = runs._file_get(run.run_id)
    assert stored is not None and stored.report == "lead answer"
    assert stored.handoffs is not None and stored.handoffs.last is not None
    assert stored.handoffs.last.report == "completed background report"
    assert run.run_id not in fusion_background._TASKS


def _usage_pair():
    target = _usage_run("usage-target", 20, 5, 25, 3)
    source = _usage_run("usage-source", 18, 7, 25, 6)
    source.role_usage["sidekick"] = {
        "shared-slot": {"prompt_tokens": 15, "completion_tokens": 6, "cached_tokens": 6}
    }
    target.role_usage["main"] = {
        "shared-slot": {"prompt_tokens": 20, "completion_tokens": 5, "cached_tokens": 3}
    }
    source.handoffs.background_usage["models"][_MODEL_IDENTITY] = {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "cached_tokens": 4,
    }
    source.handoffs.background_usage["budget"] = {"tokens": 5, "turns": 1}
    target.budget, source.budget = FusionBudgetGuard(), FusionBudgetGuard()
    target.budget._tokens, target.budget._turns = 20, 5
    source.budget._tokens, source.budget._turns = 18, 4
    return target, source


def test_parallel_usage_deltas_add_once_for_same_model():
    target, source = _usage_pair()
    fusion_background_store.merge(target, source)
    fusion_background_store.merge(target, source)

    assert target.usage["prompt_tokens"] == 25
    assert target.usage["completion_tokens"] == 7
    assert target.usage["total_tokens"] == 32
    assert target.usage["prompt_tokens_details"]["cached_tokens"] == 7
    assert target.usage_models[_MODEL_IDENTITY] == {
        "prompt_tokens": 25,
        "completion_tokens": 7,
        "cached_tokens": 7,
    }
    assert (target.budget._tokens, target.budget._turns) == (25, 6)


def _cancel_and_release(run, worker):
    task = fusion_background._TASKS[run.run_id]
    thread = task.thread
    run.close()
    fusion_background.stop(run.run_id)
    worker.release.set()
    assert fusion_background.wait(run.run_id, 1)
    thread.join(1)


def test_cancel_after_provider_return_prevents_sidekick_write(monkeypatch, race_catalog):
    worker = install(
        monkeypatch,
        [
            reply(
                "",
                call("a", "sidekick", message="write", block=False),
                call("b", "bash", command="ls"),
            ),
            reply("lead answer"),
        ],
        [
            reply("", call("w", "write_file", path="blocked.txt", content="must not write")),
            reply("unused"),
        ],
    )
    run = tool_run("cancel-write", tools=[BASH_TOOL], workspace=str(race_catalog))

    assert run.advance()["status"] == "awaiting_tools"
    assert worker.started.wait(1)
    _cancel_and_release(run, worker)

    assert not (race_catalog / "blocked.txt").exists()
    assert run.handoffs is not None and run.handoffs.last is not None
    assert run.handoffs.last.status == "cancelled"


def test_lead_tool_choice_none_hides_all_tools():
    run = tool_run("tool-choice-none")
    run.tool_choice = "none"
    manager = HandoffManager(None)

    assert fusion_background.lead_tools(manager, run) == []
