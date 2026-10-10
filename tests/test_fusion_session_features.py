"""Session persistence, cache keepalive, and handoff-history behavior."""

from __future__ import annotations

import threading
import time
from typing import Any

import fusion
import pytest


def _lane(run: Any, coordinator: Any, task: str = "broad task", name: str = "default") -> Any:
    return fusion.ExecutionLane(
        "lane0",
        run,
        coordinator,
        fusion.FusionSidekickAssignment(task=task),
        fusion.FusionWorkerProfile(name=name),
    )


def test_sidekick_context_carries_across_user_turns():
    """A healthy sidekick keeps its transcript for the next turn's lane."""
    run = fusion.FusionRun("carry", "first goal")
    coordinator = fusion.FusionCoordinator()
    lane = _lane(run, coordinator)
    lane.messages.append({"role": "assistant", "content": "prior findings about the workspace"})
    run.sidekick_lanes = [lane]

    run._apply_user_message("second question", coordinator)

    cache = run.sidekick_context_cache
    assert list(cache) == ["default"]
    assert all(msg.get("role") != "system" for msg in cache["default"])
    assert any("prior findings" in str(m.get("content")) for m in cache["default"])
    assert any("broad task" in str(m.get("content")) for m in cache["default"])
    replacement = fusion.ExecutionLane(
        "lane0",
        run,
        coordinator,
        fusion.FusionSidekickAssignment(task="follow-up work"),
        fusion.FusionWorkerProfile(name="default"),
        shared_messages=cache["default"],
    )
    assert replacement.messages[0]["role"] == "system"
    assert any(
        "prior findings" in str(m.get("content")) for m in replacement.messages[1:]
    )


def test_exhausted_sidekick_context_is_not_carried():
    """An exhausted lane starts the next turn from a clean slate."""
    run = fusion.FusionRun("carry", "goal")
    coordinator = fusion.FusionCoordinator()
    lane = _lane(run, coordinator)
    lane.exhausted = True
    run.sidekick_lanes = [lane]

    run._apply_user_message("second question", coordinator)

    assert run.sidekick_context_cache == {}


def test_context_cache_defaults_without_coordinator():
    """Snapshots work even when no coordinator is available."""
    run = fusion.FusionRun("carry", "goal")
    lane = _lane(run, fusion.FusionCoordinator())
    lane.messages.append({"role": "assistant", "content": "findings"})
    run.sidekick_lanes = [lane]

    run._apply_user_message("second question", None)

    assert list(run.sidekick_context_cache) == ["default"]


def test_handoff_history_tracks_turns_and_delegation():
    """Metadata exposes the structured handoff history of the run."""
    run = fusion.FusionRun("history", "goal")
    run._apply_user_message("turn two")
    run.plan = "the plan"
    run.sidekick_brief = "the brief"
    run._queue_sidekick_brief()

    metadata = fusion._orchestration_metadata(run)
    kinds = [entry["kind"] for entry in metadata["handoff_history"]]
    assert kinds == ["turn", "delegate"]
    assert metadata["handoff_history"][1]["subtask"] == "subtask-1"
    assert metadata["handoff_history"][1]["summary"] == "the brief"
    assert metadata["cache_keepalive_pings"] == 0


def test_handoff_history_is_bounded():
    """The history keeps only the most recent fifty entries."""
    run = fusion.FusionRun("history", "goal")
    for index in range(60):
        run._record_handoff("follow_up", summary=f"note {index}")
    assert len(run.handoff_history) == 50
    assert run.handoff_history[-1]["summary"] == "note 59"


def test_new_pickle_fields_survive_setstate():
    """Runs pickled before these features restore with safe defaults."""
    run = fusion.FusionRun("old", "goal")
    state = run.__dict__.copy()
    state.pop("lock", None)
    state.pop("request_lock", None)
    for key in ("handoff_history", "sidekick_context_cache"):
        state.pop(key, None)

    restored = fusion.FusionRun.__new__(fusion.FusionRun)
    restored.__setstate__(state)

    assert restored.handoff_history == []
    assert restored.sidekick_context_cache == {}


def test_compaction_cutoff_is_env_tunable(monkeypatch: pytest.MonkeyPatch):
    """MANTIS_COMPACTION_CUTOFF moves the demotion threshold."""
    config = fusion.FusionRoutingConfig(main=["cheap", "strong"], sidekick="side")
    router = fusion.FusionRouter(config, "main")
    monkeypatch.delenv("MANTIS_COMPACTION_CUTOFF", raising=False)
    assert router.select_at_compaction(0.7, "strong", cache_warm=False) == "cheap"
    monkeypatch.setenv("MANTIS_COMPACTION_CUTOFF", "0.5")
    assert router.select_at_compaction(0.7, "strong", cache_warm=False) == "strong"
    monkeypatch.setenv("MANTIS_COMPACTION_CUTOFF", "bogus")
    assert router.select_at_compaction(0.7, "strong", cache_warm=False) == "cheap"
    monkeypatch.setenv("MANTIS_COMPACTION_CUTOFF", "1.5")
    assert router.select_at_compaction(0.7, "strong", cache_warm=False) == "cheap"


def test_cache_keepalive_pings_warm_prefixes(monkeypatch: pytest.MonkeyPatch):
    """Keepalive re-sends a lane prefix with a tiny capped completion."""
    run = fusion.FusionRun("keepalive", "goal")
    lane = _lane(run, fusion.FusionCoordinator())
    lane.messages.append({"role": "assistant", "content": "work", "_mantis_model": "openrouter/x"})
    run.sidekick_lanes = [lane]
    run.main_lane = None
    calls: list[tuple[str, list[dict[str, Any]], int]] = []

    def fake_response(spec, messages, max_tokens, temperature, tools=None, timeout_s=None):
        calls.append((spec, messages, max_tokens))
        return {"role": "assistant", "content": "OK"}, {}

    monkeypatch.setattr(fusion.providers, "_provider_response", fake_response)

    assert run.cache_keepalive() is True
    spec, messages, max_tokens = calls[0]
    assert spec == "openrouter/x"
    assert messages[-1]["content"] == fusion._KEEPALIVE_PROMPT
    assert messages[-2]["content"] == "work"
    assert max_tokens == 4


def test_cache_keepalive_skips_routed_slots():
    """Virtual routes have no concrete provider prefix to refresh."""
    run = fusion.FusionRun("keepalive", "goal")
    lane = _lane(run, fusion.FusionCoordinator())
    run.sidekick_lanes = [lane]
    run.main_lane = None
    run.sidekick_slot = "mantis/base"

    assert run.cache_keepalive() is False


def test_keepalive_loop_caps_pings(monkeypatch: pytest.MonkeyPatch):
    """The watcher stops after the configured number of pings."""

    class FakeRun:
        status = "awaiting_tools"

        def cache_keepalive(self) -> bool:
            return True

    monkeypatch.setattr(fusion, "_keepalive_interval", lambda: 0.01)
    monkeypatch.setattr(fusion, "_keepalive_max_pings", lambda: 2)
    monkeypatch.setattr(fusion, "try_get_fusion_run", lambda run_id: FakeRun())
    fusion._KEEPALIVE_PINGS.clear()

    fusion._keepalive_loop("cap-test", threading.Event())

    assert fusion.keepalive_ping_count("cap-test") == 2
    assert "cap-test" not in fusion._KEEPALIVE_STOP


def test_keepalive_loop_stops_when_run_resumes(monkeypatch: pytest.MonkeyPatch):
    """The watcher exits without pinging once the run leaves awaiting_tools."""
    monkeypatch.setattr(fusion, "_keepalive_interval", lambda: 0.01)
    monkeypatch.setattr(fusion, "_keepalive_max_pings", lambda: 5)
    monkeypatch.setattr(fusion, "try_get_fusion_run", lambda run_id: None)
    fusion._KEEPALIVE_PINGS.clear()

    fusion._keepalive_loop("resume-test", threading.Event())

    assert fusion.keepalive_ping_count("resume-test") == 0


def test_start_keepalive_off_by_default(monkeypatch: pytest.MonkeyPatch):
    """No watcher thread is spawned unless the env knob is enabled."""
    monkeypatch.delenv("MANTIS_CACHE_KEEPALIVE", raising=False)

    fusion.start_keepalive("quiet")

    assert "quiet" not in fusion._KEEPALIVE_STOP


def test_start_keepalive_spawns_and_stops(monkeypatch: pytest.MonkeyPatch):
    """An enabled watcher runs and cleans up after stop_keepalive."""
    monkeypatch.setenv("MANTIS_CACHE_KEEPALIVE", "1")
    monkeypatch.setattr(fusion, "_keepalive_interval", lambda: 0.005)
    monkeypatch.setattr(fusion, "try_get_fusion_run", lambda run_id: None)
    fusion._KEEPALIVE_PINGS.clear()

    fusion.start_keepalive("live")
    deadline = time.monotonic() + 5
    while "live" in fusion._KEEPALIVE_STOP and time.monotonic() < deadline:
        time.sleep(0.01)

    assert "live" not in fusion._KEEPALIVE_STOP
    assert fusion.keepalive_ping_count("live") == 0


def test_advance_fusion_run_manages_keepalive(monkeypatch: pytest.MonkeyPatch):
    """Each advance stops the watcher first and restarts it on tool waits."""
    run = fusion.create_fusion_run("goal")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(fusion, "start_keepalive", lambda run_id: calls.append(("start", run_id)))
    monkeypatch.setattr(fusion, "stop_keepalive", lambda run_id: calls.append(("stop", run_id)))
    monkeypatch.setattr(fusion, "_write_learning_record", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        fusion.FusionRun,
        "advance_idempotent",
        lambda self, *args, **kwargs: {"status": "awaiting_tools", "run_id": self.run_id},
    )
    try:
        event = fusion.advance_fusion_run(run.run_id)
        assert event["status"] == "awaiting_tools"
        assert calls == [("stop", run.run_id), ("start", run.run_id)]
        monkeypatch.setattr(
            fusion.FusionRun,
            "advance_idempotent",
            lambda self, *args, **kwargs: {"status": "completed", "run_id": self.run_id},
        )
        fusion.advance_fusion_run(run.run_id)
        assert calls[-1] == ("stop", run.run_id)
    finally:
        fusion.serve_config._runs.pop(run.run_id, None)
        fusion._KEEPALIVE_PINGS.pop(run.run_id, None)

def test_orchestration_metadata_exports_role_usage():
    """Metadata reports real cumulative tokens and per-model usage."""
    run = fusion.FusionRun("history", "goal")
    run._add_role_usage(
        "main",
        "gpt-5_6-sol",
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 4},
        },
    )
    run._add_role_usage("sidekick", "gpt-5_6-luna", {"prompt_tokens": 5, "completion_tokens": 1})

    metadata = fusion._orchestration_metadata(run)

    assert metadata["usage"]["prompt_tokens"] == 15
    assert metadata["usage"]["completion_tokens"] == 3
    assert metadata["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    expected = sorted(
        [fusion._model_label("gpt-5_6-sol"), fusion._model_label("gpt-5_6-luna")]
    )
    assert sorted(metadata["usage_models"]) == expected
    assert metadata["usage_models"][fusion._model_label("gpt-5_6-sol")]["prompt_tokens"] == 10
