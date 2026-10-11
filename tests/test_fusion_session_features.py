"""Session persistence, cache keepalive, and handoff-history behavior."""

from __future__ import annotations

import fusion
import pytest

from tests.test_fusion_tool_mode import call, install, reply


def test_handoff_history_tracks_turns_and_delegation(monkeypatch):
    worker = install(
        monkeypatch,
        [reply("", call("c1", "sidekick", message="the brief")), reply("done")],
        [reply("the report")],
    )
    run = fusion.FusionRun("history", "goal")
    run.advance(message="turn two")
    metadata = fusion._orchestration_metadata(run)
    kinds = [entry["kind"] for entry in metadata["handoff_history"]]
    assert kinds == ["turn", "handoff", "answer"]
    assert metadata["handoff_history"][1]["message"] == "the brief"
    assert metadata["handoff_history"][1]["report"] == "the report"
    assert len(worker.seen["sidekick"]) == 1


def test_sidekick_context_carries_across_user_turns(monkeypatch):
    worker = install(
        monkeypatch,
        [
            reply("", call("c1", "sidekick", message="first task")),
            reply("first answer"),
            reply("", call("c2", "sidekick", message="second task")),
            reply("second answer"),
        ],
        [reply("prior findings"), reply("follow-up findings")],
    )
    run = fusion.FusionRun("carry", "goal")
    run.advance()
    run.advance(message="second question")
    second_messages = worker.seen["sidekick"][1][0]
    assert [m["content"] for m in second_messages[1:]] == [
        "first task",
        "prior findings",
        "second task",
    ]
    assert second_messages[0]["role"] == "system"


def test_handoff_history_is_bounded():
    """The history keeps only the most recent fifty entries."""
    run = fusion.FusionRun("history", "goal")
    for index in range(60):
        run._record_handoff("follow_up", summary=f"note {index}")
    assert len(run.handoff_history) == 50
    assert run.handoff_history[-1]["summary"] == "note 59"


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
    expected = sorted([fusion._model_label("gpt-5_6-sol"), fusion._model_label("gpt-5_6-luna")])
    assert sorted(metadata["usage_models"]) == expected
    assert metadata["usage_models"][fusion._model_label("gpt-5_6-sol")]["prompt_tokens"] == 10


def test_merge_native_usage_folds_serve_layer_models():
    """Serve-layer usage fills metadata when lane accounting stayed empty."""
    metadata = {
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        },
        "usage_models": {},
    }
    models = {"m|low": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4}}

    merged = fusion._merge_native_usage(metadata, models)

    assert merged["usage"]["prompt_tokens"] == 10
    assert merged["usage"]["completion_tokens"] == 2
    assert merged["usage"]["prompt_tokens_details"]["cached_tokens"] == 4
    assert merged["usage_models"] == models
    assert metadata["usage"]["prompt_tokens"] == 0
