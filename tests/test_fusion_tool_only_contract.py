"""Regression coverage for the tool-only Fusion request contracts."""

import api
import fusion
import pytest
from fusion_types import FusionToolOptions
from pydantic import ValidationError


@pytest.mark.parametrize("mode", ["marker", "tool"])
def test_removed_delegation_option_is_rejected(mode):
    with pytest.raises(ValidationError, match="Extra inputs"):
        FusionToolOptions.model_validate({"delegation": mode})


def test_chat_rejects_removed_worker_profiles():
    with pytest.raises(ValidationError, match="worker_profiles is no longer supported"):
        api.ChatRequest.model_validate(
            {"model": "mantis/fusion", "messages": [], "worker_profiles": []}
        )


def test_delegate_rejects_removed_worker_profiles():
    with pytest.raises(ValidationError, match="Extra inputs"):
        api.FusionDelegateRequest.model_validate({"brief": "task", "worker_profiles": []})


@pytest.mark.parametrize(
    "setting",
    [
        "delegation",
        "main_tools",
        "max_follow_ups",
        "sidekick_max_follow_ups",
        "worker_profiles",
    ],
)
def test_removed_catalog_controls_are_rejected(tmp_path, setting):
    path = tmp_path / "catalog.toml"
    path.write_text(f'[fusion]\n{setting} = "removed"\n')
    config = fusion.FusionConfig(path)
    with pytest.raises(ValueError, match=f"unsupported Fusion settings: {setting}"):
        config._load()
    with pytest.raises(ValueError, match=f"unsupported Fusion settings: {setting}"):
        config._load()


def test_tool_only_default_accepts_required_private_tools():
    request = api.ChatRequest.model_validate(
        {
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "task"}],
            "tool_choice": "required",
        }
    )
    assert request.tool_choice == "required"


def test_new_run_uses_tool_loop_with_budget_and_tool_bundles(monkeypatch):
    def worker(self, slot, messages, tools=None):
        assert {tool["function"]["name"] for tool in tools} >= {"sidekick", "read_subagent"}
        return {"role": "assistant", "content": "plain answer"}, {}

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "tool-only-contract",
        "task",
        budget=fusion.FusionRunBudget(max_turns=4),
        tool_options=FusionToolOptions(enabled=["shell"]),
    )
    event = run.advance()
    assert event["status"] == "completed"
    assert event["report"] == "plain answer"
    assert run.handoffs is not None
    assert run.budget.turns == 1


@pytest.mark.parametrize("stream", [False, True])
def test_default_fusion_uses_live_stream_only_for_streamed_requests(stream):
    request = api.ChatRequest.model_validate(
        {
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "task"}],
            "stream": stream,
        }
    )
    assert api._uses_live_fusion_stream(request) is stream


def test_exhausted_budget_synthesizes_terminal_answer(monkeypatch):
    calls = []

    def worker(self, slot, messages, tools=None):
        calls.append((messages, tools))
        return {"role": "assistant", "content": "Verified partial result; work remains."}, {}

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "exhausted-tool-only", "task", budget=fusion.FusionRunBudget(max_turns=0)
    )
    event = run.advance()
    assert event["status"] == "completed"
    assert run.completed_via == "budget"
    assert event["report"] == "Verified partial result; work remains."
    assert calls[-1][1] is None
    assert "turn budget exceeded" in calls[-1][0][-2]["content"]
    assert run.budget is not None
    assert run.follow_up_capped
