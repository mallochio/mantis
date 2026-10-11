"""Cache-prefix stability across ordinary Fusion turns."""

from __future__ import annotations

import copy

import fusion
import fusion_lead
import fusion_prompts


def test_fit_keeps_reasoning_and_tool_output_unchanged_while_under_budget(monkeypatch):
    coordinator = fusion.FusionCoordinator()
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "task"}]
    messages.extend(
        {"role": "assistant", "content": str(index), "reasoning": f"trace {index}"}
        for index in range(4)
    )
    original = copy.deepcopy(messages)
    monkeypatch.setattr(coordinator, "_token_sum", lambda *_: 100)

    fitted = coordinator._fit_messages("lead-slot", messages, None, 200)

    assert fitted is messages
    assert fitted == original


class LeadRun:
    def __init__(self):
        self.main_messages = [
            {"role": "system", "content": fusion_prompts.lead_system_prompt(None)},
            {"role": "user", "content": "task"},
        ]
        self.requests = []

    def _call_lane(self, coordinator, role, tools):
        self.requests.append(copy.deepcopy(self.main_messages))
        self.main_messages.append({"role": "assistant", "content": "answer"})
        return "answer", [], {}


def test_lead_turn_preserves_complete_prior_request_prefix():
    run = LeadRun()

    fusion_lead._lead_turn(run, None, [])
    run.main_messages.append({"role": "user", "content": "continue"})
    fusion_lead._lead_turn(run, None, [])

    first, second = run.requests
    assert second[: len(first)] == first
    assert len(first) == 2
    assert "parallel tool calls" in first[0]["content"]
