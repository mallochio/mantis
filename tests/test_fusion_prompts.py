from __future__ import annotations

import re
from typing import Any

from apps.api.fusion_prompts import (
    LEAD_DELEGATION_POLICY,
    READ_SUBAGENT_TOOL,
    SIDEKICK_POLICY,
    SIDEKICK_TOOL,
    lead_system_prompt,
    lead_turn_reminder,
    sidekick_system_prompt,
)

SENTENCE_STARTERS = {
    "After", "Answer", "Ask", "At", "Avoid", "Blocking", "Calling", "Delegate", "Describe",
    "Do", "For", "Include", "Follow", "Leave", "Make", "Never", "Own", "Pass", "Report",
    "Review", "Stop", "Take", "The", "This", "Use", "When", "Write", "You",
}


def test_lead_policy_covers_ownership_and_handoff_rules() -> None:
    policy = LEAD_DELEGATION_POLICY.lower()
    for concept in (
        "own the outcome",
        "correctness-critical",
        "blocking is the default",
        "interrupts it with your new message",
        "first person",
        "cannot see user messages",
        "do not hand off an unsettled ask",
        "one final full gate",
        "take over if it remains stuck",
        "hypothesis into a confirmed conclusion",
    ):
        assert concept in policy


def test_sidekick_policy_preserves_limits_and_escalation() -> None:
    policy = sidekick_system_prompt().lower()
    for concept in (
        "hands-on executor",
        "cannot talk to the user",
        "request new secrets",
        "do not commit, push, open pull requests",
        "escalate_to_main:",
        "actual command output",
        "shared with the lead",
        "leave long-running processes running",
    ):
        assert concept in policy
    assert "ESCALATE_TO_MAIN:" in sidekick_system_prompt()


def test_lead_prompt_and_turn_reminder_join_sections() -> None:
    prompt = lead_system_prompt("base instructions")
    reminder = lead_turn_reminder()
    assert prompt.startswith("base instructions")
    assert LEAD_DELEGATION_POLICY in prompt
    assert "<delegation_guidance>" in reminder
    assert "</delegation_guidance>" in reminder
    assert "parallel tool calls" in reminder
    assert "same response" in reminder


def _assert_function_schema(schema: dict[str, Any], expected_name: str) -> None:
    assert schema["type"] == "function"
    function = schema["function"]
    assert function["name"] == expected_name
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert isinstance(parameters["properties"], dict)
    assert parameters["additionalProperties"] is False


def test_tool_schemas_match_the_delegation_contract() -> None:
    _assert_function_schema(SIDEKICK_TOOL, "sidekick")
    sidekick_parameters = SIDEKICK_TOOL["function"]["parameters"]
    assert sidekick_parameters["required"] == ["message"]
    assert sidekick_parameters["properties"]["message"]["type"] == "string"
    assert sidekick_parameters["properties"]["block"]["type"] == "boolean"
    assert sidekick_parameters["properties"]["block"]["default"] is True
    sidekick_description = SIDEKICK_TOOL["function"]["description"].lower()
    assert "blocking is the default" in sidekick_description
    assert "interrupts it" in sidekick_description
    assert "tool result" in sidekick_description
    assert "notification" in sidekick_description
    _assert_function_schema(READ_SUBAGENT_TOOL, "read_subagent")
    read_parameters = READ_SUBAGENT_TOOL["function"]["parameters"]
    assert read_parameters["properties"]["block"]["type"] == "boolean"
    assert read_parameters["properties"]["timeout"]["type"] == "number"
    assert "timeout" not in read_parameters["required"]


def test_policies_avoid_external_product_names() -> None:
    text = f"{LEAD_DELEGATION_POLICY} {SIDEKICK_POLICY}"
    assert not re.search(r"(?:https?://|www\.)", text, re.IGNORECASE)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        capitalized_words = re.findall(r"\b[A-Z][a-z]+\b", sentence)
        assert all(word in SENTENCE_STARTERS for word in capitalized_words[1:])
        if capitalized_words:
            assert capitalized_words[0] in SENTENCE_STARTERS
