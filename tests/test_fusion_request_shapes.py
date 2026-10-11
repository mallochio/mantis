"""OpenAI-compatible request-shape tolerance tests for Fusion."""

from __future__ import annotations

import api


def test_stream_choices_and_parallel_tool_calls_are_accepted():
    for stream in (False, True):
        for choice in ("auto", "none", "required"):
            request = api.ChatRequest(
                model="mantis/fusion",
                messages=[{"role": "user", "content": "hello"}],
                tool_choice=choice,
                stream=stream,
                parallel_tool_calls=True,
                tool_options={},
            )
            assert request.stream is stream
            assert request.parallel_tool_calls is True
            assert api._uses_live_fusion_stream(request) is stream


def test_replayed_messages_keep_multiple_prompts_reasoning_and_content_parts():
    replay = api.ChatRequest(
        model="mantis/fusion",
        messages=[
            {"role": "system", "content": "base"},
            {"role": "developer", "content": "extra"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "r1",
                "reasoning": "r2",
                "reasoning_details": [{"type": "reasoning.text", "text": "r3"}],
            },
            {"role": "tool", "tool_call_id": "tool-1", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": "continue"},
        ],
        tool_options={},
    )
    dumped = [message.model_dump(exclude_none=True) for message in replay.messages]

    roles = ["system", "developer", "assistant", "tool", "user"]
    assert [message["role"] for message in dumped] == roles
    assert dumped[2]["reasoning_content"] == "r1"
    assert dumped[2]["reasoning_details"][0]["text"] == "r3"
    assert dumped[3]["content"][0]["text"] == "ok"
    assert api._content_to_str(replay.messages[3].content) == "ok"
