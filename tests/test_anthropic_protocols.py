"""Regression coverage for native Anthropic worker transport."""

from __future__ import annotations

import json

from anthropic_protocols import (
    anthropic_to_chat,
    assemble_anthropic_stream,
    build_anthropic_body,
    chat_to_anthropic,
)


def test_native_body_lifts_system_and_enables_thinking():
    body = build_anthropic_body(
        "bedrock/anthropic/claude-opus-5",
        [{"role": "system", "content": "Be exact."}, {"role": "user", "content": "Solve it."}],
        4096,
        "medium",
        [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read",
                    "parameters": {"type": "object"},
                },
            }
        ],
        {"type": "function", "function": {"name": "read"}},
    )
    assert body["system"] == [{"type": "text", "text": "Be exact."}]
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 4095}
    assert body["tools"] == [
        {"name": "read", "description": "Read", "input_schema": {"type": "object"}}
    ]
    assert body["tool_choice"] == {"type": "tool", "name": "read"}


def test_signed_thinking_and_provider_tool_id_survive_tool_continuation():
    raw = [
        {"type": "thinking", "thinking": "check", "signature": "signed"},
        {"type": "tool_use", "id": "provider-call", "name": "read", "input": {"path": "x"}},
    ]
    chat = anthropic_to_chat({"content": raw, "usage": {"input_tokens": 2, "output_tokens": 3}})
    assistant = chat["choices"][0]["message"]
    assistant["_anthropic_tool_ids"] = {"c0": "provider-call"}
    assistant["tool_calls"][0]["id"] = "c0"
    system, history = chat_to_anthropic(
        [
            {"role": "system", "content": "S"},
            assistant,
            {"role": "tool", "tool_call_id": "c0", "content": "file"},
        ]
    )
    assert system == [{"type": "text", "text": "S"}]
    assert history[0]["content"] == raw
    assert history[1]["content"][0]["tool_use_id"] == "provider-call"


def test_null_thinking_is_not_replayed_to_the_provider():
    raw = [
        {"type": "thinking", "thinking": None, "signature": "invalid"},
        {"type": "text", "text": "answer"},
    ]
    assistant = anthropic_to_chat({"content": raw})["choices"][0]["message"]

    assert assistant["_anthropic_content"] == [{"type": "text", "text": "answer"}]
    assert chat_to_anthropic([assistant])[1][0]["content"] == assistant["_anthropic_content"]


def test_stream_assembly_retains_thinking_signature_and_tool_input():
    result = assemble_anthropic_stream(
        [
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "thinking", "thinking": "", "signature": "sig"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "reason"},
            },
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "p1", "name": "read", "input": {}},
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps({"path": "x"})},
            },
            {"type": "message_delta", "usage": {"input_tokens": 4, "output_tokens": 5}},
        ]
    )
    message = result["choices"][0]["message"]
    assert message["_anthropic_content"][0] == {
        "type": "thinking",
        "thinking": "reason",
        "signature": "sig",
    }
    assert message["tool_calls"][0]["function"]["arguments"] == '{"path": "x"}'
    assert result["usage"]["total_tokens"] == 9
