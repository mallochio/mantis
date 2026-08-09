"""Tests for the upstream Responses API protocol adapter."""

from __future__ import annotations

import copy

import pytest
from provider_protocols import (
    assemble_responses_stream,
    build_responses_body,
    chat_messages_to_input,
    responses_to_chat,
    uses_responses_api,
)


def test_protocol_selection_is_narrow():
    assert uses_responses_api("openrouter", "openai/gpt-5.6-sol")
    assert not uses_responses_api("openrouter", "openai-ish/model")
    assert not uses_responses_api("openrouter", "anthropic/claude-sonnet-5")
    assert not uses_responses_api("opencode-go", "openai/gpt-5.6-sol")


def test_chat_history_converts_to_stateless_responses_items_without_mutation():
    reasoning = {"type": "reasoning", "id": "rs_1", "status": "completed", "summary": []}
    messages = [
        {"role": "system", "content": "Be precise."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Inspect this"},
                {"type": "image_url", "image_url": {"url": "https://image", "detail": "high"}},
            ],
        },
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "reasoning_details": [reasoning],
            "tool_calls": [
                {
                    "id": "m00000000",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "m00000000", "content": "contents"},
    ]
    original = copy.deepcopy(messages)

    items = chat_messages_to_input(messages)

    assert messages == original
    assert items[0] == {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "Be precise."}],
    }
    assert items[1]["content"][1] == {
        "type": "input_image",
        "image_url": "https://image",
        "detail": "high",
    }
    assert items[2] == reasoning
    assert items[3]["id"] == "msg_m00000002"
    assert items[3]["status"] == "completed"
    assert items[3]["content"] == [{"type": "output_text", "text": "I will inspect it."}]
    assert items[4] == {
        "type": "function_call",
        "id": "fc_m00000000",
        "call_id": "m00000000",
        "name": "read",
        "arguments": '{"path":"x"}',
    }
    assert items[5] == {
        "type": "function_call_output",
        "call_id": "m00000000",
        "output": "contents",
    }


def test_build_responses_body_maps_tools_controls_and_structured_output():
    body = build_responses_body(
        "openai/gpt-5.6-sol",
        [{"role": "user", "content": "hi"}],
        4096,
        None,
        "medium",
        [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ],
        {"type": "function", "function": {"name": "lookup"}},
        {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
        },
        {
            "max_tokens": 123,
            "reasoning_effort": "high",
            "web_search_options": {"search_context_size": "low"},
        },
    )

    assert body["max_output_tokens"] == 123
    assert body["reasoning"] == {"effort": "high"}
    assert body["tools"][0] == {
        "type": "function",
        "name": "lookup",
        "description": "Look up a value",
        "parameters": {"type": "object"},
        "strict": True,
    }
    assert body["tools"][1] == {
        "type": "openrouter:web_search",
        "parameters": {"search_context_size": "low"},
    }
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "answer",
        "schema": {"type": "object"},
        "strict": True,
    }
    assert not {"messages", "max_tokens", "response_format", "reasoning_effort"} & body.keys()


def test_responses_output_converts_to_canonical_chat_shape():
    raw = {
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "Checked."}],
            },
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Answer",
                        "annotations": [{"type": "url_citation", "url": "https://source"}],
                    },
                    {"type": "output_text", "text": " complete.", "annotations": []},
                ],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "provider-call",
                "name": "lookup",
                "arguments": '{"key":"x"}',
            },
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens_details": {"reasoning_tokens": 2},
        },
    }

    result = responses_to_chat(raw)

    message = result["choices"][0]["message"]
    assert message["content"] == "Answer complete."
    assert message["tool_calls"][0]["id"] == "provider-call"
    assert message["reasoning"] == "Checked."
    assert message["reasoning_details"] == [raw["output"][0]]
    assert message["annotations"][0]["url"] == "https://source"
    assert result["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    }


def test_responses_stream_uses_terminal_response():
    response = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{}",
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }
    result = assemble_responses_stream(
        [
            {"type": "response.output_text.delta", "delta": "ignored fallback"},
            {"type": "response.completed", "response": response},
            {},
        ]
    )
    assert result["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
    assert result["usage"]["total_tokens"] == 5


def test_responses_stream_fallback_and_error():
    result = assemble_responses_stream(
        [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
        ]
    )
    assert result["choices"][0]["message"]["content"] == "Hello world"
    with pytest.raises(RuntimeError, match="stream broke"):
        assemble_responses_stream([{"type": "error", "message": "stream broke"}])


def test_failed_response_raises():
    with pytest.raises(RuntimeError, match="provider response failed: bad request"):
        responses_to_chat({"status": "failed", "error": {"message": "bad request"}})
