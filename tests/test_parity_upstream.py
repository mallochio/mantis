"""Parity tests for upstream transport: SSE streaming, prompt-cache
breakpoints, and cache-aware cost accounting."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import serve


# ---------------------------------------------------------------------------
# SSE parsing and reassembly
# ---------------------------------------------------------------------------
def test_parse_sse_line_accepts_data_json_and_done():
    assert serve._parse_sse_line('data: {"a": 1}') == {"a": 1}
    assert serve._parse_sse_line('data:  {"a": 1} ') == {"a": 1}
    assert serve._parse_sse_line("data: [DONE]") == {}
    assert serve._parse_sse_line(": keep-alive comment") is None
    assert serve._parse_sse_line("") is None
    assert serve._parse_sse_line("data: not-json") is None
    assert serve._parse_sse_line("data: [1, 2]") is None  # non-object payload
    assert serve._parse_sse_line("event: message") is None


def test_assemble_stream_reconstructs_content_and_usage():
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}},
    ]
    out = serve._assemble_streamed_completion(chunks)
    assert out["choices"][0]["message"]["content"] == "Hello world"
    assert out["usage"] == {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4}


def test_assemble_stream_handles_empty_chunks_and_done_marker():
    out = serve._assemble_streamed_completion([{}, None, [], {"choices": []}])
    assert out["choices"][0]["message"] == {"role": "assistant", "content": None}
    assert out["usage"] is None


def test_assemble_stream_reconstructs_fragmented_tool_calls():
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "read", "arguments": '{"pa'},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": 'th": "/tmp/x"}'}}]
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"index": 1, "function": {"name": "write", "arguments": "{}"}}
                        ]
                    }
                }
            ]
        },
    ]
    out = serve._assemble_streamed_completion(chunks)
    calls = out["choices"][0]["message"]["tool_calls"]
    assert calls[0]["id"] == "call_1"
    assert calls[0]["function"]["name"] == "read"
    assert calls[0]["function"]["arguments"] == '{"path": "/tmp/x"}'
    assert calls[1]["function"]["name"] == "write"
    assert calls[1]["function"]["arguments"] == "{}"


def test_assemble_stream_keeps_reasoning_and_non_string_parts():
    chunks = [
        {"choices": [{"delta": {"reasoning": "think "}}]},
        {"choices": [{"delta": {"reasoning": "hard"}}]},
        {"choices": [{"delta": {"annotations": [{"type": "url_citation"}]}}]},
    ]
    message = serve._assemble_streamed_completion(chunks)["choices"][0]["message"]
    assert message["reasoning"] == "think hard"
    assert message["annotations"] == [{"type": "url_citation"}]


def test_assemble_stream_raises_on_provider_error_chunk():
    with pytest.raises(RuntimeError, match="provider stream error: boom"):
        serve._assemble_streamed_completion(
            [{"error": {"message": "boom"}}, {"choices": [{"delta": {"content": "x"}}]}]
        )


class _StreamResponse:
    def __init__(self, status_code: int, lines: list[str], text: str = "") -> None:
        self.status_code = status_code
        self._lines = list(lines)
        self.text = text

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://provider.test/v1/chat/completions")
            response = httpx.Response(self.status_code, text=self.text, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def iter_lines(self) -> Iterator[str]:
        yield from self._lines


class _StreamingClient:
    def __init__(self, responses: list[_StreamResponse]) -> None:
        self._responses = list(responses)
        self.posted: list[dict[str, Any]] = []

    def stream(self, _method: str, _url: str, headers=None, json=None) -> _StreamResponse:
        self.posted.append(json)
        return self._responses.pop(0)


def _sse(*chunks: dict[str, Any]) -> _StreamResponse:
    return _StreamResponse(200, ["data: " + json.dumps(c) for c in chunks] + ["data: [DONE]"])


def test_stream_completion_posts_stream_flags_and_assembles(monkeypatch):
    client = _StreamingClient(
        [
            _sse(
                {"choices": [{"delta": {"role": "assistant", "content": "hi"}}]},
                {"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            )
        ]
    )
    out = serve._stream_completion(
        client, "https://provider.test/chat/completions", {}, {"model": "m"}
    )
    assert out["choices"][0]["message"]["content"] == "hi"
    assert out["usage"]["total_tokens"] == 2
    posted = client.posted[0]
    assert posted["stream"] is True
    assert posted["stream_options"] == {"include_usage": True}
    assert posted["model"] == "m"


def test_stream_completion_raises_http_status_error():
    client = _StreamingClient([_StreamResponse(429, [], text="busy")])
    with pytest.raises(httpx.HTTPStatusError):
        serve._stream_completion(client, "u", {}, {"model": "m"})


# ---------------------------------------------------------------------------
# Prompt-cache breakpoints
# ---------------------------------------------------------------------------
def test_cache_breakpoints_mark_system_and_history_prefix(monkeypatch):
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "history turn"},
        {"role": "user", "content": "role prompt"},
    ]
    out = serve._with_cache_breakpoints(messages)
    assert out[0]["content"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1]["content"] == [
        {"type": "text", "text": "history turn", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[2] == messages[2]  # final role prompt is NOT marked
    assert messages == [  # input is not mutated
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "history turn"},
        {"role": "user", "content": "role prompt"},
    ]


def test_cache_breakpoints_preserve_existing_content_blocks():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "a"}]},
        {"role": "user", "content": "query"},
    ]
    out = serve._with_cache_breakpoints(messages)
    assert out[0]["content"] == [
        {"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}
    ]
    assert out[1]["content"] == "query"  # <3 messages: only system marked


def test_cache_breakpoints_skip_short_and_non_system_conversations():
    short = [{"role": "user", "content": "hi"}]
    assert serve._with_cache_breakpoints(short) == short
    no_system = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    out = serve._with_cache_breakpoints(no_system)
    assert out == no_system  # no system message, len == 2: nothing marked


def test_cache_breakpoints_leave_tool_call_messages_untouched():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "role prompt"},
    ]
    out = serve._with_cache_breakpoints(messages)
    assert out[1]["content"] is None  # tool-call message never gets a fake text block
    assert out[2]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_build_request_adds_breakpoints_only_for_claude_on_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    messages = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
    ]
    url, headers, body = serve._build_request(
        "openrouter/anthropic/claude-sonnet-5", messages, 100, 0.7
    )
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    url, headers, body = serve._build_request("openrouter/openai/gpt-5.6-luna", messages, 100, 0.7)
    assert body["messages"][0]["content"] == "you are helpful"
    monkeypatch.setenv("MANTIS_CACHE_BREAKPOINTS", "0")
    url, headers, body = serve._build_request(
        "openrouter/anthropic/claude-sonnet-5", messages, 100, 0.7
    )
    assert body["messages"][0]["content"] == "you are helpful"


# ---------------------------------------------------------------------------
# Cache-aware usage accounting
# ---------------------------------------------------------------------------
def test_add_usage_records_cached_tokens_per_model():
    run = serve.NativeRun("cache")
    run.add_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 8},
        },
        model="model-a",
    )
    run.add_usage(
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        model="model-b",
    )
    assert run.usage_models == {
        "model-a": {"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 8},
        "model-b": {"prompt_tokens": 1, "completion_tokens": 1},
    }


def test_usage_cost_uses_cache_read_price_when_known(monkeypatch):
    monkeypatch.setattr(serve, "_price_cache", {"m": (0.001, 0.002)})
    monkeypatch.setattr(serve, "_cache_read_price_cache", {"m": 0.0001})
    cost = serve._usage_cost(
        {"m": {"prompt_tokens": 100, "completion_tokens": 10, "cached_tokens": 90}}
    )
    assert cost == round(10 * 0.001 + 90 * 0.0001 + 10 * 0.002, 6)


def test_usage_cost_falls_back_to_full_price_without_cache_price(monkeypatch):
    monkeypatch.setattr(serve, "_price_cache", {"m": (0.001, 0.002)})
    monkeypatch.setattr(serve, "_cache_read_price_cache", {})
    cost = serve._usage_cost(
        {"m": {"prompt_tokens": 100, "completion_tokens": 0, "cached_tokens": 100}}
    )
    assert cost == 0.1  # cached at full price, conservative


def test_usage_cost_clamps_cached_to_prompt(monkeypatch):
    monkeypatch.setattr(serve, "_price_cache", {"m": (0.001, 0.002)})
    monkeypatch.setattr(serve, "_cache_read_price_cache", {"m": 0.0001})
    cost = serve._usage_cost(
        {"m": {"prompt_tokens": 10, "completion_tokens": 0, "cached_tokens": 999}}
    )
    assert cost == round(10 * 0.0001, 6)


def test_price_map_also_populates_cache_read_prices(monkeypatch):
    calls: list[str] = []

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "data": [
                    {
                        "id": "claude-a",
                        "pricing": {
                            "prompt": "3e-06",
                            "completion": "1.5e-05",
                            "input_cache_read": "3e-07",
                        },
                    },
                    {
                        "id": "gpt-a",
                        "pricing": {
                            "prompt": "0.000001",
                            "completion": "0.000002",
                            "input_cache_read": None,
                        },
                    },
                ]
            }

    def get(url: str, timeout: float) -> _Response:
        calls.append(url)
        return _Response()

    monkeypatch.setattr(serve.httpx, "get", get)
    monkeypatch.setattr(serve, "_price_cache", None)
    monkeypatch.setattr(serve, "_cache_read_price_cache", None)
    assert serve._price_map() == {
        "claude-a": (3e-06, 1.5e-05),
        "gpt-a": (0.000001, 0.000002),
    }
    assert serve._cache_read_prices() == {"claude-a": 3e-07}
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Streaming through the provider seam
# ---------------------------------------------------------------------------
def test_provider_response_streams_and_records_usage(monkeypatch):
    client = _StreamingClient(
        [
            _sse(
                {"choices": [{"delta": {"role": "assistant", "content": "ok"}}]},
                {"usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}},
            )
        ]
    )
    monkeypatch.setattr(serve, "_provider_client", client)
    run = serve.NativeRun("stream-run")
    serve._history_context.active_run = run
    try:
        data = serve._provider_response(
            "openrouter/model", [{"role": "user", "content": "hi"}], 10, 0.7
        )
    finally:
        serve._history_context.active_run = None
    assert data["choices"][0]["message"]["content"] == "ok"
    assert run.usage == {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}


def test_provider_response_uses_buffered_post_when_streaming_disabled(monkeypatch):
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "buffered"}}]}

    class Client:
        def __init__(self) -> None:
            self.posts = 0

        def post(self, *_a, **_k) -> Response:
            self.posts += 1
            return Response()

    client = Client()
    monkeypatch.setattr(serve, "_upstream_streaming_enabled", lambda: False)
    monkeypatch.setattr(serve, "_provider_client", client)
    data = serve._provider_response(
        "openrouter/model", [{"role": "user", "content": "hi"}], 10, 0.7
    )
    assert data["choices"][0]["message"]["content"] == "buffered"
    assert client.posts == 1


def test_provider_response_fails_over_between_stream_attempts(monkeypatch):
    client = _StreamingClient(
        [_StreamResponse(500, [], text="boom"), _sse({"choices": [{"delta": {"content": "ok"}}]})]
    )
    monkeypatch.setattr(serve, "_args", None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENCODE_API_KEY", "k")
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "openrouter/model-a,opencode-go/model-b")
    monkeypatch.setattr(serve, "_FAILOVER_DELAY", 0)
    monkeypatch.setattr(serve, "_provider_client", client)
    data = serve._provider_response(
        "openrouter/model-a", [{"role": "user", "content": "hi"}], 10, 0.7
    )
    assert data["choices"][0]["message"]["content"] == "ok"
    assert len(client.posted) == 2
