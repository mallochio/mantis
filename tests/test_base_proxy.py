"""Unit tests for base_proxy session and header handling."""

from __future__ import annotations

import base_proxy
import httpx
import providers


def test_session_id_prefers_switchyard_header():
    assert (
        base_proxy.session_id(
            {"x-switchyard-session-id": "switch", "x-route-session": "route"},
            type("R", (), {"metadata": None, "user": None}),
        )
        == "switch"
    )


def test_session_id_falls_back_to_metadata():
    body = type("R", (), {"metadata": {"session_id": "meta"}, "user": "user"})
    assert base_proxy.session_id({}, body) == "meta"


def test_session_id_falls_back_to_user():
    body = type("R", (), {"metadata": None, "user": "user"})
    assert base_proxy.session_id({}, body) == "user"


def test_router_headers_include_grok_conv_id_from_session():
    body = type("R", (), {"metadata": {"session_id": "conv-123"}, "user": None})
    headers = base_proxy.router_headers({}, body)
    assert headers["x-switchyard-session-id"] == "conv-123"
    assert headers["x-grok-conv-id"] == "conv-123"


def test_router_headers_omit_grok_conv_id_without_session():
    body = type("R", (), {"metadata": None, "user": None})
    headers = base_proxy.router_headers({}, body)
    assert "x-switchyard-session-id" not in headers
    assert "x-grok-conv-id" not in headers


def test_router_error_parses_json_error():
    response = httpx.Response(
        400,
        json={"error": {"type": "upstream_error", "message": "bad request"}},
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 400
    body = result.body.decode()
    assert "bad request" in body


def test_router_error_falls_back_to_text_body():
    response = httpx.Response(
        400,
        text="Service temporarily unavailable",
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 400
    body = result.body.decode()
    assert "Service temporarily unavailable" in body


def test_router_error_parses_sse_error_stream():
    sse = 'data: {"error": {"type": "server_error", "message": "model did not respond"}}\n\n'
    response = httpx.Response(
        400,
        text=sse,
        headers={"content-type": "text/event-stream"},
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 400
    body = result.body.decode()
    assert "model did not respond" in body


def _make_request(data: dict) -> object:
    class Request:
        def model_dump(self, *, exclude_none, exclude):
            excluded = {*exclude}
            return {k: v for k, v in data.items() if k not in excluded}

    return Request()


def test_router_body_strips_endpoint_bound_reasoning_details():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "pong",
                        "reasoning_details": [
                            {"type": "reasoning.summary", "text": "portable"},
                            {"type": "reasoning.encrypted", "data": "endpoint-bound"},
                            {"type": "compaction.encrypted", "data": "endpoint-bound"},
                        ],
                    }
                ],
            }
        )
    )
    assert body["messages"][0]["reasoning_details"] == [
        {"type": "reasoning.summary", "text": "portable"}
    ]


def test_router_body_drops_all_endpoint_bound_reasoning_details():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [
                    {
                        "role": "assistant",
                        "reasoning_details": [{"type": "reasoning.encrypted", "data": "secret"}],
                    }
                ],
            }
        )
    )
    assert "reasoning_details" not in body["messages"][0]


def test_router_body_normalizes_reasoning_object():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "reasoning": {"effort": "low", "max_tokens": 1000},
                "user": "u",
                "metadata": {"session_id": "s"},
            }
        )
    )
    efficient = base_proxy._base_efficient_model() or ""
    expected = providers._coerce_reasoning_effort(efficient, "low")
    assert body.get("reasoning_effort") == expected
    assert "reasoning" not in body
    assert "user" not in body
    assert "metadata" not in body


def test_router_body_preserves_existing_reasoning_effort():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "low"},
                "reasoning_effort": "high",
            }
        )
    )
    assert body["reasoning_effort"] == "high"
    assert "reasoning" not in body


def test_router_body_without_reasoning_unchanged():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            }
        )
    )
    assert body["reasoning_effort"] == "high"
    assert "reasoning" not in body


def test_router_body_coerces_medium_reasoning():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "medium"},
            }
        )
    )
    efficient = base_proxy._base_efficient_model() or ""
    expected = providers._coerce_reasoning_effort(efficient, "medium")
    assert body.get("reasoning_effort") == expected
    assert "reasoning" not in body


def test_router_body_renames_max_completion_tokens():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "max_completion_tokens": 32000,
            }
        )
    )
    assert body.get("max_tokens") == 32000
    assert "max_completion_tokens" not in body

class _SessionRequest:
    """Minimal BaseChatRequest stand-in that exposes the attributes read here."""

    def __init__(self, messages=None, metadata=None, user=None):
        self.messages = messages
        self.metadata = metadata
        self.user = user
        self.stream = False

    def model_dump(self, *, exclude_none, exclude):
        return {"messages": self.messages}


def _looping_messages(*, repeats: int, tool: str = "bash") -> list[dict]:
    """A history where the same tool call and the same error repeat."""
    messages: list[dict] = [{"role": "user", "content": "fix the failing test"}]
    for _ in range(repeats):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": tool, "arguments": '{"cmd": "pytest -q"}'}}
                ],
            }
        )
        messages.append({"role": "tool", "content": "Traceback: AssertionError in test_x"})
    return messages


def test_no_escalation_on_a_healthy_session():
    """Distinct calls and one-off errors are normal work, not looping."""
    messages = [
        {"role": "user", "content": "add a flag"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "read", "arguments": "a"}}]},
        {"role": "tool", "content": "file contents"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "edit", "arguments": "b"}}]},
        {"role": "tool", "content": "error: patch did not apply"},
    ]
    assert base_proxy.failure_signals(messages) == 0
    assert base_proxy.escalation_suffix(_SessionRequest(messages=messages)) == ""


def test_repeated_identical_tool_call_is_a_failure_signal():
    """Three identical calls plus three identical errors are two signals."""
    assert base_proxy.failure_signals(_looping_messages(repeats=2)) == 0
    assert base_proxy.failure_signals(_looping_messages(repeats=3)) == 2


def test_escalation_salts_the_stickiness_key_only():
    """The salt must move the Switchyard session and leave cache affinity alone."""
    request = _SessionRequest(
        messages=_looping_messages(repeats=3), metadata={"session_id": "s-1"}
    )
    out = base_proxy.router_headers({}, request)
    assert out[base_proxy.SWITCHYARD_SESSION_HEADER] == "s-1#esc2"
    assert out[base_proxy.GROK_CONV_HEADER] == "s-1"


def test_escalation_is_deterministic_and_idempotent():
    """Same history -> same salt, so a tier cannot oscillate within a turn."""
    messages = _looping_messages(repeats=4)
    first = base_proxy.router_headers({}, _SessionRequest(messages=messages, user="u"))
    second = base_proxy.router_headers({}, _SessionRequest(messages=messages, user="u"))
    assert first[base_proxy.SWITCHYARD_SESSION_HEADER] == second[
        base_proxy.SWITCHYARD_SESSION_HEADER
    ]


def test_escalation_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MANTIS_BASE_ESCALATE_ON_FAILURE", "0")
    request = _SessionRequest(
        messages=_looping_messages(repeats=5), metadata={"session_id": "s-2"}
    )
    assert base_proxy.router_headers({}, request)[base_proxy.SWITCHYARD_SESSION_HEADER] == "s-2"


def test_escalation_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("MANTIS_BASE_ESCALATE_REPEATS", "5")
    assert base_proxy.failure_signals(_looping_messages(repeats=3)) == 0
    assert base_proxy.failure_signals(_looping_messages(repeats=5)) == 2


def test_failure_signals_tolerates_malformed_history():
    """The proxy must never 500 on a shape it did not expect."""
    assert base_proxy.failure_signals(None) == 0
    assert base_proxy.failure_signals("not a list") == 0
    assert base_proxy.failure_signals([None, 7, {"role": "assistant"}]) == 0
    assert base_proxy.failure_signals([{"role": "tool", "content": {"blocks": []}}]) == 0
