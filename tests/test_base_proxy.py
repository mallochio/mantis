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
    assert body["reasoning_effort"] == "low"
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
    assert body["reasoning_effort"] == expected
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
