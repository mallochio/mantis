"""Unit tests for base_proxy session and header handling."""

from __future__ import annotations

import base_proxy
import httpx


def test_session_id_prefers_switchyard_header():
    assert base_proxy.session_id(
        {"x-switchyard-session-id": "switch", "x-route-session": "route"},
        type("R", (), {"metadata": None, "user": None}),
    ) == "switch"


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
