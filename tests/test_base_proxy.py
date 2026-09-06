"""Unit tests for the thin mantis/base forwarder.

Switchyard owns tier decisions. These tests pin the forwarder contract:
stable session identity, catalog-supreme body hygiene, and pass-through
forwarding with mapped response headers.
"""

from __future__ import annotations

import json

import base_proxy
import httpx
import providers


def _request(
    messages=None,
    *,
    metadata=None,
    user=None,
    stream=False,
    extra=None,
):
    class Request:
        def model_dump(self, *, exclude_none, exclude):
            body = {"messages": messages if messages is not None else []}
            body.update(extra or {})
            return {k: v for k, v in body.items() if k not in exclude}

    request = Request()
    request.messages = messages if messages is not None else []
    request.tools = None
    request.metadata = metadata
    request.user = user
    request.stream = stream
    return request


def _make_request(data: dict) -> object:
    class Request:
        def model_dump(self, *, exclude_none, exclude):
            excluded = {*exclude}
            return {k: v for k, v in data.items() if k not in excluded}

    return Request()


# -- Session identity --------------------------------------------------------


def test_session_id_prefers_switchyard_header():
    request = _request(user=None)
    assert (
        base_proxy.session_id(
            {"x-switchyard-session-id": "switch", "x-route-session": "route"},
            request,
        )
        == "switch"
    )


def test_session_id_falls_back_to_metadata():
    assert base_proxy.session_id({}, _request(metadata={"session_id": "meta"})) == "meta"


def test_session_id_falls_back_to_user():
    assert base_proxy.session_id({}, _request(user="user")) == "user"


def test_session_id_synthesizes_stable_id_without_harness_session():
    messages = [{"role": "user", "content": "fix the bug"}]
    first = base_proxy.session_id({}, _request(messages=messages))
    second = base_proxy.session_id({}, _request(messages=[dict(messages[0])]))
    assert first is not None and first == second


def test_session_id_accepts_opencode_header():
    messages = [{"role": "user", "content": "fix the bug"}]
    session = base_proxy.session_id({"x-opencode-session": "opencode"}, _request(messages))
    assert session is not None and session.startswith("opencode:")


def test_shared_harness_session_is_scoped_per_conversation():
    messages = [{"role": "user", "content": "fix the bug"}]
    other = [{"role": "user", "content": "write docs"}]
    first = base_proxy.session_id({"x-mantis-session-id": "prime-agent"}, _request(messages))
    second = base_proxy.session_id(
        {"x-mantis-session-id": "prime-agent"}, _request(messages=other)
    )
    assert first != second
    assert first.startswith("prime-agent:")


# -- Headers: auth + session only, never tier directives -----------------------


def test_router_headers_carry_session_and_auth(monkeypatch):
    monkeypatch.setenv("MANTIS_ROUTER_KEY", "gateway-key")
    headers = base_proxy.router_headers({}, _request(metadata={"session_id": "s-1"}))
    assert headers["x-switchyard-session-id"] == "s-1"
    assert headers["Authorization"] == "Bearer gateway-key"
    assert "x-switchyard-force-tier" not in headers
    assert "x-switchyard-complexity" not in headers


def test_router_headers_omit_session_without_identity(monkeypatch):
    monkeypatch.delenv("MANTIS_ROUTER_KEY", raising=False)
    assert base_proxy.router_headers({}, _request()) == {}


# -- Body hygiene: catalog supremacy + stable prefix ---------------------------


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


def test_router_body_drops_harness_reasoning_object():
    """The catalog tier governs effort; harness thinking levels are stripped."""
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
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "user" not in body
    assert "metadata" not in body


def test_router_body_drops_harness_reasoning_effort():
    body = base_proxy.router_body(
        _make_request(
            {
                "model": "mantis/base",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "low"},
                "reasoning_effort": "max",
            }
        )
    )
    assert "reasoning" not in body
    assert "reasoning_effort" not in body


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


def test_router_body_prefix_stable_across_turns():
    """Turn two extends turn one's prefix byte-for-byte for cache reuse."""

    def outbound(messages, extra=None):
        return base_proxy.router_body(_request(messages=messages, extra=extra))

    turn1 = [{"role": "user", "content": "hi"}]
    turn2 = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "thanks"},
    ]
    # A harness thinking level on either turn must not perturb the prefix.
    first = outbound([dict(m) for m in turn1], extra={"reasoning_effort": "max"})
    second = outbound([dict(m) for m in turn2], extra={"reasoning": {"effort": "low"}})
    assert second["messages"][: len(first["messages"])] == first["messages"]
    assert "reasoning_effort" not in second
    assert "reasoning" not in second


# -- Router errors ---------------------------------------------------------------


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
    assert "Service temporarily unavailable" in result.body.decode()


def test_router_error_parses_sse_error_stream():
    response = httpx.Response(
        500,
        text='data: {"error": {"message": "boom"}}\n\ndata: [DONE]\n\n',
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 500


def test_router_error_coerces_string_code():
    response = httpx.Response(
        200,
        json={"error": {"message": "upstream failed", "code": "503"}},
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 503


def test_router_error_ignores_non_http_code():
    response = httpx.Response(
        502,
        json={"error": {"message": "upstream failed", "code": "overloaded"}},
        request=httpx.Request("POST", "http://example.com"),
    )
    result = base_proxy.router_error(response)
    assert result.status_code == 502


# -- Forward: Switchyard is the only hop -----------------------------------------


def _switchyard_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_forward_serves_simple_turn_through_switchyard():
    def handler(request):
        assert request.headers["x-switchyard-session-id"] == "sess-1"
        payload = json.loads(request.content.decode())
        assert "reasoning" not in payload
        assert "reasoning_effort" not in payload
        return httpx.Response(
            200,
            json={"id": "c", "choices": [{"message": {"content": "glm-hi"}}]},
            headers={
                base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "zai-org/GLM-5.3",
                base_proxy.SWITCHYARD_SESSION_HEADER: "sess-1",
            },
        )

    request = _request(
        [{"role": "user", "content": "hi"}], metadata={"session_id": "sess-1"}
    )
    response, handed_off = base_proxy.forward(
        request, {}, "req-1", lambda: _switchyard_client(handler)
    )
    assert handed_off is False
    assert json.loads(response.body.decode())["choices"][0]["message"]["content"] == "glm-hi"
    assert response.headers["x-route-model"] == "zai-org/GLM-5.3"
    assert response.headers[base_proxy.SWITCHYARD_SESSION_HEADER] == "sess-1"


def test_forward_strips_harness_reasoning_before_switchyard():
    """A harness thinking level must not reach the router body."""
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={"id": "c", "choices": [{"message": {"content": "ok"}}]},
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "zai-org/GLM-5.3"},
        )

    request = _request(
        [{"role": "user", "content": "hi"}],
        metadata={"session_id": "sess-2"},
        extra={"reasoning_effort": "max", "reasoning": {"effort": "max"}},
    )
    response, _ = base_proxy.forward(request, {}, "req-2", lambda: _switchyard_client(handler))
    assert response.headers["x-route-model"] == "zai-org/GLM-5.3"
    assert "reasoning" not in seen
    assert "reasoning_effort" not in seen


def test_forward_rejects_non_chat_200_payload():
    def handler(_request):
        return httpx.Response(200, json={"id": "c"})

    response, handed_off = base_proxy.forward(
        _request([{"role": "user", "content": "hi"}]),
        {},
        "req-3",
        lambda: _switchyard_client(handler),
    )
    assert handed_off is False
    assert response.status_code == 502


def test_forward_streams_through_switchyard():
    def handler(_request):
        return httpx.Response(
            200,
            content=b'data: {"choices": []}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    request = _request([{"role": "user", "content": "hi"}], stream=True)
    response, handed_off = base_proxy.forward(
        request, {}, "req-4", lambda: _switchyard_client(handler)
    )
    assert handed_off is True
    assert response.media_type == "text/event-stream"


# -- Cache families ------------------------------------------------------------------


def test_grok_has_no_explicit_cache_family():
    assert providers._model_cache_family("global.xai.grok-4.6") is None
    assert providers._model_cache_family("bedrock-openai/global.xai.grok-4.6") is None


def test_base_route_families_empty_for_grok_glm_route(monkeypatch):
    from types import SimpleNamespace

    route = SimpleNamespace(
        efficient=SimpleNamespace(upstream_model="zai-org/GLM-5.3"),
        capable=SimpleNamespace(upstream_model="global.xai.grok-4.6"),
    )
    monkeypatch.setattr(base_proxy, "_load_base_route", lambda: route)
    assert base_proxy._base_route_families() == frozenset()


def test_apply_base_cache_markers_leaves_grok_glm_messages_untouched(monkeypatch):
    from types import SimpleNamespace

    route = SimpleNamespace(
        efficient=SimpleNamespace(upstream_model="zai-org/GLM-5.3"),
        capable=SimpleNamespace(upstream_model="global.xai.grok-4.6"),
    )
    monkeypatch.setattr(base_proxy, "_load_base_route", lambda: route)
    body = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    out = base_proxy._apply_base_cache_markers(body)
    assert out["messages"] == body["messages"]


def test_shipped_catalog_routes_glm_efficient_kimi_chat_capable(monkeypatch):
    route = base_proxy._load_base_route()
    assert route is not None
    assert route.efficient.upstream_model == "zai-org/GLM-5.3"
    assert route.capable.upstream_model == "moonshotai/Kimi-K3"
    assert route.capable.provider == "modal.kimi-k3"
    assert route.capable.wire_format == "openai_chat"
