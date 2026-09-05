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


def test_escalation_salts_the_stickiness_key_only(monkeypatch):
    """The salt must move the Switchyard session and leave cache affinity alone."""
    monkeypatch.setenv("MANTIS_BASE_SALT_SESSION", "1")
    request = _SessionRequest(
        messages=_looping_messages(repeats=3), metadata={"session_id": "s-1"}
    )
    out = base_proxy.router_headers({}, request)
    assert out[base_proxy.SWITCHYARD_SESSION_HEADER] == "s-1#esc2"
    assert out[base_proxy.GROK_CONV_HEADER] == "s-1"


def test_escalation_preserves_session_by_default():
    """Default keeps prefix cache and still promotes via force-tier header."""
    request = _SessionRequest(
        messages=_looping_messages(repeats=3), metadata={"session_id": "s-1"}
    )
    out = base_proxy.router_headers({}, request)
    assert out[base_proxy.SWITCHYARD_SESSION_HEADER] == "s-1"
    assert out["x-switchyard-force-tier"] == "capable"


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


# -- Improvement 4: Complexity classifier ----------------------------------


def test_classify_complexity_simple():
    messages = [{"role": "user", "content": "What time is it?"}]
    assert base_proxy._classify_complexity(messages) == "simple"


def test_classify_complexity_empty():
    assert base_proxy._classify_complexity([]) == "simple"
    assert base_proxy._classify_complexity([{"role": "system", "content": "sys"}]) == "simple"


def test_classify_complexity_reasoning():
    messages = [
        {
            "role": "user",
            "content": "Please think through this step by step and reason about the tradeoffs.",
        }
    ]
    assert base_proxy._classify_complexity(messages) == "reasoning"


def test_classify_complexity_complex():
    messages = [
        {
            "role": "user",
            "content": (
                "Write a distributed algorithm for consensus. "
                "The system must handle concurrency correctly."
            ),
        }
    ]
    tier = base_proxy._classify_complexity(messages)
    assert tier in ("complex", "reasoning")


def test_classify_complexity_medium():
    messages = [
        {
            "role": "user",
            "content": "Can you analyze this code?\n```python\ndef foo(): pass\n```",
        }
    ]
    tier = base_proxy._classify_complexity(messages)
    assert tier in ("medium", "complex")


def test_classify_complexity_uses_last_user_message():
    messages = [
        {"role": "user", "content": "Please derive the proof step by step and reason carefully."},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Thanks!"},
    ]
    # "Thanks!" is simple despite the first message being complex
    assert base_proxy._classify_complexity(messages) == "simple"


def test_classify_complexity_content_list():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Prove that the algorithm terminates."},
            ],
        }
    ]
    tier = base_proxy._classify_complexity(messages)
    assert tier in ("medium", "complex", "reasoning")


def test_router_headers_include_complexity():
    body = type("R", (), {"metadata": {"session_id": "s"}, "user": None})
    headers = base_proxy.router_headers({}, body, complexity="reasoning")
    assert headers[base_proxy.SWITCHYARD_COMPLEXITY_HEADER] == "reasoning"
    assert headers["x-switchyard-force-tier"] == "capable"


def test_router_headers_complexity_simple_no_force_tier():
    body = type("R", (), {"metadata": {"session_id": "s"}, "user": None})
    headers = base_proxy.router_headers({}, body, complexity="simple")
    assert headers[base_proxy.SWITCHYARD_COMPLEXITY_HEADER] == "simple"
    assert "x-switchyard-force-tier" not in headers


def test_router_headers_no_complexity():
    body = type("R", (), {"metadata": {"session_id": "s"}, "user": None})
    headers = base_proxy.router_headers({}, body)
    assert base_proxy.SWITCHYARD_COMPLEXITY_HEADER not in headers


# -- Direct LiteLLM leg for non-Switchyard adapters ------------------------


def _write_catalog(
    tmp_path, efficient_provider, efficient_model, capable_model="openai/gpt-5.6-sol"
):
    adapter = "bedrock" if efficient_provider == "bedrock" else "openrouter"
    base_url = (
        "https://bedrock-runtime.eu-central-1.amazonaws.com"
        if adapter == "bedrock"
        else "https://openrouter.ai/api/v1"
    )
    credential_env = "AWS_ACCESS_KEY_ID" if adapter == "bedrock" else "OPENROUTER_API_KEY"
    path = tmp_path / "catalog.toml"
    path.write_text(
        "version = 1\n"
        f'[providers.{efficient_provider}]\n'
        f'adapter = "{adapter}"\n'
        f'base_url = "{base_url}"\n'
        f'credential_env = "{credential_env}"\n'
        'protocols = ["chat_completions"]\n'
        '[providers.openrouter]\n'
        'adapter = "openrouter"\n'
        'base_url = "https://openrouter.ai/api/v1"\n'
        'credential_env = "OPENROUTER_API_KEY"\n'
        'protocols = ["chat_completions"]\n'
        "[base]\n"
        'revision = "test-direct-leg"\n'
        'picker = "efficient_first"\n'
        "confidence_threshold = 0.5\n"
        "recent_turn_window = 3\n"
        "[base.targets.efficient]\n"
        f'provider = "{efficient_provider}"\n'
        'reasoning_effort = "high"\n'
        "max_tokens = 64000\n"
        f'upstream_model = "{efficient_model}"\n'
        "[base.targets.capable]\n"
        'provider = "openrouter"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 128000\n"
        f'upstream_model = "{capable_model}"\n'
    )
    return path


def _use_catalog(monkeypatch, path):
    monkeypatch.setenv("MANTIS_CATALOG_PATH", str(path))
    monkeypatch.delenv("AI_ROUTING_CONFIG", raising=False)
    base_proxy._load_base_route.cache_clear()
    base_proxy._base_route_family.cache_clear()


def _litellm_ok_response(content="hi"):
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message={"role": "assistant", "content": content},
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def test_direct_litellm_spec_selects_bedrock_efficient(tmp_path, monkeypatch):
    _use_catalog(
        monkeypatch, _write_catalog(tmp_path, "bedrock", "global.xai.grok-4.6")
    )
    spec = base_proxy._direct_litellm_spec()
    assert spec is not None
    assert spec.adapter == "bedrock"
    assert spec.model == "global.xai.grok-4.6"
    assert spec.credential_env == "AWS_ACCESS_KEY_ID"


def test_direct_litellm_spec_skips_switchyard_servable_adapters(tmp_path, monkeypatch):
    _use_catalog(monkeypatch, _write_catalog(tmp_path, "openrouter", "openai/gpt-4o"))
    assert base_proxy._direct_litellm_spec() is None


def test_forward_uses_direct_litellm_for_bedrock_efficient(tmp_path, monkeypatch):
    _use_catalog(
        monkeypatch, _write_catalog(tmp_path, "bedrock", "global.xai.grok-4.6")
    )
    monkeypatch.setattr(
        providers, "_litellm_completion", lambda **kwargs: _litellm_ok_response("grok-hi")
    )

    def client_factory():
        raise AssertionError("Switchyard must not be called for bedrock efficient")

    request = _SessionRequest(messages=[{"role": "user", "content": "hi"}])
    response, handed_off = base_proxy.forward(request, {}, "req-1", client_factory)
    assert handed_off is False
    assert response.status_code == 200
    import json

    body = json.loads(response.body.decode())
    assert body["model"] == "mantis/base"
    assert body["choices"][0]["message"]["content"] == "grok-hi"
    assert response.headers["x-route-model"] == "global.xai.grok-4.6"


def test_direct_litellm_prefers_target_budget_over_small_client_value(
    tmp_path, monkeypatch
):
    _use_catalog(
        monkeypatch, _write_catalog(tmp_path, "bedrock", "global.xai.grok-4.6")
    )
    seen = {}
    orig_kwargs = base_proxy._direct_litellm_kwargs

    def spy(body, resolved):
        kwargs = orig_kwargs(body, resolved)
        seen.update(kwargs)
        return kwargs

    monkeypatch.setattr(base_proxy, "_direct_litellm_kwargs", spy)
    monkeypatch.setattr(
        providers, "_litellm_completion", lambda **kwargs: _litellm_ok_response("ok")
    )

    def client_factory():
        raise AssertionError("Switchyard must not be called for bedrock efficient")

    request = _SessionRequest(messages=[{"role": "user", "content": "hi"}])
    base_proxy.forward(request, {}, "req-budget", client_factory)
    # The catalog target budget (64000) wins so reasoning models keep room
    # for thinking instead of returning empty content on small client budgets.
    assert seen["max_tokens"] == 64000


def test_forward_streams_direct_litellm_for_bedrock_efficient(tmp_path, monkeypatch):
    from types import SimpleNamespace

    _use_catalog(
        monkeypatch, _write_catalog(tmp_path, "bedrock", "global.xai.grok-4.6")
    )

    def fake_completion(**kwargs):
        assert kwargs.get("stream") is True
        yield SimpleNamespace(
            id="c1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="hel", reasoning_content=None),
                    finish_reason=None,
                )
            ],
        )
        yield SimpleNamespace(
            id="c1",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                type="function",
                                index=0,
                                function=SimpleNamespace(
                                    name="ipython", arguments='{"code": "1+1"}'
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

    monkeypatch.setattr(providers, "_litellm_completion", fake_completion)

    def client_factory():
        raise AssertionError("Switchyard must not be called for bedrock efficient")

    request = _SessionRequest(messages=[{"role": "user", "content": "hi"}])
    request.stream = True
    closed = []
    response, handed_off = base_proxy.forward(
        request, {}, "req-2", client_factory, on_close=lambda: closed.append(True)
    )
    assert handed_off is True
    import asyncio
    import json

    async def _collect():
        return b"".join([chunk async for chunk in response.body_iterator])

    payload = asyncio.run(_collect()).decode()
    assert 'data: [DONE]' in payload
    assert '"content": "hel"' in payload
    assert json.loads(payload.splitlines()[0].removeprefix("data: "))["model"] == "mantis/base"
    # Tool-call deltas must survive translation or the harness loop stalls.
    tool_frames = [
        json.loads(line.removeprefix("data: "))
        for line in payload.splitlines()
        if line.startswith("data:") and "tool_calls" in line
    ]
    assert tool_frames, payload
    call = tool_frames[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["id"] == "call-1"
    assert call["function"]["name"] == "ipython"
    assert json.loads(call["function"]["arguments"]) == {"code": "1+1"}
    assert closed == [True]


def test_forward_falls_back_to_switchyard_when_direct_fails(tmp_path, monkeypatch):
    _use_catalog(
        monkeypatch, _write_catalog(tmp_path, "bedrock", "global.xai.grok-4.6")
    )

    def boom(**kwargs):
        raise RuntimeError("bedrock down")

    monkeypatch.setattr(providers, "_litellm_completion", boom)

    def handler(_request):
        return httpx.Response(
            200,
            json={"id": "c", "choices": [{"message": {"content": "via-switchyard"}}]},
        )

    request = _SessionRequest(messages=[{"role": "user", "content": "hi"}])
    response, handed_off = base_proxy.forward(
        request, {}, "req-3", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert handed_off is False
    assert response.status_code == 200
    import json

    assert json.loads(response.body.decode())["choices"][0]["message"]["content"] == (
        "via-switchyard"
    )


def test_forward_rejects_non_chat_200_payload(tmp_path, monkeypatch):
    _use_catalog(monkeypatch, _write_catalog(tmp_path, "openrouter", "openai/gpt-4o"))

    def handler(_request):
        # Raw provider error passed through with 200: no "error" key, no choices.
        return httpx.Response(
            200,
            json={
                "Output": {"__type": "com.amazon.coral.service#UnknownOperationException"},
                "Version": "1.0",
            },
        )

    request = _SessionRequest(messages=[{"role": "user", "content": "hi"}])
    response, handed_off = base_proxy.forward(
        request, {}, "req-4", lambda: httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert handed_off is False
    assert response.status_code == 502
