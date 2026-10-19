"""OpenRouter cache optimization: block markers and sticky sessions.

Anthropic models cache nothing on OpenRouter unless a request opts in with
``cache_control`` content blocks, and provider stickiness only starts at
the first request when a ``session_id`` rides the body. These tests pin
both behaviors across the LiteLLM worker legs and the Switchyard base leg.
"""

from __future__ import annotations

import json

import base_proxy
import httpx
import model_catalog
import providers


def _setup_catalog(monkeypatch):
    cat = model_catalog.load_mantis_catalog("config/catalog.toml")
    env = model_catalog.render_mantis_environment(cat)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    keys = dict.fromkeys(cat.bindings.providers, "test-key")
    monkeypatch.setenv("MANTIS_PROVIDER_KEYS", json.dumps(keys))
    for binding in cat.bindings.providers.values():
        monkeypatch.setenv(binding.credential_env, "test-key")
    return cat


def _request(messages, *, metadata=None):
    class Request:
        def model_dump(self, *, exclude_none, exclude):
            body = {"messages": messages}
            if metadata is not None:
                body["metadata"] = metadata
            return {k: v for k, v in body.items() if k not in exclude}

    request = Request()
    request.messages = messages
    request.tools = None
    request.metadata = metadata
    request.user = None
    request.stream = False
    return request


def _switchyard_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat_messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]


def test_litellm_claude_on_openrouter_gets_block_markers(monkeypatch):
    """Fusion/Ultra legs riding an openrouter claude spec need per-block markers."""
    _setup_catalog(monkeypatch)
    resolved = providers._resolve_model_spec("claude-opus-5")
    assert resolved.binding == "openrouter"
    kwargs = providers._litellm_kwargs(
        resolved, _chat_messages(), 100, 0.7, None, None, None, {}
    )
    dumped = json.dumps(kwargs["messages"])
    assert "cache_control" in dumped
    assert "prompt_cache_breakpoint" not in dumped


def test_litellm_openai_slot_keeps_breakpoint_dialect(monkeypatch):
    _setup_catalog(monkeypatch)
    resolved = providers._resolve_model_spec("gpt-5_6-sol")
    kwargs = providers._litellm_kwargs(
        resolved, _chat_messages(), 100, 0.7, None, None, None, {}
    )
    dumped = json.dumps(kwargs["messages"])
    assert "prompt_cache_breakpoint" in dumped
    assert "cache_control" not in dumped


def test_forward_body_carries_sticky_session():
    """The stable session id must ride the body so OpenRouter sticks from turn one."""

    def handler(request):
        payload = json.loads(request.content.decode())
        assert payload["session_id"] == "sticky-1"
        return httpx.Response(200, json={"id": "c", "choices": [{"message": {"content": "ok"}}]})

    base_proxy.forward(
        _request([{"role": "user", "content": "hi"}], metadata={"session_id": "sticky-1"}),
        {},
        "req-1",
        lambda: _switchyard_client(handler),
    )


def test_forward_body_without_session_has_no_sticky_key():
    def handler(request):
        payload = json.loads(request.content.decode())
        assert "session_id" not in payload
        return httpx.Response(200, json={"id": "c", "choices": [{"message": {"content": "ok"}}]})

    base_proxy.forward(
        _request([{"role": "assistant", "content": "no user turn to hash"}]),
        {},
        "req-1",
        lambda: _switchyard_client(handler),
    )


def test_sticky_session_is_capped_at_upstream_limit():
    body = base_proxy._attach_sticky_session({}, "s" * 300)
    assert body["session_id"] == "s" * 256
    assert base_proxy._attach_sticky_session({}, None) == {}
