"""Unit tests for the lean gateway's config/decision/session/cache/proxy helpers.

These tests exercise ``apps/gateway/lean/*`` directly and never import the
legacy ``server.py`` module.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest


def _ensure_lean_catalog_env():
    """Point ``lean.config`` at a policies-shaped catalog before it is imported.

    ``apps/gateway/tests/conftest.py`` already exports ``MANTIS_ROUTER_TARGETS_JSON``
    for the legacy (flat ``complexity_targets``) ``server.py`` schema. The lean
    gateway's ``lean.config._load_spec`` requires a ``policies``/``active_policy``
    wrapper, so we swap in a compatible payload only for the one-time import of
    ``lean.config``, then restore whatever was there before so the rest of the
    shared test suite (which may depend on the legacy shape) is unaffected.
    """
    if "lean.config" in sys.modules:
        return
    payload = {
        "version": 1,
        "providers": {
            "zen": {
                "base_url": "https://opencode.ai/zen/go/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions"],
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions", "responses"],
            },
        },
        "targets": {
            "cheap": {
                "provider": "zen",
                "upstream_model": "deepseek-v4-flash",
                "rank": 0,
                "protocols": ["chat_completions"],
                "fallbacks": ["middle"],
            },
            "middle": {
                "provider": "openrouter",
                "upstream_model": "openai/gpt-5.6-terra",
                "rank": 1,
                "protocols": ["chat_completions", "responses"],
                "fallbacks": ["expensive"],
            },
            "expensive": {
                "provider": "openrouter",
                "upstream_model": "openai/gpt-5.6-sol",
                "rank": 2,
                "protocols": ["chat_completions", "responses"],
                "fallbacks": ["middle"],
            },
        },
        "policies": {
            "default": {"complexity_targets": ["cheap", "cheap", "middle", "middle", "expensive"]},
        },
        "active_policy": "default",
    }
    os.environ.setdefault("MANTIS_ROUTER_TEST_CRED", "test-only")
    os.environ.setdefault("MANTIS_ROUTER_KEY", "sk-route-local")
    prev_targets = os.environ.get("MANTIS_ROUTER_TARGETS_JSON")
    prev_catalog = os.environ.pop("AI_ROUTING_CONFIG", None)
    os.environ["MANTIS_ROUTER_TARGETS_JSON"] = json.dumps(payload)
    try:
        import lean.config  # noqa: F401
    finally:
        if prev_targets is None:
            os.environ.pop("MANTIS_ROUTER_TARGETS_JSON", None)
        else:
            os.environ["MANTIS_ROUTER_TARGETS_JSON"] = prev_targets
        if prev_catalog is not None:
            os.environ["AI_ROUTING_CONFIG"] = prev_catalog


_ensure_lean_catalog_env()

import lean.cache as cache  # noqa: E402
import lean.config as config  # noqa: E402
import lean.decision as decision  # noqa: E402
import lean.proxy as proxy  # noqa: E402
import lean.session as session  # noqa: E402

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# lean.config
# ---------------------------------------------------------------------------


def test_target_for_complexity_maps_to_supra_targets():
    for c in range(1, 6):
        assert config._target_for_complexity(c) == config.SUPRA_TARGETS[c - 1]


def test_target_for_complexity_out_of_range_uses_invalid_target():
    assert config._target_for_complexity(0) == config.SUPRA_INVALID_TARGET
    assert config._target_for_complexity(6) == config.SUPRA_INVALID_TARGET
    assert config._target_for_complexity(None) == config.SUPRA_INVALID_TARGET


def test_api_routes_respects_protocols_and_fallback_for_chat():
    routes = config._api_routes("cheap", "chat")
    assert routes[0] == "cheap"
    assert "middle" in routes  # cheap's declared fallback


def test_api_routes_upgrades_and_falls_back_for_unsupported_protocol():
    # "cheap" only supports chat_completions in the fixture catalog, so a
    # responses request must be upgraded to a compatible target.
    routes = config._api_routes("cheap", "responses")
    assert routes[0] != "cheap"
    assert config._supports_api(config._backend_for(routes[0]), "responses")


def test_chat_tier_and_responses_tier_upgrade_incompatible_decision():
    assert config._chat_tier("cheap") == "cheap"
    responses_tier = config._responses_tier("cheap")
    assert responses_tier != "cheap"
    assert config._supports_api(config._backend_for(responses_tier), "responses")


def test_backend_for_returns_backend_dict_from_backends():
    backend = config._backend_for("cheap")
    assert backend is config.BACKENDS["cheap"]
    assert backend["target"] == "cheap"


def test_backend_for_unknown_target_falls_back_to_safe_target():
    backend = config._backend_for("does-not-exist")
    assert backend["target"] == config._safe_target()


# ---------------------------------------------------------------------------
# lean.decision
# ---------------------------------------------------------------------------


def test_prompt_hash_is_stable_and_deterministic():
    h1 = decision._prompt_hash("hello world")
    h2 = decision._prompt_hash("hello world")
    h3 = decision._prompt_hash("something else")
    assert h1 == h2
    assert h1 != h3
    assert isinstance(h1, str) and len(h1) == 24


@pytest.fixture
def decision_store(tmp_path, monkeypatch):
    monkeypatch.setattr(decision, "DECISION_STORE_PATH", tmp_path / "decisions.db")
    decision._init_db()
    decision._store_load()
    yield
    decision._decision_cache.clear()


def test_store_note_pins_cheap_after_pin_cheap_after_successes(decision_store, monkeypatch):
    monkeypatch.setattr(decision, "PIN_CHEAP_AFTER", 2)
    ph = decision._prompt_hash("repeat this cheap prompt")
    decision._store_note(ph, "cheap", ok=True)
    assert decision._store_pinned(ph) is None
    decision._store_note(ph, "cheap", ok=True)
    entry = decision._store_pinned(ph)
    assert entry is not None
    assert entry["decision"] == "cheap"
    assert entry.get("pin_until", 0) > time.time()


def test_store_note_pins_expensive_after_lowest_tier_fails(decision_store, monkeypatch):
    monkeypatch.setattr(decision, "PIN_EXPENSIVE_AFTER", 2)
    ph = decision._prompt_hash("refuse this prompt please")
    decision._store_note(ph, "cheap", ok=False)
    assert decision._store_pinned(ph) is None
    decision._store_note(ph, "cheap", ok=False)
    entry = decision._store_pinned(ph)
    assert entry is not None
    # "cheap" is the lowest-rank chat target, so a repeated failure pins the
    # next compatible target up (its fallback / next rank).
    assert entry["decision"] != "cheap"
    assert entry["decision"] in config.BACKENDS


def test_decide_returns_pinned_decision_without_calling_supra(decision_store, monkeypatch):
    monkeypatch.setattr(decision, "PIN_CHEAP_AFTER", 1)
    ph = decision._prompt_hash("pin me please")
    decision._store_note(ph, "cheap", ok=True)
    assert decision._store_pinned(ph) is not None

    def boom(prompt):
        raise AssertionError("_supra_complexity should not be called for a pinned prompt")

    monkeypatch.setattr(decision, "_supra_complexity", boom)
    result = decision._decide("pin me please")
    assert result[0] == "cheap"


def test_decide_calls_supra_for_unknown_prompt(decision_store, monkeypatch):
    calls = []

    def fake_supra(prompt):
        calls.append(prompt)
        return 3, 15

    monkeypatch.setattr(decision, "_supra_complexity", fake_supra)
    target, score, complexity, ms = decision._decide("a brand new prompt never seen before")
    assert calls == ["a brand new prompt never seen before"]
    assert target == config._target_for_complexity(3)
    assert complexity == 3
    assert ms == 15
    assert score is None


# ---------------------------------------------------------------------------
# lean.session
# ---------------------------------------------------------------------------


class _FakeHeaders(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


class _FakeRequest:
    def __init__(self, headers=None):
        self.headers = _FakeHeaders({k.lower(): v for k, v in (headers or {}).items()})


def test_session_id_is_opaque_hmac_digest():
    body = {"metadata": {"session_id": "my-secret-session"}}
    sid, src = session._session_id(body, _FakeRequest())
    assert sid is not None
    assert sid != "my-secret-session"
    assert src == "metadata"
    assert len(sid) == 24
    # Deterministic for identical input.
    sid2, _ = session._session_id(body, _FakeRequest())
    assert sid == sid2
    # A different raw value yields a different id.
    other, _ = session._session_id({"metadata": {"session_id": "other-session"}}, _FakeRequest())
    assert other != sid


def test_session_id_prefers_header_over_metadata():
    body = {"metadata": {"session_id": "from-metadata"}}
    req = _FakeRequest({"x-route-session": "from-header"})
    sid, src = session._session_id(body, req)
    assert src == "header"
    header_sid, _ = session._session_id({}, req)
    assert sid == header_sid


@pytest.fixture(autouse=True)
def clear_session_cache():
    session._session_cache.clear()
    yield
    session._session_cache.clear()


def test_session_route_ratchets_and_does_not_immediately_downgrade():
    sid = "sess-ratchet"
    session._session_note(sid, "expensive", 4)
    decision_, reason = session._session_route(sid, "a short cheap prompt", "cheap", 1)
    assert decision_ == "expensive"
    assert reason == "downgrade_hysteresis"


def test_session_route_new_task_allows_downgrade():
    sid = "sess-new-task"
    session._session_note(sid, "expensive", 4)
    decision_, reason = session._session_route(sid, "a short cheap prompt", "cheap", 1, new_task=True)
    assert decision_ == "cheap"
    assert reason == "new_task"


# ---------------------------------------------------------------------------
# lean.cache
# ---------------------------------------------------------------------------


def test_cache_key_excludes_tools_and_related_fields():
    base = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    assert cache._cache_key(base, "idem-1") is not None
    assert cache._cache_key({**base, "tools": [{"type": "function"}]}, "idem-1") is None
    assert cache._cache_key({**base, "functions": [{"name": "f"}]}, "idem-1") is None
    assert cache._cache_key({**base, "function_call": "auto"}, "idem-1") is None
    tool_msg = {**base, "messages": [*base["messages"], {"role": "tool", "content": "x", "tool_call_id": "1"}]}
    assert cache._cache_key(tool_msg, "idem-1") is None


def test_cache_key_requires_idempotency_key():
    base = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    assert cache._cache_key(base, None) is None
    assert cache._cache_key(base, "x" * 201) is None


def test_cache_key_changes_with_stream_flag():
    base = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    k1 = cache._cache_key(base, "idem-1")
    k2 = cache._cache_key({**base, "stream": True}, "idem-1")
    assert k1 != k2


@pytest.fixture(autouse=True)
def clear_resp_cache():
    cache._resp_cache.clear()
    cache._inflight.clear()
    yield
    cache._resp_cache.clear()
    cache._inflight.clear()


def test_cache_put_then_get_replays_stored_response():
    body = {"model": "auto", "messages": [{"role": "user", "content": "hi"}]}
    key = cache._cache_key(body, "idem-put")
    content = b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}'
    stored = cache._cache_put(key, body, content, {"x-a": "1"}, "application/json", 200)
    assert stored is not None
    got = cache._cache_get(key)
    assert got == (content, 200, "application/json", {"x-a": "1"})


def test_cache_get_missing_key_returns_none():
    assert cache._cache_get("nonexistent-key") is None
    assert cache._cache_get(None) is None


async def test_claim_and_finish_inflight_coalesce_identical_requests():
    key = "inflight-key"
    leader, fut1 = await cache._claim_inflight(key)
    assert leader is True
    follower, fut2 = await cache._claim_inflight(key)
    assert follower is False
    assert fut2 is fut1

    async def resolve():
        await asyncio.sleep(0)
        await cache._finish_inflight(key, fut1, "the-result")

    asyncio.create_task(resolve())
    result = await asyncio.wait_for(fut2, timeout=1)
    assert result == "the-result"
    # Key should be released for a new leader after completion.
    leader_again, _ = await cache._claim_inflight(key)
    assert leader_again is True


# ---------------------------------------------------------------------------
# lean.proxy
# ---------------------------------------------------------------------------


def test_is_refusal_detects_bifrost_stream_error_chunk():
    # Any Bifrost-injected stream error (status_code >= 400) is treated as a
    # refusal so the gateway can escalate to the next tier.
    assert proxy._is_refusal(200, {"status_code": 500, "is_bifrost_error": True}) is True
    assert proxy._is_refusal(200, {"status_code": 429, "is_bifrost_error": False}) is True
    assert proxy._is_refusal(200, {}) is False


def test_is_empty_completion_true_for_blank_message():
    assert proxy._is_empty_completion({"choices": [{"message": {"content": ""}}]}) is True
    assert proxy._is_empty_completion({"choices": [{"message": {"content": "   "}}]}) is True
    assert proxy._is_empty_completion({"choices": []}) is True


def test_is_empty_completion_false_when_content_present():
    assert proxy._is_empty_completion({"choices": [{"message": {"content": "hello"}}]}) is False
    assert (
        proxy._is_empty_completion({"choices": [{"message": {"tool_calls": [{"id": "1"}]}}]})
        is False
    )
