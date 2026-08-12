# Catalog-backed Supra target routing tests; all upstreams are MockTransport.

import asyncio
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

import server

AUTH = {"Authorization": "Bearer sk-route-local"}


def catalog_text(*, revision="catalog-v1", bad_key=False):
    key = 'key = "not-allowed"\n' if bad_key else ""
    return f"""
version = 1

[providers.direct]
adapter = "openai-compatible"
base_url = "https://upstream.test/v1"
credential_env = "ROUTER_CATALOG_TEST_KEY"
developer_role = "native"

[routellm]
active_policy = "coding"
revision = "{revision}"

[routellm.targets.low]
provider = "direct"
upstream_model = "vendor/low"
reasoning_effort = "none"
protocols = ["chat_completions"]
fallbacks = ["mid"]
rank = 0

[routellm.targets.mid]
provider = "direct"
upstream_model = "vendor/mid"
reasoning_effort = "medium"
protocols = ["chat_completions"]
fallbacks = ["low"]
rank = 1

[routellm.targets.work]
provider = "direct"
upstream_model = "vendor/work"
reasoning_effort = "high"
protocols = ["chat_completions"]
fallbacks = ["safe"]
rank = 2

[routellm.targets.responses]
provider = "direct"
upstream_model = "openai/responses"
reasoning_effort = "max"
force_reasoning_effort = true
protocols = ["chat_completions", "responses"]
fallbacks = ["safe"]
rank = 3

[routellm.targets.safe]
provider = "direct"
upstream_model = "openai/safe"
reasoning_effort = "xhigh"
protocols = ["chat_completions", "responses"]
fallbacks = ["responses"]
rank = 4
{key}
[routellm.policies.coding]
complexity_targets = ["low", "mid", "work", "responses", "safe"]
invalid_complexity_target = "safe"
"""


def import_catalog_server(tmp_path, monkeypatch, *, text=None, revision="catalog-v1"):
    path = tmp_path / "catalog.toml"
    path.write_text(text or catalog_text(revision=revision))
    monkeypatch.setenv("AI_ROUTING_CONFIG", str(path))
    monkeypatch.setenv("ROUTER_CATALOG_TEST_KEY", "test-only-value")
    return importlib.reload(server)


@pytest.fixture
def catalog_server(tmp_path, monkeypatch):
    value = import_catalog_server(tmp_path, monkeypatch)
    yield value
    monkeypatch.undo()
    importlib.reload(server)


def test_catalog_exact_mapping_invalid_reason_and_bounded_cycles(catalog_server, monkeypatch):
    for level, expected in enumerate(("low", "mid", "work", "responses", "safe"), 1):
        monkeypatch.setattr(catalog_server, "_supra_complexity", lambda _prompt, level=level: (level, 1))
        assert catalog_server._decide_uncached("task")[0] == expected
    for invalid in (0, 6, -1):
        monkeypatch.setattr(catalog_server, "_supra_complexity", lambda _prompt, invalid=invalid: (invalid, 1))
        decision, _, complexity, _ = catalog_server._decide_uncached("task")
        assert (decision, complexity) == ("safe", invalid)
        assert catalog_server._supra_reason(complexity) == "supra_invalid_complexity"
    assert catalog_server._failover_routes("responses") == ("responses", "safe")
    assert catalog_server._failover_routes("safe") == ("safe", "responses")


def test_catalog_direct_binding_protocols_roles_and_reasoning(catalog_server):
    assert catalog_server._backend_for("work")["model"] == "vendor/work"
    assert catalog_server._responses_routes("work") == ("responses", "safe")
    assert catalog_server._responses_routes("responses") == ("responses", "safe")
    native_messages = [{"role": "developer", "content": "keep role"}]
    chat = catalog_server._build_outgoing_body(
        {"model": "auto", "messages": native_messages}, catalog_server.BACKENDS["responses"])
    assert chat["messages"] == native_messages
    response = catalog_server._build_responses_body(
        {"model": "auto", "input": "x", "reasoning": {"effort": "low", "summary": "auto"}},
        catalog_server.BACKENDS["responses"],
    )
    assert response["reasoning"] == {"effort": "max", "summary": "auto"}


@pytest.mark.anyio
async def test_catalog_target_reaches_mock_upstream_and_headers(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("work", None, 3, 1))
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app), base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}],
        })
    assert response.status_code == 200
    assert seen["url"] == "https://upstream.test/v1/chat/completions"
    assert seen["body"]["model"] == "vendor/work"
    assert response.headers["x-route-decision"] == "work"
    assert response.headers["x-route-target"] == "work"
    assert response.headers["x-route-target-revision"].startswith("catalog-v1-")
    await mock.aclose()


def test_pins_are_revision_namespaced_and_generic_escalation(catalog_server, monkeypatch):
    monkeypatch.setattr(catalog_server, "PIN_EXPENSIVE_AFTER", 2)
    catalog_server._store_note("prompt", "low", ok=False)
    catalog_server._store_note("prompt", "low", ok=False)
    assert catalog_server._store_pinned("prompt")["decision"] == "mid"
    catalog_server._decision_store["old"] = {
        "decision": "safe", "pin_until": time.time() + 60, "target_revision": "old-revision",
    }
    assert catalog_server._store_pinned("old") is None


def test_catalog_rejects_literal_keys_and_missing_credentials(tmp_path):
    path = tmp_path / "catalog.toml"
    path.write_text(catalog_text(bad_key=True))
    env = os.environ.copy()
    env.update({"AI_ROUTING_CONFIG": str(path), "ROUTER_CATALOG_TEST_KEY": "test-only-value"})
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    bad = subprocess.run([sys.executable, "-c", "import server"], env=env, capture_output=True, text=True)
    assert bad.returncode != 0
    assert "must use credential_env, not key" in bad.stderr
    assert "not-allowed" not in bad.stderr

    path.write_text(catalog_text())
    env.pop("ROUTER_CATALOG_TEST_KEY", None)
    missing = subprocess.run([sys.executable, "-c", "import server"], env=env, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "credential environment variable ROUTER_CATALOG_TEST_KEY is not set" in missing.stderr



def _catalog_subprocess(tmp_path, text, *, json_source=None, raw_json=None, credential=True,
                        catalog_path=True, catalog_bytes=None):
    """Import a source in a clean process without exposing its values."""
    path = tmp_path / "catalog.toml"
    if catalog_path:
        if catalog_bytes is None:
            path.write_text(text)
        else:
            path.write_bytes(catalog_bytes)
    else:
        path = tmp_path / "does-not-exist.toml"
    env = os.environ.copy()
    env["AI_ROUTING_CONFIG"] = str(path)
    if credential:
        env["ROUTER_CATALOG_TEST_KEY"] = "test-only-value"
    else:
        env.pop("ROUTER_CATALOG_TEST_KEY", None)
    if raw_json is not None:
        env["ROUTELLM_TARGETS_JSON"] = raw_json
    elif json_source is None:
        env.pop("ROUTELLM_TARGETS_JSON", None)
    else:
        env["ROUTELLM_TARGETS_JSON"] = json.dumps(json_source)
    env["PYTHONPATH"] = str(Path(__file__).parents[1])
    return subprocess.run([sys.executable, "-c", "import server"], env=env,
                          capture_output=True, text=True)


def test_catalog_requires_version_and_rejects_closed_schema_and_bad_port(tmp_path):
    missing_version = catalog_text().replace("version = 1\n\n", "", 1)
    result = _catalog_subprocess(tmp_path, missing_version)
    assert result.returncode != 0 and "catalog version must be 1" in result.stderr

    typo = catalog_text().replace('rank = 0', 'rank = 0\nrank_typo = 1', 1)
    result = _catalog_subprocess(tmp_path, typo)
    assert result.returncode != 0 and "unexpected field rank_typo" in result.stderr

    bad_port = catalog_text().replace("https://upstream.test/v1", "https://upstream.test:bad/v1")
    result = _catalog_subprocess(tmp_path, bad_port)
    assert result.returncode != 0 and "base_url must be a valid URL" in result.stderr


def test_json_source_is_atomic_and_validated(tmp_path):
    # The catalog has a provider called direct, but JSON has no provider table.
    # A JSON target referring to it must fail instead of composing sources.
    atomic = {
        "version": 1,
        "targets": {
            "only": {
                "provider": "direct", "upstream_model": "vendor/only",
                "reasoning_effort": "none", "protocols": ["chat_completions"],
                "rank": 0,
            },
        },
        "complexity_targets": ["only"] * 5,
    }
    result = _catalog_subprocess(tmp_path, catalog_text(), json_source=atomic)
    assert result.returncode != 0 and "provider is unknown" in result.stderr

    malformed = {"version": 2, "targets": {}, "complexity_targets": ["only"] * 5}
    result = _catalog_subprocess(tmp_path, catalog_text(), json_source=malformed)
    assert result.returncode != 0 and "ROUTELLM_TARGETS_JSON.version must be 1" in result.stderr

    bool_version = {"version": True, "targets": {}, "complexity_targets": ["only"] * 5}
    result = _catalog_subprocess(tmp_path, catalog_text(), json_source=bool_version)
    assert result.returncode != 0 and "ROUTELLM_TARGETS_JSON.version must be 1" in result.stderr

    # A valid direct JSON source is independent of an absent catalog file.
    direct = {
        "version": 1,
        "targets": {
            "only": {
                "adapter": "openai-compatible", "base_url": "https://upstream.test/v1",
                "credential_env": "ROUTER_CATALOG_TEST_KEY", "upstream_model": "vendor/only",
                "reasoning_effort": "none", "protocols": ["chat_completions"], "rank": 0,
            },
        },
        "complexity_targets": ["only"] * 5,
    }
    result = _catalog_subprocess(tmp_path, "", json_source=direct, catalog_path=False)
    assert result.returncode == 0


def test_shared_catalog_allows_anthropic_but_routellm_targets_reject_it(tmp_path):
    shared_anthropic_provider = """
[providers.anthropic-shared]
adapter = "anthropic"
base_url = "https://anthropic.test"
credential_env = "ROUTER_CATALOG_TEST_KEY"
protocols = ["anthropic_messages"]
"""
    shared_catalog = catalog_text().replace("[routellm]", shared_anthropic_provider + "\n[routellm]")
    assert _catalog_subprocess(tmp_path, shared_catalog).returncode == 0

    invalid_target = shared_catalog.replace(
        'protocols = ["chat_completions"]',
        'protocols = ["anthropic_messages"]',
        1,
    )
    result = _catalog_subprocess(tmp_path, invalid_target)
    assert result.returncode != 0
    assert "cannot declare anthropic_messages for a RouteLLM target" in result.stderr


def test_adapter_capabilities_and_fingerprint_cover_adapter_and_policy(catalog_server):
    target = catalog_server.BACKENDS["responses"]
    assert target["adapter"] == "openai-compatible"
    assert not catalog_server._supports_api({**target, "adapter": "opencode-go"}, "responses")
    assert catalog_server._supports_api({**target, "adapter": "openai-compatible"}, "responses")

    fingerprint = catalog_server.TARGET_CONFIG_FINGERPRINT
    changed_adapter = {name: dict(value) for name, value in catalog_server.BACKENDS.items()}
    changed_adapter["responses"]["adapter"] = "modal"
    assert catalog_server._target_config_fingerprint(
        "catalog", changed_adapter, catalog_server.SUPRA_TARGETS, catalog_server.SUPRA_INVALID_TARGET,
    ) != fingerprint
    assert catalog_server._target_config_fingerprint(
        "catalog", catalog_server.BACKENDS, catalog_server.SUPRA_TARGETS, "responses",
    ) != fingerprint
    assert catalog_server.TARGET_CONFIG_REVISION.startswith("catalog-v1-")


def test_protocol_routes_promote_deterministically_beyond_first_fallback(catalog_server):
    backends = {
        "chat-only": {**catalog_server.BACKENDS["low"], "target": "chat-only", "tier": "chat-only",
                      "protocols": ("chat_completions",), "fallbacks": ("resp-a",), "rank": 0},
        "resp-a": {**catalog_server.BACKENDS["responses"], "target": "resp-a", "tier": "resp-a",
                   "protocols": ("responses",), "fallbacks": ("resp-b",), "rank": 1},
        "resp-b": {**catalog_server.BACKENDS["safe"], "target": "resp-b", "tier": "resp-b",
                   "protocols": ("responses",), "fallbacks": (), "rank": 1},
        "chat-b": {**catalog_server.BACKENDS["work"], "target": "chat-b", "tier": "chat-b",
                   "protocols": ("chat_completions",), "fallbacks": (), "rank": 2},
    }
    original = catalog_server.BACKENDS
    catalog_server.BACKENDS = backends
    try:
        assert catalog_server._responses_routes("chat-only") == ("resp-a", "resp-b")
        assert catalog_server._chat_routes("resp-a") == ("chat-b",)
    finally:
        catalog_server.BACKENDS = original


def test_legacy_session_state_never_returns_a_protocol_incompatible_target(catalog_server, monkeypatch):
    monkeypatch.setattr(catalog_server, "BACKENDS", {
        "chat": {**catalog_server.BACKENDS["low"], "target": "chat", "tier": "chat",
                 "protocols": ("chat_completions",), "rank": 0},
        "responses": {**catalog_server.BACKENDS["responses"], "target": "responses", "tier": "responses",
                      "protocols": ("responses",), "rank": 1},
    })
    # Pre-catalog in-memory state had only `tier`; it must not leak into the
    # other native endpoint merely because it still uses the compatibility form.
    assert catalog_server._session_target({"tier": "chat"}, "responses") is None
    assert catalog_server._session_target({"tier": "responses"}, "chat") is None
    assert catalog_server._session_target({"tier": "chat"}, "chat") == "chat"



@pytest.mark.anyio
async def test_catalog_endpoint_capability_filtering_and_native_responses_refusal(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("work", None, 3, 1))
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload["model"]))
        if len(seen) == 1:
            return httpx.Response(200, json={
                "id": "resp_refusal", "object": "response", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal"}]}],
            })
        return httpx.Response(200, json={
            "id": "resp_ok", "object": "response", "status": "completed", "output": [],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task",
        })
    assert response.status_code == 200
    # work is Chat-only, so native Responses starts at responses and then
    # detects output[].content[type=refusal] before bounded fallback to safe.
    assert seen == [("/v1/responses", "openai/responses"),
                    ("/v1/responses", "openai/safe")]
    assert response.headers["x-route-decision"] == "safe"
    assert response.headers["x-route-reason"] == "responses_protocol_upgrade"
    assert response.headers["x-route-attempts"] == "2"
    assert not catalog_server._store_pinned(catalog_server._prompt_hash("task"), api_format="chat")
    await mock.aclose()


def test_native_responses_refusal_shapes_are_detected(catalog_server):
    assert catalog_server._is_refusal(200, {
        "output": [{"type": "message", "content": [{"type": "refusal"}]}],
    })
    assert catalog_server._is_refusal(200, {
        "type": "response.output_item.added",
        "item": {"type": "message", "content": [{"type": "refusal"}]},
    })
    assert catalog_server._is_refusal(200, {
        "type": "response.refusal.delta", "delta": "",
    })
    assert not catalog_server._is_refusal(200, {
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
    })


@pytest.mark.anyio
async def test_catalog_responses_stream_refusal_event_falls_back_before_output(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("responses", None, 4, 1))
    calls = []
    refusal = (b'event: response.output_item.added\n'
               b'data: {"type":"response.output_item.added","item":{"type":"message","content":[{"type":"refusal"}]}}\n\n')
    success = (b'event: response.completed\n'
               b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n')

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=refusal if len(calls) == 1 else success,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task", "stream": True,
        })
    # Streaming Responses cannot translate or replay emitted events: after the
    # first native refusal event is buffered, it is discarded and the next
    # compatible target emits the actual native completion.
    assert response.status_code == 200
    assert calls == ["openai/responses", "openai/safe"]
    assert "refusal" not in response.text
    assert "response.completed" in response.text
    assert response.headers["x-route-decision"] == "safe"
    await mock.aclose()



def test_refusal_learning_is_protocol_scoped_and_uses_lowest_compatible_rank(catalog_server, monkeypatch):
    monkeypatch.setattr(catalog_server, "PIN_EXPENSIVE_AFTER", 1)
    chat_only = {**catalog_server.BACKENDS["low"], "target": "chat-only", "tier": "chat-only",
                 "protocols": ("chat_completions",), "rank": 0}
    response_low = {**catalog_server.BACKENDS["responses"], "target": "response-low", "tier": "response-low",
                    "protocols": ("responses",), "fallbacks": ("response-high",), "rank": 1}
    response_high = {**catalog_server.BACKENDS["safe"], "target": "response-high", "tier": "response-high",
                     "protocols": ("responses",), "rank": 2}
    monkeypatch.setattr(catalog_server, "BACKENDS", {
        "chat-only": chat_only, "response-low": response_low, "response-high": response_high,
    })
    attempts = [("response-low", response_low, 200, "refusal")]
    catalog_server._record_refusal_learning("prompt", attempts, None, 2, api_format="responses")
    response_pin = catalog_server._store_pinned("prompt", api_format="responses")
    assert response_pin is not None and response_pin["decision"] == "response-high"
    assert catalog_server._store_pinned("prompt", api_format="chat") is None



@pytest.mark.anyio
async def test_catalog_chat_stream_refusal_is_learned_for_the_refusing_target(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "PIN_EXPENSIVE_AFTER", 1)
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))
    calls = []
    refusal = (b'data: {"choices":[{"delta":{"content":"I cannot assist"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=refusal if len(calls) == 1 else success,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "stream": True,
        })
    assert response.status_code == 200 and calls == ["vendor/low", "vendor/mid"]
    pin = catalog_server._store_pinned(catalog_server._prompt_hash("task"), api_format="chat")
    assert pin is not None and pin["decision"] == "mid"
    await mock.aclose()



@pytest.mark.anyio
async def test_catalog_chat_endpoint_promotes_from_responses_only(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only_responses = {**catalog_server.BACKENDS["responses"], "target": "only-responses",
                      "tier": "only-responses", "protocols": ("responses",),
                      "fallbacks": (), "rank": 0}
    chat = {**catalog_server.BACKENDS["work"], "target": "chat", "tier": "chat",
            "protocols": ("chat_completions",), "fallbacks": (), "rank": 1}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only-responses": only_responses, "chat": chat})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only-responses", None, 1, 1))
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append((request.url.path, payload["model"]))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}],
        })
    assert response.status_code == 200
    assert seen == [("/v1/chat/completions", "vendor/work")]
    assert response.headers["x-route-reason"] == "chat_protocol_upgrade"
    await mock.aclose()



def test_catalog_rejects_unknown_top_level_table(tmp_path):
    result = _catalog_subprocess(tmp_path, catalog_text() + '\n[route_typo]\nenabled = true\n')
    assert result.returncode != 0
    assert "catalog has unexpected field route_typo" in result.stderr


def test_catalog_session_can_upgrade_to_intermediate_rank(catalog_server):
    session_id = "catalog-session"
    catalog_server._session_state[session_id] = {
        "routes": {"chat": "low"}, "tier": "low",
        "last_seen": time.time(), "last_complexity": 1,
        "target_revision": catalog_server.TARGET_CONFIG_REVISION,
        "turns": 1,
    }
    decision, reason = catalog_server._session_route(
        session_id, "continue implementation", "mid", 2, None, api_format="chat")
    assert (decision, reason) == ("mid", "strong_upgrade")


def test_chat_structured_delta_refusal_is_not_replay_safe(catalog_server):
    payload = {"choices": [{"delta": {"refusal": ""}, "finish_reason": "stop"}]}
    assert catalog_server._is_refusal(200, payload)
    wire = (b'data: {"choices":[{"delta":{"refusal":""},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')
    assert not catalog_server._response_replay_safe({"stream": True, "messages": []}, wire)


@pytest.mark.anyio
async def test_catalog_chat_structured_delta_refusal_falls_back_before_output(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))
    calls = []
    refusal = (b'data: {"choices":[{"delta":{"refusal":""},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=refusal if len(calls) == 1 else success,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "stream": True,
        })
    assert response.status_code == 200
    assert calls == ["vendor/low", "vendor/mid"]
    assert "refusal" not in response.text and "ok" in response.text
    await mock.aclose()



def test_catalog_malformed_unicode_and_raw_json_sources_are_redacted(tmp_path):
    result = _catalog_subprocess(tmp_path, "", catalog_bytes=b"\xff\xfe")
    assert result.returncode != 0
    assert "AI_ROUTING_CONFIG could not be read" in result.stderr
    assert "UnicodeDecodeError" not in result.stderr

    result = _catalog_subprocess(tmp_path, catalog_text(), raw_json="{")
    assert result.returncode != 0
    assert "ROUTELLM_TARGETS_JSON must be valid JSON" in result.stderr

    for raw in ("null", "[]", "{}"):
        result = _catalog_subprocess(tmp_path, catalog_text(), raw_json=raw)
        assert result.returncode != 0
        assert "ROUTELLM_TARGETS_JSON must be a non-empty object" in result.stderr


def test_revision_invalidates_cache_and_session_state(catalog_server, monkeypatch):
    body = {"model": "auto", "messages": [{"role": "user", "content": "task"}]}
    old_key = catalog_server._cache_key(body, "stable-key")
    content = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
    catalog_server._cache_put(old_key, body, content)
    assert catalog_server._cache_get(old_key) == content

    catalog_server._session_note("session", "low", 1, api_format="chat")
    assert catalog_server._session_get("session") is not None
    monkeypatch.setattr(catalog_server, "TARGET_CONFIG_REVISION", "catalog-v1-rebound")
    new_key = catalog_server._cache_key(body, "stable-key")
    assert new_key != old_key and catalog_server._cache_get(new_key) is None
    assert catalog_server._session_get("session") is None


def test_protocol_scoped_session_note_never_crosses_native_apis(catalog_server):
    catalog_server._session_note("session", "low", 1, api_format="chat")
    state = catalog_server._session_get("session")
    assert state is not None
    assert catalog_server._session_target(state, "chat") == "low"
    assert catalog_server._session_target(state, "responses") is None


@pytest.mark.anyio
@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": "auto", "messages": [{"role": "user", "content": "task"}]}),
    ("/v1/responses", {"model": "auto", "input": "task"}),
])
async def test_endpoint_malformed_json_is_an_openai_error(catalog_server, path, body):
    # A real content body is used only to establish the endpoint; malformed
    # bytes exercise the independent parser before any provider request.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post(path, headers={**AUTH, "content-type": "application/json"}, content=b"{")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_json"


@pytest.mark.anyio
async def test_catalog_chat_returns_503_when_no_chat_target(catalog_server, monkeypatch):
    only = {**catalog_server.BACKENDS["responses"], "target": "only", "tier": "only",
            "protocols": ("responses",), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}],
        })
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "chat_backend_unavailable"


@pytest.mark.anyio
async def test_final_native_responses_refusal_is_not_learned_as_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["responses"], "target": "only", "tier": "only",
            "protocols": ("responses",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={
            "id": "r", "object": "response", "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal"}]}],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={"model": "auto", "input": "task"})
    assert response.status_code == 200 and calls == ["/v1/responses"]
    entry = catalog_server._decision_store[catalog_server._store_key(catalog_server._prompt_hash("task"), "responses")]
    assert entry["fail"] == 1 and entry["ok"] == 0
    await mock.aclose()


@pytest.mark.anyio
async def test_final_chat_refusal_is_not_learned_as_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["low"], "target": "only", "tier": "only",
            "protocols": ("chat_completions",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))

    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"refusal": ""}, "finish_reason": "stop"}],
    })))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}],
        })
    assert response.status_code == 200
    entry = catalog_server._decision_store[catalog_server._store_key(catalog_server._prompt_hash("task"), "chat")]
    assert entry["fail"] == 1 and entry["ok"] == 0
    await mock.aclose()


@pytest.mark.anyio
async def test_final_responses_stream_refusal_is_not_learned_as_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["responses"], "target": "only", "tier": "only",
            "protocols": ("responses",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    wire = (b'event: response.refusal.delta\n'
            b'data: {"type":"response.refusal.delta","delta":""}\n\n')
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, content=wire, headers={"content-type": "text/event-stream"})))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task", "stream": True,
        })
    assert response.status_code == 200 and "response.refusal.delta" in response.text
    entry = catalog_server._decision_store[catalog_server._store_key(catalog_server._prompt_hash("task"), "responses")]
    assert entry["fail"] == 1 and entry["ok"] == 0
    await mock.aclose()


@pytest.mark.anyio
async def test_final_chat_stream_refusal_is_not_learned_as_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["low"], "target": "only", "tier": "only",
            "protocols": ("chat_completions",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    wire = (b'data: {"choices":[{"delta":{"refusal":""},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, content=wire, headers={"content-type": "text/event-stream"})))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "stream": True,
        })
    assert response.status_code == 200 and "refusal" in response.text
    entry = catalog_server._decision_store[catalog_server._store_key(catalog_server._prompt_hash("task"), "chat")]
    assert entry["fail"] == 1 and entry["ok"] == 0
    await mock.aclose()



def test_cache_key_is_scoped_by_opaque_session_identity(catalog_server):
    body = {"model": "auto", "messages": [{"role": "user", "content": "task"}]}
    first = catalog_server._cache_key(body, "same-key", session_id="opaque-session-a")
    second = catalog_server._cache_key(body, "same-key", session_id="opaque-session-b")
    sessionless = catalog_server._cache_key(body, "same-key")
    assert len({first, second, sessionless}) == 3



# --- Native Responses preflight + opaque-continuation affinity regressions ---

@pytest.mark.anyio
async def test_native_responses_prefetch_commits_on_first_output_delta(catalog_server):
    delta = (b'event: response.output_text.delta\n'
             b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n')

    async def open_after_delta():
        yield delta
        await asyncio.Event().wait()

    prefix, refusal = await asyncio.wait_for(
        catalog_server._responses_prefetch_sse(open_after_delta(), time.monotonic() + 30), timeout=1)
    assert refusal is False and delta in prefix


@pytest.mark.anyio
async def test_native_responses_prefetch_detects_refusal_in_first_event(catalog_server):
    created = (b'event: response.created\n'
               b'data: {"type":"response.created","response":{"id":"r","status":"in_progress"}}\n\n')
    refusal = (b'event: response.output_item.added\n'
               b'data: {"type":"response.output_item.added","item":{"type":"message","content":[{"type":"refusal"}]}}\n\n')

    # A refusal carried by the first inspected event is discarded for failover.
    async def refusal_first():
        yield refusal
        await asyncio.Event().wait()

    prefix, detected = await asyncio.wait_for(
        catalog_server._responses_prefetch_sse(refusal_first(), time.monotonic() + 30), timeout=1)
    assert detected is True and refusal in prefix

    # A refusal arriving after a committed lifecycle frame cannot be retried:
    # the preflight commits on the first event to avoid stalling sparse streams.
    async def refusal_after_lifecycle():
        yield created
        yield refusal
        await asyncio.Event().wait()

    prefix, detected = await asyncio.wait_for(
        catalog_server._responses_prefetch_sse(refusal_after_lifecycle(), time.monotonic() + 30), timeout=1)
    assert detected is False and created in prefix
@pytest.mark.anyio
async def test_responses_affinity_first_response_issues_token_and_binds(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("responses", None, 4, 1))
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={
            "id": "resp_aff", "object": "response", "status": "completed", "output": [],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        first = await client.post("/v1/responses", headers=AUTH, json={"model": "auto", "input": "task"})
        assert first.status_code == 200
        token = first.headers.get("x-route-responses-affinity")
        assert token

        second = await client.post("/v1/responses", headers={**AUTH, "x-route-responses-affinity": token},
                                   json={"model": "auto", "input": "next", "previous_response_id": "resp_aff"})
        assert second.status_code == 200
        assert second.headers["x-route-decision"] == "responses"
        assert seen == ["openai/responses", "openai/responses"]

        # A token is not a permission slip: unknown provider state fails closed
        # even when a valid router token is presented.
        third = await client.post("/v1/responses", headers={**AUTH, "x-route-responses-affinity": token},
                                  json={"model": "auto", "input": "next", "previous_response_id": "resp_unknown"})
        assert third.status_code == 409
        assert third.json()["error"]["code"] == "responses_continuation_affinity_required"
        # Known provider state resolves through the router-issued index without
        # requiring the header, and stays on the exact same origin.
        fourth = await client.post("/v1/responses", headers=AUTH,
                                   json={"model": "auto", "input": "next", "previous_response_id": "resp_aff"})
        assert fourth.status_code == 200
        assert fourth.headers["x-route-decision"] == "responses"
        assert seen == ["openai/responses", "openai/responses", "openai/responses"]
        # Affinity-bound origin traffic must not seed the prompt store.
        assert catalog_server._store_pinned(catalog_server._prompt_hash("next"), api_format="responses") is None
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_affinity_opaque_encrypted_replay_binds_origin(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("responses", None, 4, 1))
    seen = []

    def handler(request):
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(200, json={
            "id": "resp_op", "object": "response", "status": "completed",
            "output": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher-a",
                        "summary": [{"type": "summary_text", "text": "s"}]}],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        first = await client.post("/v1/responses", headers=AUTH, json={"model": "auto", "input": "task"})
        assert first.status_code == 200
        replay = {"model": "auto", "input": [
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "cipher-a"},
            {"role": "user", "content": [{"type": "input_text", "text": "continue"}]},
        ]}
        second = await client.post("/v1/responses", headers=AUTH, json=replay)
        assert second.status_code == 200
        assert seen == ["openai/responses", "openai/responses"]
    await mock.aclose()


# --- Chat preflight / replay / multi-choice regressions ---------------------

class _FailingChatStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        raise httpx.ReadError("boom")
        yield b""  # pragma: no cover

    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_chat_stream_preflight_transport_failure_fails_over(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))
    calls = []
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            return httpx.Response(200, stream=_FailingChatStream(),
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=success,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "stream": True,
        })
    assert response.status_code == 200
    assert calls == ["vendor/low", "vendor/mid"]
    assert "ok" in response.text
    await mock.aclose()


def test_truncated_nonstream_answer_is_not_replay_safe(catalog_server):
    body = {"model": "auto", "messages": [{"role": "user", "content": "task"}]}
    content = json.dumps({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]}).encode()
    assert not catalog_server._response_replay_safe(body, content)
    key = catalog_server._cache_key(body, "stable-key")
    catalog_server._cache_put(key, body, content)
    assert catalog_server._cache_get(key) is None


def test_legacy_function_shapes_are_not_cached(catalog_server):
    for body in (
        {"model": "auto", "messages": [{"role": "user", "content": "x"}], "functions": [{"name": "f"}]},
        {"model": "auto", "messages": [{"role": "user", "content": "x"}], "function_call": "auto"},
        {"model": "auto", "messages": [{"role": "user", "content": "x"},
                                       {"role": "function", "name": "f", "content": "out"}]},
        {"model": "auto", "messages": [{"role": "assistant", "content": "x",
                                        "function_call": {"name": "f", "arguments": "{}"}}]},
    ):
        assert catalog_server._cache_key(body, "same-key") is None
    content = json.dumps({"choices": [{"message": {"function_call": {"name": "f", "arguments": "{}"},
                                                   "content": ""}, "finish_reason": "function_call"}]}).encode()
    assert not catalog_server._response_replay_safe({"model": "auto", "messages": []}, content)


def test_multi_choice_refusal_is_detected_in_any_choice(catalog_server):
    assert catalog_server._is_refusal(200, {"choices": [
        {"message": {"content": "ok"}, "finish_reason": "stop"},
        {"message": {"refusal": ""}, "finish_reason": "stop"},
    ]})
    assert catalog_server._is_refusal(200, {"choices": [
        {"message": {"content": "ok"}, "finish_reason": "stop"},
        {"message": {"content": "no"}, "finish_reason": "content_filter"},
    ]})
    assert not catalog_server._is_refusal(200, {"choices": [
        {"message": {"content": "ok"}, "finish_reason": "stop"},
        {"message": {"content": "ok"}, "finish_reason": "stop"},
    ]})


@pytest.mark.anyio
async def test_native_responses_preflight_commits_lifecycle_only_stream(catalog_server):
    created = (b'event: response.created\n'
               b'data: {"type":"response.created","response":{"id":"r","status":"in_progress"}}\n\n')

    async def silent_after_created():
        yield created
        await asyncio.Event().wait()

    prefix, refusal = await asyncio.wait_for(
        catalog_server._responses_prefetch_sse(silent_after_created(), time.monotonic() + 30), timeout=1)
    assert refusal is False and created in prefix


def test_legacy_env_parse_errors_do_not_block_explicit_sources(tmp_path):
    path = tmp_path / "catalog.toml"
    path.write_text(catalog_text())
    env = os.environ.copy()
    env.update({"AI_ROUTING_CONFIG": str(path), "ROUTER_CATALOG_TEST_KEY": "test-only-value",
                "CHEAP_MAX_TOKENS": "not-an-int", "ROUTELLM_TIMEOUT_S": "bad",
                "PYTHONPATH": str(Path(__file__).parents[1])})
    ok = subprocess.run([sys.executable, "-c", "import server"], env=env,
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr[-1000:]
    # An explicit source must name an existing catalog; a missing path fails
    # loudly even when legacy env values are malformed.
    env["AI_ROUTING_CONFIG"] = str(tmp_path / "missing.toml")
    bad = subprocess.run([sys.executable, "-c", "import server"], env=env,
                         capture_output=True, text=True)
    assert bad.returncode != 0
    assert "AI_ROUTING_CONFIG does not exist" in bad.stderr



class _BlockingChatStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_chat_stream_preflight_cancellation_closes_fallback(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))
    calls = []
    refusal = (b'data: {"choices":[{"delta":{"refusal":""},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    blocking = _BlockingChatStream()

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        if len(calls) == 1:
            return httpx.Response(200, content=refusal,
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, stream=blocking,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        task = asyncio.create_task(client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "stream": True,
        }))
        await asyncio.wait_for(blocking.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert blocking.closed
    assert calls == ["vendor/low", "vendor/mid"]
    await mock.aclose()


@pytest.mark.anyio
async def test_chat_stream_n2_partial_choices_not_complete_or_cached(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))
    calls = []
    wire = (b'data: {"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')

    def handler(request):
        calls.append(json.loads(request.content)["model"])
        return httpx.Response(200, content=wire,
                              headers={"content-type": "text/event-stream"})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        headers = {**AUTH, "Idempotency-Key": "n2-key"}
        body = {"model": "auto", "messages": [{"role": "user", "content": "task"}],
                "stream": True, "n": 2}
        one = await client.post("/v1/chat/completions", headers=headers, json=body)
        two = await client.post("/v1/chat/completions", headers=headers, json=body)
        assert one.status_code == 200 and two.status_code == 200
        # Choice 1 never arrived: not complete, never replayed or coalesced.
        assert len(calls) == 2
        assert two.headers.get("x-route-cache") is None
        assert catalog_server._store_pinned(catalog_server._prompt_hash("task"), api_format="chat") is None
    await mock.aclose()


def test_length_truncation_detected_in_any_choice(catalog_server):
    assert catalog_server._length_truncated({"choices": [
        {"message": {"content": "ok"}, "finish_reason": "stop"},
        {"message": {"content": "part"}, "finish_reason": "length"},
    ]})
    assert not catalog_server._length_truncated({"choices": [
        {"message": {"content": "ok"}, "finish_reason": "stop"},
        {"message": {"content": "ok"}, "finish_reason": "stop"},
    ]})


def test_json_duplicate_keys_are_rejected_and_wrapper_version_required(tmp_path):
    dup = '{"version": 1, "targets": {"only": {"adapter": "openai-compatible", "base_url": "https://u.test/v1", "credential_env": "ROUTER_CATALOG_TEST_KEY", "upstream_model": "m", "rank": 0, "rank": 7, "protocols": ["chat_completions"]}}, "complexity_targets": ["only"] * 5}'
    result = _catalog_subprocess(tmp_path, "", json_source=None, raw_json=dup)
    assert result.returncode != 0 and "must be valid JSON" in result.stderr

    # Wrapper form now requires version == 1 (no implicit version).
    no_version = json.dumps({"targets": {"only": {"adapter": "openai-compatible",
                                                  "base_url": "https://u.test/v1",
                                                  "credential_env": "ROUTER_CATALOG_TEST_KEY",
                                                  "upstream_model": "m", "rank": 0,
                                                  "protocols": ["chat_completions"]}}})
    result = _catalog_subprocess(tmp_path, "", json_source=None, raw_json=no_version)
    assert result.returncode != 0 and "version must be 1" in result.stderr

    # Reserved wrapper keys are rejected in the plain-map form.
    reserved = json.dumps({"version": {"adapter": "openai-compatible",
                                       "base_url": "https://u.test/v1",
                                       "credential_env": "ROUTER_CATALOG_TEST_KEY",
                                       "upstream_model": "m", "rank": 0,
                                       "protocols": ["chat_completions"]}})
    result = _catalog_subprocess(tmp_path, "", json_source=None, raw_json=reserved)
    assert result.returncode != 0 and "is reserved" in result.stderr


def test_catalog_inactive_policy_is_validated_and_rank_may_be_omitted(tmp_path):
    policy_block = '[routellm.policies.coding]\ncomplexity_targets = ["low", "mid", "work", "responses", "safe"]\n'
    dormant = ('[routellm.policies.dormant]\ncomplexity_targets = ["low", "mid", "work", "responses", "safe"]\n'
               'dormant_typo = 1\n')
    text = catalog_text().replace(policy_block, policy_block + dormant, 1)
    result = _catalog_subprocess(tmp_path, text)
    assert result.returncode != 0 and "unexpected field dormant_typo" in result.stderr

    # Omitting rank is valid; declaration order supplies the rank.
    text = catalog_text().replace('fallbacks = ["mid"]\nrank = 0\n', 'fallbacks = ["mid"]\n', 1)
    result = _catalog_subprocess(tmp_path, text)
    assert result.returncode == 0, result.stderr[-1000:]


def test_whitespace_reasoning_effort_and_bad_developer_role_are_rejected(tmp_path):
    text = catalog_text().replace('reasoning_effort = "none"', 'reasoning_effort = "   "', 1)
    result = _catalog_subprocess(tmp_path, text)
    assert result.returncode != 0 and "reasoning_effort" in result.stderr

    text = catalog_text().replace('developer_role = "native"', 'developer_role = ["native"]', 1)
    result = _catalog_subprocess(tmp_path, text)
    assert result.returncode != 0 and "developer_role" in result.stderr


def test_legacy_decision_rows_without_revision_are_intentionally_ignored(catalog_server, monkeypatch, tmp_path):
    store = tmp_path / "store.jsonl"
    store.write_text(json.dumps({
        "prompt_hash": "old-prompt", "api_format": "chat", "decision": "cheap",
        "ok": 9, "fail": 0, "ts": time.time(), "pin_until": time.time() + 60,
    }) + "\n")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", store)
    catalog_server._decision_store.clear()
    catalog_server._store_load()
    assert catalog_server._store_pinned("old-prompt", api_format="chat") is None


@pytest.mark.anyio
async def test_chat_nonstream_n2_partial_choices_do_not_seed_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))

    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "a"}, "finish_reason": "stop"}],
        })

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "n": 2,
        })
    assert response.status_code == 200
    assert catalog_server._store_pinned(catalog_server._prompt_hash("task"), api_format="chat") is None
    await mock.aclose()


@pytest.mark.anyio
async def test_chat_nonstream_duplicate_choice_indexes_do_not_seed_success(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("low", None, 1, 1))

    def handler(request):
        # Two terminal records but both claim index 0; requested choice 1 absent.
        return httpx.Response(200, json={"choices": [
            {"index": 0, "message": {"content": "a"}, "finish_reason": "stop"},
            {"index": 0, "message": {"content": "dup"}, "finish_reason": "stop"},
        ]})

    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/chat/completions", headers=AUTH, json={
            "model": "auto", "messages": [{"role": "user", "content": "task"}], "n": 2,
        })
    assert response.status_code == 200
    assert catalog_server._store_pinned(catalog_server._prompt_hash("task"), api_format="chat") is None
    await mock.aclose()


@pytest.mark.anyio
async def test_chat_prefetch_commits_on_comment_keepalive_frame(catalog_server):
    comment = b': synthetic keepalive\n\n'

    async def comment_then_silence():
        yield comment
        await asyncio.Event().wait()

    prefix, refusal = await asyncio.wait_for(
        catalog_server._prefetch_sse(comment_then_silence(), asyncio.get_running_loop().time() + 30), timeout=1)
    assert refusal is False and comment in prefix


@pytest.mark.anyio
async def test_final_responses_bare_done_is_not_completed_or_learned(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["responses"], "target": "only", "tier": "only",
            "protocols": ("responses",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    wire = b'data: [DONE]\n\n'
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, content=wire, headers={"content-type": "text/event-stream"})))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task", "stream": True,
        })
    assert response.status_code == 200
    entry = catalog_server._decision_store.get(
        catalog_server._store_key(catalog_server._prompt_hash("task"), "responses"))
    assert entry is None or entry.get("ok", 0) == 0
    await mock.aclose()


@pytest.mark.anyio
async def test_streamed_responses_carry_affinity_capability_header(catalog_server, monkeypatch, tmp_path):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("responses", None, 4, 1))
    wire = (b'event: response.completed\n'
            b'data: {"type":"response.completed","response":{"id":"resp_stream_1","status":"completed"}}\n\n')
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, content=wire, headers={"content-type": "text/event-stream"})))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task", "stream": True,
        })
        assert response.status_code == 200
        assert response.headers.get("x-route-responses-affinity")
        _ = response.text  # fully consume the streamed body
        # The streamed response id is indexed; a continuation resolves without the token.
        continuation = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "next", "previous_response_id": "resp_stream_1",
        })
        assert continuation.status_code == 200
        assert continuation.headers["x-route-decision"] == "responses"
    await mock.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["incomplete", "failed", None])
async def test_responses_completed_event_with_non_completed_status_is_not_success(
        catalog_server, monkeypatch, tmp_path, status):
    monkeypatch.setattr(catalog_server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(catalog_server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(catalog_server, "DECISION_STORE_PATH", tmp_path / "store.jsonl")
    only = {**catalog_server.BACKENDS["responses"], "target": "only", "tier": "only",
            "protocols": ("responses",), "fallbacks": (), "rank": 0}
    monkeypatch.setattr(catalog_server, "BACKENDS", {"only": only})
    monkeypatch.setattr(catalog_server, "_decide", lambda *args: ("only", None, 1, 1))
    response_obj = {"id": "resp_x", "status": status} if status is not None else {"id": "resp_x"}
    wire = ("event: response.completed\n"
            'data: ' + json.dumps({"type": "response.completed", "response": response_obj}) + '\n\n').encode()
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, content=wire, headers={"content-type": "text/event-stream"})))
    monkeypatch.setattr(catalog_server, "_client", mock)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=catalog_server.app),
                                 base_url="http://router") as client:
        response = await client.post("/v1/responses", headers=AUTH, json={
            "model": "auto", "input": "task", "stream": True,
        })
        _ = response.text
    assert response.status_code == 200
    entry = catalog_server._decision_store.get(
        catalog_server._store_key(catalog_server._prompt_hash("task"), "responses"))
    assert entry is None or entry.get("ok", 0) == 0
    await mock.aclose()
