"""Provider-facing HTTP contract tests."""

import json
import threading
import time
from types import SimpleNamespace

import api
import providers
import pytest
import serve
from fastapi.testclient import TestClient
from openai import OpenAI


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_ROUTER_KEY", "gateway-key")
    return TestClient(api.app)


def _headers():
    return {"Authorization": "Bearer test-key"}


def _run():
    return SimpleNamespace(
        run_id="a" * 32,
        kind="trinity",
        terminated_by="verifier_accept",
        turns=[],
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        validate_output=lambda _text: None,
    )


def test_auth_health_and_validation(client, monkeypatch):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers=_headers()).status_code == 200
    monkeypatch.delenv("MANTIS_API_KEY")
    assert client.get("/ready").status_code == 503
    assert client.get("/v1/models", headers=_headers()).status_code == 503
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    bad = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/trinity", "messages": [], "unknown": True},
    )
    assert bad.status_code == 400
    no_tools = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        },
    )
    assert no_tools.status_code == 400
    invalid_role_fields = [
        {"role": "tool", "content": "x"},
        {"role": "user", "content": "x", "tool_calls": []},
    ]
    for message in invalid_role_fields:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": "mantis/trinity", "messages": [message]},
        )
        assert response.status_code == 400
    invalid_requests = [
        {"stream_options": {"include_usage": True}},
        {"max_tokens": 1, "max_completion_tokens": 1},
        {"reasoning": {"effort": "high"}, "reasoning_effort": "high"},
    ]
    for extra in invalid_requests:
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={
                "model": "mantis/trinity",
                "messages": [{"role": "user", "content": "hi"}],
                **extra,
            },
        )
        assert response.status_code == 400


def test_completion_reports_per_request_usage_and_hides_trace(client, monkeypatch):
    run = _run()
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/trinity", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"]
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "answer"
    # Usage describes this request's context (client messages + response), not the
    # run's accumulated orchestration usage, so context-tracking clients (e.g. the
    # prime-agent harness) do not see runaway growth across tool rounds.
    assert body["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert "mantis" not in body


def test_tool_choice_reaches_run(client, monkeypatch):
    captured = []
    run = _run()

    def create(_mode, body):
        captured.append(body["tool_choice"])
        return run

    monkeypatch.setattr(serve, "create_run", create)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "read"}}],
            "tool_choice": {"type": "function", "function": {"name": "read"}},
        },
    )
    assert response.status_code == 200
    assert captured == [{"type": "function", "function": {"name": "read"}}]


def test_buffered_stream_includes_usage(client, monkeypatch):
    run = _run()
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert response.status_code == 200
    assert response.headers["x-mantis-streaming"] == "live-status,verified-buffered-content"
    assert '"choices": []' in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_stream_sends_keepalive_while_waiting(client, monkeypatch):
    run = _run()

    def advance(*_args):
        time.sleep(0.03)
        return {"type": "final", "text": "answer"}

    monkeypatch.setattr(api, "_KEEPALIVE_SECONDS", 0.005)
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(serve, "_advance_to_boundary", advance)
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert response.status_code == 200
    assert ": keep-alive\n\n" in response.text


def test_stream_emits_live_status_before_verified_content(client, monkeypatch):
    run = serve.NativeRun("a" * 32)
    run.kind = "trinity"
    run.terminated_by = "verifier_accept"

    def advance(*_args):
        run.record_activity(
            "step",
            role="Worker",
            model="openrouter/openai/gpt-5.6-sol|medium",
            status="started",
            summary="Drafting an answer",
        )
        return {"type": "final", "text": "answer"}

    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(serve, "_advance_to_boundary", advance)
    monkeypatch.setattr(serve, "delete_run", lambda *_a, **_k: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    payloads = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]
    status = next(payload for payload in payloads if payload.get("mantis_event"))
    answer_index = next(
        index
        for index, payload in enumerate(payloads)
        if any(choice.get("delta", {}).get("content") for choice in payload.get("choices", []))
    )
    assert status["mantis_event"]["type"] == "step"
    assert status["mantis_event"]["sequence"] == 0
    assert "Drafting an answer" in status["choices"][0]["delta"]["reasoning"]
    assert payloads.index(status) < answer_index

    sdk = OpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client)
    with sdk.chat.completions.create(
        model="mantis/trinity",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    ) as stream:
        chunks = list(stream)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "answer"
    assert any(getattr(chunk.choices[0].delta, "reasoning", None) for chunk in chunks)


def test_stream_status_can_be_disabled(client, monkeypatch):
    run = serve.NativeRun("b" * 32)
    run.kind = "trinity"

    def advance(*_args):
        run.record_activity("step", role="Worker", status="started")
        return {"type": "final", "text": "answer"}

    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(serve, "_advance_to_boundary", advance)
    monkeypatch.setattr(serve, "delete_run", lambda *_a, **_k: True)
    response = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "X-Mantis-Events": "none"},
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    assert "mantis_event" not in response.text
    assert '"content": "answer"' in response.text


def test_stream_close_signals_provider_cancellation(monkeypatch):
    disconnected = threading.Event()

    def complete(_request, _headers):
        try:
            while True:
                serve._check_client_connected()
                time.sleep(0.001)
        except serve.ClientDisconnectedError:
            disconnected.set()
            raise

    monkeypatch.setattr(api, "_complete", complete)
    monkeypatch.setattr(api, "_KEEPALIVE_SECONDS", 0.005)
    request = api.ChatRequest(
        model="mantis/trinity", messages=[api.Message(role="user", content="hi")], stream=True
    )
    assert api._capacity.acquire(blocking=False)
    stream = api._stream(request)
    assert next(stream) == b": keep-alive\n\n"
    stream.close()
    assert disconnected.wait(timeout=0.2)


def test_stream_reports_errors_after_headers(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "_complete",
        lambda _request: (_ for _ in ()).throw(api.HTTPException(502, "failed")),
    )
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    )
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: {")]
    assert json.loads(events[0])["choices"][0]["finish_reason"] == "error"


def test_api_maps_run_errors(client, monkeypatch):
    request = api.ChatRequest(
        model="mantis/trinity", messages=[api.Message(role="user", content="hi")]
    )
    cases = (
        (serve.RunCapacityError("full"), 429),
        (KeyError("expired"), 409),
        (ValueError("bad"), 400),
        (RuntimeError("upstream"), 502),
    )
    for error, status in cases:
        monkeypatch.setattr(api, "_advance", lambda *_a, error=error: (_ for _ in ()).throw(error))
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json=request.model_dump(),
        )
        assert response.status_code == status


def test_structured_output_validation():
    run = serve.NativeRun("structured")
    run.response_format = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }
        },
    }
    run.validate_output('{"ok":true}')
    with pytest.raises(ValueError, match="valid JSON"):
        run.validate_output("nope")
    with pytest.raises(ValueError, match="match schema"):
        run.validate_output('{"ok":"yes"}')


def test_usage_accumulator():
    run = serve.NativeRun("usage")
    run.add_usage(None)
    run.add_usage(
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
    )
    run.add_usage(
        {
            "prompt_tokens": 4,
            "completion_tokens": 5,
            "total_tokens": 9,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
    )
    assert run.usage == {
        "prompt_tokens": 6,
        "completion_tokens": 8,
        "total_tokens": 14,
        "completion_tokens_details": {"reasoning_tokens": 5},
    }


def test_request_controls_are_stored_on_run():
    body = {
        "messages": [{"role": "user", "content": "answer"}],
        "slot_models": ["worker"],
        "max_completion_tokens": 123,
        "reasoning": {"effort": "high", "exclude": True},
        "web_search_options": {"search_context_size": "low"},
    }
    run = serve.create_run("trinity", body)
    try:
        assert run.controls == {
            "max_tokens": 123,
            "reasoning": {"effort": "high", "exclude": True},
            "web_search_options": {"search_context_size": "low"},
        }
    finally:
        serve.delete_run(run.run_id)


def test_multimodal_and_structured_request_contract(client, monkeypatch):
    captured = []
    run = _run()
    run.validate_output = lambda text: captured.append(("validated", text))

    def create(_mode, body):
        captured.append(body)
        return run

    monkeypatch.setattr(serve, "create_run", create)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": '{"ok":true}'}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/trinity",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    assert response.status_code == 200
    body = captured[0]
    assert body["messages"][0]["content"][1]["type"] == "image_url"
    assert body["response_format"]["json_schema"]["schema"]["required"] == ["ok"]
    assert captured[1] == ("validated", '{"ok":true}')


def test_invalid_structured_output_returns_502(client, monkeypatch):
    run = _run()
    run.validate_output = lambda _text: (_ for _ in ()).throw(ValueError("schema mismatch"))
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "bad"})
    monkeypatch.setattr(serve, "delete_run", lambda *_a, **_k: True)
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/trinity", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert "schema mismatch" in response.json()["error"]["message"]


def test_body_limit_returns_413():
    limited = api.FastAPI()
    limited.add_middleware(api.BodyLimitMiddleware, max_bytes=10)

    @limited.post("/")
    async def endpoint():
        return {"ok": True}

    response = TestClient(limited).post("/", content=b"x" * 11)
    assert response.status_code == 413
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_official_openai_sdk_contract(client, monkeypatch):
    run = _run()
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(
        serve, "_advance_to_boundary", lambda *_a: {"type": "final", "text": "answer"}
    )
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    sdk = OpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client)
    completion = sdk.chat.completions.create(
        model="mantis/trinity", messages=[{"role": "user", "content": "hi"}]
    )
    assert completion.choices[0].message.content == "answer"
    with sdk.chat.completions.create(
        model="mantis/trinity", messages=[{"role": "user", "content": "hi"}], stream=True
    ) as stream:
        assert "".join(chunk.choices[0].delta.content or "" for chunk in stream) == "answer"


def test_capacity_returns_429(client, monkeypatch):
    monkeypatch.setattr(api, "_capacity", SimpleNamespace(acquire=lambda **_k: False))
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/trinity", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 429
    assert response.json()["error"]["type"] == "rate_limit_error"


def test_ready_reports_only_sanitized_endpoint_metadata(client, monkeypatch):
    marker = "never-return-this-secret"
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "direct")
    monkeypatch.setenv("MANTIS_ROUTER_KEY", "gateway-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://direct.example.test/v1?token=" + marker)
    monkeypatch.setenv("OPENCODE_GO_ENDPOINT_URL", "https://opencode.example.test/private")
    body = client.get("/ready").json()
    assert body["endpoint_profile"] == "direct"
    assert body["endpoint_hosts"] == {
        "openrouter": "direct.example.test",
        "opencode": "opencode.example.test",
    }
    assert set(body["endpoint_fingerprints"]) == {"openrouter", "opencode"}
    assert body["endpoint_fingerprints"]["openrouter"] != body["endpoint_fingerprints"]["opencode"]
    assert marker not in str(body)

    first_fingerprint = body["endpoint_fingerprints"]["openrouter"]
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://direct.example.test/v2")
    changed = client.get("/ready").json()
    assert changed["endpoint_hosts"]["openrouter"] == "direct.example.test"
    assert changed["endpoint_fingerprints"]["openrouter"] != first_fingerprint


def test_ready_direct_requires_gateway_key(client, monkeypatch):
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "direct")
    monkeypatch.delenv("MANTIS_ROUTER_KEY", raising=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["message"] == "MANTIS_ROUTER_KEY is not configured"


# --- catalog readiness: credentials and binding fingerprint -------------------


def _catalog_runtime_env(monkeypatch, keys: dict | None = None, env_keys: dict | None = None):
    import json as _json

    import model_catalog
    import model_catalog_runtime

    abi = model_catalog.load_abi_manifest()
    slots = tuple(abi.slot_order)
    conductor = abi.conductor
    providers = {
        "edge": {
            "adapter": "openrouter",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }
    workers = {
        slot: {
            "provider": "edge",
            "upstream_model": f"vendor/{slot}",
            "model_identity": f"logical/{slot}",
            "protocols": ["responses"] if slot == conductor else ["chat_completions"],
        }
        for slot in slots
    }
    bindings = model_catalog._runtime_bindings(providers, workers)
    contract = model_catalog_runtime._runtime_contract_hash(bindings, slots, conductor)
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "catalog")
    monkeypatch.setenv("MANTIS_PROVIDER_BINDINGS", _json.dumps(providers))
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", _json.dumps(workers))
    monkeypatch.setenv("MANTIS_WORKER_MODELS", ",".join(slots))
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", conductor)
    monkeypatch.setenv("MANTIS_IDENTITY_CONTRACT", contract)
    monkeypatch.setenv("MANTIS_PROVIDER_KEYS", _json.dumps(keys) if keys else "")
    for name, value in (env_keys or {}).items():
        monkeypatch.setenv(name, value)
    return contract


def test_ready_catalog_requires_credentials(client, monkeypatch):
    _catalog_runtime_env(monkeypatch)
    monkeypatch.delenv("EDGE_KEY", raising=False)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "EDGE_KEY" in response.json()["error"]["message"]


def test_ready_catalog_accepts_injected_credentials(client, monkeypatch):
    _catalog_runtime_env(monkeypatch, keys={"edge": "injected-key"})
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["endpoint_profile"] == "catalog"
    assert body["endpoint_hosts"] == {"edge": "edge.example.test"}
    assert len(body["catalog_identity_contract"]) == 64
    assert len(body["binding_fingerprint"]) == 64
    assert "injected-key" not in str(body)


def test_ready_catalog_accepts_named_env_credentials(client, monkeypatch):
    _catalog_runtime_env(monkeypatch, env_keys={"EDGE_KEY": "native-key"})
    assert client.get("/ready").status_code == 200


def test_ready_catalog_rejects_partial_credentials(client, monkeypatch):
    _catalog_runtime_env(monkeypatch, keys={})
    monkeypatch.delenv("EDGE_KEY", raising=False)
    assert client.get("/ready").status_code == 503


def test_ready_catalog_rejects_stale_binding_fingerprint_change(client, monkeypatch):
    """A mutable binding change must be visible in the readiness metadata."""
    _catalog_runtime_env(monkeypatch, keys={"edge": "injected-key"})
    baseline = client.get("/ready").json()["binding_fingerprint"]
    import json as _json

    import model_catalog
    import model_catalog_runtime

    abi = model_catalog.load_abi_manifest()
    slots = tuple(abi.slot_order)
    conductor = abi.conductor
    providers = {
        "edge": {
            "adapter": "openrouter",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }
    workers = {
        slot: {
            "provider": "edge",
            "upstream_model": f"vendor/{slot}",
            "model_identity": f"logical/{slot}",
            "protocols": ["responses"] if slot == conductor else ["chat_completions"],
        }
        for slot in slots
    }
    workers[conductor]["upstream_model"] = "vendor/renamed"
    bindings = model_catalog._runtime_bindings(providers, workers)
    contract = model_catalog_runtime._runtime_contract_hash(bindings, slots, conductor)
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", _json.dumps(workers))
    monkeypatch.setenv("MANTIS_IDENTITY_CONTRACT", contract)
    changed = client.get("/ready").json()["binding_fingerprint"]
    assert changed != baseline


def _router_client(handler):
    import httpx

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_basic_model_relays_router_response_and_session(client, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return __import__("httpx").Response(
            200,
            json={"id": "chatcmpl-router", "choices": [{"message": {"content": "ok"}}]},
            headers={"x-route-decision": "middle", "x-route-reason": "strong_upgrade"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(api, "_router_client", lambda: _router_client(handler))
    response = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "X-Route-Session": "pi-session"},
        json={"model": "mantis/base", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-router"
    assert response.headers["x-route-decision"] == "middle"
    assert response.headers["x-route-reason"] == "strong_upgrade"
    assert seen["body"]["model"] == "auto"
    assert seen["headers"]["authorization"] == "Bearer router-key"
    assert seen["headers"]["x-route-session"] == "pi-session"


def test_basic_model_relays_body_session_identity_as_header(client, monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return __import__("httpx").Response(
            200,
            json={"id": "chatcmpl-router", "choices": [{"message": {"content": "ok"}}]},
            headers={"x-route-decision": "cheap"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(api, "_router_client", lambda: _router_client(handler))
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/base",
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"session_id": "meta-session"},
            "user": "user-session",
        },
    )
    assert response.status_code == 200
    # Body session identity must reach the router as X-Route-Session...
    assert seen["headers"]["x-route-session"] == "meta-session"
    # ...but must not leak into the upstream body where strict providers
    # reject unknown top-level fields.
    assert "metadata" not in seen["body"]
    assert "user" not in seen["body"]


def test_basic_model_relays_body_user_as_header_when_no_metadata(client, monkeypatch):
    seen = {}

    def handler(request):
        seen["headers"] = request.headers
        return __import__("httpx").Response(
            200,
            json={"id": "chatcmpl-router", "choices": [{"message": {"content": "ok"}}]},
            headers={"x-route-decision": "cheap"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(api, "_router_client", lambda: _router_client(handler))
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/base",
            "messages": [{"role": "user", "content": "hi"}],
            "user": "user-session",
        },
    )
    assert response.status_code == 200
    assert seen["headers"]["x-route-session"] == "user-session"


def test_basic_model_relays_router_stream(client, monkeypatch):
    def handler(_request):
        return __import__("httpx").Response(
            200,
            content=b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream", "x-route-decision": "cheap"},
        )

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(api, "_router_client", lambda: _router_client(handler))
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/base",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.headers["x-route-decision"] == "cheap"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content.endswith(b"data: [DONE]\n\n")


def test_basic_model_reports_router_connection_failure(client, monkeypatch):
    import httpx

    monkeypatch.setenv("MANTIS_ROUTER_KEY", "router-key")
    monkeypatch.setattr(
        api,
        "_router_client",
        lambda: _router_client(
            lambda request: (_ for _ in ()).throw(httpx.ConnectError("down", request=request))
        ),
    )
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/base", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_error"


def test_only_public_mantis_model_ids_are_accepted(client):
    for model in ("mantis-basic", "fugu", "conductor", "mantis-fugu"):
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 400


def test_short_model_aliases_are_accepted(client):
    # Short aliases route to the same modes and reach the gateway, which is not
    # running in tests, so they return the upstream 502 instead of 400.
    for model in ("base", "trinity", "ultra"):
        response = client.post(
            "/v1/chat/completions",
            headers=_headers(),
            json={"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code in (400, 502), f"{model}: {response.status_code}"
        body = response.json()
        assert body["error"]["type"] in ("invalid_request_error", "upstream_error")


def test_coerce_reasoning_effort_maps_deepseek_levels():
    assert providers._coerce_reasoning_effort("deepseek-v4-flash", "medium") is None
    assert providers._coerce_reasoning_effort("deepseek-v4-flash", "low") is None
    assert providers._coerce_reasoning_effort("deepseek-v4-pro", "high") == "high"
    assert providers._coerce_reasoning_effort("deepseek-v4-pro", "xhigh") == "max"
    assert providers._coerce_reasoning_effort("deepseek-v4-pro", "max") == "max"


def test_coerce_reasoning_effort_preserves_unknown_models():
    assert providers._coerce_reasoning_effort("vendor/custom", "xhigh") == "xhigh"


def test_sanitize_messages_strips_cross_model_reasoning():
    messages = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning": "long chain",
            "reasoning_details": [{"type": "reasoning"}],
            "_anthropic_content": [{"type": "thinking", "thinking": "..."}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "ok", "reasoning": "..."},
    ]
    out = providers._sanitize_messages(messages, "openai/gpt-4o", is_anthropic=False)
    assert out[0]["content"] == "answer"
    assert "reasoning" not in out[0]
    assert "reasoning_details" not in out[0]
    assert "_anthropic_content" not in out[0]
    assert "reasoning" not in out[1]


def test_sanitize_messages_keeps_deepseek_reasoning_for_tool_calls_only():
    with_tools = {
        "role": "assistant",
        "content": "plan",
        "reasoning": "long chain",
        "tool_calls": [{"id": "1", "function": {"name": "bash"}}],
    }
    without_tools = {
        "role": "assistant",
        "content": "answer",
        "reasoning": "long chain",
    }
    out = providers._sanitize_messages([with_tools, without_tools], "deepseek-v4-flash")
    assert out[0].get("reasoning") == "long chain"
    assert "reasoning" not in out[1]


def test_sanitize_messages_keeps_anthropic_blocks_for_anthropic_target():
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "_anthropic_content": [{"type": "thinking", "thinking": "..."}],
            "reasoning": "long chain",
        }
    ]
    out = providers._sanitize_messages(messages, "anthropic/claude-opus-5", is_anthropic=True)
    assert out[0]["_anthropic_content"]
    assert "reasoning" not in out[0]
