import asyncio
import json

import httpx
import pytest

import server

AUTH = {"Authorization": "Bearer sk-route-local"}
BODY = {"model": "auto", "input": "Implement this change", "max_output_tokens": 4096}


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_decide", lambda *args: ("cheap", None, 2, 10))
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(server, "DECISION_STORE_PATH", tmp_path / "decision-state.jsonl")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app),
                                base_url="http://router") as value:
        yield value


def upstream(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1)


def configure_three_tiers(monkeypatch):
    gateway = "https://unified-ai-gateway.siddsantham.workers.dev/v1"
    cheap = {**server.CHEAP, "tier": "cheap", "base": gateway,
             "model": "deepseek-v4-flash", "key": "test"}
    middle = {**server.MIDDLE, "tier": "middle", "base": gateway,
              "model": "openai/gpt-5.6-terra", "effort": "max", "key": "test"}
    expensive = {**server.EXPENSIVE, "tier": "expensive", "base": gateway,
                 "model": "openai/gpt-5.6-sol", "key": "test"}
    monkeypatch.setattr(server, "CHEAP", cheap)
    monkeypatch.setattr(server, "MIDDLE", middle)
    monkeypatch.setattr(server, "EXPENSIVE", expensive)
    monkeypatch.setattr(server, "BACKENDS", {
        "cheap": cheap, "middle": middle, "expensive": expensive,
    })
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", True)
    return cheap, middle, expensive


def test_responses_capability_requires_openai_and_supported_host():
    assert server._supports_responses({
        "base": "https://unified-ai-gateway.siddsantham.workers.dev/v1",
        "model": "openai/gpt-5.6-sol",
    })
    assert server._supports_responses({
        "base": "https://openrouter.ai/api/v1", "model": "openai/gpt-5.6-sol",
    })
    assert not server._supports_responses({
        "base": "https://modal.example/v1", "model": "kimi-k3",
    })
    assert not server._supports_responses({
        "base": "https://other.example/v1", "model": "openai/gpt-5.6-sol",
    })


@pytest.mark.anyio
async def test_responses_promotes_and_preserves_body(client, monkeypatch):
    _, middle, expensive = configure_three_tiers(monkeypatch)
    # Use the legacy incompatible middle model for the protocol-promotion test.
    middle.update(model="kimi-k3")
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        payload = {"id": "resp_1", "object": "response", "output": []}
        return httpx.Response(200, json=payload, headers={"content-type": "application/json"})

    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    body = {**BODY, "instructions": "Be concise", "metadata": {"session_id": "s"},
            "include": ["reasoning.encrypted_content"], "tools": [{"type": "web_search"}]}
    response = await client.post("/v1/responses", headers=AUTH, json=body)
    assert response.status_code == 200
    assert response.json() == {"id": "resp_1", "object": "response", "output": []}
    assert seen["path"] == "/v1/responses"
    assert seen["body"]["model"] == expensive["model"]
    assert seen["body"]["input"] == BODY["input"]
    assert seen["body"]["max_output_tokens"] == 4096
    assert seen["body"]["instructions"] == "Be concise"
    assert "messages" not in seen["body"]
    assert "max_tokens" not in seen["body"]
    assert "max_completion_tokens" not in seen["body"]
    assert "reasoning_effort" not in seen["body"]
    assert response.headers["x-route-api"] == "responses"
    assert response.headers["x-route-upstream-path"] == "/responses"
    assert response.headers["x-route-decision"] == "expensive"
    assert response.headers["x-route-reason"] == "responses_protocol_upgrade"
    await mock.aclose()


@pytest.mark.anyio
async def test_chat_still_uses_chat_completions(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    seen = []
    mock = upstream(lambda request: seen.append(request.url.path) or httpx.Response(
        200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
    ))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={
        "model": "auto", "messages": [{"role": "user", "content": "hello"}],
    })
    assert response.status_code == 200 and seen == ["/v1/chat/completions"]
    assert response.headers["x-route-api"] == "chat"
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_fails_when_no_capable_backend(client, monkeypatch):
    gateway = "https://unified-ai-gateway.siddsantham.workers.dev/v1"
    backends = {
        "cheap": {**server.CHEAP, "tier": "cheap", "base": gateway, "model": "deepseek-v4-flash"},
        "middle": {**server.MIDDLE, "tier": "middle", "base": gateway, "model": "kimi-k3"},
        "expensive": {**server.EXPENSIVE, "tier": "expensive", "base": gateway, "model": "kimi-k3"},
    }
    monkeypatch.setattr(server, "BACKENDS", backends)
    monkeypatch.setattr(server, "CHEAP", backends["cheap"])
    monkeypatch.setattr(server, "MIDDLE", backends["middle"])
    monkeypatch.setattr(server, "EXPENSIVE", backends["expensive"])
    response = await client.post("/v1/responses", headers=AUTH, json=BODY)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "responses_backend_unavailable"


@pytest.mark.anyio
async def test_responses_failover_skips_incompatible_middle(client, monkeypatch):
    cheap, middle, expensive = configure_three_tiers(monkeypatch)
    cheap.update(model="openai/gpt-5.6-luna")
    middle.update(model="kimi-k3")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append((request.url.path, payload["model"]))
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "credits unavailable"}})
        return httpx.Response(200, json={"id": "resp_fallback", "object": "response", "output": []})

    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers=AUTH, json=BODY)
    assert response.status_code == 200
    assert calls == [("/v1/responses", cheap["model"]),
                     ("/v1/responses", expensive["model"])]
    assert middle["model"] not in {model for _, model in calls}
    assert response.headers["x-route-fallback"] == "true"
    await mock.aclose()


@pytest.mark.anyio
async def test_session_affinity_cannot_force_incompatible_tier(client, monkeypatch):
    _, middle, _ = configure_three_tiers(monkeypatch)
    middle.update(model="kimi-k3")
    monkeypatch.setattr(server, "_decide", lambda *args: ("middle", None, 3, 10))
    session = "session-response"
    raw_id, _ = server._session_id({}, type("Request", (), {
        "headers": {"x-route-session": session},
    })())
    server._session_state[raw_id] = {"tier": "middle", "last_seen": __import__("time").time(),
                                     "turns": 1, "last_complexity": 3}
    seen = []
    mock = upstream(lambda request: seen.append(json.loads(request.content)["model"]) or httpx.Response(
        200, json={"id": "resp_2", "object": "response", "output": []},
    ))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers={**AUTH, "X-Route-Session": session}, json=BODY)
    assert response.status_code == 200
    assert seen == [server.EXPENSIVE["model"]]
    assert response.headers["x-route-reason"] == "responses_protocol_upgrade"
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_stream_preserves_frames_and_completion(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    wire = (b'event: response.output_text.delta\n'
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            b'event: response.completed\n'
            b'data: {"type":"response.completed","response":{"id":"resp_3","usage":{"input_tokens":10}}}\n\n'
            b'event: should.not.appear\ndata: {"type":"should.not.appear"}\n\n')
    mock = upstream(lambda request: httpx.Response(200, content=wire,
                                                    headers={"content-type": "text/event-stream"}))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers=AUTH, json={**BODY, "stream": True})
    assert response.status_code == 200
    assert "response.output_text.delta" in response.text
    assert "response.completed" in response.text
    assert "should.not.appear" not in response.text
    assert "upstream_truncated" not in response.text
    await mock.aclose()


class BlockingResponsesStream(httpx.AsyncByteStream):
    def __init__(self):
        self.blocked = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        yield (b'event: response.output_text.delta\n'
               b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
        self.blocked.set()
        await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_responses_stream_cancellation_closes_upstream(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    stream = BlockingResponsesStream()
    mock = upstream(lambda request: httpx.Response(200, stream=stream,
                                                    headers={"content-type": "text/event-stream"}))
    monkeypatch.setattr(server, "_client", mock)
    task = asyncio.create_task(client.post("/v1/responses", headers=AUTH,
                                           json={**BODY, "stream": True}))
    await stream.blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_idempotency_is_disabled(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": f"resp_{calls}", "object": "response", "output": []})

    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    headers = {**AUTH, "Idempotency-Key": "responses-key"}
    one = await client.post("/v1/responses", headers=headers, json=BODY)
    two = await client.post("/v1/responses", headers=headers, json=BODY)
    assert calls == 2 and one.json()["id"] != two.json()["id"]
    assert one.headers.get("x-route-cache") is None and two.headers.get("x-route-cache") is None
    await mock.aclose()



def test_responses_validation_and_prompt_extraction():
    assert server._validate_responses_request({"model": "auto"}) == ("'input' is required", "input")
    assert server._validate_responses_request({"model": "auto", "input": "x", "stream": 1})[1] == "stream"
    assert server._validate_responses_request({"model": "auto", "input": "x", "max_output_tokens": True})[1] == "max_output_tokens"
    body = {"model": "auto", "input": [
        {"role": "user", "content": [{"type": "input_text", "text": "old"}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": "ignore"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "latest"}]},
    ]}
    assert server._extract_responses_prompt(body) == "latest"


def test_responses_body_clamps_and_preserves_reasoning(monkeypatch):
    backend = {**server.EXPENSIVE, "model": "openai/gpt-5.6-sol", "max_tokens": 100,
               "effort": "medium"}
    body = {"model": "auto", "input": "x", "max_output_tokens": 200,
            "reasoning": {"effort": "high", "summary": "detailed"},
            "unknown_extension": {"x": 1}}
    outgoing = server._build_responses_body(body, backend)
    assert outgoing["model"] == backend["model"] and outgoing["max_output_tokens"] == 100
    assert outgoing["reasoning"] == {"effort": "high", "summary": "detailed"}
    assert outgoing["unknown_extension"] == {"x": 1}
    assert "messages" not in outgoing and "reasoning_effort" not in outgoing


def test_responses_body_uses_maximum_middle_effort_by_default():
    backend = {**server.MIDDLE, "model": "openai/gpt-5.6-terra", "effort": "max"}
    outgoing = server._build_responses_body({"model": "auto", "input": "x"}, backend)
    assert outgoing["model"] == "openai/gpt-5.6-terra"
    assert outgoing["reasoning"] == {"effort": "max"}


def test_responses_body_middle_effort_overrides_prime_default_only_for_middle():
    middle = {**server.MIDDLE, "tier": "middle", "model": "openai/gpt-5.6-terra",
              "effort": "max"}
    body = {"model": "auto", "input": "x",
            "reasoning": {"effort": "medium", "summary": "auto"}}
    outgoing = server._build_responses_body(body, middle)
    assert outgoing["reasoning"] == {"effort": "max", "summary": "auto"}

    expensive = {**server.EXPENSIVE, "tier": "expensive", "effort": "medium"}
    preserved = server._build_responses_body(body, expensive)
    assert preserved["reasoning"] == body["reasoning"]


@pytest.mark.anyio
async def test_responses_incomplete_stream_is_forwarded_without_synthetic_success(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    wire = (b'event: response.incomplete\n'
            b'data: {"type":"response.incomplete","response":{"status":"incomplete"}}\n\n'
            b'event: later\ndata: {"type":"later"}\n\n')
    mock = upstream(lambda request: httpx.Response(200, content=wire,
                                                    headers={"content-type": "text/event-stream"}))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers=AUTH, json={**BODY, "stream": True})
    assert "response.incomplete" in response.text
    assert "upstream_truncated" not in response.text and "event: later" not in response.text
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_nonstream_incomplete_is_unchanged_and_not_learned(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    mock = upstream(lambda request: httpx.Response(200, json={
        "id": "resp_incomplete", "object": "response", "status": "incomplete", "output": [],
    }))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers=AUTH, json=BODY)
    assert response.status_code == 200 and response.json()["status"] == "incomplete"
    assert not server._decision_store
    await mock.aclose()


@pytest.mark.anyio
async def test_responses_session_proceed_ignores_global_pin(client, monkeypatch):
    configure_three_tiers(monkeypatch)
    prompt_hash = server._prompt_hash("Proceed")
    server._decision_store[prompt_hash] = {
        "prompt_hash": prompt_hash, "decision": "cheap", "pin_until": __import__("time").time() + 60,
    }
    monkeypatch.setattr(server, "_decide_cached", lambda prompt: ("cheap", None, 1, 10))
    mock = upstream(lambda request: httpx.Response(200, json={
        "id": "resp_session", "object": "response", "status": "completed", "output": [],
    }))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/responses", headers={**AUTH, "X-Route-Session": "s"},
                                 json={"model": "auto", "input": "Proceed"})
    assert response.status_code == 200
    assert server._decision_store[prompt_hash]["decision"] == "cheap"
    # Session success is stored separately and the global pin remains untouched.
    assert len(server._decision_store) == 1 and server._session_state
    await mock.aclose()



def test_gateway_three_tier_configuration_points_at_cloudflare(monkeypatch):
    cheap, middle, expensive = configure_three_tiers(monkeypatch)
    host = "unified-ai-gateway.siddsantham.workers.dev"
    assert host in cheap["base"] and host in middle["base"] and host in expensive["base"]
    assert expensive["model"] == "openai/gpt-5.6-sol"
    assert middle["model"] == "openai/gpt-5.6-terra"


def test_middle_reasoning_body_preserves_cloudflare_model():
    body = {"model": "auto", "messages": [{"role": "user", "content": "refactor"}]}
    outgoing = server._build_outgoing_body(body, {
        **server.MIDDLE,
        "base": "https://unified-ai-gateway.siddsantham.workers.dev/v1",
        "model": "openai/gpt-5.6-terra", "effort": "max",
    })
    assert outgoing["model"] == "openai/gpt-5.6-terra" and outgoing["reasoning_effort"] == "max"
