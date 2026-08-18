import httpx
import pytest

import server

AUTH = {"Authorization": "Bearer sk-route-local"}

@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_decide", lambda prompt: ("cheap", 0.1, None, None))
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    monkeypatch.setattr(server, "TRAINING_LOG_PATH", tmp_path / "training.log")
    monkeypatch.setattr(server, "DECISION_STORE_PATH", tmp_path / "decision-state.jsonl")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://router") as value:
        yield value


def upstream(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=1)


@pytest.mark.anyio
async def test_nonstream_failover_and_telemetry(client, monkeypatch):
    calls = []
    def handler(request):
        calls.append(request.url.host)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert response.headers["x-route-fallback"] == "true"
    assert response.headers["x-route-attempts"] == "2"
    assert len(calls) == 2
    await mock.aclose()


@pytest.mark.anyio
async def test_stream_abrupt_eof_is_error_without_done(client, monkeypatch):
    payload = b'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}\n\n'
    mock = upstream(lambda request: httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"}))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert "upstream_truncated" in response.text
    assert "data: [DONE]" not in response.text
    await mock.aclose()


@pytest.mark.anyio
async def test_stream_stops_at_done(client, monkeypatch):
    payload = (b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n'
               b'data: {"should":"not appear"}\n\n')
    mock = upstream(lambda request: httpx.Response(200, content=payload))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hello"}]})
    assert response.text.count("[DONE]") == 1
    assert "should" not in response.text
    await mock.aclose()


@pytest.mark.anyio
async def test_body_limit_without_content_length(client, monkeypatch):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 20)
    response = await client.post("/v1/chat/completions", headers={**AUTH, "transfer-encoding": "chunked"}, content=b'{' + b' ' * 50 + b'}')
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.anyio
async def test_validation_is_openai_shaped(client):
    response = await client.post("/v1/chat/completions", headers=AUTH, json={"model": "wrong", "messages": []})
    assert response.status_code == 400
    assert set(response.json()["error"]) == {"message", "type", "param", "code"}


@pytest.mark.anyio
async def test_cache_requires_key_and_excludes_tools(client, monkeypatch):
    count = 0
    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    body = {"model": "auto", "messages": [{"role": "user", "content": "unique-normal"}]}
    await client.post("/v1/chat/completions", headers=AUTH, json=body)
    await client.post("/v1/chat/completions", headers=AUTH, json=body)
    assert count == 2
    keyed = {**AUTH, "Idempotency-Key": "safe-key"}
    first = await client.post("/v1/chat/completions", headers=keyed, json={**body, "seed": 3})
    second = await client.post("/v1/chat/completions", headers=keyed, json={**body, "seed": 3})
    assert first.headers.get("x-route-cache") is None
    assert second.headers["x-route-cache"] == "hit"
    assert count == 3
    await mock.aclose()


@pytest.mark.anyio
async def test_session_affinity_keeps_continuation_on_current_tier(client, monkeypatch):
    decisions = iter([("expensive", None, 4, 10), ("cheap", None, 1, 10)])
    monkeypatch.setattr(server, "_decide", lambda *args: next(decisions))
    calls = []
    def handler(request):
        payload = __import__("json").loads(request.content)
        calls.append(payload["model"])
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    headers = {**AUTH, "X-Route-Session": "coding-1"}
    first = await client.post("/v1/chat/completions", headers=headers,
                              json={"model": "auto", "messages": [{"role": "user", "content": "implement"}]})
    second = await client.post("/v1/chat/completions", headers=headers,
                               json={"model": "auto", "messages": [{"role": "user", "content": "Proceed"}]})
    assert first.headers["x-route-decision"] == "expensive"
    assert second.headers["x-route-decision"] == "expensive"
    assert second.headers["x-route-reason"] == "continuation_sticky"
    assert len(calls) == 2
    await mock.aclose()


def test_usage_cache_metrics_are_normalized(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "decisions.log")
    server._log("middle", None, "kimi-k3", "hello", None,
                usage={"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 75}})
    row = __import__("json").loads((tmp_path / "decisions.log").read_text().strip())
    assert row["fresh_tokens"] == 25 and row["cache_hit_ratio"] == 0.75


@pytest.mark.anyio
async def test_middle_failover_uses_middle_model(client, monkeypatch):
    monkeypatch.setattr(server, "_decide", lambda *args: ("cheap", None, 1, 10))
    calls = []
    def handler(request):
        payload = __import__("json").loads(request.content)
        calls.append((request.url.host, payload["model"]))
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "busy"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    mock = upstream(handler)
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH,
                                 json={"model": "auto", "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert calls[0][1] == server.BACKENDS["cheap"]["model"] and calls[1][1] == server.BACKENDS["middle"]["model"]
    assert response.headers["x-route-decision"] == "middle"
    await mock.aclose()


@pytest.mark.anyio
async def test_list_models_returns_aliases_and_backends(client):
    response = await client.get("/v1/models")
    assert response.status_code == 200
    data = response.json().get("data", [])
    model_ids = {item["id"] for item in data}
    assert "base" in model_ids
    assert "mantis/base" in model_ids
    assert "mantis/fusion" in model_ids
    for target in server.BACKENDS:
        assert target in model_ids


@pytest.mark.anyio
async def test_stream_done_without_explicit_finish_reason_completes_cleanly(client, monkeypatch):
    payload = (b'data: {"choices":[{"delta":{"content":"completed"}}]}\n\n'
               b'data: [DONE]\n\n')
    mock = upstream(lambda request: httpx.Response(200, content=payload, headers={"content-type": "text/event-stream"}))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH,
                                 json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hello"}]})
    assert response.status_code == 200
    assert "completed" in response.text
    assert "upstream_truncated" not in response.text
    assert "data: [DONE]" in response.text
    await mock.aclose()

