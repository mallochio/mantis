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
