import asyncio
import json
import threading

import httpx
import pytest

import pseudo_label
import server

AUTH = {"Authorization": "Bearer sk-route-local"}
BODY = {"model": "auto", "messages": [{"role": "user", "content": "hardening-case"}]}


@pytest.fixture
async def client(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_decide", lambda prompt: ("cheap", 0.1, None, None))
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "decisions.log")
    monkeypatch.setattr(server, "OUTCOME_LOG_PATH", tmp_path / "outcomes.log")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://router") as value:
        yield value


@pytest.mark.anyio
async def test_production_lifespan_owns_and_closes_client(monkeypatch):
    made = []
    class FakeClient:
        closed = False
        async def aclose(self):
            self.closed = True
    def factory(*args, **kwargs):
        value = FakeClient()
        made.append(value)
        return value
    monkeypatch.setattr(server.httpx, "AsyncClient", factory)
    monkeypatch.setattr(server, "_load_supra", lambda: (object(), object()))
    async with server.lifespan(server.app):
        assert server._READY
        assert server._client is made[0]
    assert made[0].closed
    assert server._client is None
    assert not server._READY


@pytest.mark.anyio
async def test_nonstream_refusal_falls_back(client, monkeypatch):
    calls = 0
    def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "I cannot assist"}, "finish_reason": "stop"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json=BODY)
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert response.headers["x-route-fallback"] == "true"
    assert calls == 2
    await mock.aclose()


@pytest.mark.anyio
async def test_stream_refusal_falls_back_before_output(client, monkeypatch):
    calls = 0
    refusal = b'data: {"choices":[{"message":{"content":"I cannot assist"},"finish_reason":"stop"}]}\n\n'
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=refusal if calls == 1 else success)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True})
    assert calls == 2
    assert "cannot assist" not in response.text.lower()
    assert "ok" in response.text and response.text.count("[DONE]") == 1
    await mock.aclose()


@pytest.mark.anyio
async def test_sse_comments_multiline_data_and_crlf_are_preserved(client, monkeypatch):
    first = (b': provider-comment\r\n'
             b'data: {\r\n'
             b'data: "choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\r\n\r\n')
    wire = first + b'data: [DONE]\r\n\r\n'
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wire)))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True})
    assert response.content == wire
    await mock.aclose()


@pytest.mark.anyio
async def test_identical_idempotent_requests_coalesce(client, monkeypatch):
    calls = 0
    entered = asyncio.Event()
    release = asyncio.Event()
    async def handler(request):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"choices": [{"message": {"content": "once"}, "finish_reason": "stop"}]})
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    headers = {**AUTH, "Idempotency-Key": "coalesce"}
    first = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json=BODY))
    await entered.wait()
    second = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json=BODY))
    await asyncio.sleep(0)
    release.set()
    one, two = await asyncio.gather(first, second)
    assert one.status_code == two.status_code == 200
    assert calls == 1
    assert two.headers["x-route-coalesced"] == "true"
    await mock.aclose()


@pytest.mark.anyio
async def test_stream_cache_buffer_is_capped(client, monkeypatch):
    calls = 0
    event = b'data: {"choices":[{"delta":{"content":"' + b'x' * 120 + b'"},"finish_reason":"stop"}]}\n\n'
    wire = event + b'data: [DONE]\n\n'
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=wire)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    monkeypatch.setattr(server, "RESP_CACHE_MAX_BYTES", 80)
    headers = {**AUTH, "Idempotency-Key": "too-large"}
    for _ in range(2):
        response = await client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True})
        assert response.status_code == 200
    assert calls == 2
    assert not server._resp_cache
    await mock.aclose()


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.blocked = asyncio.Event()
        self.closed = False
    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        self.blocked.set()
        await asyncio.Event().wait()
    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_downstream_cancellation_closes_upstream(client, monkeypatch):
    stream = BlockingStream()
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)))
    monkeypatch.setattr(server, "_client", mock)
    headers = {**AUTH, "Idempotency-Key": "cancel-during-stream"}
    task = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True}))
    await stream.blocked.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed
    assert not server._inflight
    await mock.aclose()


def test_journal_recovers_both_pseudo_label_outputs(tmp_path):
    output = tmp_path / "labels.jsonl"
    supra = tmp_path / "supra.jsonl"
    item = {"hash": "h", "record": {"hash": "h", "domain": "General"}, "supra": {"target": "x"}}
    pseudo_label.write_batch(output, supra, [item])
    output.write_text("corrupt\n")
    hashes = pseudo_label.recover_outputs(output, supra)
    assert hashes == {"h"}
    assert json.loads(output.read_text()) == item["record"]
    assert json.loads(supra.read_text()) == item["supra"]


@pytest.mark.anyio
async def test_error_attempt_uses_internal_immutable_occurrence(client, monkeypatch, tmp_path):
    log = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(server, "LOG_PATH", log)
    def handler(request):
        raise httpx.ConnectError("offline", request=request)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    supplied = "caller-controlled"
    response = await client.post("/v1/chat/completions", headers={**AUTH, "x-request-id": supplied}, json=BODY)
    assert response.status_code == 502
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 3
    assert [row["record_type"] for row in rows].count("attempt") == 2
    assert [row["record_type"] for row in rows].count("decision") == 1
    assert len({row["occurrence_id"] for row in rows}) == 1
    assert rows[0]["occurrence_id"] != supplied
    assert all(row["request_id"] == supplied for row in rows)
    await mock.aclose()


def test_cache_replacement_byte_accounting(monkeypatch):
    body = BODY
    one = json.dumps({"choices": [{"message": {"content": "a"}, "finish_reason": "stop"}]}).encode()
    two = json.dumps({"choices": [{"message": {"content": "longer"}, "finish_reason": "stop"}]}).encode()
    server._cache_put("key", body, one)
    server._cache_put("key", body, two)
    assert server._cache_bytes == len(two)
    assert len(server._resp_cache) == 1



def test_journal_truncates_uncommitted_tail(tmp_path):
    output = tmp_path / "labels.jsonl"
    supra = tmp_path / "supra.jsonl"
    item = {"hash": "h", "record": {"hash": "h"}, "supra": {"target": "x"}}
    pseudo_label.write_batch(output, supra, [item])
    journal = pseudo_label._journal_path(output)
    committed = journal.stat().st_size
    with journal.open("ab") as handle:
        handle.write(b'{"record":')
    assert pseudo_label.recover_outputs(output, supra) == {"h"}
    assert journal.stat().st_size == committed


@pytest.mark.anyio
async def test_realistic_delta_refusal_falls_back(client, monkeypatch):
    calls = 0
    # Refusal prose beginning in the first content chunk is caught before any
    # bytes escape and fails over. (A leading role-only chunk now commits
    # immediately so normal/sparse streams are never starved.)
    refusal = (b'data: {"choices":[{"delta":{"content":"I cannot"},"finish_reason":null}]}\n\n'
               b'data: {"choices":[{"delta":{"content":" assist with that"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=refusal if calls == 1 else success)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True})
    assert calls == 2
    assert "cannot" not in response.text.lower()
    assert "ok" in response.text
    await mock.aclose()


@pytest.mark.anyio
async def test_stream_content_filter_falls_back(client, monkeypatch):
    calls = 0
    filtered = (b'data: {"choices":[{"delta":{},"finish_reason":"content_filter"}]}\n\n')
    success = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
               b'data: [DONE]\n\n')
    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=filtered if calls == 1 else success)
    mock = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True})
    assert calls == 2
    assert "content_filter" not in response.text
    assert response.text.count("[DONE]") == 1
    await mock.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("first,second", [
    (b"\n", b"\n"), (b"\r", b"\r"), (b"\r\n", b"\r\n"),
    (b"\r\n", b"\n"), (b"\n", b"\r\n"), (b"\r\n", b"\r"),
    (b"\r", b"\r\n"), (b"\n", b"\r"),
])
async def test_sse_line_ending_matrix_preserved(client, monkeypatch, first, second):
    finish = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}' + first + second
    done = b'data: [DONE]' + first + second
    wire = finish + done
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wire)))
    monkeypatch.setattr(server, "_client", mock)
    response = await client.post("/v1/chat/completions", headers=AUTH, json={**BODY, "stream": True})
    assert response.content == wire
    await mock.aclose()


class PrefetchBlockingStream(httpx.AsyncByteStream):
    def __init__(self):
        self.started = asyncio.Event()
        self.closed = False
    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b""
    async def aclose(self):
        self.closed = True


@pytest.mark.anyio
async def test_keyed_cancel_during_prefetch_cleans_owner_and_allows_retry(client, monkeypatch):
    blocked = PrefetchBlockingStream()
    first_mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=blocked)))
    monkeypatch.setattr(server, "_client", first_mock)
    headers = {**AUTH, "Idempotency-Key": "cancel-prefetch"}
    task = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True}))
    await blocked.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert blocked.closed
    assert not server._inflight
    await first_mock.aclose()
    wire = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')
    retry_mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wire)))
    monkeypatch.setattr(server, "_client", retry_mock)
    retry = await client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True})
    assert retry.status_code == 200 and "[DONE]" in retry.text
    assert not server._inflight
    await retry_mock.aclose()


@pytest.mark.anyio
async def test_keyed_cancel_before_stream_iteration_cleans_owner_and_retries(client, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    def blocked_decide(prompt):
        started.set()
        release.wait(2)
        return "cheap", 0.1, None, None
    monkeypatch.setattr(server, "_decide", blocked_decide)
    headers = {**AUTH, "Idempotency-Key": "cancel-routing"}
    task = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True}))
    await asyncio.to_thread(started.wait, 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert not server._inflight
    monkeypatch.setattr(server, "_decide", lambda prompt: ("cheap", 0.1, None, None))
    wire = (b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n')
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=wire)))
    monkeypatch.setattr(server, "_client", mock)
    retry = await client.post("/v1/chat/completions", headers=headers, json={**BODY, "stream": True})
    assert retry.status_code == 200
    assert not server._inflight
    await mock.aclose()


def test_legacy_migration_missing_projection_fails_without_changes(tmp_path):
    output = tmp_path / "labels.jsonl"
    supra = tmp_path / "supra.jsonl"
    original = '{"hash":"h"}\n'
    output.write_text(original)
    with pytest.raises(RuntimeError, match="missing"):
        pseudo_label.recover_outputs(output, supra)
    assert output.read_text() == original
    assert not supra.exists()
    assert not pseudo_label._journal_path(output).exists()


def test_legacy_migration_mismatched_counts_fails_without_changes(tmp_path):
    output = tmp_path / "labels.jsonl"
    supra = tmp_path / "supra.jsonl"
    output.write_text('{"hash":"one"}\n{"hash":"two"}\n')
    supra.write_text('{"target":"one"}\n')
    before = output.read_bytes(), supra.read_bytes()
    with pytest.raises(RuntimeError, match="line counts"):
        pseudo_label.recover_outputs(output, supra)
    assert (output.read_bytes(), supra.read_bytes()) == before
    assert not pseudo_label._journal_path(output).exists()


@pytest.mark.anyio
async def test_chat_prefetch_holds_role_until_payload(client, monkeypatch):
    role = b'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}\n\n'
    payload = b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n'

    async def role_then_payload():
        yield role
        yield payload

    prefix, failure = await asyncio.wait_for(
        server._prefetch_sse(role_then_payload(), asyncio.get_running_loop().time() + 30), timeout=1)
    assert failure is None and role in prefix and payload in prefix
