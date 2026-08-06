"""Progressive SSE streaming parity tests."""

import json
from types import SimpleNamespace

import api
import pytest
import serve
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    return TestClient(api.app)


def _headers():
    return {"Authorization": "Bearer test-key"}


def _run(**overrides):
    run = SimpleNamespace(
        run_id="b" * 32,
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        response_metadata={},
        validate_output=lambda _text: None,
    )
    for key, value in overrides.items():
        setattr(run, key, value)
    return run


def _stream(client, monkeypatch, event, run, **request):
    monkeypatch.setattr(serve, "create_run", lambda *_a: run)
    monkeypatch.setattr(serve, "_advance_to_boundary", lambda *_a: event)
    monkeypatch.setattr(serve, "delete_run", lambda *_a: True)
    return client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            **request,
        },
    )


def _events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: {")
    ]


def _deltas(response):
    return [event["choices"][0]["delta"] for event in _events(response) if event["choices"]]


def test_long_answer_streams_progressive_content_deltas(client, monkeypatch):
    answer = "the quick brown fox jumps over the lazy dog. " * 12
    response = _stream(client, monkeypatch, {"type": "final", "text": answer}, _run())
    assert response.status_code == 200
    deltas = _deltas(response)
    content_deltas = [delta["content"] for delta in deltas if "content" in delta]
    assert len(content_deltas) > 1
    assert "".join(content_deltas) == answer
    assert all(len(part) <= 64 for part in content_deltas)
    offset = 0
    for part in content_deltas[:-1]:
        offset += len(part)
        assert answer[offset] in " \n\t"
    assert deltas[0].get("role") == "assistant"
    assert all("role" not in delta for delta in deltas[1:])
    events = _events(response)
    for event in events[:-1]:
        assert event["choices"][0]["finish_reason"] is None
    assert events[-1]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert response.text.endswith("data: [DONE]\n\n")


def test_reasoning_deltas_precede_content(client, monkeypatch):
    reasoning = "checked the claim twice before answering"
    details = [{"type": "reasoning.summary", "summary": "verified"}]
    run = _run(response_metadata={"reasoning": reasoning, "reasoning_details": details})
    response = _stream(client, monkeypatch, {"type": "final", "text": "verified answer"}, run)
    assert response.status_code == 200
    deltas = _deltas(response)
    assert deltas[0] == {"role": "assistant", "reasoning": reasoning}
    assert deltas[1] == {"reasoning_details": details}
    content_deltas = [delta for delta in deltas[2:] if "content" in delta]
    assert content_deltas
    assert "".join(delta["content"] for delta in content_deltas) == "verified answer"
    assert all("reasoning" not in delta for delta in content_deltas)
    assert all("role" not in delta for delta in deltas[1:])


def test_stream_include_usage_appends_usage_chunk(client, monkeypatch):
    run = _run()
    response = _stream(
        client,
        monkeypatch,
        {"type": "final", "text": "a slightly longer answer to chunk over"},
        run,
        stream_options={"include_usage": True},
    )
    assert response.status_code == 200
    events = _events(response)
    assert events[-1]["choices"] == []
    assert events[-1]["usage"] == run.usage
    assert events[-2]["choices"][0]["finish_reason"] == "stop"
    assert response.text.endswith("data: [DONE]\n\n")


def test_tool_call_stream_stays_single_delta(client, monkeypatch):
    event = {
        "type": "tool_calls",
        "tool_calls": [{"id": "call-1", "name": "read", "arguments": {"path": "README.md"}}],
    }
    response = _stream(client, monkeypatch, event, _run())
    assert response.status_code == 200
    deltas = _deltas(response)
    assert len(deltas) == 2
    tool_delta = deltas[0]
    assert tool_delta["role"] == "assistant"
    assert "content" not in tool_delta
    assert len(tool_delta["tool_calls"]) == 1
    call = tool_delta["tool_calls"][0]
    assert call["index"] == 0
    assert call["function"]["name"] == "read"
    assert json.loads(call["function"]["arguments"]) == {"path": "README.md"}
    assert deltas[1] == {}
    events = _events(response)
    assert events[-1]["choices"] == [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
