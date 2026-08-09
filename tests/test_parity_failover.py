"""Failover tests for openfugu-patch/serve.py: transient provider errors fail
over to the next pool worker, switching model/endpoint per spec while
messages, tools, and controls stay identical across attempts."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import serve

POOL = "openrouter/gpt-alpha,opencode-go/deepseek-beta"


class _Response:
    """Minimal httpx.Response stand-in, matching tests/test_serve.py."""

    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self.body = body if body is not None else {"error": "boom"}
        self.text = json.dumps(self.body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://provider.test/chat/completions")
            response = httpx.Response(self.status_code, text=self.text, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )

    def json(self) -> dict:
        return self.body


class _StreamResponse:
    """Context-manager stand-in for httpx's stream() response."""

    def __init__(self, response: _Response) -> None:
        self._response = response

    def __enter__(self) -> _StreamResponse:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        self._response.raise_for_status()

    def iter_lines(self):
        body = self._response.body
        message = (body.get("choices") or [{}])[0].get("message", {})
        delta: dict[str, Any] = {"role": "assistant"}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [
                {
                    "index": i,
                    "id": call.get("id"),
                    "type": "function",
                    "function": call.get("function", {}),
                }
                for i, call in enumerate(message["tool_calls"])
            ]
        yield "data: " + json.dumps({"choices": [{"delta": delta}]})
        if body.get("usage"):
            yield "data: " + json.dumps({"usage": body["usage"]})
        yield "data: [DONE]"


class _Client:
    """Queue-backed replacement for serve._provider_client that records calls."""

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self.posts: list[dict] = []

    def _record(self, url, headers, json) -> object:
        self.posts.append({"url": url, "headers": dict(headers or {}), "json": json})
        item = self._items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def post(self, url, headers=None, json=None):
        return self._record(url, headers, json)

    def stream(self, _method, url, headers=None, json=None):
        return _StreamResponse(self._record(url, headers, json))


@pytest.fixture
def pool_env(monkeypatch):
    monkeypatch.setattr(serve, "_args", None)
    monkeypatch.setattr(serve, "_FAILOVER_DELAY", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-key")
    monkeypatch.setenv("MANTIS_WORKER_MODELS", POOL)
    monkeypatch.delenv("MANTIS_WORKER_MODEL", raising=False)
    return monkeypatch


def _messages() -> list[dict]:
    return [{"role": "user", "content": "hi"}]


def test_transient_error_fails_over_to_next_pool_worker(pool_env, monkeypatch):
    usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    ok = {"choices": [{"message": {"content": "ok"}}], "usage": usage}
    client = _Client([_Response(500), _Response(200, ok)])
    monkeypatch.setattr(serve, "_provider_client", client)
    run = serve.NativeRun("failover")
    serve._history_context.active_run = run
    try:
        data = serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    finally:
        serve._history_context.active_run = None
    assert data["choices"][0]["message"]["content"] == "ok"
    # One attempt per spec, in declared pool order.
    assert [post["url"] for post in client.posts] == [
        "https://openrouter.ai/api/v1/chat/completions",
        "https://opencode.ai/zen/go/v1/chat/completions",
    ]
    assert client.posts[0]["headers"]["Authorization"] == "Bearer or-key"
    assert client.posts[1]["headers"]["Authorization"] == "Bearer oc-key"
    # Each attempt targets its own spec's model; messages are identical.
    assert client.posts[0]["json"]["model"] == "gpt-alpha"
    assert client.posts[1]["json"]["model"] == "deepseek-beta"
    assert client.posts[0]["json"]["messages"] == client.posts[1]["json"]["messages"] == _messages()
    # Usage is recorded once, from the successful attempt only.
    assert run.usage["prompt_tokens"] == 3
    assert run.usage["completion_tokens"] == 5
    assert run.usage["total_tokens"] == 8


def test_failover_switches_from_responses_to_chat_protocol(pool_env, monkeypatch):
    monkeypatch.setenv(
        "MANTIS_WORKER_MODELS",
        "openrouter/openai/gpt-alpha,opencode-go/deepseek-beta",
    )
    ok = {"choices": [{"message": {"content": "ok"}}]}
    client = _Client([_Response(500), _Response(200, ok)])
    monkeypatch.setattr(serve, "_provider_client", client)

    data = serve._provider_response("openrouter/openai/gpt-alpha", _messages(), 10, 0.7)

    assert data["choices"][0]["message"]["content"] == "ok"
    assert client.posts[0]["url"].endswith("/responses")
    assert "input" in client.posts[0]["json"]
    assert "stream_options" not in client.posts[0]["json"]
    assert client.posts[1]["url"].endswith("/chat/completions")
    assert "messages" in client.posts[1]["json"]


def test_responses_stream_failure_fails_over(pool_env, monkeypatch):
    monkeypatch.setenv(
        "MANTIS_WORKER_MODELS",
        "openrouter/openai/gpt-alpha,opencode-go/deepseek-beta",
    )
    results = iter(
        [
            RuntimeError("provider stream error: failed response"),
            {"choices": [{"message": {"content": "fallback"}}], "usage": None},
        ]
    )

    def complete(*_args, **_kwargs):
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(serve, "_stream_completion", complete)
    data = serve._provider_response("openrouter/openai/gpt-alpha", _messages(), 10, 0.7)
    assert data["choices"][0]["message"]["content"] == "fallback"


def test_exhausted_pool_raises_naming_every_attempt(pool_env, monkeypatch):
    client = _Client([_Response(429), _Response(429, {"error": "also busy"})])
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError) as excinfo:
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    message = str(excinfo.value)
    assert "openrouter/gpt-alpha" in message
    assert "opencode-go/deepseek-beta" in message
    assert message.count("HTTP 429") == 2
    assert len(client.posts) == 2


def test_non_transient_status_raises_immediately(pool_env, monkeypatch):
    client = _Client([_Response(400, {"error": "bad request"})])
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError, match="openrouter/gpt-alpha returned HTTP 400"):
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    assert len(client.posts) == 1
    assert client.posts[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"


def test_single_entry_pool_makes_one_attempt(pool_env, monkeypatch):
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "openrouter/gpt-alpha")
    client = _Client([_Response(503)])
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError, match="openrouter/gpt-alpha") as excinfo:
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    assert "HTTP 503" in str(excinfo.value)
    assert len(client.posts) == 1


def test_connection_error_fails_over(pool_env, monkeypatch):
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    client = _Client(
        [
            httpx.ConnectError("refused", request=request),
            _Response(200, {"choices": [{"message": {"content": "ok"}}]}),
        ]
    )
    monkeypatch.setattr(serve, "_provider_client", client)
    data = serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    assert data["choices"][0]["message"]["content"] == "ok"
    assert len(client.posts) == 2


def test_exhausted_pool_names_transport_failures(pool_env, monkeypatch):
    request = httpx.Request("POST", "https://provider.test/chat/completions")
    client = _Client(
        [
            httpx.ConnectError("refused", request=request),
            httpx.ReadTimeout("timed out", request=request),
        ]
    )
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError) as excinfo:
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    message = str(excinfo.value)
    assert "openrouter/gpt-alpha: ConnectError" in message
    assert "opencode-go/deepseek-beta: ReadTimeout" in message
    assert len(client.posts) == 2


def test_failover_target_without_key_is_skipped(pool_env, monkeypatch):
    monkeypatch.delenv("OPENCODE_API_KEY")
    client = _Client([_Response(502)])
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError) as excinfo:
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    message = str(excinfo.value)
    assert "openrouter/gpt-alpha: HTTP 502" in message
    assert "OPENCODE_API_KEY is required for opencode-go/deepseek-beta" in message
    assert len(client.posts) == 1


def test_failover_attempts_helper(pool_env, monkeypatch):
    # Assigned spec first, remaining pool specs in declared order, no duplicates.
    assert serve._failover_attempts("opencode-go/deepseek-beta") == [
        "opencode-go/deepseek-beta",
        "openrouter/gpt-alpha",
    ]
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "openrouter/gpt-alpha,openrouter/gpt-alpha")
    assert serve._failover_attempts("openrouter/gpt-alpha") == ["openrouter/gpt-alpha"]
    # Bare labels and unparseable entries are not routable failover targets.
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "gpt-5,not-a-spec,opencode-go/deepseek-beta")
    assert serve._failover_attempts("openrouter/gpt-alpha") == [
        "openrouter/gpt-alpha",
        "opencode-go/deepseek-beta",
    ]
    # No configured pool (upstream defaults are bare labels) means no failover.
    monkeypatch.delenv("MANTIS_WORKER_MODELS")
    assert serve._failover_attempts("openrouter/gpt-alpha") == ["openrouter/gpt-alpha"]
    # A broken pool configuration falls back to the assigned spec alone.
    monkeypatch.setenv("MANTIS_WORKER_MODELS", " , ,")
    assert serve._failover_attempts("openrouter/gpt-alpha") == ["openrouter/gpt-alpha"]


def test_unconfigured_pool_behaves_as_before(pool_env, monkeypatch):
    monkeypatch.delenv("MANTIS_WORKER_MODELS")
    ok = {"choices": [{"message": {"content": "ok"}}]}
    client = _Client([_Response(500), _Response(200, ok)])
    monkeypatch.setattr(serve, "_provider_client", client)
    with pytest.raises(RuntimeError, match="openrouter/gpt-alpha"):
        serve._provider_response("openrouter/gpt-alpha", _messages(), 10, 0.7)
    assert len(client.posts) == 1
