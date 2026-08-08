"""Parity tests for per-request usage.cost (USD), like OpenRouter."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import serve


class _PricesResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


def _run_with_usage() -> serve.NativeRun:
    run = serve.NativeRun("cost")
    run.add_usage(
        {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}, model="model-a"
    )
    return run


def test_add_usage_tracks_per_model_tokens():
    run = _run_with_usage()
    run.add_usage(
        {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}, model="model-b"
    )
    run.add_usage(
        {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}, model="model-a"
    )
    run.add_usage({"prompt_tokens": 9, "completion_tokens": 9, "total_tokens": 18})
    run.add_usage(None, model="model-c")
    assert run.usage_models == {
        "model-a": {"prompt_tokens": 9, "completion_tokens": 11},
        "model-b": {"prompt_tokens": 1, "completion_tokens": 2},
    }


def test_add_usage_aggregates_totals_as_before():
    run = serve.NativeRun("usage")
    run.add_usage(None)
    run.add_usage(
        {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
            "completion_tokens_details": {"reasoning_tokens": 2},
        },
        model="model-a",
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
    assert run.usage_models == {"model-a": {"prompt_tokens": 2, "completion_tokens": 3}}


def test_provider_response_records_bare_model_id(monkeypatch):
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }

    run = serve.NativeRun("provider")
    serve._history_context.active_run = run
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setattr(
        serve, "_provider_client", SimpleNamespace(post=lambda *_a, **_k: Response())
    )
    try:
        serve._provider_response("openrouter/openai/gpt-4o|high", [], 10, 0.7)
    finally:
        serve._history_context.active_run = None
    assert run.usage == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    # No provider prefix, no effort suffix in the attribution key.
    assert run.usage_models == {"openai/gpt-4o": {"prompt_tokens": 1, "completion_tokens": 2}}


def test_price_map_fetches_once_and_parses_payload(monkeypatch):
    calls = []

    def get(url: str, timeout: float) -> _PricesResponse:
        calls.append((url, timeout))
        return _PricesResponse(
            {
                "data": [
                    {
                        "id": "model-a",
                        "pricing": {"prompt": "0.000001", "completion": "0.000002"},
                    },
                    {"id": "model-b", "pricing": {"prompt": 0.000003, "completion": None}},
                    "junk",
                ]
            }
        )

    monkeypatch.setattr(serve.httpx, "get", get)
    monkeypatch.setattr(serve, "_price_cache", None)
    first = serve._price_map()
    second = serve._price_map()
    assert first is second
    assert first == {"model-a": (0.000001, 0.000002)}
    assert calls == [(serve._PRICES_URL, 5.0)]


def test_price_map_empty_on_any_fetch_failure(monkeypatch):
    def fail(*_a: Any, **_k: Any) -> None:
        raise serve.httpx.ConnectError("offline")

    monkeypatch.setattr(serve.httpx, "get", fail)
    monkeypatch.setattr(serve, "_price_cache", None)
    assert serve._price_map() == {}
    assert serve._price_map() == {}  # failure is cached; no second fetch
    for payload in (["not", "a", "dict"], {"data": {"not": "a list"}}):
        monkeypatch.setattr(
            serve.httpx, "get", lambda *_a, _p=payload, **_k: _PricesResponse(_p)
        )
        monkeypatch.setattr(serve, "_price_cache", None)
        assert serve._price_map() == {}


def test_completion_response_reports_cost(monkeypatch):
    monkeypatch.setattr(
        serve, "_price_map", lambda: {"model-a": (0.001, 0.002), "model-b": (0.003, 0.004)}
    )
    run = _run_with_usage()
    run.add_usage(
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, model="model-b"
    )
    messages = [{"role": "user", "content": "hi"}]
    body = serve._completion_response("mantis", messages, run, {"type": "final", "text": "answer"})
    expected = 4 * 0.001 + 6 * 0.002 + 5 * 0.003 + 7 * 0.004
    assert body["usage"]["cost"] == round(expected, 6)
    # Tokens are per-request (the client's context), while cost is the run's
    # cumulative orchestration spend.
    assert body["usage"]["prompt_tokens"] == 1
    assert body["usage"]["completion_tokens"] == 2
    assert body["usage"]["total_tokens"] == 3


def test_cost_omitted_when_any_consumed_model_is_unpriced(monkeypatch):
    monkeypatch.setattr(serve, "_price_map", lambda: {"model-a": (0.001, 0.002)})
    run = _run_with_usage()
    run.add_usage(
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}, model="model-b"
    )
    body = serve._completion_response("mantis", [], run, {"type": "final", "text": "answer"})
    assert "cost" not in body["usage"]


def test_cost_omitted_without_recorded_models(monkeypatch):
    monkeypatch.setattr(serve, "_price_map", lambda: {"model-a": (0.001, 0.002)})
    run = serve.NativeRun("cost")
    body = serve._completion_response("mantis", [], run, {"type": "final", "text": "answer"})
    assert "cost" not in body["usage"]


def test_cost_omitted_when_price_fetch_fails(monkeypatch):
    def fail(*_a: Any, **_k: Any) -> None:
        raise serve.httpx.ConnectError("offline")

    monkeypatch.setattr(serve.httpx, "get", fail)
    monkeypatch.setattr(serve, "_price_cache", None)
    run = _run_with_usage()
    body = serve._completion_response("mantis", [], run, {"type": "final", "text": "answer"})
    assert serve._price_map() == {}
    assert "cost" not in body["usage"]


def test_completion_usage_ignores_accumulated_run_usage():
    """A resumable run accumulates usage across internal steps and tool rounds.
    The response must report this request's context, not the accumulated total,
    or context-tracking clients (prime-agent harness) compact repeatedly."""
    run = serve.NativeRun("acc")
    for _ in range(6):
        run.add_usage(
            {"prompt_tokens": 42000, "completion_tokens": 1000, "total_tokens": 43000}
        )
    messages = [{"role": "user", "content": "hi"}]
    body = serve._completion_response("mantis", messages, run, {"type": "final", "text": "answer"})
    assert body["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    assert run.usage["total_tokens"] == 6 * 43000  # run accounting is untouched


def test_streaming_include_usage_chunk_carries_cost(monkeypatch):
    """api._sse streams the final chunk with {"usage": body["usage"]} verbatim, so the
    body dict from serve._completion_response must already carry the cost."""
    monkeypatch.setattr(serve, "_price_map", lambda: {"model-a": (0.000001, 0.000002)})
    run: Any = _run_with_usage()
    body = serve._completion_response("mantis", [], run, {"type": "final", "text": "answer"})
    assert body["usage"]["cost"] == round(4 * 0.000001 + 6 * 0.000002, 6)
