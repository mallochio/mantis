"""Regression tests for web_search_options passthrough and citation surfacing."""

from types import SimpleNamespace
from typing import Any

import api
import serve
from fastapi.testclient import TestClient

_HEADERS = {"Authorization": "Bearer test-key"}
_WEB_SEARCH_OPTIONS = {"search_context_size": "medium"}
_ANNOTATIONS = [
    {
        "type": "url_citation",
        "url_citation": {"url": "https://example.test/source", "title": "source"},
    }
]
_CITATIONS = ["https://example.test/source"]


class _ProviderResponse:
    status_code = 200
    text = ""

    def __init__(self, message: dict[str, Any]) -> None:
        self._body = {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


def _chat(
    monkeypatch, replies: list[dict[str, Any]], web_search_options: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Drive a deterministic Worker -> Verifier(accept) run through the HTTP API."""
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_EXPERIMENTAL_MODES", "1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "openrouter/test-worker")
    monkeypatch.setattr(serve, "_args", None)
    roles = iter([("Worker", 0), ("Verifier", 0)])
    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(
            route=lambda *_a, **_k: dict(
                zip(("role_name", "agent_id"), next(roles), strict=True)
            )
        ),
    )
    posted: list[dict[str, Any]] = []
    queue = list(replies)

    def post(_url: str, **kwargs: Any) -> _ProviderResponse:
        posted.append(kwargs["json"])
        return _ProviderResponse(queue.pop(0))

    # These tests exercise request-body passthrough, not transport; pin buffered.
    monkeypatch.setattr(serve, "_upstream_streaming_enabled", lambda: False)
    monkeypatch.setattr(serve, "_provider_client", SimpleNamespace(post=post))
    payload: dict[str, Any] = {
        "model": "mantis/trinity",
        "messages": [{"role": "user", "content": "latest mantis news?"}],
    }
    if web_search_options is not None:
        payload["web_search_options"] = web_search_options
    response = TestClient(api.app).post("/v1/chat/completions", headers=_HEADERS, json=payload)
    assert response.status_code == 200
    return response.json(), posted


def test_web_search_options_reach_worker_request_body(monkeypatch):
    _body, posted = _chat(
        monkeypatch,
        replies=[{"content": "grounded answer"}, {"content": "ACCEPT"}],
        web_search_options=_WEB_SEARCH_OPTIONS,
    )
    assert posted[0]["web_search_options"] == _WEB_SEARCH_OPTIONS
    assert "web_search_options" not in posted[1]


def test_worker_annotations_and_citations_surface_on_final_message(monkeypatch):
    body, _posted = _chat(
        monkeypatch,
        replies=[
            {"content": "grounded answer", "annotations": _ANNOTATIONS, "citations": _CITATIONS},
            {"content": "ACCEPT"},
        ],
        web_search_options=_WEB_SEARCH_OPTIONS,
    )
    message = body["choices"][0]["message"]
    assert message["content"] == "grounded answer"
    assert message["annotations"] == _ANNOTATIONS
    assert message["citations"] == _CITATIONS


def test_no_web_search_options_means_no_annotations_or_citations(monkeypatch):
    body, posted = _chat(
        monkeypatch,
        replies=[{"content": "plain answer"}, {"content": "ACCEPT"}],
    )
    message = body["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "plain answer"}
    assert "annotations" not in message
    assert "citations" not in message
    assert all("web_search_options" not in request for request in posted)
