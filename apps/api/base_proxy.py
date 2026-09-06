"""Thin forwarder from ``mantis/base`` to the local Switchyard stage router.

Switchyard is the only complexity router. It scores tool-trajectory signals
(severity, spinning, exploring, production intensity), de-escalates settled
turns, and falls open to ``efficient_first``. This module never picks a tier:
it forwards the chat body, a stable session id, and hygiene edits that keep
the reusable prefix identical across turns.

Body hygiene (catalog supremacy + prefix stability):

* harness ``reasoning`` / ``reasoning_effort`` are dropped so the catalog
  tier governs effort via Switchyard per-target ``extra_body`` (Switchyard
  ``merge_extra_body`` uses ``or_insert``: request fields would win).
* endpoint-bound encrypted reasoning is stripped; old thinking is trimmed.
* provider-native prompt-cache markers are applied for explicit dialects.

Capable Grok rides the Bedrock OpenAI-compatible endpoint
(``bedrock-runtime.../openai/v1``) with a bearer ``BEDROCK_API_KEY`` over the
``openai_responses`` wire format, which is where Bedrock reports cache reads.
Both tiers are Switchyard-servable, so there are no direct LiteLLM legs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator
from functools import lru_cache
from typing import Any, Protocol

import httpx
import providers
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("mantis.base_proxy")

SWITCHYARD_SESSION_HEADER = "x-switchyard-session-id"
SWITCHYARD_SELECTED_MODEL_HEADER = "x-model-router-selected-model"
DEFAULT_ROUTER_URL = "http://127.0.0.1:5500/v1"


def _base_route_families() -> frozenset[str]:
    """Return the set of explicit cache dialects used by the [base] route.

    Only ``anthropic`` and ``openai`` need explicit markers. Models with
    automatic prefix caching (kimi, deepseek, gemini, ...) contribute
    nothing. A mixed ``kimi + gpt`` route therefore yields ``{"openai"}``
    so the GPT leg still gets breakpoints instead of disabling all markers.
    """
    route = _load_base_route()
    if route is None:
        return frozenset()
    try:
        families = {providers._model_cache_family(m) for m in (
            route.efficient.upstream_model,
            route.capable.upstream_model,
        )}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.debug("base route cache family not loaded: %s", error)
        return frozenset()
    return frozenset(f for f in families if f in {"anthropic", "openai"})


@lru_cache(maxsize=1)
def _load_base_route() -> Any | None:
    """Load and cache the parsed [base] route from the active catalog."""
    try:
        from switchyard_config import load_switchyard_route

        return load_switchyard_route(None)
    except Exception as error:  # noqa: BLE001 - catalog may fail in many ways
        logger.debug("base route not loaded: %s", error)
    return None


class BaseChatRequest(Protocol):
    metadata: dict[str, Any] | None
    user: str | None
    stream: bool
    messages: Any
    tools: Any

    def model_dump(self, *, exclude_none: bool, exclude: set[str]) -> dict[str, Any]: ...


def router_client() -> httpx.Client:
    return httpx.Client(timeout=float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300")))


def session_id(headers: dict[str, str], body: BaseChatRequest) -> str | None:
    """Session identity for Switchyard stage-router stickiness.

    Accepts the Switchyard header, the legacy ``X-Route-Session`` client
    header, the harness ``x-opencode-session`` / ``x-mantis-session-id``
    headers, or ``metadata.session_id`` / ``user`` in the body. When the
    harness sends a shared header such as ``opencode``, combine it with a
    synthetic hash of the first user turn so different conversations do not
    share one Switchyard session and evict each other's prefix cache.
    Only ``x-switchyard-session-id`` is sent upstream.
    """
    session = (
        headers.get(SWITCHYARD_SESSION_HEADER)
        or headers.get("x-route-session")
        or headers.get("x-opencode-session")
        or headers.get("x-mantis-session-id")
        or headers.get("x-mantis-session")
    )
    if session:
        synthetic = _synthetic_session_id(body.messages, body.tools)
        if synthetic and synthetic != session:
            return f"{session}:{synthetic}"
        return session
    if isinstance(body.metadata, dict):
        for key in ("session_id", "sessionId"):
            value = body.metadata.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body.user, str) and body.user:
        return body.user
    return _synthetic_session_id(body.messages, body.tools)


def router_headers(headers: dict[str, str], body: BaseChatRequest) -> dict[str, str]:
    """Build the upstream headers: router auth plus stable session identity."""
    out: dict[str, str] = {}
    key = os.environ.get("MANTIS_ROUTER_KEY")
    if key:
        out["Authorization"] = f"Bearer {key}"
    session = session_id(headers, body)
    if session:
        out[SWITCHYARD_SESSION_HEADER] = session
    return out


def _drop_client_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Drop harness-supplied reasoning controls so the catalog tier governs.

    Harnesses send their own thinking level on every turn (Prime Agent
    defaults to ``medium``), which must not override the tier budget.
    Switchyard ``merge_extra_body`` inserts target defaults only when the
    request omits the key, so any client value would win over the catalog.
    Both the structured ``reasoning`` object and the flat
    ``reasoning_effort`` are removed here; each [base] target declares its
    own effort in ``extra_body``. Dropping (rather than coercing) also
    removes a 400 source on strict providers.
    """
    body.pop("reasoning", None)
    body.pop("reasoning_effort", None)
    return body


def _coerce_max_completion_tokens(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``max_completion_tokens`` to ``max_tokens`` for Switchyard.

    Some clients send ``max_completion_tokens`` (OpenAI o1/o3 style) while
    the Switchyard extra_body supplies ``max_tokens``. Sending both to a
    provider can produce a 400, so we collapse the client value into the
    standard ``max_tokens`` key and let the request override the catalog.
    """
    if "max_completion_tokens" in body:
        value = body.pop("max_completion_tokens")
        if "max_tokens" not in body and value is not None:
            body["max_tokens"] = value
    return body


def _strip_endpoint_bound_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Drop encrypted reasoning that cannot cross Switchyard model targets.

    OpenAI-compatible reasoning payloads may contain endpoint-bound encrypted
    items. The stage router can switch between efficient and capable targets,
    so replaying those items to a different target yields an upstream 404.
    Plain reasoning summaries remain useful and portable.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for message in messages:
        if isinstance(message, dict):
            providers._portable_reasoning_details(message)
    return body


def _apply_base_cache_markers(body: dict[str, Any]) -> dict[str, Any]:
    """Add provider-native prompt-cache markers to the outgoing chat body.

    Markers are the union of explicit dialects in the [base] route. A mixed
    ``kimi + gpt`` route still marks the GPT leg; a ``claude + gpt`` route
    marks both (Anthropic uses content blocks, OpenAI uses a message key,
    so they do not conflict). Models with automatic caching need nothing.
    """
    families = _base_route_families()
    messages = body.get("messages")
    if not families or not isinstance(messages, list) or len(messages) < 2:
        return body
    if not providers._cache_breakpoints_enabled():
        return body
    if "anthropic" in families:
        messages = providers._with_cache_breakpoints(messages)
    if "openai" in families:
        messages = providers._with_openai_cache_breakpoints(messages)
    body["messages"] = messages
    return body


def _user_texts(messages: Any) -> list[str]:
    """Return the text of user turns; outbound bodies are always dicts."""
    texts: list[str] = []
    if not isinstance(messages, list):
        return texts
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = " ".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ).strip()
        else:
            text = ""
        if text:
            texts.append(text)
    return texts


def _synthetic_session_id(messages: Any, tools: Any = None) -> str | None:
    """Hash the stable conversation prefix when the harness sends no session."""
    texts = _user_texts(messages)
    if not texts:
        return None
    names: list[str] = []
    if isinstance(tools, list):
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function")
            name = fn.get("name") if isinstance(fn, dict) else tool.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    payload = "\n".join([texts[0], *sorted(set(names))])
    return "auto-" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _trim_old_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Drop reasoning from old assistant turns to protect prefix-cache hits.

    Keeps the last two assistant messages intact. Older traces bloat the
    prefix and break cache reuse across providers. Best-effort: returns the
    body unchanged when trimming is unavailable.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return body
    try:
        body["messages"] = providers._clear_thinking(messages, keep=2)
    except Exception:  # noqa: BLE001 - trimming is best-effort
        return body
    return body


def router_body(request: BaseChatRequest) -> dict[str, Any]:
    """Build the outbound body with a stable, cache-friendly prefix.

    Harness reasoning controls are dropped so the catalog tier governs
    effort; messages are trimmed of old reasoning and endpoint-bound items
    so the reusable prefix stays identical across turns.
    """
    body = request.model_dump(exclude_none=True, exclude={"user", "metadata"})
    body = _drop_client_reasoning(body)
    body = _coerce_max_completion_tokens(body)
    body = _strip_endpoint_bound_reasoning(body)
    body = _trim_old_reasoning(body)
    return _apply_base_cache_markers(body)


def router_response_headers(upstream: httpx.Response) -> dict[str, str]:
    mapped: dict[str, str] = {}
    selected = upstream.headers.get(SWITCHYARD_SELECTED_MODEL_HEADER)
    if selected:
        mapped["x-route-model"] = selected
    session = upstream.headers.get(SWITCHYARD_SESSION_HEADER)
    if session:
        mapped[SWITCHYARD_SESSION_HEADER] = session
    return mapped


def _coerce_status(raw: Any, fallback: int) -> int:
    """Coerce an embedded error code to an int HTTP status."""
    try:
        status = int(raw)
    except (TypeError, ValueError):
        return fallback
    return status if 100 <= status <= 599 else fallback


def router_error(upstream: httpx.Response) -> JSONResponse:
    """Return an OpenAI-compatible error from a failed router response.

    Switchyard usually returns JSON, but streaming or low-level failures may
    produce an SSE error stream or a plain text body. We log the raw response
    and attempt to surface the most useful message.
    """
    try:
        body = upstream.json()
        if isinstance(body, dict) and "error" in body:
            error = body["error"]
            status = upstream.status_code
            if isinstance(error, dict):
                # Switchyard and some providers embed the HTTP code in the
                # payload while returning a 200. Trust the embedded code so the
                # response is treated as an error by OpenAI clients.
                if "type" not in error:
                    error["type"] = "upstream_error"
                if "status_code" not in error:
                    error["status_code"] = error.get("code", status)
                status = _coerce_status(error.get("code"), status)
            return JSONResponse(body, status_code=status)
        # Router returned JSON but not an error object; wrap it.
        return JSONResponse(
            {"error": {"message": json.dumps(body), "type": "upstream_error"}},
            status_code=upstream.status_code,
        )
    except (ValueError, httpx.ResponseNotRead):
        text = ""
        with contextlib.suppress(Exception):
            text = upstream.text
        logger.warning(
            "router returned non-json error status=%s content_type=%s body=%r",
            upstream.status_code,
            upstream.headers.get("content-type"),
            text[:1000],
        )
        message = text.strip() or f"router returned status {upstream.status_code}"

        # Some providers (e.g. Bifrost) return an SSE error stream for a failed
        # chat request. Look for an ``error`` field in the first data line.
        if message.startswith("data:"):
            for line in message.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line.removeprefix("data:").strip()
                if payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except ValueError:
                    continue
                if isinstance(data, dict) and isinstance(data.get("error"), dict):
                    return JSONResponse(data, status_code=upstream.status_code)
                if isinstance(data, dict) and data.get("error"):
                    message = str(data["error"])
                    break

        body = {
            "error": {
                "message": message[:500],
                "type": "upstream_error",
                "status_code": upstream.status_code,
            }
        }
    return JSONResponse(body, status_code=upstream.status_code)


def _router_stream(
    client: httpx.Client,
    stream: Any,
    upstream: httpx.Response,
    on_close: Callable[[], None] | None = None,
) -> Iterator[bytes]:
    try:
        yield from upstream.iter_bytes()
    except httpx.HTTPError as exc:
        # Upstream hung up mid-stream; stop the response cleanly instead of
        # letting the transport error crash the whole ASGI server.
        logger.warning("upstream stream closed early: %s", exc)
        # Emit a terminating SSE frame so clients see a clean end.
        yield b"data: [DONE]\n\n"
    finally:
        with contextlib.suppress(Exception):
            stream.__exit__(None, None, None)
        with contextlib.suppress(Exception):
            client.close()
        if on_close is not None:
            on_close()


def forward(
    request: BaseChatRequest,
    headers: dict[str, str],
    request_id: str,
    client_factory: Any,
    on_close: Callable[[], None] | None = None,
) -> tuple[JSONResponse | StreamingResponse, bool]:
    """Forward a Base chat request to Switchyard, which picks the tier.

    Builds a stable session identity and a cache-friendly body, then sends
    the turn to the ``mantis/base`` stage router. Returns
    ``(response, handed_off)``; ``handed_off`` is True for live streams.
    """
    outbound_body = router_body(request)
    outbound_headers = router_headers(headers, request)
    client = client_factory()
    url = os.environ.get("MANTIS_ROUTER_URL", DEFAULT_ROUTER_URL) + "/chat/completions"
    if request.stream:
        stream = client.stream("POST", url, headers=outbound_headers, json=outbound_body)
        upstream = stream.__enter__()
        if upstream.is_error:
            error = router_error(upstream)
            stream.__exit__(None, None, None)
            client.close()
            return error, False
        return (
            StreamingResponse(
                _router_stream(client, stream, upstream, on_close=on_close),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
            ),
            True,
        )
    with client:
        upstream = client.post(url, headers=outbound_headers, json=outbound_body)
    try:
        body = upstream.json()
    except ValueError:
        body = None
    if upstream.is_error or (isinstance(body, dict) and body.get("error")):
        return router_error(upstream), False
    if not isinstance(body, dict) or "choices" not in body:
        logger.warning(
            "router returned non-chat payload status=%s body=%r",
            upstream.status_code,
            str(body)[:500],
        )
        return (
            JSONResponse(
                {
                    "error": {
                        "message": "router returned a non-chat payload",
                        "type": "upstream_error",
                        "status_code": upstream.status_code,
                    }
                },
                status_code=502,
            ),
            False,
        )
    return (
        JSONResponse(
            body,
            headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
        ),
        False,
    )
