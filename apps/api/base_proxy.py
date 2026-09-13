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
* endpoint-bound encrypted reasoning is stripped; message history is
  otherwise forwarded verbatim so the reusable prefix stays identical
  across turns and prompt-cache hits survive.
* provider-native prompt-cache markers are applied for explicit dialects.

The efficient tier rides the self-hosted Modal GLM-5.3-Flash endpoint at
high reasoning effort and the capable tier rides Modal Kimi-K3 at max
effort, both over the ``openai_chat`` wire format. Both tiers are
Switchyard-servable, so there are no direct LiteLLM legs.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
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
        families = {
            providers._model_cache_family(m)
            for m in (
                route.efficient.upstream_model,
                route.capable.upstream_model,
            )
        }
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


def routed_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300")))


@dataclass(frozen=True)
class RoutedResponse:
    body: dict[str, Any]
    selected_model: str | None
    session_id: str | None


def _post_routed(
    body: dict[str, Any],
    headers: dict[str, str],
    client_factory: Callable[[], httpx.Client] | None = None,
    timeout_s: float | None = None,
) -> httpx.Response:
    url = os.environ.get("MANTIS_ROUTER_URL", DEFAULT_ROUTER_URL) + "/chat/completions"
    with (client_factory or router_client)() as client:
        kwargs = {"timeout": timeout_s} if timeout_s is not None else {}
        return client.post(url, headers=headers, json=body, **kwargs)


async def _post_routed_with_deadline(
    body: dict[str, Any],
    headers: dict[str, str],
    client_factory: Callable[[], httpx.AsyncClient] | None,
    timeout_s: float | None,
) -> httpx.Response:
    """Post and fully read a routed response within one wall-clock budget."""
    url = os.environ.get("MANTIS_ROUTER_URL", DEFAULT_ROUTER_URL) + "/chat/completions"
    async with (client_factory or routed_async_client)() as client:
        async with asyncio.timeout(timeout_s):
            return await client.post(url, headers=headers, json=body, timeout=timeout_s)


def dispatch_routed(
    body: dict[str, Any],
    session: str | None,
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
    timeout_s: float | None = None,
) -> RoutedResponse:
    """Send one non-streaming OpenAI chat body through Switchyard."""
    body = sanitize_routed_body(body)
    headers: dict[str, str] = {}
    key = os.environ.get("MANTIS_ROUTER_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if session:
        headers[SWITCHYARD_SESSION_HEADER] = session
    upstream = asyncio.run(_post_routed_with_deadline(body, headers, client_factory, timeout_s))
    upstream.raise_for_status()
    payload = upstream.json()
    if not isinstance(payload, dict) or "choices" not in payload:
        raise ValueError("router returned a non-chat payload")
    return RoutedResponse(
        body=payload,
        selected_model=upstream.headers.get(SWITCHYARD_SELECTED_MODEL_HEADER),
        session_id=upstream.headers.get(SWITCHYARD_SESSION_HEADER) or session,
    )


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
    switchyard_session = headers.get(SWITCHYARD_SESSION_HEADER)
    if switchyard_session:
        return switchyard_session
    session = (
        headers.get("x-route-session")
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


_DROP = object()
def _metadata_key(key: str) -> str:
    return "".join(character for character in key.lower() if character.isalnum())


def _is_endpoint_bound_key(key: str) -> bool:
    normalized = _metadata_key(key)
    return (
        "encrypted" in normalized
        or "compaction" in normalized
        or normalized.endswith("signature")
        or normalized == "providermetadata"
    )


def _portable_message_value(value: Any) -> Any:
    """Recursively remove opaque provider state while preserving chat/tool data."""
    if isinstance(value, list):
        return [clean for item in value if (clean := _portable_message_value(item)) is not _DROP]
    if not isinstance(value, dict):
        return value
    for discriminator in ("type", "format"):
        kind = value.get(discriminator)
        if isinstance(kind, str) and any(
            word in kind.lower() for word in ("encrypted", "compaction")
        ):
            return _DROP
    removed_opaque = False
    cleaned: dict[str, Any] = {}
    for key, item in value.items():
        if key.startswith("_") or _is_endpoint_bound_key(key):
            removed_opaque = True
            continue
        portable = _portable_message_value(item)
        if portable is not _DROP and not (key == "reasoning_details" and portable == []):
            cleaned[key] = portable
    if removed_opaque and set(cleaned) <= {"type", "format"}:
        return _DROP
    return cleaned


def _strip_endpoint_bound_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Recursively drop endpoint-bound state from dynamically routed messages."""
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    body["messages"] = [
        clean for message in messages if (clean := _portable_message_value(message)) is not _DROP
    ]
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


def router_body(request: BaseChatRequest) -> dict[str, Any]:
    """Build the outbound body with a stable, cache-friendly prefix.

    Harness reasoning controls are dropped so the catalog tier governs
    effort; endpoint-bound items are stripped while message history is
    otherwise forwarded verbatim. Older reasoning is deliberately kept:
    rewriting history every turn moves the prefix divergence point and
    forces a full context prefill on Modal, which times out past ~60s on
    long sessions and surfaces as a silently stopped conversation.
    """
    body = request.model_dump(exclude_none=True, exclude={"user", "metadata"})
    return sanitize_routed_body(body)


def sanitize_routed_body(body: dict[str, Any]) -> dict[str, Any]:
    """Copy and sanitize any body crossing the dynamic routing boundary."""
    body = {key: value for key, value in body.items() if not key.startswith("_")}
    messages = body.get("messages")
    if isinstance(messages, list):
        body["messages"] = [
            {key: value for key, value in message.items() if not key.startswith("_")}
            if isinstance(message, dict)
            else message
            for message in messages
        ]
    body = _drop_client_reasoning(body)
    body = _coerce_max_completion_tokens(body)
    body = _strip_endpoint_bound_reasoning(body)
    return _apply_base_cache_markers(body)


def router_response_headers(upstream: httpx.Response, session: str | None = None) -> dict[str, str]:
    mapped: dict[str, str] = {}
    selected = upstream.headers.get(SWITCHYARD_SELECTED_MODEL_HEADER)
    if selected:
        mapped["x-route-model"] = selected
    stable_session = upstream.headers.get(SWITCHYARD_SESSION_HEADER) or session
    if stable_session:
        mapped[SWITCHYARD_SESSION_HEADER] = stable_session
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

        # Some providers (e.g. LiteLLM) return an SSE error stream for a failed
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
        # Upstream hung up mid-stream (e.g. Modal closing a long-prefill
        # stream). End the stream truncated WITHOUT a [DONE] marker so the
        # OpenAI client raises a connection error and retries instead of
        # treating the partial turn as a clean stop with no tool calls.
        logger.warning("upstream stream closed early: %s", exc)
        return
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
    stable_session = outbound_headers.get(SWITCHYARD_SESSION_HEADER)
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
                headers={
                    "X-Request-Id": request_id,
                    **router_response_headers(upstream, stable_session),
                },
            ),
            True,
        )
    upstream = _post_routed(outbound_body, outbound_headers, lambda: client)
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
            headers={
                "X-Request-Id": request_id,
                **router_response_headers(upstream, stable_session),
            },
        ),
        False,
    )
