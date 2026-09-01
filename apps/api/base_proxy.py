"""HTTP proxy from ``mantis/base`` to the local Switchyard stage router.

The Switchyard route id is ``mantis/base``, matching the public model name, so
this module forwards the chat body without rewriting ``model``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tomllib
from collections import Counter
from collections.abc import Callable, Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("mantis.base_proxy")

SWITCHYARD_SESSION_HEADER = "x-switchyard-session-id"
SWITCHYARD_SELECTED_MODEL_HEADER = "x-model-router-selected-model"
GROK_CONV_HEADER = "x-grok-conv-id"
DEFAULT_ROUTER_URL = "http://127.0.0.1:5500/v1"


def _model_cache_family(model: str) -> str | None:
    """Return the prompt-cache dialect for an upstream model id."""
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-5.6-") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    return None


def _with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the stable prefix for prompt caching (Anthropic models).

    Adds ``cache_control: {type: ephemeral}`` to the last block of the first
    system message and to the message just before the final user/turn, so the
    reusable system+history prefix is cached and only the new tail is billed.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return messages
    out = [dict(message) for message in messages]

    def _mark(message: dict[str, Any]) -> None:
        content = message.get("content")
        if content is None:
            return
        if isinstance(content, list):
            blocks = list(content)
        else:
            blocks = [{"type": "text", "text": str(content)}]
        if blocks and isinstance(blocks[-1], dict) and "cache_control" not in blocks[-1]:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            message["content"] = blocks

    if out and out[0].get("role") == "system":
        _mark(out[0])
    if len(out) >= 3:
        _mark(out[-2])
    return out


def _with_openai_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark stable layers for OpenAI GPT-5.6+ explicit prompt caching."""
    out: list[dict[str, Any]] = []
    last_system_index = -1
    for i, message in enumerate(messages):
        if message.get("role") == "system":
            last_system_index = i
    for i, message in enumerate(messages):
        msg = dict(message)
        if i == last_system_index:
            msg["prompt_cache_breakpoint"] = {"mode": "explicit"}
        if i == len(messages) - 2 and len(messages) >= 2:
            msg["prompt_cache_breakpoint"] = {"mode": "explicit"}
        out.append(msg)
    return out


def _catalog_path() -> Path:
    configured = os.environ.get("AI_ROUTING_CONFIG") or os.environ.get("MANTIS_CATALOG_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".config" / "ai-routing" / "catalog.toml"


@lru_cache(maxsize=1)
def _load_base_route() -> Any | None:
    """Load and cache the parsed [base] route from the active catalog."""
    try:
        with open(_catalog_path(), "rb") as f:
            root = tomllib.load(f)
        from model_catalog_schema import load_base_route

        return load_base_route(root)
    except Exception as error:  # noqa: BLE001 - catalog may fail in many ways
        logger.debug("base route not loaded: %s", error)
    return None


@lru_cache(maxsize=1)
def _base_route_family() -> str | None:
    """Determine the prompt-cache family for the current [base] route.

    Returns ``anthropic`` or ``openai`` when both base targets are the same
    family, or ``None`` when the catalog cannot be read or the targets are
    mixed/unknown (e.g., Grok and Claude in the same route).  In the mixed
    case we avoid cache markers rather than risk sending Anthropic blocks to
    xAI or dropping cache controls on a Claude call.
    """
    route = _load_base_route()
    if route is None:
        return None
    try:
        models = (route.efficient.upstream_model, route.capable.upstream_model)
        families = {_model_cache_family(m) for m in models}
        if len(families) == 1 and None not in families:
            return families.pop()
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.debug("base route cache family not loaded: %s", error)
    return None


def _base_efficient_model() -> str | None:
    """Return the upstream model id for the current [base] efficient target."""
    route = _load_base_route()
    if route is None:
        return None
    try:
        return route.efficient.upstream_model  # type: ignore[no-any-return]
    except AttributeError:
        return None


class BaseChatRequest(Protocol):
    metadata: dict[str, Any] | None
    user: str | None
    stream: bool

    def model_dump(self, *, exclude_none: bool, exclude: set[str]) -> dict[str, Any]: ...


def router_client() -> httpx.Client:
    return httpx.Client(timeout=float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300")))


def session_id(headers: dict[str, str], body: BaseChatRequest) -> str | None:
    """Session identity for Switchyard stage-router stickiness.

    Accepts the Switchyard header, the legacy ``X-Route-Session`` client
    header, or ``metadata.session_id`` / ``user`` in the body. Only
    ``x-switchyard-session-id`` is sent upstream.
    """
    session = headers.get(SWITCHYARD_SESSION_HEADER) or headers.get("x-route-session")
    if session:
        return session
    if isinstance(body.metadata, dict):
        for key in ("session_id", "sessionId"):
            value = body.metadata.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body.user, str) and body.user:
        return body.user
    return None


_ESCALATE_ENV = "MANTIS_BASE_ESCALATE_ON_FAILURE"
_REPEAT_THRESHOLD_ENV = "MANTIS_BASE_ESCALATE_REPEATS"
_ERROR_MARKERS = (
    "error",
    "traceback",
    "exception",
    "command failed",
    "no such file",
    "is not defined",
)


def _escalation_enabled() -> bool:
    return os.environ.get(_ESCALATE_ENV, "1").lower() not in {"0", "false", "no", "off"}


def _repeat_threshold() -> int:
    try:
        return max(2, int(os.environ.get(_REPEAT_THRESHOLD_ENV, "3")))
    except ValueError:
        return 3


def failure_signals(messages: Any) -> int:
    """Count identical tool calls or identical errors repeated past the threshold."""
    if not isinstance(messages, list):
        return 0
    counts: Counter[tuple[str, str, str]] = Counter()
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                function = (call or {}).get("function") or {}
                name = str(function.get("name") or "")
                arguments = str(function.get("arguments") or "")
                counts[("call", name, arguments)] += 1
        elif role == "tool":
            content = message.get("content")
            if not isinstance(content, str):
                continue
            lowered = content.strip().lower()
            if any(marker in lowered for marker in _ERROR_MARKERS):
                counts[("error", "", lowered[:200])] += 1
    threshold = _repeat_threshold()
    return sum(1 for occurrences in counts.values() if occurrences >= threshold)


def escalation_suffix(body: BaseChatRequest) -> str:
    """Return the session-id suffix that forces a fresh tier decision."""
    if not _escalation_enabled():
        return ""
    signals = failure_signals(getattr(body, "messages", None))
    return f"#esc{signals}" if signals else ""


def _capable_provider_info() -> tuple[str, str, str] | None:
    """Return (base_url, credential_env, upstream_model) for the capable tier."""
    route = _load_base_route()
    if route is None:
        return None
    try:
        capable = route.capable
        provider = route.providers[capable.provider]
        return provider.base_url, provider.credential_env, capable.upstream_model
    except (AttributeError, KeyError, TypeError):
        return None


def router_headers(headers: dict[str, str], body: BaseChatRequest) -> dict[str, str]:
    out: dict[str, str] = {}
    key = os.environ.get("MANTIS_ROUTER_KEY")
    if key:
        out["Authorization"] = f"Bearer {key}"
    session = session_id(headers, body)
    if session:
        suffix = escalation_suffix(body)
        out[SWITCHYARD_SESSION_HEADER] = session + suffix
        if suffix:
            out["x-switchyard-force-tier"] = "capable"
            out["x-switchyard-escalated"] = "1"
        # xAI Grok routes prompt-cache state by conversation; pinning the same
        # conversation to the same server makes cache hits reliable. Other
        # providers ignore the custom header, so it is safe to forward whenever
        # we have a stable session identity.
        out[GROK_CONV_HEADER] = session
    return out


def _normalize_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Convert a structured ``reasoning`` object into ``reasoning_effort``.

    Clients such as Prime Agent send ``reasoning: {"effort": "low"}``.
    Switchyard's stage router only understands the flat ``reasoning_effort``
    string, so leaving the object in the body causes an "Invalid request
    payload" 400. Map ``reasoning.effort`` to ``reasoning_effort`` and drop
    the object. The top-level ``max_tokens`` already governs output length,
    so ``reasoning.max_tokens`` and ``reasoning.exclude`` are ignored here.
    """
    reasoning = body.pop("reasoning", None)
    if not isinstance(reasoning, dict):
        return body
    effort = reasoning.get("effort")
    if effort and "reasoning_effort" not in body:
        body["reasoning_effort"] = effort
    return body


def _coerce_base_reasoning(body: dict[str, Any]) -> dict[str, Any]:
    """Coerce the request's reasoning effort to a value the efficient target accepts.

    The base stage router can pick either the efficient or capable target.
    The efficient target is the most restrictive, so we coerce the requested
    level using that model's vocabulary. For example, ``kimi-k3`` does not
    accept ``medium`` or ``none``; we map them to ``high`` and ``low`` so the
    request does not fail with a 400 the moment efficient is selected.
    """
    effort = body.get("reasoning_effort")
    if effort is None:
        return body
    efficient = _base_efficient_model()
    if efficient is None:
        return body
    try:
        from providers import _coerce_reasoning_effort

        coerced = _coerce_reasoning_effort(efficient, effort)
    except Exception:  # noqa: BLE001 - coercion is best-effort, keep original body on failure
        return body
    if coerced is None:
        body.pop("reasoning_effort", None)
    else:
        body["reasoning_effort"] = coerced
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
        if not isinstance(message, dict):
            continue
        details = message.get("reasoning_details")
        if not isinstance(details, list):
            continue
        portable = [
            detail
            for detail in details
            if not isinstance(detail, dict)
            or not any(
                marker in str(detail.get(key, "")).lower()
                for key in ("type", "format")
                for marker in ("encrypted", "compaction")
            )
        ]
        if portable:
            message["reasoning_details"] = portable
        else:
            message.pop("reasoning_details", None)
    return body


def _apply_base_cache_markers(body: dict[str, Any]) -> dict[str, Any]:
    """Add provider-native prompt-cache markers to the outgoing chat body.

    The markers are chosen from the catalog [base] route so switching
    ``catalog.toml`` between Grok and Anthropic does not require code changes.
    """
    family = _base_route_family()
    messages = body.get("messages")
    if not family or not isinstance(messages, list) or len(messages) < 2:
        return body
    if family == "anthropic":
        body["messages"] = _with_cache_breakpoints(messages)
    elif family == "openai":
        body["messages"] = _with_openai_cache_breakpoints(messages)
    return body


def router_body(request: BaseChatRequest) -> dict[str, Any]:
    body = request.model_dump(exclude_none=True, exclude={"user", "metadata"})
    body = _normalize_reasoning(body)
    body = _coerce_base_reasoning(body)
    body = _coerce_max_completion_tokens(body)
    body = _strip_endpoint_bound_reasoning(body)
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
                status = error.get("code") or status
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


def _direct_capable_headers_and_body(
    headers: dict[str, str], body: dict[str, Any]
) -> tuple[dict[str, str], dict[str, Any]] | None:
    """Build headers/body for a direct capable-tier call, or None if unavailable."""
    info = _capable_provider_info()
    if info is None:
        return None
    base_url, credential_env, upstream_model = info
    api_key = os.environ.get(credential_env)
    if not api_key:
        return None
    direct_headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    # Preserve escalated routing signal for observability; providers ignore it.
    direct_headers["x-switchyard-force-tier"] = "capable"
    direct_headers["x-switchyard-escalated"] = "1"
    direct_body = dict(body)
    direct_body["model"] = upstream_model
    return direct_headers, direct_body


def forward(
    request: BaseChatRequest,
    headers: dict[str, str],
    request_id: str,
    client_factory: Any,
    on_close: Callable[[], None] | None = None,
) -> tuple[JSONResponse | StreamingResponse, bool]:
    """Proxy a Base chat request to Switchyard.

    When failure repetition is detected the request is sent directly to the
    capable tier so promotion does not depend on the Switchyard picker.
    Otherwise the request goes through Switchyard with its normal picker
    (now ``efficient_first``). Returns ``(response, handed_off)``.
    """
    outbound_headers = router_headers(headers, request)
    outbound_body = router_body(request)
    suffix = escalation_suffix(request)

    # Picker-independent promotion: if looping is detected, bypass Switchyard
    # and hit the capable provider directly. This makes efficient_first safe.
    if suffix:
        direct = _direct_capable_headers_and_body(outbound_headers, outbound_body)
        if direct is not None:
            direct_headers, direct_body = direct
            info = _capable_provider_info()
            assert info is not None
            base_url = info[0]
            url = base_url.rstrip("/") + "/chat/completions"
            client = client_factory()
            if request.stream:
                stream = client.stream("POST", url, headers=direct_headers, json=direct_body)
                upstream = stream.__enter__()
                if upstream.is_error:
                    # Capable direct failed — fall through to Switchyard.
                    try:
                        stream.__exit__(None, None, None)
                    except Exception:
                        pass
                    try:
                        client.close()
                    except Exception:
                        pass
                else:
                    return (
                        StreamingResponse(
                            _router_stream(client, stream, upstream, on_close=on_close),
                            media_type="text/event-stream",
                            headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
                        ),
                        True,
                    )
            else:
                with client:
                    upstream = client.post(url, headers=direct_headers, json=direct_body)
                try:
                    body = upstream.json()
                except ValueError:
                    body = None
                if not upstream.is_error and not (isinstance(body, dict) and body.get("error")):
                    return (
                        JSONResponse(
                            body,
                            headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
                        ),
                        False,
                    )
                # Direct capable errored — fall through to Switchyard below.
            # If we reach here direct path did not return, so continue to Switchyard.

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
    return (
        JSONResponse(
            body,
            headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
        ),
        False,
    )
