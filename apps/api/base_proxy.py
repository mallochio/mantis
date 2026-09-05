"""HTTP proxy from ``mantis/base`` to the local Switchyard stage router.

The Switchyard route id is ``mantis/base``, matching the public model name, so
this module forwards the chat body without rewriting ``model``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
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
SWITCHYARD_COMPLEXITY_HEADER = "x-switchyard-complexity"
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
        if i == last_system_index and "prompt_cache_breakpoint" not in msg:
            msg["prompt_cache_breakpoint"] = {"mode": "explicit"}
        if i == len(messages) - 2 and len(messages) >= 2 and "prompt_cache_breakpoint" not in msg:
            msg["prompt_cache_breakpoint"] = {"mode": "explicit"}
        out.append(msg)
    return out


def _cache_breakpoints_enabled() -> bool:
    return os.environ.get("MANTIS_CACHE_BREAKPOINTS", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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
        families = {_model_cache_family(m) for m in (
            route.efficient.upstream_model,
            route.capable.upstream_model,
        )}
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        logger.debug("base route cache family not loaded: %s", error)
        return frozenset()
    return frozenset(f for f in families if f in {"anthropic", "openai"})


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

    Kept for compatibility. Returns a family only when both targets share
    one explicit dialect. Prefer ``_base_route_families`` which keeps the
    usable dialect in mixed routes (e.g. ``kimi + gpt`` still caches GPT).
    """
    families = _base_route_families()
    if len(families) == 1:
        return next(iter(families))
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
    header, or ``metadata.session_id`` / ``user`` in the body. When the
    harness sends a shared header such as ``opencode``, combine it with a
    synthetic hash of the first user turn so different conversations do not
    share one Switchyard session and evict each other's prefix cache.
    Only ``x-switchyard-session-id`` is sent upstream.
    """
    session = (
        headers.get(SWITCHYARD_SESSION_HEADER)
        or headers.get("x-route-session")
        or headers.get("x-mantis-session-id")
        or headers.get("x-mantis-session")
    )
    if session:
        messages = getattr(body, "messages", None)
        tools = getattr(body, "tools", None)
        synthetic = _synthetic_session_id(messages, tools)
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
    messages = getattr(body, "messages", None)
    tools = getattr(body, "tools", None)
    return _synthetic_session_id(messages, tools)


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


def _salt_session_enabled() -> bool:
    """Return True when escalation must force a fresh Switchyard session.

    Defaults to disabled so looping tasks keep prefix-cache reuse and still
    promote to capable via ``x-switchyard-force-tier``. Set
    ``MANTIS_BASE_SALT_SESSION=1`` to restore the old salting behavior.
    """
    return os.environ.get("MANTIS_BASE_SALT_SESSION", "0").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def escalation_suffix(body: BaseChatRequest) -> str:
    """Return the session-id suffix that forces a fresh tier decision."""
    if not _escalation_enabled() or not _salt_session_enabled():
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
    except (AttributeError, KeyError, TypeError):
        return None
    else:
        return provider.base_url, provider.credential_env, capable.upstream_model


def router_headers(
    headers: dict[str, str],
    body: BaseChatRequest,
    *,
    complexity: str | None = None,
) -> dict[str, str]:
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
        elif _escalation_enabled() and failure_signals(getattr(body, "messages", None)):
            # Salt disabled: keep the session for prefix-cache reuse but still
            # promote to capable so a looping task does not stay on efficient.
            out["x-switchyard-force-tier"] = "capable"
            out["x-switchyard-escalated"] = "1"
        # xAI Grok routes prompt-cache state by conversation; pinning the same
        # conversation to the same server makes cache hits reliable. Other
        # providers ignore the custom header, so it is safe to forward whenever
        # we have a stable session identity.
        out[GROK_CONV_HEADER] = session
    # Complexity classifier hint for Switchyard: lets the stage router send
    # REASONING-tier requests directly to the capable target instead of
    # trying efficient first and escalating.
    if complexity and complexity in _COMPLEXITY_TIERS:
        out[SWITCHYARD_COMPLEXITY_HEADER] = complexity
        if complexity == "reasoning" and "x-switchyard-force-tier" not in out:
            out["x-switchyard-force-tier"] = "capable"
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

    Markers are the union of explicit dialects in the [base] route. A mixed
    ``kimi + gpt`` route still marks the GPT leg; a ``claude + gpt`` route
    marks both (Anthropic uses content blocks, OpenAI uses a message key,
    so they do not conflict). Models with automatic caching need nothing.
    """
    families = _base_route_families()
    messages = body.get("messages")
    if not families or not isinstance(messages, list) or len(messages) < 2:
        return body
    if not _cache_breakpoints_enabled():
        return body
    if "anthropic" in families:
        messages = _with_cache_breakpoints(messages)
    if "openai" in families:
        messages = _with_openai_cache_breakpoints(messages)
    body["messages"] = messages
    return body


# -- Complexity classifier --------------------------------------------------
#
# Lightweight prompt-based classifier inspired by LiteLLM's complexity_router.
# Classifies the last user message into one of four tiers so Switchyard can
# route REASONING-tier requests directly to the capable target.

_COMPLEXITY_TIERS = ("simple", "medium", "complex", "reasoning")

_REASONING_MARKERS = (
    "step by step",
    "think through",
    "reason about",
    "work through",
    "chain of thought",
    "let's think",
    "explain your reasoning",
    "show your work",
    "prove that",
    "derive",
    "analyze",
    "compare and contrast",
    "evaluate the tradeoffs",
    "what are the implications",
)

_TECHNICAL_MARKERS = (
    "algorithm",
    "complexity",
    "optimization",
    "architecture",
    "distributed",
    "concurrency",
    "deadlock",
    "race condition",
    "memory leak",
    "security vulnerability",
    "cryptograph",
    "differential equation",
    "gradient",
    "backpropag",
    "eigenvalue",
    "theorem",
    "proof",
    "formal verification",
)

_CODE_MARKERS = (
    "```",
    "def ",
    "class ",
    "function ",
    "import ",
    "SELECT ",
    "CREATE TABLE",
    "async ",
    "await ",
)

_MULTISTEP_MARKERS = (
    " then ",
    " after that ",
    " next ",
    " finally ",
    " first ",
    " second ",
    " third ",
    "1.",
    "2.",
    "3.",
)


def _classify_complexity(messages: list[dict[str, Any]]) -> str:
    """Return a complexity tier for the conversation's last user message.

    Tiers: ``simple``, ``medium``, ``complex``, ``reasoning``.

    The classifier uses cheap text heuristics (marker presence, token count,
    question density) rather than an LLM call, so it adds negligible latency.
    """
    # Find the last user message content.
    text = ""
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        break

    if not text:
        return "simple"

    lowered = text.lower()
    word_count = len(text.split())
    score = 0.0

    # Reasoning markers carry the strongest signal.
    reasoning_hits = sum(1 for m in _REASONING_MARKERS if m in lowered)
    if reasoning_hits >= 2:
        return "reasoning"
    score += reasoning_hits * 2.0

    # Technical vocabulary.
    tech_hits = sum(1 for m in _TECHNICAL_MARKERS if m in lowered)
    score += tech_hits * 1.5

    # Code presence.
    code_hits = sum(1 for m in _CODE_MARKERS if m in text)
    score += code_hits * 1.0

    # Multi-step patterns.
    step_hits = sum(1 for m in _MULTISTEP_MARKERS if m in text)
    score += step_hits * 0.5

    # Length contributes (longer prompts tend to be more complex).
    if word_count > 500:
        score += 2.0
    elif word_count > 200:
        score += 1.0
    elif word_count > 50:
        score += 0.5

    # Question density.
    question_count = text.count("?")
    if question_count >= 3:
        score += 1.5
    elif question_count >= 1:
        score += 0.5

    if score >= 6.0:
        return "reasoning"
    if score >= 3.0:
        return "complex"
    if score >= 1.0:
        return "medium"
    return "simple"


_TIER_RANK = {"simple": 0, "medium": 1, "complex": 2, "reasoning": 3}
_RANK_TIER = ("simple", "medium", "complex", "reasoning")
_SESSION_TIER_LIMIT = 1024
_session_tier_ranks: dict[str, int] = {}
_session_tier_lock = threading.Lock()


def _tier_rank(tier: str | None) -> int:
    return _TIER_RANK.get(tier or "", 0)


def _recent_turn_window() -> int:
    route = _load_base_route()
    try:
        window = int(getattr(route, "recent_turn_window", 3))
    except (AttributeError, TypeError, ValueError):
        return 3
    return max(1, window)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [_part_text(part) for part in content]
        return " ".join(p for p in parts if p).strip()
    if content is None:
        return ""
    return str(content).strip()


def _part_text(part: Any) -> str:
    text = part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
    return text if isinstance(text, str) else ""


def _message_role_content(msg: Any) -> tuple[Any, Any]:
    if isinstance(msg, dict):
        return msg.get("role"), msg.get("content")
    return getattr(msg, "role", None), getattr(msg, "content", None)


def _user_texts(messages: Any) -> list[str]:
    texts: list[str] = []
    if not isinstance(messages, list):
        return texts
    for msg in messages:
        role, content = _message_role_content(msg)
        if role != "user":
            continue
        text = _content_text(content)
        if text:
            texts.append(text)
    return texts


def _classify_window(messages: Any, window: int) -> str:
    texts = _user_texts(messages)[-max(1, window):]
    if not texts:
        return "simple"
    best = 0
    for text in texts:
        rank = _tier_rank(_classify_complexity([{"role": "user", "content": text}]))
        best = max(best, rank)
    return _RANK_TIER[best]


def _sticky_complexity(session: str | None, complexity: str) -> str:
    """Keep the max tier per session so a warm capable prefix is not dropped."""
    if not session:
        return complexity
    rank = _tier_rank(complexity)
    with _session_tier_lock:
        stored = _session_tier_ranks.get(session)
        if stored is None or rank > stored:
            while len(_session_tier_ranks) >= _SESSION_TIER_LIMIT:
                _session_tier_ranks.pop(next(iter(_session_tier_ranks)))
            _session_tier_ranks[session] = rank
            return complexity
        return _RANK_TIER[stored]


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
        from providers import _clear_thinking

        body["messages"] = _clear_thinking(messages, keep=2)
    except Exception:  # noqa: BLE001 - trimming is best-effort
        return body
    return body


def router_body(request: BaseChatRequest) -> dict[str, Any]:
    body = request.model_dump(exclude_none=True, exclude={"user", "metadata"})
    body = _normalize_reasoning(body)
    body = _coerce_base_reasoning(body)
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


# Providers Switchyard 0.2.0 cannot serve: it only speaks OpenAI-chat,
# OpenAI-responses, and Anthropic-messages wire formats with static bearer or
# x-api-key auth. Bedrock needs SigV4-signed Converse calls and Vertex needs
# OAuth, so the efficient leg for these adapters goes through LiteLLM
# directly instead of the Switchyard hop. Anything else keeps using the
# stage router (which also remains the escalation fallback below).
_DIRECT_LITELLM_ADAPTERS = frozenset({"bedrock", "vertex", "vertex_ai"})


def _direct_litellm_spec() -> Any | None:
    """Build a provider spec for the efficient target, or None.

    Returns None when there is no [base] route or when the efficient target
    rides a Switchyard-servable adapter.
    """
    import providers

    route = _load_base_route()
    if route is None:
        return None
    try:
        target = route.efficient
        binding = route.providers[target.provider]
    except (AttributeError, KeyError, TypeError):
        return None
    if binding.adapter not in _DIRECT_LITELLM_ADAPTERS:
        return None
    return providers.ResolvedModelSpec(
        adapter=binding.adapter,
        model=target.upstream_model,
        effort=target.reasoning_effort,
        base_url=binding.base_url,
        credential_env=binding.credential_env,
        binding=target.provider,
        protocols=tuple(binding.protocols),
        slot=None,
        max_tokens=target.max_tokens,
    )


def _direct_litellm_kwargs(body: dict[str, Any], resolved: Any) -> dict[str, Any]:
    """Render LiteLLM kwargs from an outbound chat body and resolved spec."""
    import providers

    # The catalog target budget wins over the client value (mirrors the
    # Switchyard extra_body override): reasoning targets burn most of a small
    # budget on thinking and would otherwise return empty content.
    max_tokens = resolved.max_tokens or body.get("max_tokens") or 64000
    controls = {
        key: body[key]
        for key in ("web_search_options", "reasoning", "reasoning_effort")
        if body.get(key) is not None
    }
    kwargs: dict[str, Any] = providers._litellm_kwargs(
        resolved,
        body.get("messages", []),
        max_tokens,
        body.get("temperature", 0.7),
        body.get("tools"),
        body.get("tool_choice"),
        body.get("response_format"),
        controls,
    )
    kwargs["timeout"] = float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300"))
    return kwargs


def _direct_litellm_chat(body: dict[str, Any], resolved: Any) -> dict[str, Any]:
    """Run the efficient leg through LiteLLM; return an OpenAI chat envelope."""
    import time

    import providers

    kwargs = _direct_litellm_kwargs(body, resolved)
    data = providers._response_to_dict(providers._litellm_completion(**kwargs))
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("efficient target returned no choices")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "id": f"chatcmpl-direct-{int(time.time() * 1000):x}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mantis/base",
        "choices": choices,
        "usage": usage,
    }


def _delta_to_dict(delta: Any) -> dict[str, Any]:
    """Convert a LiteLLM stream delta to a JSON-serializable OpenAI delta."""
    event_delta: dict[str, Any] = {"role": "assistant"}
    content = getattr(delta, "content", None)
    if content is not None:
        event_delta["content"] = content
    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning is not None:
        event_delta["reasoning_content"] = reasoning
    raw_calls = getattr(delta, "tool_calls", None)
    if raw_calls:
        calls = []
        for call in raw_calls:
            if isinstance(call, dict):
                calls.append(call)
                continue
            function = getattr(call, "function", None)
            entry: dict[str, Any] = {}
            if getattr(call, "id", None) is not None:
                entry["id"] = call.id
            if getattr(call, "type", None) is not None:
                entry["type"] = call.type
            if getattr(call, "index", None) is not None:
                entry["index"] = call.index
            if function is not None:
                if isinstance(function, dict):
                    entry["function"] = function
                else:
                    entry["function"] = {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    }
            calls.append(entry)
        if calls:
            event_delta["tool_calls"] = calls
    return event_delta


def _direct_litellm_stream(body: dict[str, Any], resolved: Any) -> Iterator[bytes]:
    """Yield OpenAI SSE frames for the efficient leg through LiteLLM."""
    import time

    import providers

    kwargs = _direct_litellm_kwargs(body, resolved)
    created = int(time.time())
    try:
        stream = providers._litellm_completion(stream=True, **kwargs)
        for chunk in stream:
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                finish = getattr(choice, "finish_reason", None)
                if delta is None:
                    if finish is None:
                        continue
                    event_delta = {"role": "assistant"}
                else:
                    event_delta = _delta_to_dict(delta)
                    if (
                        finish is None
                        and "content" not in event_delta
                        and "reasoning_content" not in event_delta
                        and "tool_calls" not in event_delta
                    ):
                        continue
                frame = {
                    "id": getattr(chunk, "id", "chatcmpl-direct"),
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "mantis/base",
                    "choices": [
                        {"index": 0, "delta": event_delta, "finish_reason": finish}
                    ],
                }
                yield ("data: " + json.dumps(frame) + "\n\n").encode()
    except Exception as exc:  # noqa: BLE001 - end the stream cleanly on failure
        logger.warning("direct efficient stream failed: %s", exc)
    finally:
        yield b"data: [DONE]\n\n"


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
    outbound_body = router_body(request)
    # Classify over the recent window and keep the max tier per session so a
    # reasoning turn warms capable once instead of flapping each turn.
    complexity = _classify_window(
        outbound_body.get("messages", []), _recent_turn_window()
    )
    # An explicit reasoning_effort already implies the client knows the task
    # needs reasoning; boost to "reasoning" tier so Switchyard skips efficient.
    if outbound_body.get("reasoning_effort") and complexity != "reasoning":
        complexity = "reasoning"
    complexity = _sticky_complexity(session_id(headers, request), complexity)
    outbound_headers = router_headers(headers, request, complexity=complexity)
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
                    with contextlib.suppress(Exception):
                        stream.__exit__(None, None, None)
                    with contextlib.suppress(Exception):
                        client.close()
                else:
                    return (
                        StreamingResponse(
                            _router_stream(client, stream, upstream, on_close=on_close),
                            media_type="text/event-stream",
                            headers={
                                "X-Request-Id": request_id,
                                **router_response_headers(upstream),
                            },
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
                            headers={
                                "X-Request-Id": request_id,
                                **router_response_headers(upstream),
                            },
                        ),
                        False,
                    )
                # Direct capable errored — fall through to Switchyard below.
            # If we reach here direct path did not return, so continue to Switchyard.

    # Efficient targets on adapters Switchyard cannot serve (Bedrock SigV4,
    # Vertex OAuth) run through LiteLLM directly. On failure we fall through
    # to Switchyard so escalation to the capable tier still applies.
    if not suffix:
        litellm_spec = _direct_litellm_spec()
        if litellm_spec is not None:
            try:
                if request.stream:

                    def _closing_stream() -> Iterator[bytes]:
                        try:
                            yield from _direct_litellm_stream(outbound_body, litellm_spec)
                        finally:
                            if on_close is not None:
                                on_close()

                    return (
                        StreamingResponse(
                            _closing_stream(),
                            media_type="text/event-stream",
                            headers={
                                "X-Request-Id": request_id,
                                "x-route-model": litellm_spec.model,
                            },
                        ),
                        True,
                    )
                envelope = _direct_litellm_chat(outbound_body, litellm_spec)
            except Exception as exc:  # noqa: BLE001 - fall through to Switchyard
                logger.warning("direct efficient leg failed, using Switchyard: %s", exc)
            else:
                return (
                    JSONResponse(
                        envelope,
                        headers={
                            "X-Request-Id": request_id,
                            "x-route-model": litellm_spec.model,
                        },
                    ),
                    False,
                )

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
        # Upstream returned 200 with a payload that is not an OpenAI chat
        # completion (e.g. a raw provider error passed through). Surfacing it
        # as success confuses clients and hides the outage; report 502.
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
