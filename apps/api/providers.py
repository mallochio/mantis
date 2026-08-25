#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY + Conductor coordinators.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

A client POSTs to /v1/chat/completions as if calling one model; internally the
requested coordinator ("trinity" or "conductor") runs the full loop. The
model field in the request selects the coordinator.

stdlib http.server only — no FastAPI/uvicorn.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import serve_config
import utils
from anthropic_protocols import (
    anthropic_headers,
    anthropic_to_chat,
    apply_anthropic_prompt_cache,
    assemble_anthropic_stream,
    build_anthropic_body,
)
from model_catalog import CatalogError, RuntimeBindings, load_runtime_bindings
from provider_protocols import (
    assemble_responses_stream,
    build_responses_body,
    responses_to_chat,
    uses_responses_api,
)
from serve_config import (
    _FAILOVER_DELAY,
    _PRICES_URL,
    _TRANSIENT_EXCEPTIONS,
    _TRANSIENT_STATUSES,
    DEFAULT_MAX_COMPLETION_TOKENS,
    PROVIDERS,
    REASONING_MODELS,
    _cache_read_price_cache,
)


def upstream_output_token_cap() -> int:
    """Return the output limit advertised by the API."""
    return int(os.environ.get("MANTIS_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS)))


def _runtime_bindings() -> RuntimeBindings | None:
    """Load rendered catalog bindings without altering legacy env behavior."""
    try:
        return load_runtime_bindings()
    except CatalogError as error:
        raise RuntimeError(f"invalid Mantis catalog bindings: {error}") from error


@dataclass(frozen=True)
class ResolvedModelSpec:
    """The complete routing decision for one model specification."""

    adapter: str
    model: str
    effort: str | None
    base_url: str
    credential_env: str
    binding: str | None
    protocols: tuple[str, ...] | None
    slot: str | None
    max_tokens: int | None = None


def _resolve_model_spec(spec: str, bindings: RuntimeBindings | None = None) -> ResolvedModelSpec:
    """Resolve legacy and catalog specs from one runtime-binding snapshot."""
    bindings = _runtime_bindings() if bindings is None else bindings
    worker = bindings.workers.get(spec) if bindings is not None else None
    provider_name: str | None = None
    if worker is not None:
        provider_name = worker.provider
        provider = bindings.providers.get(provider_name) if bindings is not None else None
        if provider is None:
            raise ValueError(f"catalog worker {spec} has no provider binding")
        return ResolvedModelSpec(
            provider.adapter,
            worker.upstream_model,
            worker.reasoning_effort,
            provider.base_url,
            provider.credential_env,
            provider_name,
            worker.protocols,
            spec,
            worker.max_tokens,
        )
    provider_name, separator, remainder = spec.partition("/")
    provider = bindings.providers.get(provider_name) if separator and bindings is not None else None
    model, marker, effort = remainder.partition("|")
    parsed_effort = effort if marker and effort != "none" else None
    if provider is not None:
        return ResolvedModelSpec(
            provider.adapter,
            model,
            parsed_effort,
            provider.base_url,
            provider.credential_env,
            provider_name,
            provider.protocols,
            None,
        )
    if not separator or provider_name not in PROVIDERS:
        supported = " or ".join((*PROVIDERS, "<catalog-provider>"))
        raise ValueError(f"model must be a catalog slot or start with {supported}: {spec}")
    base_url, credential_env = PROVIDERS[provider_name]
    return ResolvedModelSpec(
        provider_name, model, parsed_effort, base_url, credential_env, None, None, None
    )


# Kept for callers that used the old helper functions.
def _parse_model_spec(spec: str) -> tuple[str, str, str | None]:
    resolved = _resolve_model_spec(spec)
    return resolved.adapter, resolved.model, resolved.effort


def _binding_protocols(spec: str) -> tuple[str, ...] | None:
    return _resolve_model_spec(spec).protocols


def _provider_for_spec(spec: str) -> tuple[str, str, str | None]:
    resolved = _resolve_model_spec(spec)
    return resolved.base_url, resolved.credential_env, resolved.binding


def _provider_keys() -> dict[str, str]:
    """Return launch-injected catalog credentials without logging their values."""
    raw = os.environ.get("MANTIS_PROVIDER_KEYS", "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("invalid MANTIS_PROVIDER_KEYS") from error
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(key, str) and key for name, key in value.items()
    ):
        raise RuntimeError("invalid MANTIS_PROVIDER_KEYS")
    return value


def _catalog_key(binding: str, credential_env: str, spec: str) -> str:
    key = _provider_keys().get(binding) or os.environ.get(credential_env)
    if not key:
        raise RuntimeError(f"{credential_env} is required for {spec}")
    return key


class ClientDisconnectedError(Exception):
    """Raised when client disconnects during streaming or step execution."""


class RunCapacityError(Exception):
    """Raised instead of evicting a live tool run."""


def _check_client_connected() -> None:
    """Check if current request client connection is broken or aborted."""
    if getattr(serve_config._history_context, "aborted", False):
        raise ClientDisconnectedError("Client disconnected")
    is_connected = getattr(serve_config._history_context, "is_client_connected", None)
    if is_connected is not None and not is_connected():
        serve_config._history_context.aborted = True
        raise ClientDisconnectedError("Client disconnected")


def _is_reasoning_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1]
    return any(name.startswith(p) for p in REASONING_MODELS)


def _is_openai_caching_model(model: str) -> bool:
    """Models that accept OpenAI's explicit prompt_cache_options / breakpoints."""
    return _model_cache_family(model) == "openai"


def _model_cache_family(model: str) -> str | None:
    """Return the prompt-cache dialect for an upstream model id, or None.

    Family is taken from the last path segment so Bifrost ids such as
    ``bedrock/anthropic/claude-opus-5`` and ``google/gemini-3.7-flash`` match
    the same rules as OpenRouter ``anthropic/claude-…`` / ``google/gemini-…``.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-5.6-") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    return None


def _coerce_reasoning_effort(model: str, effort: str | None) -> str | None:
    """Map catalog reasoning_effort to a value the upstream model accepts.

    Providers have different reasoning level vocabularies. Passing an
    unsupported level wastes tokens or raises an API error. The deepseek-style
    pattern is to avoid reasoning on low/medium and only pay for it on high/max.
    For OpenAI GPT-5.6 the accepted set is none/low/medium/high/xhigh; `max`
    is rejected and pinning `none` is required to avoid the expensive default.
    """
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("glm-"):
        return None if effort == "none" else ("high" if effort == "medium" else effort)
    if name.startswith("deepseek-") or name.startswith("deepseek_"):
        if effort in ("none", "minimal", "low", "medium"):
            return None
        if effort == "xhigh":
            return "max"
        if effort in ("high", "max"):
            return effort
        return None
    # OpenAI-family reasoning models accept none through xhigh; `max` is invalid
    # on GPT-5.6 and is clamped to the highest supported level.
    if name.startswith("gpt-5.6-") or name.startswith("o1") or name.startswith("o3"):
        if effort is None:
            return "none"
        if effort == "max":
            return "xhigh"
        if effort in ("none", "low", "medium", "high", "xhigh"):
            return effort
        return None
    # Anthropic uses a native thinking object; keep the original for budget sizing.
    if name.startswith("claude-"):
        if effort is None:
            return None
        if effort == "max":
            return "xhigh"
        if effort == "none":
            return "none"
        return effort
    # Unknown models: keep the requested effort unless it is unsupported.
    if effort is None:
        return None
    if effort == "max":
        return "xhigh"
    return effort


def _cache_breakpoints_enabled() -> bool:
    value = os.environ.get("MANTIS_CACHE_BREAKPOINTS", "1").lower()
    return value not in {"0", "false", "no", "off"}


def _cache_retention() -> str:
    return os.environ.get("MANTIS_CACHE_RETENTION", "short").lower()


def _cache_retention_long() -> bool:
    return _cache_retention() == "long"


def _cache_retention_enabled() -> bool:
    return _cache_retention() != "none"


def _openai_cache_breakpoints_enabled() -> bool:
    """GPT-5.6 explicit cache follows the master breakpoint switch unless overridden.

    ``MANTIS_OPENAI_CACHE_BREAKPOINTS`` unset means “same as MANTIS_CACHE_BREAKPOINTS”.
    An explicit 0 still disables OpenAI markup without turning off Anthropic
    ``cache_control``.
    """
    if not _cache_breakpoints_enabled():
        return False
    value = os.environ.get("MANTIS_OPENAI_CACHE_BREAKPOINTS")
    if value is None:
        return True
    return value.lower() not in {"0", "false", "no", "off"}


def _with_openai_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark stable layers for OpenAI GPT-5.6+ explicit prompt caching.

    Places explicit breakpoints on the last system message and on the message
    just before the latest user/tool turn so the volatile tail is billed fresh
    while the stable prefix is reused. This matches the GPT-5.6 caching guide.
    """
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


def _with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the stable prefix for prompt caching (OpenRouter + Anthropic models).

    The orchestration re-sends the same system+history prefix for every internal
    step and tool round, which is exactly the pattern provider prompt caching
    rewards. Breakpoints go on the end of the system message and on the message
    before the final role prompt (the end of the stable history prefix), so the
    whole reusable prefix is cached and only the role prompt varies per step.
    The input list is not mutated.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return messages
    out = [dict(message) for message in messages]

    def _mark(message: dict[str, Any]) -> None:
        content = message.get("content")
        if content is None:
            return  # tool-call messages carry no text; nothing to cache-mark
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


def _apply_chat_prompt_cache(
    model: str, messages: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (messages, extra body fields) for the model's cache dialect.

    Gemini uses implicit prefix caching: extra markup is omitted so Bifrost /
    Vertex OpenAI-compat cannot 400 on Anthropic or GPT-5.6 fields. Unknown
    families are also left unmarked.
    """
    extra: dict[str, Any] = {}
    family = _model_cache_family(model)
    if family == "anthropic" and _cache_breakpoints_enabled():
        return _with_cache_breakpoints(messages), extra
    if family == "openai" and _openai_cache_breakpoints_enabled():
        extra["prompt_cache_options"] = {"mode": "explicit", "ttl": "30m"}
        return _with_openai_cache_breakpoints(messages), extra
    return messages, extra


def _normalize_upstream_tool_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy with compact, consistently paired tool-call IDs."""
    aliases: dict[str, str] = {}

    def alias(raw_id: str) -> str:
        if raw_id not in aliases:
            aliases[raw_id] = f"m{len(aliases):08x}"
        return aliases[raw_id]

    normalized: list[dict[str, Any]] = []
    for original in messages:
        message = dict(original)
        calls = original.get("tool_calls")
        if isinstance(calls, list):
            message["tool_calls"] = [
                {**call, "id": alias(call["id"])}
                if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]
                else call
                for call in calls
            ]
        result_id = original.get("tool_call_id")
        if isinstance(result_id, str) and result_id:
            message["tool_call_id"] = alias(result_id)
        raw_tool_ids = original.get("_anthropic_tool_ids")
        if isinstance(raw_tool_ids, dict):
            message["_anthropic_tool_ids"] = {
                alias(str(call_id)): str(provider_id)
                for call_id, provider_id in raw_tool_ids.items()
            }
        normalized.append(message)
    return normalized


_REASONING_FIELDS = frozenset(
    {"reasoning", "reasoning_content", "reasoning_details", "thinking", "thinkingSignature"}
)


def _sanitize_messages(
    messages: list[dict[str, Any]],
    model: str,
    is_anthropic: bool = False,
    is_responses: bool = False,
) -> list[dict[str, Any]]:
    """Remove provider-specific metadata and cross-model reasoning bloat.

    Reasoning/thinking generated by one model is usually invalid or billed at
    full input cost for another. Anthropic native blocks are kept so signed
    thinking can be replayed. OpenAI Responses keeps reasoning items because
    the API persists them across turns. DeepSeek only needs reasoning passed
    back on assistant turns that carried tool calls (the API requires it then);
    it is dropped on tool-call-free turns to save tokens and protect cache
    prefixes. For plain chat the model cannot reuse prior reasoning, so it is
    stripped.
    """
    is_deepseek = "deepseek" in model.lower()
    sanitized: list[dict[str, Any]] = []
    for original in messages:
        msg = dict(original)
        role = msg.get("role")
        if role == "assistant":
            has_tool_calls = bool(msg.get("tool_calls"))
            keep_reasoning = is_responses or (is_deepseek and has_tool_calls)
            content = msg.get("content")
            if isinstance(content, list):
                if is_anthropic:
                    msg["content"] = [dict(part) for part in content]
                else:
                    msg["content"] = [
                        dict(part)
                        for part in content
                        if isinstance(part, dict)
                        and part.get("type") not in ("thinking", "reasoning", "reasoning_content")
                    ]
            if not keep_reasoning:
                for key in _REASONING_FIELDS:
                    msg.pop(key, None)
            if not is_anthropic:
                for key in ("_anthropic_content", "_anthropic_tool_ids"):
                    msg.pop(key, None)
        elif role == "tool":
            for key in _REASONING_FIELDS:
                msg.pop(key, None)
            for key in ("_anthropic_content", "_anthropic_tool_ids"):
                msg.pop(key, None)
        sanitized.append(msg)
    return sanitized


def _prompt_cache_namespace(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> str:
    """Return a privacy-safe stable key for one conversation's cacheable root."""
    root: list[dict[str, Any]] = []
    for message in messages:
        root.append(message)
        if message.get("role") == "user":
            break
    payload = json.dumps({"messages": root, "tools": tools or []}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@contextmanager
def progress_events(sink: Any):
    """Install a request-local orchestration event sink."""
    previous = getattr(serve_config._history_context, "event_sink", None)
    serve_config._history_context.event_sink = sink
    try:
        yield
    finally:
        serve_config._history_context.event_sink = previous


def _emit_progress(event: dict[str, Any]) -> None:
    sink = getattr(serve_config._history_context, "event_sink", None)
    if sink is not None:
        sink(event)


@contextmanager
def client_connection(is_connected: Any):
    """Install a request-local client connection check."""
    previous = getattr(serve_config._history_context, "is_client_connected", None)
    serve_config._history_context.is_client_connected = is_connected
    try:
        yield
    finally:
        serve_config._history_context.is_client_connected = previous


def _uses_anthropic_messages(spec: str | ResolvedModelSpec) -> bool:
    resolved = _resolve_model_spec(spec) if isinstance(spec, str) else spec
    return resolved.protocols is not None and "anthropic_messages" in resolved.protocols


def _build_request(
    spec: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    response_format: dict[str, Any] | None = None,
    controls: dict[str, Any] | None = None,
    resolved: ResolvedModelSpec | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    resolved = _resolve_model_spec(spec) if resolved is None else resolved
    provider, model, effort = resolved.adapter, resolved.model, resolved.effort
    if resolved.max_tokens is not None and max_tokens > resolved.max_tokens:
        # Catalog-declared provider output cap (models.dev) for this worker.
        max_tokens = resolved.max_tokens
    base_url, key_env, binding = resolved.base_url, resolved.credential_env, resolved.binding
    key = _catalog_key(binding, key_env, spec) if binding else os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is required for {spec}")
    active_controls = controls or {}
    normalized_messages = _normalize_upstream_tool_ids(messages)
    is_anthropic = _uses_anthropic_messages(resolved)
    is_responses = uses_responses_api(provider, model, resolved.protocols)
    sanitized_messages = _sanitize_messages(normalized_messages, model, is_anthropic, is_responses)
    coerced_effort = _coerce_reasoning_effort(model, effort)
    if _uses_anthropic_messages(resolved):
        body = build_anthropic_body(
            model, sanitized_messages, max_tokens, effort, tools, tool_choice
        )
        if _cache_breakpoints_enabled() and _model_cache_family(model) == "anthropic":
            body = apply_anthropic_prompt_cache(body)
        headers = anthropic_headers(key)
        path = "v1/messages"
    elif is_responses:
        body = build_responses_body(
            model,
            sanitized_messages,
            max_tokens,
            None if effort or _is_reasoning_model(model) else temperature,
            coerced_effort,
            tools,
            tool_choice,
            response_format,
            active_controls,
        )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python",
        }
        path = "responses"
    else:
        body = {"model": model, "messages": sanitized_messages, "max_tokens": max_tokens}
        cached_messages, cache_extra = _apply_chat_prompt_cache(model, sanitized_messages)
        body["messages"] = cached_messages
        body.update(cache_extra)
        if coerced_effort:
            body["reasoning_effort"] = coerced_effort
        if not effort and not _is_reasoning_model(model):
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if "reasoning" in active_controls:
            body.pop("reasoning_effort", None)
        body.update(active_controls)
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python",
        }
        path = "chat/completions"
    return f"{base_url}/{path}", headers, body


def _upstream_streaming_enabled() -> bool:
    """Stream upstream provider responses (SSE) so long generations survive
    pass-through proxies; disable with MANTIS_UPSTREAM_STREAM=0."""
    return os.environ.get("MANTIS_UPSTREAM_STREAM", "1").lower() not in {"0", "false", "no", "off"}


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one SSE `data:` line; None for comments/junk/keep-alives, {} for [DONE]."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:") :].strip()
    if payload == "[DONE]":
        return {}
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _assemble_streamed_completion(chunks: Any) -> dict[str, Any]:
    """Reassemble OpenAI-style SSE chunks into one completion object."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] | None = None
    for chunk in chunks:
        if not chunk:
            continue
        if chunk.get("error"):
            error = chunk["error"]
            detail = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"provider stream error: {detail}")
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            if delta.get("role"):
                message["role"] = delta["role"]
            if delta.get("content"):
                message["content"] = (message.get("content") or "") + delta["content"]
            for key in ("reasoning", "reasoning_details", "annotations", "citations"):
                part = delta.get(key)
                if isinstance(part, str):
                    message[key] = (message.get(key) or "") + part
                elif part is not None:
                    message[key] = part
            for call in delta.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                index = call.get("index")
                if not isinstance(index, int):
                    index = len(tool_calls)
                while len(tool_calls) <= index:
                    tool_calls.append(
                        {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                    )
                slot = tool_calls[index]
                if call.get("id"):
                    slot["id"] = call["id"]
                if call.get("type"):
                    slot["type"] = call["type"]
                function = call.get("function")
                if isinstance(function, dict):
                    if function.get("name"):
                        slot["function"]["name"] += function["name"]
                    if function.get("arguments"):
                        slot["function"]["arguments"] += function["arguments"]
    named = [call for call in tool_calls if call.get("id") or call["function"]["name"]]
    if named:
        message["tool_calls"] = named
    return {"choices": [{"message": message}], "usage": usage}


def _stream_completion(
    client: Any,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    responses_api: bool = False,
    anthropic_messages: bool = False,
) -> dict:
    """POST with SSE streaming; returns the canonical Chat-shaped result."""
    stream_body = dict(body)
    stream_body["stream"] = True
    if not responses_api:
        stream_body["stream_options"] = {"include_usage": True}
    with client.stream("POST", url, headers=headers, json=stream_body) as response:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError:
            # Streamed error bodies stay unread until consumed; read real httpx
            # responses before re-raising so callers can inspect `response.text`.
            if hasattr(response, "read"):
                response.read()
            raise

        def chunks() -> Any:
            for line in response.iter_lines():
                _check_client_connected()
                yield _parse_sse_line(line)

        if anthropic_messages:
            return assemble_anthropic_stream(chunks())
        if responses_api:
            return assemble_responses_stream(chunks())
        return _assemble_streamed_completion(chunks())


def _failover_attempts(spec: str, bindings: RuntimeBindings | None = None) -> list[str]:
    """Attempt order on transient failures: the assigned spec, then the rest of
    the configured pool in declared order, each at most once. Falls back to
    [spec] when no provider pool is configured (tests, local workers)."""
    try:
        pool = utils._configured_slot_models()
    except (TypeError, ValueError):
        return [spec]
    attempts = [spec]
    for candidate in pool:
        if candidate in attempts:
            continue
        try:
            _resolve_model_spec(candidate, bindings)
        except ValueError:
            continue  # bare labels are not routable failover targets
        attempts.append(candidate)
    return attempts


def _provider_response(
    spec: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run = getattr(serve_config._history_context, "active_run", None)
    tool_choice = getattr(run, "active_tool_choice", None)
    response_format = getattr(run, "active_response_format", None)
    controls = getattr(run, "active_controls", None) or {}
    normalized_messages = _normalize_upstream_tool_ids(messages)
    failures: list[str] = []
    bindings = _runtime_bindings()
    for index, attempt in enumerate(_failover_attempts(spec, bindings)):
        _check_client_connected()
        if index:
            time.sleep(_FAILOVER_DELAY)
        try:
            resolved = _resolve_model_spec(attempt, bindings)
            # Failover switches model/effort/endpoint per spec; messages, tools,
            # and controls stay identical across attempts.
            url, headers, body = _build_request(
                attempt,
                normalized_messages,
                max_tokens,
                temperature,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format,
                controls=controls,
                resolved=resolved,
            )
        except RuntimeError as error:  # provider key missing for this attempt
            failures.append(str(error))
            continue
        provider, model = resolved.adapter, resolved.model
        responses_api = uses_responses_api(provider, model, resolved.protocols)
        anthropic_messages = _uses_anthropic_messages(resolved)
        # Session stickiness pins OpenRouter to one model+provider per
        # conversation to maximize prompt-cache hits. Apply it to all OpenRouter
        # backends (native Responses for "openai/*" and Chat Completions for
        # every other OpenRouter model), not just the Responses path.
        if provider == "openrouter" and _cache_retention_enabled():
            namespace = getattr(run, "cache_namespace", None) or _prompt_cache_namespace(
                messages, tools
            )
            body["session_id"] = f"mantis-{namespace}"
            cache_key = hashlib.sha256(f"{model}:{namespace}".encode()).hexdigest()[:32]
            body["prompt_cache_key"] = cache_key
            if _cache_retention_long():
                body["prompt_cache_retention"] = "24h"
            if responses_api and _cache_breakpoints_enabled():
                body["cache_control"] = {"type": "ephemeral"}
        _emit_progress(
            {
                "type": "provider",
                "status": "started",
                "model": attempt,
                "protocol": (
                    "anthropic_messages"
                    if anthropic_messages
                    else "responses"
                    if responses_api
                    else "chat_completions"
                ),
                "attempt": index + 1,
                "summary": "Calling a model",
            }
        )
        try:
            if _upstream_streaming_enabled():
                data = _stream_completion(
                    serve_config._provider_client,
                    url,
                    headers,
                    body,
                    responses_api=responses_api,
                    anthropic_messages=anthropic_messages,
                )
            else:
                response = serve_config._provider_client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
                if anthropic_messages:
                    data = anthropic_to_chat(data)
                elif responses_api:
                    data = responses_to_chat(data)
            break
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in _TRANSIENT_STATUSES:
                _emit_progress(
                    {
                        "type": "provider",
                        "status": "failed",
                        "model": attempt,
                        "attempt": index + 1,
                        "summary": f"Model call failed with HTTP {status}",
                    }
                )
                raise RuntimeError(
                    f"{attempt} returned HTTP {status}: {error.response.text[:500]}"
                ) from error
            failures.append(f"{attempt}: HTTP {status}")
            run_record = getattr(serve_config._history_context, "active_run", None)
            if run_record is not None:
                run_record.record_activity(
                    "failover",
                    model=attempt,
                    status="failed",
                    attempt=index + 1,
                    detail=f"HTTP {status}: {error.response.text[:300]}",
                )
        except (RuntimeError, *_TRANSIENT_EXCEPTIONS) as error:
            failures.append(f"{attempt}: {type(error).__name__}")
            run_record = getattr(serve_config._history_context, "active_run", None)
            if run_record is not None:
                run_record.record_activity(
                    "failover",
                    model=attempt,
                    status="failed",
                    attempt=index + 1,
                    detail=str(error)[:300],
                )
    else:
        raise RuntimeError(f"{spec} failed on every pool worker: " + "; ".join(failures))
    if not isinstance(data, dict):
        raise TypeError(f"{spec} returned a non-object response")
    raw_usage = data.get("usage")
    usage = cast(dict[str, Any], raw_usage) if isinstance(raw_usage, dict) else {}
    if run is not None:
        # Catalog slots are the trained-router identities. Legacy raw specs
        # retain their upstream model attribution for existing cost reports.
        usage_model = attempt if resolved.slot is not None else body["model"]
        run.add_usage(usage, model=usage_model)
        message = (data.get("choices") or [{}])[0].get("message", {})
        if run.capture_metadata and isinstance(message, dict):
            run.response_metadata = {
                key: message[key]
                for key in ("reasoning", "reasoning_details", "annotations", "citations")
                if message.get(key) is not None
            }
    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = (
        prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0
    )
    _emit_progress(
        {
            "type": "provider",
            "status": "completed",
            "model": attempt,
            "protocol": (
                "anthropic_messages"
                if anthropic_messages
                else "responses"
                if responses_api
                else "chat_completions"
            ),
            "attempt": index + 1,
            "usage": {
                key: usage[key]
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(usage.get(key), int)
            },
            "cached_tokens": cached_tokens,
            "summary": (
                f"Model call completed ({cached_tokens} cached input tokens)"
                if cached_tokens
                else "Model call completed"
            ),
        }
    )
    return data


def _direct_completion(
    spec: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, timeout: float
) -> str:
    del timeout  # the shared client uses MANTIS_WORKER_TIMEOUT
    data = _provider_response(spec, messages, max_tokens, temperature)
    return str(data["choices"][0]["message"].get("content") or "")


def _price_entry(entry: Any) -> tuple[str, tuple[float, float], float | None] | None:
    """(model id, (prompt, completion) price, cache-read price) or None."""
    try:
        pricing = entry["pricing"]
        cache_read = pricing.get("input_cache_read")
        cache_price = None if cache_read in (None, "") else float(cache_read)
        return (
            str(entry["id"]),
            (float(pricing["prompt"]), float(pricing["completion"])),
            cache_price,
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _fetch_prices() -> dict[str, tuple[float, float]]:
    """Fetch (prompt, completion) USD prices and cache-read prices; empty on failure."""
    global _cache_read_price_cache
    try:
        response = httpx.get(_PRICES_URL, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        _cache_read_price_cache = {}
        return {}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        _cache_read_price_cache = {}
        return {}
    prices: dict[str, tuple[float, float]] = {}
    cache_read: dict[str, float] = {}
    for entry in data:
        pair = _price_entry(entry)
        if pair is not None:
            prices[pair[0]] = pair[1]
            if pair[2] is not None:
                cache_read[pair[0]] = pair[2]
    _cache_read_price_cache = cache_read
    return prices


def _price_map() -> dict[str, tuple[float, float]]:
    """Per-token USD prices keyed by model id, fetched lazily once per process."""
    if serve_config._price_cache is None:
        serve_config._price_cache = _fetch_prices()
    return serve_config._price_cache


def _cache_read_prices() -> dict[str, float]:
    """Per-model cache-read USD prices, populated by the same fetch as the map."""
    global _cache_read_price_cache
    if _cache_read_price_cache is None:
        _price_map()  # the shared lazy fetch fills both caches
    return _cache_read_price_cache or {}


def _price_key(model: str, prices: dict[str, tuple[float, float]]) -> str | None:
    """Resolve a usage key to one unambiguous priced model id."""
    if model in prices:
        return model
    try:
        resolved = _resolve_model_spec(model).model
    except (RuntimeError, ValueError):
        resolved = None
    for candidate in (model, resolved):
        if not candidate:
            continue
        if candidate in prices:
            return candidate
        matches = [key for key in prices if key.endswith(f"/{candidate}")]
        if len(matches) == 1:
            return matches[0]
    return None


def _model_cost(
    model: str, tokens: dict[str, int], prices: dict[str, tuple[float, float]]
) -> float | None:
    """USD cost for one model's tokens, cache-aware; None when unpriced."""
    key = _price_key(model, prices)
    if key is None:
        return None
    price = prices[key]
    prompt = int(tokens.get("prompt_tokens", 0))
    completion = int(tokens.get("completion_tokens", 0))
    cached = min(int(tokens.get("cached_tokens", 0)), prompt)
    cache_price = _cache_read_prices().get(key, price[0])
    return (prompt - cached) * price[0] + cached * cache_price + completion * price[1]


def _usage_cost(usage_models: dict[str, dict[str, int]]) -> float | None:
    """Total USD cost, or None when any consumed model has no known price."""
    if not usage_models:
        return None
    prices = _price_map()
    cost = 0.0
    for model, tokens in usage_models.items():
        model_cost = _model_cost(model, tokens, prices)
        if model_cost is None:
            return None
        cost += model_cost
    return round(cost, 6)


def _activity_summary(activity_type: str, role: str | None = None) -> str:
    """Deterministic one-line label for an orchestration activity."""
    label = {
        "step": {
            "Planner": "Planned the workflow",
            "Thinker": "Analyzed the task",
            "Verifier": "Verified the answer",
            "Worker": "Drafted the answer",
        }.get(role or "", "Called a model"),
        "tool_call": "Requested a tool call",
        "tool_result": "Processed a tool result",
        "failover": "Provider failed; switched to the next configured model",
        "verify_accept": "Final answer accepted by verifier",
        "verify_reject": "Verifier rejected the draft; requesting revision",
        "retry": "Retrying the step",
        "complete": "Run completed",
        "provider": "Called a provider model",
        "run": "Started Mantis orchestration",
        "validation": "Validated the final answer",
        "error": "Run ended with an error",
    }.get(activity_type, "Orchestration step")
    return label


def _running_summary(role: str) -> str:
    return {
        "Planner": "Planning the workflow",
        "Thinker": "Analyzing the task",
        "Verifier": "Checking the draft",
        "Worker": "Drafting an answer",
    }.get(role, "Calling a model")


def _cost_breakdown(usage_models: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Per-model cost and cache effectiveness for internal orchestration calls."""
    if not usage_models:
        return {"total": None, "known": False, "source": "unavailable", "models": []}
    prices = _price_map()
    models: list[dict[str, Any]] = []
    total = 0.0
    total_prompt = 0
    total_cached = 0
    known = True
    for model, tokens in usage_models.items():
        prompt = tokens["prompt_tokens"]
        cached = min(tokens.get("cached_tokens", 0), prompt)
        total_prompt += prompt
        total_cached += cached
        model_cost = _model_cost(model, tokens, prices)
        if model_cost is None:
            known = False
        else:
            total += model_cost
        models.append(
            {
                "model": model,
                "prompt_tokens": prompt,
                "completion_tokens": tokens["completion_tokens"],
                "cached_tokens": cached,
                "cache_hit_ratio": round(cached / prompt, 4) if prompt else 0.0,
                "cost": round(model_cost, 6) if model_cost is not None else None,
                "source": "price_table" if model_cost is not None else "unknown",
            }
        )
    return {
        "total": round(total, 6) if known else None,
        "known": known,
        "source": "price_table" if known else "partial",
        "prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "cache_hit_ratio": round(total_cached / total_prompt, 4) if total_prompt else 0.0,
        "models": models,
    }


def _run_mantis_details(run: Any, level: str) -> dict[str, Any]:
    """Build the additive `mantis` response object for the given detail level."""
    steps = list(getattr(run, "turns", getattr(run, "steps", [])))
    recorded = list(getattr(run, "_activity", []))
    activity: list[dict[str, Any]] = []
    if recorded:
        keep = None if level == "debug" else ("type", "role", "model", "status", "summary")
        activity = [
            dict(entry) if keep is None else {k: v for k, v in entry.items() if k in keep}
            for entry in recorded
            if any(v is not None for v in entry.values())
        ]
    else:
        for step in steps:
            role = str(step.get("role", ""))
            entry = {
                "type": "step",
                "role": role,
                "model": step.get("model_name"),
                "status": "completed",
                "summary": _activity_summary("step", role),
            }
            activity.append(entry)
        activity.extend(
            {
                "type": "tool_result",
                "status": "failed" if item.get("is_error") else "completed",
                "summary": _activity_summary("tool_result"),
            }
            for item in getattr(run, "tool_observations", [])
        )
    if activity and activity[-1].get("type") != "complete":
        activity.append(
            {
                "type": "complete",
                "status": "completed",
                "summary": _activity_summary("complete"),
            }
        )
    started = getattr(run, "_started_monotonic", None)
    duration_ms = round((time.monotonic() - started) * 1000.0, 1) if started else None
    outcome = str(getattr(run, "terminated_by", "") or "")
    return {
        "run_id": getattr(run, "run_id", ""),
        "mode": getattr(run, "kind", ""),
        "outcome": outcome,
        "duration_ms": duration_ms,
        "activity": activity,
        "usage": _cost_breakdown(getattr(run, "usage_models", {})),
    }


__all__ = [k for k in globals() if not k.startswith("__")]
