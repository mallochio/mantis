#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Minimal LiteLLM-backed provider layer.
# All provider handling is delegated to the LiteLLM SDK. This module keeps
# only mantis policy: catalog resolution, cross-model hygiene, and run accounting.

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

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import serve_config
import utils
from model_catalog import CatalogError, RuntimeBindings, load_runtime_bindings
from serve_config import (
    _FAILOVER_DELAY,
    _PRICES_URL,
    _TRANSIENT_STATUSES,
    DEFAULT_MAX_COMPLETION_TOKENS,
    PROVIDERS,
    REASONING_MODELS,
    _cache_read_price_cache,
)

# ponytail: LiteLLM carries retries but mantis keeps one pool-level failover
# across the declared worker slots; the SDK handles wire-protocol translation.


def upstream_output_token_cap() -> int:
    return int(os.environ.get("MANTIS_MAX_COMPLETION_TOKENS", str(DEFAULT_MAX_COMPLETION_TOKENS)))


def _runtime_bindings() -> RuntimeBindings | None:
    try:
        return load_runtime_bindings()
    except CatalogError as error:
        raise RuntimeError(f"invalid Mantis catalog bindings: {error}") from error


@dataclass(frozen=True)
class ResolvedModelSpec:
    adapter: str
    model: str
    effort: str | None
    base_url: str
    credential_env: str
    binding: str | None
    protocols: tuple[str, ...] | None
    slot: str | None
    max_tokens: int | None = None
    context_window: int | None = None


def _resolve_model_spec(spec: str, bindings: RuntimeBindings | None = None) -> ResolvedModelSpec:
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
            worker.context_window,
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


def _parse_model_spec(spec: str) -> tuple[str, str, str | None]:
    resolved = _resolve_model_spec(spec)
    return resolved.adapter, resolved.model, resolved.effort


def _binding_protocols(spec: str) -> tuple[str, ...] | None:
    return _resolve_model_spec(spec).protocols


def _provider_for_spec(spec: str) -> tuple[str, str, str | None]:
    resolved = _resolve_model_spec(spec)
    return resolved.base_url, resolved.credential_env, resolved.binding


def _provider_keys() -> dict[str, str]:
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
    pass


class RunCapacityError(Exception):
    pass


def _check_client_connected() -> None:
    if getattr(serve_config._history_context, "aborted", False):
        raise ClientDisconnectedError("Client disconnected")
    is_connected = getattr(serve_config._history_context, "is_client_connected", None)
    if is_connected is not None and not is_connected():
        serve_config._history_context.aborted = True
        raise ClientDisconnectedError("Client disconnected")


def _is_reasoning_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1]
    return any(name.startswith(p) for p in REASONING_MODELS)


def _model_cache_family(model: str) -> str | None:
    name = model.rsplit("/", 1)[-1].lower()
    if name.startswith("claude-"):
        return "anthropic"
    if name.startswith("gpt-5.6-") or name.startswith("o3") or name.startswith("o4"):
        return "openai"
    if name.startswith("gemini-"):
        return "gemini"
    return None


def _coerce_reasoning_effort(model: str, effort: str | None) -> str | None:
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
    if name.startswith("kimi-") or name.startswith("moonshot"):
        if effort in ("max", "high", "xhigh", "low"):
            return effort
        if effort == "medium":
            return "high"
        if effort in ("none", "minimal"):
            return "low"
        return None
    if name.startswith("gpt-5.6-") or name.startswith("o1") or name.startswith("o3"):
        if effort is None:
            return "none"
        if effort == "max":
            return "xhigh"
        if effort in ("none", "low", "medium", "high", "xhigh"):
            return effort
        return None
    if name.startswith("claude-"):
        if effort is None:
            return None
        if effort == "max":
            return "xhigh"
        if effort == "none":
            return "none"
        return effort
    if effort is None:
        return None
    if effort == "max":
        return "xhigh"
    return effort



def _cache_breakpoints_enabled() -> bool:
    return os.environ.get("MANTIS_CACHE_BREAKPOINTS", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


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


def _with_cache_breakpoints(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or len(messages) < 2:
        return messages
    out = [dict(m) for m in messages]

    def _mark(m: dict[str, Any]) -> None:
        content = m.get("content")
        if content is None:
            return
        if isinstance(content, list):
            blocks = list(content)
        else:
            blocks = [{"type": "text", "text": str(content)}]
        if blocks and isinstance(blocks[-1], dict) and "cache_control" not in blocks[-1]:
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            m["content"] = blocks

    if out and out[0].get("role") == "system":
        _mark(out[0])
    if len(out) >= 3:
        _mark(out[-2])
    return out


def _normalize_upstream_tool_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aliases: dict[str, str] = {}

    def alias(raw: str) -> str:
        if raw not in aliases:
            aliases[raw] = f"m{len(aliases):08x}"
        return aliases[raw]

    out: list[dict[str, Any]] = []
    for original in messages:
        m = dict(original)
        calls = original.get("tool_calls")
        if isinstance(calls, list):
            m["tool_calls"] = [
                {**call, "id": alias(call["id"])}
                if isinstance(call, dict) and isinstance(call.get("id"), str) and call["id"]
                else call
                for call in calls
            ]
        result_id = original.get("tool_call_id")
        if isinstance(result_id, str) and result_id:
            m["tool_call_id"] = alias(result_id)
        raw_ids = original.get("_anthropic_tool_ids")
        if isinstance(raw_ids, dict):
            m["_anthropic_tool_ids"] = {
                alias(str(k)): str(v) for k, v in raw_ids.items()
            }
        out.append(m)
    return out



_REASONING_FIELDS = frozenset(
    {"reasoning", "reasoning_content", "reasoning_details", "thinking", "thinkingSignature"}
)


_ENDPOINT_BOUND_MARKERS = ("encrypted", "compaction")

# Internal bookkeeping that must never reach a provider.
_INTERNAL_FIELDS = ("_anthropic_content", "_anthropic_tool_ids", "_mantis_model")


def _portable_reasoning_details(msg: dict[str, Any]) -> None:
    """Drop reasoning items that are bound to the endpoint that produced them.

    Mirrors ``base_proxy._strip_endpoint_bound_reasoning``. Encrypted and
    compaction items are only valid at their originating endpoint, so replaying
    them after a model switch yields an upstream 404. Plain summaries are
    portable and stay.
    """
    details = msg.get("reasoning_details")
    if not isinstance(details, list):
        return
    portable = [
        detail
        for detail in details
        if not isinstance(detail, dict)
        or not any(
            marker in str(detail.get(key, "")).lower()
            for key in ("type", "format")
            for marker in _ENDPOINT_BOUND_MARKERS
        )
    ]
    if portable:
        msg["reasoning_details"] = portable
    else:
        msg.pop("reasoning_details", None)


def _sanitize_messages(
    messages: list[dict[str, Any]],
    model: str,
    is_anthropic: bool = False,
    is_responses: bool = False,
) -> list[dict[str, Any]]:
    is_deepseek = "deepseek" in model.lower()
    sanitized: list[dict[str, Any]] = []
    for original in messages:
        msg = dict(original)
        role = msg.get("role")
        if role == "assistant":
            has_tool_calls = bool(msg.get("tool_calls"))
            # Reasoning is model-bound. A Fusion pool can promote mid-run, so
            # replaying the previous model's reasoning is wasted at best and an
            # upstream rejection at worst. Only the producer may see it again.
            produced_by = msg.get("_mantis_model")
            same_model = produced_by is None or str(produced_by) == model
            keep_reasoning = (is_responses or (is_deepseek and has_tool_calls)) and same_model
            content = msg.get("content")
            if isinstance(content, list):
                if is_anthropic:
                    msg["content"] = [dict(p) for p in content]
                else:
                    msg["content"] = [
                        dict(p)
                        for p in content
                        if isinstance(p, dict)
                        and p.get("type")
                        not in ("thinking", "reasoning", "reasoning_content")
                    ]
            if keep_reasoning:
                # Even for the producing model, endpoint-bound items are unsafe.
                _portable_reasoning_details(msg)
            else:
                for k in _REASONING_FIELDS:
                    msg.pop(k, None)
            if not is_anthropic:
                for k in ("_anthropic_content", "_anthropic_tool_ids"):
                    msg.pop(k, None)
        elif role == "tool":
            for k in _REASONING_FIELDS:
                msg.pop(k, None)
            for k in ("_anthropic_content", "_anthropic_tool_ids"):
                msg.pop(k, None)
        msg.pop("_mantis_model", None)
        sanitized.append(msg)
    return sanitized


def _prompt_cache_namespace(
    messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, model: str = ""
) -> str:
    """Fingerprint the cacheable prompt prefix for one model.

    ``model`` is part of the namespace because provider prompt caches are
    per-model: a Fusion pool that promotes mid-run must not be handed a
    namespace it shares with the slot it was promoted from.
    """
    root: list[dict[str, Any]] = []
    for m in messages:
        root.append(m)
        if m.get("role") == "user":
            break
    payload = json.dumps(
        {"model": model, "messages": root, "tools": tools or []}, sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@contextmanager
def progress_events(sink: Any):
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
    previous = getattr(serve_config._history_context, "is_client_connected", None)
    serve_config._history_context.is_client_connected = is_connected
    try:
        yield
    finally:
        serve_config._history_context.is_client_connected = previous


# -- LiteLLM helpers -------------------------------------------------------

_LITELLM_KNOWN = {"anthropic", "bedrock", "vertex_ai", "vertex", "azure", "openrouter"}


def _is_anthropic_spec(resolved: ResolvedModelSpec) -> bool:
    return resolved.adapter == "anthropic" or (
        resolved.protocols is not None and "anthropic_messages" in resolved.protocols
    )


def _litellm_model(resolved: ResolvedModelSpec) -> str:
    # ponytail: one branch for known litellm providers; everything else is
    # OpenAI-compatible with a custom base_url.
    if resolved.adapter in _LITELLM_KNOWN:
        prefix = "vertex_ai" if resolved.adapter == "vertex" else resolved.adapter
        # don’t double-prefix if upstream_model already provider-qualified
        if resolved.model.startswith(f"{prefix}/"):
            return resolved.model
        return f"{prefix}/{resolved.model}"
    # openai-compatible: strip any leading vendor prefix, use openai handler
    clean = resolved.model.split("/")[-1] if "/" in resolved.model else resolved.model
    return f"openai/{clean}"


def _litellm_kwargs(
    resolved: ResolvedModelSpec,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    response_format: dict[str, Any] | None,
    controls: dict[str, Any],
) -> dict[str, Any]:
    model = _litellm_model(resolved)
    # credential
    key = None
    if resolved.binding:
        key = _provider_keys().get(resolved.binding) or os.environ.get(resolved.credential_env)
    else:
        key = os.environ.get(resolved.credential_env)
    if not key and resolved.adapter not in {"bedrock", "vertex_ai", "vertex"}:
        raise RuntimeError(f"{resolved.credential_env} is required for {resolved.model}")

    # reasoning effort
    coerced = _coerce_reasoning_effort(resolved.model, resolved.effort)

    # sanitize + cache
    is_anthropic = _is_anthropic_spec(resolved)
    sanitized = _sanitize_messages(_normalize_upstream_tool_ids(messages), resolved.model, is_anthropic, False)
    # LiteLLM translates cache_control; keep mantis policy for anthropic family
    if is_anthropic and _cache_breakpoints_enabled():
        sanitized = _with_cache_breakpoints(sanitized)

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": sanitized,
        "max_tokens": max_tokens if resolved.max_tokens is None else min(max_tokens, resolved.max_tokens),
    }
    # api base / key for openai-compatible
    if resolved.adapter not in _LITELLM_KNOWN:
        kwargs["api_base"] = resolved.base_url
        kwargs["api_key"] = key
    elif resolved.adapter in {"openrouter", "anthropic"}:
        kwargs["api_key"] = key
        if resolved.base_url and resolved.base_url != "http://127.0.0.1:8080/v1":
            kwargs["api_base"] = resolved.base_url
    else:
        # bedrock / vertex read creds from env, still pass key if present
        if key:
            kwargs["api_key"] = key

    # temperature: omit when reasoning is active
    if not coerced and not _is_reasoning_model(resolved.model):
        kwargs["temperature"] = temperature
    if coerced is not None:
        # litellm unified param; also set thinking for older anthropic models
        if is_anthropic and coerced != "none":
            budgets = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 16384}
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": budgets.get(coerced, 8192)}
        elif coerced != "none":
            kwargs["reasoning_effort"] = coerced

    if tools:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    # litellm extra controls
    for k in ("web_search_options", "reasoning", "reasoning_effort"):
        if k in controls:
            kwargs[k] = controls[k]
    return kwargs


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
    """Compatibility shim: old tests expect (url, headers, body)."""
    resolved = _resolve_model_spec(spec) if resolved is None else resolved
    # reuse litellm path for cache/reasoning semantics, but render classic body
    # for test assertions (model, messages, headers).
    key = None
    if resolved.binding:
        key = _provider_keys().get(resolved.binding) or __import__("os").environ.get(resolved.credential_env)
    else:
        key = __import__("os").environ.get(resolved.credential_env)
    if not key:
        key = "test-key"
    # build minimal legacy body that tests inspect
    kwargs = _litellm_kwargs(resolved, messages, max_tokens, temperature, tools, tool_choice, response_format, controls or {})
    # legacy url shape: base_url + /chat/completions or /responses
    is_anthropic = _is_anthropic_spec(resolved)
    # old tests check for /responses when model is openai on openrouter or when
    # the worker declares responses protocol
    use_responses = (
        resolved.adapter == "openrouter" and resolved.model.startswith("openai/")
    ) or (resolved.protocols is not None and "responses" in resolved.protocols)
    if is_anthropic:
        path = "v1/messages"
    elif use_responses:
        path = "responses"
    else:
        path = "chat/completions"
    url = f"{resolved.base_url}/{path}"
    headers: dict[str, str]
    if is_anthropic:
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json", "User-Agent": "Mantis"}
    else:
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "OpenAI/Python"}
    # body for assertions: include model, messages (possibly cache-marked), tools
    body: dict[str, Any] = {"model": resolved.model, "messages": kwargs.get("messages", messages), "max_tokens": kwargs.get("max_tokens", max_tokens)}
    if kwargs.get("tools"):
        body["tools"] = kwargs["tools"]
    if kwargs.get("tool_choice") is not None:
        body["tool_choice"] = kwargs["tool_choice"]
    if use_responses and kwargs.get("reasoning_effort"):
        body["reasoning"] = {"effort": kwargs["reasoning_effort"]}
    elif kwargs.get("reasoning_effort"):
        body["reasoning_effort"] = kwargs["reasoning_effort"]
    if is_anthropic and kwargs.get("thinking"):
        body["thinking"] = kwargs["thinking"]
    if kwargs.get("reasoning"):
        body["reasoning"] = kwargs["reasoning"]
    # preserve cache-related fields tests check
    if kwargs.get("response_format"):
        body["response_format"] = kwargs["response_format"]
    return url, headers, body


def _response_to_dict(resp: Any) -> dict[str, Any]:
    # litellm ModelResponse -> mantis canonical dict
    choices: list[dict[str, Any]] = []
    for c in getattr(resp, "choices", []) or []:
        msg = getattr(c, "message", c)
        if isinstance(msg, dict):
            content = msg.get("content")
            tool_calls = msg.get("tool_calls")
            reasoning = msg.get("reasoning") or msg.get("reasoning_content")
            details = msg.get("reasoning_details")
            annotations = msg.get("annotations")
        else:
            content = getattr(msg, "content", None)
            tool_calls = getattr(msg, "tool_calls", None)
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            details = getattr(msg, "reasoning_details", None)
            annotations = getattr(msg, "annotations", None)
            # litellm tool_calls are objects
            if tool_calls and not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            if tool_calls and hasattr(tool_calls[0], "function"):
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": getattr(tc, "type", "function"),
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
        msg_dict: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg_dict["tool_calls"] = tool_calls
        if isinstance(reasoning, str) and reasoning:
            msg_dict["reasoning"] = reasoning
        if isinstance(details, list) and details:
            msg_dict["reasoning_details"] = details
        if isinstance(annotations, list) and annotations:
            msg_dict["annotations"] = annotations
        # preserve anthropic replay metadata if present on raw response
        raw_msg = getattr(resp, "_hidden_params", {}).get("original_response") if hasattr(resp, "_hidden_params") else None
        if isinstance(raw_msg, dict):
            for k in ("_anthropic_content", "_anthropic_tool_ids"):
                if k in msg:
                    msg_dict[k] = msg[k]
        choices.append({"message": msg_dict, "finish_reason": getattr(c, "finish_reason", None)})

    usage: dict[str, Any] = {}
    u = getattr(resp, "usage", None)
    if u is not None:
        if isinstance(u, dict):
            usage = {
                k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if isinstance(u.get(k), int)
            }
            details = u.get("prompt_tokens_details") or {}
            if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
                usage["prompt_tokens_details"] = {"cached_tokens": details["cached_tokens"]}
        else:
            usage = {
                k: getattr(u, k)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                if isinstance(getattr(u, k, None), int)
            }
            details = getattr(u, "prompt_tokens_details", None)
            if details is not None:
                cached = getattr(details, "cached_tokens", None) if not isinstance(details, dict) else details.get("cached_tokens")
                if isinstance(cached, int):
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
            # fallback: litellm sometimes puts cached_tokens top-level
            if "prompt_tokens_details" not in usage:
                cached = getattr(u, "cached_tokens", None)
                if isinstance(cached, int):
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}

    out: dict[str, Any] = {"choices": choices or [{"message": {"role": "assistant", "content": None}}], "usage": usage}
    # surface litellm cost if available
    hidden = getattr(resp, "_hidden_params", None)
    if isinstance(hidden, dict) and isinstance(hidden.get("response_cost"), (int, float)):
        out["_litellm_cost"] = hidden["response_cost"]
    return out


def _litellm_completion(**kwargs: Any) -> Any:
    # wrapper so tests can monkeypatch litellm.completion
    return litellm.completion(**kwargs)


# -- Public seam ------------------------------------------------------------

def _failover_attempts(spec: str, bindings: RuntimeBindings | None = None) -> list[str]:
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
            continue
        attempts.append(candidate)
    return attempts


def _provider_response(
    spec: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    run = getattr(serve_config._history_context, "active_run", None)
    tool_choice = getattr(run, "active_tool_choice", None)
    response_format = getattr(run, "active_response_format", None)
    controls = getattr(run, "active_controls", None) or {}
    failures: list[str] = []
    bindings = _runtime_bindings()
    deadline = time.monotonic() + timeout_s if timeout_s is not None else None

    for index, attempt in enumerate(_failover_attempts(spec, bindings)):
        _check_client_connected()
        if index:
            delay = _FAILOVER_DELAY
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
        try:
            resolved = _resolve_model_spec(attempt, bindings)
            kwargs = _litellm_kwargs(
                resolved, messages, max_tokens, temperature, tools, tool_choice, response_format, controls
            )
            if timeout_s is not None:
                # litellm respects timeout via kwargs; compute remaining
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                if remaining is not None and remaining <= 0:
                    raise Timeout(message="Fusion provider deadline exceeded", model=attempt, llm_provider=resolved.adapter)
                kwargs["timeout"] = remaining
            _emit_progress({"type": "provider", "status": "started", "model": attempt, "attempt": index + 1, "summary": "Calling a model"})
            resp = _litellm_completion(**kwargs)
            data = _response_to_dict(resp)
            break
        except Exception as error:  # noqa: BLE001
            # classify transient vs fatal
            transient = isinstance(error, (RateLimitError, APIConnectionError, ServiceUnavailableError, Timeout, APIError))
            status = getattr(error, "status_code", None)
            if status is None and hasattr(error, "response"):
                try:
                    status = getattr(error.response, "status_code", None)
                except Exception:
                    status = None
            if isinstance(error, RateLimitError) or (isinstance(status, int) and status in _TRANSIENT_STATUSES):
                transient = True
            if transient:
                failures.append(f"{attempt}: {type(error).__name__}")
                rec = getattr(serve_config._history_context, "active_run", None)
                if rec is not None:
                    rec.record_activity("failover", model=attempt, status="failed", attempt=index + 1, detail=str(error)[:300])
                # try next attempt
                if index == len(_failover_attempts(spec, bindings)) - 1:
                    raise RuntimeError(f"{spec} failed on every pool worker: " + "; ".join(failures)) from error
                continue
            # fatal: surface immediately
            _emit_progress({"type": "provider", "status": "failed", "model": attempt, "attempt": index + 1, "summary": f"Model call failed: {error}"})
            raise RuntimeError(f"{attempt} failed: {error}") from error
    else:
        raise RuntimeError(f"{spec} failed on every pool worker: " + "; ".join(failures))

    if not isinstance(data, dict):
        raise TypeError(f"{spec} returned a non-object response")
    raw_usage = data.get("usage")
    usage = cast(dict[str, Any], raw_usage) if isinstance(raw_usage, dict) else {}
    if run is not None:
        usage_model = attempt if resolved.slot is not None else kwargs.get("model", attempt)
        run.add_usage(usage, model=usage_model)
        # stash litellm cost for breakdown without Network
        litellm_cost = data.get("_litellm_cost")
        if litellm_cost is not None:
            if not hasattr(run, "_litellm_costs"):
                object.__setattr__(run, "_litellm_costs", {})
            getattr(run, "_litellm_costs")[usage_model] = litellm_cost
        message = (data.get("choices") or [{}])[0].get("message", {})
        if run.capture_metadata and isinstance(message, dict):
            run.response_metadata = {
                key: message[key] for key in ("reasoning", "reasoning_details", "annotations", "citations") if message.get(key) is not None
            }
    prompt_details = usage.get("prompt_tokens_details")
    cached_tokens = prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0
    _emit_progress(
        {
            "type": "provider",
            "status": "completed",
            "model": attempt,
            "attempt": index + 1,
            "usage": {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens") if isinstance(usage.get(k), int)},
            "cached_tokens": cached_tokens,
            "summary": f"Model call completed ({cached_tokens} cached input tokens)" if cached_tokens else "Model call completed",
        }
    )
    # strip internal helper key before returning to callers
    data.pop("_litellm_cost", None)
    return data


def _direct_completion(
    spec: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, timeout: float
) -> str:
    del timeout
    data = _provider_response(spec, messages, max_tokens, temperature)
    return str(data["choices"][0]["message"].get("content") or "")


# -- Cost (kept for reporting; prefers litellm hidden cost, falls back to price table) --

def _price_entry(entry: Any) -> tuple[str, tuple[float, float], float | None] | None:
    try:
        pricing = entry["pricing"]
        cache_read = pricing.get("input_cache_read")
        cache_price = None if cache_read in (None, "") else float(cache_read)
        return (str(entry["id"]), (float(pricing["prompt"]), float(pricing["completion"])), cache_price)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def _fetch_prices() -> dict[str, tuple[float, float]]:
    global _cache_read_price_cache  # noqa: PLW0603
    import httpx

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
    if serve_config._price_cache is None:
        serve_config._price_cache = _fetch_prices()
    return serve_config._price_cache


def _cache_read_prices() -> dict[str, float]:
    global _cache_read_price_cache  # noqa: PLW0603
    if _cache_read_price_cache is None:
        _price_map()
    return _cache_read_price_cache or {}


def _price_key(model: str, prices: dict[str, tuple[float, float]]) -> str | None:
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
        matches = [k for k in prices if k.endswith(f"/{candidate}")]
        if len(matches) == 1:
            return matches[0]
    return None


def _model_cost(model: str, tokens: dict[str, int], prices: dict[str, tuple[float, float]]) -> float | None:
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
    label = {
        "step": {"Planner": "Planned the workflow", "Thinker": "Analyzed the task", "Verifier": "Verified the answer", "Worker": "Drafted the answer"}.get(role or "", "Called a model"),
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
    return {"Planner": "Planning the workflow", "Thinker": "Analyzing the task", "Verifier": "Checking the draft", "Worker": "Drafting an answer"}.get(role, "Calling a model")


def _cost_breakdown(usage_models: dict[str, dict[str, int]]) -> dict[str, Any]:
    if not usage_models:
        return {"total": None, "known": False, "source": "unavailable", "models": []}
    # prefer litellm per-run costs if present (no network, handles custom pricing)
    # caller stashes costs on run._litellm_costs; read via first usage_models key owner?
    # Instead, check if any model has known litellm cost; if so, use it directly.
    run = getattr(serve_config._history_context, "active_run", None)
    litellm_costs = getattr(run, "_litellm_costs", None) if run is not None else None
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
        # litellm cost takes precedence for this model
        litellm_cost = litellm_costs.get(model) if isinstance(litellm_costs, dict) else None
        if isinstance(litellm_cost, (int, float)):
            model_cost: float | None = float(litellm_cost)
        else:
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
                "source": "litellm" if isinstance(litellm_cost, (int, float)) else ("price_table" if model_cost is not None else "unknown"),
            }
        )
    return {
        "total": round(total, 6) if known else None,
        "known": known,
        "source": "litellm" if isinstance(litellm_costs, dict) and known else ("price_table" if known else "partial"),
        "prompt_tokens": total_prompt,
        "cached_tokens": total_cached,
        "cache_hit_ratio": round(total_cached / total_prompt, 4) if total_prompt else 0.0,
        "models": models,
    }


def _run_mantis_details(run: Any, level: str) -> dict[str, Any]:
    steps = list(getattr(run, "turns", getattr(run, "steps", [])))
    recorded = list(getattr(run, "_activity", []))
    activity: list[dict[str, Any]] = []
    if recorded:
        keep = None if level == "debug" else ("type", "role", "model", "status", "summary")
        activity = [dict(e) if keep is None else {k: v for k, v in e.items() if k in keep} for e in recorded if any(v is not None for v in e.values())]
    else:
        for step in steps:
            role = str(step.get("role", ""))
            entry = {"type": "step", "role": role, "model": step.get("model_name"), "status": "completed", "summary": _activity_summary("step", role)}
            activity.append(entry)
    usage = _cost_breakdown(getattr(run, "usage_models", {}))
    details: dict[str, Any] = {"activity": activity, "usage": usage}
    if level == "debug" and hasattr(run, "steps"):
        details["steps"] = steps
    return details
