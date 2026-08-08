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

import argparse
import hashlib
import json
import os
import pickle
import re
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import jsonschema
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's openfugu-patch overlay, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

from mini import (
    DEFAULT_SLOT_LABELS,
    HEAD_ROWS,
    HIDDEN,
    ROUTER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    THINKER_PROMPT,
    VERIFICATION_PROMPT,
    Coordinator,
    FuguRouter,
)
from ultra import ConductorExecutor, conductor_prompt, parse_workflow, visible_indices

ROUTER: FuguRouter | None = None
_router_lock = threading.Lock()
MODEL_NAME = os.environ.get("MANTIS_MODEL_NAME", "mantis")
MODEL_MODES = {
    MODEL_NAME: "trinity",
    "mantis-trinity": "trinity",
    "trinity": "trinity",
    "fugu": "trinity",
    "mantis-ultra": "conductor",
    "conductor": "conductor",
    "ultra": "conductor",
}
MAX_TURNS = 5
WORKER_TIMEOUT = float(os.environ.get("MANTIS_WORKER_TIMEOUT", "240"))

# These providers reject temperature != 1 when reasoning is enabled.
REASONING_MODELS = ("claude-", "gpt-5.6-")
PROVIDERS = {
    "openrouter": (
        os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        "OPENROUTER_API_KEY",
    ),
    "opencode-go": (
        os.environ.get("OPENCODE_GO_ENDPOINT_URL", "https://opencode.ai/zen/go/v1"),
        "OPENCODE_API_KEY",
    ),
}

_args: argparse.Namespace | None = None
_coordinators: dict[str, object] = {}
_coordinator_lock = threading.Lock()
_history_context = threading.local()
_provider_client = httpx.Client(timeout=WORKER_TIMEOUT)


class ClientDisconnectedError(Exception):
    """Raised when client disconnects during streaming or step execution."""


class RunCapacityError(Exception):
    """Raised instead of evicting a live tool run."""


def _check_client_connected() -> None:
    """Check if current request client connection is broken or aborted."""
    if getattr(_history_context, "aborted", False):
        raise ClientDisconnectedError("Client disconnected")
    is_connected = getattr(_history_context, "is_client_connected", None)
    if is_connected is not None and not is_connected():
        _history_context.aborted = True
        raise ClientDisconnectedError("Client disconnected")


def _parse_model_spec(spec: str) -> tuple[str, str, str | None]:
    """Parse provider/model[|reasoning_effort]."""
    provider, sep, rest = spec.partition("/")
    if not sep or provider not in PROVIDERS:
        raise ValueError(f"model must start with {' or '.join(PROVIDERS)}: {spec}")
    model, marker, effort = rest.partition("|")
    return provider, model, effort if marker and effort != "none" else None


def _is_reasoning_model(model: str) -> bool:
    name = model.rsplit("/", 1)[-1]
    return any(name.startswith(p) for p in REASONING_MODELS)


def _cache_breakpoints_enabled() -> bool:
    value = os.environ.get("MANTIS_CACHE_BREAKPOINTS", "1").lower()
    return value not in {"0", "false", "no", "off"}


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


def _build_request(
    spec: str, messages: list[dict[str, str]], max_tokens: int, temperature: float
) -> tuple[str, dict[str, str], dict[str, Any]]:
    provider, model, effort = _parse_model_spec(spec)
    base_url, key_env = PROVIDERS[provider]
    key = os.environ.get(key_env)
    if not key:
        raise RuntimeError(f"{key_env} is required for {spec}")
    body: dict[str, Any] = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if _cache_breakpoints_enabled() and model.startswith("anthropic/claude-"):
        body["messages"] = _with_cache_breakpoints(messages)
    if effort:
        body["reasoning_effort"] = effort
    if not effort and not _is_reasoning_model(model):
        body["temperature"] = temperature
    return (
        f"{base_url}/chat/completions",
        {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "OpenAI/Python",
        },
        body,
    )


# Provider failures worth retrying on the next pool worker.
_TRANSIENT_STATUSES = {408, 429, 500, 502, 503, 504}
_TRANSIENT_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.StreamError)
_FAILOVER_DELAY = 0.5  # short, constant pause between attempts


def _upstream_streaming_enabled() -> bool:
    """Stream upstream provider responses (SSE) so long generations survive
    pass-through proxies; disable with MANTIS_UPSTREAM_STREAM=0."""
    return os.environ.get("MANTIS_UPSTREAM_STREAM", "1").lower() not in {"0", "false", "no", "off"}


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse one SSE `data:` line; None for comments/junk/keep-alives, {} for [DONE]."""
    stripped = line.strip()
    if not stripped.startswith("data:"):
        return None
    payload = stripped[len("data:"):].strip()
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
    client: Any, url: str, headers: dict[str, str], body: dict[str, Any]
) -> dict:
    """POST with SSE streaming; returns the reassembled completion object."""
    stream_body = dict(body)
    stream_body["stream"] = True
    stream_body["stream_options"] = {"include_usage": True}
    with client.stream("POST", url, headers=headers, json=stream_body) as response:
        response.raise_for_status()
        chunks = (_parse_sse_line(line) for line in response.iter_lines())
        return _assemble_streamed_completion(chunks)


def _failover_attempts(spec: str) -> list[str]:
    """Attempt order on transient failures: the assigned spec, then the rest of
    the configured pool in declared order, each at most once. Falls back to
    [spec] when no provider pool is configured (tests, local workers)."""
    try:
        pool = _configured_slot_models()
    except (TypeError, ValueError):
        return [spec]
    attempts = [spec]
    for candidate in pool:
        if candidate in attempts:
            continue
        try:
            _parse_model_spec(candidate)
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
    run = getattr(_history_context, "active_run", None)
    tool_choice = getattr(run, "active_tool_choice", None)
    response_format = getattr(run, "active_response_format", None)
    controls = getattr(run, "active_controls", None) or {}
    failures: list[str] = []
    for index, attempt in enumerate(_failover_attempts(spec)):
        if index:
            time.sleep(_FAILOVER_DELAY)
        try:
            # Failover switches model/effort/endpoint per spec; messages, tools,
            # and controls stay identical across attempts.
            url, headers, body = _build_request(attempt, messages, max_tokens, temperature)
        except RuntimeError as error:  # provider key missing for this attempt
            failures.append(str(error))
            continue
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if response_format is not None:
            body["response_format"] = response_format
        if "reasoning" in controls:
            body.pop("reasoning_effort", None)
        body.update(controls)
        try:
            if _upstream_streaming_enabled():
                data = _stream_completion(_provider_client, url, headers, body)
            else:
                response = _provider_client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
            break
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status not in _TRANSIENT_STATUSES:
                raise RuntimeError(
                    f"{attempt} returned HTTP {status}: {error.response.text[:500]}"
                ) from error
            failures.append(f"{attempt}: HTTP {status}")
            run_record = getattr(_history_context, "active_run", None)
            if run_record is not None:
                run_record.record_activity(
                    "failover",
                    model=attempt,
                    status="failed",
                    attempt=index + 1,
                )
        except _TRANSIENT_EXCEPTIONS as error:
            failures.append(f"{attempt}: {type(error).__name__}")
            run_record = getattr(_history_context, "active_run", None)
            if run_record is not None:
                run_record.record_activity(
                    "failover",
                    model=attempt,
                    status="failed",
                    attempt=index + 1,
                )
    else:
        raise RuntimeError(f"{spec} failed on every pool worker: " + "; ".join(failures))
    if not isinstance(data, dict):
        raise TypeError(f"{spec} returned a non-object response")
    if run is not None:
        run.add_usage(data.get("usage"), model=body["model"])
        message = (data.get("choices") or [{}])[0].get("message", {})
        if run.capture_metadata and isinstance(message, dict):
            run.response_metadata = {
                key: message[key]
                for key in ("reasoning", "reasoning_details", "annotations", "citations")
                if message.get(key) is not None
            }
    return data


def _direct_completion(
    spec: str, messages: list[dict[str, str]], max_tokens: int, temperature: float, timeout: float
) -> str:
    del timeout  # the shared client uses MANTIS_WORKER_TIMEOUT
    data = _provider_response(spec, messages, max_tokens, temperature)
    return str(data["choices"][0]["message"].get("content") or "")


_PRICES_URL = "https://openrouter.ai/api/v1/models"
_price_cache: dict[str, tuple[float, float]] | None = None
_cache_read_price_cache: dict[str, float] | None = None


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
    global _price_cache
    if _price_cache is None:
        _price_cache = _fetch_prices()
    return _price_cache


def _cache_read_prices() -> dict[str, float]:
    """Per-model cache-read USD prices, populated by the same fetch as the map."""
    global _cache_read_price_cache
    if _cache_read_price_cache is None:
        _price_map()  # the shared lazy fetch fills both caches
    return _cache_read_price_cache or {}


def _model_cost(
    model: str, tokens: dict[str, int], prices: dict[str, tuple[float, float]]
) -> float | None:
    """USD cost for one model's tokens, cache-aware; None when unpriced."""
    price = prices.get(model)
    if price is None:
        return None
    prompt = int(tokens.get("prompt_tokens", 0))
    completion = int(tokens.get("completion_tokens", 0))
    cached = min(int(tokens.get("cached_tokens", 0)), prompt)
    cache_price = _cache_read_prices().get(model, price[0])
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
        "error": "Run ended with an error",
    }.get(activity_type, "Orchestration step")
    return label


def _cost_breakdown(usage_models: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Per-model and aggregate cost with an explicit known flag."""
    if not usage_models:
        return {"total": None, "known": False, "source": "unavailable", "models": []}
    prices = _price_map()
    models: list[dict[str, Any]] = []
    total = 0.0
    known = True
    for model, tokens in usage_models.items():
        model_cost = _model_cost(model, tokens, prices)
        if model_cost is None:
            known = False
            models.append(
                {
                    "model": model,
                    "prompt_tokens": tokens["prompt_tokens"],
                    "completion_tokens": tokens["completion_tokens"],
                    "cached_tokens": tokens.get("cached_tokens", 0),
                    "cost": None,
                    "source": "unknown",
                }
            )
            continue
        total += model_cost
        models.append(
            {
                "model": model,
                "prompt_tokens": tokens["prompt_tokens"],
                "completion_tokens": tokens["completion_tokens"],
                "cached_tokens": tokens.get("cached_tokens", 0),
                "cost": round(model_cost, 6),
                "source": "price_table",
            }
        )
    return {
        "total": round(total, 6) if known else None,
        "known": known,
        "source": "price_table" if known else "partial",
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


class RejectAwareRouter:
    """Force a revision after a verifier rejects instead of re-verifying unchanged text."""

    def __init__(self, router: Any) -> None:
        self._router = router

    def route(self, *args: Any, **kwargs: Any) -> Any:
        result = self._router.route(*args, **kwargs)
        if getattr(_history_context, "force_worker", False):
            _history_context.force_worker = False
            result = {**result, "role_id": 0, "role_name": "Worker"}
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._router, name)


class HistoryWorker:
    """Wrap a worker so every LLM call sees the conversation history.

    The coordinator classes only know the current query; this wrapper prepends
    the prior user/assistant turns (carried in a per-request thread-local) to
    the messages list handed to the underlying worker. This makes multi-turn
    coding sessions work without modifying the upstream Coordinator code."""

    def __init__(self, worker: Any) -> None:
        self._worker = worker

    def _combine(self, messages: Any) -> Any:
        history = getattr(_history_context, "history", None) or []
        if not history or not isinstance(messages, list):
            return messages
        system_context = "\n\n".join(
            str(m.get("content", ""))
            for m in history
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        )
        prior = [m for m in history if isinstance(m, dict) and m.get("role") != "system"]
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            first = messages[0]
            if system_context:
                first = {**first, "content": f"{first.get('content', '')}\n\n{system_context}"}
            return [first] + prior + list(messages[1:])
        if system_context:
            return [{"role": "system", "content": system_context}] + prior + list(messages)
        return prior + list(messages)

    def _model_name(self, agent_id: int) -> str:
        labels = (
            getattr(self._worker, "slot_models", None) or getattr(self._worker, "names", None) or []
        )
        if not labels:
            return f"slot-{agent_id}"
        return str(labels[agent_id % len(labels)])

    def _last_user_prompt(self, messages: Any) -> str:
        if not isinstance(messages, list):
            return ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content", ""))
            if getattr(m, "role", None) == "user":
                return str(getattr(m, "content", ""))
        return ""

    def __call__(self, *args: Any) -> Any:
        _check_client_connected()
        if len(args) == 3:
            role_or_subtask, messages, agent_id = args
            combined = self._combine(messages)
            is_conductor = getattr(_history_context, "conductor_mode", False)
            known_roles = {"Worker", "Thinker", "Verifier"}
            role = role_or_subtask if role_or_subtask in known_roles else "Worker"

            if is_conductor:
                original_prompt = self._last_user_prompt(messages)
                call: dict[str, Any] = {
                    "role": role,
                    "agent_id": agent_id,
                    "model_name": self._model_name(agent_id),
                    "messages": combined,
                    "prompt": original_prompt,
                }
                calls = getattr(_history_context, "calls", None)
                if calls is not None:
                    calls.append(call)
                write_line = getattr(_history_context, "write_line", None)
                turn_index = len(calls) - 1 if calls else 0
                if write_line:
                    write_line(
                        {
                            "type": "step-start",
                            "turn": turn_index,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": call["model_name"],
                            "prompt": call["prompt"],
                        }
                    )
                reply = self._worker(role_or_subtask, combined, agent_id)
                call["reply"] = reply
                if write_line:
                    write_line(
                        {
                            "type": "step-end",
                            "turn": turn_index,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": call["model_name"],
                            "prompt": call["prompt"],
                            "reply": reply,
                        }
                    )

                if reply and reply.strip():
                    return reply

                # Bounded retry once for an empty Conductor node
                retry_instruction = (
                    "\n\nPrevious attempt produced an empty response; produce a complete answer."
                )
                last_msg = (
                    combined[-1]
                    if combined and isinstance(combined[-1], dict)
                    else {"role": "user", "content": ""}
                )
                retry_content = str(last_msg.get("content", "")) + retry_instruction
                retry_messages = combined[:-1] + [{**last_msg, "content": retry_content}]
                retry_prompt = self._last_user_prompt(retry_messages)
                retry_call: dict[str, Any] = {
                    "role": role,
                    "agent_id": agent_id,
                    "model_name": self._model_name(agent_id),
                    "messages": retry_messages,
                    "prompt": retry_prompt,
                }
                if calls is not None:
                    calls.append(retry_call)
                turn_index_retry = len(calls) - 1 if calls else 0
                if write_line:
                    write_line(
                        {
                            "type": "step-start",
                            "turn": turn_index_retry,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": retry_call["model_name"],
                            "prompt": retry_call["prompt"],
                        }
                    )
                reply_retry = self._worker(role_or_subtask, retry_messages, agent_id)
                retry_call["reply"] = reply_retry
                if write_line:
                    write_line(
                        {
                            "type": "step-end",
                            "turn": turn_index_retry,
                            "role": role,
                            "agent_id": agent_id,
                            "model_name": retry_call["model_name"],
                            "prompt": retry_call["prompt"],
                            "reply": reply_retry,
                        }
                    )

                if reply_retry and reply_retry.strip():
                    return reply_retry

                raise ValueError(
                    f"Conductor subtask node {agent_id} returned empty response after retry."
                )

            if role == "Worker":
                feedback = getattr(_history_context, "revision_feedback", None)
                if feedback and combined and isinstance(combined[-1], dict):
                    prev_content = combined[-1].get("content", "")
                    combined[-1] = {
                        **combined[-1],
                        "content": (
                            f"{prev_content}\n\n"
                            f"Revise the answer to address this verifier feedback:\n{feedback}"
                        ),
                    }
                    _history_context.revision_feedback = None
            original_prompt = self._last_user_prompt(messages)
            call = {
                "role": role,
                "agent_id": agent_id,
                "model_name": self._model_name(agent_id),
                "messages": combined,
                "prompt": original_prompt,
            }
            calls = getattr(_history_context, "calls", None)
            if calls is not None:
                calls.append(call)
            write_line = getattr(_history_context, "write_line", None)
            turn_index = len(calls) - 1 if calls else 0
            if write_line:
                write_line(
                    {
                        "type": "step-start",
                        "turn": turn_index,
                        "role": role,
                        "agent_id": agent_id,
                        "model_name": call["model_name"],
                        "prompt": call["prompt"],
                    }
                )
            reply = self._worker(role_or_subtask, combined, agent_id)
            call["reply"] = reply
            if role == "Worker" and not reply.strip():
                _history_context.force_worker = True
                _history_context.revision_feedback = (
                    "Previous worker returned no response; produce a complete answer."
                )
            elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
                _history_context.force_worker = True
                _history_context.revision_feedback = reply
            if write_line:
                write_line(
                    {
                        "type": "step-end",
                        "turn": turn_index,
                        "role": role,
                        "agent_id": agent_id,
                        "model_name": call["model_name"],
                        "prompt": call["prompt"],
                        "reply": reply,
                    }
                )
            return reply
        return self._worker(*args)

    def conduct(self, *args: Any) -> Any:
        _check_client_connected()
        if not (getattr(_history_context, "history", None) or []):
            return self._worker.conduct(*args)
        if len(args) == 2:
            model, messages = args
            return self._worker.conduct(model, self._combine(messages))
        if len(args) == 1:
            (messages,) = args
            return self._worker.conduct(self._combine(messages))
        return self._worker.conduct(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._worker, name)


class DirectTrinityWorker:
    def __init__(
        self,
        slot_models: list[str],
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: float | None = None,
    ) -> None:
        self.slot_models = slot_models
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout if timeout is not None else WORKER_TIMEOUT

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        model = self.slot_models[agent_id % len(self.slot_models)]
        msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
        return _direct_completion(model, msgs, self.max_tokens, self.temperature, self.timeout)


class DirectConductorWorker(DirectTrinityWorker):
    def _call(self, model: str, messages: list) -> str:
        return _direct_completion(model, messages, self.max_tokens, self.temperature, self.timeout)

    def conduct(self, model: str, messages: list) -> str:
        return self._call(model, messages)


class LocalPoolWorker:
    """Serving-time local worker pool — the same protocol the per-step trainer
    used. The Coordinator calls (role_name, messages, agent_id) -> reply; we
    dispatch to model[agent_id % n], each model resident on its own GPU. Replies
    are decoded greedily so serving is deterministic. No external API."""

    def __init__(self, specs: list[tuple[str, str, str]], max_new: int = 384) -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = _torch
        self.max_new = max_new
        self.names: list[str] = []
        self.toks: list[Any] = []
        self.models: list[Any] = []
        self.devs: list[str] = []
        for name, path, dev in specs:
            tk = AutoTokenizer.from_pretrained(path)
            if tk.pad_token is None:
                tk.pad_token = tk.eos_token
            dtype = _torch.bfloat16 if dev.startswith("cuda") else _torch.float32
            m: Any = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
            m = m.to(dev).eval()
            self.names.append(name)
            self.toks.append(tk)
            self.models.append(m)
            self.devs.append(dev)

    def __call__(self, role_name: str, messages: list, agent_id: int) -> str:
        torch = self.torch
        wid = agent_id % len(self.models)
        tk, model, dev = self.toks[wid], self.models[wid], self.devs[wid]
        try:
            text = tk.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (ValueError, TypeError, AttributeError):
            text = "\n".join(m["content"] for m in messages)
        ids = tk(text, return_tensors="pt", truncation=True, max_length=2048).to(dev)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=self.max_new,
                do_sample=False,
                pad_token_id=tk.pad_token_id,
            )
        return str(tk.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True))


def _resolve_conductor_model(worker) -> str:
    """Pick the model used for the Conductor planning call."""
    conductor_model = os.environ.get("MANTIS_CONDUCTOR_MODEL")
    if conductor_model is None and getattr(worker, "slot_models", None):
        conductor_model = worker.slot_models[0]
    if conductor_model is None:
        conductor_model = "openai/gpt-4o-mini"
    return conductor_model


def _run_conductor_workflow(
    worker: Any, query: str, slot_labels: list[str], completion: str, verbose: bool = False
):
    """Parse a Conductor completion and execute the resulting DAG."""
    mids, subs, acc = parse_workflow(completion)
    if not subs:
        raise ValueError(f"Conductor did not emit a parseable workflow. Raw: {completion[:200]}")
    res = ConductorExecutor(worker, slot_labels=slot_labels).execute(
        mids, subs, acc, verbose=verbose
    )
    # expose a turns attribute for _chat_response
    res.turns = res.steps
    return res


class ConductorCoordinator:
    """Per-request Conductor wrapper: one Conductor LM call produces a workflow
    DAG, then ConductorExecutor runs it. Exposes the same .run(query) interface
    as the TRINITY Coordinator."""

    def __init__(self, worker: Any, slot_labels: list[str] | None = None) -> None:
        self.worker = worker
        self.slot_labels = (
            slot_labels or getattr(worker, "slot_models", None) or DEFAULT_SLOT_LABELS
        )

    def _prepare_planning(self, query: str) -> tuple[str, list[dict[str, Any]], Any]:
        conductor_model = _resolve_conductor_model(self.worker)
        prompt_msgs = conductor_prompt(query, self.slot_labels)

        def _get_completion() -> str:
            return str(self.worker.conduct(conductor_model, prompt_msgs))

        return conductor_model, prompt_msgs, _get_completion

    def run(self, query: str, verbose: bool = False):
        _history_context.conductor_mode = True
        try:
            conductor_model, prompt_msgs, get_completion = self._prepare_planning(query)

            calls = getattr(_history_context, "calls", None)
            if calls is None:
                calls = []
                _history_context.calls = calls

            write_line = getattr(_history_context, "write_line", None)
            planner_turn = len(calls)
            planner_call: dict[str, Any] = {
                "role": "Planner",
                "agent_id": 0,
                "model_name": conductor_model,
                "messages": prompt_msgs,
                "prompt": query,
            }
            calls.append(planner_call)

            if write_line:
                write_line(
                    {
                        "type": "step-start",
                        "turn": planner_turn,
                        "role": "Planner",
                        "agent_id": 0,
                        "model_name": conductor_model,
                        "prompt": query,
                    }
                )

            completion = get_completion()
            planner_call["reply"] = completion

            if write_line:
                write_line(
                    {
                        "type": "step-end",
                        "turn": planner_turn,
                        "role": "Planner",
                        "agent_id": 0,
                        "model_name": conductor_model,
                        "prompt": query,
                        "reply": completion,
                    }
                )

            if not completion or not str(completion).strip():
                raise ValueError("Conductor planning returned an empty completion.")

            res = _run_conductor_workflow(self.worker, query, self.slot_labels, completion, verbose)

            if len(calls) > 1:
                turns = []
                for idx, call in enumerate(calls):
                    t = SimpleNamespace(
                        idx=idx,
                        turn=idx,
                        t=idx,
                        agent_id=call.get("agent_id", 0),
                        role=call.get("role", "Worker"),
                        role_name=call.get("role", "Worker"),
                        subtask=call.get("prompt", ""),
                        prompt=call.get("prompt", ""),
                        reply=call.get("reply", ""),
                        text=call.get("reply", ""),
                        model_name=call.get("model_name", ""),
                        sees=[],
                    )
                    turns.append(t)
                res.turns = turns
            else:
                planner_turn_obj = SimpleNamespace(
                    idx=0,
                    turn=0,
                    t=0,
                    agent_id=0,
                    role="Planner",
                    role_name="Planner",
                    subtask=query,
                    prompt=query,
                    reply=completion,
                    text=completion,
                    model_name=conductor_model,
                    sees=[],
                )
                dag_steps = getattr(res, "steps", [])
                for i, s in enumerate(dag_steps, start=1):
                    if hasattr(s, "idx"):
                        s.idx = i
                    if hasattr(s, "t"):
                        s.t = i
                    if hasattr(s, "turn"):
                        s.turn = i
                res.turns = [planner_turn_obj] + dag_steps

            return res
        finally:
            _history_context.conductor_mode = False


def choose_conductor_device(torch_module: Any, env_device: str | None = None) -> str:
    """Pick mps > cuda:0 > cpu, allowing an explicit env override."""
    if env_device and env_device != "auto":
        return env_device
    if torch_module.backends.mps.is_available():
        return "mps"
    if torch_module.cuda.is_available():
        return "cuda:0"
    return "cpu"


def choose_conductor_dtype(device: str, torch_module: Any, env_dtype: str | None = None) -> Any:
    """Return torch dtype for a Conductor device, honoring an explicit override."""
    if env_dtype:
        return getattr(torch_module, env_dtype)
    return torch_module.bfloat16 if device in ("mps", "cuda", "cuda:0") else torch_module.float32


class EnvLocalConductor:
    """Load a GRPO-trained Conductor checkpoint locally with transformers.

    Env overrides: MANTIS_CONDUCTOR_DEVICE (cpu/cuda:0/mps/auto),
                    MANTIS_CONDUCTOR_DTYPE (float32/bfloat16/float16),
                    MANTIS_CONDUCTOR_MAX_NEW.
    Defaults to bfloat16 on mps/cuda and float32 on cpu."""

    def __init__(self, ckpt: str, device: str | None = None, max_new: int | None = None) -> None:
        import torch as _torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = _torch
        self.ckpt = ckpt
        env_device = device if device is not None else os.environ.get("MANTIS_CONDUCTOR_DEVICE")
        self.device = choose_conductor_device(_torch, env_device)
        self.max_new = max_new or int(os.environ.get("MANTIS_CONDUCTOR_MAX_NEW", "512"))
        dtype_env = os.environ.get("MANTIS_CONDUCTOR_DTYPE")
        self.dtype = choose_conductor_dtype(self.device, _torch, dtype_env)
        self.temperature = float(os.environ.get("MANTIS_CONDUCTOR_TEMPERATURE", "0.7"))
        self.top_p = float(os.environ.get("MANTIS_CONDUCTOR_TOP_P", "0.9"))
        do_sample_env = os.environ.get("MANTIS_CONDUCTOR_DO_SAMPLE")
        if do_sample_env:
            self.do_sample = do_sample_env.lower() not in ("0", "false", "no", "")
        else:
            self.do_sample = True
        print(
            f"[serve] loading local Conductor ({ckpt}) on {self.device} dtype={self.dtype} "
            f"do_sample={self.do_sample} temp={self.temperature} top_p={self.top_p} ...",
            flush=True,
        )
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model: Any = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=self.dtype)
        if self.device == "auto" and _torch.cuda.device_count() > 1:
            pass  # leave device_map behavior to from_pretrained
        else:
            self.model = self.model.to(self.device)
        self.model.eval()
        print("[serve] Conductor ready", flush=True)

    def _build_messages(self, messages: list) -> list[dict[str, str]]:
        """Add an assistant prefill that nudges the Conductor into the 3-list format.

        One-shot examples are intentionally avoided here: the 3B Conductor
        checkpoints tend to collapse into repeating a fixed example rather
        than following the actual user query.
        """
        return list(messages) + [{"role": "assistant", "content": "Plan:\n"}]

    def conduct(self, messages: list) -> str:
        torch = self.torch
        messages = self._build_messages(messages)
        try:
            text = self.tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
                truncation=True,
                max_length=2048,
            )
        except (ValueError, TypeError, AttributeError):
            # Fallback for tokenizers without chat_template or old transformers.
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            text = "\n\n".join(parts)
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new,
            "pad_token_id": self.tok.pad_token_id,
        }
        if self.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self.model.generate(**ids, **gen_kwargs)
        completion = str(
            self.tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        )
        print(f"[serve] raw conductor completion: {completion[:1000]!r}", flush=True)
        return completion


class EnvConductorCoordinator(ConductorCoordinator):
    """ConductorCoordinator that can use a local transformers checkpoint
    (Llama-3.2-3B Conductor) or LiteLLM for the planning call."""

    def __init__(
        self,
        worker: Any,
        conductor: EnvLocalConductor | None = None,
        slot_labels: list[str] | None = None,
    ) -> None:
        super().__init__(worker, slot_labels=slot_labels)
        self.local_conductor = conductor

    def _prepare_planning(self, query: str) -> tuple[str, list[dict[str, Any]], Any]:
        prompt_msgs = conductor_prompt(query, self.slot_labels)
        lc = self.local_conductor
        if lc is not None:
            conductor_model = getattr(lc, "ckpt", "local-conductor")

            def _get_completion_local() -> str:
                return str(lc.conduct(prompt_msgs))

            return conductor_model, prompt_msgs, _get_completion_local

        conductor_model = _resolve_conductor_model(self.worker)

        def _get_completion_worker() -> str:
            return str(self.worker.conduct(conductor_model, prompt_msgs))

        return conductor_model, prompt_msgs, _get_completion_worker


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return the text query and history; multimodal content stays in the run."""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        return "", messages
    return _message_text(messages[last_user_idx].get("content")), messages[:last_user_idx]


def _json_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw or b"{}")
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return "" if content is None else str(content)


def _with_images(text: str, content: Any) -> Any:
    if not isinstance(content, list):
        return text
    images = [
        part
        for part in content
        if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    return [{"type": "text", "text": text}, *images] if images else text


def _last_user_content(messages: list[dict[str, Any]]) -> Any:
    return next(
        (message.get("content") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )


def _mode_for_model(model: Any) -> str:
    if not isinstance(model, str) or model not in MODEL_MODES:
        raise ValueError(f"unknown model: {model}")
    return MODEL_MODES[model]


def _public_tool_id(run_id: str, internal_id: str) -> str:
    return f"call_mantis_{run_id}_{internal_id}"


def _parse_public_tool_id(tool_id: Any) -> tuple[str, str] | None:
    if not isinstance(tool_id, str) or not tool_id.startswith("call_mantis_"):
        return None
    rest = tool_id[len("call_mantis_") :]
    run_id, sep, internal_id = rest.partition("_")
    if not sep or len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id.lower()):
        return None
    return run_id, internal_id


def _continuation(messages: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """Extract a Mantis continuation from trailing standard OpenAI tool messages."""
    if not isinstance(messages, list):
        return None
    trailing: list[dict[str, Any]] = []
    i = len(messages) - 1
    while i >= 0 and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
        trailing.append(messages[i])
        i -= 1
    if not trailing or i < 0:
        return None
    assistant = messages[i]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None
    advertised = {
        call.get("id") for call in assistant.get("tool_calls", []) if isinstance(call, dict)
    }
    results: list[dict[str, Any]] = []
    run_id: str | None = None
    for message in reversed(trailing):
        public_id = message.get("tool_call_id")
        parsed = _parse_public_tool_id(public_id)
        if parsed is None or public_id not in advertised:
            raise ValueError("tool result does not belong to a Mantis tool call")
        current_run, internal_id = parsed
        if run_id is not None and current_run != run_id:
            raise ValueError("tool results span multiple Mantis runs")
        run_id = current_run
        results.append(
            {
                "tool_call_id": internal_id,
                "content": _message_text(message.get("content"))[:RUN_MAX_MSG_BYTES],
                "is_error": False,
            }
        )
    return (cast(str, run_id), results)


def _advance_to_boundary(run_id: str, tool_results: Any = None) -> dict[str, Any]:
    event = advance_run(run_id, tool_results)
    for _ in range(128):
        if event.get("type") != "step_complete":
            return event
        event = advance_run(run_id, None)
    return {"type": "error", "error": "orchestration exceeded 128 internal steps"}


def _run_trace(run: Any) -> dict[str, Any]:
    steps = list(getattr(run, "turns", getattr(run, "steps", [])))
    return {
        "mode": run.kind,
        "terminated_by": run.terminated_by,
        "steps": [
            {
                key: step.get(key)
                for key in ("turn", "role", "agent_id", "model_name")
                if step.get(key) is not None
            }
            for step in steps
        ],
    }


def _request_usage(messages: list[dict[str, Any]], completion_text: str) -> dict[str, int]:
    """Estimate per-request usage from the client's messages and this response.

    OpenAI clients (including the prime-agent harness) track their own context
    size from the response usage and compact when it approaches the window. The
    run's accumulated usage spans every internal orchestrator call and tool
    round, so reporting it would make the client see the context grow by the
    full orchestration cost each round and compact repeatedly. Report only the
    current request's context instead, using the same chars/4 estimate the
    client itself applies to messages without usage.
    """
    prompt_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        prompt_chars += len(_message_text(message.get("content")))
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            prompt_chars += len(str(function.get("name", ""))) + len(arguments)
    prompt_tokens = max(1, (prompt_chars + 3) // 4)
    completion_tokens = max(1, (len(completion_text) + 3) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion_response(
    model: str,
    messages: list[dict[str, Any]],
    run: Any,
    event: dict[str, Any],
    details: str = "none",
) -> dict[str, Any]:
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if event.get("type") == "tool_calls":
        message["tool_calls"] = [
            _openai_tool_call(
                str(call.get("name", "")),
                _public_tool_id(run.run_id, str(call.get("id", ""))),
                call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
            )
            for call in event.get("tool_calls", [])
        ]
        finish_reason = "tool_calls"
        completion_text = "".join(call["function"]["arguments"] for call in message["tool_calls"])
    else:
        message["content"] = str(event.get("text", ""))
        message.update(getattr(run, "response_metadata", {}))
        finish_reason = "stop"
        completion_text = message["content"]
    usage: dict[str, Any] = _request_usage(messages, completion_text)
    cost = _usage_cost(getattr(run, "usage_models", {}))
    if cost is not None:
        usage["cost"] = cost
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
    if details != "none":
        body["mantis"] = _run_mantis_details(run, details)
    return body



def _parse_args() -> argparse.Namespace:
    global _args
    if _args is not None:
        return _args
    ap = argparse.ArgumentParser(description="Serve Mantis as one OpenAI-compatible model.")
    ap.add_argument(
        "--model",
        default=os.environ.get("MANTIS_MODEL", "Qwen/Qwen3-0.6B"),
        help="Qwen3-0.6B dir or HF id",
    )
    ap.add_argument(
        "--vector",
        default=os.environ.get("MANTIS_VECTOR", "model_iter_60.npy"),
        help="base vector (19456) — SVF + head",
    )
    ap.add_argument(
        "--head",
        default=os.environ.get("MANTIS_HEAD"),
        help="optional trained head-only vector/safetensors; overrides the "
        "head from --vector after SVF is applied",
    )
    default_workers = os.environ.get("MANTIS_WORKER_MODELS") or os.environ.get(
        "MANTIS_WORKER_MODEL"
    )
    ap.add_argument(
        "--slot-models",
        metavar="CSV",
        default=default_workers,
        help="provider/model[|reasoning_effort] worker specs (CSV); also MANTIS_WORKER_MODELS",
    )
    ap.add_argument(
        "--local-models",
        metavar="CSV",
        default=os.environ.get("MANTIS_LOCAL_MODELS"),
        help="local HF worker model paths (CSV). "
        "Optional 'path@device' per entry; also MANTIS_LOCAL_MODELS",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("MANTIS_HOST", "0.0.0.0"),  # noqa: S104
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MANTIS_PORT", "8088")),
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MANTIS_MAX_TURNS", "5")),
    )
    _args, _ = ap.parse_known_args()  # ignore uvicorn's own argv (api:app --app-dir ...)
    return _args


def get_router() -> FuguRouter:
    global ROUTER
    if ROUTER is None:
        with _router_lock:
            if ROUTER is None:
                args = _parse_args()
                device = os.environ.get("MANTIS_DEVICE")
                print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
                router = FuguRouter(args.model, args.vector, device=device, seed=0)
                if args.head:  # layer a trained head over base SVF
                    head = _load_head(args.head)
                    router.head = (
                        router.torch.from_numpy(head.copy())
                        .float()
                        .reshape(HEAD_ROWS, HIDDEN)
                        .to(router.device)
                    )
                    print(f"[serve] applied trained head from {args.head}", flush=True)
                ROUTER = router
    return ROUTER


def _load_head(path: str) -> np.ndarray:
    """Load a 10240-float head from .npy or from a safetensors file."""
    if path.endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(path, framework="pt") as f:
            head = f.get_tensor("trinity_router_head")
        head = head.to(torch.float32).numpy().reshape(-1)
    else:
        head = np.load(path).astype(np.float64)
    if head.shape != (HEAD_ROWS * HIDDEN,):
        raise ValueError(f"head must be {HEAD_ROWS * HIDDEN} floats, got {head.shape}")
    return np.asarray(head, dtype=np.float64)


def _worker_from_args(args: argparse.Namespace, mode: str) -> Any:
    if args.local_models:
        specs = []
        n_gpu = 0
        try:
            import torch as _torch

            n_gpu = _torch.cuda.device_count() if _torch.cuda.is_available() else 0
        except (ImportError, ModuleNotFoundError):
            pass
        for i, entry in enumerate(args.local_models.split(",")):
            if "@" in entry:
                path, dev = entry.rsplit("@", 1)
            else:
                path = entry
                dev = f"cuda:{(i % max(n_gpu - 1, 1)) + 1}" if n_gpu > 1 else "cpu"
            specs.append((os.path.basename(path.rstrip("/")) or f"w{i}", path, dev))
        return LocalPoolWorker(specs)

    # 4096 tokens to leave room for high reasoning effort (max/xhigh) while still
    # capping cost on long code outputs.
    slot_models = args.slot_models.split(",") if args.slot_models else []
    if not slot_models:
        raise ValueError("MANTIS_WORKER_MODELS is required")
    if mode == "conductor":
        return DirectConductorWorker(slot_models=slot_models, max_tokens=4096)
    if mode == "trinity":
        return DirectTrinityWorker(slot_models=slot_models, max_tokens=4096)
    raise ValueError(f"unknown coordinator mode: {mode}")


def load_coordinator(mode: str):
    global MAX_TURNS
    if mode not in ("trinity", "conductor"):
        raise ValueError(f"unknown coordinator mode: {mode}")
    args = _parse_args()
    MAX_TURNS = args.max_turns
    worker = HistoryWorker(_worker_from_args(args, mode))
    if mode == "trinity":
        return Coordinator(
            RejectAwareRouter(get_router()), worker, max_turns=args.max_turns, sample=True
        )
    local_ckpt = os.environ.get("MANTIS_LOCAL_CONDUCTOR")
    conductor = EnvLocalConductor(local_ckpt) if local_ckpt else None
    return EnvConductorCoordinator(
        worker, conductor=conductor, slot_labels=getattr(worker, "slot_models", None)
    )


def get_coordinator(mode: str):
    if mode not in _coordinators:
        with _coordinator_lock:
            if mode not in _coordinators:
                _coordinators[mode] = load_coordinator(mode)
    return _coordinators[mode]



# ---------------------------------------------------------------------------
# Resumable native-tool runs
# ---------------------------------------------------------------------------
# Stateful orchestration runs back standard Chat Completions tool calls.
# Internal steps stay server-side; the calling harness executes only its own tools.
# State lives in this bounded, expiring in-memory registry.
RUN_TTL = float(os.environ.get("MANTIS_RUN_TTL", "600"))
MAX_TOOL_ROUNDS = int(os.environ.get("MANTIS_MAX_TOOL_ROUNDS_PER_STEP", "8"))
MAX_RUNS = int(os.environ.get("MANTIS_MAX_CONCURRENT_RUNS", "32"))
RUN_MAX_MSG_BYTES = 400_000
RUN_STORE = os.environ.get("MANTIS_RUN_STORE", "memory").lower()
if RUN_STORE not in {"memory", "redis"}:
    raise ValueError("MANTIS_RUN_STORE must be memory or redis")
REDIS_URL = os.environ.get("MANTIS_REDIS_URL", "")
_REDIS_PREFIX = os.environ.get("MANTIS_REDIS_PREFIX", "mantis:run:")
REDIS_LOCK_TIMEOUT = max(300, int(WORKER_TIMEOUT * MAX_TURNS + 60))
_redis_client: Any | None = None

_runs: dict[str, NativeRun] = {}
_runs_lock = threading.Lock()
_runs_sweeper_started = False


def _redis() -> Any:
    global _redis_client
    if _redis_client is None:
        if not REDIS_URL:
            raise RuntimeError("MANTIS_REDIS_URL is required when MANTIS_RUN_STORE=redis")
        try:
            import redis
        except ImportError as error:
            raise RuntimeError("install the redis extra to use MANTIS_RUN_STORE=redis") from error
        _redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
    return _redis_client


def _redis_key(run_id: str) -> str:
    return f"{_REDIS_PREFIX}{run_id}"


def _redis_index_key() -> str:
    return f"{_REDIS_PREFIX}index"


def _redis_get(run_id: str) -> NativeRun | None:
    raw = _redis().get(_redis_key(run_id))
    return pickle.loads(raw) if raw else None  # noqa: S301 - Redis is a trusted deployment dependency


def _redis_put(run: NativeRun) -> None:
    client = _redis()
    payload = pickle.dumps(run, protocol=pickle.HIGHEST_PROTOCOL)
    client.setex(_redis_key(run.run_id), max(1, int(RUN_TTL)), payload)
    client.sadd(_redis_index_key(), run.run_id)
    client.expire(_redis_index_key(), max(1, int(RUN_TTL)))


@contextmanager
def _redis_run_lock(run_id: str):
    lock = _redis().lock(f"{_REDIS_PREFIX}lock:{run_id}", timeout=REDIS_LOCK_TIMEOUT)
    acquired = lock.acquire(blocking=True, blocking_timeout=REDIS_LOCK_TIMEOUT)
    if not acquired:
        raise RunCapacityError("Mantis run is busy")
    try:
        yield
    finally:
        lock.release()


@contextmanager
def _redis_registry_lock():
    lock = _redis().lock(f"{_REDIS_PREFIX}registry-lock", timeout=60)
    acquired = lock.acquire(blocking=True, blocking_timeout=60)
    if not acquired:
        raise RunCapacityError("Mantis tool-run registry is busy")
    try:
        yield
    finally:
        lock.release()


def _sweep_runs() -> None:
    if RUN_STORE == "redis":
        client = _redis()
        now = time.time()
        for raw_id in client.smembers(_redis_index_key()):
            run_id = raw_id.decode() if isinstance(raw_id, bytes) else str(raw_id)
            run = _redis_get(run_id)
            if run is None or now - run.last_active > RUN_TTL:
                client.delete(_redis_key(run_id))
                client.srem(_redis_index_key(), run_id)
        return
    now = time.time()
    stale = [
        rid for rid, run in _runs.items() if run.in_flight == 0 and now - run.last_active > RUN_TTL
    ]
    for rid in stale:
        run = _runs.pop(rid, None)
        if run is not None:
            run.close()


def _ensure_runs_sweeper() -> None:
    global _runs_sweeper_started
    if _runs_sweeper_started:
        return
    _runs_sweeper_started = True

    def _loop() -> None:
        while True:
            time.sleep(RUN_TTL / 2 if RUN_TTL > 0 else 60)
            with _runs_lock:
                _sweep_runs()

    threading.Thread(target=_loop, daemon=True).start()


def _register_run(run: NativeRun) -> str:
    if RUN_STORE == "redis":
        with _redis_registry_lock():
            client = _redis()
            if _redis_get(run.run_id) is not None:
                raise ValueError("run id already exists")
            if client.scard(_redis_index_key()) >= MAX_RUNS:
                _sweep_runs()
            if client.scard(_redis_index_key()) >= MAX_RUNS:
                raise RunCapacityError("Mantis tool-run capacity is full")
            _redis_put(run)
        return cast(str, run.run_id)
    with _runs_lock:
        _ensure_runs_sweeper()
        if run.run_id in _runs:
            raise ValueError("run id already exists")
        if len(_runs) >= MAX_RUNS:
            _sweep_runs()
        if len(_runs) >= MAX_RUNS:
            raise RunCapacityError("Mantis tool-run capacity is full")
        _runs[run.run_id] = run
    return cast(str, run.run_id)


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    """Validate and copy standard OpenAI function tools."""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        out.append({"type": "function", "function": dict(function)})
    return out


def _model_completion(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Call the selected provider directly; return (text, tool_calls)."""
    data = _provider_response(model, messages, 4096, 0.7, tools)
    msg = data["choices"][0]["message"]
    text = str(msg.get("content") or "")
    tcs = msg.get("tool_calls") or []
    calls: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for tc in tcs:
        fn = tc.get("function", {})
        raw_arguments = fn.get("arguments")
        try:
            args = json.loads(raw_arguments) if raw_arguments else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        call_id = str(tc.get("id") or f"tc_{uuid.uuid4().hex}")
        if call_id in seen_ids:
            call_id = f"tc_{uuid.uuid4().hex}"
        seen_ids.add(call_id)
        calls.append(
            {
                "id": call_id,
                "name": str(fn.get("name")),
                "arguments": args,
            }
        )
    return text, calls


def _openai_tool_call(name: str, _id: str, arguments: dict) -> dict[str, Any]:
    return {
        "id": _id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def _validate_tool_results(tool_results: Any, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list")
    if not all(isinstance(result, dict) for result in tool_results):
        raise ValueError("each tool result must be an object")
    results = cast(list[dict[str, Any]], tool_results)
    ids = [result.get("tool_call_id") for result in results]
    if not all(isinstance(tool_id, str) and tool_id for tool_id in ids):
        raise ValueError("each tool result requires a tool_call_id")
    string_ids = cast(list[str], ids)
    if len(string_ids) != len(set(string_ids)) or set(string_ids) != expected_ids:
        raise ValueError(
            f"tool result id mismatch: expected {sorted(expected_ids)} got {sorted(string_ids)}"
        )
    return results


def _configured_slot_models(override: Any = None) -> list[str]:
    value = override
    if value is None:
        configured = getattr(_args, "slot_models", None) if _args is not None else None
        configured = (
            configured
            or os.environ.get("MANTIS_WORKER_MODELS")
            or os.environ.get("MANTIS_WORKER_MODEL")
        )
        value = configured.split(",") if configured else list(DEFAULT_SLOT_LABELS)
    if not isinstance(value, list):
        raise TypeError("slot_models must be a non-empty list of model names")
    models = [model.strip() for model in value if isinstance(model, str) and model.strip()]
    if len(models) != len(value) or not models:
        raise ValueError("slot_models must be a non-empty list of model names")
    return models


_learning_lock = threading.Lock()
_TEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*|\s)(?:python\d*\s+-m\s+(?:pytest|unittest)|pytest|npm\s+(?:run\s+)?test|"
    r"pnpm\s+(?:run\s+)?test|yarn\s+test|bun\s+test|cargo\s+test|go\s+test|dotnet\s+test|"
    r"mvn\s+test|gradle\s+test)(?:\s|$)",
    re.IGNORECASE,
)
_SECRET_PATTERNS = (
    re.compile(r"\b(?:sk|hf)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\b(?:AKIA[A-Z0-9]{16}|github_pat_[A-Za-z0-9_]{20,})\b"),
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|xox[baprs]-[A-Za-z0-9-]{20,})\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)\b(api[_ -]?key|token|password|secret|authorization)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+\S+"),
)


def _learning_enabled() -> bool:
    return os.environ.get("MANTIS_LEARNING", "").lower() in {"1", "true", "yes", "on"}


def _redact_learning_task(task: str) -> str:
    redacted = task[: int(os.environ.get("MANTIS_LEARNING_MAX_TASK_CHARS", "12000"))]
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _learning_path() -> Path:
    root = Path(
        os.path.expanduser(os.environ.get("MANTIS_LEARNING_DIR", "~/.local/share/mantis/learning"))
    )
    instance = os.environ.get("MANTIS_LEARNING_INSTANCE") or socket.gethostname()
    safe_instance = re.sub(r"[^A-Za-z0-9_.-]", "_", instance)[:80] or "local"
    return root / f"runs-{safe_instance}.jsonl"


def _tool_call_by_id(pending: dict[str, Any], tool_call_id: str) -> dict[str, Any] | None:
    for call in pending.get("asst", {}).get("tool_calls", []):
        if isinstance(call, dict) and call.get("id") == tool_call_id:
            return cast(dict[str, Any], call)
    return None


def _learning_record(run: NativeRun, event: dict[str, Any]) -> dict[str, Any]:
    turns = cast(list[dict[str, Any]], getattr(run, "turns", getattr(run, "steps", [])))
    final_worker = next((turn for turn in reversed(turns) if turn.get("role") == "Worker"), None)
    tests = [item for item in run.tool_observations if item["is_test"]]
    last_test_passed = bool(tests) and not tests[-1]["is_error"]
    accepted = event.get("terminated_by") == "verifier_accept"
    trainable = bool(run.kind == "trinity" and accepted and last_test_passed and final_worker)
    task = _redact_learning_task(str(getattr(run, "query", "")))
    return {
        "schema_version": 1,
        "timestamp": int(time.time()),
        "run_id": run.run_id,
        "mode": run.kind,
        "task": task,
        "task_hash": hashlib.sha256(task.encode()).hexdigest(),
        "pool": list(getattr(run, "slot_models", [])),
        "terminated_by": event.get("terminated_by", event.get("type", "")),
        "duration_seconds": round(time.time() - run.created, 3),
        "turn_count": len(turns),
        "test_seen": bool(tests),
        "last_test_passed": last_test_passed,
        "tool_error_count": sum(item["is_error"] for item in run.tool_observations),
        "verifier_accepted": accepted,
        "trainable": trainable,
        "label_worker": (
            final_worker.get("agent_id") if trainable and final_worker is not None else None
        ),
        "label_role": 0 if trainable else None,
    }


def _write_learning_record(run: NativeRun, event: dict[str, Any]) -> None:
    if not _learning_enabled():
        return
    with _learning_lock:
        if run.learning_logged:
            return
        path = _learning_path()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        os.chmod(path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(_learning_record(run, event), ensure_ascii=False) + "\n")
        run.learning_logged = True


class NativeRun:
    """Base for a resumable orchestration run."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.created = time.time()
        self.last_active = time.time()
        self.cancelled = False
        self.finished = False
        self.final_text = ""
        self.terminated_by: str | None = None
        self.kind = "run"
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_events: dict[str, dict[str, Any]] = {}
        self.tool_observations: list[dict[str, Any]] = []
        self.learning_logged = False
        self.in_flight = 0
        self.tool_choice: Any = None
        self.active_tool_choice: Any = None
        self.response_format: dict[str, Any] | None = None
        self.active_response_format: dict[str, Any] | None = None
        self.controls: dict[str, Any] = {}
        self.active_controls: dict[str, Any] = {}
        self.capture_metadata = False
        self.response_metadata: dict[str, Any] = {}
        self.usage: dict[str, Any] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.usage_models: dict[str, dict[str, int]] = {}
        self._activity: list[dict[str, Any]] = []
        self._started_monotonic = time.monotonic()

    def record_activity(
        self,
        activity_type: str,
        *,
        role: str | None = None,
        model: str | None = None,
        status: str = "completed",
        duration_ms: float | None = None,
        summary: str | None = None,
        attempt: int | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "type": activity_type,
            "role": role,
            "model": model,
            "status": status,
            "summary": summary or _activity_summary(activity_type, role),
        }
        if duration_ms is not None:
            entry["duration_ms"] = round(duration_ms, 1)
        if attempt is not None:
            entry["attempt"] = attempt
        self._activity.append(entry)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("lock", None)
        state.pop("request_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self.request_lock = threading.Lock()
        self.request_events = getattr(self, "request_events", {})

    def add_usage(self, usage: Any, model: str | None = None) -> None:
        if not isinstance(usage, dict):
            return
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                self.usage[key] += value
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            target = self.usage.setdefault("completion_tokens_details", {})
            for key, value in details.items():
                if isinstance(value, int) and value >= 0:
                    target[key] = target.get(key, 0) + value
        if model:
            model_usage = self.usage_models.setdefault(
                model, {"prompt_tokens": 0, "completion_tokens": 0}
            )
            for key in ("prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if isinstance(value, int) and value >= 0:
                    model_usage[key] += value
        prompt_details = usage.get("prompt_tokens_details")
        if model and isinstance(prompt_details, dict):
            cached = prompt_details.get("cached_tokens")
            if isinstance(cached, int) and cached > 0:
                model_usage = self.usage_models.setdefault(
                    model, {"prompt_tokens": 0, "completion_tokens": 0}
                )
                model_usage["cached_tokens"] = model_usage.get("cached_tokens", 0) + cached

    def validate_output(self, text: str) -> None:
        if not self.response_format:
            return
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"structured output is not valid JSON: {error.msg}") from error
        if self.response_format.get("type") == "json_schema":
            schema = self.response_format.get("json_schema", {}).get("schema", {})
            try:
                jsonschema.validate(value, schema)
            except jsonschema.ValidationError as error:
                message = f"structured output does not match schema: {error.message}"
                raise ValueError(message) from error

    def touch(self) -> None:
        self.last_active = time.time()

    def advance(self, tool_results: Any) -> dict[str, Any]:
        raise NotImplementedError

    def advance_idempotent(self, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
        if request_id is None:
            return self.advance(tool_results)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cached
            event = self.advance(tool_results)
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event

    def record_tool_results(
        self, pending: dict[str, Any], tool_results: list[dict[str, Any]]
    ) -> None:
        for result in tool_results:
            call = _tool_call_by_id(pending, str(result.get("tool_call_id", ""))) or {}
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}
            command = arguments.get("command", "") if isinstance(arguments, dict) else ""
            self.tool_observations.append(
                {
                    "name": str(function.get("name", "")),
                    "is_error": bool(result.get("is_error", False)),
                    "is_test": bool(
                        function.get("name") == "bash"
                        and isinstance(command, str)
                        and _TEST_COMMAND.search(command)
                    ),
                }
            )

    def close(self) -> None:
        self.cancelled = True


class TrinityRun(NativeRun):
    """Resumable TRINITY loop with native tool support.

    Replicates Coordinator semantics (role sampling, Thinker suggestion,
    Verifier accept/reject, cold-verifier -> Worker, empty-response recovery,
    multi-turn history) but lets each role's model call client-provided tools before its text
    reply finalizes."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_turns: int = MAX_TURNS,
    ) -> None:
        super().__init__(run_id)
        self.kind = "trinity"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_turns = max_turns
        query, history = _split_messages(messages)
        self.query = query or ""
        self.query_content = _last_user_content(messages)
        self.history = history
        self.obs = self.query
        self.ref_id = 0
        self.last_response: str | None = None
        self.suggestion: str | None = None
        self.suggested_role: str | None = None
        self.force_worker = False
        self.revision_feedback: str | None = None
        self.turns: list[dict[str, Any]] = []
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _model_name(self, agent_id: int) -> str:
        return str(self.slot_models[agent_id % len(self.slot_models)])

    def _route(self) -> tuple[str, int]:
        msgs = [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT.format(num_agents=len(self.slot_models)),
            },
            {"role": "user", "content": self.obs},
        ]
        r = get_router().route(msgs, sample=True)
        role = r["role_name"]
        if self.suggested_role:
            role, self.suggested_role = self.suggested_role, None
        if self.force_worker:
            self.force_worker = False
            role = "Worker"
        if role == "Verifier" and self.last_response is None:
            role = "Worker"  # nothing to verify yet [FC]
        if role == "Thinker" and self.last_response is None:
            role = "Worker"  # a Thinker with no response to reason about is noise
        return role, int(r["agent_id"])

    def _role_prompt(self, role: str) -> str:
        if role == "Thinker":
            info = self.query
            if self.last_response:
                info += f"\n\nCurrent response:\n{self.last_response}"
            return cast(str, THINKER_PROMPT.format(info=info))
        if role == "Verifier":
            vp = VERIFICATION_PROMPT.format(query=self.query, response=self.last_response or "")
            if self.suggestion:
                vp += (
                    f"These are useful suggestions when drafting your response:\n"
                    f"<suggestion>{self.suggestion}</suggestion>"
                )
            return cast(str, vp)
        content = self.query
        if self.suggestion:
            content += (
                f"when drafting your response, thinking of following:\n"
                f"<suggestion>{self.suggestion}</suggestion>"
            )
        return cast(str, content)

    def _build_messages(self, role: str) -> list[dict[str, Any]]:
        prior_sys = "\n\n".join(
            str(m.get("content", ""))
            for m in self.history
            if isinstance(m, dict) and m.get("role") == "system" and m.get("content")
        )
        sys_content = SYSTEM_PROMPT
        if prior_sys:
            sys_content = f"{sys_content}\n\n{prior_sys}"
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        msgs: list[dict[str, Any]] = [{"role": "system", "content": sys_content}]
        msgs.extend(prior)
        user_content = self._role_prompt(role)
        if role == "Worker" and self.revision_feedback:
            user_content = (
                f"{user_content}\n\n"
                f"Revise the answer to address this verifier feedback:\n{self.revision_feedback}"
            )
            self.revision_feedback = None
        msgs.append(
            {
                "role": "user",
                "content": _with_images(user_content, self.query_content)
                if role == "Worker"
                else user_content,
            }
        )
        return msgs

    def _role_complete(self, role: str, agent_id: int, turn: int, messages: list, reply: str):
        if role == "Worker":
            self.last_response = reply
            self.suggestion = None
            thought = self._extract_thought(reply)
            if thought:
                self.obs += (
                    f"\n<reference_thought_{self.ref_id}>{thought}"
                    f"</reference_thought_{self.ref_id}>"
                )
                self.ref_id += 1
        elif role == "Thinker":
            self.suggested_role, self.suggestion = self._parse_thinker(reply)
        elif role == "Verifier":
            self.suggestion = None
            if self._parse_verification(reply):
                self.terminated_by = "verifier_accept"
                self.final_text = self.last_response or reply
        if role == "Worker" and not reply.strip():
            self.force_worker = True
            nope = "produce a complete answer."
            self.revision_feedback = f"Previous worker returned no response; {nope}"
        elif role == "Verifier" and reply.strip().upper().startswith("REJECT"):
            self.force_worker = True
            self.revision_feedback = reply
        step = {
            "turn": turn,
            "role": role,
            "agent_id": agent_id,
            "model_name": self._model_name(agent_id),
            "prompt": messages[-1]["content"] if messages else "",
            "reply": reply,
        }
        self.turns.append(step)
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        return {"type": "step_complete", **step}

    def _run_model(self, role: str, agent_id: int, turn: int, messages: list):
        model = self._model_name(agent_id)
        self.active_response_format = self.response_format if role == "Worker" else None
        self.active_tool_choice = (
            self.tool_choice if role == "Worker" and self._tool_rounds == 0 else None
        )
        self.active_controls = self.controls if role == "Worker" else {}
        self.capture_metadata = role == "Worker"
        started = time.monotonic()
        try:
            text, calls = _model_completion(model, messages, self.tools)
            duration_ms = (time.monotonic() - started) * 1000.0
        except Exception:
            self.record_activity(
                "step",
                role=role,
                model=model,
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            raise
        finally:
            self.active_response_format = None
            self.active_tool_choice = None
            self.active_controls = {}
            self.capture_metadata = False
        self.record_activity(
            "step",
            role=role,
            model=model,
            duration_ms=duration_ms,
        )
        if calls:
            self.record_activity("tool_call", role=role, model=model)
            asst: dict[str, Any] = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    _openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            self._pending = {
                "role": role,
                "agent_id": agent_id,
                "turn": turn,
                "messages": messages,
                "asst": asst,
            }
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "agent_id": agent_id,
                "model_name": model,
                "turn": turn,
                "tool_calls": calls,
            }
        return self._role_complete(role, agent_id, turn, messages, text)

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = _validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        failed = any(bool(result.get("is_error")) for result in tool_results)
        self.record_activity(
            "tool_result",
            role=str(pending.get("role") or ""),
            model=str(pending.get("model") or pending.get("model_name") or ""),
            status="failed" if failed else "completed",
        )
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any],
                        self._run_model(p["role"], p["agent_id"], p["turn"], p["messages"]),
                    )

                # Finish conditions before starting a new coordinator turn.
                if self.terminated_by is not None:
                    self.finished = True
                    return self._final()
                if len(self.turns) >= self.max_turns:
                    self.terminated_by = "max_turns"
                    self.finished = True
                    return self._final()

                turn = len(self.turns)
                role, agent_id = self._route()
                messages = self._build_messages(role)
                return cast(dict[str, Any], self._run_model(role, agent_id, turn, messages))
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001 - model/network failures abort the run
                self.close()
                return {"type": "error", "error": str(e)}

    def _final(self) -> dict[str, Any]:
        text = self.final_text
        if not text:
            # Reasoning workers can return content:null when effort eats the
            # budget; fall back to the last non-empty reply so the client
            # never receives an empty final answer.
            for step in reversed(self.turns):
                if step.get("reply", "").strip():
                    text = step["reply"]
                    break
        return {
            "type": "final",
            "text": text,
            "terminated_by": self.terminated_by or "",
            "steps": self.turns,
        }

    @staticmethod
    def _extract_thought(reply: str) -> str:
        return reply.strip()

    @staticmethod
    def _parse_thinker(text: str):
        import re

        role = None
        m = re.search(
            r"<suggested_role>\s*(solver|thinker|verifier)\s*</suggested_role>",
            text,
            re.IGNORECASE,
        )
        if m:
            role = {"solver": "Worker", "thinker": "Thinker", "verifier": "Verifier"}[
                m.group(1).lower()
            ]
        sug = None
        s = re.search(r"<suggestion>\s*([\s\S]*?)\s*</suggestion>", text, re.IGNORECASE)
        if s:
            sug = s.group(1).strip() or None
        return role, sug

    @staticmethod
    def _parse_verification(text: str) -> bool:
        return text.strip().upper().startswith("ACCEPT")


class ConductorRun(NativeRun):
    """Resumable Conductor run: planning step then DAG nodes, all tool-capable."""

    def __init__(
        self,
        run_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        slot_models: list[str] | None = None,
        max_steps: int = 5,
    ) -> None:
        super().__init__(run_id)
        self.kind = "conductor"
        self.tools = tools
        self.slot_models = slot_models or list(DEFAULT_SLOT_LABELS)
        self.max_steps = max_steps
        query, history = _split_messages(messages)
        self.query = query or ""
        self.query_content = _last_user_content(messages)
        self.history = history
        self.conductor_model = _resolve_conductor_model(
            SimpleNamespace(slot_models=self.slot_models)
        )
        self.steps: list[dict[str, Any]] = []
        self._workflow: tuple[list, list, list] | None = None
        self._outputs: list[str] = []
        self._next_node = 0
        self._pending: dict[str, Any] | None = None
        self._expected_ids: set[str] = set()
        self._tool_rounds = 0

    def _planner_messages(self) -> list[dict[str, Any]]:
        prior = [m for m in self.history if isinstance(m, dict) and m.get("role") != "system"]
        return cast(
            list[dict[str, Any]],
            conductor_prompt(self.query, self.slot_models) + prior[-4:],
        )

    def _node_messages(self, node_index: int, mid: int, sub: str) -> list[dict[str, Any]]:
        if self._workflow is None:
            raise ValueError("workflow required")
        sees = visible_indices(self._workflow[2], node_index)
        mids = self._workflow[0]
        subs = self._workflow[1]
        ctx = ""
        for j in sees:
            prev_mid = mids[j]
            ctx += (
                f"\n<Subtask assigned to Agent {prev_mid}>{subs[j]}"
                f"</Subtask assigned to Agent {prev_mid}>"
                f"\n<Agent {prev_mid} response>{self._outputs[j].strip()}"
                f"</Agent {prev_mid} response>"
            )
        user = (
            f"USER QUESTION context:\n{ctx}\n\nYour subtask: {sub}"
            if ctx
            else f"Your subtask: {sub}"
        )
        return [{"role": "user", "content": _with_images(user, self.query_content)}]

    def _run_model(self, role: str, model: str, messages: list) -> dict[str, Any]:
        seq = len(self.steps)
        is_final_worker = bool(
            role == "Worker"
            and self._workflow is not None
            and self._next_node >= len(self._workflow[1])
        )
        self.active_response_format = self.response_format if is_final_worker else None
        self.active_tool_choice = (
            self.tool_choice if role == "Worker" and self._tool_rounds == 0 else None
        )
        self.active_controls = self.controls if role == "Worker" else {}
        self.capture_metadata = is_final_worker
        started = time.monotonic()
        try:
            text, calls = _model_completion(model, messages, self.tools)
            duration_ms = (time.monotonic() - started) * 1000.0
        except Exception:
            self.record_activity(
                "step",
                role=role,
                model=model,
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000.0,
            )
            raise
        finally:
            self.active_response_format = None
            self.active_tool_choice = None
            self.active_controls = {}
            self.capture_metadata = False
        self.record_activity(
            "step",
            role=role,
            model=model,
            duration_ms=duration_ms,
        )
        if calls:
            self.record_activity("tool_call", role=role, model=model)
            asst = {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    _openai_tool_call(c["name"], c["id"], c["arguments"]) for c in calls
                ],
            }
            self._pending = {"role": role, "model": model, "messages": messages, "asst": asst}
            self._expected_ids = {c["id"] for c in calls}
            self._tool_rounds = 1
            return {
                "type": "tool_calls",
                "role": role,
                "model_name": model,
                "turn": seq,
                "tool_calls": calls,
            }
        return self._finalize_text(role, text, seq)

    def _finalize_text(self, role: str, text: str, seq: int) -> dict[str, Any]:
        self._pending = None
        self._expected_ids = set()
        self._tool_rounds = 0
        if role == "Planner":
            try:
                self._workflow = parse_workflow(text)
            except Exception as e:  # noqa: BLE001
                return {
                    "type": "error",
                    "error": f"Conductor did not emit a parseable workflow: {e}",
                }
            mids, subs, access = self._workflow
            if not subs or not (len(mids) == len(subs) == len(access)):
                return {
                    "type": "error",
                    "error": "Conductor emitted an empty or malformed workflow",
                }
            self.steps.append(
                {
                    "turn": seq,
                    "role": "Planner",
                    "agent_id": 0,
                    "model_name": self.conductor_model,
                    "prompt": self.query,
                    "reply": text,
                }
            )
            return {
                "type": "step_complete",
                "turn": seq,
                "role": "Planner",
                "agent_id": 0,
                "model_name": self.conductor_model,
                "prompt": self.query,
                "reply": text,
            }

        node_index = self._next_node - 1
        if self._workflow is None:
            raise ValueError("workflow required")
        mids = self._workflow[0]
        subs = self._workflow[1]
        mid = int(mids[node_index]) % len(self.slot_models)
        self._outputs.append(text)
        self.steps.append(
            {
                "turn": seq,
                "role": "Worker",
                "agent_id": mid,
                "model_name": self.slot_models[mid],
                "prompt": subs[node_index],
                "reply": text,
            }
        )
        return {
            "type": "step_complete",
            "turn": seq,
            "role": "Worker",
            "agent_id": mid,
            "model_name": self.slot_models[mid],
            "prompt": subs[node_index],
            "reply": text,
        }

    def _apply_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = self._pending
        if pending is None:
            raise RuntimeError("no pending tool call")  # noqa: TRY301
        tool_results = _validate_tool_results(tool_results, self._expected_ids)
        self.record_tool_results(pending, tool_results)
        failed = any(bool(result.get("is_error")) for result in tool_results)
        self.record_activity(
            "tool_result",
            role=str(pending.get("role") or ""),
            model=str(pending.get("model") or pending.get("model_name") or ""),
            status="failed" if failed else "completed",
        )
        self._tool_rounds += 1
        if self._tool_rounds > MAX_TOOL_ROUNDS:
            raise ValueError(f"exceeded max tool rounds per step ({MAX_TOOL_ROUNDS})")
        pending["messages"].append(pending["asst"])
        for r in tool_results:
            pending["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": r.get("tool_call_id"),
                    "content": str(r.get("content", ""))[:RUN_MAX_MSG_BYTES],
                }
            )
        self._expected_ids = set()
        self._pending = pending

    def advance(self, tool_results: Any) -> dict[str, Any]:
        with self.lock:
            self.touch()
            if self.cancelled:
                return {"type": "error", "error": "run cancelled"}
            if self.finished:
                return {"type": "error", "error": "run already finished"}
            try:
                if self._pending is None and tool_results is not None:
                    return {"type": "error", "error": "unexpected tool results"}
                if self._pending is not None:
                    if tool_results is None:
                        return {"type": "error", "error": "missing tool results"}
                    self._apply_tool_results(tool_results)
                    p = self._pending
                    if p is None:
                        raise RuntimeError("no pending tool call")  # noqa: TRY301
                    return cast(
                        dict[str, Any], self._run_model(p["role"], p["model"], p["messages"])
                    )

                if self._workflow is None:
                    return self._run_model(
                        "Planner", self.conductor_model, self._planner_messages()
                    )
                mids, subs, access = self._workflow
                if self._next_node >= len(subs):
                    self.finished = True
                    self.terminated_by = "conductor_done"
                    return self._final_conductor("conductor_done")
                if self._next_node >= self.max_steps:
                    self.finished = True
                    self.terminated_by = "max_steps"
                    return self._final_conductor("max_steps")
                node_index = self._next_node
                self._next_node += 1
                mid = int(mids[node_index]) % len(self.slot_models)
                model = self.slot_models[mid]
                return self._run_model(
                    "Worker", model, self._node_messages(node_index, mid, subs[node_index])
                )
            except ValueError as e:
                return {"type": "error", "error": str(e)}
            except Exception as e:  # noqa: BLE001
                self.close()
                return {"type": "error", "error": str(e)}

    def close(self) -> None:
        self.cancelled = True

    def _final_conductor(self, terminated_by: str) -> dict[str, Any]:
        text = ""
        for out in reversed(self._outputs):
            if str(out or "").strip():
                text = str(out)
                break
        self.final_text = text
        return {
            "type": "final",
            "text": text,
            "terminated_by": terminated_by,
            "steps": self.steps,
        }


def create_run(mode: str, body: dict[str, Any]) -> NativeRun:
    messages = body.get("messages") or []
    if (
        not isinstance(messages, list)
        or not messages
        or not all(isinstance(message, dict) for message in messages)
    ):
        raise ValueError("messages must be a non-empty list of objects")
    tools = _convert_tools(body.get("tools"))
    slot_models = _configured_slot_models(body.get("slot_models"))
    requested_id = body.get("run_id")
    if requested_id is not None and (
        not isinstance(requested_id, str)
        or len(requested_id) != 32
        or any(char not in "0123456789abcdef" for char in requested_id.lower())
    ):
        raise ValueError("run_id must be a 32-character hexadecimal string")
    run_id = requested_id or uuid.uuid4().hex
    run: NativeRun
    if mode == "conductor":
        run = ConductorRun(run_id, messages, tools, slot_models=slot_models)
    else:
        run = TrinityRun(run_id, messages, tools, slot_models=slot_models)
    run.tool_choice = body.get("tool_choice")
    run.response_format = body.get("response_format")
    output_limit = body.get("max_completion_tokens", body.get("max_tokens"))
    if output_limit is not None:
        run.controls["max_tokens"] = output_limit
    if body.get("reasoning"):
        run.controls["reasoning"] = body["reasoning"]
    elif body.get("reasoning_effort") is not None:
        run.controls["reasoning_effort"] = body["reasoning_effort"]
    if body.get("web_search_options") is not None:
        run.controls["web_search_options"] = body["web_search_options"]
    _register_run(run)
    return run


def get_run(run_id: str) -> NativeRun:
    run = _redis_get(run_id) if RUN_STORE == "redis" else _runs.get(run_id)
    if run is None:
        raise KeyError(f"unknown or expired run: {run_id}")
    return run


def advance_run(run_id: str, tool_results: Any, request_id: Any = None) -> dict[str, Any]:
    lock = _redis_run_lock(run_id) if RUN_STORE == "redis" else nullcontext()
    with lock:
        run = get_run(run_id)
        run.in_flight += 1
        if RUN_STORE == "redis":
            _redis_put(run)
        try:
            _history_context.active_run = run
            event = run.advance_idempotent(tool_results, request_id)
            if event.get("type") in ("final", "error"):
                _write_learning_record(run, event)
            return event
        finally:
            _history_context.active_run = None
            run.in_flight -= 1
            run.touch()
            if RUN_STORE == "redis":
                _redis_put(run)


def delete_run(run_id: str) -> bool:
    if RUN_STORE == "redis":
        with _redis_run_lock(run_id):
            run = _redis_get(run_id)
            if run is None:
                return False
            _redis().delete(_redis_key(run_id))
            _redis().srem(_redis_index_key(), run_id)
    else:
        with _runs_lock:
            run = _runs.pop(run_id, None)
        if run is None:
            return False
    _write_learning_record(run, {"type": "error", "terminated_by": "deleted"})
    run.close()
    return True
