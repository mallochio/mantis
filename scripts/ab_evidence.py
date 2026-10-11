"""Extract canonical tool evidence from routed-run transcripts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_EXIT_PATTERNS = (
    re.compile(
        r"(?:^|\n)\s*exit(?:[_ ](?:code|status))?\s*[:=]\s*(-?\d+)",
        re.IGNORECASE,
    ),
    re.compile(r"exit\s+status\s+(-?\d+)", re.IGNORECASE),
)


def _walk(value: Any) -> Iterator[dict[str, Any]]:
    """Yield nested dictionaries and list items from JSON-shaped data."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _identifier(value: dict[str, Any]) -> str:
    """Return the provider-supplied canonical ID for a call or result."""
    for key in ("tool_call_id", "call_id", "id"):
        identifier = value.get(key)
        if isinstance(identifier, str) and identifier:
            return identifier
    return ""


def _call_arrays(value: Any) -> list[dict[str, Any]]:
    """Find tool calls in a transcript without synthesizing missing calls."""
    calls: list[dict[str, Any]] = []
    for node in _walk(value):
        for key in ("tool_calls", "pending_tool_calls"):
            entries = node.get(key)
            if isinstance(entries, list):
                calls.extend(item for item in entries if isinstance(item, dict))
    return calls


def _result_arrays(value: Any) -> list[dict[str, Any]]:
    """Find tool results in a transcript without parsing summary text as output."""
    results: list[dict[str, Any]] = []
    for node in _walk(value):
        entries = node.get("tool_results")
        if isinstance(entries, list):
            results.extend(item for item in entries if isinstance(item, dict))
    return results


def _message_records(
    messages: list[Any],
    calls: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    """Index tool messages from the client-side conversation."""
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    _remember(calls, order, call)
        if message.get("role") == "tool":
            _remember(results, order, message)


def _event_records(
    sources: list[Any],
    calls: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
    order: list[str],
) -> None:
    """Index server-side calls and results from responses and run details."""
    for source in sources:
        for call in _call_arrays(source):
            _remember(calls, order, call)
        for result in _result_arrays(source):
            _remember(results, order, result)


def tool_records(observation: dict[str, Any]) -> list[dict[str, Any]]:
    """Join client and server tool records once per canonical call ID."""
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    messages = observation.get("messages") or []
    sources = [
        observation.get("response_payloads") or [],
        observation.get("server_run_details") or {},
    ]
    _message_records(messages, calls, results, order)
    _event_records(sources, calls, results, order)
    return [
        {
            "tool_call_id": _identifier(calls.get(identifier) or results.get(identifier, {}))
            or None,
            "call": calls.get(identifier),
            "result": results.get(identifier),
        }
        for identifier in order
    ]


def _remember(records: dict[str, dict[str, Any]], order: list[str], item: dict[str, Any]) -> None:
    """Retain the richest observation once for each ID or exact ID-less call."""
    identifier = _identifier(item)
    key = identifier or json.dumps(item, sort_keys=True, default=str)
    if key not in records:
        records[key] = item
        if key not in order:
            order.append(key)
    elif len(json.dumps(item, default=str)) > len(json.dumps(records[key], default=str)):
        records[key] = item


def call_command(call: dict[str, Any] | None) -> tuple[str, str]:
    """Return the observed tool name and shell command from a call record."""
    if not isinstance(call, dict):
        return "", ""
    function = call.get("function") or {}
    if not isinstance(function, dict):
        return "", ""
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    command = arguments.get("command") if isinstance(arguments, dict) else ""
    return str(function.get("name") or ""), str(command or "")


def result_text(result: Any) -> str:
    """Return observed output text without interpreting a failure marker."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("content", "output", "stdout", "text", "result"):
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = result_text(value)
                if nested:
                    return nested
    return ""


def exit_code(result: Any) -> int | None:
    """Read a structured exit code or a recognized shell output field."""
    if isinstance(result, dict):
        for key in ("exit_code", "exit_status", "returncode", "exit"):
            value = result.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value.strip()):
                return int(value)
        for key in ("content", "output", "stdout", "text", "result"):
            value = result.get(key)
            code = exit_code(value)
            if code is not None:
                return code
    text = result_text(result)
    for pattern in _EXIT_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def sidekick_handoff_count(observation: dict[str, Any]) -> int:
    """Count unique sidekick delegations reported in run events."""
    handoffs: list[dict[str, Any]] = []
    sources: list[Any] = [
        observation.get("response_payloads") or [],
        observation.get("server_run_details") or {},
    ]
    for source in sources:
        for node in _walk(source):
            entries = node.get("handoff_history")
            if isinstance(entries, list):
                handoffs.extend(
                    entry
                    for entry in entries
                    if isinstance(entry, dict)
                    and str(entry.get("kind", "")).lower() in {"delegate", "sidekick"}
                )
    unique = {json.dumps(item, sort_keys=True, default=str) for item in handoffs}
    records = tool_records(observation)
    sidekick_calls = sum(call_command(item.get("call"))[0] == "sidekick" for item in records)
    return max(sidekick_calls, len(unique))


def tool_use_count(observation: dict[str, Any]) -> int:
    """Count calls and sidekick delegation as tool use, without double-counting."""
    records = tool_records(observation)
    sidekicks = sidekick_handoff_count(observation)
    sidekick_calls = sum(call_command(item.get("call"))[0] == "sidekick" for item in records)
    return len(records) + max(0, sidekicks - sidekick_calls)
