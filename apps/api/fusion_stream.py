"""Streaming helpers for Fusion provider calls and chat responses."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Queue
from typing import Any, cast

import serve_config
from fusion_stream_delta import DeltaEncoder as DeltaEncoder


class FusionStream:
    """Carry live provider deltas and a completed event to the SSE consumer."""

    def __init__(self) -> None:
        """Create an empty stream queue and a cooperative cancellation signal."""
        self.events: Queue[tuple[str, Any]] = Queue()
        self.cancelled = threading.Event()

    def emit_delta(self, run_id: str, delta: dict[str, Any]) -> None:
        """Queue one lead delta without consuming the producer visibility preview."""
        if not self.cancelled.is_set():
            self.events.put(("delta", (run_id, delta)))


@contextmanager
def bind_stream(
    stream: FusionStream, visibility: Callable[[dict[str, Any]], bool] | None = None
) -> Iterator[None]:
    """Bind the stream callback and disconnect check to the current worker thread."""
    context = serve_config._history_context
    previous_callback = getattr(context, "fusion_stream_callback", None)
    previous_connection = getattr(context, "is_client_connected", None)
    previous_visibility = getattr(context, "fusion_stream_is_visible", None)
    context.fusion_stream_is_visible = visibility
    context.fusion_stream_callback = stream.emit_delta
    context.is_client_connected = lambda: not stream.cancelled.is_set()
    try:
        yield
    finally:
        _restore(context, "fusion_stream_callback", previous_callback)
        _restore(context, "is_client_connected", previous_connection)
        _restore(context, "fusion_stream_is_visible", previous_visibility)


def _restore(context: Any, name: str, value: Any) -> None:
    """Restore one optional history-context attribute."""
    if value is None:
        try:
            delattr(context, name)
        except AttributeError:
            return
    else:
        setattr(context, name, value)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read one response field from either a mapping or an SDK object."""
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _dict(value: Any) -> dict[str, Any]:
    """Convert a response object to a shallow mapping."""
    if isinstance(value, dict):
        return dict(value)
    method = getattr(value, "model_dump", None) or getattr(value, "dict", None)
    return cast(dict[str, Any], method(exclude_none=True)) if method else {}


def _append_tool_calls(target: dict[int, dict[str, Any]], fragments: Any) -> list[dict[str, Any]]:
    """Merge one streamed tool-call fragment and return its public delta form."""
    visible = []
    for ordinal, raw in enumerate(fragments or []):
        fragment = _dict(raw)
        index = int(fragment.get("index", ordinal))
        call = target.setdefault(index, {"id": "", "type": "function"})
        call.setdefault("function", {"name": "", "arguments": ""})
        function = _dict(fragment.get("function"))
        call["id"] = call["id"] or fragment.get("id") or ""
        for key in ("name", "arguments"):
            call["function"][key] += str(function.get(key) or "")
        visible.append({"index": index, **fragment, "function": function})
    return visible


def _usage_dict(raw: Any) -> dict[str, Any]:
    """Normalize streamed token usage from mapping or SDK object responses."""
    result = {
        key: _field(raw, key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if isinstance(_field(raw, key), int)
    }
    details = _field(raw, "prompt_tokens_details")
    cached = _field(details, "cached_tokens")
    if isinstance(cached, int):
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    return result


def _append_details(target: dict[int, dict[str, Any]], fragments: Any) -> list[dict[str, Any]]:
    """Merge indexed reasoning-detail fragments without changing their fields."""
    emitted = []
    for ordinal, raw in enumerate(fragments or []):
        fragment = _dict(raw)
        index = int(fragment.get("index", ordinal))
        detail = target.setdefault(index, {})
        excluded = {"index", "text", "summary"}
        detail.update({key: value for key, value in fragment.items() if key not in excluded})
        if isinstance(fragment.get("text"), str):
            detail["text"] = str(detail.get("text", "")) + fragment["text"]
        if isinstance(fragment.get("summary"), list):
            detail["summary"] = [*detail.get("summary", []), *fragment["summary"]]
        elif "summary" in fragment:
            detail["summary"] = fragment["summary"]
        emitted.append(fragment)
    return emitted


def consume_provider_stream(
    response: Any,
    on_delta: Callable[[dict[str, Any]], None],
    check_connected: Callable[[], None],
) -> dict[str, Any]:
    """Consume a provider stream into the regular worker-response shape."""
    message: dict[str, Any] = {"role": "assistant", "content": ""}
    tools: dict[int, dict[str, Any]] = {}
    details: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    finish: str | None = None
    on_delta({"_mantis_turn_start": True})
    for chunk in response:
        check_connected()
        finish = _consume_chunk(chunk, message, tools, details, usage, finish, on_delta)
    on_delta({"_mantis_turn_end": True})
    if tools:
        message["tool_calls"] = [tools[index] for index in sorted(tools)]
    if details:
        message["reasoning_details"] = [details[index] for index in sorted(details)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


def _consume_chunk(
    chunk: Any,
    message: dict[str, Any],
    tools: dict[int, dict[str, Any]],
    details: dict[int, dict[str, Any]],
    usage: dict[str, Any],
    finish: str | None,
    on_delta: Callable[[dict[str, Any]], None],
) -> str | None:
    """Accumulate one provider chunk and publish its user-visible delta."""
    usage.update(_usage_dict(_field(chunk, "usage")))
    choices = _field(chunk, "choices", []) or []
    if not choices:
        return finish
    choice = choices[0]
    finish = _field(choice, "finish_reason", finish)
    raw_delta = _field(choice, "delta", {}) or {}
    delta = _dict(raw_delta)
    _accumulate_text(message, delta)
    published = _published_delta(delta, tools, details)
    if published:
        on_delta(published)
    return finish


def _accumulate_text(message: dict[str, Any], delta: dict[str, Any]) -> None:
    """Accumulate content and reasoning fields from a provider delta."""
    for key in ("content", "reasoning", "reasoning_content"):
        value = delta.get(key)
        if isinstance(value, str) and value:
            message[key] = str(message.get(key, "")) + value


def _published_delta(
    delta: dict[str, Any], tools: dict[int, dict[str, Any]], details: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    """Normalize supported fields for a live chat-completion delta."""
    published: dict[str, Any] = {}
    if isinstance(delta.get("content"), str) and delta["content"]:
        published["content"] = delta["content"]
    reasoning = delta.get("reasoning") or delta.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        published["reasoning"] = reasoning
        published["reasoning_content"] = reasoning
    tool_fragments = _append_tool_calls(tools, delta.get("tool_calls"))
    if tool_fragments:
        published["tool_calls"] = tool_fragments
    detail_fragments = _append_details(details, delta.get("reasoning_details"))
    if detail_fragments and not any(delta.get(key) for key in ("reasoning", "reasoning_content")):
        detail_text = _reasoning_detail_text(detail_fragments)
        if detail_text:
            published["reasoning"] = detail_text
            published["reasoning_content"] = detail_text
    return published


def _reasoning_detail_text(details: list[dict[str, Any]]) -> str:
    """Extract portable text from streamed reasoning details."""
    pieces: list[str] = []
    for detail in details:
        kind = " ".join(str(detail.get(key, "")).lower() for key in ("type", "format"))
        if any(marker in kind for marker in ("encrypted", "compaction")):
            continue
        summary = detail.get("summary")
        if isinstance(summary, list):
            pieces.extend(
                part["text"]
                for part in summary
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        if isinstance(detail.get("text"), str):
            pieces.append(detail["text"])
    return "".join(pieces)
