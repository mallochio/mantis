"""Background execution and live SSE delivery for Fusion chat."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Any, cast

import fusion_api_response as response
import fusion_stream
import providers
from fastapi import HTTPException


@dataclass
class LiveChatCallbacks:
    """Carry explicit HTTP adapter operations into the live stream."""

    compute: Callable[[], dict[str, Any]]
    release: Callable[[], None]
    completed: Callable[[dict[str, Any]], Iterator[bytes]]
    encode_delta: Callable[[fusion_stream.DeltaEncoder, str, dict[str, Any]], list[dict[str, Any]]]
    encode_tool_id: Callable[[str, str], str]


def iter_chat_stream(
    request: Any,
    base: dict[str, Any],
    callbacks: LiveChatCallbacks,
    hidden_names: set[str],
    keepalive_seconds: float,
    include_reasoning: bool,
) -> Iterator[bytes]:
    """Run tool-based Fusion in the background and stream its lead deltas."""
    preview = fusion_stream.DeltaEncoder(hidden_names)
    stream = fusion_stream.FusionStream()
    visibility = partial(_is_visible, preview, include_reasoning)
    threading.Thread(target=_worker, args=(callbacks, stream, visibility), daemon=True).start()
    try:
        yield from _stream_items(request, base, callbacks, hidden_names, stream, keepalive_seconds)
    finally:
        stream.cancelled.set()


def _preview_tool_id(_run_id: str, call_id: str) -> str:
    """Return a placeholder public ID without storing preview state on the run."""
    return call_id


def _is_visible(
    preview: fusion_stream.DeltaEncoder, include_reasoning: bool, delta: dict[str, Any]
) -> bool:
    """Classify a provider fragment without changing persisted public IDs."""
    return bool(response.encode_deltas(preview, "", delta, include_reasoning, _preview_tool_id))


def _worker(
    callbacks: LiveChatCallbacks,
    stream: fusion_stream.FusionStream,
    visibility: Callable[[dict[str, Any]], bool],
) -> None:
    """Compute a response while provider callbacks feed the SSE queue."""
    try:
        with fusion_stream.bind_stream(stream, visibility):
            stream.events.put(("result", callbacks.compute()))
    except (
        HTTPException,
        providers.ClientDisconnectedError,
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        OSError,
    ) as error:
        stream.events.put(("error", error))
    finally:
        callbacks.release()


def _next_item(stream: fusion_stream.FusionStream, timeout: float) -> tuple[str, Any]:
    """Return the next event or a keepalive marker when the queue is idle."""
    try:
        return cast(tuple[str, Any], stream.events.get(timeout=timeout))
    except queue.Empty:
        return "keepalive", None


def _stream_items(
    request: Any,
    base: dict[str, Any],
    callbacks: LiveChatCallbacks,
    hidden_names: set[str],
    stream: fusion_stream.FusionStream,
    keepalive_seconds: float,
) -> Iterator[bytes]:
    """Yield each queue event as keepalive, provider delta, result, or error."""
    encoder = fusion_stream.DeltaEncoder(hidden_names)
    state: dict[str, Any] = {"started": False, "visible": False, "content": False}
    while True:
        kind, value = _next_item(stream, keepalive_seconds)
        if kind == "keepalive":
            yield response._fusion_stream_keepalive()
        elif kind == "delta":
            yield from _delta_chunks(encoder, value, base, callbacks, state)
        elif kind == "result":
            yield from _result_chunks(request, value, base, state, encoder, callbacks)
            return
        else:
            yield from response._fusion_stream_error(base, value)
            return


def _delta_chunks(
    encoder: fusion_stream.DeltaEncoder,
    value: tuple[str, dict[str, Any]],
    base: dict[str, Any],
    callbacks: LiveChatCallbacks,
    state: dict[str, Any],
) -> Iterator[bytes]:
    """Turn one queued provider delta into public SSE chunks."""
    run_id, raw_delta = value
    if raw_delta.get("_mantis_turn_start"):
        state["content"] = False
    for delta in callbacks.encode_delta(encoder, run_id, raw_delta):
        if not state["started"]:
            yield response._fusion_stream_chunk(base, {"role": "assistant"}, None)
            state["started"] = True
        state["visible"] = True
        state["content"] = state["content"] or bool(delta.get("content"))
        yield response._fusion_stream_chunk(base, delta, None)


def _result_chunks(
    request: Any,
    event: dict[str, Any],
    base: dict[str, Any],
    state: dict[str, Any],
    encoder: fusion_stream.DeltaEncoder,
    callbacks: LiveChatCallbacks,
) -> Iterator[bytes]:
    """Complete a live stream or fall back to completed-event chunks."""
    if not state["visible"]:
        yield from callbacks.completed(event)
        return
    fusion_event = event.get("mantis_event")
    final_base = {**base, **({"mantis_event": fusion_event} if fusion_event else {})}
    finish, completion_text = response._fusion_stream_completion(event)
    delta = _completion_delta(event, completion_text, state, encoder, callbacks)
    yield response._fusion_stream_chunk(final_base, delta, finish)
    if request.stream_options and request.stream_options.include_usage:
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        yield response.usage_chunk(final_base, messages, completion_text)
    yield b"data: [DONE]\n\n"


def _completion_delta(
    event: dict[str, Any],
    completion_text: str,
    state: dict[str, Any],
    encoder: fusion_stream.DeltaEncoder,
    callbacks: LiveChatCallbacks,
) -> dict[str, Any]:
    """Return final client tool calls or any unstreamed completion content."""
    if event.get("status") == "awaiting_tools":
        return {
            "tool_calls": response._missing_fusion_tool_calls(
                event, str(event.get("run_id") or ""), encoder, callbacks.encode_tool_id
            )
        }
    if event.get("status") != "completed" or not state["content"]:
        return {"content": completion_text}
    return {}
