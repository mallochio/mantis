"""Provider dispatch and reasoning replay hooks for Fusion tool delegation."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import fusion_stream
import serve_config

_PRESERVE_REASONING_FIELD = "_mantis_preserve_reasoning"


def request_controls(run: Any) -> tuple[Any, dict[str, Any]]:
    """Read thread-local Fusion controls, falling back to ordinary run controls."""
    scoped = getattr(serve_config._history_context, "fusion_tool_controls", None)
    if isinstance(scoped, dict):
        return scoped["tool_choice"], scoped["controls"]
    return getattr(run, "active_tool_choice", None), getattr(run, "active_controls", None) or {}


def normalize_reasoning(
    message: dict[str, Any],
    keep_reasoning: bool,
    same_model: bool,
    portable_details: Callable[[dict[str, Any]], None],
) -> bool:
    """Apply same-model reasoning replay and normalize ordinary reasoning details."""
    preserve = bool(message.pop(_PRESERVE_REASONING_FIELD, False))
    keep_reasoning = keep_reasoning or (preserve and same_model)
    if keep_reasoning and not preserve:
        portable_details(message)
    return keep_reasoning


def complete_provider(
    kwargs: dict[str, Any],
    on_delta: Callable[[dict[str, Any]], None] | None,
    completion: Callable[..., Any],
    response_to_dict: Callable[[Any], dict[str, Any]],
    check_connected: Callable[[], None],
) -> dict[str, Any]:
    """Dispatch a provider call and consume either its response or live stream."""
    if on_delta is not None:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
    progress = StreamProgress(on_delta)
    try:
        response = completion(**kwargs)
        if on_delta is not None:
            return fusion_stream.consume_provider_stream(response, progress.emit, check_connected)
        return response_to_dict(response)
    except Exception as error:
        if progress.visible:
            raise RuntimeError(
                "Provider stream failed after client-visible output; retry disabled"
            ) from error
        raise


def portable_reasoning_details(message: dict[str, Any], markers: tuple[str, ...]) -> None:
    """Remove endpoint-bound reasoning details from an ordinary replay message."""
    details = message.get("reasoning_details")
    if not isinstance(details, list):
        return
    portable = [
        detail
        for detail in details
        if not isinstance(detail, dict)
        or not any(
            marker in str(detail.get(key, "")).lower()
            for key in ("type", "format")
            for marker in markers
        )
    ]
    if portable:
        message["reasoning_details"] = portable
    else:
        message.pop("reasoning_details", None)


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize worker tool-call IDs and arguments to the canonical wire shape."""
    calls: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        arguments = function.get("arguments") or ""
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(
            {
                "id": str(tool_call.get("id", f"call_{len(calls)}")),
                "type": "function",
                "function": {"name": str(function.get("name", "")), "arguments": arguments},
            }
        )
    return calls


def call_provider(
    response: Callable[..., dict[str, Any]],
    slot: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: list[dict[str, Any]] | None,
    timeout_s: float | None,
    on_delta: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    """Pass optional provider controls only when they are active."""
    options: dict[str, Any] = {}
    if timeout_s is not None:
        options["timeout_s"] = timeout_s
    if on_delta is not None:
        options["on_delta"] = on_delta
    return response(slot, messages, max_tokens, 0.7, tools, **options)


def call_lane(
    run: Any,
    coordinator: Any,
    role: str,
    slot: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply request tool controls only to the lead provider call."""
    context = serve_config._history_context
    previous = getattr(context, "fusion_tool_controls", None)
    context.fusion_tool_controls = {
        "tool_choice": run.tool_choice if role == "main" else None,
        "controls": getattr(run, "controls", {}),
    }
    try:
        return _call_lane(run, coordinator, role, slot, messages, tools)
    finally:
        context.fusion_tool_controls = previous


def _call_lane(
    run: Any,
    coordinator: Any,
    role: str,
    slot: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream tool-delegation lead turns and mark their reasoning for replay."""
    stream_sink = getattr(serve_config._history_context, "fusion_stream_callback", None)
    if role == "main" and run.handoffs is not None and callable(stream_sink):
        if slot == "mantis/base":
            stream_sink(run.run_id, {"_mantis_turn_start": True})
        result = coordinator._call_worker(
            slot, messages, tools, on_delta=lambda delta: stream_sink(run.run_id, delta)
        )
        if slot == "mantis/base":
            stream_sink(run.run_id, {"_mantis_turn_end": True})
    else:
        result = coordinator._call_worker(slot, messages, tools)
    message, usage = result
    if role == "main" and run.handoffs is not None and message.get("reasoning_details"):
        message[_PRESERVE_REASONING_FIELD] = True
    return message, usage


def cache_sidekick_history(run: Any, coordinator: Any, cache: dict[str, Any]) -> None:
    """Carry tool-delegation sidekick history into the next follow-up."""
    if run.handoffs is None or len(run.sidekick_messages) <= 1:
        return
    history = run.sidekick_messages[1:]
    window = getattr(coordinator, "context_window", 0) or 16384
    cache["sidekick"] = (
        coordinator._trim_messages(history, max(1, window // 8))
        if coordinator is not None
        else history[-256:]
    )


class StreamProgress:
    """Track whether a provider attempt has emitted client-visible output."""

    def __init__(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        """Start an attempt with the request's visibility filter when available."""
        self.callback = callback
        self.visible = False
        self.is_visible = getattr(serve_config._history_context, "fusion_stream_is_visible", None)
        self.encoder = fusion_stream.DeltaEncoder()

    def emit(self, delta: dict[str, Any]) -> None:
        """Publish a delta and remember only content visible to the client."""
        if self.callback is None:
            return
        visible = (
            self.is_visible(delta)
            if callable(self.is_visible)
            else bool(self.encoder.encode("", delta, lambda _run_id, call_id: call_id))
        )
        self.visible = self.visible or bool(visible)
        self.callback(delta)
