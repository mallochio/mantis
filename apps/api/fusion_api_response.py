"""SSE encoding for completed and live Fusion chat responses."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import fusion_stream
import utils
from fastapi.responses import JSONResponse


def _fusion_stream_keepalive() -> bytes:
    """Return a standards-compliant SSE comment keepalive."""
    return b": keep-alive\n\n"


def _fusion_stream_chunk(base: dict[str, Any], delta: dict[str, Any], finish: str | None) -> bytes:
    """Encode one Fusion chat-completion stream chunk."""
    payload = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _fusion_stream_completion(event: dict[str, Any]) -> tuple[str, str]:
    """Return finish reason and the visible completion text for usage accounting."""
    if event.get("status") == "completed":
        return "stop", str(event.get("report") or "")
    if event.get("status") == "awaiting_tools":
        calls = event.get("pending_tool_calls") or []
        text = "".join(
            f"{call['function'].get('name', '')}:{call['function'].get('arguments', '')}"
            for call in calls
        )
        return "tool_calls", text
    return "stop", str(event.get("report") or "Fusion run failed")


def _fusion_stream_error(base: dict[str, Any], error: Exception) -> Iterator[bytes]:
    """Report an orchestration error and end the live stream."""
    payload = {
        **base,
        "error": {"message": str(error), "type": "upstream_error"},
        "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
    }
    yield f"data: {json.dumps(payload)}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _missing_fusion_tool_calls(
    event: dict[str, Any],
    run_id: str,
    encoder: fusion_stream.DeltaEncoder,
    encode_tool_id: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """Add pending client calls that were not emitted by the streamed lead."""
    calls = event.get("pending_tool_calls") or []
    index = max(encoder.calls, default=-1) + 1
    output = []
    for call in calls:
        call_id = str(call.get("id") or "")
        if call_id in encoder.emitted_call_ids:
            continue
        output.append(
            {
                "index": index,
                "id": encode_tool_id(run_id, call_id),
                "type": call.get("type", "function"),
                "function": call.get("function") or {},
            }
        )
        index += 1
    return output


def stream_base(request_id: str, created: int) -> dict[str, Any]:
    """Return common fields for a Fusion chat-completion chunk."""
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": "mantis/fusion",
    }


def usage_chunk(
    base: dict[str, Any], messages: list[dict[str, Any]], completion_text: str
) -> bytes:
    """Encode the optional usage-only chunk."""
    usage = utils._request_usage(messages, completion_text)
    return f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n".encode()


def pending_tool_calls(
    event: dict[str, Any], encode_tool_id: Callable[[str, str], str]
) -> list[dict[str, Any]]:
    """Return completed-event client tool calls with public IDs."""
    return [
        {
            "index": index,
            "id": encode_tool_id(event["run_id"], call.get("id", "")),
            "type": call.get("type", "function"),
            "function": call.get("function", {}),
        }
        for index, call in enumerate(event.get("pending_tool_calls") or [])
    ]


def event_completion(
    event: dict[str, Any], encode_tool_id: Callable[[str, str], str]
) -> tuple[dict[str, Any], str, str]:
    """Return the completed-event delta, finish reason, and usage text."""
    status = event.get("status")
    if status == "awaiting_tools":
        calls = pending_tool_calls(event, encode_tool_id)
        text = "".join(
            f"{call['function'].get('name', '')}:{call['function'].get('arguments', '')}"
            for call in calls
        )
        return {"tool_calls": calls}, "tool_calls", text
    if status == "completed":
        report = event.get("report") or ""
        return {"content": report}, "stop", report
    message = event.get("report") or f"fusion run failed with status {status}"
    return {"content": message}, "stop", message


def iter_chat_event(
    base: dict[str, Any],
    event: dict[str, Any],
    messages: list[dict[str, Any]],
    include_usage: bool,
    encode_tool_id: Callable[[str, str], str],
) -> Iterator[bytes]:
    """Yield role, optional reasoning, final payload, and optional usage."""
    fusion_event = event.get("mantis_event")
    if fusion_event:
        base = {**base, "mantis_event": fusion_event}
    delta, finish, completion_text = event_completion(event, encode_tool_id)
    yield _fusion_stream_chunk(base, {"role": "assistant"}, None)
    if event.get("status") in {"completed", "awaiting_tools"}:
        reasoning_trace = event.get("reasoning_trace")
        if reasoning_trace:
            yield _fusion_stream_chunk(base, {"reasoning": reasoning_trace}, None)
    yield _fusion_stream_chunk(base, delta, finish)
    if include_usage:
        yield usage_chunk(base, messages, completion_text)


def completion_headers(request_id: str, event: dict[str, Any]) -> dict[str, str]:
    """Return request, run, and final model routing headers."""
    headers = {"X-Request-Id": request_id}
    if event.get("run_id"):
        headers["X-Mantis-Run-Id"] = str(event["run_id"])
    slot_models = event.get("usage_models")
    if isinstance(slot_models, dict) and slot_models:
        headers["x-route-model"] = list(slot_models.keys())[-1]
    elif isinstance(slot_models, list) and slot_models:
        headers["x-route-model"] = str(slot_models[-1])
    return headers


def completion_message(
    event: dict[str, Any], encode_tool_id: Callable[[str, str], str]
) -> tuple[dict[str, Any], str, str]:
    """Return assistant payload and usage text for a successful event."""
    event = {**event, "run_id": str(event.get("run_id") or "")}
    delta, finish, text = event_completion(event, encode_tool_id)
    calls = delta.get("tool_calls")
    if calls is not None:
        delta = {
            "content": "",
            "tool_calls": [
                {key: value for key, value in call.items() if key != "index"} for call in calls
            ],
        }
    message = {"role": "assistant", **delta}
    if event.get("reasoning_trace"):
        message["reasoning"] = event["reasoning_trace"]
    return message, finish, text


def failed_chat_response(event: dict[str, Any]) -> JSONResponse:
    """Return a failed Fusion event as the established HTTP error envelope."""
    status = event.get("status")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": event.get("report") or f"fusion run failed with status {status}",
                "type": "upstream_error",
            }
        },
    )


def build_chat_response(
    request_id: str,
    model: str,
    event: dict[str, Any],
    messages: list[dict[str, Any]],
    encode_tool_id: Callable[[str, str], str],
) -> JSONResponse:
    """Build an OpenAI chat completion for a completed Fusion event."""
    status = event.get("status")
    if status not in {"completed", "awaiting_tools"}:
        return failed_chat_response(event)
    message, finish, text = completion_message(event, encode_tool_id)
    fusion_event = event.get("mantis_event")
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": utils._request_usage(messages, text),
            **({"mantis_event": fusion_event} if fusion_event else {}),
        },
        headers=completion_headers(request_id, event),
    )


def encode_deltas(
    encoder: fusion_stream.DeltaEncoder,
    run_id: str,
    delta: dict[str, Any],
    include_reasoning: bool,
    encode_tool_id: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """Filter reasoning according to the established trace opt-in and scope."""
    output = cast(list[dict[str, Any]], encoder.encode(run_id, delta, encode_tool_id))
    if include_reasoning:
        return output
    return [
        {key: value for key, value in item.items() if key not in {"reasoning", "reasoning_content"}}
        for item in output
        if item.get("content") or item.get("tool_calls")
    ]
