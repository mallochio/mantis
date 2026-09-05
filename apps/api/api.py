"""Typed HTTP adapter for the Mantis orchestration engine."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import base_proxy
import fusion
import httpx
import model_catalog
import openai
import providers
import serve
import utils
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi import Request as HttpRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fusion_types import FusionRunBudget, FusionToolOptions, FusionWorkerProfile
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    function: FunctionDefinition


class FunctionChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)


class NamedToolChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["function"]
    function: FunctionChoice


class TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="ignore")

    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] | None = None


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["image_url"]
    image_url: ImageURL


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart | ImagePart] | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_role_fields(self) -> Message:
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")
        if self.role != "assistant" and self.tool_calls is not None:
            raise ValueError("tool_calls are only valid on assistant messages")
        return self


class JsonSchemaDefinition(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class JsonObjectFormat(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["json_object"]


class ReasoningOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    exclude: bool | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str = serve.MODEL_NAME
    messages: list[Message] = Field(min_length=1)
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: JsonSchemaFormat | JsonObjectFormat | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning: ReasoningOptions | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    web_search_options: dict[str, Any] | None = None
    # Fusion-specific orchestration controls.
    worker_profiles: list[FusionWorkerProfile] | None = None
    budget: FusionRunBudget | None = None
    tool_options: FusionToolOptions | None = None
    # Session identity accepted from the body so conversations can reach
    # Switchyard. Never forwarded upstream: unknown top-level fields can be
    # rejected by strict providers. See base_proxy.router_body.
    user: str | None = None
    metadata: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_tools(self) -> ChatRequest:
        if self.tool_choice not in (None, "none") and not self.tools:
            raise ValueError("tool_choice requires tools")
        if isinstance(self.tool_choice, NamedToolChoice):
            names = {tool.function.name for tool in self.tools or []}
            if self.tool_choice.function.name not in names:
                raise ValueError("tool_choice function must be present in tools")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("set only one of max_tokens or max_completion_tokens")
        if self.reasoning is not None and self.reasoning_effort is not None:
            raise ValueError("set reasoning.effort or reasoning_effort, not both")
        return self


_MAX_REQUESTS = int(os.environ.get("MANTIS_MAX_CONCURRENT_REQUESTS", "32"))
_MAX_BODY_BYTES = int(os.environ.get("MANTIS_MAX_BODY_BYTES", str(50 * 1024 * 1024)))
_KEEPALIVE_SECONDS = float(os.environ.get("MANTIS_SSE_KEEPALIVE_SECONDS", "10"))
_STREAM_EVENTS_DEFAULT = os.environ.get("MANTIS_STREAM_EVENTS", "1").lower() not in {
    "0",
    "false",
    "no",
    "off",
}
_FINAL_CHUNK_DELAY_SECONDS = max(
    0.0, float(os.environ.get("MANTIS_FINAL_CHUNK_DELAY_MS", "5")) / 1000.0
)
_capacity = threading.BoundedSemaphore(_MAX_REQUESTS)


class BodyLimitMiddleware:
    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length and int(content_length) > self.max_bytes:
            await _error(413, "request body exceeds limit", "invalid_request_error")(
                scope, receive, send
            )
            return
        received = 0

        async def limited_receive() -> dict:
            nonlocal received
            message = cast(dict[str, Any], await receive())
            received += len(message.get("body", b""))
            if received > self.max_bytes:
                raise OverflowError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except OverflowError:
            await _error(413, "request body exceeds limit", "invalid_request_error")(
                scope, receive, send
            )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    serve._provider_client.close()


app = FastAPI(title="Mantis", version="0.3.0", lifespan=lifespan)
app.add_middleware(BodyLimitMiddleware, max_bytes=_MAX_BODY_BYTES)


def _error(status: int, message: str, error_type: str) -> JSONResponse:
    return JSONResponse(
        status_code=status, content={"error": {"message": message, "type": error_type}}
    )


def _authorize(authorization: str | None = Header(default=None)) -> None:
    expected = os.environ.get("MANTIS_API_KEY")
    if not expected:
        raise HTTPException(503, "MANTIS_API_KEY is not configured")
    supplied = authorization[7:] if authorization and authorization.startswith("Bearer ") else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(401, "unauthorized")


def _advance(request: ChatRequest, body: dict[str, Any]) -> tuple[Any, str, dict[str, Any]]:
    mode = serve._mode_for_model(request.model)
    continuation = serve._continuation(body["messages"])
    run_id: str | None = None
    try:
        if continuation is None:
            run = serve.create_run(mode, body)
            run_id = run.run_id
            event = serve._advance_to_boundary(run_id)
        else:
            run_id, tool_results = continuation
            run = serve.get_run(run_id)
            if run.kind != mode:
                raise ValueError("model does not match the active Mantis run")
            event = serve._advance_to_boundary(run_id, tool_results)
    except serve.ClientDisconnectedError:
        if run_id:
            serve.delete_run(run_id, error="client disconnected")
        raise
    if event.get("type") == "error":
        raise RuntimeError(str(event.get("error", "orchestration failed")))
    # External stores deserialize a fresh run object on every advance.
    # Reload so response metadata and aggregate usage come from the updated state.
    if serve.RUN_STORE != "memory":
        run = serve.get_run(run_id)
    return run, run_id, event


_DETAIL_LEVELS = ("none", "summary", "debug")


def _detail_level(headers: dict[str, str] | None) -> str:
    value = (headers or {}).get("x-mantis-details", "none").strip().lower()
    if value == "debug" and os.environ.get("MANTIS_ALLOW_DEBUG_TRACE", "0") != "1":
        return "summary"
    return value if value in _DETAIL_LEVELS else "none"


def _fusion_return_reasoning(headers: dict[str, str] | None) -> bool:
    """Return True if the client has opted in to receiving Fusion reasoning traces."""
    value = (headers or {}).get("x-mantis-return-reasoning", "").strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return fusion._return_reasoning_by_default()


def _mantis_headers(mantis: dict[str, Any], body: dict[str, Any]) -> dict[str, str]:
    headers = {
        "X-Mantis-Run-Id": str(mantis.get("run_id", "")),
        "X-Mantis-Mode": str(mantis.get("mode", "")),
        "X-Mantis-Outcome": str(mantis.get("outcome", "")),
    }
    duration = mantis.get("duration_ms")
    if isinstance(duration, (int, float)):
        headers["X-Mantis-Duration-Ms"] = str(int(duration))
    cost = mantis.get("usage", {}).get("total")
    if isinstance(cost, (int, float)):
        headers["X-Mantis-Cost-Usd"] = f"{cost:.6f}"
    return headers


_AZURE_ROUTER_MODEL = "mantis/azure-router"
_AZURE_ROUTER_ALIAS = "azure-router"
_AZURE_ROUTER_PROVIDER = "azure-foundry-router"
_AZURE_ROUTER_DEPLOYMENT = "model-router"
_AZURE_SESSION_LIMIT = 1024
_azure_sessions: dict[str, tuple[str, list[dict[str, Any]]]] = {}


def _azure_router_spec(request: ChatRequest) -> str:
    effort = _azure_router_effort(request)
    if effort:
        return f"{_AZURE_ROUTER_PROVIDER}/{_AZURE_ROUTER_DEPLOYMENT}|{effort}"
    return f"{_AZURE_ROUTER_PROVIDER}/{_AZURE_ROUTER_DEPLOYMENT}"


def _azure_router_effort(request: ChatRequest) -> str:
    effort = request.reasoning_effort
    if effort is None and request.reasoning is not None:
        effort = request.reasoning.effort
    return effort or "medium"


def _azure_router_max_tokens(request: ChatRequest) -> int:
    cap = 131072
    value = request.max_completion_tokens or request.max_tokens
    if value is not None:
        return min(value, cap)
    return cap


def _azure_part_text(part: Any) -> str | None:
    """Return stripped text from one content part, or None."""
    text = getattr(part, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    if isinstance(part, dict) and isinstance(part.get("text"), str):
        stripped = part["text"].strip()
        if stripped:
            return stripped
    return None


def _azure_list_text(content: list) -> str:
    """Join text parts from a list content payload."""
    parts = [text for part in content if (text := _azure_part_text(part))]
    return " ".join(parts)


def _azure_first_user_text(request: ChatRequest) -> str:
    """Return the first user turn text for conversation fingerprinting."""
    for msg in request.messages:
        if msg.role != "user":
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            joined = _azure_list_text(content)
            if joined:
                return joined
    return ""


def _azure_conversation_fingerprint(request: ChatRequest) -> str | None:
    """Hash the stable conversation prefix for per-conversation isolation.

    Uses only the first user turn plus sorted tool names so the key stays
    stable across turns but differs across conversations sharing one harness
    header such as ``opencode`` or ``prime-agent``.
    """
    first = _azure_first_user_text(request)
    if not first:
        return None
    names: list[str] = []
    for tool in request.tools or []:
        name = getattr(getattr(tool, "function", None), "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    payload = first + "\n" + ",".join(sorted(set(names)))
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _azure_session_key(request: ChatRequest, headers: dict[str, str] | None) -> str | None:
    """Resolve a stable conversation key for Azure prompt-cache reuse.

    Combines the harness namespace with a conversation fingerprint so two
    different tasks sharing ``x-mantis-session-id: opencode`` do not evict
    each other. Falls back to the bare header when no fingerprint exists.
    """
    if headers:
        explicit = headers.get("x-mantis-session-id") or headers.get("x-mantis-session")
        if explicit:
            cleaned = explicit.strip()
            fingerprint = _azure_conversation_fingerprint(request)
            if fingerprint:
                return f"explicit:{cleaned}:{fingerprint}"
            return f"explicit:{cleaned}"
    if request.user:
        return f"user:{request.user.strip()}"
    metadata = request.metadata or {}
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return f"metadata:{metadata['session_id']}"
    return None


def _azure_content_item(role: str, part: TextPart | ImagePart) -> dict[str, Any]:
    """Convert a Mantis message part into an Azure Responses input content item."""
    if isinstance(part, TextPart):
        text_type = "output_text" if role == "assistant" else "input_text"
        return {"type": text_type, "text": part.text}
    if part.type == "image_url":
        return {
            "type": "input_image",
            "image_url": {
                "url": part.image_url.url,
                "detail": part.image_url.detail or "auto",
            },
        }
    return {"type": "input_text", "text": str(part)}


def _azure_canonical_content(role: str, content: str | list[TextPart | ImagePart] | None) -> str | list[dict[str, Any]] | None:
    """Return Azure Responses-compatible content for a message role.

    System and developer messages are sent as a single text string.
    User and assistant messages are sent as content-part lists with the
    correct part types (``input_text`` / ``output_text`` / ``input_image``).
    """
    if content is None:
        return None
    if isinstance(content, str):
        if role in ("system", "developer"):
            return content
        part_type = "output_text" if role == "assistant" else "input_text"
        return [{"type": part_type, "text": content}]
    parts = [_azure_content_item(role, p) for p in content]
    if role in ("system", "developer"):
        return "\n".join(p["text"] for p in parts if p.get("type") in ("input_text", "output_text"))
    return parts


def _azure_canonical_message(message: Message) -> dict[str, Any] | None:
    """Return an Azure Responses-compatible input item for a Mantis message."""
    content = _azure_canonical_content(message.role, message.content)
    if not content:
        return None
    return {"type": "message", "role": message.role, "content": content}


def _azure_input_for_request(
    request: ChatRequest,
    headers: dict[str, str] | None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return the input list and previous_response_id for an Azure request.

    When the new messages extend the stored conversation, only the new tail is
    sent with ``previous_response_id`` so the prior prompt prefix is cached.
    When the conversation is reset, shortened, or has no session key, the full
    message list is sent without a previous response id.
    """
    input_messages = [m for m in (_azure_canonical_message(msg) for msg in request.messages) if m]
    key = _azure_session_key(request, headers)
    if not key:
        return input_messages, None
    previous = _azure_sessions.get(key)
    if not previous:
        return input_messages, None
    previous_id, previous_messages = previous
    previous_len = len(previous_messages)
    if previous_len and previous_len < len(input_messages) and input_messages[:previous_len] == previous_messages:
        return input_messages[previous_len:], previous_id
    return input_messages, None


def _azure_output_text(item: openai.types.responses.ResponseOutputMessage) -> str:
    return "\n".join(
        part.text for part in item.content if part.type == "output_text" and hasattr(part, "text")
    )


def _azure_output_reasoning(item: openai.types.responses.ResponseOutputMessage) -> str:
    parts: list[str] = []
    for part in item.content:
        if part.type != "reasoning" or not hasattr(part, "summary"):
            continue
        summaries = getattr(part, "summary", []) or []
        parts.extend(
            s.text
            for s in summaries
            if s.type == "summary_text" and hasattr(s, "text")
        )
    return "\n".join(parts)


def _azure_response_to_chat_completion(model: str, resp: openai.types.responses.Response) -> dict[str, Any]:
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in resp.output:
        if item.type != "message":
            continue
        text_parts.append(_azure_output_text(item))
        reasoning_parts.append(_azure_output_reasoning(item))
    message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
    if reasoning := "\n".join(reasoning_parts):
        message["reasoning"] = reasoning
    usage = resp.usage
    return {
        "id": resp.id,
        "object": "chat.completion",
        "created": int(resp.created_at or 0),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.input_tokens if usage else 0,
            "completion_tokens": usage.output_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        },
    }


def _azure_store_session(
    request: ChatRequest,
    headers: dict[str, str] | None,
    input_messages: list[dict[str, Any]],
    response_id: str,
) -> None:
    """Store the latest Azure response id for a session, pruning on overflow."""
    key = _azure_session_key(request, headers)
    if not key:
        return
    while len(_azure_sessions) >= _AZURE_SESSION_LIMIT:
        _azure_sessions.pop(next(iter(_azure_sessions)))
    _azure_sessions[key] = (response_id, input_messages)


def _complete_direct(request: ChatRequest, headers: dict[str, str] | None = None) -> dict[str, Any]:
    resolved = providers._resolve_model_spec(_azure_router_spec(request))
    key = providers._provider_keys().get(_AZURE_ROUTER_PROVIDER) or os.environ.get(resolved.credential_env)
    client = openai.OpenAI(
        base_url=resolved.base_url,
        api_key=key or "",
        default_headers={"api-key": key or ""},
    )
    input_messages, previous_id = _azure_input_for_request(request, headers)
    create_kwargs: dict[str, Any] = {
        "model": resolved.model,
        "input": input_messages,
        "max_output_tokens": _azure_router_max_tokens(request),
        "reasoning": {"effort": _azure_router_effort(request)},
    }
    if previous_id:
        create_kwargs["previous_response_id"] = previous_id
    resp = client.responses.create(**create_kwargs, stream=False)
    body = _azure_response_to_chat_completion(request.model, resp)
    full_messages: list[dict[str, Any]] = [m for m in (_azure_canonical_message(msg) for msg in request.messages) if m]
    assistant = body["choices"][0]["message"]
    assistant_content = _azure_canonical_content(assistant["role"], assistant["content"])
    if assistant_content:
        full_messages.append({"type": "message", "role": assistant["role"], "content": assistant_content})
    _azure_store_session(request, headers, full_messages, resp.id)
    return body


def _stream_direct(
    request: ChatRequest,
    headers: dict[str, str] | None,
    completion_id: str | None,
) -> Iterator[bytes]:
    results: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancelled = threading.Event()
    stream_id = completion_id or "chatcmpl-" + uuid.uuid4().hex[:24]
    base = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
    }

    def complete() -> None:
        try:
            result = _complete_direct(request, headers)
            result["id"] = stream_id
            result["model"] = request.model
            results.put(("result", result))
        except HTTPException as error:
            results.put(("error", error))
        except Exception as error:  # noqa: BLE001 - never leave the stream hanging
            results.put(("error", HTTPException(502, f"direct provider failed: {error}")))
        finally:
            _capacity.release()

    threading.Thread(target=complete, daemon=True).start()
    include_usage = bool(request.stream_options and request.stream_options.include_usage)
    try:
        while True:
            try:
                kind, value = results.get(timeout=_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield b": keep-alive\n\n"
                continue
            if kind == "error":
                payload = {
                    **base,
                    "error": {"message": str(value.detail), "type": "upstream_error"},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            yield from _sse(value, include_usage)
            return
    finally:
        cancelled.set()


def _complete(request: ChatRequest, headers: dict[str, str] | None = None) -> dict[str, Any]:
    detail_level = _detail_level(headers)
    body = request.model_dump(exclude_none=True, by_alias=True)
    run_id: str | None = None
    try:
        run, run_id, event = _advance(request, body)
    except serve.RunCapacityError as error:
        raise HTTPException(429, str(error)) from error
    except KeyError as error:
        raise HTTPException(409, str(error)) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(400, str(error)) from error
    except RuntimeError as error:
        if run_id:
            serve.delete_run(run_id, error=str(error))
        raise HTTPException(502, str(error)) from error
    if event.get("type") == "final":
        record_activity = getattr(run, "record_activity", lambda *_a, **_k: None)
        record_activity("validation", status="started", summary="Validating the final answer")
        try:
            run.validate_output(str(event.get("text", "")))
        except ValueError as error:
            record_activity("validation", status="failed", summary="Final answer validation failed")
            serve.delete_run(run_id, error=str(error))
            raise HTTPException(502, str(error)) from error
        record_activity(
            "validation", status="completed", summary="Final answer validation completed"
        )
        record_activity("complete", summary="Final answer ready")
    response = serve._completion_response(
        request.model, body["messages"], run, event, details=detail_level
    )
    if event.get("type") == "final":
        serve.delete_run(run_id)
    return cast(dict[str, Any], response)


def _text_chunks(text: str, size: int = 64) -> Iterator[str]:
    while len(text) > size:
        split = max(text.rfind(char, 0, size + 1) for char in " \n\t")
        if split <= 0:
            split = size
        yield text[:split]
        text = text[split:]
    if text:
        yield text


def _sse(body: dict[str, Any], include_usage: bool) -> Iterator[bytes]:
    choice = body["choices"][0]
    message = choice["message"]
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }

    def chunk(delta: dict[str, Any], finish_reason: Any = None) -> bytes:
        choices = [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
        return f"data: {json.dumps({**base, 'choices': choices})}\n\n".encode()

    if message.get("tool_calls"):
        yield chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {**call, "index": index} for index, call in enumerate(message["tool_calls"])
                ],
            }
        )
    else:
        deltas: list[dict[str, Any]] = []
        reasoning = message.get("reasoning")
        if isinstance(reasoning, str) and reasoning:
            deltas.append({"reasoning": reasoning})
        details = message.get("reasoning_details")
        if isinstance(details, list) and details:
            deltas.append({"reasoning_details": details})
        deltas.extend(
            {key: message[key]} for key in ("annotations", "citations") if message.get(key)
        )
        content = str(message.get("content") or "")
        deltas.extend({"content": part} for part in _text_chunks(content))
        if not deltas:
            deltas.append({"content": ""})
        for number, delta in enumerate(deltas):
            if number and delta.get("content") and _FINAL_CHUNK_DELAY_SECONDS:
                time.sleep(_FINAL_CHUNK_DELAY_SECONDS)
            yield chunk({"role": "assistant", **delta} if number == 0 else delta)
    yield chunk({}, choice["finish_reason"])
    if include_usage:
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': body['usage']})}\n\n".encode()
    if body.get("mantis"):
        yield f"data: {json.dumps({**base, 'choices': [], 'mantis': body['mantis']})}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _stream_events_enabled(headers: dict[str, str] | None) -> bool:
    value = (headers or {}).get("x-mantis-events")
    if value is None:
        return _STREAM_EVENTS_DEFAULT
    return value.strip().lower() not in {"0", "false", "none", "off"}


def _progress_sse(base: dict[str, Any], event: dict[str, Any], first: bool) -> bytes:
    summary = str(event.get("summary") or "Mantis is working")
    model = event.get("model")
    status_line = f"Mantis · {summary}" + (f" · {model}" if model else "") + "\n"
    delta = {"reasoning": status_line}
    if first:
        delta["role"] = "assistant"
    payload = {
        **base,
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        "mantis_event": event,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _stream(
    request: ChatRequest,
    headers: dict[str, str] | None = None,
    completion_id: str | None = None,
) -> Iterator[bytes]:
    results: queue.Queue[tuple[str, Any]] = queue.Queue()
    cancelled = threading.Event()
    started = time.monotonic()
    sequence = 0
    sequence_lock = threading.Lock()
    stream_id = completion_id or "chatcmpl-" + uuid.uuid4().hex[:24]
    base = {
        "id": stream_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": request.model,
    }

    def emit(raw: dict[str, Any]) -> None:
        nonlocal sequence
        if cancelled.is_set():
            return
        # Parallel Fusion lanes emit concurrently; keep sequence unique.
        with sequence_lock:
            event = {
                "version": 1,
                "sequence": sequence,
                "elapsed_ms": round((time.monotonic() - started) * 1000.0, 1),
                **raw,
            }
            sequence += 1
        results.put(("event", event))

    def complete() -> None:
        try:
            with (
                serve.progress_events(emit),
                serve.client_connection(lambda: not cancelled.is_set()),
            ):
                result = _complete(request, headers)
                result["id"] = stream_id
                results.put(("result", result))
        except HTTPException as error:
            results.put(("error", error))
        except Exception as error:  # noqa: BLE001 - never leave the stream hanging
            results.put(("error", HTTPException(502, f"orchestration failed: {error}")))
        finally:
            _capacity.release()

    threading.Thread(target=complete, daemon=True).start()
    requested_events = (headers or {}).get("x-mantis-events")
    if requested_events is None:
        event_mode = "summary" if _STREAM_EVENTS_DEFAULT else "none"
    else:
        event_mode = requested_events.strip().lower()
        if event_mode in {"0", "false", "off", "none"}:
            event_mode = "none"
        elif event_mode != "debug":
            event_mode = "summary"
    show_events = event_mode != "none"
    first_event = True
    last_summary_line = ""
    last_role = ""
    try:
        while True:
            try:
                kind, value = results.get(timeout=_KEEPALIVE_SECONDS)
            except queue.Empty:
                yield b": keep-alive\n\n"
                continue
            if kind == "event":
                if show_events:
                    output = value
                    if event_mode == "summary":
                        # Provider activity is intentionally reduced at the
                        # streaming boundary; debug retains the original event.
                        activity_type = str(value.get("type", ""))
                        status = str(value.get("status", ""))
                        role = str(value.get("role", ""))
                        if activity_type in {"provider", "run"} or (
                            activity_type == "step" and status != "started"
                        ):
                            continue
                        if activity_type == "complete":
                            summary = "Answer ready"
                        elif activity_type == "tool_call":
                            summary = "Worker requested a tool"
                        elif activity_type == "tool_result":
                            summary = "Tool result received"
                        elif activity_type == "verify_accept":
                            summary = "Answer ready"
                        elif activity_type == "verify_reject":
                            summary = "Worker revising"
                        else:
                            summary = {
                                "Worker": "Worker drafting",
                                "Verifier": "Verifier reviewing",
                                "Planner": "Planner planning",
                            }.get(role, str(value.get("summary") or "Mantis is working"))
                        model = value.get("model") if role != last_role else None
                        line = f"Mantis · {summary}" + (f" · {model}" if model else "")
                        if line == last_summary_line:
                            continue
                        output = {**value, "summary": summary, "model": model}
                        last_summary_line = line
                        last_role = role or last_role
                    yield _progress_sse(base, output, first_event)
                    first_event = False
                continue
            if kind == "error":
                payload = {
                    **base,
                    "error": {"message": str(value.detail), "type": "upstream_error"},
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            yield from _sse(
                value,
                bool(request.stream_options and request.stream_options.include_usage),
            )
            return
    finally:
        cancelled.set()


@app.exception_handler(RequestValidationError)
def validation_error(_request: Any, error: RequestValidationError) -> JSONResponse:
    message = "; ".join(str(item.get("msg", "invalid request")) for item in error.errors())
    return _error(400, message, "invalid_request_error")


@app.exception_handler(HTTPException)
def http_error(_request: Any, error: HTTPException) -> JSONResponse:
    error_type = "authentication_error" if error.status_code == 401 else "invalid_request_error"
    if error.status_code == 429:
        error_type = "rate_limit_error"
    elif error.status_code >= 500:
        error_type = "upstream_error"
    return _error(error.status_code, str(error.detail), error_type)


@app.get("/health")
def health(response: Response) -> dict[str, Any]:
    response.headers["X-Request-Id"] = uuid.uuid4().hex
    return {"status": "ok", "model": serve.MODEL_NAME}


def _endpoint_readiness(url: str) -> tuple[str | None, str]:
    parsed = urlsplit(url)
    hostname = parsed.hostname
    host = f"[{hostname}]" if hostname and ":" in hostname else hostname or ""
    port = parsed.port
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = f"{host}:{port}" if port is not None and not default_port else host
    normalized = urlunsplit(
        (parsed.scheme.lower(), netloc.lower(), parsed.path.rstrip("/"), "", "")
    )
    return hostname, hashlib.sha256(normalized.encode()).hexdigest()[:12]


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    response.headers["X-Request-Id"] = uuid.uuid4().hex
    if not os.environ.get("MANTIS_API_KEY"):
        raise HTTPException(503, "MANTIS_API_KEY is not configured")
    profile = os.environ.get("MANTIS_ENDPOINT_PROFILE", "direct")
    if profile == "catalog":
        try:
            bindings = model_catalog.load_runtime_bindings()
        except model_catalog.CatalogError as error:
            raise HTTPException(503, str(error)) from error
        if bindings is None:
            raise HTTPException(503, "Mantis catalog bindings are not configured")
        try:
            keys = model_catalog.resolve_runtime_credentials(bindings)
        except model_catalog.CatalogError as error:
            raise HTTPException(503, str(error)) from error
        contract = os.environ.get("MANTIS_IDENTITY_CONTRACT", "")
        if not contract:
            raise HTTPException(503, "Mantis catalog bindings are not configured")
        if not keys:
            raise HTTPException(503, "Mantis catalog provider credentials are not configured")
        endpoint_urls = {name: binding.base_url for name, binding in bindings.providers.items()}
    else:
        contract = ""
        keys = {}
        endpoint_urls = {
            "openrouter": os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            "opencode": os.environ.get("OPENCODE_GO_ENDPOINT_URL", "https://opencode.ai/zen/go/v1"),
        }
    metadata = {name: _endpoint_readiness(url) for name, url in endpoint_urls.items()}
    body: dict[str, Any] = {
        "status": "ready",
        "model": serve.MODEL_NAME,
        "runtime_mode": "experimental" if _experimental_modes_enabled() else "normal",
        "available_modes": (
            ["base", "trinity", "ultra", "fusion"]
            if _experimental_modes_enabled()
            else ["base", "fusion"]
        ),
        "endpoint_profile": profile,
        "endpoint_hosts": {name: values[0] for name, values in metadata.items()},
        "endpoint_fingerprints": {name: values[1] for name, values in metadata.items()},
    }
    if contract:
        body["catalog_identity_contract"] = contract
        body["binding_fingerprint"] = model_catalog.runtime_binding_fingerprint()
    return body


_MODEL_CREATED = int(time.time())
_BASIC_MODEL = "mantis/base"
# Short aliases clients may send when a provider namespace is already "mantis".
_MODEL_ALIASES = {
    "mantis": "mantis/base",
    "base": "mantis/base",
    "mantis-trinity": "mantis/trinity",
    "trinity": "mantis/trinity",
    "mantis-ultra": "mantis/ultra",
    "ultra": "mantis/ultra",
    "mantis-fusion": "mantis/fusion",
    "fusion": "mantis/fusion",
    "mantis-azure-router": _AZURE_ROUTER_MODEL,
    "azure-router": _AZURE_ROUTER_MODEL,
}
_SUPPORTED_PARAMETERS = [
    "tools",
    "tool_choice",
    "response_format",
    "reasoning",
    "reasoning_effort",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "stream",
    "stream_options",
    "web_search_options",
]


def _router_client() -> httpx.Client:
    return base_proxy.router_client()


_FUSION_TOOL_ID_PREFIX = "f"
_FUSION_TOOL_ID_SEP = "~"


def _encode_fusion_tool_call_id(run_id: str, call_id: str) -> str:
    return f"{_FUSION_TOOL_ID_PREFIX}{run_id}{_FUSION_TOOL_ID_SEP}{call_id}"


def _decode_fusion_tool_call_id(encoded: str) -> tuple[str, str]:
    if not encoded.startswith(_FUSION_TOOL_ID_PREFIX) or _FUSION_TOOL_ID_SEP not in encoded:
        raise ValueError("invalid fusion tool_call_id")
    parts = encoded[1:].split(_FUSION_TOOL_ID_SEP, 1)
    if len(parts) != 2:
        raise ValueError("invalid fusion tool_call_id")
    return parts[0], parts[1]


def _content_to_str(content: str | list[TextPart | ImagePart] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.text if isinstance(part, TextPart) else str(part) for part in content)
    return str(content)


def _build_fusion_chat_response(
    request_id: str,
    model: str,
    event: dict[str, Any],
    request: ChatRequest,
) -> JSONResponse:
    status = event.get("status")
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
    run_id = str(event.get("run_id") or "")
    if status == "completed":
        content = event.get("report") or ""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        finish = "stop"
        completion_text = content
    elif status == "awaiting_tools":
        pending = event.get("pending_tool_calls") or []
        tool_calls = [
            {
                "id": _encode_fusion_tool_call_id(run_id, call.get("id", "")),
                "type": call.get("type", "function"),
                "function": call.get("function", {}),
            }
            for call in pending
        ]
        message = {"role": "assistant", "content": "", "tool_calls": tool_calls}
        finish = "tool_calls"
        completion_text = "".join(
            f"{call['function'].get('name', '')}:{call['function'].get('arguments', '')}"
            for call in tool_calls
        )
    else:
        return _error(
            500,
            event.get("report") or f"fusion run failed with status {status}",
            "upstream_error",
        )
    reasoning_trace = event.get("reasoning_trace")
    if reasoning_trace:
        message["reasoning"] = reasoning_trace
    usage = utils._request_usage(messages, completion_text)
    headers = {"X-Request-Id": request_id}
    if run_id:
        headers["X-Mantis-Run-Id"] = run_id
    return JSONResponse(
        {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish,
                }
            ],
            "usage": usage,
        },
        headers=headers,
    )


def _iter_fusion_chat_event(
    request: ChatRequest,
    request_id: str,
    event: dict[str, Any],
) -> Iterator[bytes]:
    """Yield SSE chunks for an already-computed Fusion chat event."""
    base = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": _MODEL_CREATED,
        "model": "mantis/fusion",
    }
    try:
        status = event.get("status")

        def _chunk(delta: dict[str, Any], finish: str | None) -> bytes:
            return (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                )
                + "\n\n"
            ).encode()

        if status == "completed":
            report = event.get("report") or ""
            reasoning_trace = event.get("reasoning_trace")
            yield _chunk({"role": "assistant"}, None)
            if reasoning_trace:
                yield _chunk({"reasoning": reasoning_trace}, None)
            yield _chunk({"content": report}, "stop")
            completion_text = report
        elif status == "awaiting_tools":
            run_id = event["run_id"]
            pending = event.get("pending_tool_calls") or []
            tool_calls = [
                {
                    "index": i,
                    "id": _encode_fusion_tool_call_id(run_id, call.get("id", "")),
                    "type": call.get("type", "function"),
                    "function": call.get("function", {}),
                }
                for i, call in enumerate(pending)
            ]
            reasoning_trace = event.get("reasoning_trace")
            yield _chunk({"role": "assistant"}, None)
            if reasoning_trace:
                yield _chunk({"reasoning": reasoning_trace}, None)
            yield _chunk({"tool_calls": tool_calls}, "tool_calls")
            completion_text = "".join(
                f"{call['function'].get('name', '')}:{call['function'].get('arguments', '')}"
                for call in tool_calls
            )
        else:
            msg = event.get("report") or f"fusion run failed with status {status}"
            yield _chunk({"role": "assistant"}, None)
            yield _chunk({"content": msg}, "stop")
            completion_text = msg
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        if include_usage:
            usage = utils._request_usage(
                [msg.model_dump(exclude_none=True) for msg in request.messages],
                completion_text,
            )
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n".encode()
    finally:
        _capacity.release()
    yield b"data: [DONE]\n\n"


def _fusion_run_id_from_headers(headers: dict[str, str] | None) -> str | None:
    if not headers:
        return None
    for key in ("x-mantis-run-id", "X-Mantis-Run-Id"):
        value = headers.get(key, "").strip()
        if value:
            return value
    # Starlette lowercases header names; also accept mixed maps.
    for name, value in headers.items():
        if name.lower() == "x-mantis-run-id" and str(value).strip():
            return str(value).strip()
    return None


def _fusion_run_id_from_messages(messages: list[Any]) -> str | None:
    """Infer the latest Fusion run id from encoded tool call ids in history."""
    found: str | None = None
    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        if role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is None and isinstance(msg, dict):
                tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                continue
            try:
                found, _ = _decode_fusion_tool_call_id(str(tool_call_id))
            except ValueError:
                continue
        elif role == "assistant":
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls is None and isinstance(msg, dict):
                tool_calls = msg.get("tool_calls")
            for call in tool_calls or []:
                call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
                if not call_id:
                    continue
                try:
                    found, _ = _decode_fusion_tool_call_id(str(call_id))
                except ValueError:
                    continue
    return found


def _run_fusion_chat(
    request: ChatRequest,
    return_reasoning: bool = False,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resume or start a Fusion run from an OpenAI-style chat request."""
    # Find a trailing block of tool messages; if present, this is a tool follow-up.
    tool_results: list[dict[str, Any]] = []
    run_id: str | None = None
    i = len(request.messages) - 1
    while i >= 0 and request.messages[i].role == "tool":
        tool_msg = request.messages[i]
        try:
            rid, original_id = _decode_fusion_tool_call_id(tool_msg.tool_call_id or "")
        except ValueError:
            break
        if run_id is None:
            run_id = rid
        elif run_id != rid:
            break
        tool_results.insert(
            0,
            {
                "tool_call_id": original_id,
                "content": _content_to_str(tool_msg.content),
                "is_error": False,
            },
        )
        i -= 1

    if run_id is not None and tool_results:
        event = fusion.advance_fusion_run(
            run_id,
            tool_results=tool_results,
        )
    else:
        brief = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                brief = _content_to_str(msg.content)
                break
        if not brief:
            raise ValueError("mantis/fusion requires at least one user message")

        resume_id = _fusion_run_id_from_headers(headers) or _fusion_run_id_from_messages(
            request.messages
        )
        resumed = fusion.try_get_fusion_run(resume_id) if resume_id else None
        if resumed is not None and resumed.status != "awaiting_tools":
            if request.tools is not None:
                fusion.refresh_fusion_run_tools(
                    resumed, [tool.model_dump() for tool in request.tools]
                )
            event = fusion.advance_fusion_run(resumed.run_id, message=brief)
        else:
            tools = [tool.model_dump() for tool in (request.tools or [])]
            messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
            run = fusion.create_fusion_run(
                brief,
                tools,
                messages=messages,
                delegation_mode="available",
                worker_profiles=request.worker_profiles,
                budget=request.budget,
                tool_options=request.tool_options,
            )
            event = fusion.advance_fusion_run(run.run_id)

    if event.get("run_id"):
        run_obj = fusion.get_run(event["run_id"])
        if isinstance(run_obj, fusion.FusionRun):
            # Plans and delegation briefs are public orchestration output.
            # Only provider summaries are gated by the existing opt-in.
            event["reasoning_trace"] = fusion._orchestration_trace(
                run_obj, include_reasoning=return_reasoning
            )
    return event


_EXPERIMENTAL_MODELS = {"mantis/trinity", "mantis/ultra"}


def _experimental_modes_enabled() -> bool:
    return os.environ.get("MANTIS_EXPERIMENTAL_MODES", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _experimental_gate(request: ChatRequest) -> JSONResponse | None:
    if request.model not in _EXPERIMENTAL_MODELS or _experimental_modes_enabled():
        return None
    return _error(
        400,
        f"{request.model} is experimental; restart the local stack with --experimental",
        "invalid_request_error",
    )


@app.get("/v1/models", dependencies=[Depends(_authorize)])
def models() -> dict[str, Any]:
    descriptor = {
        "object": "model",
        "owned_by": "mantis",
        "created": _MODEL_CREATED,
        "context_length": int(os.environ.get("MANTIS_CONTEXT_LENGTH", "262144")),
        "max_completion_tokens": serve.upstream_output_token_cap(),
        "supported_parameters": _SUPPORTED_PARAMETERS,
        "pricing": {"prompt": "0", "completion": "0"},
    }
    basic = {
        **descriptor,
        "context_length": descriptor["context_length"],
        "max_completion_tokens": 131072,
    }
    data = [
        {"id": _BASIC_MODEL, "status": "stable", **basic},
        {"id": _AZURE_ROUTER_MODEL, "status": "stable", **basic},
        {"id": "mantis/fusion", "status": "stable", **descriptor},
    ]
    if _experimental_modes_enabled():
        data[2:2] = [
            {"id": "mantis/trinity", "status": "experimental", **descriptor},
            {"id": "mantis/ultra", "status": "experimental", **descriptor},
        ]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions", dependencies=[Depends(_authorize)])
def chat(request: ChatRequest, response: Response, http: HttpRequest) -> Response:
    request_id = uuid.uuid4().hex
    response.headers["X-Request-Id"] = request_id
    headers = dict(http.headers)
    if request.model in _MODEL_ALIASES:
        request = request.model_copy(update={"model": _MODEL_ALIASES[request.model]})
    if experimental_error := _experimental_gate(request):
        return experimental_error
    if not _capacity.acquire(blocking=False):
        return _error(429, "Mantis is at capacity", "rate_limit_error")
    if request.model == "mantis/fusion":
        return_reasoning = _fusion_return_reasoning(headers)
        try:
            event = _run_fusion_chat(request, return_reasoning=return_reasoning, headers=headers)
        except Exception as exc:  # noqa: BLE001 - chat adapter error boundary
            _capacity.release()
            return _error(500, f"fusion chat failed: {exc}", "upstream_error")
        run_headers = {"X-Request-Id": request_id}
        if event.get("run_id"):
            run_headers["X-Mantis-Run-Id"] = str(event["run_id"])
        if request.stream:
            return StreamingResponse(
                _iter_fusion_chat_event(request, request_id, event),
                media_type="text/event-stream",
                headers=run_headers,
            )
        try:
            return _build_fusion_chat_response(request_id, "mantis/fusion", event, request)
        finally:
            _capacity.release()
    if request.model == _AZURE_ROUTER_MODEL:
        if request.stream:
            return StreamingResponse(
                _stream_direct(request, headers, "chatcmpl-" + request_id[:24]),
                media_type="text/event-stream",
                headers={
                    "X-Request-Id": request_id,
                    "X-Mantis-Streaming": "live-status,verified-buffered-content",
                },
            )
        try:
            body = _complete_direct(request, headers)
            return JSONResponse(body, headers={"X-Request-Id": request_id})
        finally:
            _capacity.release()
    if request.model == _BASIC_MODEL:
        handed_off = False
        try:
            response, handed_off = base_proxy.forward(
                request,
                headers,
                request_id,
                _router_client,
                on_close=_capacity.release,
            )
        except httpx.HTTPError as error:
            return _error(502, f"router unavailable: {error}", "upstream_error")
        else:
            return response
        finally:
            if not handed_off:
                _capacity.release()
    if request.stream:
        return StreamingResponse(
            _stream(request, headers, "chatcmpl-" + request_id[:24]),
            media_type="text/event-stream",
            headers={
                "X-Request-Id": request_id,
                "X-Mantis-Streaming": "live-status,verified-buffered-content",
            },
        )
    try:
        body = _complete(request, headers)
        extra = _mantis_headers(body.get("mantis", {}), body) if body.get("mantis") else {}
        return JSONResponse(body, headers={"X-Request-Id": request_id, **extra})
    finally:
        _capacity.release()


class FusionDelegateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    brief: str = Field(min_length=1)
    tools: list[FunctionTool] | None = None
    worker_profiles: list[FusionWorkerProfile] | None = None
    budget: FusionRunBudget | None = None
    tool_options: FusionToolOptions | None = None


class FusionToolResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool_call_id: str = Field(min_length=1)
    content: str
    is_error: bool = False


class FusionFollowUpRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: str = Field(min_length=1, max_length=128)
    tool_results: list[FusionToolResult] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_tool_results(self) -> FusionFollowUpRequest:
        seen: set[str] = set()
        for item in self.tool_results:
            if item.tool_call_id in seen:
                raise ValueError("duplicate tool_call_id in tool_results")
            seen.add(item.tool_call_id)
        return self


class FusionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: str
    status: Literal[
        "main_planning", "sidekick_pending", "awaiting_tools", "main_review", "completed", "error"
    ]
    report: str | None
    pending_tool_calls: list[dict[str, Any]] | None
    usage: dict[str, Any]
    activity: list[dict[str, Any]]
    request_id: str | None = None
    usage_models: dict[str, Any] = Field(default_factory=dict)
    cost: float | None = None
    follow_up_count: int = 0
    follow_up_capped: bool = False


@app.post("/v1/fusion/delegate", dependencies=[Depends(_authorize)])
def fusion_delegate(request: FusionDelegateRequest) -> JSONResponse:
    if not _capacity.acquire(blocking=False):
        return _error(429, "Mantis is at capacity", "rate_limit_error")
    try:
        tools = [tool.model_dump() for tool in (request.tools or [])]
        run = fusion.create_fusion_run(
            request.brief,
            tools,
            worker_profiles=request.worker_profiles,
            budget=request.budget,
            tool_options=request.tool_options,
        )
        event = fusion.advance_fusion_run(run.run_id)
        return JSONResponse(FusionResponse(**event).model_dump())
    finally:
        _capacity.release()


@app.post("/v1/fusion/follow_up/{run_id}", dependencies=[Depends(_authorize)])
def fusion_follow_up(run_id: str, request: FusionFollowUpRequest) -> JSONResponse:
    if not _capacity.acquire(blocking=False):
        return _error(429, "Mantis is at capacity", "rate_limit_error")
    try:
        tool_results = [item.model_dump() for item in request.tool_results]
        event = fusion.advance_fusion_run(
            run_id,
            request_id=request.request_id,
            tool_results=tool_results,
        )
        return JSONResponse(FusionResponse(**event, request_id=request.request_id).model_dump())
    finally:
        _capacity.release()


@app.get("/v1/fusion/runs/{run_id}", dependencies=[Depends(_authorize)])
def fusion_status(run_id: str) -> JSONResponse:
    event = fusion.fusion_run_status(run_id)
    return JSONResponse(FusionResponse(**event).model_dump())
