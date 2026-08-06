"""Typed HTTP adapter for the Mantis orchestration engine."""

from __future__ import annotations

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

import serve
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class FunctionTool(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionDefinition


class FunctionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)


class NamedToolChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["function"]
    function: FunctionChoice


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)
    detail: Literal["auto", "low", "high"] | None = None


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str | None = None
    schema_: dict[str, Any] = Field(alias="schema")
    strict: bool = True


class JsonSchemaFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_schema"]
    json_schema: JsonSchemaDefinition


class JsonObjectFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["json_object"]


class ReasoningOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    exclude: bool | None = None


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = serve.MODEL_NAME
    messages: list[Message] = Field(min_length=1)
    tools: list[FunctionTool] | None = None
    tool_choice: Literal["auto", "none", "required"] | NamedToolChoice | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    response_format: JsonSchemaFormat | JsonObjectFormat | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    reasoning: ReasoningOptions | None = None
    reasoning_effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    web_search_options: dict[str, Any] | None = None

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
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}})


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
    if event.get("type") == "error":
        raise RuntimeError(str(event.get("error", "orchestration failed")))
    return run, run_id, event


def _complete(request: ChatRequest) -> dict[str, Any]:
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
            serve.delete_run(run_id)
        raise HTTPException(502, str(error)) from error
    if event.get("type") == "final":
        try:
            run.validate_output(str(event.get("text", "")))
        except ValueError as error:
            serve.delete_run(run_id)
            raise HTTPException(502, str(error)) from error
    response = serve._completion_response(request.model, body["messages"], run, event)
    if event.get("type") == "final":
        serve.delete_run(run_id)
    return response


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
                    {**call, "index": index}
                    for index, call in enumerate(message["tool_calls"])
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
            yield chunk({"role": "assistant", **delta} if number == 0 else delta)
    yield chunk({}, choice["finish_reason"])
    if include_usage:
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': body['usage']})}\n\n".encode()
    yield b"data: [DONE]\n\n"


def _stream(request: ChatRequest) -> Iterator[bytes]:
    results: queue.Queue[dict[str, Any] | HTTPException] = queue.Queue(maxsize=1)

    def complete() -> None:
        try:
            results.put(_complete(request))
        except HTTPException as error:
            results.put(error)
        finally:
            _capacity.release()

    threading.Thread(target=complete, daemon=True).start()
    while True:
        try:
            result = results.get(timeout=_KEEPALIVE_SECONDS)
            break
        except queue.Empty:
            yield b": keep-alive\n\n"
    if isinstance(result, HTTPException):
        payload = {
            "error": {"message": str(result.detail), "type": "upstream_error"},
            "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(payload)}\n\n".encode()
        yield b"data: [DONE]\n\n"
        return
    yield from _sse(
        result,
        bool(request.stream_options and request.stream_options.include_usage),
    )


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


@app.get("/ready")
def ready(response: Response) -> dict[str, Any]:
    response.headers["X-Request-Id"] = uuid.uuid4().hex
    if not os.environ.get("MANTIS_API_KEY"):
        raise HTTPException(503, "MANTIS_API_KEY is not configured")
    return {"status": "ready", "model": serve.MODEL_NAME}


_MODEL_CREATED = int(time.time())
_SUPPORTED_PARAMETERS = [
    "tools",
    "tool_choice",
    "response_format",
    "reasoning",
    "reasoning_effort",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "web_search_options",
]


@app.get("/v1/models", dependencies=[Depends(_authorize)])
def models() -> dict[str, Any]:
    descriptor = {
        "object": "model",
        "owned_by": "mantis",
        "created": _MODEL_CREATED,
        "context_length": int(os.environ.get("MANTIS_CONTEXT_LENGTH", "262144")),
        "max_completion_tokens": int(os.environ.get("MANTIS_MAX_COMPLETION_TOKENS", "32768")),
        "supported_parameters": _SUPPORTED_PARAMETERS,
        "pricing": {"prompt": "0", "completion": "0"},
    }
    return {
        "object": "list",
        "data": [
            {"id": model, **descriptor}
            for model in (serve.MODEL_NAME, "mantis-trinity", "mantis-ultra")
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_authorize)])
def chat(request: ChatRequest, response: Response) -> Response:
    request_id = uuid.uuid4().hex
    response.headers["X-Request-Id"] = request_id
    if not _capacity.acquire(blocking=False):
        return _error(429, "Mantis is at capacity", "rate_limit_error")
    if request.stream:
        return StreamingResponse(
            _stream(request),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id, "X-Mantis-Streaming": "buffered"},
        )
    try:
        body = _complete(request)
        return JSONResponse(body, headers={"X-Request-Id": request_id})
    finally:
        _capacity.release()
