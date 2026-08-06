"""Typed HTTP adapter for the Mantis orchestration engine."""

from __future__ import annotations

import hmac
import json
import os
import threading
import uuid
from collections.abc import Iterator
from typing import Any, Literal

import serve
from fastapi import Depends, FastAPI, Header, HTTPException, Response
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
        return self


_MAX_REQUESTS = int(os.environ.get("MANTIS_MAX_CONCURRENT_REQUESTS", "32"))
_capacity = threading.BoundedSemaphore(_MAX_REQUESTS)
app = FastAPI(title="Mantis", version="0.3.0")


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


def _sse(body: dict[str, Any], include_usage: bool) -> Iterator[bytes]:
    choice = body["choices"][0]
    message = choice["message"]
    delta: dict[str, Any] = {"role": "assistant"}
    if message.get("tool_calls"):
        delta["tool_calls"] = [
            {**call, "index": index} for index, call in enumerate(message["tool_calls"])
        ]
    else:
        delta["content"] = message.get("content") or ""
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n".encode()
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': choice['finish_reason']}]})}\n\n".encode()
    if include_usage:
        yield f"data: {json.dumps({**base, 'choices': [], 'usage': body['usage']})}\n\n".encode()
    yield b"data: [DONE]\n\n"


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


@app.get("/v1/models", dependencies=[Depends(_authorize)])
def models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": model, "object": "model", "owned_by": "mantis"}
            for model in (serve.MODEL_NAME, "mantis-trinity", "mantis-ultra")
        ],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_authorize)])
def chat(request: ChatRequest, response: Response) -> Response:
    request_id = uuid.uuid4().hex
    response.headers["X-Request-Id"] = request_id
    if not _capacity.acquire(blocking=False):
        return _error(429, "Mantis is at capacity", "rate_limit_error")
    try:
        body = _complete(request)
    finally:
        _capacity.release()
    if request.stream:
        return StreamingResponse(
            _sse(body, bool(request.stream_options and request.stream_options.include_usage)),
            media_type="text/event-stream",
            headers={"X-Request-Id": request_id, "X-Mantis-Streaming": "buffered"},
        )
    return JSONResponse(body, headers={"X-Request-Id": request_id})
