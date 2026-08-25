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
import serve
import utils
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from fastapi import Request as HttpRequest
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
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
    reasoning: ReasoningOptions | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] | None = None
    web_search_options: dict[str, Any] | None = None
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
}
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
        return " ".join(
            part.text if isinstance(part, TextPart) else str(part)
            for part in content
        )
    return str(content)


def _build_fusion_chat_response(
    request_id: str,
    model: str,
    event: dict[str, Any],
    request: ChatRequest,
) -> JSONResponse:
    status = event.get("status")
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
    if status == "completed":
        content = event.get("report") or ""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        finish = "stop"
        completion_text = content
    elif status == "awaiting_tools":
        run_id = event["run_id"]
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
        headers={"X-Request-Id": request_id},
    )


def _stream_fusion_chat(
    request: ChatRequest,
    request_id: str,
    return_reasoning: bool = False,
) -> Iterator[bytes]:
    """Yield SSE chunks for a Fusion chat completion."""
    base = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": _MODEL_CREATED,
        "model": "mantis/fusion",
    }
    try:
        event = _run_fusion_chat(request, return_reasoning=return_reasoning)
        status = event.get("status")

        def _chunk(delta: dict[str, Any], finish: str | None) -> bytes:
            return (
                "data: "
                + json.dumps(
                    {
                        **base,
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": finish}
                        ],
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
        include_usage = bool(
            request.stream_options and request.stream_options.include_usage
        )
        if include_usage:
            messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
            usage = utils._request_usage(messages, completion_text)
            yield f"data: {json.dumps({**base, 'choices': [], 'usage': usage})}\n\n".encode()
    finally:
        _capacity.release()
    yield b"data: [DONE]\n\n"


def _run_fusion_chat(
    request: ChatRequest,
    return_reasoning: bool = False,
) -> dict[str, Any]:
    """Resume or start a Fusion run from an OpenAI-style chat request."""
    # Find a trailing block of tool messages; if present, this is a follow-up.
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
        # Initial call: the brief is the last user message.
        brief = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                brief = _content_to_str(msg.content)
                break
        if not brief:
            raise ValueError("mantis/fusion requires at least one user message")
        tools = [tool.model_dump() for tool in (request.tools or [])]
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        run = fusion.create_fusion_run(brief, tools, messages=messages)
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
        {"id": "mantis/fusion", "status": "stable", **descriptor},
    ]
    if _experimental_modes_enabled():
        data[1:1] = [
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
        if request.stream:
            return StreamingResponse(
                _stream_fusion_chat(request, request_id, return_reasoning=return_reasoning),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id},
            )
        try:
            event = _run_fusion_chat(request, return_reasoning=return_reasoning)
            return _build_fusion_chat_response(
                request_id, "mantis/fusion", event, request
            )
        except Exception as exc:  # noqa: BLE001 - chat adapter error boundary
            return _error(500, f"fusion chat failed: {exc}", "upstream_error")
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
    status: Literal["main_planning", "sidekick_pending", "awaiting_tools", "main_review", "completed", "error"]
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
        run = fusion.create_fusion_run(request.brief, tools)
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
