"""HTTP proxy from ``mantis/base`` to the local Switchyard stage router.

The Switchyard route id is ``mantis/base``, matching the public model name, so
this module forwards the chat body without rewriting ``model``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, Protocol

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

SWITCHYARD_SESSION_HEADER = "x-switchyard-session-id"
SWITCHYARD_SELECTED_MODEL_HEADER = "x-model-router-selected-model"
DEFAULT_ROUTER_URL = "http://127.0.0.1:5500/v1"


class BaseChatRequest(Protocol):
    metadata: dict[str, Any] | None
    user: str | None
    stream: bool

    def model_dump(
        self, *, exclude_none: bool, exclude: set[str]
    ) -> dict[str, Any]: ...


def router_client() -> httpx.Client:
    return httpx.Client(timeout=float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300")))


def session_id(headers: dict[str, str], body: BaseChatRequest) -> str | None:
    """Session identity for Switchyard stage-router stickiness.

    Accepts the Switchyard header, the legacy ``X-Route-Session`` client
    header, or ``metadata.session_id`` / ``user`` in the body. Only
    ``x-switchyard-session-id`` is sent upstream.
    """
    session = headers.get(SWITCHYARD_SESSION_HEADER) or headers.get("x-route-session")
    if session:
        return session
    if isinstance(body.metadata, dict):
        for key in ("session_id", "sessionId"):
            value = body.metadata.get(key)
            if isinstance(value, str) and value:
                return value
    if isinstance(body.user, str) and body.user:
        return body.user
    return None


def router_headers(headers: dict[str, str], body: BaseChatRequest) -> dict[str, str]:
    out: dict[str, str] = {}
    key = os.environ.get("MANTIS_ROUTER_KEY")
    if key:
        out["Authorization"] = f"Bearer {key}"
    session = session_id(headers, body)
    if session:
        out[SWITCHYARD_SESSION_HEADER] = session
    return out


def router_body(request: BaseChatRequest) -> dict[str, Any]:
    return request.model_dump(exclude_none=True, exclude={"user", "metadata"})


def router_response_headers(upstream: httpx.Response) -> dict[str, str]:
    mapped: dict[str, str] = {}
    selected = upstream.headers.get(SWITCHYARD_SELECTED_MODEL_HEADER)
    if selected:
        mapped["x-route-model"] = selected
    session = upstream.headers.get(SWITCHYARD_SESSION_HEADER)
    if session:
        mapped[SWITCHYARD_SESSION_HEADER] = session
    return mapped


def router_error(upstream: httpx.Response) -> JSONResponse:
    try:
        body = upstream.json()
    except (ValueError, httpx.ResponseNotRead):
        body = {
            "error": {
                "message": "router returned an invalid response",
                "type": "upstream_error",
            }
        }
    return JSONResponse(body, status_code=upstream.status_code)


def _router_stream(client: httpx.Client, stream: Any, upstream: httpx.Response) -> Iterator[bytes]:
    try:
        yield from upstream.iter_bytes()
    finally:
        stream.__exit__(None, None, None)
        client.close()


def forward(
    request: BaseChatRequest,
    headers: dict[str, str],
    request_id: str,
    client_factory: Any,
) -> tuple[JSONResponse | StreamingResponse, bool]:
    """Proxy a Base chat request to Switchyard.

    Returns ``(response, handed_off)``. ``handed_off`` is True when a streaming
    response now owns the HTTP client; the caller should not release capacity
    until that stream ends.
    """
    client = client_factory()
    url = os.environ.get("MANTIS_ROUTER_URL", DEFAULT_ROUTER_URL) + "/chat/completions"
    outbound_headers = router_headers(headers, request)
    outbound_body = router_body(request)
    if request.stream:
        stream = client.stream("POST", url, headers=outbound_headers, json=outbound_body)
        upstream = stream.__enter__()
        if upstream.is_error:
            error = router_error(upstream)
            stream.__exit__(None, None, None)
            client.close()
            return error, False
        return (
            StreamingResponse(
                _router_stream(client, stream, upstream),
                media_type="text/event-stream",
                headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
            ),
            True,
        )
    with client:
        upstream = client.post(url, headers=outbound_headers, json=outbound_body)
    if upstream.is_error:
        return router_error(upstream), False
    return (
        JSONResponse(
            upstream.json(),
            headers={"X-Request-Id": request_id, **router_response_headers(upstream)},
        ),
        False,
    )
