"""Upstream send, failover, refusal detection, and retryability."""

from __future__ import annotations

import asyncio
import time

import httpx

from lean.config import (
    RETRY_STATUSES,
    TIMEOUT_S,
    _api_routes,
    _apply_effort_override,
    _backend_for,
)
from lean.helpers import _is_empty_completion, _is_refusal, _safe_json
from lean.sse import _iter_sse, _prefetch_chat, _prefetch_resp

_client = None


def _set_client(client: httpx.AsyncClient | None):
    global _client
    _client = client


def _url_body(backend, body, api):
    if api == "responses":
        from lean.build import _build_responses

        return backend["base_url"].rstrip("/") + "/responses", _build_responses(body, backend)
    from lean.build import _build_chat

    return backend["base_url"].rstrip("/") + "/chat/completions", _build_chat(body, backend)


async def _send_upstream(backend, body, *, stream, api):
    if _client is None:
        raise RuntimeError("client not ready")
    url, out = _url_body(backend, body, api)
    req = _client.build_request(
        "POST",
        url,
        json=out,
        headers={"Authorization": f"Bearer {backend['key']}", "Content-Type": "application/json"},
        timeout=TIMEOUT_S,
    )
    return await _client.send(req, stream=stream)


def _retryable(status):
    return status in RETRY_STATUSES


async def _open_with_failover(body, decision, deadline, *, stream, api, effort=None, allow_failover=True):
    attempts = []
    routes = _api_routes(decision, api) if allow_failover else (decision,)
    for idx, route in enumerate(routes):
        backend = _backend_for(route)
        if idx == 0 and effort:
            backend = _apply_effort_override(backend, effort)
        resp = None
        try:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError
            resp = await asyncio.wait_for(_send_upstream(backend, body, stream=stream, api=api), timeout=rem)
        except (httpx.TransportError, TimeoutError, asyncio.TimeoutError) as e:
            attempts.append((route, backend, None, type(e).__name__))
            continue
        if resp.status_code != 200:
            content = await resp.aread()
            data = _safe_json(content)
            refusal = _is_refusal(resp.status_code, data)
            attempts.append((route, backend, resp.status_code, "refusal" if refusal else None))
            if (refusal or _retryable(resp.status_code)) and idx != len(routes) - 1:
                await resp.aclose()
                continue
            return route, backend, resp, attempts, None
        if not stream:
            content = await resp.aread()
            data = _safe_json(content)
            refusal = _is_refusal(200, data)
            empty = _is_empty_completion(data) if api == "chat" else False
            failure = "refusal" if refusal else ("empty_completion" if empty else None)
            attempts.append((route, backend, 200, failure))
            if failure and idx != len(routes) - 1:
                await resp.aclose()
                continue
            return route, backend, resp, attempts, None
        events = _iter_sse(resp)
        prefix, failure = (
            await _prefetch_resp(events, deadline) if api == "responses" else await _prefetch_chat(events, deadline)
        )
        attempts.append((route, backend, 200, failure))
        if failure and idx != len(routes) - 1:
            await resp.aclose()
            continue
        return route, backend, resp, attempts, (prefix, events)
    if not attempts:
        backend = _backend_for(decision)
        if effort:
            backend = _apply_effort_override(backend, effort)
        return decision, backend, None, [], None
    return attempts[-1][0], attempts[-1][1], None, attempts, None
