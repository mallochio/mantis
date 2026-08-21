"""TTLCache response/in-flight/replay caches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading

from cachetools import TTLCache
from fastapi.responses import Response, StreamingResponse

from lean.config import (
    RESP_CACHE_MAX_BYTES,
    RESP_CACHE_MAX_ENTRIES,
    RESP_CACHE_TTL_S,
    TARGET_CONFIG_REVISION,
)
from lean.sse import _response_replay_safe

_resp_cache = TTLCache(maxsize=RESP_CACHE_MAX_ENTRIES, ttl=RESP_CACHE_TTL_S)
_cache_lock = threading.RLock()
_cache_bytes = 0
_cache_metrics = {"hits": 0, "misses": 0, "stores": 0, "evictions": 0}
_inflight: dict[str, asyncio.Future] = {}
_inflight_lock = asyncio.Lock()


def _prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest()[:24]


def _request_body_hash(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _cache_key(body, idem, *, sid=None):
    if not idem or len(idem) > 200:
        return None
    if body.get("tools") or body.get("functions") or body.get("function_call"):
        return None
    if any(
        isinstance(m, dict) and (m.get("role") in {"tool", "function"} or m.get("tool_calls") or m.get("function_call"))
        for m in body.get("messages", [])
    ):
        return None
    return hashlib.sha256(
        (TARGET_CONFIG_REVISION + ":" + (sid or "sessionless") + ":" + idem + ":" + _request_body_hash(body)).encode()
    ).hexdigest()


def _cache_get(key):
    if key is None:
        return None
    with _cache_lock:
        if key in _resp_cache:
            _cache_metrics["hits"] += 1
            return _resp_cache[key]
        _cache_metrics["misses"] += 1
        return None


def _cache_put(key, body, content, headers, media, status=200):
    global _cache_bytes
    if key is None or len(content) > RESP_CACHE_MAX_BYTES or not _response_replay_safe(body, content):
        return None
    with _cache_lock:
        while _resp_cache and (
            len(_resp_cache) >= RESP_CACHE_MAX_ENTRIES or _cache_bytes + len(content) > RESP_CACHE_MAX_BYTES
        ):
            k, v = _resp_cache.popitem(last=False)
            _cache_bytes -= len(v[0])
            _cache_metrics["evictions"] += 1
        _resp_cache[key] = (content, status, media, headers)
        _cache_bytes += len(content)
        _cache_metrics["stores"] += 1
        return _resp_cache[key]


async def _claim_inflight(key):
    if key is None:
        return True, None
    async with _inflight_lock:
        fut = _inflight.get(key)
        if fut is None:
            fut = asyncio.get_running_loop().create_future()
            _inflight[key] = fut
            return True, fut
        return False, fut


async def _finish_inflight(key, fut, result):
    if key is None or fut is None:
        return
    async with _inflight_lock:
        if _inflight.get(key) is fut:
            _inflight.pop(key, None)
        if not fut.done():
            fut.set_result(result)


def _finish_inflight_nowait(key, fut, result):
    if key is None or fut is None:
        return
    if _inflight.get(key) is fut:
        _inflight.pop(key, None)
    if fut and not fut.done():
        fut.set_result(result)


def _replayed_response(result, request_id):
    content, status, media, headers = result
    headers = {**headers, "x-request-id": request_id, "x-route-coalesced": "true"}
    if media == "text/event-stream":
        return StreamingResponse(iter([content]), status_code=status, media_type=media, headers=headers)
    return Response(content=content, status_code=status, media_type=media, headers=headers)
