"""Shared helpers for request auth, body parsing, errors, and startup validation."""

from __future__ import annotations

import json
import os
import secrets

from fastapi.responses import JSONResponse

from lean.config import MAX_BODY_BYTES, SERVER_KEY


def _authorize(a):
    return bool(a and a.startswith("Bearer ") and secrets.compare_digest(a[7:].encode(), SERVER_KEY.encode()))


async def _read_json_body(request):
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise OverflowError
        chunks.append(chunk)
    try:
        data = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError("Invalid JSON body")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def _openai_err(msg, status, *, err="invalid_request_error", param=None, code=None):
    return JSONResponse({"error": {"message": msg, "type": err, "param": param, "code": code}}, status_code=status)


def _bind_loopback(h):
    return h in {"127.0.0.1", "::1", "localhost"}


def _validate_bind():
    from lean.config import HOST

    if not _bind_loopback(HOST) and (not os.environ.get("MANTIS_ROUTER_KEY") or SERVER_KEY == "sk-route-local"):
        raise RuntimeError("non-loopback binding requires an externally supplied, non-default MANTIS_ROUTER_KEY")
