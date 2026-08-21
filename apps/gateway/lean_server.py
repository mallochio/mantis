"""Minimal Mantis gateway: Supra routing + Bifrost transport."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request

import lean.proxy
import lean.state
from lean.common import _openai_err, _validate_bind
from lean.config import BACKENDS, HOST, MAX_BODY_BYTES, PORT, TARGET_CONFIG_REVISION, TIMEOUT_S
from lean.decision import _init_db, _store_load
from lean.routes import chat_completions, healthz, models, responses
from lean.supra import _load_supra


@asynccontextmanager
async def lifespan(app):
    lean.state._READY = False
    _init_db()
    lean.proxy._set_client(httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S, connect=min(10.0, TIMEOUT_S))))
    try:
        await asyncio.to_thread(_load_supra)
        await asyncio.to_thread(_store_load)
        lean.state._READY = True
        yield
    finally:
        lean.state._READY = False
        client = lean.proxy._client
        if client:
            await client.aclose()
        lean.proxy._set_client(None)


app = FastAPI(title="Mantis lean gateway", lifespan=lifespan)


@app.middleware("http")
async def _body_limit(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
        return _openai_err("Request body is too large", 413, code="request_too_large")
    return await call_next(request)


app.get("/healthz")(healthz)
app.get("/v1/models")(models)
app.post("/v1/chat/completions")(chat_completions)
app.post("/v1/responses")(responses)


if __name__ == "__main__":
    import uvicorn

    _validate_bind()
    print(f"lean gateway: targets={','.join(BACKENDS)} port={PORT} revision={TARGET_CONFIG_REVISION}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
