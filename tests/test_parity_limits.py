"""Request body size limits and 413 error-shape parity tests."""

import importlib

import api
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient


def _limited_app(max_bytes: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(api.BodyLimitMiddleware, max_bytes=max_bytes)

    @app.post("/")
    async def endpoint(request: Request):
        await request.body()
        return {"ok": True}

    return app


def test_default_body_limit_is_50mb():
    assert api._MAX_BODY_BYTES == 50 * 1024 * 1024


def test_oversized_content_length_returns_413_standard_error():
    response = TestClient(_limited_app(10)).post("/", content=b"x" * 11)
    assert response.status_code == 413
    error = response.json()["error"]
    assert set(error) == {"message", "type"}
    assert error["type"] == "invalid_request_error"
    assert "limit" in error["message"]


def test_oversized_streamed_body_returns_413_standard_error():
    chunks = [b"x" * 6, b"x" * 6]
    response = TestClient(_limited_app(10)).post("/", content=iter(chunks))
    assert response.status_code == 413
    error = response.json()["error"]
    assert set(error) == {"message", "type"}
    assert error["type"] == "invalid_request_error"


def test_body_under_limit_passes():
    response = TestClient(_limited_app(10)).post("/", content=b"x" * 10)
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_env_override_changes_effective_limit(monkeypatch):
    monkeypatch.setenv("MANTIS_MAX_BODY_BYTES", "1024")
    importlib.reload(api)
    try:
        assert api._MAX_BODY_BYTES == 1024
        response = TestClient(api.app).post("/health", content=b"x" * 1025)
        assert response.status_code == 413
    finally:
        monkeypatch.delenv("MANTIS_MAX_BODY_BYTES")
        importlib.reload(api)
    assert api._MAX_BODY_BYTES == 50 * 1024 * 1024
