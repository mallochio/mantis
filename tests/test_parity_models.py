"""OpenRouter-style /v1/models descriptor tests."""

import api
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.delenv("MANTIS_CONTEXT_LENGTH", raising=False)
    monkeypatch.delenv("MANTIS_MAX_COMPLETION_TOKENS", raising=False)
    return TestClient(api.app)


def _models(client):
    response = client.get("/v1/models", headers={"Authorization": "Bearer test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    return body["data"]


def test_descriptor_fields_and_defaults(client):
    for entry in _models(client):
        assert entry["object"] == "model"
        assert entry["owned_by"] == "mantis"
        assert isinstance(entry["id"], str)
        assert isinstance(entry["created"], int) and entry["created"] > 0
        assert entry["context_length"] == 262144
        assert entry["max_completion_tokens"] == 32768
        assert entry["pricing"] == {"prompt": "0", "completion": "0"}
        assert isinstance(entry["supported_parameters"], list)


def test_env_overrides(client, monkeypatch):
    monkeypatch.setenv("MANTIS_CONTEXT_LENGTH", "65536")
    monkeypatch.setenv("MANTIS_MAX_COMPLETION_TOKENS", "4096")
    for entry in _models(client):
        assert entry["context_length"] == 65536
        assert entry["max_completion_tokens"] == 4096


def test_supported_parameters_accepted_by_chat_request(client):
    fields = set(api.ChatRequest.model_fields)
    for entry in _models(client):
        supported = entry["supported_parameters"]
        assert len(supported) == len(set(supported))
        unknown = set(supported) - fields
        assert not unknown, f"ChatRequest would reject: {unknown}"
    expected = {
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
    }
    assert set(_models(client)[0]["supported_parameters"]) == expected
