"""Tests for mantis-fusion main/sidekick orchestration."""

from __future__ import annotations

import uuid
from typing import Any

import api
import fusion
import pytest
import serve_config
from fastapi.testclient import TestClient

import runs


class FakeWorker:
    """Deterministic fake for the FusionCoordinator._call_worker boundary."""

    def __init__(self) -> None:
        self.main_calls = 0
        self.sidekick_calls = 0
        self.calls: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]] = []
        self.accept_after = 0
        self.reviews = 0

    def __call__(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.calls.append((slot, messages, tools))
        first_content = messages[0].get("content", "")
        if first_content == fusion.MAIN_PREAMBLE:
            return self._main_response(messages)
        return self._sidekick_response(messages, tools)

    def _main_response(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.main_calls += 1
        # Planning call: the user brief is the last user message before an assistant response.
        # Review call: the user message contains REVIEW_PROMPT.
        is_review = any(
            msg.get("role") == "user" and fusion.REVIEW_PROMPT in msg.get("content", "")
            for msg in messages
        )
        if not is_review:
            return (
                "PLAN: implement and test the brief\nBRIEF: implement, run tests, and lint",
                [],
                {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )
        self.reviews += 1
        if self.reviews > self.accept_after:
            return (
                "ACCEPT",
                [],
                {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
            )
        return (
            "FOLLOW_UP: add more tests before reporting",
            [],
            {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58},
        )

    def _sidekick_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        self.sidekick_calls += 1
        # First call: request a tool. Subsequent calls: return a final report.
        if self.sidekick_calls == 1:
            return (
                "",
                [
                    {
                        "id": "call_bash_1",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "echo hello"}',
                        },
                    }
                ],
                {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            )
        return (
            "Final report: implemented and tested successfully.",
            [],
            {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35},
        )


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_RUN_STORE", "memory")
    return TestClient(api.app)


@pytest.fixture
def file_client(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_RUN_STORE", "file")
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(fusion, "RUN_STORE", "file")
    monkeypatch.setattr(runs, "RUN_STORE", "file")
    monkeypatch.setattr(serve_config, "RUN_STORE", "file")
    return TestClient(api.app)


@pytest.fixture
def fake_worker(monkeypatch):
    worker = FakeWorker()
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    return worker


def _headers():
    return {"Authorization": "Bearer test-key"}


SAMPLE_DELEGATE = {
    "brief": "write a hello world script",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                },
            },
        }
    ],
}


def test_fusion_smoke(client, fake_worker):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_tools"
    assert body["pending_tool_calls"]
    assert body["pending_tool_calls"][0]["function"]["name"] == "bash"
    run_id = body["run_id"]
    assert fake_worker.main_calls == 1
    assert fake_worker.sidekick_calls == 1

    # Resume with the tool result.
    response = client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [
                {
                    "tool_call_id": "call_bash_1",
                    "content": "hello",
                }
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "Final report" in (body["report"] or "")
    assert fake_worker.sidekick_calls == 2
    assert fake_worker.main_calls >= 2


def test_fusion_status(client, fake_worker):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    run_id = response.json()["run_id"]
    response = client.get(f"/v1/fusion/runs/{run_id}", headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["status"] == "awaiting_tools"
    assert body["pending_tool_calls"]
    assert body["report"] is None


def test_fusion_idempotent_follow_up(client, fake_worker):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    run_id = response.json()["run_id"]
    request_id = uuid.uuid4().hex
    payload = {
        "request_id": request_id,
        "tool_results": [
            {"tool_call_id": "call_bash_1", "content": "hello"}
        ],
    }
    first = client.post(f"/v1/fusion/follow_up/{run_id}", headers=_headers(), json=payload)
    assert first.status_code == 200
    second = client.post(f"/v1/fusion/follow_up/{run_id}", headers=_headers(), json=payload)
    assert second.status_code == 200
    assert first.json() == second.json()
    assert fake_worker.sidekick_calls == 2


def test_fusion_missing_tool_result(client, fake_worker):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    run_id = response.json()["run_id"]
    response = client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [
                {"tool_call_id": "wrong_id", "content": "hello"}
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "error"


def test_fusion_auth(client):
    response = client.post("/v1/fusion/delegate", json=SAMPLE_DELEGATE)
    assert response.status_code == 401


def test_fusion_delegate_requires_brief(client):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json={},
    )
    assert response.status_code == 400


def test_fusion_run_advance_cycle(fake_worker):
    run = fusion.FusionRun("test-run", "write a test")
    coordinator = fusion.FusionCoordinator()
    event = run.advance(coordinator=coordinator)
    assert event["status"] == "awaiting_tools"

    event = run.advance(
        tool_results=[{"tool_call_id": "call_bash_1", "content": "hello"}],
        request_id="req-1",
        coordinator=coordinator,
    )
    assert event["status"] == "completed"
    assert "Final report" in (event["report"] or "")


def test_fusion_run_max_follow_ups(fake_worker):
    # accept_after=2 means the main rejects the first two sidekick reports.
    fake_worker.accept_after = 2
    run = fusion.FusionRun("test-run-2", "write a test")
    coordinator = fusion.FusionCoordinator()
    event = run.advance(coordinator=coordinator)
    assert event["status"] == "awaiting_tools"

    # The sidekick first resumes to a report. The main rejects and asks for a
    # follow-up. The sidekick retries up to max_follow_ups=3, then the main
    # accepts on the third review.
    event = run.advance(
        tool_results=[{"tool_call_id": "call_bash_1", "content": "hello"}],
        request_id="req-1",
        coordinator=coordinator,
    )
    assert event["status"] == "completed"


def test_fusion_file_persistence_round_trip(file_client, fake_worker, tmp_path):
    """A FusionRun survives a simulated process restart through the file store."""
    response = file_client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "awaiting_tools"
    run_id = body["run_id"]

    # Confirm the run was written to disk.
    assert (tmp_path / f"{run_id}.pkl").exists()

    # Resume: a fresh coordinator is created on the server side and the run is
    # reloaded from the file store.
    response = file_client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [
                {"tool_call_id": "call_bash_1", "content": "hello"}
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert "Final report" in (body["report"] or "")
