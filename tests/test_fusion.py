"""Tests for mantis-fusion main/sidekick orchestration."""

from __future__ import annotations

import uuid
from typing import Any

import api
import fusion
import providers
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

    @staticmethod
    def _message(
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning: str | None = None,
        reasoning_details: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning:
            message["reasoning"] = reasoning
        if reasoning_details:
            message["reasoning_details"] = reasoning_details
        return message

    def __call__(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append((slot, messages, tools))
        first_content = messages[0].get("content", "")
        if first_content == fusion.MAIN_PREAMBLE:
            return self._main_response(messages)
        return self._sidekick_response(messages, tools)

    def _main_response(
        self,
        messages: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.main_calls += 1
        # Planning call: the user brief is the last user message before an assistant response.
        # Review call: the user message contains REVIEW_PROMPT.
        is_review = any(
            msg.get("role") == "user" and fusion.REVIEW_PROMPT in msg.get("content", "")
            for msg in messages
        )
        if not is_review:
            return (
                self._message(
                    "PLAN: implement and test the brief\nBRIEF: implement, run tests, and lint"
                ),
                {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )
        self.reviews += 1
        if self.reviews > self.accept_after:
            return (
                self._message("ACCEPT"),
                {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
            )
        return (
            self._message("FOLLOW_UP: add more tests before reporting"),
            {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58},
        )

    def _sidekick_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.sidekick_calls += 1
        # First call: request a tool. Subsequent calls: return a final report.
        if self.sidekick_calls == 1:
            return (
                self._message(
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
                    reasoning="I will call bash to verify the environment.",
                ),
                {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            )
        return (
            self._message(
                "Final report: implemented and tested successfully.",
                reasoning="The shell output confirms the implementation works.",
            ),
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


def test_fusion_context_window_trims_old_messages():
    coordinator = fusion.FusionCoordinator()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "old brief"},
        {"role": "assistant", "content": "old plan"},
        {"role": "user", "content": "new brief"},
    ]
    # Force a tiny budget so only the system and the newest user message fit.
    trimmed = coordinator._trim_messages(messages, 6)
    assert trimmed[0]["role"] == "system"
    assert [m["role"] for m in trimmed[1:]] == ["user"]
    assert trimmed[-1]["content"] == "new brief"


def test_fusion_trimmer_keeps_tool_call_result_pairs():
    coordinator = fusion.FusionCoordinator()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "do work"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    trimmed = coordinator._trim_messages(messages, 7)
    roles = [m["role"] for m in trimmed]
    assert roles == ["system", "assistant", "tool"]


def test_fusion_output_tokens_are_model_specific(monkeypatch):
    resolved = providers.ResolvedModelSpec(
        adapter="openai",
        model="gpt-5.6-luna",
        effort="medium",
        base_url="http://test",
        credential_env="TEST_KEY",
        binding=None,
        protocols=("chat_completions",),
        slot="gpt-5_6-luna",
        max_tokens=128000,
    )
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda _slot: resolved)
    coordinator = fusion.FusionCoordinator()
    assert coordinator._output_tokens_for("gpt-5_6-luna") == 128000
    assert coordinator.max_output_tokens == 4096


def test_fusion_chat_initial_call_returns_tool_calls(client, fake_worker):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "write a hello world script"}],
            "tools": tools,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mantis/fusion"
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    message = body["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == ""
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "bash"
    tool_call_id = message["tool_calls"][0]["id"]
    assert tool_call_id.startswith("f")
    assert "~" in tool_call_id
    assert fake_worker.main_calls == 1
    assert fake_worker.sidekick_calls == 1


def test_fusion_chat_follow_up_returns_report(client, fake_worker):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    # First call to get tool calls.
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "write a hello world script"}],
            "tools": tools,
        },
    )
    first = response.json()
    tool_call = first["choices"][0]["message"]["tool_calls"][0]
    tool_call_id = tool_call["id"]

    # Second call with the tool result.
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "user", "content": "write a hello world script"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call],
                },
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": "hello",
                },
            ],
            "tools": tools,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "Final report" in body["choices"][0]["message"]["content"]
    assert fake_worker.sidekick_calls == 2
    assert fake_worker.main_calls >= 2


def test_extract_reasoning_trace():
    messages = [
        {"role": "assistant", "content": "hi", "reasoning": "Plain reasoning."},
        {
            "role": "assistant",
            "content": "",
            "_anthropic_content": [
                {"type": "thinking", "thinking": "Anthropic thought."},
                {"type": "redacted_thinking", "data": "secret"},
            ],
        },
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": [
                {
                    "type": "reasoning",
                    "id": "r1",
                    "summary": [{"type": "summary_text", "text": "Responses summary."}],
                }
            ],
        },
    ]
    trace = fusion._extract_reasoning_trace(messages)
    assert "Plain reasoning." in trace
    assert "Anthropic thought." in trace
    assert "Responses summary." in trace
    assert "secret" not in trace


def test_extract_reasoning_trace_truncation():
    long_text = "x" * 10000
    messages = [{"role": "assistant", "content": "", "reasoning": long_text}]
    trace = fusion._extract_reasoning_trace(messages, max_chars=100)
    assert trace.endswith("\n...")
    assert len(trace) <= 105


def test_fusion_preserves_reasoning_metadata(monkeypatch):
    reasoning_details = [
        {
            "type": "reasoning",
            "id": "r1",
            "summary": [{"type": "summary_text", "text": "I need to run bash."}],
        }
    ]
    usage = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}

    class MetadataFakeWorker:
        def __init__(self) -> None:
            self.sidekick_calls: list[list[dict[str, Any]]] = []

        def __call__(
            self,
            slot: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            first_content = messages[0].get("content", "")
            if first_content == fusion.MAIN_PREAMBLE:
                return (
                    {"role": "assistant", "content": "PLAN: p\nBRIEF: b"},
                    usage,
                )
            self.sidekick_calls.append([dict(m) for m in messages])
            if len(self.sidekick_calls) == 1:
                return (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_bash_1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": '{"command": "echo hello"}',
                                },
                            }
                        ],
                        "reasoning_details": reasoning_details,
                    },
                    usage,
                )
            return (
                {"role": "assistant", "content": "Final report: done."},
                usage,
            )

    worker = MetadataFakeWorker()
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("metadata-run", "write a test")
    event = run.advance(coordinator=fusion.FusionCoordinator())
    assert event["status"] == "awaiting_tools"
    assert len(worker.sidekick_calls) == 1

    event = run.advance(
        tool_results=[{"tool_call_id": "call_bash_1", "content": "hello"}],
        request_id="req-1",
        coordinator=fusion.FusionCoordinator(),
    )
    assert event["status"] == "completed"
    # The second sidekick call must have received the first sidekick's
    # reasoning_details so it can be replayed by the upstream provider.
    second_call_messages = worker.sidekick_calls[1]
    assistant_msgs = [m for m in second_call_messages if m.get("role") == "assistant"]
    assert any(m.get("reasoning_details") == reasoning_details for m in assistant_msgs)


def test_fusion_chat_returns_reasoning_when_requested(client, fake_worker):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    headers = _headers()
    headers["X-Mantis-Return-Reasoning"] = "true"
    response = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "write a hello world script"}],
            "tools": tools,
        },
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message.get("finish_reason") == "tool_calls" or "tool_calls" in message
    assert "reasoning" in message
    assert "I will call bash" in message["reasoning"]


def test_fusion_chat_no_reasoning_by_default(client, fake_worker):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        }
    ]
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "write a hello world script"}],
            "tools": tools,
        },
    )
    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert "reasoning" not in message


def test_fusion_reasoning_not_exposed_to_other_providers():
    messages = [
        {
            "role": "assistant",
            "content": "text",
            "reasoning": "hidden",
            "reasoning_details": [{"id": "r"}],
            "_anthropic_content": [{"type": "thinking", "thinking": "hidden"}],
            "_anthropic_tool_ids": {"a": "b"},
        }
    ]
    sanitized = providers._sanitize_messages(
        messages,
        model="gpt-5.6-sol",
        is_anthropic=False,
        is_responses=False,
    )
    msg = sanitized[0]
    assert "reasoning" not in msg
    assert "reasoning_content" not in msg
    assert "reasoning_details" not in msg
    assert "_anthropic_content" not in msg
    assert "_anthropic_tool_ids" not in msg


def test_fusion_file_persistence_retains_reasoning_metadata(file_client, fake_worker, tmp_path):
    response = file_client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    assert response.status_code == 200
    body = response.json()
    run_id = body["run_id"]
    assert (tmp_path / f"{run_id}.pkl").exists()

    response = file_client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [{"tool_call_id": "call_bash_1", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"

    run = fusion.get_run(run_id)
    assert isinstance(run, fusion.FusionRun)
    assistant_msgs = [m for m in run.sidekick_messages if m.get("role") == "assistant"]
    assert any("reasoning" in m or "reasoning_details" in m for m in assistant_msgs)


def test_fusion_main_driver_can_call_tools_during_planning(client, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    calls = []

    def mock_worker(slot, messages, max_tokens=4096, temperature=0.7, worker_tools=None):
        calls.append((slot, messages, worker_tools))
        first_content = messages[0].get("content", "")
        if first_content == fusion.MAIN_PREAMBLE:
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if not has_tool_result:
                # Main calls a tool during planning
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{
                                "id": "call_read_1",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path": "config.py"}'},
                            }],
                        }
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            # After tool result, emit plan and brief
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "PLAN: inspected config, now edit\nBRIEF: edit config.py",
                    }
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        # Sidekick completes task
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Done editing config.py.",
                }
            }],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

    monkeypatch.setattr(providers, "_provider_response", mock_worker)

    # Initial call triggers main tool call
    res1 = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "system", "content": "Harness system rules"},
                {"role": "user", "content": "Update the config file"},
            ],
            "tools": tools,
        },
    )
    assert res1.status_code == 200
    body1 = res1.json()
    assert body1["choices"][0]["finish_reason"] == "tool_calls"
    t_call = body1["choices"][0]["message"]["tool_calls"][0]
    assert t_call["function"]["name"] == "read_file"

    # Follow up with tool result
    res2 = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "system", "content": "Harness system rules"},
                {"role": "user", "content": "Update the config file"},
                body1["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": t_call["id"],
                    "content": "PORT = 8080",
                },
            ],
            "tools": tools,
        },
    )
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["choices"][0]["finish_reason"] == "stop"
    assert "Done editing config.py." in body2["choices"][0]["message"]["content"]


def test_fusion_review_receives_sidekick_tool_activity(client, monkeypatch):
    tools = [
        {
            "type": "function",
            "function": {"name": "run_test", "parameters": {"type": "object"}},
        }
    ]
    review_prompts = []

    def mock_worker(slot, messages, max_tokens=4096, temperature=0.7, worker_tools=None):
        first_content = messages[0].get("content", "")
        if first_content == fusion.MAIN_PREAMBLE:
            is_review = any(fusion.REVIEW_PROMPT in m.get("content", "") for m in messages)
            if not is_review:
                return {
                    "choices": [{"message": {"role": "assistant", "content": "PLAN: test\nBRIEF: run tests"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            # Record the review prompt
            for m in messages:
                if fusion.REVIEW_PROMPT in m.get("content", ""):
                    review_prompts.append(m["content"])
            return {
                "choices": [{"message": {"role": "assistant", "content": "ACCEPT"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            }
        # Sidekick calls run_test then finishes
        has_tool_res = any(m.get("role") == "tool" for m in messages)
        if not has_tool_res:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_t1", "type": "function", "function": {"name": "run_test", "arguments": "{}"}}],
                    }
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": "All 5 tests passed."}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }

    monkeypatch.setattr(providers, "_provider_response", mock_worker)

    res1 = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={"model": "mantis/fusion", "messages": [{"role": "user", "content": "Run test suite"}], "tools": tools},
    )
    assert res1.status_code == 200
    t_call = res1.json()["choices"][0]["message"]["tool_calls"][0]

    res2 = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "user", "content": "Run test suite"},
                res1.json()["choices"][0]["message"],
                {"role": "tool", "tool_call_id": t_call["id"], "content": "OK: 5 passed"},
            ],
            "tools": tools,
        },
    )
    assert res2.status_code == 200
    assert len(review_prompts) == 1
    assert "Tool Activity by Sidekick:" in review_prompts[0]
    assert "run_test" in review_prompts[0]
    assert "OK: 5 passed" in review_prompts[0]

