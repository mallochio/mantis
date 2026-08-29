"""Tests for mantis-fusion main/sidekick orchestration."""

from __future__ import annotations

import json
import pickle
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import api
import fusion
import providers
import pytest
import serve_config
import tool_exec
import utils
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
        if first_content.startswith(fusion.MAIN_PREAMBLE):
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


DEFAULT_USAGE = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}

BASH_TOOL = {
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

BASH_CALL = [
    {
        "id": "call_0",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "echo hello"}'},
    }
]


class SequenceWorker:
    """Deterministic worker that returns a configured sequence of outputs."""

    def __init__(
        self,
        main_outputs: list[tuple[str, list[dict[str, Any]] | None, dict[str, Any]]],
        sidekick_outputs: list[tuple[str, list[dict[str, Any]] | None, dict[str, Any]]],
    ) -> None:
        self.main_outputs = main_outputs
        self.sidekick_outputs = sidekick_outputs
        self.main_idx = 0
        self.sidekick_idx = 0
        self.calls: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    @staticmethod
    def _message(
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    def _next(
        self,
        outputs: list[tuple[str, list[dict[str, Any]] | None, dict[str, Any]]],
        idx: int,
    ) -> tuple[str, list[dict[str, Any]] | None, dict[str, Any]]:
        if idx < len(outputs):
            return outputs[idx]
        if outputs:
            return outputs[-1]
        return ("", None, DEFAULT_USAGE)

    def __call__(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.calls.append((slot, messages, tools))
        first_content = messages[0].get("content", "")
        if first_content.startswith(fusion.MAIN_PREAMBLE):
            content, tool_calls, usage = self._next(self.main_outputs, self.main_idx)
            self.main_idx += 1
        else:
            content, tool_calls, usage = self._next(self.sidekick_outputs, self.sidekick_idx)
            self.sidekick_idx += 1
        return self._message(content, tool_calls), usage


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


def _fusion_config_file(
    tmp_path,
    *,
    main_tools: str = "plan+review",
    max_follow_ups: int = 2,
    main: str = "gpt-5_6-sol",
    sidekick: str = "deepseek-v4-flash",
):
    path = tmp_path / "catalog.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[fusion]\n"
        f'main = "{main}"\n'
        f'sidekick = "{sidekick}"\n'
        f'main_tools = "{main_tools}"\n'
        f"max_follow_ups = {max_follow_ups}\n"
    )
    return fusion.FusionConfig(path)


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
        "tool_results": [{"tool_call_id": "call_bash_1", "content": "hello"}],
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
            "tool_results": [{"tool_call_id": "wrong_id", "content": "hello"}],
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
            "tool_results": [{"tool_call_id": "call_bash_1", "content": "hello"}],
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


def test_fusion_trim_prunes_tool_results_before_dropping_prefix():
    coordinator = fusion.FusionCoordinator()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "keep me"},
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
        {"role": "tool", "tool_call_id": "call_1", "content": "x" * 20_000},
        {"role": "user", "content": "latest"},
    ]
    trimmed = coordinator._trim_messages(messages, 500)
    assert [m["role"] for m in trimmed] == ["system", "user", "assistant", "tool", "user"]
    assert trimmed[1]["content"] == "keep me"
    assert len(trimmed[-2]["content"]) < 3_000
    assert trimmed[-1]["content"] == "latest"


def test_fusion_trim_keeps_frozen_prefix_when_budget_allows():
    coordinator = fusion.FusionCoordinator()
    prefix = [
        {"role": "system", "content": fusion.MAIN_PREAMBLE},
        {"role": "user", "content": "client history"},
        {"role": "assistant", "content": "ack"},
    ]
    first = coordinator._trim_messages([*prefix, {"role": "user", "content": "plan now"}], 100_000)
    second = coordinator._trim_messages(
        [
            *prefix,
            {"role": "user", "content": "plan now"},
            {"role": "assistant", "content": "PLAN:\nx\nBRIEF:\ny"},
        ],
        100_000,
    )
    assert first[:3] == prefix
    assert second[:3] == prefix


def test_fusion_freezes_tools_in_name_order():
    tools = [
        {"type": "function", "function": {"name": "zsh", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
    ]
    run = fusion.FusionRun("tool-order", "brief", tools)
    assert [tool["function"]["name"] for tool in run.tools] == ["bash", "zsh"]


def test_fusion_repeat_tool_reminder_is_a_user_message():
    run = fusion.FusionRun("repeat", "brief")
    run.active_role = "main"
    run.pending_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": "{}"},
        }
    ]
    for _ in range(3):
        run._append_tool_results([{"tool_call_id": "call_1", "content": "ok"}])
    reminders = [
        message
        for message in run.main_messages
        if message.get("role") == "user" and "<system-reminder>" in str(message.get("content", ""))
    ]
    assert reminders
    assert not any(
        message.get("role") == "system" and "Advisory Notice" in str(message.get("content", ""))
        for message in run.main_messages
    )


def test_fusion_compaction_replays_same_slot_and_keeps_system(monkeypatch):
    coordinator = fusion.FusionCoordinator()
    monkeypatch.setattr(coordinator, "_output_tokens_for", lambda _slot: 256)
    calls: list[dict[str, Any]] = []

    def fake_provider(spec, messages, max_tokens, temperature, tools=None):
        calls.append({"spec": spec, "messages": messages, "tools": tools})
        last = messages[-1]["content"] if messages else ""
        if last == fusion.COMPACTION_INSTRUCTION:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<compacted-summary>inspected files</compacted-summary>",
                        }
                    }
                ],
                "usage": {},
            }
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}], "usage": {}}

    monkeypatch.setattr(providers, "_provider_response", fake_provider)
    pad = "x" * 320
    messages = [{"role": "system", "content": "system prompt"}]
    for index in range(6):
        messages.append({"role": "user", "content": f"{index}-{pad}"})
        messages.append({"role": "assistant", "content": f"a{index}-{pad}"})
    messages.append({"role": "user", "content": "latest"})
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "read"}},
    ]
    fitted = coordinator._fit_messages("gpt-5_6-sol", messages, tools, 400)
    assert fitted[0]["content"] == "system prompt"
    assert fitted[1]["content"].startswith("<compacted-summary>")
    assert "inspected files" in fitted[1]["content"]
    assert fitted[-1]["content"] == "latest"
    assert calls
    compact = calls[0]
    assert compact["spec"] == "gpt-5_6-sol"
    assert compact["tools"] == tools
    assert compact["messages"][0]["content"] == "system prompt"
    assert compact["messages"][-1]["content"] == fusion.COMPACTION_INSTRUCTION


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
                is_review = any(
                    msg.get("role") == "user" and fusion.REVIEW_PROMPT in msg.get("content", "")
                    for msg in messages
                )
                if is_review:
                    return ({"role": "assistant", "content": "ACCEPT"}, usage)
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


def test_fusion_plan_and_brief_match_streaming_and_non_streaming(client, fake_worker):
    payload = {
        "model": "mantis/fusion",
        "messages": [{"role": "user", "content": "write a hello world script"}],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }
    regular = client.post("/v1/chat/completions", headers=_headers(), json=payload)
    fake_worker.sidekick_calls = 0
    fake_worker.main_calls = 0
    fake_worker.reviews = 0
    stream = client.post(
        "/v1/chat/completions", headers=_headers(), json={**payload, "stream": True}
    )
    trace = regular.json()["choices"][0]["message"]["reasoning"]
    streamed = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("reasoning", "")
        for line in stream.text.splitlines()
        if line.startswith("data: {")
    )
    assert "implement and test the brief" in trace
    assert "implement, run tests, and lint" in trace
    assert trace == streamed


def test_fusion_trace_excludes_opaque_metadata(client, fake_worker, monkeypatch):
    opaque_values = {
        "signature": "signed-provider-signature-123",
        "encrypted": "encrypted-thinking-payload-456",
        "provider_id": "provider-internal-id-789",
    }

    def worker_with_opaque_metadata(slot, messages, tools):
        message, usage = fake_worker(slot, messages, tools)
        message["reasoning_details"] = [
            {
                "type": "reasoning",
                "id": opaque_values["provider_id"],
                "summary": [{"type": "summary_text", "text": "Safe summary."}],
                "signature": opaque_values["signature"],
                "encrypted_content": opaque_values["encrypted"],
            }
        ]
        message["_anthropic_content"] = [
            {
                "type": "thinking",
                "thinking": "Safe textual thinking.",
                "signature": opaque_values["signature"],
            }
        ]
        message["_anthropic_tool_ids"] = {"call": opaque_values["provider_id"]}
        return message, usage

    # Metadata is stored for provider replay, but only safe textual summaries
    # may enter the explicit opt-in provider section.
    message, _usage = worker_with_opaque_metadata(
        "gpt-5_6-sol", [{"role": "system", "content": fusion.MAIN_PREAMBLE}], None
    )
    trace = fusion._extract_reasoning_trace([message])
    assert "Safe summary." in trace
    assert "Safe textual thinking." in trace
    for value in opaque_values.values():
        assert value not in trace

    run = fusion.FusionRun("opaque-trace", "x")
    run.plan = "safe plan"
    run.sidekick_brief = "safe brief"
    run.main_messages.append(message)
    public_trace = fusion._orchestration_trace(run, include_reasoning=True)
    for value in opaque_values.values():
        assert value not in public_trace


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
    assert "reasoning" in message
    assert "implement and test the brief" in message["reasoning"]
    assert "implement, run tests, and lint" in message["reasoning"]
    assert "I will call bash" not in message["reasoning"]


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


def test_fusion_main_driver_can_call_tools_during_planning(client, monkeypatch, tmp_path):
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", _fusion_config_file(tmp_path))
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
        if first_content.startswith(fusion.MAIN_PREAMBLE):
            is_review = any(
                m.get("role") == "user" and fusion.REVIEW_PROMPT in m.get("content", "")
                for m in messages
            )
            if is_review:
                return {
                    "choices": [{"message": {"role": "assistant", "content": "ACCEPT"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                }
            has_tool_result = any(m.get("role") == "tool" for m in messages)
            if not has_tool_result:
                # Main calls a tool during planning
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_read_1",
                                        "type": "function",
                                        "function": {
                                            "name": "read_file",
                                            "arguments": '{"path": "config.py"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            # After tool result, emit plan and brief
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "PLAN: inspected config, now edit\nBRIEF: edit config.py",
                        }
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
        # Sidekick completes task
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Done editing config.py.",
                    }
                }
            ],
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
        if first_content.startswith(fusion.MAIN_PREAMBLE):
            is_review = any(fusion.REVIEW_PROMPT in m.get("content", "") for m in messages)
            if not is_review:
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "PLAN: test\nBRIEF: run tests",
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
                }
            # Record the review prompt
            review_prompts.extend(
                m["content"] for m in messages if fusion.REVIEW_PROMPT in m.get("content", "")
            )
            return {
                "choices": [{"message": {"role": "assistant", "content": "ACCEPT"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            }
        # Sidekick calls run_test then finishes
        has_tool_res = any(m.get("role") == "tool" for m in messages)
        if not has_tool_res:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_t1",
                                    "type": "function",
                                    "function": {
                                        "name": "run_test",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
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
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "Run test suite"}],
            "tools": tools,
        },
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


def test_fusion_retries_malformed_plan(monkeypatch):
    """If the main model emits prose instead of PLAN:/BRIEF:, Fusion retries once."""
    main_outputs = [
        ("I will inspect the codebase and then make a plan.", None, DEFAULT_USAGE),
        (
            "PLAN: inspect and extract abstractions\nBRIEF: implement the first two targets",
            None,
            DEFAULT_USAGE,
        ),
        ("ACCEPT", None, DEFAULT_USAGE),
    ]
    sidekick_outputs = [
        ("", BASH_CALL, DEFAULT_USAGE),
        ("Done.", None, DEFAULT_USAGE),
    ]
    worker = SequenceWorker(main_outputs, sidekick_outputs)
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)

    run = fusion.FusionRun("retry-run", "spin out Cerberus", tools=[BASH_TOOL])
    event = run.advance(coordinator=fusion.FusionCoordinator())
    assert event["status"] == "awaiting_tools"
    assert worker.main_idx == 2  # bad plan + retry produced a good plan

    event = run.advance(
        tool_results=[{"tool_call_id": "call_0", "content": "hello"}],
        request_id="req-1",
        coordinator=fusion.FusionCoordinator(),
    )
    assert event["status"] == "completed"
    assert "Done." in (event["report"] or "")


def test_fusion_rejects_unknown_planning_tools(monkeypatch, tmp_path):
    """If the main calls a tool not in the allowed set, Fusion forces it back to text."""
    ipython_call = [
        {
            "id": "call_ipython",
            "type": "function",
            "function": {
                "name": "ipython",
                "arguments": '{"code": "1+1"}',
            },
        }
    ]
    main_outputs = [
        ("", ipython_call, DEFAULT_USAGE),
        ("PLAN: use bash\nBRIEF: implement the task", None, DEFAULT_USAGE),
        ("ACCEPT", None, DEFAULT_USAGE),
    ]
    sidekick_outputs = [("Done.", None, DEFAULT_USAGE)]
    worker = SequenceWorker(main_outputs, sidekick_outputs)
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    config = _fusion_config_file(tmp_path)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)

    run = fusion.FusionRun("unknown-tool-run", "do work", tools=[BASH_TOOL])
    event = run.advance(coordinator=fusion.FusionCoordinator(config))
    assert event["status"] == "completed"
    assert "Done." in (event["report"] or "")
    # The retry should have been invoked with tools=None to stop hallucination.
    assert any(t is None for _s, _m, t in worker.calls)


def test_fusion_enforces_planning_tool_budget(monkeypatch, tmp_path):
    """After two planning tool rounds, the main must produce text or error."""
    bash_call_1 = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"ls"}'},
        }
    ]
    bash_call_2 = [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"cat"}'},
        }
    ]
    bash_call_3 = [
        {
            "id": "call_3",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command":"more"}'},
        }
    ]
    main_outputs = [
        ("", bash_call_1, DEFAULT_USAGE),
        ("", bash_call_2, DEFAULT_USAGE),
        ("", bash_call_3, DEFAULT_USAGE),
    ]
    sidekick_outputs = [("Done.", None, DEFAULT_USAGE)]
    worker = SequenceWorker(main_outputs, sidekick_outputs)
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    config = _fusion_config_file(tmp_path)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)

    run = fusion.FusionRun("budget-run", "do work", tools=[BASH_TOOL])
    event = run.advance(coordinator=fusion.FusionCoordinator(config))
    assert event["status"] == "awaiting_tools"
    assert run.planning_tool_rounds == 1

    event = run.advance(
        tool_results=[{"tool_call_id": "call_1", "content": "ok"}],
        request_id="req-1",
        coordinator=fusion.FusionCoordinator(config),
    )
    assert event["status"] == "awaiting_tools"
    assert run.planning_tool_rounds == 2

    event = run.advance(
        tool_results=[{"tool_call_id": "call_2", "content": "ok"}],
        request_id="req-2",
        coordinator=fusion.FusionCoordinator(config),
    )
    assert event["status"] == "error"
    assert run.status == "error"
    assert "plan was required" in (run.error or "")


def test_fusion_default_lead_has_no_tools_and_sidekick_does(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: delegate work\nBRIEF: implement and test", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    config = fusion.FusionConfig(Path("config/catalog.toml"))
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)

    run = fusion.FusionRun("default-tools", "do work", tools=[BASH_TOOL])
    event = run.advance(coordinator=fusion.FusionCoordinator(config))

    assert event["status"] == "completed"
    main_calls = [call for call in worker.calls if call[1][0]["content"] == fusion.MAIN_PREAMBLE]
    sidekick_calls = [
        call for call in worker.calls if call[1][0]["content"] == fusion.SIDEKICK_PREAMBLE
    ]
    assert [call[2] for call in main_calls] == [None, None]
    assert sidekick_calls[0][2] == [BASH_TOOL]


def test_fusion_structured_brief_packet(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: inspect then edit\nBRIEF: edit the target", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "brief-packet",
        "latest ask",
        messages=[
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "latest ask"},
        ],
        delegation_mode="forced",
    )
    run.advance()

    packet = run.sidekick_messages[1]["content"]
    assert packet.startswith(fusion.FUSION_BRIEF_OPEN)
    assert packet.endswith(fusion.FUSION_BRIEF_CLOSE)
    assert "goal: fix the bug" in packet
    assert "latest_user: latest ask" in packet
    assert "plan: inspect then edit" in packet
    assert "brief: edit the target" in packet
    assert run.goal == "fix the bug"
    assert run.latest_user == "latest ask"


def test_fusion_follow_up_cap_and_event_telemetry(monkeypatch, tmp_path):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("FOLLOW_UP: revise", None, DEFAULT_USAGE),
        ],
        [("report", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    config = _fusion_config_file(tmp_path, main_tools="none", max_follow_ups=1)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    run = fusion.FusionRun("capped", "goal")

    event = run.advance(coordinator=fusion.FusionCoordinator(config))

    assert event["status"] == "completed"
    assert event["follow_up_capped"] is True
    assert event["follow_up_count"] == 1
    assert {"usage_models", "cost", "follow_up_count"} <= event.keys()
    follow_ups = [m["content"] for m in run.sidekick_messages if m.get("role") == "user"][1:]
    assert len(follow_ups) == 1  # only one follow-up was actually sent
    assert follow_ups[0].startswith(fusion.FUSION_FOLLOW_UP_OPEN)
    assert follow_ups[0].endswith(fusion.FUSION_FOLLOW_UP_CLOSE)


def test_fusion_lane_cache_namespaces(monkeypatch):
    """Lane namespaces are content-addressed: stable per lane, differ across lanes,
    and identical for a second run with the same brief/tools (cross-run reuse)."""

    def run_once(run_id: str) -> tuple[list[str], fusion.FusionRun]:
        worker = SequenceWorker(
            [
                ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
                ("ACCEPT", None, DEFAULT_USAGE),
            ],
            [("Done.", None, DEFAULT_USAGE)],
        )
        namespaces: list[str] = []
        run = fusion.FusionRun(run_id, "same goal", tools=[BASH_TOOL])

        def capture(_coordinator, slot, messages, tools):
            namespaces.append(run.cache_namespace)
            return worker(slot, messages, tools)

        monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", capture)
        event = run.advance()
        assert event["status"] == "completed", event
        return namespaces, run

    # 3 worker calls per run: planning (main), sidekick, review (main).
    ns1, run1 = run_once("cache-lanes-a")
    ns2, _run2 = run_once("cache-lanes-b")

    assert ns1[0] == ns1[2]  # stable across calls within the main lane
    assert ns1[0] != ns1[1]  # main and sidekick lanes differ
    assert ns1 == ns2  # same brief/tools -> same namespaces despite different run ids
    assert run1.cache_namespace == ns1[2]


def test_fusion_records_tool_observations(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("", BASH_CALL, DEFAULT_USAGE), ("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("tool-observation", "goal", tools=[BASH_TOOL])
    assert run.advance()["status"] == "awaiting_tools"

    event = run.advance([{"tool_call_id": "call_0", "content": "hello"}])

    assert event["status"] == "completed"
    assert run.tool_observations == [{"name": "bash", "is_error": False, "is_test": False}]


def test_fusion_uses_slots_frozen_on_run(monkeypatch, tmp_path):
    run_config = _fusion_config_file(
        tmp_path / "run", main="run-main", sidekick="run-sidekick", main_tools="none"
    )
    coordinator_config = _fusion_config_file(
        tmp_path / "coordinator", main="other-main", sidekick="other-sidekick", main_tools="none"
    )
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", run_config)
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("frozen-slots", "goal")

    event = run.advance(coordinator=fusion.FusionCoordinator(coordinator_config))

    assert event["status"] == "completed"
    assert [slot for slot, _messages, _tools in worker.calls] == [
        "run-main",
        "run-sidekick",
        "run-main",
    ]


def test_fusion_new_fields_survive_pickle(monkeypatch, tmp_path):
    config = _fusion_config_file(tmp_path, main_tools="plan+review")
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    run = fusion.FusionRun("pickled", "goal")
    run.follow_up_capped = True
    run.cache_namespace = "pickled:sidekick"

    restored = pickle.loads(pickle.dumps(run))  # noqa: S301 - trusted local round-trip

    assert restored.follow_up_capped is True
    assert restored.main_tools_policy == frozenset({"plan", "review"})
    assert restored.cache_namespace == "pickled:sidekick"


def test_fusion_writes_learning_record_on_accept(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_LEARNING", "1")
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_LEARNING_INSTANCE", "fusion-test")
    monkeypatch.setattr(fusion, "RUN_STORE", "memory")
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.create_fusion_run("goal")

    event = fusion.advance_fusion_run(run.run_id)

    assert event["status"] == "completed"
    record = json.loads(runs._learning_path().read_text().splitlines()[-1])
    assert record["mode"] == "fusion"
    assert record["terminated_by"] == "fusion_accept"
    assert record["task"] == "goal"
    assert record["pool"] == [run.main_slot, run.sidekick_slot]
    assert record["turn_count"] > 0
    assert record["steps"]
    for step in record["steps"]:
        assert step["role"] in ("main", "sidekick")
        assert step["model"] in (run.main_slot, run.sidekick_slot)


def test_fusion_writes_learning_record_on_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_LEARNING", "1")
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_LEARNING_INSTANCE", "fusion-test-error")
    monkeypatch.setattr(fusion, "RUN_STORE", "memory")
    # Main keeps emitting tool calls instead of the required PLAN/BRIEF text.
    worker = SequenceWorker(
        [("", BASH_CALL, DEFAULT_USAGE)],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.create_fusion_run("goal", tools=[BASH_TOOL])

    event = fusion.advance_fusion_run(run.run_id)

    assert event["status"] == "error"
    record = json.loads(runs._learning_path().read_text().splitlines()[-1])
    assert record["mode"] == "fusion"
    assert record["terminated_by"] == "fusion_error"


def test_fusion_status_reuses_event_telemetry(monkeypatch):
    monkeypatch.setattr(providers, "_price_map", lambda: {})
    run = fusion.FusionRun("status-telemetry", "goal")
    fusion._put_run(run)
    status = fusion.fusion_run_status(run.run_id)
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= status.keys()


def test_fusion_cost_resolves_catalog_slot(monkeypatch):
    prices = {"openai/gpt-5.6-sol": (0.001, 0.002)}
    resolved = providers.ResolvedModelSpec(
        adapter="openai",
        model="gpt-5.6-sol",
        effort=None,
        base_url="http://test",
        credential_env="TEST_KEY",
        binding=None,
        protocols=("chat_completions",),
        slot="gpt-5_6-sol",
    )
    monkeypatch.setattr(providers, "_price_map", lambda: prices)
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda _model: resolved)
    monkeypatch.setattr(providers, "_cache_read_prices", lambda: {})

    usage = {"gpt-5_6-sol": {"prompt_tokens": 10, "completion_tokens": 5}}
    assert providers._usage_cost(usage) == 0.02
    breakdown = providers._cost_breakdown(usage)
    assert breakdown["known"] is True
    assert breakdown["total"] == 0.02
    assert breakdown["models"][0]["model"] == "gpt-5_6-sol"
    assert breakdown["models"][0]["cost"] == 0.02


def test_fusion_price_suffix_collision_is_unpriced(monkeypatch):
    """An ambiguous bare-suffix match is not attributed to a price.

    Guessing among equally-suffixed namespaced ids could charge the wrong
    provider, so an ambiguous key is reported as unpriced instead.
    """

    def not_a_slot(_model):
        raise RuntimeError("not a catalog slot")

    prices = {
        "zvendor/bare-model": (0.003, 0.004),
        "avendor/bare-model": (0.001, 0.002),
    }
    monkeypatch.setattr(providers, "_price_map", lambda: prices)
    monkeypatch.setattr(providers, "_resolve_model_spec", not_a_slot)
    monkeypatch.setattr(providers, "_cache_read_prices", lambda: {})

    usage = {"bare-model": {"prompt_tokens": 10, "completion_tokens": 5}}
    # Two namespaced keys share the bare suffix, so no single price is chosen.
    assert providers._usage_cost(usage) is None


def test_fusion_stray_planning_tool_calls_are_not_honored(monkeypatch):
    """With main_tools="none", stray lead tool calls are forced back to text."""
    worker = SequenceWorker(
        [
            ("", BASH_CALL, DEFAULT_USAGE),
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    config = fusion.FusionConfig(Path("config/catalog.toml"))
    assert config.main_tools() == "none"
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)

    run = fusion.FusionRun("stray-tools", "do work", tools=[BASH_TOOL])
    event = run.advance(coordinator=fusion.FusionCoordinator(config))

    # The run never suspends for tool results; it completes via the reminder path.
    assert event["status"] == "completed"
    assert run.planning_tool_rounds == 0
    # stray tool-call + forced retry + review
    assert worker.main_idx == 3
    main_tool_args = [
        tools
        for (_slot, messages, tools) in worker.calls
        if messages[0]["content"] == fusion.MAIN_PREAMBLE
    ]
    assert main_tool_args == [None, None, None]
    stray_index = next(
        i for i, message in enumerate(run.main_messages) if message.get("tool_calls")
    )
    assert run.main_messages[stray_index + 1] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "Tool unavailable during Fusion planning.",
        "is_error": True,
    }


def test_fusion_review_can_call_tools_under_review_policy(monkeypatch, tmp_path):
    config = _fusion_config_file(tmp_path, main_tools="review")
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    review_tool_call = [
        {
            "id": "call_review",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "echo check"}'},
        }
    ]
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("", review_tool_call, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    coordinator = fusion.FusionCoordinator(config)
    run = fusion.FusionRun("review-tools", "goal", tools=[BASH_TOOL])

    event = run.advance(coordinator=coordinator)
    assert event["status"] == "awaiting_tools"
    assert run.active_role == "main"
    review_calls = [
        tools
        for (_slot, messages, tools) in worker.calls
        if messages[0]["content"] == fusion.MAIN_PREAMBLE and tools is not None
    ]
    assert review_calls == [[BASH_TOOL]]  # review got client tools; planning did not

    event = run.advance(
        tool_results=[{"tool_call_id": "call_review", "content": "check ok"}],
        request_id="req-1",
        coordinator=coordinator,
    )
    assert event["status"] == "completed"
    assert "Done." in (event["report"] or "")


def test_fusion_old_pickle_restores_legacy_defaults():
    """Pickles from before the config gate keep the old lead-tool behavior."""
    run = fusion.FusionRun("old-pickle", "goal")
    state = run.__dict__.copy()
    state.pop("lock", None)
    state.pop("request_lock", None)
    for key in ("main_tools_policy", "follow_up_capped", "query", "slot_models", "turns"):
        state.pop(key, None)

    restored = fusion.FusionRun.__new__(fusion.FusionRun)
    restored.__setstate__(state)

    assert restored.main_tools_policy == frozenset({"plan", "review"})
    assert restored.follow_up_capped is False
    assert restored.query == ""
    assert restored.slot_models == []
    assert restored.turns == []


def test_fusion_api_events_expose_telemetry(client, fake_worker):
    response = client.post(
        "/v1/fusion/delegate",
        headers=_headers(),
        json=SAMPLE_DELEGATE,
    )
    assert response.status_code == 200
    body = response.json()
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= body.keys()
    run_id = body["run_id"]

    response = client.post(
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
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= body.keys()

    status = client.get(f"/v1/fusion/runs/{run_id}", headers=_headers()).json()
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= status.keys()


def test_fusion_available_answer_skips_sidekick(monkeypatch):
    worker = SequenceWorker(
        [("ANSWER: the third word is are", None, DEFAULT_USAGE)],
        [("should not run", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("answer-run", "what is the third word?", delegation_mode="available")
    event = run.advance(coordinator=fusion.FusionCoordinator())
    assert event["status"] == "completed"
    assert event["report"] == "the third word is are"
    assert worker.sidekick_idx == 0
    assert run.completed_via == "answer"


def test_fusion_forced_rejects_answer_then_plans(monkeypatch):
    worker = SequenceWorker(
        [
            ("ANSWER: nope", None, DEFAULT_USAGE),
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("forced-answer", "do work", delegation_mode="forced")
    event = run.advance(coordinator=fusion.FusionCoordinator())
    assert event["status"] == "completed"
    assert "Done." in (event["report"] or "")
    assert worker.sidekick_idx >= 1


def test_fusion_message_resume_answers_without_sidekick(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
            ("ANSWER: resumed clarification", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("resume-msg", "goal", delegation_mode="available")
    assert run.advance()["status"] == "completed"
    sidekick_before = worker.sidekick_idx
    event = run.advance(message="what did you just do?")
    assert event["status"] == "completed"
    assert event["report"] == "resumed clarification"
    assert worker.sidekick_idx == sidekick_before


def test_fusion_chat_resume_via_run_id_header(client, monkeypatch):
    worker = SequenceWorker(
        [
            ("ANSWER: first", None, DEFAULT_USAGE),
            ("ANSWER: second", None, DEFAULT_USAGE),
        ],
        [("unused", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    first = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "say first"}],
        },
    )
    assert first.status_code == 200
    run_id = first.headers.get("X-Mantis-Run-Id")
    assert run_id
    assert first.json()["choices"][0]["message"]["content"] == "first"

    second = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "X-Mantis-Run-Id": run_id},
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "user", "content": "say first"},
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "say second"},
            ],
        },
    )
    assert second.status_code == 200
    assert second.headers.get("X-Mantis-Run-Id") == run_id
    assert second.json()["choices"][0]["message"]["content"] == "second"
    assert worker.sidekick_idx == 0


def test_fusion_chat_resume_via_encoded_tool_call_id(client, monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
            ("ANSWER: from history", None, DEFAULT_USAGE),
        ],
        [("", BASH_CALL, DEFAULT_USAGE), ("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    tools = [BASH_TOOL]
    first = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "do work"}],
            "tools": tools,
        },
    )
    assert first.status_code == 200
    tool_call = first.json()["choices"][0]["message"]["tool_calls"][0]
    run_id = first.headers["X-Mantis-Run-Id"]

    second = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "user", "content": "do work"},
                first.json()["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "ok",
                },
            ],
            "tools": tools,
        },
    )
    assert second.status_code == 200
    assert "Done." in second.json()["choices"][0]["message"]["content"]

    third = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [
                {"role": "user", "content": "do work"},
                first.json()["choices"][0]["message"],
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "ok",
                },
                second.json()["choices"][0]["message"],
                {"role": "user", "content": "remind me"},
            ],
            "tools": tools,
        },
    )
    assert third.status_code == 200
    assert third.headers.get("X-Mantis-Run-Id") == run_id
    assert third.json()["choices"][0]["message"]["content"] == "from history"


def test_fusion_chat_missing_run_id_starts_new_available_run(client, monkeypatch):
    worker = SequenceWorker(
        [("ANSWER: fresh", None, DEFAULT_USAGE)],
        [("unused", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    response = client.post(
        "/v1/chat/completions",
        headers={**_headers(), "X-Mantis-Run-Id": "does-not-exist"},
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fresh"
    assert response.headers.get("X-Mantis-Run-Id")
    assert response.headers.get("X-Mantis-Run-Id") != "does-not-exist"


def test_fusion_sidekick_escalate_to_main_answer(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ANSWER: escalated and answered", None, DEFAULT_USAGE),
        ],
        [("ESCALATE_TO_MAIN: brief is ambiguous", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("escalate", "goal", delegation_mode="available")
    event = run.advance()
    assert event["status"] == "completed"
    assert event["report"] == "escalated and answered"
    assert run.follow_up_count == 1
    assert run.completed_via == "answer"


def test_fusion_sidekick_tool_round_cap_escalates(monkeypatch, tmp_path):
    bash_calls = [
        (
            "",
            [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
            DEFAULT_USAGE,
        )
        for i in range(3)
    ]
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ANSWER: capped", None, DEFAULT_USAGE),
        ],
        [*bash_calls, ("ESCALATE_TO_MAIN: over budget", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    path = tmp_path / "catalog.toml"
    path.write_text(
        "[fusion]\n"
        'main = "gpt-5_6-sol"\n'
        'sidekick = "gpt-5_6-luna"\n'
        'main_tools = "none"\n'
        "max_follow_ups = 2\n"
        "sidekick_max_tool_rounds = 2\n"
    )
    config = fusion.FusionConfig(path)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    coordinator = fusion.FusionCoordinator(config)
    run = fusion.FusionRun("cap-run", "goal", tools=[BASH_TOOL], delegation_mode="forced")

    event = run.advance(coordinator=coordinator)
    assert event["status"] == "awaiting_tools"
    event = run.advance(
        tool_results=[{"tool_call_id": "call_0", "content": "ok"}],
        request_id="r1",
        coordinator=coordinator,
    )
    assert event["status"] == "awaiting_tools"
    event = run.advance(
        tool_results=[{"tool_call_id": "call_1", "content": "ok"}],
        request_id="r2",
        coordinator=coordinator,
    )
    assert event["status"] == "completed"
    assert event["report"] == "capped"
    assert run.sidekick_tool_rounds == 2


def test_fusion_old_pickle_restores_continuity_defaults():
    run = fusion.FusionRun("old-continuity", "goal")
    state = run.__dict__.copy()
    state.pop("lock", None)
    state.pop("request_lock", None)
    for key in (
        "delegation_mode",
        "goal",
        "latest_user",
        "sidekick_tool_rounds",
        "completed_via",
    ):
        state.pop(key, None)

    restored = fusion.FusionRun.__new__(fusion.FusionRun)
    restored.__setstate__(state)

    assert restored.delegation_mode == "forced"
    assert restored.goal == "goal"
    assert restored.latest_user == "goal"
    assert restored.sidekick_tool_rounds == 0
    assert restored.completed_via == ""


def test_adaptive_router_escalates_through_sidekick_pool(monkeypatch, tmp_path):
    path = tmp_path / "catalog.toml"
    path.write_text(
        "[fusion]\n"
        'main = "run-main"\n'
        'sidekick = ["run-s1", "run-s2"]\n'
        "fallback_on_escalate = true\n"
        'main_tools = "none"\n'
        "max_follow_ups = 2\n"
    )
    config = fusion.FusionConfig(path)
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [
            ("ESCALATE_TO_MAIN: retry", None, DEFAULT_USAGE),
            ("done", None, DEFAULT_USAGE),
        ],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    coordinator = fusion.FusionCoordinator(config)
    run = fusion.FusionRun("router", "goal", delegation_mode="forced")

    event = run.advance(coordinator=coordinator)

    assert event["status"] == "completed"
    sidekick_slots = [call[0] for call in worker.calls if call[0].startswith("run-s")]
    assert sidekick_slots == ["run-s1", "run-s2"]


def test_fusion_structured_plan_with_worker_profile(monkeypatch):
    plan_json = json.dumps(
        {
            "complexity": 0.5,
            "main_task": "verify script",
            "sidekick_assignments": [{"task": "write a python utility", "profile": "coder"}],
        }
    )
    worker = SequenceWorker(
        [(plan_json, None, DEFAULT_USAGE), ("ACCEPT", None, DEFAULT_USAGE)],
        [("I wrote the script.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "struct",
        "goal",
        worker_profiles=[
            fusion.FusionWorkerProfile(
                name="coder",
                model="deepseek-v4-flash",
                instructions="You are a focused python coder.",
            )
        ],
    )

    event = run.advance()

    assert event["status"] == "completed"
    assert "wrote the script" in event["report"].lower()
    sidekick_calls = [call for call in worker.calls if call[0] == "deepseek-v4-flash"]
    assert sidekick_calls


def test_fusion_run_budget_enforces_max_turns(monkeypatch):
    worker = SequenceWorker(
        [
            ("PLAN: p\nBRIEF: b", None, DEFAULT_USAGE),
            ("ACCEPT", None, DEFAULT_USAGE),
        ],
        [("done", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "budget",
        "goal",
        delegation_mode="forced",
        budget=fusion.FusionRunBudget(max_turns=0),
    )

    event = run.advance()

    assert event["status"] == "error"
    assert "turn budget" in event["report"].lower()


def test_filter_tools_by_options_uses_bundles():
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "edit_file"}},
        {"type": "function", "function": {"name": "python"}},
    ]
    options = fusion.FusionToolOptions(enabled=["shell", "files"])
    filtered = utils._filter_tools_by_options(tools, options)
    names = {t["function"]["name"] for t in filtered}
    assert names == {"bash", "edit_file"}


def test_fusion_structured_lanes_step_in_parallel(monkeypatch):
    """Two sidekick lanes must rendezvous; sequential stepping breaks the barrier."""
    plan_json = json.dumps(
        {
            "complexity": 0.5,
            "main_task": "verify both",
            "sidekick_assignments": [{"task": "task a"}, {"task": "task b"}],
        }
    )
    barrier = threading.Barrier(2)
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    def worker(_self, slot, messages, tools):
        first = messages[0].get("content", "")
        if first.startswith(fusion.MAIN_PREAMBLE):
            is_review = any(
                msg.get("role") == "user" and fusion.REVIEW_PROMPT in msg.get("content", "")
                for msg in messages
            )
            text = "ACCEPT" if is_review else plan_json
            return ({"role": "assistant", "content": text}, dict(usage))
        barrier.wait(timeout=5)
        return ({"role": "assistant", "content": "done"}, dict(usage))

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "par",
        "goal",
        budget=fusion.FusionRunBudget(max_turns=10),
    )

    event = run.advance()

    assert event["status"] == "completed", event
    assert len(run.sidekick_reports) == 2
    assert event["usage"]["total_tokens"] == 8


def test_fusion_budget_guard_pickle_roundtrip():
    guard = fusion.FusionBudgetGuard(max_turns=3, max_tokens=100, timeout_ms=1000)
    guard.consume_turn()
    restored = pickle.loads(pickle.dumps(guard))  # noqa: S301
    assert restored.turns == 1
    restored.consume_turn()
    assert restored.turns == 2


def test_fusion_structured_preamble_delegates_and_legacy_unchanged():
    structured_run = fusion.FusionRun(
        "p-struct",
        "goal",
        budget=fusion.FusionRunBudget(max_turns=5),
    )
    preamble = structured_run.main_messages[0]["content"]
    assert preamble.startswith(fusion.MAIN_PREAMBLE)
    assert "Delegate all execution work" in preamble
    assert "frontier" in preamble

    legacy_run = fusion.FusionRun("p-legacy", "goal", delegation_mode="forced")
    assert legacy_run.main_messages[0]["content"] == fusion.MAIN_PREAMBLE


def test_fusion_frontier_profile_uses_main_slot(monkeypatch):
    plan_json = json.dumps(
        {
            "complexity": 0.9,
            "main_task": "verify integration",
            "sidekick_assignments": [
                {"task": "hard integration task", "profile": "frontier"},
                {"task": "mechanical rename"},
            ],
        }
    )
    worker = SequenceWorker(
        [(plan_json, None, DEFAULT_USAGE), ("ACCEPT", None, DEFAULT_USAGE)],
        [("hard done", None, DEFAULT_USAGE), ("rename done", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "frontier",
        "goal",
        budget=fusion.FusionRunBudget(max_turns=10),
    )

    event = run.advance()

    assert event["status"] == "completed"
    lane_slots = [
        call[0]
        for call in worker.calls
        if not call[1][0].get("content", "").startswith(fusion.MAIN_PREAMBLE)
    ]
    assert run.main_slot in lane_slots
    assert len(lane_slots) == 2


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_tool_exec_file_and_shell_tools(tmp_path):
    workspace = tool_exec.workspace_for("t", root=tmp_path)
    assert (
        tool_exec.execute(
            "write_file", {"path": "a.py", "content": "print('hi')"}, workspace
        )
        == "ok"
    )
    assert tool_exec.execute("read_file", {"path": "a.py"}, workspace) == "print('hi')"
    assert "a.py" in tool_exec.execute("list_files", {}, workspace)
    assert "print" in tool_exec.execute("search_files", {"query": "print"}, workspace)
    assert "hello" in tool_exec.execute("bash", {"command": "echo hello"}, workspace)
    assert "2" in tool_exec.execute("run_code", {"code": "print(1 + 1)"}, workspace)
    tool_exec.execute(
        "edit_file", {"path": "a.py", "old_text": "hi", "new_text": "bye"}, workspace
    )
    assert "bye" in tool_exec.execute("read_file", {"path": "a.py"}, workspace)


def test_tool_exec_rejects_escape_and_unknown(tmp_path):
    workspace = tool_exec.workspace_for("t2", root=tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        tool_exec.execute("read_file", {"path": "../secret"}, workspace)
    results = tool_exec.execute_calls([_tool_call("x", "unknown_tool", {})], workspace)
    assert results[0]["is_error"]


def test_tool_exec_calls_parallel_and_ordered(tmp_path):
    workspace = tool_exec.workspace_for("parallel", root=tmp_path)
    calls = [
        _tool_call("a", "bash", {"command": "sleep 0.5; printf A"}),
        _tool_call("b", "bash", {"command": "sleep 0.5; printf B"}),
    ]
    start = time.monotonic()
    results = tool_exec.execute_calls(calls, workspace)
    elapsed = time.monotonic() - start
    assert [result["tool_call_id"] for result in results] == ["a", "b"]
    assert [result["content"] for result in results] == ["A", "B"]
    assert elapsed < 0.9


def test_fusion_server_side_execution_completes_without_client(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOT", str(tmp_path))
    plan_json = json.dumps(
        {
            "complexity": 0.5,
            "main_task": "verify script",
            "sidekick_assignments": [{"task": "write and run a script"}],
        }
    )
    worker = SequenceWorker(
        [(plan_json, None, DEFAULT_USAGE), ("ACCEPT", None, DEFAULT_USAGE)],
        [
            (
                "",
                [
                    _tool_call(
                        "w1", "write_file", {"path": "script.py", "content": "print(40 + 2)"}
                    )
                ],
                DEFAULT_USAGE,
            ),
            (
                "",
                [
                    _tool_call(
                        "r1",
                        "run_code",
                        {"code": "import pathlib; print(pathlib.Path('script.py').read_text())"},
                    )
                ],
                DEFAULT_USAGE,
            ),
            ("script written and verified", None, DEFAULT_USAGE),
        ],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "server-exec",
        "goal",
        tool_options=fusion.FusionToolOptions(enabled=["files", "code"], server_execution=True),
    )

    event = run.advance()

    assert event["status"] == "completed", event
    assert event["pending_tool_calls"] is None
    assert "script written" in event["report"]
    assert (tmp_path / "server-exec" / "script.py").read_text() == "print(40 + 2)"


def test_fusion_server_execution_mixed_batch_suspends(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_FUSION_WORKSPACE_ROOT", str(tmp_path))
    plan_json = json.dumps(
        {
            "complexity": 0.5,
            "main_task": "verify",
            "sidekick_assignments": [{"task": "use a client tool too"}],
        }
    )
    worker = SequenceWorker(
        [(plan_json, None, DEFAULT_USAGE), ("ACCEPT", None, DEFAULT_USAGE)],
        [
            (
                "",
                [
                    _tool_call("w1", "write_file", {"path": "a.txt", "content": "x"}),
                    _tool_call("c1", "client_tool", {}),
                ],
                DEFAULT_USAGE,
            ),
            ("done", None, DEFAULT_USAGE),
        ],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun(
        "mixed",
        "goal",
        tool_options=fusion.FusionToolOptions(enabled=["files"], server_execution=True),
    )

    event = run.advance()

    assert event["status"] == "awaiting_tools"
    assert len(event["pending_tool_calls"]) == 2


def test_provider_stream_assembly_emits_output_and_reasoning_deltas(monkeypatch):
    events: list[dict[str, Any]] = []
    monkeypatch.setattr(providers, "_emit_progress", events.append)
    chunks = [
        {"choices": [{"delta": {"role": "assistant", "content": "hel"}}]},
        {"choices": [{"delta": {"content": "lo", "reasoning": "think"}}]},
        {"usage": {"prompt_tokens": 1, "completion_tokens": 2}},
    ]

    result = providers._assemble_streamed_completion(chunks)

    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["message"]["reasoning"] == "think"
    assert events == [
        {"type": "provider.output.delta", "delta": "hel"},
        {"type": "provider.output.delta", "delta": "lo"},
        {"type": "provider.reasoning.delta", "delta": "think"},
    ]
