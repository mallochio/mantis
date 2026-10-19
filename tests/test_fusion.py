"""Tests for mantis-fusion main/sidekick orchestration."""

from __future__ import annotations

import json
import pickle
import re
import time
import uuid
from pathlib import Path
from typing import Any

import api
import base_proxy
import fusion
import fusion_background
import fusion_config
import fusion_limits
import fusion_prompts
import httpx
import providers
import pytest
import runs
import serve_config
import tool_exec
import utils
from fastapi.testclient import TestClient

from tests.test_fusion_tool_mode import call, reply, use_tool_catalog

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


class FakeWorker:
    """Deterministic tool-delegation worker with client tool suspension."""

    def __init__(self) -> None:
        self.main_calls = 0
        self.sidekick_calls = 0
        self.calls = []

    def __call__(self, slot, messages, tools):
        self.calls.append((slot, messages, tools))
        if messages[0]["content"] != fusion_prompts.SIDEKICK_POLICY:
            self.main_calls += 1
            if self.main_calls == 1:
                return reply(
                    "", call("delegate", "sidekick", message="implement and test")
                ), DEFAULT_USAGE
            return reply("Final report: implemented and tested successfully."), DEFAULT_USAGE
        self.sidekick_calls += 1
        if self.sidekick_calls == 1:
            return reply(
                "",
                call("call_bash_1", "bash", command="echo hello"),
                reasoning="I will call bash to verify the environment.",
            ), DEFAULT_USAGE
        return reply(
            "Final report: implemented and tested successfully.",
            reasoning="The shell output confirms the implementation works.",
        ), DEFAULT_USAGE


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
        if first_content != fusion_prompts.SIDEKICK_POLICY:
            content, tool_calls, usage = self._next(self.main_outputs, self.main_idx)
            self.main_idx += 1
        else:
            content, tool_calls, usage = self._next(self.sidekick_outputs, self.sidekick_idx)
            self.sidekick_idx += 1
        return self._message(content, tool_calls), usage


@pytest.fixture(autouse=True)
def fusion_catalog(monkeypatch, tmp_path):
    serve_config._runs.clear()
    use_tool_catalog(monkeypatch, tmp_path)
    yield
    serve_config._runs.clear()


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


def _stream_reasoning(response):
    return [
        json.loads(line[6:])["choices"][0]["delta"]["reasoning"]
        for line in response.text.splitlines()
        if line.startswith("data: {") and "reasoning" in json.loads(line[6:])["choices"][0]["delta"]
    ]


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
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

    response = client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [
                {
                    "tool_call_id": "sidekick:call_bash_1",
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
    assert fake_worker.main_calls == 2


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
        "tool_results": [{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}],
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
        tool_results=[{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}],
        request_id="req-1",
        coordinator=coordinator,
    )
    assert event["status"] == "completed"
    assert "Final report" in (event["report"] or "")


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

    assert (tmp_path / f"{run_id}.pkl").exists()

    response = file_client.post(
        f"/v1/fusion/follow_up/{run_id}",
        headers=_headers(),
        json={
            "request_id": uuid.uuid4().hex,
            "tool_results": [{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}],
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


def test_fusion_fitting_keeps_latest_prompt_and_prunes_old_history(monkeypatch):
    """Over-budget history is dropped oldest-first; the latest prompt survives.

    Model-driven full-history compaction was removed (it re-fired on every
    call and swallowed the live review/follow-up contract), so fitting must be
    purely local: prune tool bodies, then drop old groups newest-first.
    """
    coordinator = fusion.FusionCoordinator()
    pad = "x" * 320
    messages = [{"role": "system", "content": "system prompt"}]
    for index in range(6):
        messages.append({"role": "user", "content": f"{index}-{pad}"})
        messages.append({"role": "assistant", "content": f"a{index}-{pad}"})
    messages.append({"role": "user", "content": "latest"})
    fitted = coordinator._fit_messages("gpt-5_6-sol", messages, None, 400)
    assert fitted[0]["content"] == "system prompt"
    assert fitted[-1]["content"] == "latest"
    assert len(fitted) < len(messages)
    assert all("0-" not in str(msg.get("content", "")) for msg in fitted[1:])


def test_fusion_fitting_never_drops_only_latest_group(monkeypatch):
    """Even a single oversized group is kept (pruned), never dropped."""
    coordinator = fusion.FusionCoordinator()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "plan the work"},
    ]
    fitted = coordinator._fit_messages("gpt-5_6-sol", messages, None, 1)
    assert fitted[-1]["content"] == "plan the work"
    assert fitted[0]["content"] == "system prompt"


def test_fusion_token_sum_counts_replay_metadata():
    coordinator = fusion.FusionCoordinator()
    base = {
        "role": "assistant",
        "content": "ok",
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
        ],
    }
    replay = {
        **base,
        "reasoning": "r" * 40_000,
        "reasoning_details": [{"text": "d" * 40_000}],
        "_anthropic_content": [{"type": "thinking", "thinking": "a" * 40_000}],
    }
    assert coordinator._token_sum([base]) < coordinator._token_sum([replay]) // 100


def test_fusion_token_sum_counts_tool_schemas():
    coordinator = fusion.FusionCoordinator()
    messages = [{"role": "user", "content": "hello"}]
    big_tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{index}",
                "description": "d" * 4000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for index in range(10)
    ]
    without = coordinator._token_sum(messages)
    with_tools = coordinator._token_sum(messages, big_tools)
    assert with_tools > without * 5


def test_fusion_fitting_counts_tool_schema_overhead(monkeypatch):
    coordinator = fusion.FusionCoordinator()
    big_tools = [
        {
            "type": "function",
            "function": {
                "name": "tool",
                "description": "d" * 200_000,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "go"},
    ]
    monkeypatch.setattr(coordinator, "_tool_tokens", lambda _tools: 50_000)
    fitted = coordinator._fit_messages("gpt-5_6-sol", messages, big_tools, 10_000)
    assert fitted[-1]["content"] == "go"


def test_fusion_call_worker_persists_fitted_history(monkeypatch):
    """Trimmed (pruned) history is written back so old tool bodies are not
    re-sent on every later provider call."""
    coordinator = fusion.FusionCoordinator()
    coordinator.context_window = 5000
    coordinator.max_output_tokens = 256
    big = "x" * 30_000
    messages = [{"role": "system", "content": "system prompt"}]
    for index in range(6):
        messages.append({"role": "user", "content": f"{index}-{big}"})
        messages.append({"role": "assistant", "content": f"a{index}"})
    messages.append({"role": "user", "content": "latest"})

    sent_sizes: list[int] = []

    def recording(spec, sent, max_tokens, temperature, tools=None, timeout_s=None):
        sent_sizes.append(len(sent))
        return {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(providers, "_provider_response", recording)
    coordinator._call_worker("gpt-5_6-sol", messages, None)
    assert serve_config._history_context.fusion_compacted is True
    assert len(messages) < 14
    coordinator._call_worker("gpt-5_6-sol", messages, None)
    assert len(sent_sizes) == 2
    assert sent_sizes[1] == len(messages)


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
    default_limits = fusion_limits.ModelLimits(200000, 128000)
    monkeypatch.setattr(
        fusion_limits, "resolve_model_limits", lambda slot: default_limits
    )
    coordinator = fusion.FusionCoordinator()
    assert coordinator._output_tokens_for("gpt-5_6-luna") == 128000
    assert coordinator.max_output_tokens is None


def test_fusion_chat_initial_call_returns_tool_calls(client, fake_worker):
    response = client.post(
        "/v1/chat/completions",
        headers=_headers(),
        json={
            "model": "mantis/fusion",
            "messages": [{"role": "user", "content": "write a hello world script"}],
            "tools": [BASH_TOOL],
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
    assert len(tool_call_id) <= 40
    assert re.fullmatch(r"[A-Za-z0-9_-]+", tool_call_id)
    run_id, original_id = api._decode_fusion_tool_call_id(tool_call_id)
    assert run_id == response.headers["x-mantis-run-id"]
    assert original_id == "sidekick:call_bash_1"
    assert ":" not in tool_call_id
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
    assert fake_worker.main_calls == 2


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
    headers["X-Mantis-Trace-Scope"] = "sidekick"
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
    assert "Fusion · Planning" not in message["reasoning"]


def test_generic_sse_dedupes_reasoning_text_and_details():
    body = {
        "id": "completion-1",
        "created": 1,
        "model": "test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "done",
                    "reasoning": "shared summary",
                    "reasoning_details": [
                        {
                            "type": "reasoning.summary",
                            "summary": [
                                {"type": "summary_text", "text": "shared summary"},
                                {"type": "summary_text", "text": "new detail"},
                            ],
                        }
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    chunks = [
        json.loads(chunk.decode()[6:])
        for chunk in api._sse(body, include_usage=False)
        if chunk.startswith(b"data: {")
    ]
    deltas = [chunk["choices"][0]["delta"] for chunk in chunks]
    reasoning = [delta["reasoning"] for delta in deltas if "reasoning" in delta]
    details = [delta["reasoning_details"] for delta in deltas if "reasoning_details" in delta]

    assert reasoning == ["shared summary"]
    assert details[0][0]["summary"] == [{"type": "summary_text", "text": "new detail"}]


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
            "tool_results": [{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"

    run = fusion.get_run(run_id)
    assert isinstance(run, fusion.FusionRun)
    assistant_msgs = [m for m in run.sidekick_messages if m.get("role") == "assistant"]
    assert any("reasoning" in m or "reasoning_details" in m for m in assistant_msgs)


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
    assert providers._usage_cost(usage) is None


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
            "tool_results": [{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= body.keys()

    status = client.get(f"/v1/fusion/runs/{run_id}", headers=_headers()).json()
    assert {"usage_models", "cost", "follow_up_count", "follow_up_capped"} <= status.keys()


def test_orchestration_cursor_emits_fresh_third_turn_reasoning():
    run = fusion.FusionRun("reasoning-turns", "first")
    run.sidekick_messages.append(
        {"role": "assistant", "content": "", "reasoning": "first reasoning"}
    )
    first = fusion._orchestration_trace(run, trace_scope="orchestration")
    run.sidekick_messages.append(
        {"role": "assistant", "content": "", "reasoning": "fresh third reasoning"}
    )
    second = fusion._orchestration_trace(run, trace_scope="orchestration")

    assert "first reasoning" in first
    assert "fresh third reasoning" in second
    assert "first reasoning" not in second


def test_orchestration_trace_sidekick_delta():
    """The sidekick-scope trace emits only reasoning added since the previous
    response; the same trace is never replayed to the client."""
    run = fusion.FusionRun("delta", "goal")
    run.sidekick_messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "", "reasoning": "first thought"},
    ]
    assert fusion._orchestration_trace(run, trace_scope="sidekick") == "first thought"
    assert fusion._orchestration_trace(run, trace_scope="sidekick") == ""
    run.sidekick_messages.append(
        {"role": "assistant", "content": "", "reasoning": "second thought"}
    )
    assert fusion._orchestration_trace(run, trace_scope="sidekick") == "second thought"


def test_orchestration_trace_without_preamble_is_idempotent():
    run = fusion.FusionRun("empty-trace", "goal")
    assert fusion._orchestration_trace(run, include_reasoning=True) == ""
    assert fusion._orchestration_trace(run, include_reasoning=True) == ""


def test_generic_reasoning_detail_filter_handles_unique_and_non_text_items():
    details = api._dedupe_reasoning_details(
        "same",
        [
            {"type": "reasoning.summary", "summary": [{"text": "same"}]},
            {"type": "reasoning.summary", "summary": [{"text": "new"}]},
            {"type": "reasoning.encrypted", "data": "opaque"},
            "not a mapping",
        ],
    )
    assert details[0] == {"type": "reasoning.summary"}
    assert details[1]["summary"] == [{"text": "new"}]
    assert details[2]["data"] == "opaque"
    assert details[3] == "not a mapping"


def test_extract_reasoning_trace_dedupes():
    """A summary mirrored under multiple reasoning keys appears once."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning": "shared summary",
            "reasoning_details": [
                {
                    "type": "reasoning.summary",
                    "summary": [{"type": "summary_text", "text": "shared summary"}],
                }
            ],
        }
    ]
    trace = fusion._extract_reasoning_trace(messages)
    assert trace.count("shared summary") == 1


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


def test_fusion_filter_tools_intersects_client_tools_with_enabled():
    run = fusion.FusionRun(
        "tool-auth",
        "goal",
        tools=[
            {"type": "function", "function": {"name": "bash"}},
            {"type": "function", "function": {"name": "read_file"}},
            {"type": "function", "function": {"name": "unknown"}},
        ],
        tool_options=fusion.FusionToolOptions(enabled=["files"]),
    )
    fusion.fusion_lead.prepare_run(run)
    filtered = fusion_background.sidekick_tools(run.handoffs, run) or []
    names = {t["function"]["name"] for t in filtered}
    assert names == {"read_file"}


def test_fusion_filter_tools_generates_server_schemas_for_enabled():
    run = fusion.FusionRun(
        "tool-auth-server",
        "goal",
        tool_options=fusion.FusionToolOptions(enabled=["files"], server_execution=True),
    )
    fusion.fusion_lead.prepare_run(run)
    filtered = fusion_background.sidekick_tools(run.handoffs, run) or []
    names = {t["function"]["name"] for t in filtered}
    assert names == {
        "list_files",
        "read_file",
        "search_files",
        "write_file",
        "edit_file",
    }


def test_fusion_chat_streams_per_request_usage(client, fake_worker, monkeypatch):
    monkeypatch.setattr(api, "_uses_live_fusion_stream", lambda request: False)
    """Streaming usage chunks must also be per-request, not run-lifetime."""
    request_messages = [{"role": "user", "content": "write a hello world script"}]
    payload = {
        "model": "mantis/fusion",
        "messages": request_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    response = client.post("/v1/chat/completions", headers=_headers(), json=payload)
    assert response.status_code == 200
    usage_chunks: list[dict[str, Any]] = []
    completion_text = ""
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunk = json.loads(line[len("data: ") :])
        for choice in chunk.get("choices") or []:
            completion_text += str((choice.get("delta") or {}).get("content") or "")
        if chunk.get("usage"):
            usage_chunks.append(chunk["usage"])
    assert usage_chunks, response.text
    expected = utils._request_usage(request_messages, completion_text)
    assert usage_chunks[0]["prompt_tokens"] == expected["prompt_tokens"]
    run = fusion.get_run(response.headers["X-Mantis-Run-Id"])
    assert usage_chunks[0] != run.usage


def test_fusion_chat_reports_per_request_usage(client, fake_worker):
    """Public chat usage must be this request's context only.

    Run-lifetime usage (every internal planner/sidekick/review call) would
    make context-tracking clients compact after nearly every turn.
    """
    headers = _headers()
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
    body = {
        "model": "mantis/fusion",
        "messages": [{"role": "user", "content": "write a hello world script"}],
        "tools": tools,
    }
    first = client.post("/v1/chat/completions", headers=headers, json=body)
    assert first.status_code == 200
    run_id = first.headers["X-Mantis-Run-Id"]
    first_usage = first.json()["usage"]
    run = fusion.get_run(run_id)
    assert first_usage != run.usage

    second_messages = [
        {"role": "user", "content": "write a hello world script"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "now lint it"},
    ]
    second = client.post(
        "/v1/chat/completions",
        headers={**headers, "X-Mantis-Run-Id": run_id},
        json={
            **body,
            "messages": second_messages,
        },
    )
    assert second.status_code == 200
    second_body = second.json()
    second_usage = second_body["usage"]
    response_message = second_body["choices"][0]["message"]
    completion_text = str(response_message.get("content") or "")
    completion_text += "".join(
        f"{call['function'].get('name', '')}:{call['function'].get('arguments', '')}"
        for call in response_message.get("tool_calls") or []
    )
    assert second_usage == utils._request_usage(second_messages, completion_text)


def test_fusion_budget_guard_pickle_roundtrip():
    guard = fusion.FusionBudgetGuard(max_turns=3, max_tokens=100, timeout_ms=1000)
    guard.consume_turn()
    restored = pickle.loads(pickle.dumps(guard))  # noqa: S301
    assert restored.turns == 1
    restored.consume_turn()
    assert restored.turns == 2


def test_tool_exec_file_and_shell_tools(tmp_path):
    workspace = tool_exec.workspace_for("t", root=tmp_path)
    assert (
        tool_exec.execute("write_file", {"path": "a.py", "content": "print('hi')"}, workspace)
        == "ok"
    )
    assert tool_exec.execute("read_file", {"path": "a.py"}, workspace) == "print('hi')"
    assert "a.py" in tool_exec.execute("list_files", {}, workspace)
    assert "print" in tool_exec.execute("search_files", {"query": "print"}, workspace)
    assert "hello" in tool_exec.execute("bash", {"command": "echo hello"}, workspace)
    assert "2" in tool_exec.execute("run_code", {"code": "print(1 + 1)"}, workspace)
    tool_exec.execute("edit_file", {"path": "a.py", "old_text": "hi", "new_text": "bye"}, workspace)
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


def test_provider_stream_assembly_emits_output_and_reasoning_deltas(monkeypatch):
    pass


def test_fusion_budget_remaining_timeout():
    guard = fusion.FusionBudgetGuard(timeout_ms=1000)
    remaining = guard.remaining_timeout_s()
    assert remaining is not None and 0 < remaining <= 1
    guard._started -= 2
    with pytest.raises(RuntimeError, match="wall-time"):
        guard.remaining_timeout_s()


def test_fusion_router_pools_are_cheapest_first_for_every_role():
    """A higher pool index must always be the stronger, costlier slot."""
    config = fusion.FusionRoutingConfig(
        main=["main-cheap", "main-strong"],
        sidekick=["side-cheap", "side-strong"],
    )
    main = fusion.FusionRouter(config, "main")
    sidekick = fusion.FusionRouter(config, "sidekick")
    assert main.select(escalation_count=0) == "main-cheap"
    assert main.select(escalation_count=1) == "main-strong"
    assert sidekick.select(escalation_count=0) == "side-cheap"
    assert sidekick.select(escalation_count=1) == "side-strong"


def test_fusion_role_reasoning_fields_annotate_resolved_specs():
    """main_reasoning/sidekick_reasoning become the spec's |effort suffix."""
    config = fusion.FusionRoutingConfig(
        main="openrouter/openai/gpt-5.6-sol",
        main_reasoning="low",
        sidekick="openrouter/openai/gpt-5.6-luna",
        sidekick_reasoning="high",
    )
    assert fusion.FusionRouter(config, "main").select() == "openrouter/openai/gpt-5.6-sol|low"
    assert fusion.FusionRouter(config, "sidekick").select() == "openrouter/openai/gpt-5.6-luna|high"


def test_fusion_role_reasoning_is_only_a_default():
    """An explicit |suffix or the virtual route ignores the role field."""
    config = fusion.FusionRoutingConfig(
        main=["openrouter/openai/gpt-5.6-sol|xhigh", "openrouter/openai/gpt-5.6-luna"],
        main_reasoning="low",
        sidekick="mantis/base",
        sidekick_reasoning="low",
    )
    main = fusion.FusionRouter(config, "main")
    assert main.select(escalation_count=0) == "openrouter/openai/gpt-5.6-sol|xhigh"
    assert main.select(escalation_count=1) == "openrouter/openai/gpt-5.6-luna|low"
    assert fusion.FusionRouter(config, "sidekick").select() == "mantis/base"


def test_fusion_role_reasoning_keeps_base_target_effort():
    """A base: route already carries its target's effort; the field yields."""
    catalog = Path(__file__).resolve().parent.parent / "config" / "catalog.toml"
    config = fusion.FusionRoutingConfig(
        main="base:capable", main_reasoning="low", sidekick="gpt-5_6-luna"
    )
    router = fusion.FusionRouter.from_config(config, "main", catalog=catalog)
    assert router.select() == "openrouter/~anthropic/claude-opus-latest|high"


def test_record_selected_slot_tracks_promotions():
    """Trace and learning records must name the model that actually ran."""
    run = fusion.FusionRun("promote", "goal")
    run.main_slot = "main-cheap"
    run.sidekick_slot = "side-cheap"
    run.slot_models = ["main-cheap", "side-cheap"]

    run._record_selected_slot("main", "main-strong")
    assert run.main_slot == "main-strong"
    assert "main-strong" in run.slot_models

    run._record_selected_slot("sidekick", "lane-model", primary=False)
    assert run.sidekick_slot == "side-cheap"
    assert "lane-model" in run.slot_models

    run._record_selected_slot("main", "main-strong")
    assert run.slot_models.count("main-strong") == 1


def test_base_route_remains_virtual_while_explicit_capable_target_resolves():
    """Switchyard, not FusionRouter, resolves the dynamic mantis/base route."""
    catalog = Path(__file__).resolve().parent.parent / "config" / "catalog.toml"
    config = fusion.FusionRoutingConfig(main="mantis/base", sidekick="gpt-5_6-luna")
    router = fusion.FusionRouter.from_config(config, "main", catalog=catalog)
    spec = router.select(escalation_count=0)
    assert spec == "mantis/base"
    assert router.strongest() == "mantis/base"

    coordinator = fusion.FusionCoordinator()
    if coordinator.context_window:
        assert coordinator._context_window_for(spec) == coordinator.context_window
    explicit = fusion.FusionRouter.from_config(
        fusion.FusionRoutingConfig(main="base:capable", sidekick="gpt-5_6-luna"),
        "main",
        catalog=catalog,
    )
    assert explicit.select(escalation_count=0) == "openrouter/~anthropic/claude-opus-latest|high"


def test_virtual_worker_dispatches_complete_tool_trajectory_to_switchyard(monkeypatch):
    monkeypatch.setenv("MANTIS_CACHE_BREAKPOINTS", "0")
    trajectory = {
        "messages": [
            {"role": "user", "content": "inspect"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("call_1", "read_file", {"path": "README.md"})],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    seen: list[tuple[dict, str]] = []

    async def handler(request):
        seen.append(
            (
                json.loads(request.content.decode()),
                request.headers[base_proxy.SWITCHYARD_SESSION_HEADER],
            )
        )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [_tool_call("next", "read_file", {"path": "x"})],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 5},
            },
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/chosen"},
        )

    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    coordinator = fusion.FusionCoordinator()
    with fusion._fusion_route_session("run-1:main"):
        first, _ = coordinator._call_worker(
            "mantis/base", trajectory["messages"], trajectory["tools"]
        )
        coordinator._call_worker("mantis/base", trajectory["messages"], trajectory["tools"])
    with fusion._fusion_route_session("run-1:sidekick"):
        coordinator._call_worker("mantis/base", trajectory["messages"], trajectory["tools"])

    assert [session for _, session in seen] == [
        "run-1:main",
        "run-1:main",
        "run-1:sidekick",
    ]
    assert seen[0][0]["messages"] == trajectory["messages"]
    assert seen[0][0]["tools"] == trajectory["tools"]
    assert "max_tokens" not in seen[0][0]
    assert first["tool_calls"] == [_tool_call("next", "read_file", {"path": "x"})]
    assert "_mantis_model" not in first


def test_virtual_worker_does_not_fit_to_a_fictional_route_context(monkeypatch):
    monkeypatch.setattr(
        fusion.FusionCoordinator,
        "_fit_messages",
        lambda *_args, **_kwargs: pytest.fail(
            "virtual route must choose its target before fitting"
        ),
    )

    async def handler(_request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    fusion.FusionCoordinator()._call_worker(
        "mantis/base", [{"role": "user", "content": "hi"}], None
    )


def test_virtual_worker_sanitizes_replayed_messages_before_every_request(monkeypatch):
    requests: list[dict[str, Any]] = []

    async def handler(request):
        requests.append(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "done",
                            "reasoning_details": [
                                {"type": "reasoning.encrypted", "data": "secret"},
                                {"type": "reasoning", "text": "portable"},
                            ],
                            "_provider_private": "must not replay",
                        }
                    }
                ]
            },
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/chosen"},
        )

    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    messages = [{"role": "user", "content": "first"}]
    coordinator = fusion.FusionCoordinator()
    first, _ = coordinator._call_worker("mantis/base", messages, None)
    messages.extend([first, {"role": "user", "content": "second"}])
    coordinator._call_worker("mantis/base", messages, None)

    replayed = requests[1]["messages"][1]
    assert "_mantis_model" not in first
    assert all(not key.startswith("_") for key in replayed)
    assert replayed["reasoning_details"] == [{"type": "reasoning", "text": "portable"}]


def test_virtual_worker_http_timeout_uses_remaining_fusion_budget(monkeypatch):
    from types import SimpleNamespace

    effective_timeouts: list[dict[str, float | None]] = []

    async def handler(request):
        effective_timeouts.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    monkeypatch.setattr(
        base_proxy,
        "routed_async_client",
        lambda: httpx.AsyncClient(timeout=99, transport=httpx.MockTransport(handler)),
    )
    budget = SimpleNamespace(remaining_timeout_s=lambda: 1.25)
    monkeypatch.setattr(
        serve_config._history_context,
        "active_run",
        SimpleNamespace(budget=budget, add_usage=lambda *a, **k: None),
        raising=False,
    )

    fusion.FusionCoordinator()._call_worker(
        "mantis/base", [{"role": "user", "content": "hi"}], None
    )

    assert effective_timeouts == [{"connect": 1.25, "read": 1.25, "write": 1.25, "pool": 1.25}]


def test_concrete_worker_still_uses_provider_path(monkeypatch):
    calls: list[str] = []

    def provider(slot, *_args, **_kwargs):
        calls.append(slot)
        return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    monkeypatch.setattr(fusion.providers, "_provider_response", provider)
    coordinator = fusion.FusionCoordinator()
    message, _ = coordinator._call_worker("gpt-5_6-sol", [{"role": "user", "content": "hi"}], None)
    assert calls == ["gpt-5_6-sol"]
    assert message["content"] == "ok"


def test_base_route_lead_and_sidekick_have_distinct_provenance():
    """Reasoning must not cross between the lead and sidekick models."""
    coordinator = fusion.FusionCoordinator()
    run = fusion.FusionRun("provenance", "goal")
    main = coordinator._upstream_model_for(run.main_router.select())
    side = coordinator._upstream_model_for(run.sidekick_router.select())
    assert main != side


def test_shipped_slots_get_the_full_endpoint_budget(monkeypatch):
    """The harness owns compaction, so Fusion must not trim early."""
    coordinator = fusion.FusionCoordinator()
    for slot in (coordinator.main_slot, coordinator.sidekick_slot):
        limits = fusion_limits.resolve_model_limits(slot)
        expected = limits.context_window
        if coordinator.context_window:
            expected = min(expected, coordinator.context_window)
        assert coordinator._context_window_for(slot) == expected


def test_context_trim_emits_an_observable_event(monkeypatch):
    """Fusion-side trimming is a safety net and must never be silent."""
    coordinator = fusion.FusionCoordinator()
    coordinator.context_window = 4000
    coordinator.max_output_tokens = 500
    monkeypatch.setattr(
        fusion.providers,
        "_provider_response",
        lambda *a, **k: {"choices": [{"message": {"content": "ok"}}], "usage": {}},
    )
    events: list[dict[str, Any]] = []
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "g"}]
    for i in range(400):
        messages.append({"role": "assistant", "content": f"turn {i} " + "x" * 400})
        messages.append({"role": "user", "content": f"next {i}"})

    with fusion.providers.progress_events(events.append):
        coordinator._call_worker("gpt-5_6-sol", messages, None)

    trimmed = [e for e in events if e.get("type") == "fusion_context_trimmed"]
    assert trimmed, "trimming must emit fusion_context_trimmed"
    assert trimmed[0]["tokens_dropped"] > 0
    assert trimmed[0]["tokens_after"] <= trimmed[0]["input_budget"]
    assert trimmed[0]["model"] == "gpt-5_6-sol"


def test_context_window_clamps_to_the_selected_slot(monkeypatch):
    """A pool may mix window sizes; each turn is sized for the slot it uses."""
    coordinator = fusion.FusionCoordinator()
    coordinator.context_window = 262144
    windows = {"big": 500000, "mid": 128000, "small": 65536, "undeclared": 262144}

    monkeypatch.setattr(
        fusion_limits,
        "resolve_model_limits",
        lambda slot: fusion_limits.ModelLimits(windows.get(slot, 262144), 4096),
    )

    assert coordinator._context_window_for("big") == 262144
    assert coordinator._context_window_for("mid") == 128000
    assert coordinator._context_window_for("small") == 65536
    assert coordinator._context_window_for("undeclared") == 262144


def test_call_worker_sizes_input_budget_for_the_promoted_slot(monkeypatch):
    """Promoting into a smaller-window model must shrink the transcript."""
    coordinator = fusion.FusionCoordinator()
    coordinator.context_window = None
    coordinator.max_output_tokens = None
    seen: list[int] = []
    windows = {"mid": 128000, "small": 65536}

    monkeypatch.setattr(
        fusion_limits,
        "resolve_model_limits",
        lambda slot: fusion_limits.ModelLimits(windows.get(slot, 128000), 4096),
    )

    def fake_fit(slot, messages, tools, input_budget):
        seen.append(input_budget)
        return messages

    monkeypatch.setattr(coordinator, "_fit_messages", fake_fit)
    monkeypatch.setattr(
        fusion.providers,
        "_provider_response",
        lambda *a, **k: {"choices": [{"message": {"content": "ok"}}], "usage": {}},
    )

    coordinator._call_worker("mid", [{"role": "user", "content": "hi"}], None)
    coordinator._call_worker("small", [{"role": "user", "content": "hi"}], None)
    assert seen == [128000 - 4096, 65536 - 4096]


def test_fusion_router_strongest_is_last_pool_entry():
    config = fusion.FusionRoutingConfig(
        main=["main-cheap", "main-strong"],
        sidekick="side-only",
    )
    main = fusion.FusionRouter(config, "main")
    sidekick = fusion.FusionRouter(config, "sidekick")
    assert main.select(escalation_count=0) == "main-cheap"
    assert main.strongest() == "main-strong"
    assert sidekick.strongest() == "side-only"


def test_refresh_fusion_run_tools_replaces_stale_harness_schemas():
    """A harness that compacts or loads a skill resumes with a new tool set."""
    run = fusion.FusionRun(
        "refresh",
        "goal",
        tools=[{"type": "function", "function": {"name": "old_tool"}}],
    )
    assert [t["function"]["name"] for t in run.tools] == ["old_tool"]

    fusion.refresh_fusion_run_tools(
        run,
        [
            {"type": "function", "function": {"name": "zebra"}},
            {"type": "function", "function": {"name": "alpha"}},
        ],
    )
    assert [t["function"]["name"] for t in run.tools] == ["alpha", "zebra"]

    fusion.refresh_fusion_run_tools(run, [])
    assert run.tools == []


def test_fusion_router_reroutes_only_at_compaction():
    config = fusion.FusionRoutingConfig(
        main=["main-cheap", "main-strong"],
        sidekick=["side-cheap", "side-strong"],
    )
    main = fusion.FusionRouter(config, "main")
    sidekick = fusion.FusionRouter(config, "sidekick")
    assert main.select_at_compaction(0.5, "main-strong") == "main-cheap"
    assert main.select_at_compaction(0.9, "main-strong") == "main-strong"
    assert sidekick.select_at_compaction(0.5, "side-cheap", 0) == "side-cheap"
    assert sidekick.select_at_compaction(0.5, "side-cheap", 1) == "side-strong"


def test_native_protocol_stream_deltas_emit_progress(monkeypatch):
    pass


def test_select_at_compaction_stays_when_cache_warm():
    """When cache is warm and no failures, keep the current model."""
    config = fusion.FusionRoutingConfig(main=["cheap", "strong"], sidekick="side")
    router = fusion.FusionRouter(config, "main")
    selected = router.select_at_compaction(0.5, "strong", failure_count=0, cache_warm=True)
    assert selected == "strong"


def test_select_at_compaction_switches_when_cache_warm_but_failing():
    """Even with warm cache, failures should force a promotion."""
    config = fusion.FusionRoutingConfig(main=["cheap", "strong", "strongest"], sidekick="side")
    router = fusion.FusionRouter(config, "main")
    selected = router.select_at_compaction(0.5, "strong", failure_count=1, cache_warm=True)
    assert selected == "strongest"


def test_select_at_compaction_promotes_cheap_main_on_high_complexity():
    config = fusion.FusionRoutingConfig(main=["cheap", "strong"], sidekick="side")
    router = fusion.FusionRouter(config, "main")
    assert router.select_at_compaction(0.95, "cheap") == "strong"
    assert router.select_at_compaction(0.95, "strong") == "strong"


def test_select_at_compaction_never_demotes_sidekick():
    config = fusion.FusionRoutingConfig(
        main="main", sidekick=["side-cheap", "side-mid", "side-strong"]
    )
    router = fusion.FusionRouter(config, "sidekick")
    assert router.select_at_compaction(0.9, "side-strong") == "side-strong"
    assert router.select_at_compaction(0.2, "side-mid") == "side-mid"


def test_select_at_compaction_promotes_sidekick_past_second_slot_on_failure():
    config = fusion.FusionRoutingConfig(
        main="main", sidekick=["side-cheap", "side-mid", "side-strong"]
    )
    router = fusion.FusionRouter(config, "sidekick")
    selected = router.select_at_compaction(0.5, "side-mid", failure_count=1)
    assert selected == "side-strong"


def test_select_at_compaction_cold_cache_allows_reroute():
    """With cold cache, normal complexity-based rerouting applies."""
    config = fusion.FusionRoutingConfig(main=["cheap", "strong"], sidekick="side")
    router = fusion.FusionRouter(config, "main")
    selected = router.select_at_compaction(0.3, "strong", failure_count=0, cache_warm=False)
    assert selected == "cheap"


def test_base_route_rejects_unknown_role():
    catalog = Path(__file__).resolve().parent.parent / "config" / "catalog.toml"
    config = fusion.FusionRoutingConfig(main="base:typo", sidekick="gpt-5_6-luna")
    router = fusion.FusionRouter.from_config(config, "main", catalog=catalog)
    with pytest.raises(ValueError, match="unknown base route role"):
        router.select()


def test_fusion_pool_ordering_rejects_known_price_regression(monkeypatch):
    prices = {
        "cheap-model": (1.0, 2.0),
        "strong-model": (5.0, 10.0),
    }
    monkeypatch.setattr(providers, "_price_map", lambda: prices)
    bad = fusion.FusionRoutingConfig(main=["strong-model", "cheap-model"], sidekick="side")
    with pytest.raises(ValueError, match="not cheapest-first"):
        fusion_config._validate_pool_ordering(bad)
    good = fusion.FusionRoutingConfig(main=["cheap-model", "strong-model"], sidekick="side")
    fusion_config._validate_pool_ordering(good)


def test_fusion_pool_ordering_skips_unpriced_slots(monkeypatch):
    monkeypatch.setattr(providers, "_price_map", lambda: {"cheap-model": (1.0, 2.0)})
    config = fusion.FusionRoutingConfig(main=["cheap-model", "mystery-model"], sidekick="side")
    fusion_config._validate_pool_ordering(config)


def test_trailing_error_streak_counts_tail_failures_only():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_call_id": "b", "content": "bad", "is_error": True},
        {"role": "tool", "tool_call_id": "c", "content": "bad", "is_error": True},
    ]
    assert fusion._trailing_error_streak(messages) == 2
    assert fusion._trailing_error_streak(messages[:2]) == 0
    assert fusion._trailing_error_streak([]) == 0


def test_fusion_savings_prices_sidekick_tokens_at_main_rate(monkeypatch):
    prices = {
        "main-model": (10.0, 20.0),
        "side-model": (1.0, 2.0),
    }
    monkeypatch.setattr(providers, "_price_map", lambda: prices)
    monkeypatch.setattr(providers, "_cache_read_prices", lambda: {})
    run = fusion.FusionRun("savings", "goal")
    run.main_slot = "main-model"
    run._add_role_usage("sidekick", "side-model", {"prompt_tokens": 100, "completion_tokens": 50})
    run._add_role_usage("main", "main-model", {"prompt_tokens": 999, "completion_tokens": 999})

    savings = run._fusion_savings_usd()

    assert savings is not None
    assert savings == 100 * 10.0 + 50 * 20.0 - (100 * 1.0 + 50 * 2.0)


def test_fusion_savings_none_without_sidekick_usage(monkeypatch):
    monkeypatch.setattr(providers, "_price_map", lambda: {"main-model": (10.0, 20.0)})
    run = fusion.FusionRun("no-savings", "goal")
    run.main_slot = "main-model"
    assert run._fusion_savings_usd() is None


def test_fusion_savings_none_when_prices_unknown(monkeypatch):
    monkeypatch.setattr(providers, "_price_map", lambda: {"main-model": (10.0, 20.0)})
    run = fusion.FusionRun("unknown-prices", "goal")
    run.main_slot = "main-model"
    run._add_role_usage("sidekick", "mystery", {"prompt_tokens": 10, "completion_tokens": 5})
    assert run._fusion_savings_usd() is None


def test_ok_event_reports_fusion_savings(monkeypatch):
    prices = {
        "main-model": (10.0, 20.0),
        "side-model": (1.0, 2.0),
    }
    monkeypatch.setattr(providers, "_price_map", lambda: prices)
    monkeypatch.setattr(providers, "_cache_read_prices", lambda: {})
    monkeypatch.setattr(providers, "_usage_cost", lambda usage_models: 0.0)
    run = fusion.FusionRun("savings-event", "goal")
    run.main_slot = "main-model"
    run._add_role_usage("sidekick", "side-model", {"prompt_tokens": 100, "completion_tokens": 50})

    event = run._ok_event(None)

    assert event["fusion_savings_usd"] == 100 * 10.0 + 50 * 20.0 - (100 * 1.0 + 50 * 2.0)


def test_has_cache_hits_true():
    usage = {"prompt_tokens_details": {"cached_tokens": 1024}}
    assert fusion.FusionRun._has_cache_hits(usage) is True


def test_has_cache_hits_false_zero():
    usage = {"prompt_tokens_details": {"cached_tokens": 0}}
    assert fusion.FusionRun._has_cache_hits(usage) is False


def test_has_cache_hits_missing():
    assert fusion.FusionRun._has_cache_hits({}) is False
    assert fusion.FusionRun._has_cache_hits({"prompt_tokens_details": {}}) is False


def test_fusion_message_resume_answers_without_sidekick(monkeypatch):
    worker = SequenceWorker(
        [
            ("", [call("delegate", "sidekick", message="implement and test")], DEFAULT_USAGE),
            ("Done.", None, DEFAULT_USAGE),
            ("resumed clarification", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("resume-msg", "goal")
    assert run.advance()["status"] == "completed"
    sidekick_before = worker.sidekick_idx
    event = run.advance(message="what did you just do?")
    assert event["status"] == "completed"
    assert event["report"] == "resumed clarification"
    assert worker.sidekick_idx == sidekick_before


def test_fusion_chat_completed_run_header_starts_new_run(client, monkeypatch):
    worker = SequenceWorker(
        [
            ("first", None, DEFAULT_USAGE),
            ("second", None, DEFAULT_USAGE),
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
    assert second.headers.get("X-Mantis-Run-Id") != run_id
    assert second.json()["choices"][0]["message"]["content"] == "second"
    assert worker.sidekick_idx == 0


def test_fusion_chat_resume_via_encoded_tool_call_id(client, monkeypatch):
    worker = SequenceWorker(
        [
            ("", [call("delegate", "sidekick", message="implement and test")], DEFAULT_USAGE),
            ("Done.", None, DEFAULT_USAGE),
            ("fresh ordinary turn", None, DEFAULT_USAGE),
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
    assert second.headers.get("X-Mantis-Run-Id") == run_id
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
    assert third.headers.get("X-Mantis-Run-Id") != run_id
    assert third.json()["choices"][0]["message"]["content"] == "fresh ordinary turn"


def test_fusion_chat_three_turns_return_fresh_report(client, monkeypatch):
    worker = SequenceWorker(
        [
            ("TURN 1 REPORT", None, DEFAULT_USAGE),
            ("TURN 2 REPORT", None, DEFAULT_USAGE),
            ("TURN 3 FRESH REPORT", None, DEFAULT_USAGE),
        ],
        [],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)

    def send(messages, run_id=None):
        headers = _headers()
        if run_id:
            headers["X-Mantis-Run-Id"] = run_id
        return client.post(
            "/v1/chat/completions",
            headers=headers,
            json={"model": "mantis/fusion", "messages": messages},
        )

    first = send([{"role": "user", "content": "turn one"}])
    first_run_id = first.headers["X-Mantis-Run-Id"]
    second = send(
        [
            {"role": "user", "content": "turn one"},
            first.json()["choices"][0]["message"],
            {"role": "user", "content": "turn two"},
        ],
        first_run_id,
    )
    second_run_id = second.headers["X-Mantis-Run-Id"]
    third = send(
        [
            {"role": "user", "content": "turn one"},
            first.json()["choices"][0]["message"],
            {"role": "user", "content": "turn two"},
            second.json()["choices"][0]["message"],
            {"role": "user", "content": "turn three"},
        ],
        second_run_id,
    )

    assert first.status_code == second.status_code == third.status_code == 200
    assert len({first_run_id, second_run_id, third.headers["X-Mantis-Run-Id"]}) == 3
    assert first.json()["choices"][0]["message"]["content"] == "TURN 1 REPORT"
    assert second.json()["choices"][0]["message"]["content"] == "TURN 2 REPORT"
    assert third.json()["choices"][0]["message"]["content"] == "TURN 3 FRESH REPORT"
    assert "TURN 2 REPORT" not in third.json()["choices"][0]["message"]["content"]
    for run_id in (first_run_id, second_run_id, third.headers["X-Mantis-Run-Id"]):
        serve_config._runs.pop(run_id, None)


def test_resumed_run_clears_report_before_new_review(monkeypatch):
    worker = SequenceWorker(
        [
            ("", [call("delegate", "sidekick", message="implement and test")], DEFAULT_USAGE),
            ("FRESH REVIEW REPORT", None, DEFAULT_USAGE),
        ],
        [("FRESH REVIEW REPORT", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.FusionRun("stale-report", "old task")
    run.report = "STALE REPORT"
    run.status = "completed"

    event = run.advance(message="new task")

    assert event["status"] == "completed"
    assert event["report"] == "FRESH REVIEW REPORT"
    assert "STALE REPORT" not in event["report"]


def test_repeated_request_id_returns_cached_event_without_advancing(monkeypatch):
    worker = SequenceWorker(
        [
            ("", [call("delegate", "sidekick", message="implement and test")], DEFAULT_USAGE),
            ("Done.", None, DEFAULT_USAGE),
        ],
        [("FRESH REPORT", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.create_fusion_run("request cache")

    first = fusion.advance_fusion_run(run.run_id, request_id="request-1")
    call_count = len(worker.calls)
    second = fusion.advance_fusion_run(run.run_id, request_id="request-1")

    assert second == first
    assert len(worker.calls) == call_count
    serve_config._runs.pop(run.run_id, None)


def test_fusion_chat_missing_run_id_starts_new_available_run(client, monkeypatch):
    worker = SequenceWorker(
        [("fresh", None, DEFAULT_USAGE)],
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


def test_fusion_writes_learning_record_on_accept(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_LEARNING", "1")
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_LEARNING_INSTANCE", "fusion-test")
    monkeypatch.setattr(fusion, "RUN_STORE", "memory")
    worker = SequenceWorker(
        [
            ("", [call("delegate", "sidekick", message="implement and test")], DEFAULT_USAGE),
            ("Done.", None, DEFAULT_USAGE),
        ],
        [("Done.", None, DEFAULT_USAGE)],
    )
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    run = fusion.create_fusion_run("goal")

    event = fusion.advance_fusion_run(run.run_id)

    assert event["status"] == "completed"
    record = json.loads(runs._learning_path().read_text().splitlines()[-1])
    assert record["mode"] == "fusion"
    assert record["terminated_by"] == "fusion_answer"
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

    def fail(*args):
        raise RuntimeError("worker failed")

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", fail)
    run = fusion.create_fusion_run("goal")
    event = fusion.advance_fusion_run(run.run_id)
    assert event["status"] == "error"
    record = json.loads(runs._learning_path().read_text().splitlines()[-1])
    assert record["mode"] == "fusion"
    assert record["terminated_by"] == "fusion_error"


def test_virtual_route_is_not_rerouted_by_fusion_after_compaction():
    config = fusion.FusionRoutingConfig(main="mantis/base", sidekick="mantis/base")
    run = fusion.FusionRun("virtual-compaction", "goal")
    run.main_router = fusion.FusionRouter(config, "main")
    run.sidekick_router = fusion.FusionRouter(config, "sidekick")
    run.main_slot = "mantis/base"
    run.sidekick_slot = "mantis/base"

    run._reroute_after_compaction("main", "mantis/base")
    run._reroute_after_compaction("sidekick", "mantis/base")

    assert run.main_compaction_slot is None
    assert run.sidekick_compaction_slot is None
    assert not [entry for entry in run._activity if entry["type"] == "fusion_reroute"]


def test_established_main_system_prefix_is_not_mutated_during_dispatch(monkeypatch):
    run = fusion.FusionRun(
        "stable-prefix",
        "goal",
        tool_options=fusion.FusionToolOptions(enabled=["files"]),
    )
    original = dict(run.main_messages[0])
    coordinator = fusion.FusionCoordinator()
    monkeypatch.setattr(
        coordinator,
        "_call_worker",
        lambda *_args: ({"role": "assistant", "content": "done"}, {}),
    )

    run._call_lane(coordinator, "main")

    assert run.main_messages[0] == original


def test_fusion_run_records_compaction_reroute():
    config = fusion.FusionRoutingConfig(
        main=["main-cheap", "main-strong"],
        sidekick=["side-cheap", "side-strong"],
    )
    run = fusion.FusionRun(
        "reroute",
        "goal",
        budget=fusion.FusionRunBudget(max_turns=10),
    )
    run.main_router = fusion.FusionRouter(config, "main")
    run.sidekick_router = fusion.FusionRouter(config, "sidekick")
    run.main_slot = "main-strong"
    run.sidekick_slot = "side-cheap"

    run._reroute_after_compaction("main", "main-strong")
    run.follow_up_count = 1
    run._reroute_after_compaction("sidekick", "side-cheap")

    assert run.main_compaction_slot is None
    assert run.sidekick_compaction_slot == "side-strong"
    assert run._activity[-1]["type"] == "fusion_reroute"


def test_reroute_after_compaction_counts_error_streak_as_failure():
    config = fusion.FusionRoutingConfig(
        main=["main-cheap", "main-strong"],
        sidekick=["side-cheap", "side-strong"],
    )
    run = fusion.FusionRun(
        "streak-reroute",
        "goal",
        budget=fusion.FusionRunBudget(max_turns=10),
    )
    run.main_router = fusion.FusionRouter(config, "main")
    run.sidekick_router = fusion.FusionRouter(config, "sidekick")
    run.main_slot = "main-cheap"
    run.sidekick_slot = "side-cheap"
    run.follow_up_count = 0
    run.sidekick_messages = [
        {"role": "system", "content": fusion_prompts.SIDEKICK_POLICY},
        {"role": "tool", "tool_call_id": "x", "content": "bad", "is_error": True},
    ]

    run._reroute_after_compaction("sidekick", "side-cheap")

    assert run.sidekick_compaction_slot == "side-strong"


def test_fusion_trim_keeps_frozen_prefix_when_budget_allows():
    coordinator = fusion.FusionCoordinator()
    prefix = [
        {"role": "system", "content": fusion_prompts.lead_system_prompt(None)},
        {"role": "user", "content": "client history"},
        {"role": "assistant", "content": "ack"},
    ]
    first = coordinator._trim_messages([*prefix, {"role": "user", "content": "plan now"}], 100_000)
    second = coordinator._trim_messages(
        [
            *prefix,
            {"role": "user", "content": "plan now"},
            {"role": "assistant", "content": "task result"},
        ],
        100_000,
    )
    assert first[:3] == prefix
    assert second[:3] == prefix


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

    message, _usage = worker_with_opaque_metadata(
        "gpt-5_6-sol",
        [{"role": "system", "content": fusion_prompts.lead_system_prompt(None)}],
        None,
    )
    trace = fusion._extract_reasoning_trace([message])
    assert "Safe summary." in trace
    assert "Safe textual thinking." in trace
    for value in opaque_values.values():
        assert value not in trace

    run = fusion.FusionRun("opaque-trace", "x")
    run.main_messages.append(message)
    public_trace = fusion._orchestration_trace(run, include_reasoning=True)
    for value in opaque_values.values():
        assert value not in public_trace


def test_fusion_preserves_reasoning_metadata(monkeypatch):
    details = [{"type": "reasoning", "id": "r1", "summary": [{"text": "run bash"}]}]
    worker = FakeWorker()

    def with_metadata(_coordinator, slot, messages, tools):
        message, usage = worker(slot, messages, tools)
        message["reasoning_details"] = details
        return message, usage

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", with_metadata)
    run = fusion.FusionRun("metadata", "goal", tools=[BASH_TOOL])
    assert run.advance()["status"] == "awaiting_tools"
    assert (
        run.advance(tool_results=[{"tool_call_id": "sidekick:call_bash_1", "content": "hello"}])[
            "status"
        ]
        == "completed"
    )
    second_sidekick = [
        messages
        for _, messages, _ in worker.calls
        if messages[0]["content"] == fusion_prompts.SIDEKICK_POLICY
    ][1]
    assert any(
        m.get("reasoning_details") == details for m in second_sidekick if m["role"] == "assistant"
    )
