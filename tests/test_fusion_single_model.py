"""Fusion chat responses present as a single model by default."""

from __future__ import annotations

import json
from typing import Any

import api
import fusion
import fusion_prompts
import pytest
import serve_config
from fastapi.testclient import TestClient

from tests.test_fusion_tool_mode import call, reply

USAGE = {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}

READ_FILE_CALL = {
    "id": "call_read_1",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path": "config.py"}'},
}

MAIN_PLANNING_REASONING = "main planning reasoning"
MAIN_POST_TOOL_REASONING = "main post-tool reasoning"
MAIN_REVIEW_REASONING = "main review reasoning"
SIDEKICK_REASONING = "sidekick hidden reasoning"
SIDEKICK_REPORT = "Done editing config.py."
PLAN_TEXT = "PLAN: inspected config, now edit\nBRIEF: edit config.py"


@pytest.fixture(autouse=True)
def _clear_run_store():
    """Clear shared in-memory runs around each test; chat requests fill them.

    Without this, later memory-mode tests hit the run capacity guard.
    """
    serve_config._runs.clear()
    yield
    serve_config._runs.clear()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    monkeypatch.setenv("MANTIS_RUN_STORE", "memory")
    return TestClient(api.app)


@pytest.fixture
def plan_tools_config(monkeypatch, tmp_path):
    path = tmp_path / "catalog.toml"
    path.write_text(
        "[fusion]\n"
        'main = "gpt-5_6-sol"\n'
        'sidekick = "deepseek-v4-flash"\n'
        "sidekick_max_tool_rounds = 16\n"
        "run_max_turns = 64\n"
    )
    config = fusion.FusionConfig(path)
    monkeypatch.setattr(fusion, "_FUSION_CONFIG", config)
    return config


class SingleModelWorker:
    """Lead uses a client tool, delegates work, and answers plainly."""

    def __init__(self):
        self.main_calls = 0
        self.sidekick_calls = 0

    def __call__(self, slot, messages, tools):
        if messages[0]["content"] == fusion_prompts.SIDEKICK_POLICY:
            self.sidekick_calls += 1
            return reply(SIDEKICK_REPORT, reasoning=SIDEKICK_REASONING), USAGE
        self.main_calls += 1
        if self.main_calls == 1:
            return reply("", READ_FILE_CALL, reasoning=MAIN_PLANNING_REASONING), USAGE
        if self.main_calls == 2:
            return reply(
                "",
                call("delegate", "sidekick", message="edit config.py"),
                reasoning=MAIN_POST_TOOL_REASONING,
            ), USAGE
        return reply(SIDEKICK_REPORT, reasoning=MAIN_REVIEW_REASONING), USAGE


def _headers():
    return {"Authorization": "Bearer test-key"}


def _payload(stream: bool = False) -> dict[str, Any]:
    return {
        "model": "mantis/fusion",
        "messages": [{"role": "user", "content": "Update the config file"}],
        "tools": [READ_FILE_TOOL],
        "stream": stream,
    }


def _drive_to_completion(client, headers) -> tuple[dict[str, Any], dict[str, Any]]:
    first = client.post("/v1/chat/completions", headers=headers, json=_payload())
    assert first.status_code == 200
    body = first.json()
    message = body["choices"][0]["message"]
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    call = message["tool_calls"][0]
    follow_up = {
        **_payload(),
        "stream": False,
        "messages": [
            {"role": "user", "content": "Update the config file"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call["id"], "content": "PORT = 8080"},
        ],
    }
    second = client.post("/v1/chat/completions", headers=headers, json=follow_up)
    assert second.status_code == 200
    return body, second.json()


def _stream_reasoning(response) -> str:
    return "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("reasoning", "")
        for line in response.text.splitlines()
        if line.startswith("data: {")
    )


def test_default_scope_shows_main_lane_only(client, plan_tools_config, monkeypatch):
    worker = SingleModelWorker()
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    first_body, second_body = _drive_to_completion(client, _headers())

    first_message = first_body["choices"][0]["message"]
    assert first_message["tool_calls"][0]["function"]["name"] == "read_file"
    first_reasoning = first_message.get("reasoning", "")
    assert MAIN_PLANNING_REASONING in first_reasoning
    assert "Tool call: read_file" in first_reasoning
    for hidden in (
        SIDEKICK_REASONING,
        SIDEKICK_REPORT,
        "Fusion · Planning",
        "PLAN:",
        "BRIEF:",
        "Delegated to sidekick",
    ):
        assert hidden not in first_reasoning

    second_message = second_body["choices"][0]["message"]
    assert second_message["content"] == SIDEKICK_REPORT
    second_reasoning = second_message.get("reasoning", "")
    assert MAIN_POST_TOOL_REASONING in second_reasoning
    assert MAIN_REVIEW_REASONING in second_reasoning
    assert "PORT = 8080" in second_reasoning
    assert MAIN_PLANNING_REASONING not in second_reasoning
    for hidden in (
        SIDEKICK_REASONING,
        SIDEKICK_REPORT,
        "Fusion · Planning",
        "PLAN:",
        "BRIEF:",
        "Delegated to sidekick",
    ):
        assert hidden not in second_reasoning

    event = second_body["mantis_event"]
    assert event["trace_scope"] == "main"
    assert isinstance(event["handoff_history"], list) and event["handoff_history"]
    assert all("kind" in entry for entry in event["handoff_history"])


def test_default_streaming_scope_matches_non_streaming(client, plan_tools_config, monkeypatch):
    worker = SingleModelWorker()
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    monkeypatch.setattr(api, "_uses_live_fusion_stream", lambda request: False)
    headers = _headers()
    first = client.post("/v1/chat/completions", headers=headers, json=_payload(stream=True))
    assert first.status_code == 200
    streamed_reasoning = _stream_reasoning(first)
    assert MAIN_PLANNING_REASONING in streamed_reasoning
    assert "Tool call: read_file" in streamed_reasoning
    assert SIDEKICK_REASONING not in streamed_reasoning
    assert "PLAN:" not in streamed_reasoning
    tool_chunk = next(
        json.loads(line[6:])
        for line in first.text.splitlines()
        if line.startswith("data: {")
        and "tool_calls" in json.loads(line[6:])["choices"][0]["delta"]
    )
    streamed_call = tool_chunk["choices"][0]["delta"]["tool_calls"][0]
    assert streamed_call["function"]["name"] == "read_file"
    event_chunk = next(
        json.loads(line[6:])
        for line in first.text.splitlines()
        if line.startswith("data: {") and "mantis_event" in json.loads(line[6:])
    )
    assert event_chunk["mantis_event"]["trace_scope"] == "main"


def test_full_scope_still_exposes_everything(client, plan_tools_config, monkeypatch):
    worker = SingleModelWorker()
    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    headers = {**_headers(), "X-Mantis-Trace-Scope": "orchestration"}
    _, second_body = _drive_to_completion(client, headers)

    second_reasoning = second_body["choices"][0]["message"]["reasoning"]
    assert SIDEKICK_REASONING in second_reasoning
    assert second_body["mantis_event"]["trace_scope"] == "orchestration"


def test_ab_consumed_metadata_is_identical_across_scopes(client, plan_tools_config, monkeypatch):
    events = {}
    for scope in ("main", "orchestration"):
        worker = SingleModelWorker()
        monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
        headers = {**_headers(), "X-Mantis-Trace-Scope": scope}
        _, second_body = _drive_to_completion(client, headers)
        events[scope] = second_body["mantis_event"]

    for key in (
        "usage",
        "usage_models",
        "main_model",
        "sidekick_model",
        "tool_count",
        "tool_results",
        "fusion_savings_usd",
        "review",
        "status",
        "handoff_history",
    ):
        assert events["main"][key] == events["orchestration"][key], key
    assert events["main"]["trace_scope"] == "main"
    assert events["orchestration"]["trace_scope"] == "orchestration"

    handoffs = events["main"]["handoff_history"]
    assert handoffs and all(isinstance(entry.get("kind"), str) for entry in handoffs)
    assert any(entry["kind"] not in ("turn", "answer") for entry in handoffs)


def test_trace_scope_header_and_env_resolution(monkeypatch):
    monkeypatch.delenv("MANTIS_FUSION_TRACE_SCOPE", raising=False)
    assert api._fusion_trace_scope(None) == "main"
    assert api._fusion_trace_scope({}) == "main"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "full"}) == "orchestration"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "meta"}) == "orchestration"
    assert api._fusion_trace_scope({"x-mantis-fusion-trace": "orchestration"}) == "orchestration"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "sidekick"}) == "sidekick"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "worker"}) == "sidekick"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "main"}) == "main"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "single"}) == "main"
    assert api._fusion_trace_scope({"x-mantis-trace-scope": "bogus"}) == "main"
    monkeypatch.setenv("MANTIS_FUSION_TRACE_SCOPE", "sidekick")
    assert api._fusion_trace_scope({}) == "sidekick"
    monkeypatch.setenv("MANTIS_FUSION_TRACE_SCOPE", "full")
    assert api._fusion_trace_scope({}) == "orchestration"
    monkeypatch.setenv("MANTIS_FUSION_TRACE_SCOPE", "bogus")
    assert api._fusion_trace_scope({}) == "main"
