"""Hermetic checks for the routed A/B runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import run_routed_ab as runner
from evaluate_routed_harness import EvalCase, _validate_record, aggregate_records


def _catalog() -> dict[str, Any]:
    return runner.load_catalog(runner.DEFAULT_CATALOG)


def _case(case_id: str = "no-tool-chat") -> EvalCase:
    return EvalCase(
        case_id,
        "no_tool_conversation",
        "Answer with exactly ROUTING_OK and do not call tools.",
        {"description": "exact echo", "checks": ["compare exact response"]},
    )


def test_request_plan_maps_every_variant():
    catalog = _catalog()
    efficient = runner.request_plan("efficient-only", catalog)
    capable = runner.request_plan("capable-only", catalog)
    assert efficient["model"] == "openrouter/~openai/gpt-sol-latest"
    assert efficient["reasoning_effort"] == "medium"
    assert capable["model"] == "openrouter/~openai/gpt-sol-latest"
    assert capable["reasoning_effort"] == "high"
    assert runner.request_plan("mantis-base", catalog) == {"model": "mantis/base"}
    pinned = runner.request_plan("fusion-pinned", catalog)
    routed = runner.request_plan("fusion-routed", catalog)
    assert pinned["model"] == routed["model"] == "mantis/fusion"
    assert pinned["worker_profiles"][0]["model"] == catalog["fusion"]["main"]
    assert "worker_profiles" not in routed
    assert routed["tool_options"]["server_execution"] is True
    with pytest.raises(ValueError):
        runner.request_plan("unknown-variant", catalog)


def test_auto_judge_only_decides_machine_checkable_cases():
    assert runner.auto_judge("no-tool-chat", "ROUTING_OK", 0) is True
    assert runner.auto_judge("no-tool-chat", "ROUTING_OK", 2) is False
    assert runner.auto_judge("no-tool-chat", " ROUTING_OK. ", 0) is False
    assert runner.auto_judge("easy-mechanical-edit", "done", 3) is None


def test_execute_tool_call_round_trips_in_workspace(tmp_path: Path):
    write = {
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": "a.txt", "content": "hello"}),
        }
    }
    assert runner.execute_tool_call(write, tmp_path) == "written"
    listing = {"function": {"name": "list_files", "arguments": "{}"}}
    assert "a.txt" in runner.execute_tool_call(listing, tmp_path)
    bash = {"function": {"name": "bash", "arguments": json.dumps({"command": "cat a.txt"})}}
    assert "hello" in runner.execute_tool_call(bash, tmp_path)
    edit = {
        "function": {
            "name": "edit_file",
            "arguments": json.dumps({"path": "a.txt", "old": "hello", "new": "hi"}),
        }
    }
    assert runner.execute_tool_call(edit, tmp_path) == "edited"
    search = {"function": {"name": "search_files", "arguments": json.dumps({"query": "hi"})}}
    assert "a.txt" in runner.execute_tool_call(search, tmp_path)
    read = {"function": {"name": "read_file", "arguments": json.dumps({"path": "a.txt"})}}
    assert runner.execute_tool_call(read, tmp_path) == "hi"
    unknown = {"function": {"name": "detonate", "arguments": "{}"}}
    assert "unknown tool" in runner.execute_tool_call(unknown, tmp_path)
    bad_json = {"function": {"name": "bash", "arguments": "{not json"}}
    assert "exit=" in runner.execute_tool_call(bad_json, tmp_path)


def test_build_record_matches_summarizer_schema():
    event = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 40},
        "cost": 0.001,
        "usage_models": {"~openai/gpt-luna-latest": {"prompt_tokens": 60, "completion_tokens": 30}},
        "fusion_savings_usd": 0.002,
        "main_model": "astra",
        "sidekick_model": "luna",
    }
    observation = {
        "content": "ROUTING_OK",
        "event": event,
        "response": {},
        "tool_rounds": 0,
        "wall_time_s": 1.25,
    }
    plan = {"model": "mantis/fusion"}
    record = runner.build_record("fusion-routed", _case(), observation, plan)
    assert record["passed"] is True
    assert record["needs_review"] is False
    assert record["selected_models"] == ["~openai/gpt-luna-latest"]
    assert record["input_tokens"] == 100
    assert record["provider_cost"] == 0.001
    assert record["fusion_savings_usd"] == 0.002
    _validate_record(record)
    summary = aggregate_records([record])
    assert summary["fusion-routed"]["runs"] == 1
    assert summary["fusion-routed"]["passed"] == 1


def test_build_record_flags_review_for_judgment_cases():
    observation = {
        "content": "renamed the label",
        "event": {},
        "response": {},
        "tool_rounds": 3,
        "wall_time_s": 9.0,
    }
    plan = {"model": "mantis/base"}
    case = EvalCase(
        "easy-mechanical-edit",
        "easy_mechanical_edit",
        "rename",
        {"description": "d", "checks": ["c"]},
    )
    record = runner.build_record("mantis-base", case, observation, plan)
    assert record["passed"] is False
    assert record["needs_review"] is True
    assert record["selected_models"] == ["mantis/base"]
    assert record["provider_cost"] == 0.0
    _validate_record(record)


def test_usage_numbers_falls_back_to_response_usage():
    response = {
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    }
    numbers = runner.usage_numbers({}, response)
    assert numbers == {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_read_tokens": 4,
        "provider_cost": 0.0,
    }
    empty = runner.usage_numbers(None, {})
    assert empty["input_tokens"] == 0
