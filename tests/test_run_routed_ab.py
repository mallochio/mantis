"""Hermetic checks for the routed A/B runner."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import run_routed_ab as runner
from evaluate_routed_harness import (
    REQUIRED_GROUPS,
    EvalCase,
    _validate_record,
    aggregate_records,
    load_cases,
)
from routed_ab_tools import OPENROUTER_BASE_URL, priced_cost, usage_totals


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
    assert efficient["model"] == "~openai/gpt-sol-latest"
    assert efficient["base_url"] == OPENROUTER_BASE_URL
    assert efficient["reasoning_effort"] == "medium"
    assert capable["model"] == "~openai/gpt-sol-latest"
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
        "cost_basis": "priced_estimate",
    }
    empty = runner.usage_numbers(None, {})
    assert empty["input_tokens"] == 0


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    @property
    def headers(self) -> dict[str, str]:
        return {}

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(self, payloads: list[dict[str, Any]]):
        self._iter = iter(payloads)

    def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(next(self._iter))


def _usage_payload(prompt: int, completion: int, cost: float) -> dict[str, Any]:
    return {
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": 3},
            "cost_details": {"upstream_inference_cost": cost},
        },
    }


def test_usage_totals_sums_rounds_and_upstream_cost():
    totals = usage_totals([_usage_payload(10, 2, 0.5), _usage_payload(7, 1, 0.25)])
    assert totals["usage"]["prompt_tokens"] == 17
    assert totals["usage"]["completion_tokens"] == 3
    assert totals["usage"]["prompt_tokens_details"]["cached_tokens"] == 6
    assert totals["cost"] == 0.75


def test_drive_case_returns_budget_observation_instead_of_raising(tmp_path: Path):
    plan = {"model": "~openai/gpt-sol-latest", "base_url": "https://example.invalid"}
    tool_call = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "true"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {},
    }
    client = _FakeClient([tool_call, tool_call])
    observation = runner.drive_case(client, "https://example.invalid", plan, "do it", tmp_path, 2)
    assert observation["content"] == "EXCEEDED_TOOL_BUDGET"
    assert observation["tool_rounds"] == 2


def test_drive_case_sums_usage_across_direct_rounds(tmp_path: Path):
    plan = {"model": "~openai/gpt-sol-latest", "base_url": "https://example.invalid"}
    tool_call = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "t1",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "true"}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "is_byok": True},
    }
    client = _FakeClient([tool_call, _usage_payload(10, 2, 0.5)])
    observation = runner.drive_case(client, "https://example.invalid", plan, "do it", tmp_path, 4)
    assert observation["content"] == "done"
    assert observation["event"]["usage"]["prompt_tokens"] == 15
    assert observation["event"]["cost"] == 0.5


def test_priced_cost_uses_rate_card_and_strips_effort_suffix():
    prices = {
        "models": {
            "~openai/gpt-astra-latest": {
                "input_per_token": 2e-06,
                "cached_input_per_token": 1e-07,
                "output_per_token": 1e-05,
            }
        }
    }
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    models = {"~openai/gpt-astra-latest|low": {"prompt_tokens": 1000, "completion_tokens": 10}}
    assert priced_cost(usage, models, prices) == pytest.approx(0.0021)
    prefixed = {
        "openrouter/~openai/gpt-astra-latest": {"prompt_tokens": 1000, "completion_tokens": 10}
    }
    assert priced_cost(usage, prefixed, prices) == pytest.approx(0.0021)
    cached = {
        "~openai/gpt-astra-latest": {
            "prompt_tokens": 1000,
            "cached_tokens": 750,
            "completion_tokens": 10,
        }
    }
    assert priced_cost(usage, cached, prices) == pytest.approx(0.000675)


def test_auto_judge_accepts_fusion_answer_prefix():
    assert runner.auto_judge("no-tool-chat", "ANSWER:\nROUTING_OK", 0) is True
    assert runner.auto_judge("no-tool-chat", "ROUTING_OK", 0) is True
    assert runner.auto_judge("no-tool-chat", "ANSWER:\nSomething else", 0) is False


def _seed_workspace(case_id: str, root: Path) -> Path:
    workspace = root / case_id
    workspace.mkdir(parents=True)
    return runner.seed_workspace(case_id, workspace, runner.DEFAULT_FIXTURES)


def _run_in(workspace: Path, *command: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *command], cwd=workspace, capture_output=True, text=True)


def test_case_corpus_loads_with_every_required_group():
    cases = load_cases(runner.DEFAULT_CASES)
    assert {case.case_id for case in cases} == {
        "easy-mechanical-edit",
        "slow-test-execution",
        "upstream-reuse",
        "ambiguous-feature",
        "error-recovery",
        "no-tool-chat",
    }
    assert {case.group for case in cases} == REQUIRED_GROUPS


def test_workspace_cases_name_their_seeded_files(tmp_path: Path):
    for case in load_cases(runner.DEFAULT_CASES):
        fixture_dir = runner.DEFAULT_FIXTURES / case.case_id
        if not fixture_dir.is_dir():
            continue
        seeded = {
            path.relative_to(fixture_dir).as_posix()
            for path in fixture_dir.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        assert seeded, case.case_id
        assert any(name in case.prompt for name in seeded), case.case_id


def test_seed_workspace_copies_fixtures_idempotently(tmp_path: Path):
    workspace = _seed_workspace("upstream-reuse", tmp_path)
    first = (workspace / "caller.py").read_text()
    assert (workspace / "dispatch_helper.py").is_file()
    assert (workspace / "tests" / "test_caller.py").is_file()
    assert (workspace / "check_reuse.py").is_file()
    runner.seed_workspace("upstream-reuse", workspace, runner.DEFAULT_FIXTURES)
    assert (workspace / "caller.py").read_text() == first


def test_seed_workspace_without_fixtures_leaves_workspace_empty(tmp_path: Path):
    workspace = _seed_workspace("no-tool-chat", tmp_path)
    assert list(workspace.rglob("*")) == []
    workspace2 = tmp_path / "elsewhere"
    workspace2.mkdir(parents=True)
    assert runner.seed_workspace("no-tool-chat", workspace2, tmp_path / "absent") == workspace2
    assert list(workspace2.rglob("*")) == []


def test_easy_mechanical_edit_fixture_round_trips_checker(tmp_path: Path):
    workspace = _seed_workspace("easy-mechanical-edit", tmp_path)
    pristine = _run_in(workspace, "check_settings.py")
    assert pristine.returncode != 0
    settings = workspace / "settings.toml"
    settings.write_text(settings.read_text().replace("legacy_dispatch", "routed_dispatch"))
    renamed = _run_in(workspace, "check_settings.py")
    assert renamed.returncode == 0
    assert "CONFIG_OK" in renamed.stdout


def test_upstream_reuse_checker_rejects_duplicate_and_accepts_reuse(tmp_path: Path):
    workspace = _seed_workspace("upstream-reuse", tmp_path)
    duplicate = _run_in(workspace, "check_reuse.py")
    assert duplicate.returncode != 0
    assert "REUSE_FAILED" in duplicate.stdout
    workspace.joinpath("caller.py").write_text(
        '"""Internal caller that routes requests through the shared helper."""\n\n'
        "from dispatch_helper import routed_dispatch\n\n\n"
        "def handle(request: dict) -> dict:\n"
        '    """Serve one request through the shared dispatch helper."""\n'
        "    return routed_dispatch(request)\n"
    )
    reused = _run_in(workspace, "check_reuse.py")
    assert reused.returncode == 0
    assert "REUSE_OK" in reused.stdout
    suite = _run_in(workspace, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v")
    assert suite.returncode == 0


def test_ambiguous_feature_fixture_passes_and_lacks_dry_run(tmp_path: Path):
    workspace = _seed_workspace("ambiguous-feature", tmp_path)
    suite = _run_in(workspace, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v")
    assert suite.returncode == 0
    assert "OK" in suite.stderr
    assert "dry_run" not in workspace.joinpath("tests/test_router.py").read_text()


def test_slow_suite_and_error_probe_behave_as_prompted(tmp_path: Path):
    workspace = _seed_workspace("slow-test-execution", tmp_path)
    suite = _run_in(workspace, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v")
    assert suite.returncode == 0
    assert "Ran 4 tests" in suite.stderr
    probe_root = _seed_workspace("error-recovery", tmp_path)
    probe = _run_in(probe_root, "upstream_check.py")
    assert probe.returncode == 1
    assert "UPSTREAM_TIMEOUT" in probe.stdout
