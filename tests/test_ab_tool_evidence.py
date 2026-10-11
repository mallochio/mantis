"""Tests for evidence-aware evaluation checks across client and server tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_routed_ab as runner
from ab_evidence import exit_code, tool_records, tool_use_count
from ab_seeds import ERROR_LINE, _error_recovery_check, _slow_suite_check, run_case_checks


def _seed_workspace(case_id: str, root: Path) -> Path:
    workspace = root / case_id
    workspace.mkdir(parents=True)
    return runner.seed_workspace(case_id, workspace, runner.DEFAULT_FIXTURES)


def _suite_call() -> dict[str, Any]:
    return {
        "id": "suite-1",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": "python -m unittest discover -s tests -t . -v"}),
        },
    }


def _client_suite_observation() -> dict[str, Any]:
    call = _suite_call()
    return {
        "messages": [
            {"role": "assistant", "tool_calls": [call]},
            {
                "role": "tool",
                "tool_call_id": "suite-1",
                "content": "exit=0\nRan 4 tests in 1.0s\n\nOK\n",
            },
        ],
    }


def _server_suite_observation() -> dict[str, Any]:
    return {
        "response_payloads": [
            {
                "mantis_event": {
                    "tool_calls": [_suite_call()],
                    "tool_results": [
                        {
                            "tool_call_id": "suite-1",
                            "exit_code": 0,
                            "content": "Ran 4 tests in 1.0s\n\nOK\n",
                        }
                    ],
                },
            }
        ],
    }


def test_slow_suite_requires_agent_execution_result_and_report(tmp_path: Path):
    workspace = _seed_workspace("slow-test-execution", tmp_path)
    result = run_case_checks("slow-test-execution", workspace, "", [], 0)
    assert result["check_passed"] is False
    report = "Ran 4 tests\nOK\nExit status 0"
    assert _slow_suite_check(report, _client_suite_observation())["check_passed"] is True
    assert _slow_suite_check(report, _server_suite_observation())["check_passed"] is True


def test_error_recovery_requires_two_observed_failures():
    records = []
    for identifier in ("probe-1", "probe-2"):
        records.extend(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": identifier,
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "python upstream_check.py"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": identifier, "content": f"exit=1\n{ERROR_LINE}\n"},
            ]
        )
    observation = {"messages": records}
    assert (
        _error_recovery_check(f"Failure: {ERROR_LINE}. Both runs exit status 1.", observation)[
            "check_passed"
        ]
        is True
    )
    assert _error_recovery_check("failure", {"messages": records[:2]})["check_passed"] is False


def test_tool_evidence_joins_server_records_by_canonical_id():
    call = {
        "id": "server-1",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": "python upstream_check.py"}),
        },
    }
    result = {"tool_call_id": "server-1", "content": "exit status 1\nfailed"}
    observation = {
        "response_payloads": [
            {"mantis_event": {"tool_calls": [call], "tool_results": [result]}},
            {"mantis_event": {"tool_calls": [call], "tool_results": [result]}},
        ],
        "server_run_details": {"tool_results": [result]},
    }
    records = tool_records(observation)
    assert len(records) == 1
    assert records[0]["tool_call_id"] == "server-1"
    assert exit_code(records[0]["result"]) == 1
    assert tool_use_count(observation) == 1


def test_sidekick_handoff_counts_as_tool_use_for_no_tool_case():
    observation = {
        "response_payloads": [
            {
                "mantis_event": {
                    "handoff_history": [
                        {"kind": "delegate", "lane": "sidekick-1", "task": "inspect"}
                    ],
                },
            }
        ],
    }
    uses = tool_use_count(observation)
    assert uses == 1
    assert runner.auto_judge("no-tool-chat", "ROUTING_OK", 0, uses) is False
