"""Network-free tests for routed-harness evaluation records."""

from __future__ import annotations

import json

import evaluate_routed_harness as evaluator
import pytest


def _record(**overrides):
    record = {
        "variant": "mantis-base",
        "passed": True,
        "selected_models": ["provider/efficient"],
        "input_tokens": 10,
        "output_tokens": 4,
        "cache_read_tokens": 3,
        "provider_cost": 0.02,
        "tool_rounds": 2,
        "wall_time_s": 1.5,
    }
    record.update(overrides)
    return record


def test_shipped_cases_are_non_executable_and_cover_required_groups():
    cases = evaluator.load_cases(evaluator.DEFAULT_CASES_PATH)
    assert {case.group for case in cases} == evaluator.REQUIRED_GROUPS
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.expected for case in cases)
    assert all(set(case.expected) <= evaluator.EXPECTATION_KEYS for case in cases)
    assert not hasattr(evaluator, "run_evaluation")
    assert not hasattr(evaluator, "EndpointRunner")


def test_case_manifest_rejects_executable_fields(tmp_path):
    rows = [
        {
            "id": group,
            "group": group,
            "prompt": "prompt",
            "expected": {"description": "observable outcome"},
        }
        for group in sorted(evaluator.REQUIRED_GROUPS)
    ]
    rows[0]["expected"] = {"test_command": "false"}
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="unsupported expectations"):
        evaluator.load_cases(path)


def test_aggregate_records_computes_totals_and_selected_models():
    summary = evaluator.aggregate_records(
        [
            _record(selected_models=["provider/efficient", "provider/capable"]),
            _record(
                passed=False,
                selected_models=["provider/efficient"],
                input_tokens=8,
                output_tokens=2,
                cache_read_tokens=1,
                provider_cost=0.01,
                tool_rounds=1,
                wall_time_s=0.5,
            ),
        ]
    )
    assert summary["mantis-base"] == {
        "runs": 2,
        "passed": 1,
        "pass_rate": 0.5,
        "input_tokens": 18,
        "output_tokens": 6,
        "cache_read_tokens": 4,
        "provider_cost": 0.03,
        "tool_rounds": 3,
        "wall_time_s": 2.0,
        "selected_models": ["provider/capable", "provider/efficient"],
    }


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"variant": "unknown"}, "unknown evaluation variant"),
        ({"selected_models": []}, "non-empty selected_models"),
        ({"input_tokens": True}, "numeric input_tokens"),
        ({"passed": "yes"}, "boolean passed"),
    ],
)
def test_record_schema_is_strict(change, match):
    with pytest.raises((TypeError, ValueError), match=match):
        evaluator.aggregate_records([_record(**change)])


def test_jsonl_records_round_trip_through_summary(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(_record(variant="capable-only")) + "\n")
    assert evaluator.summarize_records(path)["capable-only"]["runs"] == 1


def test_cli_has_only_network_free_commands():
    parser = evaluator.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"validate-cases", "summarize"}


def test_declared_variants_are_fixed():
    assert evaluator.VARIANTS == (
        "efficient-only",
        "capable-only",
        "mantis-base",
        "fusion-pinned",
        "fusion-routed",
    )
