#!/usr/bin/env python3
"""Validate benchmark manifests and summarize external run records.

This module is deliberately network-free and never executes commands. Comparative
live evaluation is deferred to a separately controlled benchmark. Records passed
to ``summarize`` are observations supplied by that benchmark, not trusted proof;
this tool makes no quality or cost claim and does not change routing defaults.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CASES_PATH = REPO / "eval" / "routed-harness-cases.jsonl"
REQUIRED_GROUPS = {
    "easy_mechanical_edit",
    "slow_test_execution",
    "hard_mechanical_upstream_reuse",
    "ambiguous_multifile_judgment",
    "error_recovery",
    "no_tool_conversation",
}
VARIANTS = (
    "efficient-only",
    "capable-only",
    "mantis-base",
    "fusion-pinned",
    "fusion-routed",
)
METRICS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "provider_cost",
    "tool_rounds",
    "wall_time_s",
)
EXPECTATION_KEYS = frozenset({"description", "checks"})


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    group: str
    prompt: str
    expected: dict[str, Any]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number}: row must be an object")
        rows.append(row)
    return rows


def _validate_expectation(case_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"case {case_id} requires a described expected outcome")
    unknown = sorted(set(value) - EXPECTATION_KEYS)
    if unknown:
        raise ValueError(f"case {case_id} has unsupported expectations: {', '.join(unknown)}")
    description = value.get("description")
    if not isinstance(description, str) or not description.strip():
        raise TypeError(f"case {case_id} expected.description must be a non-empty string")
    checks = value.get("checks", [])
    if not isinstance(checks, list) or not all(isinstance(item, str) and item for item in checks):
        raise TypeError(f"case {case_id} expected.checks must contain strings")
    return value


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen: set[str] = set()
    for row in _read_jsonl(path):
        case_id = str(row.get("id") or "")
        group = str(row.get("group") or "")
        prompt = str(row.get("prompt") or "")
        if not case_id or case_id in seen:
            raise ValueError(f"case id must be non-empty and unique: {case_id!r}")
        if group not in REQUIRED_GROUPS:
            raise ValueError(f"case {case_id} has unsupported group {group!r}")
        if not prompt:
            raise ValueError(f"case {case_id} requires a prompt")
        expected = _validate_expectation(case_id, row.get("expected"))
        cases.append(EvalCase(case_id, group, prompt, expected))
        seen.add(case_id)
    missing = REQUIRED_GROUPS - {case.group for case in cases}
    if missing:
        raise ValueError(f"case corpus is missing groups: {', '.join(sorted(missing))}")
    return cases


def _validate_record(record: dict[str, Any]) -> None:
    variant = record.get("variant")
    if variant not in VARIANTS:
        raise ValueError(f"unknown evaluation variant: {variant!r}")
    if not isinstance(record.get("passed"), bool):
        raise TypeError(f"run {variant} requires boolean passed")
    selected = record.get("selected_models")
    if not isinstance(selected, list) or not selected or not all(
        isinstance(model, str) and model for model in selected
    ):
        raise ValueError("every run requires non-empty selected_models")
    for metric in METRICS:
        value = record.get(metric)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"run {variant} requires numeric {metric}")


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    models: dict[str, set[str]] = {}
    for record in records:
        _validate_record(record)
        variant = str(record["variant"])
        summary = summaries.setdefault(
            variant, {"runs": 0, "passed": 0, **dict.fromkeys(METRICS, 0)}
        )
        summary["runs"] += 1
        summary["passed"] += int(record["passed"])
        for metric in METRICS:
            summary[metric] += record[metric]
        models.setdefault(variant, set()).update(record["selected_models"])
    for variant, summary in summaries.items():
        summary["pass_rate"] = summary["passed"] / summary["runs"]
        summary["selected_models"] = sorted(models[variant])
        summary["provider_cost"] = round(float(summary["provider_cost"]), 12)
        summary["wall_time_s"] = round(float(summary["wall_time_s"]), 6)
        summaries[variant] = {
            "runs": summary["runs"],
            "passed": summary["passed"],
            "pass_rate": summary["pass_rate"],
            **{metric: summary[metric] for metric in METRICS},
            "selected_models": summary["selected_models"],
        }
    return summaries


def summarize_records(path: Path) -> dict[str, dict[str, Any]]:
    return aggregate_records(_read_jsonl(path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-cases")
    validate.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--records", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-cases":
        cases = load_cases(args.cases)
        print(json.dumps({"cases": len(cases), "groups": sorted(REQUIRED_GROUPS)}))
    else:
        print(json.dumps(summarize_records(args.records), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
