"""Tests for routed A/B report aggregation and paired deltas."""

from __future__ import annotations

import json
from pathlib import Path

import ab_report
import pytest


def _rows(variant: str, costs: list[float], checks: list[bool]) -> list[dict]:
    return [
        {
            "variant": variant,
            "case_id": "edit",
            "seed": seed,
            "provider_cost": cost,
            "input_tokens": tokens - 20,
            "output_tokens": 20,
            "tool_rounds": rounds,
            "check_passed": check,
        }
        for seed, (cost, check, tokens, rounds) in enumerate(
            zip(costs, checks, [100, 300] if variant == "fusion-pinned" else [80, 200],
                [2, 4] if variant == "fusion-pinned" else [1, 3], strict=True),
            start=1,
        )
    ]


def _assessments(variant: str, nouls: list[float], qualities: list[float]) -> list[dict]:
    return [
        {
            "variant": variant,
            "case_id": "edit",
            "seed": seed,
            "answers": {
                "success": {"noul": noul},
                "quality": {"score": quality},
            },
        }
        for seed, (noul, quality) in enumerate(zip(nouls, qualities, strict=True), start=1)
    ]


def test_aggregate_stats_and_paired_case_deltas():
    records = _rows("fusion-pinned", [1.0, 3.0], [True, False])
    records += _rows("fusion-tool", [0.5, 1.5], [True, True])
    assessments = _assessments("fusion-pinned", [0.5, 1.0], [2.0, 2.0])
    assessments += _assessments("fusion-tool", [0.75, 0.75], [3.0, 3.0])
    summary, matched = ab_report.aggregate(records, assessments)
    assert summary["fusion-pinned"]["cost_usd"] == (2.0, 2**0.5)
    assert summary["fusion-pinned"]["tokens"] == (200.0, 2**0.5 * 100)
    assert summary["fusion-pinned"]["check_pass_rate"] == (0.5, 2**-0.5)
    assert summary["fusion-tool"]["jev_noul"] == (0.75, 0.0)
    assert summary["fusion-tool"]["quality_per_dollar"] == (4.0, 2**1.5)
    deltas = ab_report.paired_deltas(records, matched, "fusion-pinned")
    by_metric = {metric: delta for _, _, metric, delta in deltas}
    assert by_metric["cost_usd"] == -1.0
    assert by_metric["tokens"] == -60.0
    assert by_metric["rounds"] == -1.0
    assert by_metric["check_pass_rate"] == 0.5
    assert by_metric["jev_noul"] == 0.0
    assert by_metric["jev_quality"] == 1.0
    assert by_metric["quality_per_dollar"] == pytest.approx(8 / 3)


def test_report_cli_writes_and_prints_markdown(tmp_path: Path, capsys):
    records = _rows("fusion-pinned", [1.0, 3.0], [True, False])
    assessments = _assessments("fusion-pinned", [0.5, 1.0], [2.0, 2.0])
    records_path = tmp_path / "runs.jsonl"
    assessments_path = tmp_path / "jev.jsonl"
    output_path = tmp_path / "nested" / "report.md"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assessments_path.write_text("".join(json.dumps(row) + "\n" for row in assessments))
    result = ab_report.main([
        "--records", str(records_path), "--assessments", str(assessments_path),
        "--out", str(output_path),
    ])
    assert result == 0
    assert "| fusion-pinned | cost_usd | 2 |" in output_path.read_text()
    assert capsys.readouterr().out == output_path.read_text()
