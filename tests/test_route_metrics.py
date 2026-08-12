"""Regression coverage for router evaluation result accounting."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("route_metrics", ROOT / "eval" / "route_metrics.py")
assert _SPEC and _SPEC.loader
route_metrics = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(route_metrics)


def test_load_result_rows_skips_metadata(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(
        "\n".join(
            [
                json.dumps({"record_type": "metadata", "metadata": {"seed": 7}}),
                json.dumps({"record_type": "result", "instance_id": "one", "arm": "cheap-only"}),
                json.dumps({"record_type": "metadata", "metadata": {"spent_usd": 1}}),
            ]
        )
        + "\n"
    )

    assert route_metrics.load_result_rows(results) == [
        {"record_type": "result", "instance_id": "one", "arm": "cheap-only"}
    ]


def test_metrics_include_failed_attempts_in_quality_and_cost():
    metrics = route_metrics.compute_metrics(
        [
            {"instance_id": "one", "arm": "cheap-only", "resolved": True, "cost_usd": 1.0},
            {
                "instance_id": "two",
                "arm": "cheap-only",
                "resolved": False,
                "cost_usd": 2.0,
                "error": "timeout",
            },
        ]
    )

    summary = metrics["arms"]["cheap-only"]
    assert summary == {
        "total": 2,
        "completed": 1,
        "resolved": 1,
        "errors": 1,
        "quality_mean": 0.5,
        "cost_usd": 3.0,
    }
