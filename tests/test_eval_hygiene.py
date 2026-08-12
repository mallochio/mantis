"""Regression tests for evaluation aggregation hygiene."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("score", ROOT / "eval" / "score.py")
score = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(score)


def test_error_rows_are_not_in_aggregate_metrics():
    fixtures = {"ok": {"expect": ["answer"]}, "failed": {"expect": ["answer"]}}
    rows = [
        {"id": "ok", "response_text": "answer", "latency_s": 2, "est_cost_usd": 3},
        {
            "id": "failed",
            "response_text": "",
            "latency_s": 100,
            "est_cost_usd": 50,
            "error": "HTTP 500",
        },
    ]

    metrics = score.aggregate_metrics(rows, fixtures)

    assert metrics["success_count"] == 1
    assert metrics["error_count"] == 1
    assert metrics["auto_score_mean"] == 1.0
    assert metrics["latency_mean_s"] == 2
    assert metrics["cost_sum_usd"] == 3
