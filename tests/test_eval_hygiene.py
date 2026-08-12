"""Regression tests for evaluation aggregation hygiene."""

import importlib.util
import json
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


def test_report_suppresses_low_success_means_and_zero_baseline_delta(tmp_path, monkeypatch):
    fixtures = tmp_path / "fixtures.jsonl"
    fixture_rows = [{"id": "t1", "tier": "simple", "expect": ["answer"], "prompt": "test"}]
    fixture_rows.extend(
        {"id": f"t{i}", "tier": "medium", "expect": ["answer"], "prompt": "test"}
        for i in range(2, 5)
    )
    fixtures.write_text("\n".join(json.dumps(row) for row in fixture_rows) + "\n")
    results = tmp_path / "results.jsonl"
    rows = []
    for config in ("direct", "trinity"):
        rows.extend(
            {
                "id": f"t{i}",
                "config": config,
                "tier": next(row["tier"] for row in fixture_rows if row["id"] == f"t{i}"),
                "response_text": "" if config == "direct" else "answer",
                "latency_s": 1,
                "est_cost_usd": 1,
                "error": None,
            }
            for i in range(1, 5)
        )
    for config in ("conductor-old", "conductor-new"):
        rows.append(
            {
                "id": "t1",
                "config": config,
                "tier": "simple",
                "response_text": "answer",
                "latency_s": 1,
                "est_cost_usd": 1,
                "error": None,
            }
        )
        rows.extend(
            {
                **rows[-1],
                "id": f"t{i}",
                "tier": next(row["tier"] for row in fixture_rows if row["id"] == f"t{i}"),
                "error": "HTTP 500",
                "response_text": "",
            }
            for i in range(2, 5)
        )
    results.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = tmp_path / "report.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "score.py",
            "--fixtures",
            str(fixtures),
            "--results",
            str(results),
            "--output",
            str(report),
        ],
    )

    score.main()
    text = report.read_text()

    assert "n/a (1/4 successful)" in text
    assert "| medium | conductor-old | 0/3 | 3 | n/a (0/3 successful) |" in text
    assert "- conductor-old mean auto_score: n/a (1/4 successful)" in text
    assert "### b) conductor-new vs conductor-old" in text
    assert "quality delta:" not in text.split("### b) conductor-new vs conductor-old", 1)[1].split(
        "### c)", 1
    )[0]
    assert "- trinity mean auto_score: 1.000" in text
    assert "quality delta percentage:" not in text


def test_paired_comparison_uses_successful_intersection():
    fixtures = {f"t{i}": {"expect": ["answer"]} for i in range(1, 6)}
    left = [
        {"id": f"t{i}", "response_text": "answer"} for i in range(1, 5)
    ]
    right = [
        {"id": "t1", "response_text": "answer"},
        {"id": "t5", "response_text": "answer"},
    ]

    paired_left, paired_right = score.paired_comparable(left, right, fixtures)

    assert paired_left == []
    assert paired_right == []
    assert score.metric_display(score.aggregate_metrics(left, fixtures)) == "1.000"
