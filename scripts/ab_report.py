"""Summarize routed A/B records and matched assessment deltas."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

REPORT_METRICS = (
    "cost_usd", "tokens", "rounds", "check_pass_rate", "jev_noul",
    "jev_quality", "quality_per_dollar",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read non-empty JSON Lines records from a file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assessment_scores(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, float]]:
    """Index Jev success and quality values by run identity."""
    scores = {}
    for row in rows:
        answers = row.get("answers") or {}
        success = answers.get("success") or {}
        quality = answers.get("quality") or {}
        key = (str(row["variant"]), str(row["case_id"]), int(row.get("seed", 1)))
        values = {}
        if isinstance(success.get("noul"), (int, float)):
            values["jev_noul"] = float(success["noul"])
        if isinstance(quality.get("score"), (int, float)):
            values["jev_quality"] = float(quality["score"])
        scores[key] = values
    return scores


def _run_metrics(row: dict[str, Any], assessment: dict[str, float]) -> dict[str, float]:
    """Convert one run and its judge answers into report metrics."""
    cost = float(row.get("provider_cost") or 0.0)
    metrics = {
        "cost_usd": cost,
        "tokens": float(row.get("input_tokens") or 0) + float(row.get("output_tokens") or 0),
        "rounds": float(row.get("tool_rounds") or 0),
        "check_pass_rate": float(bool(row.get("check_passed", row.get("passed", False)))),
    }
    metrics.update(assessment)
    if assessment.get("jev_quality") is not None and cost > 0:
        metrics["quality_per_dollar"] = assessment["jev_quality"] / cost
    return metrics


def _stats(values: list[float]) -> tuple[float | None, float | None]:
    """Return the mean and sample standard deviation for observed values."""
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate(
    records: list[dict[str, Any]], assessments: list[dict[str, Any]]
) -> tuple[
    dict[str, dict[str, tuple[float | None, float | None]]],
    dict[tuple[str, str, int], dict[str, float]],
]:
    """Aggregate per-run metrics by variant and retain matched values."""
    scores = _assessment_scores(assessments)
    values: dict[str, dict[str, list[float]]] = {}
    matched = {}
    for row in records:
        key = (str(row["variant"]), str(row["case_id"]), int(row.get("seed", 1)))
        metrics = _run_metrics(row, scores.get(key, {}))
        matched[key] = metrics
        bucket = values.setdefault(key[0], {})
        for metric, value in metrics.items():
            bucket.setdefault(metric, []).append(value)
    result = {
        variant: {metric: _stats(metrics.get(metric, [])) for metric in REPORT_METRICS}
        for variant, metrics in values.items()
    }
    return result, matched


def paired_deltas(
    records: list[dict[str, Any]],
    matched: dict[tuple[str, str, int], dict[str, float]],
    baseline: str,
) -> list[tuple[str, str, str, float]]:
    """Return per-case mean deltas against the same-seed baseline rows."""
    baseline_rows = {
        (case_id, seed): metrics for (variant, case_id, seed), metrics in matched.items()
        if variant == baseline
    }
    variants = sorted({str(row["variant"]) for row in records} - {baseline})
    deltas: list[tuple[str, str, str, float]] = []
    for variant in variants:
        grouped: dict[tuple[str, str], list[float]] = {}
        for (row_variant, case_id, seed), metrics in matched.items():
            reference = baseline_rows.get((case_id, seed))
            if row_variant != variant or reference is None:
                continue
            for metric in (
                "cost_usd", "tokens", "rounds", "check_pass_rate", "jev_noul",
                "jev_quality", "quality_per_dollar",
            ):
                if metric in metrics and metric in reference:
                    delta = metrics[metric] - reference[metric]
                    grouped.setdefault((case_id, metric), []).append(delta)
        deltas.extend(
            (variant, case_id, metric, statistics.mean(samples))
            for (case_id, metric), samples in sorted(grouped.items())
        )
    return deltas


def _format(value: float | None) -> str:
    """Format one metric value for a compact Markdown table."""
    return "n/a" if value is None else f"{value:.6g}"


def render_report(
    records: list[dict[str, Any]], assessments: list[dict[str, Any]], baseline: str
) -> str:
    """Render aggregate statistics and paired per-case deltas as Markdown."""
    summaries, matched = aggregate(records, assessments)
    lines = ["# Routed A/B report", "", f"Baseline: `{baseline}`", "",
             "| Variant | Metric | Mean | Stdev | N |", "|---|---|---:|---:|---:|"]
    for variant in sorted(summaries):
        for metric in REPORT_METRICS:
            mean, deviation = summaries[variant][metric]
            count = sum(1 for key, values in matched.items()
                        if key[0] == variant and metric in values)
            row = f"| {variant} | {metric} | {_format(mean)} | {_format(deviation)} | {count} |"
            lines.append(row)
    lines.extend(["", f"## Paired deltas by case (variant minus `{baseline}`)", "",
                  "| Variant | Case | Metric | Mean delta |", "|---|---|---|---:|"])
    lines.extend(
        f"| {variant} | {case_id} | {metric} | {delta:+.6g} |"
        for variant, case_id, metric, delta in paired_deltas(records, matched, baseline)
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Return report input and output arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--assessments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", default="fusion-pinned")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Print and save the Markdown report."""
    args = build_parser().parse_args(argv)
    report = render_report(_read_jsonl(args.records), _read_jsonl(args.assessments), args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
