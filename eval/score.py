#!/usr/bin/env python3
"""Score eval results and write eval/report.md.

Scoring rubric (auto_score only):
- For each fixture, `expect` is a list of keywords/phrases a good answer
  should contain. Keywords are matched case-insensitively as literal
  substrings.
- `auto_score` = (number of expect terms found in response_text) / (total expect terms).
- Responses with an error field are excluded from aggregate metrics.
- `human_score` is left null for the user to fill in later.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
MIN_SUCCESSFUL_ROWS = 4


def provenance(
    *,
    config: str,
    fixtures_path: Path,
    cost_method: str,
    generated_at: datetime | None = None,
) -> list[str]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unknown"
    try:
        catalog = tomllib.loads((REPO / "config" / "catalog.toml").read_text())
        revision = catalog["base"]["revision"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        revision = "unknown"
    fixture_sha = sha256(fixtures_path.read_bytes()).hexdigest()
    try:
        fixture_display = fixtures_path.relative_to(REPO)
    except ValueError:
        fixture_display = fixtures_path
    timestamp = generated_at or datetime.now(UTC)
    return [
        "## Provenance",
        f"- config/arm: `{config}`",
        f"- git SHA: `{git_sha}`",
        f"- catalog revision: `{revision}`",
        f"- fixtures: `{fixture_display}` (sha256 `{fixture_sha}`)",
        f"- generated at: `{timestamp.isoformat()}`",
        f"- cost method: `{cost_method}`",
        "",
    ]


def score_response(response_text: str, expect: list[str]) -> float:
    if not response_text:
        return 0.0
    text = response_text.lower()
    found = sum(1 for kw in expect if kw.lower() in text)
    return found / len(expect) if expect else 0.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt(x: float) -> str:
    return f"{x:.2f}"


def aggregate_metrics(
    rows: list[dict[str, Any]], fx_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("error")]
    scores = [
        score_response(r.get("response_text", ""), fx_by_id.get(r["id"], {}).get("expect", []))
        for r in valid
    ]
    latencies = [float(r.get("latency_s", 0) or 0) for r in valid]
    costs = [float(r.get("est_cost_usd", 0) or 0) for r in valid]
    return {
        "count": len(rows),
        "success_count": len(valid),
        "error_count": len(rows) - len(valid),
        "auto_score_mean": statistics.mean(scores) if scores else 0.0,
        "auto_score_median": statistics.median(scores) if scores else 0.0,
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
        "cost_sum_usd": sum(costs),
    }


def metric_display(
    metrics: dict[str, Any], key: str = "auto_score_mean", digits: int = 3
) -> str:
    success = metrics["success_count"]
    total = metrics["count"]
    if success < MIN_SUCCESSFUL_ROWS:
        return f"n/a ({success}/{total} successful)"
    return f"{metrics[key]:.{digits}f}"


def paired_scores(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    fx_by_id: dict[str, dict[str, Any]],
) -> tuple[list[float], list[float]]:
    left = {r["id"]: r for r in left_rows if not r.get("error")}
    right = {r["id"]: r for r in right_rows if not r.get("error")}
    ids = left.keys() & right.keys()
    left_scores = [
        score_response(left[i].get("response_text", ""), fx_by_id.get(i, {}).get("expect", []))
        for i in ids
    ]
    right_scores = [
        score_response(right[i].get("response_text", ""), fx_by_id.get(i, {}).get("expect", []))
        for i in ids
    ]
    return left_scores, right_scores


def paired_comparable(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    fx_by_id: dict[str, dict[str, Any]],
) -> tuple[list[float], list[float]]:
    left_scores, right_scores = paired_scores(left_rows, right_rows, fx_by_id)
    if len(left_scores) < MIN_SUCCESSFUL_ROWS:
        return [], []
    return left_scores, right_scores


def paired_success_count(
    left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]]
) -> int:
    left = {r["id"] for r in left_rows if not r.get("error")}
    right = {r["id"] for r in right_rows if not r.get("error")}
    return len(left & right)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score eval results.")
    parser.add_argument("--fixtures", default=str(REPO / "eval" / "fixtures.jsonl"))
    parser.add_argument("--results", default=str(REPO / "eval" / "results.jsonl"))
    parser.add_argument("--output", default=str(REPO / "eval" / "report.md"))
    args = parser.parse_args()

    fixtures = load_jsonl(Path(args.fixtures))
    fx_by_id = {f["id"]: f for f in fixtures}

    results = load_jsonl(Path(args.results))

    # Score and write scored results
    scored_path = Path(args.results).with_stem(Path(args.results).stem + "-scored")
    with open(scored_path, "w") as out:
        for r in results:
            fx = fx_by_id.get(r["id"], {})
            expect = fx.get("expect", [])
            auto = None if r.get("error") else score_response(r.get("response_text", ""), expect)
            row = {
                **r,
                "auto_score": round(auto, 3) if auto is not None else None,
                "human_score": None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build per-config metrics
    configs = ["direct", "trinity", "conductor-old", "conductor-new"]
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    def config_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregate_metrics(rows, fx_by_id)

    metrics = {c: config_metrics(by_config.get(c, [])) for c in configs}

    # Per-tier averages
    tiers = ["simple", "medium", "hard", "debug"]
    tier_table: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        tier_table[tier] = {}
        for c in configs:
            rows = [r for r in by_config.get(c, []) if r.get("tier") == tier]
            tier_table[tier][c] = config_metrics(rows)

    # Headline comparisons (we will use string builders later)
    direct_rows = by_config.get("direct", [])
    trinity_rows = by_config.get("trinity", [])
    old_rows = by_config.get("conductor-old", [])
    new_rows = by_config.get("conductor-new", [])

    def mean_score(rows: list[dict[str, Any]]) -> float:
        vals = []
        for r in rows:
            fx = fx_by_id.get(r["id"], {})
            if not r.get("error"):
                vals.append(score_response(r.get("response_text", ""), fx.get("expect", [])))
        return statistics.mean(vals) if vals else 0.0

    def sum_cost(rows: list[dict[str, Any]]) -> float:
        return sum(float(r.get("est_cost_usd", 0) or 0) for r in rows if not r.get("error"))

    def sum_latency(rows: list[dict[str, Any]]) -> float:
        return sum(float(r.get("latency_s", 0) or 0) for r in rows if not r.get("error"))

    def tier_rows(rows: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("tier") == tier]

    # Build report
    lines: list[str] = provenance(
        config=",".join(configs),
        fixtures_path=Path(args.fixtures),
        cost_method="native estimated prices; 2K prompt + 1K completion assumption",
    )
    lines.append("# Comparative Eval: direct vs TRINITY vs Conductor-old vs Conductor-new\n")
    lines.append(
        f"Scoring rule: report a mean only with at least {MIN_SUCCESSFUL_ROWS} "
        "successful rows; comparisons use successful-item intersections.\n"
    )
    lines.append(
        "Scoring note: an empty response without an error scores 0.0 under the "
        "current keyword scorer; row output does not distinguish an empty answer "
        "from a wrong answer.\n"
    )
    lines.append("## Summary\n")
    lines.append(
        "| config | success | errors | auto_score_mean | latency_mean_s | est_cost_sum_usd |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in configs:
        m = metrics[c]
        summary_row = (
            f"| {c} | {m['success_count']}/{m['count']} | {m['error_count']} | "
            f"{metric_display(m)} | {metric_display(m, 'latency_mean_s', 1)} | "
            f"${m['cost_sum_usd']:.4f} |"
        )
        lines.append(summary_row)
    lines.append("")

    # Results matrix
    lines.append("## Results matrix (auto_score, latency, cost)\n")
    lines.append("| id | tier | direct | trinity | conductor-old | conductor-new |")
    lines.append("|---|---|---|---|---|---|")
    for fx in fixtures:
        fid = fx["id"]
        row_cells = [fid, fx["tier"]]
        for c in configs:
            r = next((x for x in by_config.get(c, []) if x["id"] == fid), {})
            if not r:
                row_cells.append("-")
            elif r.get("error"):
                row_cells.append(f"error: {r['error']}")
            else:
                auto = score_response(r.get("response_text", ""), fx.get("expect", []))
                row_cells.append(f"{auto:.2f} / {r['latency_s']:.1f}s / ${r['est_cost_usd']:.4f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")

    # Per-tier averages
    lines.append("## Per-tier averages\n")
    lines.append(
        "| tier | config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for tier in tiers:
        for c in configs:
            m = tier_table[tier][c]
            tier_latency = sum_latency(
                [r for r in by_config.get(c, []) if r.get("tier") == tier]
            )
            tier_row = (
                f"| {tier} | {c} | {m['success_count']}/{m['count']} | {m['error_count']} | "
                f"{metric_display(m)} | {tier_latency:.1f} | "
                f"${m['cost_sum_usd']:.4f} |"
            )
            lines.append(tier_row)
    lines.append("")

    # Headline comparisons
    lines.append("## Headline comparisons\n")

    # a) trinity vs direct
    ds = mean_score(direct_rows)
    ts = mean_score(trinity_rows)
    dc = sum_cost(direct_rows)
    tc = sum_cost(trinity_rows)
    dl = sum_latency(direct_rows)
    tl = sum_latency(trinity_rows)
    lines.append("### a) trinity vs direct\n")
    direct_trinity_left, direct_trinity_right = paired_comparable(
        direct_rows, trinity_rows, fx_by_id
    )
    lines.append(f"- direct mean auto_score: {metric_display(metrics['direct'])}")
    lines.append(f"- trinity mean auto_score: {metric_display(metrics['trinity'])}")
    if direct_trinity_left:
        paired_ds = statistics.mean(direct_trinity_left)
        paired_ts = statistics.mean(direct_trinity_right)
        lines.append(
            f"- quality delta (paired n={len(direct_trinity_left)}): "
            f"{paired_ts - paired_ds:+.3f}"
        )
        if paired_ds:
            lines.append(
                f"- quality delta percentage (paired n={len(direct_trinity_left)}): "
                f"{'+' if paired_ts >= paired_ds else ''}"
                f"{((paired_ts - paired_ds) / paired_ds * 100):.1f}% vs direct"
            )
    else:
        lines.append(
            f"- paired successful intersection: "
            f"n={paired_success_count(direct_rows, trinity_rows)} "
            f"(<{MIN_SUCCESSFUL_ROWS}); quality comparison omitted"
        )
    lines.append(f"- direct total cost: ${dc:.4f}, latency: {dl:.1f}s")
    lines.append(f"- trinity total cost: ${tc:.4f}, latency: {tl:.1f}s")
    cd_direct = f"- cost delta: ${tc - dc:+.4f} " f"({((tc - dc) / dc * 100):.1f}% vs direct)"
    lines.append(cd_direct if dc else "")
    lines.append("")

    # b) conductor-new vs conductor-old
    os = mean_score(old_rows)
    ns = mean_score(new_rows)
    oc = sum_cost(old_rows)
    nc = sum_cost(new_rows)
    ol = sum_latency(old_rows)
    nl = sum_latency(new_rows)
    lines.append("### b) conductor-new vs conductor-old\n")
    old_new_left, old_new_right = paired_comparable(old_rows, new_rows, fx_by_id)
    lines.append(f"- conductor-old mean auto_score: {metric_display(metrics['conductor-old'])}")
    lines.append(f"- conductor-new mean auto_score: {metric_display(metrics['conductor-new'])}")
    if old_new_left:
        paired_os = statistics.mean(old_new_left)
        paired_ns = statistics.mean(old_new_right)
        lines.append(
            f"- quality delta (paired n={len(old_new_left)}): "
            f"{paired_ns - paired_os:+.3f}"
        )
        if paired_os:
            lines.append(
                f"- quality delta percentage (paired n={len(old_new_left)}): "
                f"{'+' if paired_ns >= paired_os else ''}"
                f"{((paired_ns - paired_os) / paired_os * 100):.1f}% vs old"
            )
    else:
        lines.append(
            f"- paired successful intersection: "
            f"n={paired_success_count(old_rows, new_rows)} "
            f"(<{MIN_SUCCESSFUL_ROWS}); quality comparison omitted"
        )
    lines.append(f"- conductor-old total cost: ${oc:.4f}, latency: {ol:.1f}s")
    lines.append(f"- conductor-new total cost: ${nc:.4f}, latency: {nl:.1f}s")
    lines.append("")

    # c) conductor-new vs trinity
    lines.append("### c) conductor-new vs trinity\n")
    new_trinity_left, new_trinity_right = paired_comparable(
        new_rows, trinity_rows, fx_by_id
    )
    lines.append(f"- trinity mean auto_score: {metric_display(metrics['trinity'])}")
    lines.append(f"- conductor-new mean auto_score: {metric_display(metrics['conductor-new'])}")
    if new_trinity_left:
        paired_ns = statistics.mean(new_trinity_left)
        paired_ts = statistics.mean(new_trinity_right)
        lines.append(
            f"- quality delta (paired n={len(new_trinity_left)}): "
            f"{paired_ns - paired_ts:+.3f}"
        )
        if paired_ts:
            lines.append(
                f"- quality delta percentage (paired n={len(new_trinity_left)}): "
                f"{'+' if paired_ns >= paired_ts else ''}"
                f"{((paired_ns - paired_ts) / paired_ts * 100):.1f}% vs trinity"
            )
    else:
        lines.append(
            f"- paired successful intersection: "
            f"n={paired_success_count(new_rows, trinity_rows)} "
            f"(<{MIN_SUCCESSFUL_ROWS}); quality comparison omitted"
        )
    lines.append(f"- trinity total cost: ${tc:.4f}, latency: {tl:.1f}s")
    lines.append(f"- conductor-new total cost: ${nc:.4f}, latency: {nl:.1f}s")
    lines.append("")

    # Hard-tier comparison
    lines.append("### Hard-tier comparison\n")
    for c in configs:
        rows = tier_rows(by_config.get(c, []), "hard")
        m = tier_table["hard"][c]
        hard_line = (
            f"- {c}: mean auto_score ({m['success_count']} successful, {m['error_count']} errors) "
            f"= {metric_display(m)}, "
        )
        hard_line += f"cost = ${sum_cost(rows):.4f}"
        lines.append(hard_line)
    lines.append("")

    # Recommendation
    lines.append("## Recommendation\n")

    # Compute error rates for context
    def error_rate(rows: list[dict[str, Any]]) -> float:
        return sum(1 for r in rows if r.get("error")) / len(rows) if rows else 0.0

    new_err = error_rate(new_rows)

    # Decision rule from task, plus a guard for the failure case observed here
    if not new_trinity_left or not paired_comparable(
        direct_rows, new_rows, fx_by_id
    )[0]:
        rec = (
            "Insufficient comparable successful rows for a Conductor quality "
            "comparison. Recommendation: collect at least "
            f"{MIN_SUCCESSFUL_ROWS} successful rows per arm with matched counts."
        )
    elif ns < ds - 0.05 and new_err > 0.5:
        rec = (
            f"Conductor-new scores well below direct ({ns:.3f} vs {ds:.3f}) "
            f"and fails on {new_err:.0%} of prompts. Recommendation: do not "
            "deploy the local Conductor checkpoints as-is; use TRINITY in "
            "`/fugu auto` mode, and do not spend more on Conductor training "
            "until the checkpoint reliably emits valid DAGs on CPU/float32 serving."
        )
    elif abs(ns - ts) < 0.05 and abs(os - ts) < 0.05:
        rec = (
            "All three orchestrated configs perform within 0.05 auto_score "
            "of each other. Recommendation: keep using TRINITY in `/fugu auto` "
            "mode and stop further Conductor training spend."
        )
    elif (
        ns > os + 0.05
        and tier_table["hard"]["conductor-new"]["auto_score_mean"]
        > tier_table["hard"]["conductor-old"]["auto_score_mean"]
    ):
        rec = (
            "Conductor-new is meaningfully better than conductor-old, "
            "especially on hard tasks. Recommendation: continue with "
            "outcome-reward / more-steps tuning rather than full retraining "
            "from scratch."
        )
    elif ns > ds and ns < ts:
        rec = (
            "Conductor-new beats direct but not TRINITY. Recommendation: "
            "use conductor only for hard-tier prompts where a multi-step "
            "DAG is expected to help; keep TRINITY as default."
        )
    else:
        rec = (
            "Results are mixed. Recommendation: run a targeted hard-tier "
            "eval with more prompts and human scoring before committing to "
            "more Conductor training."
        )
    lines.append(rec)
    lines.append("")

    # Error list
    lines.append("## Errors / timeouts\n")
    errors = [r for r in results if r.get("error")]
    if not errors:
        lines.append("No errors or timeouts recorded.\n")
    else:
        lines.extend(f"- `{r['config']}` / `{r['id']}`: {r['error']}" for r in errors)
        lines.append("")

    # Fixture list
    lines.append("## Fixture list\n")
    lines.extend(
        f"- `{fx['id']}` ({fx['tier']}): {fx['prompt'][:80].replace(chr(10), ' ')}..."
        for fx in fixtures
    )
    lines.append("")

    # Total spend
    total_cost = sum(r.get("est_cost_usd", 0) for r in results)
    lines.append("## Total spend\n")
    lines.append(f"Estimated total API spend across all configs: **${total_cost:.4f}**.")
    lines.append(
        "This is a per-call estimate based on `config/worker-costs.json`; "
        "actual OpenRouter spend may differ.\n"
    )

    Path(args.output).write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
