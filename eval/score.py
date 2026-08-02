#!/usr/bin/env python3
"""Score eval results and write eval/report.md.

Scoring rubric (auto_score only):
- For each fixture, `expect` is a list of keywords/phrases a good answer
  should contain. Keywords are matched case-insensitively as literal
  substrings.
- `auto_score` = (number of expect terms found in response_text) / (total expect terms).
- Responses with an error field or empty response_text score 0.
- `human_score` is left null for the user to fill in later.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent


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
            auto = 0.0
            if not r.get("error"):
                auto = score_response(r.get("response_text", ""), expect)
            row = {
                **r,
                "auto_score": round(auto, 3),
                "human_score": None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Build per-config metrics
    configs = ["direct", "trinity", "conductor-old", "conductor-new"]
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_config[r["config"]].append(r)

    def config_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        scores = []
        latencies = []
        costs = []
        for r in rows:
            fx = fx_by_id.get(r["id"], {})
            expect = fx.get("expect", [])
            auto = 0.0
            if not r.get("error"):
                auto = score_response(r.get("response_text", ""), expect)
            scores.append(auto)
            latencies.append(r.get("latency_s", 0))
            costs.append(r.get("est_cost_usd", 0))
        return {
            "count": len(rows),
            "auto_score_mean": statistics.mean(scores) if scores else 0.0,
            "auto_score_median": statistics.median(scores) if scores else 0.0,
            "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
            "cost_sum_usd": sum(costs),
        }

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
            auto = 0.0
            if not r.get("error"):
                auto = score_response(r.get("response_text", ""), fx.get("expect", []))
            vals.append(auto)
        return statistics.mean(vals) if vals else 0.0

    def sum_cost(rows: list[dict[str, Any]]) -> float:
        return sum(r.get("est_cost_usd", 0) for r in rows)

    def sum_latency(rows: list[dict[str, Any]]) -> float:
        return sum(r.get("latency_s", 0) for r in rows)

    def tier_rows(rows: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
        return [r for r in rows if r.get("tier") == tier]

    # Build report
    lines: list[str] = []
    lines.append("# Comparative Eval: direct vs TRINITY vs Conductor-old vs Conductor-new\n")
    lines.append("## Summary\n")
    lines.append("| config | auto_score_mean | latency_mean_s | cost_sum_usd |")
    lines.append("|---|---|---|---|")
    for c in configs:
        m = metrics[c]
        summary_row = (
            f"| {c} | {m['auto_score_mean']:.3f} | {m['latency_mean_s']:.1f} | "
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
            r = next((x for x in by_config.get(c, []) if x["id"] == fid), None)
            if r is None:
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
    lines.append("| tier | config | auto_score_mean | latency_sum_s | cost_sum_usd |")
    lines.append("|---|---|---|---|---|")
    for tier in tiers:
        for c in configs:
            m = tier_table[tier][c]
            tier_latency = sum_latency(
                [r for r in by_config.get(c, []) if r.get("tier") == tier]
            )
            tier_row = (
                f"| {tier} | {c} | {m['auto_score_mean']:.3f} | {tier_latency:.1f} | "
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
    lines.append(f"- direct mean auto_score: {ds:.3f}")
    lines.append(f"- trinity mean auto_score: {ts:.3f}")
    qd_direct = (
        f"- quality delta: {ts - ds:+.3f} "
        f"({'+' if ts >= ds else ''}{((ts - ds) / ds * 100):.1f}% vs direct)"
    )
    lines.append(qd_direct if ds else "")
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
    lines.append(f"- conductor-old mean auto_score: {os:.3f}")
    lines.append(f"- conductor-new mean auto_score: {ns:.3f}")
    qd_old = (
        f"- quality delta: {ns - os:+.3f} "
        f"({'+' if ns >= os else ''}{((ns - os) / os * 100):.1f}% vs old)"
    )
    lines.append(qd_old if os else "")
    lines.append(f"- conductor-old total cost: ${oc:.4f}, latency: {ol:.1f}s")
    lines.append(f"- conductor-new total cost: ${nc:.4f}, latency: {nl:.1f}s")
    lines.append("")

    # c) conductor-new vs trinity
    lines.append("### c) conductor-new vs trinity\n")
    lines.append(f"- trinity mean auto_score: {ts:.3f}")
    lines.append(f"- conductor-new mean auto_score: {ns:.3f}")
    qd_trinity = (
        f"- quality delta: {ns - ts:+.3f} "
        f"({'+' if ns >= ts else ''}{((ns - ts) / ts * 100):.1f}% vs trinity)"
    )
    lines.append(qd_trinity if ts else "")
    lines.append(f"- trinity total cost: ${tc:.4f}, latency: {tl:.1f}s")
    lines.append(f"- conductor-new total cost: ${nc:.4f}, latency: {nl:.1f}s")
    lines.append("")

    # Hard-tier comparison
    lines.append("### Hard-tier comparison\n")
    for c in configs:
        rows = tier_rows(by_config.get(c, []), "hard")
        hard_line = f"- {c}: mean auto_score = {mean_score(rows):.3f}, "
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
    if ns < ds - 0.05 and new_err > 0.5:
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
        "This is a per-call estimate based on `configs/worker-costs.json`; "
        "actual OpenRouter spend may differ.\n"
    )

    Path(args.output).write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
