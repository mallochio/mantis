#!/usr/bin/env python3
"""Generate eval/report-luna-conductor.md from tracked raw results.

Combines native-v2 direct/trinity results with a fresh
`eval/results-luna.jsonl` run of `conductor-luna`.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from score import load_jsonl, provenance, score_response

REPO = Path(__file__).resolve().parent.parent


def config_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if not r.get("error")]
    scores = []
    latencies = []
    costs = []
    for r in valid:
        scores.append(r.get("auto_score", 0.0) or 0.0)
        latencies.append(r.get("latency_s", 0) or 0)
        costs.append(r.get("est_cost_usd", 0) or 0)
    return {
        "count": len(rows),
        "success_count": len(valid),
        "error_count": len(rows) - len(valid),
        "auto_score_mean": statistics.mean(scores) if scores else 0.0,
        "latency_sum_s": sum(latencies),
        "cost_sum_usd": sum(costs),
    }


def wrap(s: str, width: int = 100) -> str:
    """Simple word-wrap for f-strings that would exceed ruff's line length."""
    parts = s.split(" ")
    lines: list[str] = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) + 1 > width:
            lines.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        lines.append(cur)
    return " ".join(lines)


def main() -> None:
    fixtures = load_jsonl(REPO / "eval" / "fixtures.jsonl")
    fx_by_id = {f["id"]: f for f in fixtures}
    tiers = ["simple", "medium", "hard", "debug"]

    baseline_raw = load_jsonl(REPO / "eval" / "results-native-v2.jsonl")
    baseline: list[dict[str, Any]] = []
    for r in baseline_raw:
        fx = fx_by_id.get(r["id"], {})
        auto = (
            None
            if r.get("error")
            else score_response(r.get("response_text", ""), fx.get("expect", []))
        )
        baseline.append(
            {
                **r,
                "auto_score": round(auto, 3) if auto is not None else None,
            }
        )
    direct_rows = [r for r in baseline if r["config"] == "direct"]
    trinity_rows = [r for r in baseline if r["config"] == "trinity"]

    luna_path = REPO / "eval" / "results-luna.jsonl"
    luna_raw = load_jsonl(luna_path) if luna_path.exists() else []
    luna_rows: list[dict[str, Any]] = []
    for r in luna_raw:
        fx = fx_by_id.get(r["id"], {})
        expect = fx.get("expect", [])
        auto = None if r.get("error") else score_response(r.get("response_text", ""), expect)
        row = {
            **r,
            "config": "conductor-luna",
            "auto_score": round(auto, 3) if auto is not None else None,
        }
        luna_rows.append(row)

    # Write scored luna results
    scored_luna_path = REPO / "eval" / "results-luna-scored.jsonl"
    with open(scored_luna_path, "w") as out:
        for r in luna_rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_config = {
        "direct": direct_rows,
        "trinity": trinity_rows,
        "conductor-luna": luna_rows,
    }
    configs = ["direct", "trinity", "conductor-luna"]

    metrics = {c: config_metrics(rows) for c, rows in by_config.items()}

    tier_table: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        tier_table[tier] = {}
        for c in configs:
            rows = [r for r in by_config[c] if r.get("tier") == tier]
            tier_table[tier][c] = config_metrics(rows)

    lines: list[str] = provenance(
        config="direct,trinity,conductor-luna",
        fixtures_path=REPO / "eval" / "fixtures.jsonl",
        cost_method="native estimated prices; 2K prompt + 1K completion assumption",
    )
    lines.append("# Comparative Eval: direct vs TRINITY vs Conductor-Luna\n")
    lines.append(
        "Conductor-Luna uses the LiteLLM planner `gpt-5.6-luna-max` "
        "instead of a local 3B checkpoint.\n"
    )

    lines.append("## Summary\n")
    lines.append(
        "| config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for c in configs:
        m = metrics[c]
        lines.append(
            f"| {c} | {m['success_count']}/{m['count']} | {m['error_count']} | "
            f"{m['auto_score_mean']:.3f} | {m['latency_sum_s']:.1f} | "
            f"${m['cost_sum_usd']:.4f} |"
        )
    lines.append("")

    lines.append("## Results matrix (auto_score, latency, cost)\n")
    lines.append("| id | tier | direct | trinity | conductor-luna |")
    lines.append("|---|---|---|---|---|")
    for fx in fixtures:
        fid = fx["id"]
        cells = [fid, fx["tier"]]
        for c in configs:
            r = next((x for x in by_config[c] if x["id"] == fid), {})
            if not r:
                cells.append("-")
            elif r.get("error"):
                err = str(r["error"])[:60]
                if len(str(r["error"])) > 60:
                    err += "..."
                cells.append(f"error: {err}")
            else:
                cells.append(
                    f"{r['auto_score']:.2f} / {r['latency_s']:.1f}s / "
                    f"${r['est_cost_usd']:.4f}"
                )
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-tier averages\n")
    lines.append(
        "| tier | config | success | errors | auto_score_mean | latency_sum_s | est_cost_sum_usd |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for tier in tiers:
        for c in configs:
            m = tier_table[tier][c]
            lines.append(
                f"| {tier} | {c} | {m['success_count']}/{m['count']} | {m['error_count']} | "
                f"{m['auto_score_mean']:.3f} | {m['latency_sum_s']:.1f} | "
                f"${m['cost_sum_usd']:.4f} |"
            )
    lines.append("")

    # Headline comparisons
    trinity_overall = metrics["trinity"]["auto_score_mean"]
    luna_overall = metrics["conductor-luna"]["auto_score_mean"]
    trinity_cost = metrics["trinity"]["cost_sum_usd"]
    luna_cost = metrics["conductor-luna"]["cost_sum_usd"]

    hard_trinity = tier_table["hard"]["trinity"]["auto_score_mean"]
    hard_luna = tier_table["hard"]["conductor-luna"]["auto_score_mean"]
    hard_trinity_cost = tier_table["hard"]["trinity"]["cost_sum_usd"]
    hard_luna_cost = tier_table["hard"]["conductor-luna"]["cost_sum_usd"]

    luna_total = len(luna_rows)
    luna_failures = sum(1 for r in luna_rows if r.get("error"))
    failure_rate = luna_failures / luna_total if luna_total else 0.0

    lines.append("## Headline comparisons\n")

    lines.append("### a) conductor-luna vs trinity\n")
    lines.append(
        f"- trinity overall mean ({metrics['trinity']['success_count']} successful, "
        f"{metrics['trinity']['error_count']} errors): {trinity_overall:.3f}"
    )
    lines.append(
        f"- conductor-luna overall mean ({metrics['conductor-luna']['success_count']} successful, "
        f"{metrics['conductor-luna']['error_count']} errors): {luna_overall:.3f}"
    )
    lines.append(f"- overall quality delta: {luna_overall - trinity_overall:+.3f}")
    lines.append(
        f"- hard-tier trinity mean ({tier_table['hard']['trinity']['success_count']} successful, "
        f"{tier_table['hard']['trinity']['error_count']} errors): {hard_trinity:.3f}"
    )
    lines.append(
        f"- hard-tier conductor-luna mean "
        f"({tier_table['hard']['conductor-luna']['success_count']} successful, "
        f"{tier_table['hard']['conductor-luna']['error_count']} errors): {hard_luna:.3f}"
    )
    lines.append(f"- hard-tier quality delta: {hard_luna - hard_trinity:+.3f}")
    lines.append("")

    lines.append("### b) cost-per-quality-point\n")
    score_delta = luna_overall - trinity_overall
    cost_delta = luna_cost - trinity_cost
    if score_delta > 0 and cost_delta >= 0:
        cpp = cost_delta / score_delta if score_delta else 0.0
        lines.append(
            wrap(
                "- overall: "
                f"${cost_delta:.4f} extra spend for +{score_delta:.3f} quality "
                f"= ${cpp:.4f} per quality point"
            )
        )
    elif score_delta > 0 and cost_delta < 0:
        lines.append(
            wrap(
                "- overall: conductor-luna is both cheaper "
                f"(save ${-cost_delta:.4f}) and better (+{score_delta:.3f}), "
                "so cost per quality point is negative"
            )
        )
    else:
        lines.append(
            wrap(
                "- overall: conductor-luna is not a quality win "
                f"(delta {score_delta:+.3f}) despite ${cost_delta:+.4f} cost delta"
            )
        )

    hard_score_delta = hard_luna - hard_trinity
    hard_cost_delta = hard_luna_cost - hard_trinity_cost
    if hard_score_delta > 0 and hard_cost_delta >= 0:
        hard_cpp = hard_cost_delta / hard_score_delta if hard_score_delta else 0.0
        lines.append(
            wrap(
                "- hard tier: "
                f"${hard_cost_delta:.4f} extra spend for +{hard_score_delta:.3f} "
                f"quality = ${hard_cpp:.4f} per quality point"
            )
        )
    elif hard_score_delta > 0 and hard_cost_delta < 0:
        lines.append(
            wrap(
                "- hard tier: conductor-luna is both cheaper "
                f"(save ${-hard_cost_delta:.4f}) and better "
                f"(+{hard_score_delta:.3f})"
            )
        )
    else:
        lines.append(
            wrap(
                "- hard tier: conductor-luna is not a quality win "
                f"(delta {hard_score_delta:+.3f}) on hard tasks"
            )
        )
    lines.append("")

    lines.append("### c) failure rate\n")
    lines.append(
        f"- conductor-luna HTTP 500/timeout count: {luna_failures} / {luna_total}"
    )
    lines.append(f"- failure rate: {failure_rate:.1%}")
    lines.append("")

    lines.append("## Decision table\n")
    if hard_luna >= hard_trinity + 0.10:
        decision = "conductor-luna hard-tier mean >= trinity + 0.10"
        threshold = 4
        reason = (
            "LiteLLM-planned Conductor meaningfully outperforms TRINITY on "
            "hard tasks; enable it in `/fugu auto` with threshold 4."
        )
    elif hard_luna < hard_trinity - 0.10:
        decision = "conductor-luna hard-tier mean < trinity - 0.10"
        threshold = 6
        reason = (
            "TRINITY remains the better orchestrator for hard tasks; "
            "keep Conductor out of auto mode."
        )
    else:
        decision = "conductor-luna within ±0.10 of trinity on hard tier"
        threshold = 6
        reason = (
            "No clear win demonstrated; Conductor is manual-only for curiosity."
        )

    lines.append(f"- Decision branch: {decision}")
    lines.append(f"- Recommended Conductor routing threshold: {threshold}")
    lines.append(f"- Reasoning: {reason}")
    if failure_rate > 0.10:
        lines.append(
            wrap(
                "- Additional note: "
                f"{failure_rate:.0%} failure rate is too high for auto "
                "deployment even if quality were comparable."
            )
        )
    lines.append("")

    lines.append("## Errors / timeouts\n")
    errors = [r for r in luna_rows if r.get("error")]
    if not errors:
        lines.append("No errors or timeouts recorded for conductor-luna.\n")
    else:
        for r in errors:
            err = str(r["error"])[:120]
            lines.append(f"- `{r['id']}` ({r.get('tier')}): {err}")
        lines.append("")

    lines.append("## Total spend\n")
    lines.append(f"- conductor-luna estimated API spend: **${luna_cost:.4f}**.")
    lines.append(
        f"- direct baseline (for reference): "
        f"**${metrics['direct']['cost_sum_usd']:.4f}**."
    )
    lines.append(
        f"- trinity baseline (for reference): "
        f"**${metrics['trinity']['cost_sum_usd']:.4f}**."
    )
    lines.append(
        "Costs are per-call estimates from `config/worker-costs.json` "
        "(2K prompt + 1K completion); actual OpenRouter spend may differ.\n"
    )

    lines.append("## Fixture list\n")
    for fx in fixtures:
        prompt = fx["prompt"][:80].replace("\n", " ")
        lines.append(f"- `{fx['id']}` ({fx['tier']}): {prompt}...")
    lines.append("")

    out_path = REPO / "eval" / "report-luna-conductor.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
