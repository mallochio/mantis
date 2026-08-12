#!/usr/bin/env python3
"""A/B comparison harness: mantis (local) vs sakana/fugu-ultra (OpenRouter).

Runs identical fixtures through both targets, records latency, usage, and
real cost, then (with --summarize) writes eval/report-ab.md with quality,
cost, and latency aggregates per target and tier.

Usage:
    uv run python eval/ab_compare.py --targets mantis,openrouter-fugu \
        --output eval/results-ab.jsonl
    uv run python eval/ab_compare.py --summarize --results eval/results-ab.jsonl

Cost: OpenRouter's fugu-ultra response carries usage.cost (real USD). Mantis
reports usage.cost from its price map (P0) and usage.fugu_trace (internal
worker calls) when available; when cost is absent (older builds) the cost
column stays null so the summary compares only what both sides report.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from score import provenance, score_response

REPO = Path(__file__).resolve().parent.parent

MANTIS_DEFAULTS = {
    "url": "http://127.0.0.1:8088/v1/chat/completions",
    "key_env": "MANTIS_API_KEY",
    "model": "mantis",
}
FUGU_DEFAULTS = {
    "url": "https://openrouter.ai/api/v1/chat/completions",
    "key_env": "OPENROUTER_API_KEY",
    "model": "sakana/fugu-ultra",
}


def run_target(name: str, cfg: dict[str, str], fx: dict[str, Any], timeout: float) -> dict[str,
            Any]:
    rec: dict[str, Any] = {
        "id": fx["id"],
        "target": name,
        "tier": fx["tier"],
        "prompt": fx["prompt"],
        "response_text": "",
        "latency_s": 0.0,
        "usage": None,
        "cost_usd": None,
        "error": None,
    }
    payload: dict[str, Any] = {
        "model": cfg["model"],
        "messages": [{"role": "user", "content": fx["prompt"]}],
    }
    if name != "openrouter-fugu":
        payload["max_tokens"] = 1024
    # fugu-ultra: mandatory reasoning at its default effort (xhigh); do not
    # send temperature/max_tokens — the model rejects unsupported params.
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ[cfg['key_env']]}",
    }
    start = time.time()
    try:
        r = requests.post(cfg["url"], json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        rec["latency_s"] = round(time.time() - start, 3)
        rec["response_text"] = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        )
        usage = data.get("usage") or {}
        rec["usage"] = {k: usage[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        if k in usage}
        if isinstance(usage.get("cost"), (int, float)):
            rec["cost_usd"] = round(float(usage["cost"]), 6)
    except requests.exceptions.Timeout:
        rec["latency_s"] = round(time.time() - start, 3)
        rec["error"] = "timeout"
    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        rec["latency_s"] = round(time.time() - start, 3)
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(results: list[dict[str, Any]], output: str) -> None:
    targets = sorted({r["target"] for r in results})
    by_target: dict[str, list[dict[str, Any]]] = {t: [] for t in targets}
    for r in results:
        by_target[r["target"]].append(r)

    lines = provenance(
        config=",".join(targets),
        fixtures_path=REPO / "eval" / "fixtures.jsonl",
        cost_method="provider-reported usage.cost (real USD when supplied)",
    )
    lines.extend(["# A/B: mantis vs sakana/fugu-ultra", ""])
    lines.append(
        f"Fixtures: {len(results) // max(len(targets), 1)} tasks x {len(targets)} targets. "
        "Scoring: keyword hit rate on `expect` terms (eval/score.py)."
    )
    lines.append("")
    header = (
        "| target | success | errors | score | cost_usd | score/$ | latency s (med/p95) | "
        "completion tokens (med) |"
    )
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---|---:|")
    for t in targets:
        rs = by_target[t]
        ok = [r for r in rs if not r["error"]]
        scores = [score_response(r["response_text"], _expect(r)) for r in ok]
        costs = [r["cost_usd"] for r in ok if r["cost_usd"] is not None]
        lats = [r["latency_s"] for r in ok]
        comps = [r["usage"]["completion_tokens"] for r in ok if r.get("usage")]
        mean_score = statistics.mean(scores) if scores else 0.0
        total_cost = sum(costs)
        eff = mean_score / total_cost if total_cost else float("nan")
        lines.append(
            f"| {t} | {len(ok)}/{len(rs)} | {len(rs) - len(ok)} | {mean_score:.2f} | "
            f"{total_cost:.4f} | "
            f"{eff:.1f} | {statistics.median(lats) if lats else 0:.1f}/{pct(lats, 0.95):.1f} | "
            f"{int(statistics.median(comps)) if comps else 0} |"
        )
    lines.append("")
    lines.append("## Per tier (mean score / total cost $)")
    lines.append("")
    tiers = sorted({r["tier"] for r in results})
    for tier in tiers:
        lines.append(f"### {tier}")
        lines.append("")
        lines.append("| target | success | errors | score | cost_usd | latency med s |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for t in targets:
            all_rs = [r for r in by_target[t] if r["tier"] == tier]
            rs = [r for r in all_rs if not r["error"]]
            if not all_rs:
                continue
            scores = [score_response(r["response_text"], _expect(r)) for r in rs]
            costs = [r["cost_usd"] for r in rs if r["cost_usd"] is not None]
            lats = [r["latency_s"] for r in rs]
            lines.append(
                f"| {t} | {len(rs)}/{len(all_rs)} | {len(all_rs) - len(rs)} | "
                f"{statistics.mean(scores):.2f} | {sum(costs):.4f} | "
                f"{statistics.median(lats):.1f} |"
            )
        lines.append("")
    Path(output).write_text("\n".join(lines))
    print(f"wrote {output}")


def _expect(rec: dict[str, Any]) -> list[str]:
    """Expect terms are not stored in results; reload from fixtures."""
    with open(REPO / "eval" / "fixtures.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                fx = json.loads(line)
                if fx["id"] == rec["id"]:
                    return fx.get("expect", [])
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B: mantis vs fugu-ultra.")
    parser.add_argument("--targets", default="mantis,openrouter-fugu")
    parser.add_argument("--fixtures", default=str(REPO / "eval" / "fixtures.jsonl"))
    parser.add_argument("--output", default=str(REPO / "eval" / "results-ab.jsonl"))
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--mantis-url", default=MANTIS_DEFAULTS["url"])
    parser.add_argument("--mantis-model", default=MANTIS_DEFAULTS["model"])
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--results", default=str(REPO / "eval" / "results-ab.jsonl"))
    parser.add_argument("--report", default=str(REPO / "eval" / "report-ab.md"))
    args = parser.parse_args()

    if args.summarize:
        results = [json.loads(line) for line in Path(args.results).read_text().splitlines() if
    line.strip()]
        summarize(results, args.report)
        return

    targets = {}
    for name in (t.strip() for t in args.targets.split(",") if t.strip()):
        if name == "mantis":
            targets[name] = {
                "url": args.mantis_url, "key_env": "MANTIS_API_KEY", "model": args.mantis_model
            }
        elif name == "openrouter-fugu":
            targets[name] = dict(FUGU_DEFAULTS)
        else:
            raise SystemExit(f"unknown target: {name}")
    for cfg in targets.values():
        if cfg["key_env"] not in os.environ:
            raise SystemExit(f"{cfg['key_env']} is not set")

    fixtures = [
        json.loads(line)
        for line in Path(args.fixtures).read_text().splitlines()
        if line.strip()
    ]
    out_path = Path(args.output)
    out_path.write_text("")  # fresh run

    for fx in fixtures:
        for name, cfg in targets.items():
            rec = run_target(name, cfg, fx, args.timeout)
            with open(out_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{name}] {fx['id']} lat={rec['latency_s']} "
                  f"cost={rec['cost_usd']} err={rec['error']}", flush=True)


if __name__ == "__main__":
    main()
