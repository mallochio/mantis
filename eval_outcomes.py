#!/usr/bin/env python3
"""Weekly router outcome evaluation.

Reads training.jsonl (routed calls), outcomes.jsonl (retried/refused/truncated/
upstream_error feedback) and pseudo-labels.jsonl (Gemini difficulty labels),
then reports failure rates per score band and route decision, and says whether
the data supports moving the threshold.

Usage:
    python eval_outcomes.py
    python eval_outcomes.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DATA_DIR = Path.home() / ".local/share/mantis/router"
BANDS = [("<0.1", 0.0, 0.1), ("0.1-0.156", 0.1, 0.156), ("0.156-0.2", 0.156, 0.2), (">=0.2", 0.2, 1.01)]
FAILURES = {"retried", "refused", "truncated", "upstream_error"}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def band_of(score: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= score < hi:
            return name
    return BANDS[-1][0]


def evaluate(decisions: list[dict], outcomes: list[dict], labels: list[dict]) -> dict:
    # Modern rows join one outcome occurrence to one routed request. Legacy rows
    # without IDs retain the old prompt-hash join for backward compatibility.
    failed_ids: set[str] = set()
    legacy_failed_hashes: set[str] = set()
    for outcome in outcomes:
        if outcome.get("outcome") not in FAILURES:
            continue
        occurrence = outcome.get("related_request_id") or outcome.get("request_id") or outcome.get("decision_occurrence_id")
        if occurrence:
            failed_ids.add(str(occurrence))
        elif outcome.get("prompt_hash"):
            legacy_failed_hashes.add(str(outcome["prompt_hash"]))

    lab = {}
    for label in labels:
        if "domain" in label and label.get("prompt"):
            lab[label["prompt"].strip()] = "expensive" if label.get("route") == "big model" else "cheap"

    cells = {d: {b: [0, 0] for b, _, _ in BANDS} for d in ("cheap", "expensive")}
    lab_cells = {d: {r: [0, 0] for r in ("cheap", "expensive")} for d in ("cheap", "expensive")}
    for r in decisions:
        score = r.get("score")
        if not isinstance(score, (int, float)):
            continue
        d = r.get("decision")
        if d not in cells:
            continue
        cells[d][band_of(score)][0] += 1
        row_id = r.get("request_id") or r.get("occurrence_id")
        is_failed = (str(row_id) in failed_ids if row_id else r.get("prompt_hash") in legacy_failed_hashes)
        if is_failed:
            cells[d][band_of(score)][1] += 1
        lr = lab.get(str(r.get("prompt", "")).strip())
        if lr:
            lab_cells[d][lr][0] += 1
            if is_failed:
                lab_cells[d][lr][1] += 1
    return {"cells": cells, "lab_cells": lab_cells}


def rate(cell) -> float | None:
    n, f = cell
    return f / n if n else None


def verdicts(res: dict) -> list[str]:
    c = res["cells"]
    out = []
    # cheap failing more on high-score prompts than low -> under-routing
    cheap_high = [c["cheap"][b] for b in ("0.156-0.2", ">=0.2")]
    n_high, f_high = sum(x[0] for x in cheap_high), sum(x[1] for x in cheap_high)
    n_low, f_low = c["cheap"]["<0.1"]
    if n_high >= 20 and n_low >= 20:
        rh, rl = f_high / n_high, f_low / n_low
        if rh > 2 * rl:
            out.append(f"RAISE: cheap failures at {rh:.0%} on high-score prompts vs {rl:.0%} low — move high band to expensive")
    # labeler cross: cheap routed on labeler-hard prompts failing a lot
    for band_name, key, msg in (("labeler-hard", "expensive", "cheap fails on labeler-hard prompts"),
                                ("labeler-easy", "cheap", "expensive fails on labeler-easy prompts")):
        n, f = res["lab_cells"]["cheap"][key] if band_name == "labeler-hard" else res["lab_cells"]["expensive"][key]
        if n >= 20 and rate([n, f]) and rate([n, f]) >= 0.15:
            out.append(f"CHECK: {msg}: {f}/{n} = {f/n:.0%}")
    if not out:
        out.append("No threshold move supported by data yet.")
    return out


def report(decisions, outcomes, labels) -> str:
    res = evaluate(decisions, outcomes, labels)
    lines = [
        f"Router outcome evaluation ({len(decisions)} routed calls, {len(outcomes)} outcome events)",
        "",
        "Failure rate (retried/refused/truncated/upstream_error) by score band x decision:",
    ]
    header = "  band       | cheap        | expensive"
    lines.append(header)
    lines.append("  -----------+--------------+-------------")
    for b, _, _ in BANDS:
        row = []
        for d in ("cheap", "expensive"):
            n, f = res["cells"][d][b]
            row.append(f"{f}/{n} ({rate([n, f]) or 0:.0%})" if n else "-")
        lines.append(f"  {b:10s} | {row[0]:12s} | {row[1]:11s}")
    lines.append("")
    lines.append("Failure rate by route decision x labeler call:")
    for d in ("cheap", "expensive"):
        parts = [f"{r}:{res['lab_cells'][d][r][1]}/{res['lab_cells'][d][r][0]}" for r in ("cheap", "expensive")]
        lines.append(f"  routed {d:9s} | " + "  ".join(parts))
    lines.append("")
    lines.append("Verdict:")
    lines.extend("  " + v for v in verdicts(res))
    return "\n".join(lines)


def demo() -> None:
    dec = [
        {"score": 0.05, "decision": "cheap", "prompt": "a", "prompt_hash": "h1"},
        {"score": 0.18, "decision": "cheap", "prompt": "b", "prompt_hash": "h2"},
        {"score": 0.18, "decision": "cheap", "prompt": "b", "prompt_hash": "h2"},
        {"score": 0.05, "decision": "cheap", "prompt": "c", "prompt_hash": "h3"},
    ] * 30  # 120 calls: high band mostly failing
    out = [{"prompt_hash": "h2", "outcome": "retried"}, {"prompt_hash": "h2", "outcome": "retried"}]
    labs = [{"domain": "x", "prompt": "b", "route": "big model"}, {"domain": "x", "prompt": "a", "route": "small model"}]
    v = verdicts(evaluate(dec, out, labs))
    assert any(x.startswith("RAISE") for x in v), v
    assert not any(x.startswith("RAISE") for x in verdicts(evaluate(dec, [], labs)))
    print("self-test passed:", v[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = ap.parse_args()
    if args.self_test:
        demo()
        sys.exit(0)
    print(report(
        load_jsonl(args.data_dir / "training.jsonl"),
        load_jsonl(args.data_dir / "outcomes.jsonl"),
        load_jsonl(args.data_dir / "pseudo-labels.jsonl"),
    ))
