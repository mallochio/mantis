"""Calibrate the Switchyard stage-router confidence threshold from paired runs.

Pure-tier runs bypass ``mantis/base`` (direct Modal GLM and Bedrock Grok)
so the router does not contaminate its own baseline. Each row records the
task outcome, the billed cost, and the escalation score observed on the
efficient probe run. This module classifies RESCUE/LOSS/SAFE/HARD
quadrants and sweeps candidate thresholds with a budget-first rule.

Row format (JSONL, one object per line)::

    {"instance_id": "t01", "tier": "efficient", "resolved": true,
     "cost_usd": 0.04, "score": 0.62}

``record_type`` metadata lines from ``eval/router_eval.py`` are ignored.

Run: ``uv run python scripts/calibrate_router.py --input runs.jsonl``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO

EFFICIENT_TIER = "efficient"
CAPABLE_TIER = "capable"

RESCUE = "rescue"
LOSS = "loss"
SAFE = "safe"
HARD = "hard"


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Load paired-run rows, skipping metadata and incomplete records."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("record_type") == "metadata":
            continue
        if not _is_complete_row(record):
            continue
        rows.append(record)
    return rows


def _is_complete_row(record: dict[str, Any]) -> bool:
    for key in ("instance_id", "tier", "resolved", "cost_usd", "score"):
        if record.get(key) is None:
            return False
    return record["tier"] in (EFFICIENT_TIER, CAPABLE_TIER)


def pair_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Group rows by task with one efficient and one capable entry each."""
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        task = str(row["instance_id"])
        tier = str(row["tier"])
        slot = pairs.setdefault(task, {})
        slot[tier] = row
    return {task: slot for task, slot in pairs.items() if _is_paired(slot)}


def _is_paired(slot: dict[str, dict[str, Any]]) -> bool:
    return EFFICIENT_TIER in slot and CAPABLE_TIER in slot


def classify_quadrants(
    pairs: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, list[str]]:
    """Classify paired tasks into RESCUE/LOSS/SAFE/HARD quadrants."""
    quadrants: dict[str, list[str]] = {RESCUE: [], LOSS: [], SAFE: [], HARD: []}
    for task in sorted(pairs):
        slot = pairs[task]
        efficient_ok = bool(slot[EFFICIENT_TIER]["resolved"])
        capable_ok = bool(slot[CAPABLE_TIER]["resolved"])
        if capable_ok and not efficient_ok:
            quadrants[RESCUE].append(task)
        elif efficient_ok and not capable_ok:
            quadrants[LOSS].append(task)
        elif efficient_ok and capable_ok:
            quadrants[SAFE].append(task)
        else:
            quadrants[HARD].append(task)
    return quadrants


def sweep_thresholds(
    pairs: dict[str, dict[str, dict[str, Any]]],
    candidates: list[float],
) -> list[dict[str, Any]]:
    """Score each candidate threshold on paired tasks.

    The escalation signal is the efficient-run score: a task escalates
    when ``score >= threshold``. Predicted outcome and cost follow the
    tier that would have served the task.
    """
    results: list[dict[str, Any]] = []
    for threshold in candidates:
        solved = 0
        cost = 0.0
        escalated = 0
        rescued = 0
        lost = 0
        total = 0
        for task in sorted(pairs):
            slot = pairs[task]
            score = float(slot[EFFICIENT_TIER]["score"])
            escalate = score >= threshold
            chosen = slot[CAPABLE_TIER] if escalate else slot[EFFICIENT_TIER]
            total += 1
            if escalate:
                escalated += 1
            if bool(chosen["resolved"]):
                solved += 1
            cost += float(chosen["cost_usd"])
            if _is_rescued(slot, escalate):
                rescued += 1
            if _is_lost(slot, escalate):
                lost += 1
        results.append(
            {
                "threshold": threshold,
                "tasks": total,
                "solve_rate": solved / total if total else 0.0,
                "total_cost_usd": cost,
                "cost_per_solved": cost / solved if solved else float("inf"),
                "escalation_rate": escalated / total if total else 0.0,
                "rescued": rescued,
                "lost": lost,
            }
        )
    return results


def _is_rescued(slot: dict[str, dict[str, Any]], escalated: bool) -> bool:
    return (
        escalated
        and bool(slot[CAPABLE_TIER]["resolved"])
        and not bool(slot[EFFICIENT_TIER]["resolved"])
    )


def _is_lost(slot: dict[str, dict[str, Any]], escalated: bool) -> bool:
    return (
        escalated
        and not bool(slot[CAPABLE_TIER]["resolved"])
        and bool(slot[EFFICIENT_TIER]["resolved"])
    )


def recommend(
    sweep: list[dict[str, Any]],
    max_escalation_rate: float,
) -> dict[str, Any] | None:
    """Pick the best threshold within the escalation budget.

    Budget-first: keep candidates at or under the affordable Grok-call
    share, then take the highest solve rate, lowest cost per solved task,
    and lowest threshold in that order.
    """
    affordable = [row for row in sweep if row["escalation_rate"] <= max_escalation_rate]
    if not affordable:
        return None
    return min(affordable, key=_recommend_key)


def _recommend_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (-float(row["solve_rate"]), float(row["cost_per_solved"]), float(row["threshold"]))


def parse_thresholds(raw: str) -> list[float]:
    """Parse a comma-separated candidate list into sorted unique values."""
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted(set(values))


def suggested_run_command(manifest: str) -> str:
    """Return the pinned-tier eval command for pure GLM/Grok baselines."""
    return (
        "uv run python eval/router_eval.py --manifest "
        f"{manifest} --arms glm-only,grok-only "
        "--fixed-model glm-only=modal.glm-5-3/zai-org/GLM-5.3 "
        "--fixed-model grok-only=bedrock-openai/global.xai.grok-4.6 "
        "--mantis-endpoint http://127.0.0.1:8088/v1/chat/completions"
    )


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """CLI entry point for threshold calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.3,0.5,0.7")
    parser.add_argument("--max-escalation-rate", type=float, default=0.3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        candidates = parse_thresholds(args.thresholds)
    except ValueError:
        print(f"invalid --thresholds value: {args.thresholds!r}")
        return 2
    if not candidates:
        print("no candidate thresholds parsed")
        return 2
    rows = load_rows(args.input)
    pairs = pair_rows(rows)
    if not pairs:
        print("no paired efficient/capable tasks found")
        return 1
    quadrants = classify_quadrants(pairs)
    sweep = sweep_thresholds(pairs, candidates)
    picked = recommend(sweep, args.max_escalation_rate)
    report = {
        "tasks": len(pairs),
        "quadrants": {name: len(tasks) for name, tasks in quadrants.items()},
        "sweep": sweep,
        "recommended_threshold": picked["threshold"] if picked else None,
    }
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text, file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
