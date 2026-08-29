"""Metrics for router evaluation rows and frozen tier oracles."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TIERS = ("cheap", "middle", "expensive")


def _completed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row.get("error")]


def load_result_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [row for row in rows if row.get("record_type") in {None, "result"} and "arm" in row]


def oracle_labels(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    """Return the cheapest tier that resolved each instance."""
    by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") in TIERS:
            tier = row["arm"]
        elif row.get("arm", "").endswith("-only"):
            tier = row["arm"].removesuffix("-only")
        else:
            continue
        by_instance[row["instance_id"]][tier] = row
    labels: dict[str, str | None] = {}
    for instance_id, tiers in by_instance.items():
        labels[instance_id] = next(
            (tier for tier in TIERS if tiers.get(tier, {}).get("resolved")), None
        )
    return labels


def _arm_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row.get(key, float(bool(row.get("resolved")))))
        for row in rows
        if row.get(key) is not None or key == "quality"
    ]
    return statistics.mean(values) if values else None


def _cost_sum(rows: list[dict[str, Any]]) -> float | None:
    costs = [row.get("cost_usd") for row in rows]
    if any(cost is None for cost in costs):
        return None
    return sum(float(cost or 0) for cost in costs)


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = _completed(rows)
    resolved = [row for row in rows if row.get("resolved")]
    return {
        "total": len(rows),
        "completed": len(completed),
        "resolved": len(resolved),
        "errors": len(rows) - len(completed),
        "quality_mean": _arm_mean(rows, "quality"),
        "cost_usd": _cost_sum(rows),
    }


def _regret(
    chosen: str | None,
    oracle: str | None,
    costs: dict[str, float | None],
    qualities: dict[str, float],
) -> tuple[float, float | None]:
    if not chosen or not oracle:
        return 0.0, 0.0
    chosen_rank, oracle_rank = TIERS.index(chosen), TIERS.index(oracle)
    quality_loss = max(0.0, qualities.get(oracle, 0.0) - qualities.get(chosen, 0.0))
    chosen_cost, oracle_cost = costs.get(chosen), costs.get(oracle)
    if chosen_rank < oracle_rank:
        return quality_loss, (
            max(0.0, oracle_cost - chosen_cost)
            if chosen_cost is not None and oracle_cost is not None
            else None
        )
    if chosen_rank > oracle_rank:
        return 0.0, (
            max(0.0, chosen_cost - oracle_cost)
            if chosen_cost is not None and oracle_cost is not None
            else None
        )
    return 0.0, 0.0


def _tier_cost(row: dict[str, Any] | None) -> float | None:
    cost = (row or {}).get("cost_usd")
    return float(cost) if cost is not None else None


def _tier_quality(row: dict[str, Any] | None) -> float:
    return float((row or {}).get("quality", 0) or 0)


def _tier_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float | None]], dict[str, dict[str, float]]]:
    """Per-instance cost and quality maps for tier arms (tier or <tier>-only)."""
    tier_by_instance: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("arm") in TIERS or row.get("arm", "").endswith("-only"):
            tier = row.get("tier") or row["arm"].removesuffix("-only")
            tier_by_instance[row["instance_id"]][tier] = row
    tier_costs = {
        instance: {tier: _tier_cost(tier_by_instance[instance].get(tier)) for tier in TIERS}
        for instance in tier_by_instance
    }
    tier_quality = {
        instance: {tier: _tier_quality(tier_by_instance[instance].get(tier)) for tier in TIERS}
        for instance in tier_by_instance
    }
    return tier_costs, tier_quality


def _accuracy_and_regret(
    by_arm: dict[str, list[dict[str, Any]]],
    oracle: dict[str, str | None],
    tier_costs: dict[str, dict[str, float | None]],
    tier_quality: dict[str, dict[str, float]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Oracle accuracy plus under/over-routing regret totals per arm."""
    accuracy: dict[str, Any] = {}
    regrets: dict[str, Any] = {}
    for arm, arm_rows in by_arm.items():
        arm_rows_by_instance = {row["instance_id"]: row for row in _completed(arm_rows)}
        chosen = {
            row["instance_id"]: (
                row.get("chosen_tier")
                or row.get("tier")
                or str(row["arm"]).removesuffix("-only")
            )
            for row in _completed(arm_rows)
        }
        comparable = [instance for instance in chosen if oracle.get(instance) is not None]
        correct = sum(chosen[i] == oracle[i] for i in comparable)
        totals = {
            "under_routing": {"quality_lost": 0.0, "dollars_wasted": 0.0},
            "over_routing": {"quality_lost": 0.0, "dollars_wasted": 0.0},
        }
        for instance in comparable:
            oracle_tier = oracle[instance]
            assert oracle_tier is not None
            _, wasted = _regret(
                chosen[instance],
                oracle_tier,
                tier_costs.get(instance, {}),
                tier_quality.get(instance, {}),
            )
            chosen_rank, oracle_rank = TIERS.index(chosen[instance]), TIERS.index(oracle_tier)
            if chosen_rank == oracle_rank:
                continue
            direction = "under_routing" if chosen_rank < oracle_rank else "over_routing"
            bucket = totals[direction]
            bucket["quality_lost"] += max(
                0.0,
                tier_quality.get(instance, {}).get(oracle_tier, 0.0)
                - float(arm_rows_by_instance[instance].get("quality", 0.0)),
            )
            bucket["dollars_wasted"] = (
                bucket["dollars_wasted"] + wasted
                if bucket["dollars_wasted"] is not None and wasted is not None
                else None
            )
        accuracy[arm] = {
            "correct": correct,
            "eligible": len(comparable),
            "accuracy": correct / len(comparable) if comparable else None,
        }
        regrets[arm] = {
            "under_routing": dict(totals["under_routing"]),
            "over_routing": dict(totals["over_routing"]),
        }
    return accuracy, regrets


def _cheap_expensive_interpolation(arms: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    """Router quality vs linear interpolation between the cheap and expensive baselines."""
    cheap, expensive = arms.get("cheap-only"), arms.get("expensive-only")
    router = arms.get("mantis-direct")
    if not (
        cheap
        and expensive
        and router
        and cheap["quality_mean"] is not None
        and expensive["quality_mean"] is not None
        and router["cost_usd"] is not None
        and cheap["cost_usd"] is not None
        and expensive["cost_usd"] is not None
    ):
        return None
    span = expensive["cost_usd"] - cheap["cost_usd"]
    fraction = (router["cost_usd"] - cheap["cost_usd"]) / span if span else None
    expected = (
        cheap["quality_mean"] + fraction * (expensive["quality_mean"] - cheap["quality_mean"])
        if fraction is not None
        else None
    )
    return {
        "router_cost_usd": router["cost_usd"],
        "router_quality": router["quality_mean"],
        "interpolated_quality": expected,
        "beats_interpolation": (
            router["quality_mean"] is not None
            and expected is not None
            and router["quality_mean"] > expected
        ),
    }


def compute_metrics(
    rows: list[dict[str, Any]], *, oracle: dict[str, str | None] | None = None
) -> dict[str, Any]:
    """Compute oracle accuracy, regret, frontier, and endpoint interpolation."""
    oracle = oracle or oracle_labels(rows)
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)

    tier_costs, tier_quality = _tier_maps(rows)
    accuracy, regrets = _accuracy_and_regret(by_arm, oracle, tier_costs, tier_quality)

    arms = {arm: _arm_summary(arm_rows) for arm, arm_rows in by_arm.items()}
    return {
        "oracle": oracle,
        "accuracy": accuracy,
        "regret": regrets,
        "arms": arms,
        "frontier": [{"arm": arm, **summary} for arm, summary in arms.items()],
        "cheap_expensive_interpolation": _cheap_expensive_interpolation(arms),
    }


def complexity_confusion(
    rows: list[dict[str, Any]], *, targets: tuple[str, ...]
) -> dict[str, dict[str, int]]:
    """Build complexity-level to oracle-tier counts from offline decisions."""
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        complexity = str(row["complexity"])
        oracle = row.get("oracle_tier")
        if oracle:
            matrix[complexity][oracle] += 1
    return {level: dict(counts) for level, counts in sorted(matrix.items())}


def replay_complexity(
    prompts: list[dict[str, Any]], decider: Any, oracle: dict[str, str | None]
) -> dict[str, Any]:
    """Replay prompts through an injected offline gateway decision function."""
    rows = []
    for prompt in prompts:
        complexity, target = decider(prompt["prompt"])
        rows.append(
            {
                "instance_id": prompt["instance_id"],
                "complexity": complexity,
                "decision": target,
                "oracle_tier": oracle.get(prompt["instance_id"]),
            }
        )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row["oracle_tier"]:
            confusion[row["complexity"]][row["oracle_tier"]] += 1
    return {
        "rows": rows,
        "confusion": {level: dict(counts) for level, counts in sorted(confusion.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute router metrics from JSONL rows.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_result_rows(args.results)
    output = json.dumps(compute_metrics(rows), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
