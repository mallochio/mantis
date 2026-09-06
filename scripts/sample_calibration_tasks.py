"""Sample a stratified calibration task list from a task pool.

The sampler is dataset-agnostic: it reads a JSONL pool where each line
has ``task_id`` and an optional difficulty label, then writes a manifest
with a fixed seed and per-stratum quotas. For Hugging Face pools, first
dump the pool to JSONL (for example with ``datasets``), then sample here
so this script needs no network access and no extra dependencies.

Pool row example::

    {"task_id": "dsbench-001", "difficulty": "hard", "prompt": "..."}

Manifest output::

    {"dataset": "dsbench-data-modeling", "seed": 7, "tasks": [...]}
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, TextIO

DEFAULT_STRATA = ("easy", "hard")
FALLBACK_STRATUM = "unspecified"


def load_pool(path: Path) -> list[dict[str, Any]]:
    """Load pool rows that carry a usable task id."""
    tasks: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("task_id"):
            tasks.append(record)
    return tasks


def stratum_of(task: dict[str, Any]) -> str:
    """Return the difficulty stratum, falling back when unlabeled."""
    difficulty = task.get("difficulty")
    if isinstance(difficulty, str) and difficulty.strip():
        return difficulty.strip().lower()
    return FALLBACK_STRATUM


def sample_tasks(
    pool: list[dict[str, Any]],
    count: int,
    seed: int,
    strata: tuple[str, ...] = DEFAULT_STRATA,
) -> list[dict[str, Any]]:
    """Sample up to ``count`` tasks with equal per-stratum quotas.

    Strata listed in ``strata`` split the quota evenly; pool tasks from
    unlisted strata fill remaining slots so a thin stratum never aborts
    the sample. Ordering is deterministic for a fixed seed.
    """
    if count <= 0:
        return []
    rng = random.Random(seed)
    grouped = _group_by_stratum(pool, strata)
    quota, remainder = divmod(count, len(strata))
    picked: list[dict[str, Any]] = []
    for index, stratum in enumerate(strata):
        want = quota + (1 if index < remainder else 0)
        picked.extend(rng.sample(grouped[stratum], min(want, len(grouped[stratum]))))
    picked_ids = {str(task["task_id"]) for task in picked}
    if len(picked) < count:
        rest = [task for task in pool if str(task["task_id"]) not in picked_ids]
        rng.shuffle(rest)
        picked.extend(rest[: count - len(picked)])
    ordered = sorted(picked, key=lambda task: str(task["task_id"]))
    return ordered[:count]


def _group_by_stratum(
    pool: list[dict[str, Any]],
    strata: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in strata}
    for task in pool:
        name = stratum_of(task)
        grouped.setdefault(name, []).append(task)
    for tasks in grouped.values():
        tasks.sort(key=lambda task: str(task["task_id"]))
    return grouped


def build_manifest(
    dataset: str,
    tasks: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    """Build a calibration manifest from sampled tasks."""
    return {
        "dataset": dataset,
        "seed": seed,
        "instance_count": len(tasks),
        "instances": [
            {"instance_id": str(task["task_id"]), **task} for task in tasks
        ],
    }


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """CLI entry point for calibration sampling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--dataset", default="dsbench-data-modeling")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--strata", default="easy,hard")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    strata = tuple(part.strip().lower() for part in args.strata.split(",") if part.strip())
    if not strata:
        print("no strata parsed")
        return 2
    if args.count <= 0:
        print("--count must be positive")
        return 2
    pool = load_pool(args.pool)
    if not pool:
        print("no usable tasks in pool")
        return 1
    tasks = sample_tasks(pool, args.count, args.seed, strata)
    text = json.dumps(build_manifest(args.dataset, tasks, args.seed), indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text, file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
