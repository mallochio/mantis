#!/usr/bin/env python3
"""Build a frozen, repository-disjoint SWE-rebench manifest.

This script downloads revision-pinned monthly parquet files directly from the
Hugging Face repository, excludes repositories present in one or more existing
manifests, and deterministically samples whole task rows. It does not make model
calls. PyArrow and pandas are optional command-time dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_DATASET = "nebius/SWE-rebench-leaderboard"
DEFAULT_REVISION = "34d5a58864acf91613740a09ec5d205228dcfa39"
DEFAULT_MONTHS = ("2025_12", "2026_01", "2026_02", "2026_03")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def _download(dataset: str, revision: str, month: str, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{month}.parquet"
    if path.exists():
        return path
    url = (
        f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/"
        f"data/{month}-00000-of-00001.parquet"
    )
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def _excluded_repos(paths: list[Path]) -> set[str]:
    repos: set[str] = set()
    for path in paths:
        data = json.loads(path.read_text())
        repos.update(str(row["repo"]) for row in data["instances"])
    return repos


def _manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    install = _jsonable(row.get("install_config")) or {}
    return {
        "FAIL_TO_PASS": _jsonable(row.get("FAIL_TO_PASS")) or [],
        "PASS_TO_PASS": _jsonable(row.get("PASS_TO_PASS")) or [],
        "base_commit": str(row["base_commit"]),
        "created_at": str(row.get("created_at") or ""),
        "docker_image": str(row["docker_image"]),
        "install": str(install.get("install") or ""),
        "instance_id": str(row["instance_id"]),
        "interface": str(row.get("interface") or ""),
        "patch": str(row.get("patch") or ""),
        "problem_statement": str(row["problem_statement"]),
        "python": str(install.get("python") or ""),
        "repo": str(row["repo"]),
        "test_cmd": str(install.get("test_cmd") or ""),
        "test_patch": str(row.get("test_patch") or ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--months", default=",".join(DEFAULT_MONTHS))
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--cache", type=Path, default=Path("eval/runs/dataset-cache"))
    args = parser.parse_args()

    months = [month.strip() for month in args.months.split(",") if month.strip()]
    frames = []
    source_hashes: dict[str, str] = {}
    for month in months:
        path = _download(args.dataset, args.revision, month, args.cache)
        source_hashes[month] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_parquet(path)
        frame["_source_month"] = month
        frames.append(frame)

    frame = pd.concat(frames, ignore_index=True).drop_duplicates("instance_id")
    excluded = _excluded_repos(args.exclude_manifest)
    candidates = frame[~frame["repo"].isin(excluded)].copy()
    candidates = candidates.sort_values(["repo", "created_at", "instance_id"])
    rows = candidates.to_dict(orient="records")
    random.Random(args.seed).shuffle(rows)
    if len(rows) < args.count:
        raise SystemExit(f"only {len(rows)} repository-disjoint tasks available")
    selected = rows[: args.count]

    result = {
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "source_months": months,
        "source_sha256": source_hashes,
        "instance_count": len(selected),
        "instances": [_manifest_row(row) for row in selected],
        "repo_distribution": dict(
            sorted(pd.Series([row["repo"] for row in selected]).value_counts().to_dict().items())
        ),
        "selection": {
            "method": "seeded shuffle after sorting and repository exclusion",
            "seed": args.seed,
            "candidate_rows": len(rows),
            "excluded_repositories": len(excluded),
            "excluded_manifests": [str(path) for path in args.exclude_manifest],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(selected)} tasks to {args.output}")


if __name__ == "__main__":
    main()
