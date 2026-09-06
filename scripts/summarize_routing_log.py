"""Summarize Switchyard per-request routing records for threshold mining.

Reads the JSONL written by ``switchyard-server --routing-log-file`` and
reports per-model calls, token totals, and cache hit rates. The capable
share of calls is the measured escalation rate; token totals show where
spend actually goes. Combine with task outcomes (see
``scripts/calibrate_router.py``) to judge a candidate threshold.

Run: ``uv run python scripts/summarize_routing_log.py --input \\
    ~/.local/share/mantis/switchyard/routing-log.jsonl``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TextIO


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load routing records, skipping blank and malformed lines."""
    records: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("model"):
            records.append(record)
    return records


def _number(record: dict[str, Any], key: str) -> float:
    try:
        return float(record.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate calls and tokens per model plus overall totals."""
    models: dict[str, dict[str, float]] = {}
    for record in records:
        model = str(record["model"])
        slot = models.setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
        )
        slot["calls"] += 1
        slot["prompt_tokens"] += _number(record, "prompt_tokens")
        slot["cached_tokens"] += _number(record, "cached_tokens")
        slot["total_tokens"] += _number(record, "total_tokens")
    summary_models = {
        model: {**totals, "cache_hit_rate": _hit_rate(totals)}
        for model, totals in sorted(models.items())
    }
    total_tokens = sum(t["total_tokens"] for t in models.values())
    return {
        "records": len(records),
        "total_tokens": total_tokens,
        "models": summary_models,
    }


def _hit_rate(totals: dict[str, float]) -> float:
    prompt = totals["prompt_tokens"]
    return totals["cached_tokens"] / prompt if prompt else 0.0


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """CLI entry point for routing-log summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"no routing log at {args.input}")
        return 1
    text = json.dumps(summarize(load_records(args.input)), indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text, file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
