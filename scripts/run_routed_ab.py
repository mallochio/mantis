#!/usr/bin/env python3
"""Drive routed harness cases against a live server and emit run records."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import httpx
from ab_run import (
    auto_judge,
    base_target_spec,
    build_record,
    drive_case,
    load_catalog,
    load_prices,
    request_plan,
    run_variant,
    seed_workspace,
    selected_models,
    usage_numbers,
)
from ab_seeds import DEFAULT_CHECKS, completed_runs
from evaluate_routed_harness import load_cases
from routed_ab_tools import (
    OPENROUTER_BASE_URL,
    execute_tool_call,
    openrouter_headers,
    post_round,
    priced_cost,
    request_body,
)

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval"
DEFAULT_CATALOG = REPO / "config" / "catalog.toml"
DEFAULT_CASES = EVAL / "routed-harness-cases.jsonl"
DEFAULT_FIXTURES = EVAL / "routed-harness-fixtures"
DEFAULT_OUT = EVAL / "results-routed-ab.jsonl"
DEFAULT_TRANSCRIPTS = EVAL / "runs" / "routed-ab"
DEFAULT_PRICES = EVAL / "model_prices.json"
LIVE_VARIANTS = ("efficient-only", "capable-only", "mantis-base", "fusion-tool")
__all__ = [
    "OPENROUTER_BASE_URL", "auto_judge", "base_target_spec", "build_record", "drive_case",
    "execute_tool_call", "load_catalog", "load_prices", "openrouter_headers", "post_round",
    "priced_cost", "request_body", "request_plan", "run_variant", "seed_workspace",
    "completed_runs",
    "selected_models", "usage_numbers",
]


def build_parser() -> argparse.ArgumentParser:
    """Return the runner CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    base_url = os.environ.get("MANTIS_BASE_URL", "http://127.0.0.1:8000")
    parser.add_argument("--base-url", default=base_url)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--checks", type=Path, default=DEFAULT_CHECKS)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--variants", default=",".join(LIVE_VARIANTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument("--workspace-root", type=Path, default=REPO / ".scratch" / "routed-ab")
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute selected variants and write run records."""
    args = build_parser().parse_args(argv)
    if args.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in LIVE_VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    cases = load_cases(args.cases)
    catalog = load_catalog(args.catalog)
    prices = load_prices(args.prices)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = completed_runs(args.out)
    api_key = os.environ.get("MANTIS_API_KEY", "")
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    total = 0
    with httpx.Client(timeout=args.timeout, headers=auth) as client:
        for variant in variants:
            total += run_variant(
                client, args.base_url.rstrip("/"), variant, cases, catalog, args.out,
                args.transcripts_dir, args.workspace_root.resolve(), args.fixtures,
                args.max_rounds, prices, args.seeds, args.force, args.checks, done,
            )
    print(f"runs={total} records={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
