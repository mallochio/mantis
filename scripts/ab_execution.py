"""Execute seed-isolated variants and persist their observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from ab_drive import drive_case
from ab_records import _persist_run, build_record, request_plan, seed_workspace
from ab_seeds import (
    DEFAULT_CHECKS,
    completed_runs,
    execution_workspace,
    run_case_checks,
    transcript_name,
)
from evaluate_routed_harness import EvalCase
from routed_ab_tools import openrouter_headers


@dataclass(frozen=True)
class RunConfig:
    """Hold shared options for one routed A/B variant run."""

    client: httpx.Client
    base_url: str
    variant: str
    cases: list[EvalCase]
    catalog: dict[str, Any]
    out_path: Path
    transcripts_dir: Path
    workspace_root: Path
    fixtures_root: Path
    max_rounds: int
    prices: dict[str, Any] | None = None
    seeds: int = 1
    force: bool = False
    checks_path: Path = DEFAULT_CHECKS
    completed: set[tuple[str, str, int]] | None = None


def _execute_seed(
    config: RunConfig, case: EvalCase, workspace: Path, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run, verify, and record one isolated seed workspace."""
    workspace.mkdir(parents=True, exist_ok=False)
    seed_workspace(case.case_id, workspace, config.fixtures_root)
    plan = request_plan(config.variant, config.catalog, workspace)
    url = str(plan.get("base_url") or config.base_url)
    auth = openrouter_headers() if plan.get("base_url") else {}
    observation = drive_case(
        config.client, url, plan, case.prompt, workspace, config.max_rounds, auth
    )
    observation["workspace_path"] = str(workspace.resolve())
    checks = run_case_checks(
        case.case_id,
        workspace,
        observation["content"],
        [],
        observation["tool_rounds"],
        config.checks_path,
        observation=observation,
    )
    record = build_record(config.variant, case, observation, plan, config.prices, seed, checks)
    attempt = workspace.name if workspace.name.startswith("attempt-") else None
    record["workspace_path"] = str(workspace.resolve())
    record["transcript_file"] = transcript_name(config.variant, case.case_id, seed, attempt)
    return record, observation


def _run_seed(config: RunConfig, case: EvalCase, seed: int, done: set[tuple[str, str, int]]) -> int:
    """Execute one seed unless the output already records it."""
    key = (config.variant, case.case_id, seed)
    if key in done and not config.force:
        return 0
    workspace = execution_workspace(
        config.workspace_root, config.variant, case.case_id, seed, config.force
    )
    record, observation = _execute_seed(config, case, workspace, seed)
    _persist_run(record, observation, case, config.out_path, config.transcripts_dir)
    done.add(key)
    passed = record["passed"]
    cost = record["provider_cost"]
    print(f"{config.variant} {case.case_id} seed{seed}: {passed} ({cost:.6f} USD)")
    return 1


def _execute_variant(config: RunConfig) -> int:
    """Run every case and seed under one configured variant."""
    done = config.completed if config.completed is not None else completed_runs(config.out_path)
    return sum(
        _run_seed(config, case, seed, done)
        for case in config.cases
        for seed in range(1, config.seeds + 1)
    )


def run_variant(*args: Any, **kwargs: Any) -> int:
    """Execute a variant with the established positional and keyword options."""
    return _execute_variant(RunConfig(*args, **kwargs))
