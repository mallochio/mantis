"""Join Harbor trial results with Switchyard routing records into calibration rows.

Produces the JSONL row format that ``scripts/calibrate_router.py`` consumes::

    {"instance_id": "...", "tier": "efficient"|"capable",
     "resolved": bool, "cost_usd": float, "score": float}

Three runs feed the join:

* ``--efficient-job`` — Harbor job with the agent pinned to the efficient
  model (pure-efficient outcome, no router in the path).
* ``--capable-job`` — Harbor job pinned to the capable model.
* ``--routing-log`` — routing records from the *probe* run: the same tasks
  routed through the calibration Switchyard route (``cal/base`` on :5501).
  The probe's per-turn ``tier`` tells us which turns the router escalated.

The Switchyard routing log carries no confidence value — only the served
tier per request — so ``score`` is the fraction of a task's calls that the
router sent to the capable model (``capable_calls / total_calls``). It is a
task-level escalation propensity, not Switchyard's per-turn confidence: a
sweep threshold T reads as "pin capable when at least T of the task's turns
escalated at theta=0.5". Capable-tier rows carry ``score=0.0``; the field is
required by the row schema but never read for that tier.

Run::

    uv run python scripts/join_calibration_scores.py \
      --efficient-job jobs/tb4-eff --capable-job jobs/tb4-cap \
      --routing-log ~/.local/share/mantis/switchyard/routing-log-cal.jsonl \
      --capable-model muse-spark-1.3-contributor \
      --output eval/runs/calibration-rows.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

EFFICIENT_TIER = "efficient"
CAPABLE_TIER = "capable"

_RESULT_FILES = ("result.json", "results.json")


def find_trial_results(job_dir: Path) -> list[dict[str, Any]]:
    """Load every Harbor TrialResult under a job directory."""
    trials: list[dict[str, Any]] = []
    for name in _RESULT_FILES:
        for path in sorted(job_dir.rglob(name)):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(record, dict) and record.get("task_name"):
                trials.append(record)
    return trials


def trial_resolved(trial: dict[str, Any]) -> bool:
    """A trial is resolved when any verifier reward reaches 1.0."""
    verifier = trial.get("verifier_result") or {}
    rewards = verifier.get("rewards") or {}
    values = rewards.values() if isinstance(rewards, dict) else [rewards]
    for value in values:
        try:
            if float(value) >= 1.0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def trial_session(trial: dict[str, Any]) -> str | None:
    """The session id the agent sent on its model calls.

    Harbor assigns ``<trial_name>__agent`` to ``agent.session_id``; agents
    that forward it (mini-swe-agent ``session_id_headers``) produce routing
    records with that value. Also accepts a ``cal_session_id`` the agent may
    have stored in its context metadata.
    """
    agent = trial.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    session = metadata.get("cal_session_id")
    if isinstance(session, str) and session:
        return session
    name = trial.get("trial_name")
    return f"{name}__agent" if isinstance(name, str) and name else None


def trial_cost(trial: dict[str, Any], prices: dict[str, Any]) -> float:
    """Reported cost, else token counts priced from a shadow price table."""
    agent = trial.get("agent_result") or {}
    cost = agent.get("cost_usd")
    if isinstance(cost, int | float) and not isinstance(cost, bool):
        return float(cost)
    metadata = agent.get("metadata") or {}
    model = str(metadata.get("model_requested") or "")
    price = prices.get(model) or {}
    prompt = float(agent.get("n_input_tokens") or 0)
    completion = float(agent.get("n_output_tokens") or 0)
    return (
        prompt * float(price.get("input_per_token") or 0)
        + completion * float(price.get("output_per_token") or 0)
    )


def extract_arm_rows(
    job_dir: Path, tier: str, prices: dict[str, Any]
) -> list[dict[str, Any]]:
    """Normalize a Harbor job dir into per-task arm rows."""
    return [
        {
            "instance_id": str(trial["task_name"]),
            "tier": tier,
            "resolved": trial_resolved(trial),
            "cost_usd": trial_cost(trial, prices),
            "session_id": trial_session(trial),
            "trial_name": trial.get("trial_name"),
        }
        for trial in find_trial_results(job_dir)
    ]


def load_routing_scores(
    log_path: Path, capable_model: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Capable-call fraction per task and per session from the routing log.

    Returns ``(by_task, by_session)``; the task key comes from the
    ``x-switchyard-intake-task`` header, the session key from
    ``x-switchyard-session-id``.
    """
    counts: dict[str, dict[str, list[int]]] = {"task": {}, "session": {}}
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or not record.get("model"):
            continue
        capable = 1 if record.get("model") == capable_model else 0
        # ``task`` comes from x-switchyard-intake-task, ``session``/``trial``
        # from x-switchyard-session-id (metadata.session_id) and
        # x-switchyard-trial-id respectively.
        for kind, key in (
            ("task", record.get("task")),
            ("session", record.get("session_id")),
            ("session", record.get("trial_id")),
        ):
            if isinstance(key, str) and key:
                slot = counts[kind].setdefault(key, [0, 0])
                slot[0] += capable
                slot[1] += 1
    by_task = {k: c / t for k, (c, t) in counts["task"].items() if t}
    by_session = {k: c / t for k, (c, t) in counts["session"].items() if t}
    return by_task, by_session


def attach_scores(
    rows: list[dict[str, Any]],
    by_task: dict[str, float],
    by_session: dict[str, float],
) -> list[dict[str, Any]]:
    """Fill ``score`` on efficient rows; capable rows get 0.0 (schema filler)."""
    out = []
    for row in rows:
        row = dict(row)
        if row["tier"] == EFFICIENT_TIER:
            score = by_task.get(row["instance_id"])
            if score is None:
                for key in (row.get("session_id"), row.get("trial_name")):
                    if isinstance(key, str) and key in by_session:
                        score = by_session[key]
                        break
            row["score"] = 0.0 if score is None else score
        else:
            row["score"] = 0.0
        row.pop("session_id", None)
        row.pop("trial_name", None)
        out.append(row)
    return out


def _load_prices(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    table = json.loads(path.read_text())
    return table.get("models", table) if isinstance(table, dict) else {}


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    """CLI entry point for the calibration join."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--efficient-job", type=Path, required=True)
    parser.add_argument("--capable-job", type=Path, required=True)
    parser.add_argument("--routing-log", type=Path, required=True)
    parser.add_argument("--capable-model", required=True)
    parser.add_argument("--price-table", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    if not args.routing_log.exists():
        print(f"no routing log at {args.routing_log}", file=sys.stderr)
        return 1
    prices = _load_prices(args.price_table)
    rows = extract_arm_rows(args.efficient_job, EFFICIENT_TIER, prices)
    rows += extract_arm_rows(args.capable_job, CAPABLE_TIER, prices)
    if not rows:
        print("no trial results found in the job dirs", file=sys.stderr)
        return 1
    by_task, by_session = load_routing_scores(args.routing_log, args.capable_model)
    rows = attach_scores(rows, by_task, by_session)
    unscored = sum(
        1 for row in rows if row["tier"] == EFFICIENT_TIER and row["score"] == 0.0
    )
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    if args.output is not None:
        args.output.write_text(text)
    print(text, end="", file=stdout)
    print(
        f"{len(rows)} rows written; {unscored} efficient rows have no probe score",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
