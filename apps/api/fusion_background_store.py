"""Persistence and merge helpers for background Fusion handoffs."""

from __future__ import annotations

import copy
from typing import Any

import runs
import serve_config


def persist(source: Any, task: Any) -> bool:
    """Persist worker-owned state while respecting an active lead continuation."""
    if source.cancelled or source.finished:
        return True
    if runs.RUN_STORE == "memory":
        if not task.consumed:
            runs.serve_config._runs[task.run.run_id] = task.run
        return True
    lock, load, save = _store_io()
    with lock(source.run_id):
        if task.consumed:
            return True
        target = load(source.run_id)
        if target is not None:
            merge(target, source)
            save(target)
    return True


def merge(target: Any, source: Any) -> None:
    """Merge sidekick state without replacing newer lead turns."""
    target.sidekick_messages = copy.deepcopy(source.sidekick_messages)
    target.subtask_report = source.subtask_report
    target.handoff_history = _merge_history(target.handoff_history, source.handoff_history)
    target.role_usage["sidekick"] = copy.deepcopy(source.role_usage.get("sidekick", {}))
    target.turns = _merge_history(target.turns, source.turns)
    target_handoffs, source_handoffs = target.handoffs, source.handoffs
    if target_handoffs is None or source_handoffs is None:
        target.handoffs = copy.deepcopy(source_handoffs)
        return
    _merge_handoffs(target, source, target_handoffs, source_handoffs)


def _merge_handoffs(target: Any, source: Any, target_handoffs: Any, source_handoffs: Any) -> None:
    """Merge manager state and sidekick usage without losing lead-side increments."""
    source_usage = source_handoffs.background_usage
    target_usage = target_handoffs.background_usage
    if target_usage.get("token") != source_usage.get("token"):
        target_usage = target_handoffs.background_usage = copy.deepcopy(source_usage)
    applied = target_usage.setdefault("applied", {})
    _merge_sidekick_usage(
        target, source, source_usage["baseline_role"], applied.setdefault("role", {})
    )
    _merge_model_usage(
        target.usage_models, source_usage["models"], {}, applied.setdefault("models", {})
    )
    _merge_budget(target.budget, source_usage["budget"], applied.setdefault("budget", {}))
    for name in (
        "current",
        "last",
        "sidekick_batch",
        "sidekick_held",
        "sidekick_waiting",
        "tool_rounds",
        "error_streak",
        "delegation_names",
    ):
        setattr(target_handoffs, name, copy.deepcopy(getattr(source_handoffs, name)))
    target_handoffs.pending_notifications = _merge_history(
        target_handoffs.pending_notifications, source_handoffs.pending_notifications
    )
    target_handoffs.notified_handoffs |= source_handoffs.notified_handoffs


def _store_io() -> tuple[Any, Any, Any]:
    """Return the active run-store lock, loader, and saver."""
    if runs.RUN_STORE == "file":
        return runs._file_run_lock, runs._file_get, runs._file_put
    return runs._redis_run_lock, runs._redis_get, runs._redis_put


def _merge_history(
    target: list[dict[str, Any]], source: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep existing lead history and append unique worker entries."""
    merged = list(target)
    for entry in source:
        if entry not in merged:
            merged.append(copy.deepcopy(entry))
    return merged[-50:]


def _merge_sidekick_usage(
    target: Any, source: Any, baseline: dict[str, Any], applied: dict[str, Any]
) -> None:
    """Add sidekick role deltas once without copying lead progress from the source run."""
    prompt = completion = cached = 0
    for slot, counts in source.role_usage.get("sidekick", {}).items():
        base, done = baseline.get(slot, {}), applied.setdefault(slot, {})
        delta = {
            key: max(0, int(counts.get(key, 0)) - int(base.get(key, 0)))
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens")
        }
        added = {key: max(0, value - int(done.get(key, 0))) for key, value in delta.items()}
        done.update(delta)
        prompt += added["prompt_tokens"]
        completion += added["completion_tokens"]
        cached += added["cached_tokens"]
    target.usage["prompt_tokens"] = int(target.usage.get("prompt_tokens", 0)) + prompt
    target.usage["completion_tokens"] = int(target.usage.get("completion_tokens", 0)) + completion
    target.usage["total_tokens"] = int(target.usage.get("total_tokens", 0)) + prompt + completion
    details = target.usage.setdefault("prompt_tokens_details", {})
    details["cached_tokens"] = int(details.get("cached_tokens", 0)) + cached


def _merge_model_usage(
    target: dict[str, Any],
    source: dict[str, Any],
    baseline: dict[str, Any],
    applied: dict[str, Any],
) -> None:
    """Merge actual provider usage identities with an idempotent background delta."""
    for model, counts in source.items():
        prior = baseline.get(model, {})
        done = applied.setdefault(model, {})
        totals = target.setdefault(model, {})
        for key, value in counts.items():
            if isinstance(value, int):
                delta = max(0, value - int(prior.get(key, 0)))
                increment = max(0, delta - int(done.get(key, 0)))
                totals[key] = int(totals.get(key, 0)) + increment
                done[key] = max(int(done.get(key, 0)), delta)


def _merge_budget(target: Any, delta: dict[str, int], applied: dict[str, int]) -> None:
    """Add sidekick token and turn deltas once without losing lead spend."""
    if target is None:
        return
    with target._lock:
        for name, attribute in (("tokens", "_tokens"), ("turns", "_turns")):
            total = int(delta.get(name, 0))
            previous = applied.get(name, 0)
            setattr(target, attribute, getattr(target, attribute) + max(0, total - previous))
            applied[name] = max(previous, total)


def usage_baseline(run: Any) -> dict[str, Any]:
    """Create per-handoff usage cursors from the run's sidekick role."""
    return {
        "token": f"{run.run_id}:{len(run.sidekick_messages)}",
        "baseline_role": copy.deepcopy(run.role_usage.get("sidekick", {})),
        "models": {},
        "budget": {"tokens": 0, "turns": 0},
        "applied": {},
    }


def sidekick_turn(
    manager: Any, run: Any, coordinator: Any, tools: Any
) -> tuple[str, list[dict[str, Any]]]:
    """Run one sidekick call and record usage under its actual returned model key."""
    text, calls, usage = run._call_lane(coordinator, "sidekick", tools=tools)
    budget = manager.background_usage["budget"]
    candidates = [usage.get(key) for key in ("total_tokens", "input_tokens", "prompt_tokens")]
    spent = max((value for value in candidates if isinstance(value, int)), default=0)
    if not spent:
        spent = int(usage.get("prompt_tokens", 0) or 0) + int(
            usage.get("completion_tokens", 0) or 0
        )
    budget["tokens"] += spent
    budget["turns"] += 1
    message = run.sidekick_messages[-1]
    model = (
        getattr(serve_config._history_context, "fusion_selected_model", None)
        or message.get("_mantis_model")
        or run.sidekick_slot
    )
    totals = manager.background_usage.setdefault("models", {}).setdefault(str(model), {})
    for key in ("prompt_tokens", "completion_tokens"):
        if isinstance(usage.get(key), int):
            totals[key] = int(totals.get(key, 0)) + usage[key]
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    if isinstance(cached, int):
        totals["cached_tokens"] = int(totals.get("cached_tokens", 0)) + cached
    return text, calls
