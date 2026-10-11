"""Rebase native lead history at explicit public request boundaries."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import serve_config
from fusion_harness import lead_prompt_with_names
from fusion_prompts import lead_system_prompt


def rebase(run: Any, history: list[dict[str, Any]], tool_ids: dict[str, str]) -> None:
    """Replace public history while retaining only the current private working span."""
    if not history or all(item.get("role") == "tool" for item in history):
        return
    if run.status == "awaiting_tools":
        if not tool_ids:
            return
        boundary = _pending_boundary(run, history, tool_ids)
        checkpoint = run.fusion_public_checkpoint
        if checkpoint is None:
            raise ValueError("Fusion run has no public history checkpoint")
        suffix = run.main_messages[checkpoint:]
        _replace(run, history[:boundary], suffix)
    else:
        prefix = history[:-1] if history[-1].get("role") == "user" else history
        _replace(run, prefix, [])


def _pending_boundary(run: Any, history: list[dict[str, Any]], ids: dict[str, str]) -> int:
    """Find the exact current public assistant batch by its pending call identities."""
    expected = {str(call["id"]) for call in run.pending_tool_calls}
    for index in range(len(history) - 1, -1, -1):
        item = history[index]
        calls = item.get("tool_calls") or []
        actual = {ids.get(str(call.get("id"))) for call in calls}
        if item.get("role") == "assistant" and actual == expected and len(calls) == len(expected):
            return index
    raise ValueError("client history must contain the current pending assistant tool calls")


def _replace(run: Any, history: list[dict[str, Any]], suffix: list[dict[str, Any]]) -> None:
    """Rebuild orchestration policy from authoritative client system instructions."""
    history = deepcopy(history)
    system = []
    while history and history[0].get("role") == "system":
        system.append(str(history.pop(0).get("content") or ""))
    run.fusion_lead_prompt_base = lead_system_prompt("\n\n".join(system) or None)
    prompt = lead_prompt_with_names(run.fusion_lead_prompt_base, run.handoffs.delegation_names)
    run.main_messages = [{"role": "system", "content": prompt}, *history, *suffix]
    run.fusion_public_checkpoint = 1 + len(history)
    sidekick_cursor = run.trace_cursors.get("sidekick", 0)
    run.trace_cursors = {"main": len(run.main_messages), "sidekick": sidekick_cursor}

def requested_output(coordinator: Any, slot: str) -> int | None:
    """Return this request's explicit lead output cap, when present."""
    if slot != getattr(coordinator, "main_slot", None):
        return None
    run = getattr(serve_config._history_context, "active_run", None)
    requested = getattr(run, "controls", {}).get("max_output_tokens")
    return requested if isinstance(requested, int) and requested > 0 else None
