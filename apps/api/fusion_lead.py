"""Tool-based fusion delegation: the lead agent loop.

In tool delegation the lead is an ordinary agent loop. It sees the client's tools plus the
server-resolved ``sidekick`` and ``read_subagent`` tools. Client tool calls return to the
client through the ``awaiting_tools`` flow; delegation calls are resolved by the
``HandoffManager`` and never reach the client. A plain assistant reply ends the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import fusion_background as background
import fusion_session
import utils
from fusion_budget import FusionBudgetExceededError
from fusion_handoff import DELEGATE_CALL_PREFIX, HandoffManager
from fusion_harness import (
    delegation_name_policy,
    effective_delegation_names,
    lead_prompt_with_names,
)
from fusion_prompts import lead_system_prompt, lead_turn_reminder, sidekick_system_prompt

if TYPE_CHECKING:
    from fusion import FusionCoordinator, FusionRun

__all__ = ["HandoffManager", "advance", "uses_tool_delegation", "visible_lead_messages"]

BUDGET_STOP = "Not run: the run budget is exhausted."


def set_tool_schemas(run: FusionRun, tools: list[dict[str, Any]]) -> None:
    """Replace one run's schemas and refresh tool-delegation aliases."""
    run.tools = utils._convert_tools(tools)
    if run.structured and run.structured_plan is not None:
        run.tools = run._filter_tools()
    if run.handoffs is not None:
        refresh_tool_names(run)


def copy_refreshed_tool_state(target: FusionRun, source: FusionRun) -> None:
    """Reflect a persistent-store refresh on the caller's previously loaded run copy."""
    target.tools = source.tools
    target.main_messages = source.main_messages
    target.handoffs = source.handoffs
    target.fusion_lead_prompt_base = source.fusion_lead_prompt_base


def visible_lead_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop delegation calls and their results from lead messages shown in traces."""
    visible: list[dict[str, Any]] = []
    for message in messages:
        if str(message.get("tool_call_id", "")).startswith(DELEGATE_CALL_PREFIX):
            continue
        calls = message.get("tool_calls")
        if calls:
            kept = [c for c in calls if not str(c.get("id", "")).startswith(DELEGATE_CALL_PREFIX)]
            message = {**message, "tool_calls": kept}
        visible.append(message)
    return visible


def uses_tool_delegation(run: FusionRun, coordinator: FusionCoordinator) -> bool:
    """Return True for tool-delegation runs, fixing the mode on a run's first advance."""
    if run.handoffs is not None:
        return True
    if run.turns or run.status != "main_planning":
        return False
    if (run.tool_options.delegation or coordinator.config.delegation()) != "tool":
        return False
    prepare_run(run)
    return True


def prepare_run(run: FusionRun) -> None:
    """Replace the marker-protocol prompts with the lead and sidekick policy prompts."""
    history = run.main_messages[1:]
    base_parts: list[str] = []
    while history and history[0].get("role") == "system":
        base_parts.append(str(history.pop(0).get("content") or ""))
    base = "\n\n".join(part for part in base_parts if part) or None
    names = effective_delegation_names(run.tools or [])
    run.fusion_lead_prompt_base = lead_system_prompt(base)
    lead_prompt = lead_prompt_with_names(run.fusion_lead_prompt_base, names)
    run.main_messages = [{"role": "system", "content": lead_prompt}, *history]
    run.sidekick_messages = [{"role": "system", "content": sidekick_system_prompt()}]
    manager = HandoffManager(run.tool_options.workspace)
    manager.delegation_names = names
    run.handoffs = manager
    fusion_session.restore_sidekick(run, run.fusion_session_id)


def refresh_tool_names(run: FusionRun) -> None:
    """Refresh per-run delegation aliases and the lead prompt after client tools change."""
    manager = run.handoffs
    if manager is None:
        return
    previous = manager.delegation_names
    base = run.fusion_lead_prompt_base
    if base is None and run.main_messages:
        base = str(run.main_messages[0].get("content") or "")
        if any(previous[name] != name for name in previous):
            base = base.removesuffix(f"\n\n{delegation_name_policy(previous)}")
        run.fusion_lead_prompt_base = base
    manager.delegation_names = effective_delegation_names(run.tools or [])
    if run.main_messages and base is not None:
        run.main_messages[0] = {
            **run.main_messages[0],
            "content": lead_prompt_with_names(base, manager.delegation_names),
        }


def advance(
    run: FusionRun,
    tool_results: list[dict[str, Any]],
    request_id: str | None,
    message: str | None,
    coordinator: FusionCoordinator,
) -> dict[str, Any]:
    """Advance a tool-delegation run until it waits on client tools or completes."""
    manager = _manager(run)
    _check_request(run, tool_results, message)
    if message is not None:
        add_user_message(run, message)
    if run.status == "awaiting_tools":
        background.accept_client_results(manager, run, run._validate_tool_results(tool_results))
        run.pending_tool_calls = []
        run.status = "main_planning"
    try:
        event = _lead_loop(run, coordinator, request_id)
    except FusionBudgetExceededError:
        manager.close_open_calls(run, BUDGET_STOP)
        raise
    else:
        if run.status == "completed":
            fusion_session.save_sidekick(run, run.fusion_session_id)
        return event


def add_user_message(run: FusionRun, message: str) -> None:
    """Append a new user instruction to the lead conversation and reopen the run."""
    run._record_handoff("turn", summary=message[:200])
    run.latest_user = message
    run.main_messages.append({"role": "user", "content": message})
    run.report = None
    run.completed_via = ""
    run.status = "main_planning"


def _manager(run: FusionRun) -> HandoffManager:
    """Return the run's handoff manager, synchronizing live worker state first."""
    if run.handoffs is None:
        raise RuntimeError("run does not use tool delegation")
    background.sync_run(run)
    return run.handoffs


def _check_request(run: FusionRun, tool_results: list[dict[str, Any]], message: str | None) -> None:
    """Reject advance requests that do not fit the run's current state."""
    if run.cancelled:
        background.stop(run.run_id)
        raise RuntimeError("run is cancelled")
    if message is not None and tool_results:
        raise ValueError("cannot combine message follow-up with tool_results")
    if message is not None and run.status == "error":
        raise ValueError("cannot follow up an errored fusion run")
    if tool_results and run.status != "awaiting_tools":
        raise ValueError("tool_results are only valid when status is awaiting_tools")


def _lead_loop(
    run: FusionRun, coordinator: FusionCoordinator, request_id: str | None
) -> dict[str, Any]:
    """Alternate lead turns and tool resolution until the lead answers or must wait."""
    manager = _manager(run)
    for _ in range(run.run_max_turns):
        if run.budget is not None:
            run.budget.check_timeout()
        if manager.lead_calls:
            if not manager.resolve_lead_calls(run, coordinator):
                run.pending_tool_calls = background.client_calls(manager)
                run.status = "awaiting_tools"
                return run._ok_event(request_id)
            background.flush_lead_results(manager, run)
            _drain_user_messages(run)
        manager.take_completion_notification(run)
        text, calls = _lead_turn(run, coordinator, background.lead_tools(manager, run))
        if not calls:
            background.stop(run.run_id)
            run._complete_answer(text)
            return run._ok_event(request_id)
        background.open_lead_batch(manager, calls)
    raise FusionBudgetExceededError("lead reached the run turn limit")


def _lead_turn(
    run: FusionRun, coordinator: FusionCoordinator, tools: list[dict[str, Any]]
) -> tuple[str, list[dict[str, Any]]]:
    """Call the lead with the per-turn reminder at the tail of its context.

    The reminder is removed after the call so it never accumulates: the stored history,
    and so the cacheable prefix of the next turn, contains only real turns.
    """
    reminder = lead_turn_reminder()
    run.main_messages.append({"role": "user", "content": reminder})
    try:
        text, calls, _ = run._call_lane(coordinator, "main", tools=tools)
    finally:
        _drop_last_user_message(run.main_messages, reminder)
    return text, calls


def _drop_last_user_message(messages: list[dict[str, Any]], content: str) -> None:
    """Remove the newest user message with exactly this content."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and message.get("content") == content:
            del messages[index]
            return


def _drain_user_messages(run: FusionRun) -> None:
    """Apply user messages that arrived while the run waited on client tools."""
    queued, run.pending_user_messages = run.pending_user_messages, []
    for text in queued:
        add_user_message(run, text)
