"""Persistent sidekick handoffs for tool-based fusion delegation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import fusion_background as background
import fusion_background_store as state
import fusion_harness
import tool_exec
from fusion_background import tool_name, tool_result

if TYPE_CHECKING:
    from fusion import FusionCoordinator, FusionRun

LEAD_CALL_PREFIX = "lead:"
DELEGATE_CALL_PREFIX = "delegate:"
SIDEKICK_CALL_PREFIX = "sidekick:"
ESCALATION_PREFIX = "ESCALATE_TO_MAIN:"


@dataclass
class Handoff:
    """One lead-to-sidekick handoff and its outcome."""

    call_id: str
    message: str
    block: bool = True
    status: str = "running"
    report: str | None = None
    background: bool = False


class HandoffManager:
    """Own the persistent sidekick and the lead's open tool-call batch."""

    def __init__(self, workspace: str | None) -> None:
        """Start with no open batch, no handoff, and the given shared workspace."""
        self.workspace = workspace
        self.delegation_names = {"sidekick": "sidekick", "read_subagent": "read_subagent"}
        self.lead_calls: list[dict[str, Any]] = []
        self.lead_results: dict[str, dict[str, Any]] = {}
        self.current: Handoff | None = None
        self.last: Handoff | None = None
        self.sidekick_batch: list[dict[str, Any]] = []
        self.sidekick_held: list[dict[str, Any]] = []
        self.sidekick_waiting: list[dict[str, Any]] = []
        self.tool_rounds = 0
        self.error_streak = 0
        self.pending_notifications: list[dict[str, str]] = []
        self.notified_handoffs: set[str] = set()

    def resolve_lead_calls(self, run: FusionRun, coordinator: FusionCoordinator) -> bool:
        """Resolve delegation calls; return True once every call in the batch has a result."""
        for call in self.lead_calls:
            call_id = str(call["id"])
            if call_id in self.lead_results or not call_id.startswith(DELEGATE_CALL_PREFIX):
                continue
            result = self._delegation_result(run, coordinator, call)
            if result is None:
                return False
            self.lead_results[call_id] = result
        return all(str(call["id"]) in self.lead_results for call in self.lead_calls)

    def close_open_calls(self, run: FusionRun, text: str) -> None:
        """Pair every unanswered lead call with an error result so the transcript stays valid."""
        if self.current is not None and self.current.background:
            background.stop(run.run_id)
        if self.current is not None:
            self._finish(run, self.current, f"{ESCALATION_PREFIX} {text}")
        last = run.main_messages[-1] if run.main_messages else {}
        if not self.lead_calls and last.get("role") == "assistant" and last.get("tool_calls"):
            background.open_lead_batch(self, last["tool_calls"])
        if not self.lead_calls:
            return
        for call in self.lead_calls:
            call_id = str(call["id"])
            self.lead_results.setdefault(call_id, tool_result(call_id, text, is_error=True))
        background.flush_lead_results(self, run)

    def latest_report(self) -> str:
        """Return the newest sidekick report for ``read_subagent``."""
        return self.last.report if self.last is not None and self.last.report else "No report yet."

    def start(self, run: FusionRun, call_id: str, arguments: dict[str, Any]) -> str | None:
        """Open a handoff by appending the lead's message as a sidekick user turn."""
        message = str(arguments.get("message") or "").strip()
        if not message:
            return "The sidekick tool requires a non-empty message."
        self.current = Handoff(call_id, message, arguments.get("block", True) is not False)
        self.background_usage = state.usage_baseline(run)
        self.tool_rounds = self.error_streak = 0
        run.sidekick_messages.append({"role": "user", "content": message})
        return None

    def take_completion_notification(self, run: FusionRun) -> None:
        """Append each completed background report once before a lead turn."""
        for item in self.pending_notifications:
            call_id, report = item["call_id"], item["report"]
            if call_id not in self.notified_handoffs:
                content = f"<handoff_completion>\n{report}\n</handoff_completion>"
                run.main_messages.append({"role": "user", "content": content})
                self.notified_handoffs.add(call_id)

    def interrupt(self, run: FusionRun, call_id: str, message: str) -> dict[str, Any]:
        """Queue a lead-authored interrupt for the active background sidekick."""
        background.interrupt(run.run_id, message)
        return tool_result(
            call_id, "The sidekick received the interrupt and will use it at its next turn."
        )

    def drive(self, run: FusionRun, coordinator: FusionCoordinator) -> str | None:
        """Run the sidekick until it reports; return None while it waits on client tools."""
        handoff = self.current
        if handoff is None or self.sidekick_waiting:
            return self.latest_report() if handoff is None else None
        handoff.status = "running"
        for _ in range(run.run_max_turns):
            stop = self._stop_reason(run, handoff)
            if stop is not None:
                return self._finish(run, handoff, f"{ESCALATION_PREFIX} {stop}")
            self._inject_interrupts(run)
            tools = background.sidekick_tools(self, run)
            try:
                text, calls = state.sidekick_turn(self, run, coordinator, tools)
            except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
                return self._finish(run, handoff, f"{ESCALATION_PREFIX} {exc}")
            stop = self._stop_reason(run, handoff)
            if stop is not None:
                return self._finish(run, handoff, f"{ESCALATION_PREFIX} {stop}")
            if not calls and not self._inject_interrupts(run):
                return self._finish(run, handoff, background.report_text(run, text))
            if calls:
                self.tool_rounds += 1
                if not self._execute(run, calls):
                    handoff.status = "waiting_tools"
                    return None
        return self._finish(run, handoff, f"{ESCALATION_PREFIX} sidekick reached the turn limit")

    def _stop_reason(self, run: FusionRun, handoff: Handoff) -> str | None:
        """Return the cancellation, run guard, or sidekick limit reason."""
        if (
            run.cancelled
            or run.finished
            or (handoff.background and background.should_stop(run.run_id))
        ):
            return "run cancelled"
        if run.budget is not None:
            try:
                run.budget.check_timeout()
            except RuntimeError as exc:
                return str(exc)
        return self._limit_reason(run)

    def _inject_interrupts(self, run: FusionRun) -> bool:
        """Append queued interrupts as tagged sidekick user turns."""
        messages = background.take_interrupts(run.run_id)
        run.sidekick_messages.extend(
            {"role": "user", "content": f"<sidekick_interrupt>\n{text}\n</sidekick_interrupt>"}
            for text in messages
        )
        return bool(messages)

    def _delegation_result(
        self, run: FusionRun, coordinator: FusionCoordinator, call: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Resolve one delegation call, or return None while the sidekick waits on tools."""
        call_id, arguments = str(call["id"]), background.call_arguments(call)
        name = fusion_harness.canonicalize(tool_name(call), self.delegation_names)
        if name == "read_subagent":
            return tool_result(call_id, self._read_subagent(run, coordinator, arguments))
        if self.current is not None and background.running(run.run_id):
            return self.interrupt(run, call_id, str(arguments.get("message") or ""))
        if self.current is None or self.current.call_id != call_id:
            error = self.start(run, call_id, arguments)
            if error is not None:
                return tool_result(call_id, error, is_error=True)
        return self._start_or_drive(run, coordinator, call_id, arguments)

    def _start_or_drive(
        self, run: FusionRun, coordinator: FusionCoordinator, call_id: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Start eligible asynchronous work or complete it in the lead request."""
        if (
            self.current is not None
            and args.get("block", True) is False
            and self._can_background(run)
        ):
            self.current.background = True
            if background.start(run, self, coordinator):
                return tool_result(call_id, "The sidekick started in the background.")
            return self.interrupt(run, call_id, str(args.get("message") or ""))
        report = self.drive(run, coordinator)
        if report is None:
            return None
        if args.get("block", True) is False and run.tools:
            report = (
                f"Non-blocking handoff requires server-side tools; waited for report.\n{report}"
            )
        return tool_result(call_id, report)

    def _can_background(self, run: FusionRun) -> bool:
        """Require server execution and no tools that would round-trip to the client."""
        if not (self.workspace or run.tool_options.server_execution):
            return False
        server = {tool_name(schema) for schema in background.server_schemas(self, run)}
        tools = background.sidekick_tools(self, run) or []
        return all(tool_name(tool) in server for tool in tools)

    def _read_subagent(
        self, run: FusionRun, coordinator: FusionCoordinator, args: dict[str, Any]
    ) -> str:
        """Return the latest status or wait for the active background handoff."""
        if self.current is None:
            return self.latest_report()
        if args.get("block", True) is False:
            if not background.running(run.run_id):
                background.sync_run(run)
                return self.latest_report()
            return f"Sidekick status: {self.current.status}. {self.latest_report()}"
        raw_timeout = args.get("timeout")
        if raw_timeout is None:
            raw_timeout = coordinator.config._load().get("handoff_wait_timeout_s", 2700)
        try:
            timeout = max(0.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout = 2700.0
        if not background.wait(run.run_id, timeout):
            return f"Sidekick is still running; wait timed out after {timeout:g} seconds."
        background.sync_run(run)
        return self.latest_report()

    def _limit_reason(self, run: FusionRun) -> str | None:
        """Return why the sidekick must hand back, using the run's configured limits."""
        errors = run.sidekick_max_consecutive_errors
        if errors and self.error_streak >= errors:
            return f"sidekick hit {errors} consecutive failed tool results"
        rounds = run.sidekick_max_tool_rounds
        if rounds and self.tool_rounds >= rounds:
            return "sidekick exceeded tool budget"
        return None

    def _execute(self, run: FusionRun, calls: list[dict[str, Any]]) -> bool:
        """Run server-side sidekick calls; return False when some need the client."""
        server_names = {tool_name(schema) for schema in background.server_schemas(self, run)}
        ready: list[dict[str, Any]] = []
        server: list[dict[str, Any]] = []
        waiting: list[dict[str, Any]] = []
        for call in calls:
            call["id"] = f"{SIDEKICK_CALL_PREFIX}{call.get('id', '')}"
            denial = tool_exec.lead_only_command_error(background.call_arguments(call))
            if denial is not None:
                ready.append(tool_result(call["id"], denial, is_error=True))
            else:
                (server if tool_name(call) in server_names else waiting).append(call)
        if server:
            ready.extend(
                background.execute_server_calls(self, run, server, self._workspace_path(run))
            )
        if waiting:
            self.sidekick_batch, self.sidekick_held, self.sidekick_waiting = calls, ready, waiting
            return False
        self._append_sidekick_results(run, calls, ready)
        return True

    def _workspace_path(self, run: FusionRun) -> Path:
        """Return the shared client workspace, or the run's private sandbox."""
        return Path(self.workspace) if self.workspace else tool_exec.workspace_for(run.run_id)

    def _append_sidekick_results(
        self, run: FusionRun, calls: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> None:
        """Append sidekick results in call order and update the failure streak."""
        order = {str(call["id"]): index for index, call in enumerate(calls)}
        ordered = sorted(results, key=lambda r: order.get(str(r["tool_call_id"]), len(order)))
        for result in ordered:
            self.error_streak = self.error_streak + 1 if result.get("is_error") else 0
        background.append_results(run, "sidekick", calls, ordered)

    def _finish(self, run: FusionRun, handoff: Handoff, report: str) -> str:
        """Close a handoff, record it in the run's handoff history, and return the report."""
        handoff.report = report
        status = "escalated" if report.startswith(ESCALATION_PREFIX) else "completed"
        cancelled = run.cancelled or report.endswith("run cancelled")
        handoff.status = "cancelled" if cancelled else status
        self.current = None
        self.last = handoff
        run.subtask_report = report
        run._record_handoff(
            "handoff", message=handoff.message[:500], report=report[:500], status=handoff.status
        )
        if handoff.background and handoff.status != "cancelled":
            self.pending_notifications.append({"call_id": handoff.call_id, "report": report})
        return report
