"""mantis-fusion: catalog-driven lead/sidekick resumable orchestration."""

from __future__ import annotations

import json
import os
import re
import threading
import tomllib
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import model_catalog
import providers
import serve_config

from runs import (
    RUN_STORE,
    NativeRun,
    _file_put,
    _file_run_lock,
    _redis_put,
    _redis_run_lock,
    get_run,
)

__all__ = [
    "FusionCoordinator",
    "FusionRun",
    "advance_fusion_run",
    "FUSION_STATUS",
]

FUSION_STATUS = frozenset(
    {"main_planning", "sidekick_pending", "awaiting_tools", "main_review", "completed", "error"}
)

MAIN_PREAMBLE = (
    "You are the lead engineer on a software task. Your job is to plan work and "
    "review a sidekick's output. When given a brief, respond with exactly two "
    "sections: 'PLAN:' containing the high-level plan, and 'BRIEF:' containing a "
    "self-contained brief for the sidekick. "
    "When reviewing a sidekick report, reply exactly 'ACCEPT' if the report is "
    "satisfactory. Otherwise reply 'FOLLOW_UP:' followed by concise feedback."
)

SIDEKICK_PREAMBLE = (
    "You are a fast, cheap coding sidekick. Implement, test, and lint according to "
    "the brief you are given. You may use the provided tools. When finished, "
    "return a concise final report."
)

REVIEW_PROMPT = (
    "The sidekick produced the following report. Review it. If it is satisfactory, "
    "reply exactly 'ACCEPT'. Otherwise reply 'FOLLOW_UP:' followed by concise feedback "
    "so the sidekick can revise.\n\nReport:\n"
)


class FusionConfig:
    """Runtime fusion settings loaded from the shared catalog."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or model_catalog.catalog_path()[0]
        self._raw: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if self._raw is not None:
            return self._raw
        with self._lock:
            if self._raw is not None:
                return self._raw
            if not self.path.exists():
                self._raw = {}
            else:
                with self.path.open("rb") as handle:
                    root = tomllib.load(handle)
                self._raw = root.get("fusion", {})
            return self._raw

    def main_slot(self) -> str:
        return (
            self._load().get("main")
            or os.environ.get("MANTIS_FUSION_MAIN_MODEL")
            or "gpt-5_6-sol"
        )

    def sidekick_slot(self) -> str:
        return (
            self._load().get("sidekick")
            or os.environ.get("MANTIS_FUSION_SIDEKICK_MODEL")
            or "gemini-3_6-flash"
        )

    def max_follow_ups(self) -> int:
        raw = self._load().get("max_follow_ups") or os.environ.get(
            "MANTIS_FUSION_MAX_FOLLOW_UPS", "3"
        )
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 3


_FUSION_CONFIG = FusionConfig()


def _fusion_config() -> FusionConfig:
    """Return the global FusionConfig, reloaded on first call in a new process."""
    return _FUSION_CONFIG


class FusionCoordinator:
    """Thin wrapper that calls catalog-bound worker slots."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or _fusion_config()
        self.main_slot = self.config.main_slot()
        self.sidekick_slot = self.config.sidekick_slot()
        self.max_follow_ups = self.config.max_follow_ups()

    def _call_worker(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Call a worker slot and return (text, tool_calls, usage)."""
        data = providers._provider_response(slot, messages, 4096, 0.7, tools)
        msg = data["choices"][0]["message"]
        text = str(msg.get("content") or "")
        tcs = msg.get("tool_calls") or []
        calls: list[dict[str, Any]] = []
        for tc in tcs:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or ""
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            call = {
                "id": str(tc.get("id", f"call_{len(calls)}")),
                "type": "function",
                "function": {"name": str(fn.get("name", "")), "arguments": raw_args},
            }
            calls.append(call)
        usage = data.get("usage") or {}
        return text, calls, usage


class FusionRun(NativeRun):
    """A resumable lead/sidekick run whose roles are bound by the catalog."""

    def __init__(self, run_id: str, brief: str, tools: list[dict[str, Any]] | None = None) -> None:
        super().__init__(run_id)
        self.kind = "fusion"
        self.brief = brief
        self.tools = tools or []
        self.main_messages: list[dict[str, Any]] = [
            {"role": "system", "content": MAIN_PREAMBLE},
            {"role": "user", "content": brief},
        ]
        self.sidekick_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SIDEKICK_PREAMBLE},
        ]
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.report: str | None = None
        self.plan: str = ""
        self.sidekick_brief: str = ""
        self.error: str | None = None
        self.follow_up_count = 0
        self.status = "main_planning"

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Ensure new fields are present for forward compatibility.
        for key, default in {
            "plan": "",
            "sidekick_brief": "",
            "follow_up_count": 0,
        }.items():
            if not hasattr(self, key):
                setattr(self, key, default)

    def _append_sidekick_tool_call(self, text: str, tool_calls: list[dict[str, Any]]) -> None:
        self.sidekick_messages.append(
            {"role": "assistant", "content": text, "tool_calls": tool_calls}
        )

    def _append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        for result in tool_results:
            self.sidekick_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": result.get("content", ""),
                    "is_error": bool(result.get("is_error", False)),
                }
            )

    def _call_main(
        self,
        coordinator: FusionCoordinator,
        prompt: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if prompt is not None:
            self.main_messages.append({"role": "user", "content": prompt})
        text, calls, usage = coordinator._call_worker(
            coordinator.main_slot, self.main_messages, None
        )
        self.main_messages.append({"role": "assistant", "content": text, "tool_calls": calls})
        self.add_usage(usage, model=coordinator.main_slot)
        self.record_activity(
            "main_turn",
            role="main",
            model=coordinator.main_slot,
            status="completed",
            summary=text[:200] if text else "tool-calls",
        )
        return text, calls, usage

    def _call_sidekick(
        self,
        coordinator: FusionCoordinator,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        text, calls, usage = coordinator._call_worker(
            coordinator.sidekick_slot, self.sidekick_messages, self.tools
        )
        self.add_usage(usage, model=coordinator.sidekick_slot)
        self.record_activity(
            "sidekick_turn",
            role="sidekick",
            model=coordinator.sidekick_slot,
            status="completed",
            summary=text[:200] if text else "tool-calls",
        )
        return text, calls, usage

    def _parse_main_plan(self, text: str) -> tuple[str, str]:
        plan_match = re.search(r"PLAN:(.*?)(?:BRIEF:|$)", text, re.DOTALL)
        brief_match = re.search(r"BRIEF:(.*)", text, re.DOTALL)
        if not plan_match or not brief_match:
            raise ValueError("main did not produce both PLAN and BRIEF sections")
        plan = plan_match.group(1).strip()
        sidekick_brief = brief_match.group(1).strip()
        if not plan or not sidekick_brief:
            raise ValueError("main produced empty PLAN or BRIEF")
        return plan, sidekick_brief

    def _parse_main_review(self, text: str) -> tuple[bool, str]:
        stripped = text.strip()
        if re.search(r"\bACCEPT\b", stripped):
            return True, ""
        follow_match = re.match(r"FOLLOW_UP:\s*(.*)", stripped, re.DOTALL)
        if follow_match:
            return False, follow_match.group(1).strip()
        # Default to acceptance when the main model does not issue a clear follow-up.
        return True, ""

    def _validate_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        if not isinstance(tool_results, list):
            raise TypeError("tool_results must be a list")
        pending_ids = {str(tc.get("id")) for tc in self.pending_tool_calls if tc.get("id")}
        seen: set[str] = set()
        for item in tool_results:
            tid = str(item.get("tool_call_id"))
            if tid in pending_ids and tid not in seen:
                seen.add(tid)
        missing = pending_ids - seen
        if missing:
            raise ValueError(f"missing tool results for {sorted(missing)}")

    def _error_event(self, exc: Exception, request_id: str | None) -> dict[str, Any]:
        self.status = "error"
        self.error = str(exc)
        self.record_activity(
            "error",
            role="fusion",
            model="",
            status="error",
            detail=self.error,
        )
        event = {
            "run_id": self.run_id,
            "status": "error",
            "report": None,
            "pending_tool_calls": None,
            "usage": self.usage,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def _ok_event(self, request_id: str | None) -> dict[str, Any]:
        event = {
            "run_id": self.run_id,
            "status": self.status,
            "report": self.report if self.status == "completed" else None,
            "pending_tool_calls": (
                self.pending_tool_calls if self.status == "awaiting_tools" else None
            ),
            "usage": self.usage,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def advance(
        self,
        tool_results: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        message: str | None = None,
        coordinator: FusionCoordinator | None = None,
    ) -> dict[str, Any]:
        """Run the Fusion state machine until suspension, completion, or error."""
        coordinator = coordinator or FusionCoordinator()
        tool_results = tool_results or []
        try:
            return self._advance(tool_results, request_id, message, coordinator)
        except Exception as exc:  # noqa: BLE001 - catch-all guard for worker/state errors
            return self._error_event(exc, request_id)

    def _advance(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        if self.cancelled:
            raise RuntimeError("run is cancelled")

        # Apply client tool results only when awaiting tools.
        if self.status == "awaiting_tools":
            self._validate_tool_results(tool_results)
            self._append_tool_results(tool_results)
            self.pending_tool_calls = []
            self.status = "sidekick_pending"
        elif tool_results:
            raise ValueError("tool_results are only valid when status is awaiting_tools")

        # Main planning on a fresh run.
        if self.status == "main_planning":
            main_text, _, _ = self._call_main(coordinator)
            self.plan, self.sidekick_brief = self._parse_main_plan(main_text)
            self.sidekick_messages.append(
                {"role": "user", "content": self.sidekick_brief}
            )
            self.status = "sidekick_pending"

        # Bounded sidekick-main loop.
        max_iterations = max(1, coordinator.max_follow_ups + 1)
        for _ in range(max_iterations * 4):  # generous step ceiling
            if self.status == "sidekick_pending":
                sidekick_text, sidekick_calls, _ = self._call_sidekick(coordinator)
                if sidekick_calls:
                    self._append_sidekick_tool_call(sidekick_text, sidekick_calls)
                    self.pending_tool_calls = sidekick_calls
                    self.status = "awaiting_tools"
                    break
                # Sidekick produced a report; ask the main to review.
                self.status = "main_review"
                review_prompt = REVIEW_PROMPT + sidekick_text
                review_text, _, _ = self._call_main(coordinator, review_prompt)
                accepted, feedback = self._parse_main_review(review_text)
                if accepted:
                    self.report = sidekick_text
                    self.status = "completed"
                    break
                # Main requested a sidekick follow-up.
                self.follow_up_count += 1
                if self.follow_up_count > coordinator.max_follow_ups:
                    self.report = sidekick_text
                    self.status = "completed"
                    break
                self.sidekick_messages.append({"role": "user", "content": feedback})
                self.record_activity(
                    "follow_up",
                    role="main",
                    model=coordinator.main_slot,
                    status="completed",
                    summary=feedback[:200],
                )
                self.status = "sidekick_pending"
                continue
            if self.status in ("completed", "awaiting_tools", "error"):
                break

        return self._ok_event(request_id)

    def advance_idempotent(
        self,
        tool_results: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        message: str | None = None,
        coordinator: FusionCoordinator | None = None,
    ) -> dict[str, Any]:
        if request_id is None:
            return self.advance(tool_results, request_id, message, coordinator)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cached
            event = self.advance(tool_results, request_id, message, coordinator)
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event


def _put_run(run: NativeRun) -> None:
    if RUN_STORE == "redis":
        _redis_put(run)
    elif RUN_STORE == "file":
        _file_put(run)
    else:
        serve_config._runs[run.run_id] = run


def _run_lock(run_id: str) -> Any:
    if RUN_STORE == "redis":
        return _redis_run_lock(run_id)
    if RUN_STORE == "file":
        return _file_run_lock(run_id)
    return nullcontext()


def advance_fusion_run(
    run_id: str,
    request_id: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Load, advance, and save a FusionRun with the same locking as advance_run."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        run.in_flight += 1
        _put_run(run)
        try:
            serve_config._history_context.active_run = run
            coordinator = FusionCoordinator()
            event = run.advance_idempotent(tool_results, request_id, message, coordinator)
            if event.get("status") in ("completed", "error"):
                # Persist learning record only for terminal states.
                pass
            return event
        finally:
            serve_config._history_context.active_run = None
            run.in_flight -= 1
            run.touch()
            _put_run(run)


def create_fusion_run(brief: str, tools: list[dict[str, Any]] | None = None) -> FusionRun:
    run_id = uuid.uuid4().hex
    run = FusionRun(run_id, brief, tools)
    _put_run(run)
    return run


def fusion_run_status(run_id: str) -> dict[str, Any]:
    """Return the current status of a FusionRun without advancing it."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "report": run.report if run.status == "completed" else None,
            "pending_tool_calls": (
                run.pending_tool_calls if run.status == "awaiting_tools" else None
            ),
            "usage": run.usage,
            "activity": run._activity,
        }
