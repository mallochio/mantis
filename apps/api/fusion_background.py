"""Background execution and synchronization for Fusion handoffs."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import fusion_background_store as store
import fusion_harness
import serve_config
import tool_exec
import utils
from fusion_prompts import READ_SUBAGENT_TOOL, SIDEKICK_TOOL

_TASKS: dict[str, BackgroundTask] = {}
_TASKS_LOCK = threading.Lock()
_RESULT_LIMIT = 64


@dataclass
class BackgroundTask:
    """Hold process-local controls for one background handoff."""

    run: Any
    manager: Any
    coordinator: Any
    stop: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    interrupts: list[str] = field(default_factory=list)
    thread: threading.Thread | None = None
    persisted: bool = False
    consumed: bool = False


def start(run: Any, manager: Any, coordinator: Any) -> bool:
    """Start one background handoff for a run unless one is already active."""
    with _TASKS_LOCK:
        existing = _TASKS.get(run.run_id)
        if existing is not None and existing.thread is not None and existing.thread.is_alive():
            return False
        task = BackgroundTask(run, manager, coordinator)
        thread = threading.Thread(target=_drive, args=(task,), daemon=True)
        task.thread = thread
        _TASKS[run.run_id] = task
        thread.start()
    return True


def running(run_id: str) -> bool:
    """Return whether a background handoff thread is still active."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
        return bool(task and task.thread and task.thread.is_alive() and not task.done.is_set())


def interrupt(run_id: str, message: str) -> bool:
    """Queue a lead message for the next sidekick turn boundary."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
        if task is None or task.done.is_set():
            return False
        task.interrupts.append(message)
        return True


def take_interrupts(run_id: str) -> list[str]:
    """Remove and return queued interrupt messages for the sidekick."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
        if task is None:
            return []
        interrupts, task.interrupts = task.interrupts, []
        return interrupts


def wait(run_id: str, timeout: float) -> bool:
    """Wait for a completed handoff result or its worker event."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
    return task is not None and task.done.wait(timeout)


def stop(run_id: str) -> None:
    """Ask the background handoff to stop at its next turn boundary."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
    if task is not None:
        task.stop.set()


def should_stop(run_id: str) -> bool:
    """Return whether cancellation or run completion stopped the handoff."""
    with _TASKS_LOCK:
        task = _TASKS.get(run_id)
    return task is None or task.stop.is_set()


def sync_run(run: Any) -> None:
    """Copy completed or live sidekick state before a run is persisted."""
    with _TASKS_LOCK:
        task = _TASKS.get(run.run_id)
    if task is None or task.run is run:
        return
    store.merge(run, task.run)
    if task.done.is_set():
        with _TASKS_LOCK:
            task.consumed = True
            if task.persisted or task.thread is None:
                _TASKS.pop(run.run_id, None)


def tool_name(call: dict[str, Any]) -> str:
    """Return the function name of one tool call or tool schema."""
    return str((call.get("function") or {}).get("name", ""))


def call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Parse one tool call's JSON arguments, returning an empty dict when malformed."""
    raw = (call.get("function") or {}).get("arguments") or "{}"
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def tool_result(call_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    """Build one tool result in the shape the run transcripts store."""
    return {"tool_call_id": call_id, "content": content, "is_error": is_error}


def lead_tools(manager: Any, run: Any) -> list[dict[str, Any]]:
    """Return client tools plus the lead's private delegation tools."""
    if run.tool_choice == "none":
        return []
    names = manager.delegation_names
    client = [tool for tool in run.tools or [] if tool_name(tool) not in names.values()]
    schemas = fusion_harness.rename_schemas((SIDEKICK_TOOL, READ_SUBAGENT_TOOL), names)
    return [*client, *schemas]


def server_schemas(manager: Any, run: Any) -> list[dict[str, Any]]:
    """Return tool schemas executable by the sidekick on the server."""
    options = run.tool_options
    if manager.workspace:
        return tool_exec.server_tool_schemas(options.enabled or ["files", "shell", "code"])
    if not options.server_execution:
        return []
    declared = {tool_name(tool) for tool in run.tools or []}
    schemas = tool_exec.server_tool_schemas(options.enabled)
    return [schema for schema in schemas if tool_name(schema) not in declared]


def sidekick_tools(manager: Any, run: Any) -> list[dict[str, Any]] | None:
    """Return sidekick-visible client tools and server-side tools."""
    server = server_schemas(manager, run)
    hidden = {tool_name(schema) for schema in server} | set(manager.delegation_names.values())
    client = utils._filter_tools_by_options(run.tools, run.tool_options)
    return [*(tool for tool in client if tool_name(tool) not in hidden), *server] or None


def execute_server_calls(
    manager: Any, run: Any, calls: list[dict[str, Any]], workspace: Any
) -> list[dict[str, Any]]:
    """Check stop state before every dispatch and execute independent calls in parallel."""

    def dispatch(call: dict[str, Any]) -> dict[str, Any]:
        """Execute one call after checking the handoff stop state."""
        return _dispatch_call(manager, run, call, workspace)

    if len(calls) < 2:
        return [dispatch(call) for call in calls]
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        return list(executor.map(dispatch, calls))


def _dispatch_call(manager: Any, run: Any, call: dict[str, Any], workspace: Any) -> dict[str, Any]:
    """Execute a tool only while its sidekick handoff remains active."""
    stopped = (
        run.cancelled
        or run.finished
        or (manager.current is not None and manager.current.background and should_stop(run.run_id))
    )
    if stopped:
        return tool_result(call["id"], "run cancelled", is_error=True)
    return tool_exec.execute_calls([call], workspace)[0]


def open_lead_batch(manager: Any, calls: list[dict[str, Any]]) -> None:
    """Namespace a lead tool-call batch and store it on its manager."""
    for call in calls:
        prefix = "delegate:" if tool_name(call) in manager.delegation_names.values() else "lead:"
        call["id"] = f"{prefix}{call.get('id', '')}"
    manager.lead_calls = list(calls)
    manager.lead_results = {}


def client_calls(manager: Any) -> list[dict[str, Any]]:
    """Return the open calls the client must execute."""
    results = manager.lead_results
    lead = [
        call
        for call in manager.lead_calls
        if call["id"].startswith("lead:") and call["id"] not in results
    ]
    return [*lead, *manager.sidekick_waiting]


def accept_client_results(manager: Any, run: Any, results: list[dict[str, Any]]) -> None:
    """Route client tool results to the lead batch or a waiting sidekick turn."""
    waiting = {str(call["id"]) for call in manager.sidekick_waiting}
    for result in results:
        if str(result["tool_call_id"]) not in waiting:
            manager.lead_results[str(result["tool_call_id"])] = result
    if waiting:
        sidekick = [result for result in results if str(result["tool_call_id"]) in waiting]
        sidekick = [*manager.sidekick_held, *sidekick]
        manager._append_sidekick_results(run, manager.sidekick_batch, sidekick)
        manager.sidekick_batch, manager.sidekick_held, manager.sidekick_waiting = [], [], []


def flush_lead_results(manager: Any, run: Any) -> None:
    """Append a finished call batch to the lead transcript in call order."""
    results = [manager.lead_results[str(call["id"])] for call in manager.lead_calls]
    append_results(run, "main", manager.lead_calls, results)
    manager.lead_calls, manager.lead_results = [], {}


def append_results(
    run: Any, role: str, calls: list[dict[str, Any]], results: list[dict[str, Any]]
) -> None:
    """Append tool results to one role's transcript through the run's shared recorder."""
    with run.lock:
        previous_role, previous_calls = run.active_role, run.pending_tool_calls
        run.active_role = role
        run.pending_tool_calls = calls
        try:
            run._append_tool_results(results)
        finally:
            run.pending_tool_calls = previous_calls
            run.active_role = previous_role


def report_text(text: str) -> str:
    """Return sidekick final text, marking escalations for the lead."""
    return text.strip() or "The sidekick finished without a report."


def _drive(task: BackgroundTask) -> None:
    """Run a handoff and persist its plain-data result before releasing the worker."""
    context = serve_config._history_context
    previous = getattr(context, "active_run", None)
    context.active_run = task.run
    try:
        try:
            task.manager.drive(task.run, task.coordinator)
        except (RuntimeError, ValueError, TypeError, KeyError, OSError) as exc:
            handoff = task.manager.current
            if handoff is not None:
                task.manager._finish(task.run, handoff, f"ESCALATE_TO_MAIN: {exc}")
    finally:
        context.active_run = previous
        task.done.set()
    persisted = False
    try:
        persisted = store.persist(task.run, task)
    finally:
        with _TASKS_LOCK:
            task.persisted = persisted
            task.thread = None
            task.manager = task.coordinator = None
            if persisted or task.consumed:
                _TASKS.pop(task.run.run_id, None)
            else:
                completed = [key for key, item in _TASKS.items() if item.thread is None]
                while len(completed) > _RESULT_LIMIT:
                    _TASKS.pop(completed.pop(0), None)
