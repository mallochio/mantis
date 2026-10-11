"""Request context and new-run setup for the Fusion HTTP adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import fusion
import fusion_harness
from fusion_types import FusionToolOptions


def _resolve_fusion_workspace(
    request: Any, headers: dict[str, str] | None, content_to_str: Callable[[Any], str]
) -> Any:
    """Apply an allowlisted workspace discovered from request and harness context."""
    options = request.tool_options or FusionToolOptions()
    mode = options.delegation or fusion._fusion_config().delegation()
    if mode != "tool" or options.workspace is not None:
        return request
    header = next(
        (value for key, value in (headers or {}).items() if key.lower() == "x-mantis-workspace"),
        None,
    )
    prompt = "\n".join(
        content_to_str(message.content)
        for message in request.messages
        if message.role in {"system", "developer"}
    )
    default = fusion._fusion_config()._load().get("default_workspace")
    workspace = fusion_harness.resolve_workspace(None, header, prompt, default)
    if workspace is None:
        return request
    updated = request.model_copy(
        update={"tool_options": options.model_copy(update={"workspace": workspace})}
    )
    return updated


def _fusion_session_id(headers: dict[str, str] | None, request: Any) -> str | None:
    """Return an explicit session identity without inventing one for stateless calls."""
    session_headers = {"x-mantis-session-id", "x-mantis-session", "x-route-session"}
    header_id = next(
        (
            value.strip()
            for key, value in (headers or {}).items()
            if key.lower() in session_headers and value.strip()
        ),
        None,
    )
    if header_id:
        return header_id
    if isinstance(request.metadata, dict):
        value = request.metadata.get("session_id") or request.metadata.get("sessionId")
        if isinstance(value, str) and value:
            return value
    return None


def create_chat_run(request: Any, brief: str, session_id: str | None) -> Any:
    """Create a run with its original request history and explicit session identity."""
    tools = [tool.model_dump() for tool in (request.tools or [])]
    messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
    run = fusion.create_fusion_run(
        brief,
        tools,
        messages=messages,
        delegation_mode="available",
        worker_profiles=request.worker_profiles,
        budget=request.budget,
        tool_options=request.tool_options,
    )
    run.fusion_session_id = session_id
    apply_request_controls(run, request)
    return run


def apply_request_controls(run: Any, request: Any) -> None:
    """Store each HTTP turn's tool constraints on the existing native run fields."""
    choice = request.tool_choice
    run.tool_choice = choice.model_dump() if hasattr(choice, "model_dump") else choice
    run.controls = (
        {"parallel_tool_calls": request.parallel_tool_calls}
        if request.parallel_tool_calls is not None
        else {}
    )
    fusion._put_run(run)


def refresh_request_controls(run_id: str, request: Any) -> None:
    """Refresh controls before resuming the run with client tool results."""
    apply_request_controls(fusion.get_run(run_id), request)


def hidden_tool_names(request: Any, run_id: str | None) -> set[str]:
    """Return effective server tool names for this request or its stored run."""
    run = fusion.try_get_fusion_run(run_id) if run_id else None
    if isinstance(run, fusion.FusionRun) and run.handoffs is not None:
        if request.tools is not None:
            fusion.refresh_fusion_run_tools(run, [tool.model_dump() for tool in request.tools])
        return set(run.handoffs.delegation_names.values())
    client_tools = (
        run.tools
        if isinstance(run, fusion.FusionRun)
        else [tool.model_dump() for tool in request.tools or []]
    )
    return set(fusion_harness.effective_delegation_names(client_tools or []).values())
