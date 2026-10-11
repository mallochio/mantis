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
    if options.workspace is not None:
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
    controls: dict[str, Any] = {}
    if request.parallel_tool_calls is not None:
        controls["parallel_tool_calls"] = request.parallel_tool_calls
    requested_output = request.max_completion_tokens or request.max_tokens
    if requested_output is not None:
        controls["max_output_tokens"] = requested_output
    run.controls = controls
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

def trailing_tool_results(
    request: Any,
    decode_tool_id: Callable[[str], tuple[str, str]],
    content_to_str: Callable[[Any], str],
) -> tuple[list[dict[str, Any]], dict[str, str], str | None]:
    """Decode one trailing block of client tool results, if the request ends in one."""
    tool_results: list[dict[str, Any]] = []
    client_tool_ids: dict[str, str] = {}
    run_id: str | None = None
    index = len(request.messages) - 1
    while index >= 0 and request.messages[index].role == "tool":
        tool_msg = request.messages[index]
        try:
            rid, original_id = decode_tool_id(tool_msg.tool_call_id or "")
        except ValueError:
            break
        if run_id is None:
            run_id = rid
        elif run_id != rid:
            break
        client_tool_ids[str(tool_msg.tool_call_id)] = original_id
        result = _tool_result_entry(tool_msg, original_id, content_to_str)
        tool_results.insert(0, result)
        index -= 1
    return tool_results, client_tool_ids, run_id


def _tool_result_entry(
    tool_msg: Any, original_id: str, content_to_str: Callable[[Any], str]
) -> dict[str, Any]:
    """Build one decoded tool-result entry."""
    return {
        "tool_call_id": original_id,
        "content": content_to_str(tool_msg.content),
        "is_error": False,
    }
