#!/usr/bin/env python3
"""Tool schemas and workspace executor for the routed A/B runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the run workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the run workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the run workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the run workspace.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace one exact substring in a workspace file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search workspace files for a substring.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)


def parse_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Return the decoded arguments of one tool call."""
    function = call.get("function") or {}
    try:
        args = json.loads(function.get("arguments") or "{}")
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def _run_bash(args: dict[str, Any], workspace: Path) -> str:
    """Run one sandboxed shell command inside the workspace."""
    completed = subprocess.run(  # noqa: S603 - reason: fixed bash -c shape with a workspace cwd jail
        ["bash", "-c", str(args.get("command") or "")],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return f"exit={completed.returncode}\n{completed.stdout}{completed.stderr}"


def _read(args: dict[str, Any], workspace: Path) -> str:
    """Read one workspace file."""
    return (workspace / str(args.get("path") or "")).read_text()[:20000]


def _write(args: dict[str, Any], workspace: Path) -> str:
    """Write one workspace file."""
    target = workspace / str(args.get("path") or "")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(args.get("content") or ""))
    return "written"


def _list(workspace: Path) -> str:
    """List every path under the workspace."""
    return "\n".join(str(path.relative_to(workspace)) for path in workspace.rglob("*"))


def _edit(args: dict[str, Any], workspace: Path) -> str:
    """Replace one exact substring in a workspace file."""
    target = workspace / str(args.get("path") or "")
    text = target.read_text()
    target.write_text(text.replace(str(args.get("old")), str(args.get("new")), 1))
    return "edited"


def _search(args: dict[str, Any], workspace: Path) -> str:
    """Return workspace files containing the query substring."""
    query = str(args.get("query") or "")
    hits = [
        str(path.relative_to(workspace))
        for path in workspace.rglob("*")
        if path.is_file() and query in path.read_text(errors="ignore")
    ]
    return "\n".join(hits)


def execute_tool_call(call: dict[str, Any], workspace: Path) -> str:
    """Run one client-side tool call inside the run workspace."""
    args = parse_arguments(call)
    name = str((call.get("function") or {}).get("name") or "")
    if name not in _NAMES:
        return f"unknown tool {name!r}"
    try:
        if name == "bash":
            return _run_bash(args, workspace)
        if name == "read_file":
            return _read(args, workspace)
        if name == "write_file":
            return _write(args, workspace)
        if name == "list_files":
            return _list(workspace)
        if name == "edit_file":
            return _edit(args, workspace)
        return _search(args, workspace)
    except (OSError, ValueError) as error:
        return f"tool error: {error}"
