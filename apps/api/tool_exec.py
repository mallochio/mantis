"""Opt-in server-side Fusion tools scoped to one run workspace."""

from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import serve_config

SERVER_TOOL_NAMES = frozenset(
    {
        "list_files",
        "read_file",
        "search_files",
        "write_file",
        "edit_file",
        "bash",
        "shell",
        "sh",
        "execute_command",
        "execute_code",
        "python",
        "exec",
        "run_code",
    }
)


def workspace_for(run_id: str, root: Path | None = None) -> Path:
    base = (
        root or Path(os.environ.get("MANTIS_FUSION_WORKSPACE_ROOT", ".mantis/fusion"))
    ).resolve()
    workspace = (base / run_id).resolve()
    if workspace.parent != base:
        raise ValueError("invalid run id")
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _path(workspace: Path, value: Any) -> Path:
    path = (workspace / str(value or ".")).resolve()
    if path != workspace and workspace not in path.parents:
        raise ValueError("path escapes the run workspace")
    return path


def _text(args: dict[str, Any], *keys: str) -> str:
    return next((str(args[key]) for key in keys if args.get(key) is not None), "")


def _run(command: list[str] | str, workspace: Path, timeout_s: float, *, shell: bool) -> str:
    # ponytail: cwd jail + scrubbed env is not tenant-grade isolation; use a
    # container backend when untrusted callers are allowed to opt in.
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=workspace,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(workspace)},
        capture_output=True,
        text=True,
        timeout=timeout_s,
        shell=shell,
        check=False,
    )
    output = completed.stdout + completed.stderr
    if completed.returncode:
        raise RuntimeError(f"command exited {completed.returncode}: {output}")
    return output


def execute(
    name: str,
    arguments: str | dict[str, Any],
    workspace: Path,
    timeout_s: float = 30,
) -> str:
    """Execute one built-in tool and return its textual result."""
    if name not in SERVER_TOOL_NAMES:
        raise KeyError(name)
    args = json.loads(arguments or "{}") if isinstance(arguments, str) else arguments
    if not isinstance(args, dict):
        raise TypeError("tool arguments must be an object")

    if name == "list_files":
        root = _path(workspace, args.get("path", "."))
        pattern = _text(args, "pattern") or "*"
        return "\n".join(
            str(path.relative_to(workspace))
            for path in sorted(root.rglob("*"))
            if path.is_file() and fnmatch.fnmatch(path.name, pattern)
        )
    if name == "read_file":
        return _path(workspace, args.get("path")).read_text()
    if name == "search_files":
        query = _text(args, "query", "text")
        root = _path(workspace, args.get("path", "."))
        matches = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                for number, line in enumerate(path.read_text().splitlines(), 1):
                    if query in line:
                        matches.append(f"{path.relative_to(workspace)}:{number}:{line}")
            except UnicodeDecodeError:
                continue
        return "\n".join(matches)
    if name == "write_file":
        path = _path(workspace, args.get("path"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_text(args, "content", "text"))
        return "ok"
    if name == "edit_file":
        path = _path(workspace, args.get("path"))
        old = _text(args, "old_text", "oldText")
        new = _text(args, "new_text", "newText")
        content = path.read_text()
        count = content.count(old)
        if not old or count != 1:
            raise ValueError(f"old text must occur exactly once; found {count}")
        path.write_text(content.replace(old, new, 1))
        return "ok"
    if name in {"bash", "shell", "sh", "execute_command"}:
        command = _text(args, "command", "script")
        return _run(["/bin/sh", "-c", command], workspace, timeout_s, shell=False)
    if name in {"execute_code", "python", "exec", "run_code"}:
        language = _text(args, "language") or "python"
        if language not in {"python", "py"}:
            raise ValueError("server execution supports python only")
        return _run([sys.executable, "-c", _text(args, "code")], workspace, timeout_s, shell=False)
    raise AssertionError("unreachable")


def execute_calls(
    calls: list[dict[str, Any]], workspace: Path, timeout_s: float = 30
) -> list[dict[str, Any]]:
    def one(call: dict[str, Any]) -> dict[str, Any]:
        function = call.get("function") or {}
        try:
            content = execute(
                str(function.get("name", "")),
                function.get("arguments") or "{}",
                workspace,
                timeout_s,
            )
            is_error = False
        except Exception as error:  # noqa: BLE001 - tool failures return to the model
            content, is_error = str(error), True
        return {
            "tool_call_id": str(call.get("id", "")),
            "content": content[: serve_config.RUN_MAX_MSG_BYTES],
            "is_error": is_error,
        }

    if len(calls) < 2:
        return [one(call) for call in calls]
    with ThreadPoolExecutor(max_workers=len(calls)) as executor:
        return list(executor.map(one, calls))


def _fn(
    name: str, description: str, properties: dict[str, Any], required: list[str]
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_SCHEMAS = {
    "list_files": _fn(
        "list_files",
        "List files in the run workspace.",
        {"path": {"type": "string"}, "pattern": {"type": "string"}},
        [],
    ),
    "read_file": _fn(
        "read_file", "Read a file from the run workspace.", {"path": {"type": "string"}}, ["path"]
    ),
    "search_files": _fn(
        "search_files",
        "Search file contents in the run workspace.",
        {"query": {"type": "string"}, "path": {"type": "string"}},
        ["query"],
    ),
    "write_file": _fn(
        "write_file",
        "Write a file in the run workspace.",
        {"path": {"type": "string"}, "content": {"type": "string"}},
        ["path", "content"],
    ),
    "edit_file": _fn(
        "edit_file",
        "Replace one unique occurrence of old_text with new_text in a workspace file.",
        {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        ["path", "old_text", "new_text"],
    ),
    "bash": _fn(
        "bash",
        "Run a shell command in the run workspace.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    "run_code": _fn(
        "run_code",
        "Run Python code in the run workspace.",
        {"code": {"type": "string"}, "language": {"type": "string"}},
        ["code"],
    ),
}

# Canonical schema per bundle; alias names still execute if the client declares them.
_BUNDLE_SCHEMA_NAMES = {
    "files": ["list_files", "read_file", "search_files", "write_file", "edit_file"],
    "shell": ["bash"],
    "code": ["run_code"],
}


def server_tool_schemas(enabled: list[str]) -> list[dict[str, Any]]:
    names: list[str] = []
    for capability in enabled:
        names.extend(_BUNDLE_SCHEMA_NAMES.get(capability, [capability]))
    return [_SCHEMAS[name] for name in dict.fromkeys(names) if name in _SCHEMAS]
