#!/usr/bin/env python3
"""Headless Fusion session runner.

Starts a local Mantis API, delegates a brief to the Fusion main/sidekick loop,
executes pending tool calls in a scratch worktree, and follows up until the run
completes or errors.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

import httpx
import model_catalog

DEFAULT_BRIEF = (
    "Write a small Python project that prints a friendly greeting, "
    "run any tests you write, and lint the code. Report the final state."
)


def _wait_for_server(url: str, token: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(
                f"{url}/v1/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            continue
        time.sleep(0.5)
    raise RuntimeError("mantis API did not become ready in time")


def _start_server(url: str, workdir: Path) -> subprocess.Popen:
    port = url.rsplit(":", 1)[-1].rstrip("/")
    env = os.environ.copy()
    env["MANTIS_RUN_STORE"] = "file"
    env["MANTIS_RUN_DIR"] = str(workdir / "runs")
    env["MANTIS_API_KEY"] = env.get("MANTIS_API_KEY", "sk-mantis-headless")
    env["PYTHONPATH"] = "scripts:apps/api"
    catalog = model_catalog.load_mantis_catalog()
    if catalog is not None:
        env.update(model_catalog.render_mantis_environment(catalog))
    (workdir / "runs").mkdir(parents=True, exist_ok=True)
    cwd = Path(__file__).resolve().parent.parent
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "apps.api.api:app",
            "--host", "127.0.0.1", "--port", port, "--no-access-log",
        ],
        env=env,
        cwd=str(cwd),
    )


def _execute_tool_call(workdir: Path, tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function", {})
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")
    tool_call_id = str(tool_call.get("id", ""))
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return {
            "tool_call_id": tool_call_id,
            "content": f"invalid arguments: {exc}",
            "is_error": True,
        }

    if name == "bash":
        command = args.get("command", "")
        if not command:
            return {
                "tool_call_id": tool_call_id,
                "content": "empty command",
                "is_error": True,
            }
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=60.0,
            )
        except subprocess.TimeoutExpired:
            return {
                "tool_call_id": tool_call_id,
                "content": "tool timed out",
                "is_error": True,
            }
        except (OSError, ValueError) as exc:
            return {
                "tool_call_id": tool_call_id,
                "content": f"exec failed: {exc}",
                "is_error": True,
            }
        output = (result.stdout or "") + (result.stderr or "")
        if not output:
            output = "<no output>"
        return {
            "tool_call_id": tool_call_id,
            "content": output,
            "is_error": result.returncode != 0,
        }

    return {
        "tool_call_id": tool_call_id,
        "content": f"unknown tool: {name}",
        "is_error": True,
    }


def _delegate(
    client: httpx.Client, url: str, brief: str, tools: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    response = client.post(
        f"{url}/v1/fusion/delegate",
        json={"brief": brief, "tools": tools},
    )
    response.raise_for_status()
    body = response.json()
    return body["run_id"], body


def _follow_up(
    client: httpx.Client,
    url: str,
    run_id: str,
    request_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    response = client.post(
        f"{url}/v1/fusion/follow_up/{run_id}",
        json={"request_id": request_id, "tool_results": results},
    )
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


def _log_tool_call(tc: dict[str, Any], res: dict[str, Any]) -> None:
    function = tc.get("function", {})
    name = function.get("name", "")
    args = function.get("arguments", "")[:80]
    error = res.get("is_error", False)
    print(f"[tool] {name}: {args!r} -> error={error}")


def run_session(url: str, token: str, brief: str, max_iterations: int = 10) -> dict[str, Any]:
    client = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=600.0)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command in the project worktree",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]

    run_id, event = _delegate(client, url, brief, tools)
    print(f"[fusion] run {run_id} started: {event['status']}")

    with tempfile.TemporaryDirectory(prefix="fusion-pi-") as worktree:
        iterations = 0
        while event.get("status") == "awaiting_tools" and iterations < max_iterations:
            pending = event["pending_tool_calls"]
            tool_results = [_execute_tool_call(Path(worktree), tc) for tc in pending]
            for tc, res in zip(pending, tool_results, strict=True):
                _log_tool_call(tc, res)
            request_id = uuid.uuid4().hex
            event = _follow_up(client, url, run_id, request_id, tool_results)
            iterations += 1
            print(f"[fusion] status: {event['status']}")

    client.close()
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a headless Fusion session.")
    parser.add_argument("--url", default="http://127.0.0.1:5501")
    parser.add_argument(
        "--token",
        default=os.environ.get("MANTIS_API_KEY", "sk-mantis-headless"),
    )
    parser.add_argument("--brief", default=DEFAULT_BRIEF)
    parser.add_argument(
        "--managed",
        action="store_true",
        help="start and stop the API server automatically",
    )
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the final event JSON to this file",
    )
    args = parser.parse_args()

    server = None
    workdir = None
    try:
        if args.managed:
            workdir = Path(tempfile.mkdtemp(prefix="fusion-runs-"))
            server = _start_server(args.url, workdir)
            _wait_for_server(args.url, args.token)

        event = run_session(args.url, args.token, args.brief, args.max_iterations)
        if args.output:
            with args.output.open("w") as handle:
                json.dump(event, handle, indent=2)
            print(f"[fusion] wrote final event to {args.output}")
        print(json.dumps(event, indent=2))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
        if workdir is not None:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
