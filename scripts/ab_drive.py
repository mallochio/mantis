"""Drive routed cases while preserving client and server tool evidence."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

import httpx
from routed_ab_tools import case_observation, execute_tool_call, post_round, request_body


def _apply_tool_results(
    message: dict[str, Any], workspace: Path, messages: list[dict[str, Any]]
) -> None:
    """Execute client calls and retain their canonical request and result messages."""
    messages.append(
        {
            "role": "assistant",
            "content": str(message.get("content") or ""),
            "tool_calls": message["tool_calls"],
        }
    )
    for call in message["tool_calls"]:
        result = execute_tool_call(call, workspace)
        messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": result})


def _response_run_id(payloads: list[dict[str, Any]], headers: dict[str, str]) -> str:
    """Return the run ID actually reported by the response transport or event."""
    if headers.get("X-Mantis-Run-Id"):
        return headers["X-Mantis-Run-Id"]
    for payload in reversed(payloads):
        event = payload.get("mantis_event") or {}
        if isinstance(event, dict) and event.get("run_id"):
            return str(event["run_id"])
    return ""


def _server_run_details(
    client: httpx.Client,
    base_url: str,
    plan: dict[str, Any],
    payloads: list[dict[str, Any]],
    headers: dict[str, str],
) -> dict[str, Any] | None:
    """Fetch run evidence from the existing status endpoint when identified."""
    if plan.get("model") != "mantis/fusion":
        return None
    run_id = _response_run_id(payloads, headers)
    if not run_id:
        return None
    try:
        response = client.get(f"{base_url}/v1/fusion/runs/{run_id}", headers=headers)
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except (httpx.HTTPError, ValueError) as error:
        return {"run_id": run_id, "lookup_error": str(error)}


def _observed(
    client: httpx.Client,
    base_url: str,
    plan: dict[str, Any],
    payloads: list[dict[str, Any]],
    message: dict[str, Any],
    messages: list[dict[str, Any]],
    headers: dict[str, str],
    rounds: int,
    started: float,
) -> dict[str, Any]:
    """Retain every response, conversation message, and available server detail."""
    observation = case_observation(plan, payloads, message, rounds, started)
    observation["messages"] = messages
    observation["response_payloads"] = payloads
    observation["server_run_details"] = _server_run_details(
        client, base_url, plan, payloads, headers
    )
    observation["run_id"] = _response_run_id(payloads, headers) or None
    return observation


def _initial_request(
    plan: dict[str, Any], prompt: str, plan_auth: dict[str, str] | None
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Build the mutable message list, request body, and auth headers."""
    messages = [{"role": "user", "content": prompt}]
    return messages, request_body(plan, messages), dict(plan_auth or {})


def _run_rounds(
    client: httpx.Client,
    base_url: str,
    plan: dict[str, Any],
    workspace: Path,
    max_rounds: int,
    body: dict[str, Any],
    messages: list[dict[str, Any]],
    headers: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any], int, dict[str, str]]:
    """Run tool turns until the model returns a final message or the budget ends."""
    payloads: list[dict[str, Any]] = []
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        payload, headers = post_round(client, base_url, body, headers)
        payloads.append(payload)
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        if choice.get("finish_reason") != "tool_calls":
            return payloads, message, rounds - 1, headers
        _apply_tool_results(message, workspace, messages)
    return payloads, {"content": "EXCEEDED_TOOL_BUDGET"}, max_rounds, headers


def drive_case(
    client: httpx.Client,
    base_url: str,
    plan: dict[str, Any],
    prompt: str,
    workspace: Path,
    max_rounds: int,
    plan_auth: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one case and retain its full request and response transcript."""
    messages, body, headers = _initial_request(plan, prompt, plan_auth)
    started = time.monotonic()
    payloads, message, rounds, headers = _run_rounds(
        client, base_url, plan, workspace, max_rounds, body, messages, headers
    )
    return _observed(client, base_url, plan, payloads, message, messages, headers, rounds, started)
