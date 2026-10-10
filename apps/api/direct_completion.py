"""Internal worker calls with shared Switchyard dispatch."""

from __future__ import annotations

from typing import Any, cast

import base_proxy
import providers
import serve_config


def _routed_response(
    messages: list[dict[str, Any]],
    temperature: float,
    timeout: float | None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Dispatch through Switchyard with run identity and selected-model usage.

    Output limits and reasoning effort belong to the selected catalog target,
    not to the virtual route's caller.
    """
    context = serve_config._history_context
    run = getattr(context, "active_run", None)
    session = getattr(context, "fusion_route_session_id", None) or getattr(run, "run_id", None)
    body: dict[str, Any] = {
        "model": "mantis/base",
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        body["tools"] = tools
    result = base_proxy.dispatch_routed(
        body,
        session,
        timeout_s=timeout,
    )
    if run is not None:
        run.add_usage(result.body.get("usage") or {}, model=result.selected_model or "mantis/base")
    return cast(dict[str, Any], result.body)


def complete(
    spec: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    """Return assistant text, reusing routed dispatch only for mantis/base."""
    data = (
        _routed_response(messages, temperature, timeout)
        if spec == "mantis/base"
        else providers._provider_response(spec, messages, max_tokens, temperature)
    )
    return str(data["choices"][0]["message"].get("content") or "")


def model_response(
    spec: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Return a tool-capable worker response through the appropriate dispatch."""
    return (
        _routed_response(messages, 0.7, None, tools)
        if spec == "mantis/base"
        else providers._provider_response(spec, messages, 4096, 0.7, tools)
    )
