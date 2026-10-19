"""Cache-preserving tier effort for the mantis/base stage router.

Switchyard picks the efficient or capable target per turn. When both
targets share one upstream model, the tiers differ only in the
top-level ``reasoning_effort`` they inject through ``extra_body``, and
changing that key between requests restarts the upstream prompt cache
on GPT-6 and Claude 5.5 model families. This module pins the
request-level effort to the efficient tier's value and, when the
Switchyard decision endpoint selects a tier with a stronger effort,
appends the OpenRouter chat-completions ``configuration_update`` system
message directly before the pending turn. OpenRouter translates that
message into the cache-preserving per-message mechanism of each model
family, so the reusable prefix survives tier changes.

Set ``MANTIS_TIER_EFFORT=0`` to disable the mechanism and keep the
legacy per-target top-level effort. Any decision-endpoint failure
leaves the body untouched, so Switchyard per-target ``extra_body``
keeps governing effort exactly as before.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import httpx
from model_catalog_schema import BaseRoute
from switchyard_config import load_switchyard_route

SWITCHYARD_SESSION_HEADER = "x-switchyard-session-id"
DECISION_TIMEOUT_S = 5.0
_BASE_TARGETS = ("efficient", "capable")


def _tier_effort_enabled() -> bool:
    """Return False when the operator disabled per-message tier effort."""
    return os.environ.get("MANTIS_TIER_EFFORT", "1").lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def _load_base_route() -> BaseRoute | None:
    """Load and cache the parsed [base] route from the active catalog."""
    try:
        return load_switchyard_route(None)
    except (OSError, ValueError):
        return None


def attach_sticky_session(body: dict[str, Any], session: str | None) -> dict[str, Any]:
    """Pin OpenRouter provider stickiness; the derived key dies on compaction."""
    if session:
        body.setdefault("session_id", str(session)[:256])
    return body


def decision_url() -> str:
    """Return the Switchyard decision endpoint for the active router."""
    return os.environ.get("MANTIS_ROUTER_URL", "http://127.0.0.1:5500/v1") + "/decision"


def decision_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Wrap one sanitized chat body for the decision endpoint."""
    return {"input_format": "openai_chat", "request": body}


def fetch_decision_sync(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any] | None:
    """Post one decision request; return parsed JSON or None on failure."""
    try:
        response = client.post(url, headers=headers, json=payload, timeout=DECISION_TIMEOUT_S)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


async def fetch_decision_async(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float | None,
) -> dict[str, Any] | None:
    """Post one decision request inside the run budget; None on failure."""
    budget = DECISION_TIMEOUT_S if timeout_s is None else min(DECISION_TIMEOUT_S, timeout_s)
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=budget)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def effort_baseline(route: BaseRoute | None) -> str | None:
    """Return the constant request-level effort for a route, when active."""
    if not _tier_effort_enabled():
        return None
    if route is None or route.efficient.wire_format != "openai_chat":
        return None
    return route.efficient.reasoning_effort


def resolve_effort(
    body: dict[str, Any],
    headers: dict[str, str],
    client_factory: Any,
) -> dict[str, Any]:
    """Attach stickiness and resolve the tier effort for one routed turn."""
    body = attach_sticky_session(body, headers.get(SWITCHYARD_SESSION_HEADER))
    route = _load_base_route()
    if effort_baseline(route) is None:
        return body
    with client_factory() as client:
        decision = fetch_decision_sync(client, decision_url(), decision_payload(body), headers)
    return apply_router_effort(body, route, decision)


async def resolve_effort_async(
    body: dict[str, Any],
    headers: dict[str, str],
    client_factory: Any,
    timeout_s: float | None,
) -> dict[str, Any]:
    """Async twin of resolve_effort inside the run's wall-clock budget."""
    body = attach_sticky_session(body, headers.get(SWITCHYARD_SESSION_HEADER))
    route = _load_base_route()
    if effort_baseline(route) is None:
        return body
    async with client_factory() as client:
        decision = await fetch_decision_async(
            client, decision_url(), decision_payload(body), headers, timeout_s
        )
    return apply_router_effort(body, route, decision)


def apply_router_effort(
    body: dict[str, Any],
    route: BaseRoute | None,
    decision: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pin the baseline effort and inject the selected tier's update.

    The baseline is the efficient tier's catalog effort and is sent on
    every request, so the per-tier ``extra_body`` value in Switchyard can
    no longer change it (``merge_extra_body`` fills only absent keys).
    The update message carries the capable tier's effort for the pending
    turn only.
    """
    if not isinstance(body, dict) or not isinstance(decision, dict):
        return body
    baseline = effort_baseline(route)
    if baseline is None:
        return body
    target, effort = _selected_tier(decision)
    if target is None:
        return body
    if effort is None:
        effort = route.capable.reasoning_effort if target == "capable" else baseline
    amended = dict(body)
    amended["reasoning_effort"] = baseline
    if effort is None or effort == baseline:
        return amended
    return _inject_effort_update(amended, effort)


def _selected_tier(decision: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return the selected target name and its declared reasoning effort."""
    selected = decision.get("selected")
    if not isinstance(selected, Mapping):
        return None, None
    effort = None
    extra = selected.get("extra_body")
    if isinstance(extra, Mapping):
        declared = extra.get("reasoning_effort")
        if isinstance(declared, str):
            effort = declared
    target = selected.get("target")
    if not isinstance(target, str) or target not in _BASE_TARGETS:
        return None, None
    return target, effort


def _inject_effort_update(body: dict[str, Any], effort: str) -> dict[str, Any]:
    """Insert the per-message effort update ahead of the pending turn."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    if _carries_update(messages):
        return body
    update = {
        "role": "system",
        "content": "",
        "configuration_update": {"reasoning": {"effort": effort}},
    }
    amended = list(messages)
    amended.insert(_insertion_index(messages), update)
    out = dict(body)
    out["messages"] = amended
    return out


def _insertion_index(messages: list[Any]) -> int:
    """Position of the update relative to the pending turn.

    A turn ending in tool results gets the update after the results so it
    lands between the assistant turn and the pending reply; any other
    turn gets it directly before the final message, matching the
    placement OpenRouter expects.
    """
    last = messages[-1]
    if isinstance(last, dict) and last.get("role") == "tool":
        return len(messages)
    return len(messages) - 1


def _carries_update(messages: list[Any]) -> bool:
    """Return True when history already holds an effort-update message."""
    return any(
        isinstance(message, dict) and "configuration_update" in message for message in messages
    )
