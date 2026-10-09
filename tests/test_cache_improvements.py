"""Cache-improvement tests: per-conversation isolation and prefix stability."""

from __future__ import annotations

import base_proxy


class _BaseBody:
    def __init__(self, messages=None, tools=None, metadata=None, user=None):
        self.messages = messages
        self.tools = tools
        self.metadata = metadata
        self.user = user
        self.stream = False


def test_base_session_id_isolates_shared_harness_header():
    messages_a = [{"role": "user", "content": "fix failing test"}]
    messages_b = [{"role": "user", "content": "write migration docs"}]
    key_a = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"}, _BaseBody(messages=messages_a)
    )
    key_b = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"}, _BaseBody(messages=messages_b)
    )
    assert key_a != key_b
    assert key_a.startswith("opencode:auto-")
    assert key_b.startswith("opencode:auto-")


def test_base_session_id_stable_across_tool_rounds():
    turn_one = [{"role": "user", "content": "fix failing test"}]
    turn_two = [
        {"role": "user", "content": "fix failing test"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "run pytest again"},
    ]
    headers = {"x-mantis-session-id": "prime-agent"}
    assert base_proxy.session_id(headers, _BaseBody(messages=turn_one)) == (
        base_proxy.session_id(headers, _BaseBody(messages=turn_two))
    )


def test_router_body_preserves_old_reasoning():
    """History round-trips verbatim so the Modal prefix cache survives.

    Rewriting old assistant turns every request moves the divergence point
    and forces a full context prefill, which times out past ~60s on long
    sessions and surfaces as a silently stopped conversation.
    """
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "a", "reasoning_content": "old-one"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "b", "reasoning_content": "old-two"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "c", "reasoning_content": "keep-one"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "d", "reasoning_content": "keep-two"},
        {"role": "user", "content": "final"},
    ]

    class Request:
        def model_dump(self, *, exclude_none, exclude):
            return {"messages": [dict(m) for m in messages]}

    body = base_proxy.router_body(Request())
    kept = [m for m in body["messages"] if m.get("role") == "assistant"]
    assert [m.get("reasoning_content") for m in kept] == [
        "old-one",
        "old-two",
        "keep-one",
        "keep-two",
    ]
