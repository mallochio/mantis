"""Cache-improvement tests: per-conversation isolation and prefix stability."""

from __future__ import annotations

import api
import base_proxy


class _BaseBody:
    def __init__(self, messages=None, tools=None, metadata=None, user=None):
        self.messages = messages
        self.tools = tools
        self.metadata = metadata
        self.user = user
        self.stream = False


def _azure_request(first: str, second: str | None = None):
    msgs = [api.Message(role="user", content=first)]
    if second is not None:
        msgs.append(api.Message(role="assistant", content="ok"))
        msgs.append(api.Message(role="user", content=second))
    return api.ChatRequest(model="mantis/azure-router", messages=msgs)


def test_azure_session_key_isolates_conversations_sharing_harness_header():
    headers = {"x-mantis-session-id": "opencode"}
    first = _azure_request("fix failing test")
    second = _azure_request("write migration docs")
    key_first = api._azure_session_key(first, headers)
    key_second = api._azure_session_key(second, headers)
    assert key_first is not None and key_second is not None
    assert key_first != key_second
    assert key_first.startswith("explicit:opencode:")
    assert key_second.startswith("explicit:opencode:")


def test_azure_session_key_stable_across_turns():
    headers = {"x-mantis-session-id": "opencode"}
    turn_one = _azure_request("fix failing test")
    turn_two = _azure_request("fix failing test", "run pytest again")
    assert api._azure_session_key(turn_one, headers) == api._azure_session_key(
        turn_two, headers
    )


def test_azure_tail_only_send_with_isolated_key():
    api._azure_sessions.clear()
    try:
        headers = {"x-mantis-session-id": "prime-agent"}
        first = _azure_request("remember falcon")
        full_first = [
            m
            for m in (api._azure_canonical_message(m) for m in first.messages)
            if m
        ]
        full_first.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        )
        api._azure_sessions[api._azure_session_key(first, headers)] = (
            "resp-1",
            full_first,
        )
        second = api.ChatRequest(
            model="mantis/azure-router",
            messages=[
                api.Message(role="user", content="remember falcon"),
                api.Message(role="assistant", content="ok"),
                api.Message(role="user", content="which word?"),
            ],
        )
        tail, previous = api._azure_input_for_request(second, headers)
        assert previous == "resp-1"
        assert len(tail) == 1
    finally:
        api._azure_sessions.clear()


def test_azure_max_tokens_cap_matches_harness():
    request = _azure_request("hi")
    assert api._azure_router_max_tokens(request) == 131072


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


def test_router_body_trims_old_reasoning():
    messages = [
        {"role": "user", "content": "start"},
        {"role": "assistant", "content": "a", "reasoning": "old-one"},
        {"role": "user", "content": "next"},
        {"role": "assistant", "content": "b", "reasoning": "old-two"},
        {"role": "user", "content": "more"},
        {"role": "assistant", "content": "c", "reasoning": "keep-one"},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "d", "reasoning": "keep-two"},
        {"role": "user", "content": "final"},
    ]

    class Request:
        def model_dump(self, *, exclude_none, exclude):
            return {"messages": [dict(m) for m in messages]}

    body = base_proxy.router_body(Request())
    kept = [m for m in body["messages"] if m.get("role") == "assistant"]
    assert "reasoning" not in kept[0]
    assert "reasoning" not in kept[1]
    assert kept[2]["reasoning"] == "keep-one"
    assert kept[3]["reasoning"] == "keep-two"
