"""Measure prompt-cache gains for old vs new session handling.

Simulates multi-turn coding sessions without network calls. Compares:

- Azure-router input tokens sent with a shared static key (old) vs
  per-conversation isolated keys (new).
- Base session isolation when two conversations share one harness header.
- Reasoning-trim savings from keeping only recent thinking blocks.

Run: ``uv run python scripts/measure_cache_gains.py``
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

import api  # noqa: E402
import base_proxy  # noqa: E402
import providers  # noqa: E402


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 4 chars per token."""
    return max(1, len(text) // 4)


def _messages_tokens(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        total += _estimate_tokens(str(content))
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(key)
            if isinstance(value, str):
                total += _estimate_tokens(value)
    return total


def _canonical_for(conv: list) -> list:
    request = api.ChatRequest(model="mantis/azure-router", messages=list(conv))
    return [m for m in (api._azure_canonical_message(m) for m in request.messages) if m]


def _charge(store: dict, key: str, canonical: list, turn: int) -> int:
    prev = store.get(key)
    if prev and canonical[: len(prev[1])] == prev[1]:
        sent = _messages_tokens([{"content": str(m)} for m in canonical[len(prev[1]) :]])
    else:
        sent = _messages_tokens([{"content": str(m)} for m in canonical])
    store[key] = (f"resp-{turn}", list(canonical))
    return sent


def _append_turn(conv: list, turn: int, extra: str) -> None:
    conv.append(api.Message(role="assistant", content=f"work update {turn} " + "x" * 2000))
    conv.append(api.Message(role="user", content=f"follow-up{extra} " + "y" * 800))


def simulate_azure_interleaved(turns: int = 8) -> dict[str, int]:
    """Two interleaved conversations sharing one harness header."""
    api._azure_sessions.clear()
    headers = {"x-mantis-session-id": "opencode"}
    conv_a = [api.Message(role="user", content="fix failing test in auth module")]
    conv_b = [api.Message(role="user", content="write migration docs for v2")]
    old_sent = 0
    new_sent = 0
    old_store: dict[str, tuple[str, list]] = {}
    new_store: dict[str, tuple[str, list]] = {}

    for turn in range(turns):
        for conv, extra in ((conv_a, f" pytest run {turn}"), (conv_b, f" docs part {turn}")):
            _append_turn(conv, turn, extra)
            canonical = _canonical_for(conv)
            old_sent += _charge(old_store, "explicit:opencode", canonical, turn)
            request = api.ChatRequest(model="mantis/azure-router", messages=list(conv))
            new_key = api._azure_session_key(request, headers)
            assert new_key is not None
            new_sent += _charge(new_store, new_key, canonical, turn)
    api._azure_sessions.clear()
    return {"old_tokens_sent": old_sent, "new_tokens_sent": new_sent}


def simulate_base_session_isolation() -> dict[str, str]:
    """Show per-conversation session keys when the harness header is shared."""

    def _req(messages: list[dict]) -> Any:
        return type(
            "R",
            (SimpleNamespace,),
            {"messages": messages, "tools": None},
        )()

    key_a = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"},
        _req([{"role": "user", "content": "prove theorem"}]),
    )
    key_b = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"},
        _req([{"role": "user", "content": "what time is it"}]),
    )
    return {"conversation_a": str(key_a), "conversation_b": str(key_b)}


def simulate_reasoning_trim(assistant_turns: int = 8) -> dict[str, int]:
    """Compare full reasoning history vs keep-last-2 trimming."""
    messages: list[dict] = [{"role": "system", "content": "sys " + "s" * 800}]
    for i in range(assistant_turns):
        messages.append({"role": "user", "content": f"step {i} " + "u" * 400})
        messages.append(
            {"role": "assistant", "content": f"answer {i}", "reasoning": "r" * 4000}
        )
    before = _messages_tokens(messages)
    trimmed = providers._clear_thinking(messages, keep=2)
    after = _messages_tokens(trimmed)
    return {"before_tokens": before, "after_tokens": after}


def main() -> None:
    azure = simulate_azure_interleaved()
    azure_saved = azure["old_tokens_sent"] - azure["new_tokens_sent"]
    azure_pct = 100.0 * azure_saved / max(1, azure["old_tokens_sent"])
    sessions = simulate_base_session_isolation()
    trim = simulate_reasoning_trim()
    trim_saved = trim["before_tokens"] - trim["after_tokens"]
    trim_pct = 100.0 * trim_saved / max(1, trim["before_tokens"])
    print("Azure-router interleaved 2x8 turns (estimated tokens sent)")
    print(f"  old shared key : {azure['old_tokens_sent']}")
    print(f"  new isolated   : {azure['new_tokens_sent']}")
    print(f"  saved          : {azure_saved} ({azure_pct:.1f}%)")
    print("Base session isolation under one shared harness header")
    print(f"  conversation A : {sessions['conversation_a']}")
    print(f"  conversation B : {sessions['conversation_b']}")
    print("Reasoning trim keep-last-2 over 8 assistant turns")
    print(f"  before : {trim['before_tokens']}")
    print(f"  after  : {trim['after_tokens']}")
    print(f"  saved  : {trim_saved} ({trim_pct:.1f}%)")


if __name__ == "__main__":
    main()
