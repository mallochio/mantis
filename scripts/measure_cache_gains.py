"""Measure prompt-cache gains for old vs new session handling.

Simulates multi-turn coding sessions without network calls. Compares:

- Base session isolation when two conversations share one harness header.
- Reasoning-trim savings from keeping only recent thinking blocks.

Run: ``uv run python scripts/measure_cache_gains.py``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

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


def _reusable_prefix_tokens(previous: list[dict], current: list[dict]) -> int:
    """Estimate reusable serialized prefix tokens without provider calls."""
    left = json.dumps(previous, sort_keys=True, separators=(",", ":"))
    right = json.dumps(current, sort_keys=True, separators=(",", ":"))
    common = 0
    for old, new in zip(left, right, strict=False):
        if old != new:
            break
        common += 1
    return common // 4


def simulate_gateway_prefixes() -> dict[str, int]:
    """Compare stable, independent, compacted, and accidentally mutable prefixes."""
    system: dict[str, Any] = {
        "role": "system",
        "content": "stable harness policy " + "s" * 1200,
    }
    first: list[dict[str, Any]] = [
        system,
        {"role": "user", "content": "inspect repository"},
    ]
    second: list[dict[str, Any]] = [
        *first,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contents"},
    ]
    third = [*second, {"role": "user", "content": "summarize"}]
    stable = _reusable_prefix_tokens(first, second) + _reusable_prefix_tokens(second, third)

    main_first: list[dict[str, Any]] = [system, {"role": "user", "content": "main task"}]
    main_next = [*main_first, {"role": "assistant", "content": "main update"}]
    side_system: dict[str, Any] = {
        "role": "system",
        "content": "stable sidekick policy " + "k" * 900,
    }
    side_first = [side_system, {"role": "user", "content": "side task"}]
    side_next = [*side_first, {"role": "assistant", "content": "side update"}]

    compacted = [system, second[-2], second[-1], {"role": "user", "content": "continue"}]
    mutated = [
        {"role": "system", "content": "dynamic run=2 " + "s" * 1200},
        *second[1:],
    ]
    return {
        "stable_session_reused_tokens": stable,
        "main_lane_reused_tokens": _reusable_prefix_tokens(main_first, main_next),
        "sidekick_lane_reused_tokens": _reusable_prefix_tokens(side_first, side_next),
        "compaction_reused_tokens": _reusable_prefix_tokens(second, compacted),
        "mutable_prefix_reused_tokens": _reusable_prefix_tokens(second, mutated),
    }


def main() -> None:
    sessions = simulate_base_session_isolation()
    trim = simulate_reasoning_trim()
    trim_saved = trim["before_tokens"] - trim["after_tokens"]
    trim_pct = 100.0 * trim_saved / max(1, trim["before_tokens"])
    gateway = simulate_gateway_prefixes()
    print("Base session isolation under one shared harness header")
    print(f"  conversation A : {sessions['conversation_a']}")
    print(f"  conversation B : {sessions['conversation_b']}")
    print("Reasoning trim keep-last-2 over 8 assistant turns")
    print(f"  before : {trim['before_tokens']}")
    print(f"  after  : {trim['after_tokens']}")
    print(f"  saved  : {trim_saved} ({trim_pct:.1f}%)")
    print("Gateway/Fusion reusable-prefix estimates (not measured cost savings)")
    for name, tokens in gateway.items():
        print(f"  {name}: {tokens}")


if __name__ == "__main__":
    main()
