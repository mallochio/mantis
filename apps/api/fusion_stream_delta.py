"""Filter provider-local Fusion tool fragments into public streaming calls."""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any


class DeltaEncoder:
    """Filter server-resolved delegation calls from client-visible deltas."""

    def __init__(self, hidden_tool_names: Collection[str] | None = None) -> None:
        """Track provider-local fragments and public calls across lead responses."""
        self.hidden_tool_names = frozenset(hidden_tool_names or ("sidekick", "read_subagent"))
        self.calls: dict[int, dict[str, Any]] = {}
        self.turn_calls: dict[int, dict[str, Any]] = {}
        self.emitted_call_ids: set[str] = set()

    def encode(
        self,
        run_id: str,
        delta: dict[str, Any],
        encode_tool_id: Callable[[str, str], str],
    ) -> list[dict[str, Any]]:
        """Encode public deltas while keeping provider response boundaries internal."""
        if delta.get("_mantis_turn_start"):
            self.turn_calls = {}
        output = []
        basic = {
            key: value
            for key, value in delta.items()
            if key in {"content", "reasoning", "reasoning_content"}
        }
        if basic:
            output.append(basic)
        for call in delta.get("tool_calls") or []:
            encoded = self._encode_call(run_id, call, encode_tool_id)
            if encoded is not None:
                output.append({"tool_calls": [encoded]})
        if delta.get("_mantis_turn_end"):
            output.extend(self._flush_calls(run_id, encode_tool_id))
        return output

    def _encode_call(
        self,
        run_id: str,
        fragment: dict[str, Any],
        encode_tool_id: Callable[[str, str], str],
    ) -> dict[str, Any] | None:
        """Accumulate an unsent name or emit continuation fields for a public call."""
        index = int(fragment.get("index", 0))
        state = self.turn_calls.setdefault(index, {"id": "", "name": "", "args": "", "sent": False})
        function = dict(fragment.get("function") or {})
        state["id"] = state["id"] or str(fragment.get("id") or "")
        state["name"] += str(function.get("name") or "")
        state["args"] += str(function.get("arguments") or "")
        if not state["sent"] and (
            not state["name"]
            or any(name.startswith(state["name"]) for name in self.hidden_tool_names)
        ):
            return None
        return self._client_call_delta(state, run_id, encode_tool_id)

    def _flush_calls(
        self, run_id: str, encode_tool_id: Callable[[str, str], str]
    ) -> list[dict[str, Any]]:
        """Emit completed client names that share a prefix with an internal tool."""
        output = []
        for state in self.turn_calls.values():
            if not state["sent"] and state["name"] and state["name"] not in self.hidden_tool_names:
                call = self._client_call_delta(state, run_id, encode_tool_id)
                if call is not None:
                    output.append({"tool_calls": [call]})
        return output

    def _client_call_delta(
        self,
        state: dict[str, Any],
        run_id: str,
        encode_tool_id: Callable[[str, str], str],
    ) -> dict[str, Any] | None:
        """Build a public fragment without reusing a prior response's tool index."""
        if not state["id"]:
            return None
        first = not state["sent"]
        if first:
            state["index"] = len(self.calls)
            self.calls[state["index"]] = state
        function = {
            key: state[field]
            for key, field in (("name", "name"), ("arguments", "args"))
            if state[field]
        }
        state["name"] = state["args"] = ""
        call: dict[str, Any] = {"index": state["index"], "function": function}
        if first:
            state["sent"] = True
            call_id = f"lead:{state['id']}"
            self.emitted_call_ids.add(call_id)
            call.update(id=encode_tool_id(run_id, call_id), type="function")
        return call
