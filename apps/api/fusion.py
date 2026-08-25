"""mantis-fusion: catalog-driven lead/sidekick resumable orchestration."""

from __future__ import annotations

import json
import os
import re
import threading
import tomllib
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import model_catalog
import providers
import serve_config
import utils
from runs import (
    RUN_STORE,
    NativeRun,
    _file_put,
    _file_run_lock,
    _redis_put,
    _redis_run_lock,
    get_run,
)

__all__ = [
    "FusionCoordinator",
    "FusionRun",
    "advance_fusion_run",
    "FUSION_STATUS",
]

FUSION_STATUS = frozenset(
    {"main_planning", "sidekick_pending", "awaiting_tools", "main_review", "completed", "error"}
)

MAIN_PREAMBLE = (
    "You are the lead engineer on a software task. You may use tools to inspect the "
    "codebase, then plan work and review a sidekick's output. When given a brief or "
    "conversation, respond with exactly two sections: 'PLAN:' containing the high-level "
    "plan, and 'BRIEF:' containing a self-contained brief for the sidekick. "
    "When reviewing a sidekick report, reply exactly 'ACCEPT' if the report is "
    "satisfactory. Otherwise reply 'FOLLOW_UP:' followed by concise feedback."
)

SIDEKICK_PREAMBLE = (
    "You are a fast, cheap coding sidekick. Implement, test, and lint according to "
    "the brief you are given. You may use the provided tools. When finished, "
    "return a concise final report."
)

REVIEW_PROMPT = (
    "The sidekick produced the following report. Review it. If it is satisfactory, "
    "reply exactly 'ACCEPT'. Otherwise reply 'FOLLOW_UP:' followed by concise feedback "
    "so the sidekick can revise.\n\nReport:\n"
)

PLAN_REMINDER_PROMPT = (
    "Your previous response did not contain the required sections. "
    "Reply with exactly two sections and nothing else: "
    "'PLAN:' containing the high-level plan, and "
    "'BRIEF:' containing a self-contained brief for the sidekick. "
    "Do not include commentary, markdown, or tool calls."
)

PLAN_TOOL_BUDGET_PROMPT = (
    "You have already used the allowed planning tool budget. "
    "Stop calling tools and reply with exactly two sections: "
    "'PLAN:' and 'BRIEF:'. No other text or tool calls."
)

COMPACTION_INSTRUCTION = (
    "Condense the conversation above into a checkpoint another model can resume from. "
    "Output exactly one <compacted-summary> block with terse bullets for: goal, "
    "files touched, decisions, and remaining work. Do not call tools."
)
COMPACTION_RETAIN_RATIO = 0.16
COMPACTION_MAX_OUTPUT_TOKENS = 1024

PLAN_UNKNOWN_TOOL_PROMPT = (
    "You tried to call tools that are not available for planning: {names}. "
    "Stop calling tools and reply with exactly two sections: "
    "'PLAN:' and 'BRIEF:'. No other text or tool calls."
)


class FusionConfig:
    """Runtime fusion settings loaded from the shared catalog."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or model_catalog.catalog_path()[0]
        self._raw: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if self._raw is not None:
            return self._raw
        with self._lock:
            if self._raw is not None:
                return self._raw
            if not self.path.exists():
                self._raw = {}
            else:
                with self.path.open("rb") as handle:
                    root = tomllib.load(handle)
                self._raw = root.get("fusion", {})
            return self._raw

    def main_slot(self) -> str:
        return self._load().get("main") or "gpt-5_6-sol"

    def sidekick_slot(self) -> str:
        return self._load().get("sidekick") or "gpt-5_6-luna"

    def max_follow_ups(self) -> int:
        raw = self._load().get("max_follow_ups") or "3"
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 3

    def context_window(self) -> int:
        raw = self._load().get("context_window") or os.environ.get(
            "MANTIS_CONTEXT_LENGTH", "262144"
        )
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 262144

    def max_output_tokens(self) -> int:
        raw = self._load().get("max_output_tokens") or "4096"
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 4096


_FUSION_CONFIG = FusionConfig()


def _fusion_config() -> FusionConfig:
    """Return the global FusionConfig, reloaded on first call in a new process."""
    return _FUSION_CONFIG


def _return_reasoning_by_default() -> bool:
    return os.environ.get("MANTIS_FUSION_RETURN_REASONING", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def _extract_reasoning_trace(
    messages: list[dict[str, Any]],
    max_chars: int = 8192,
) -> str:
    """Return safe textual provider summaries, excluding replay-only metadata."""
    pieces: list[str] = []
    length = 0

    def add(piece: str) -> bool:
        nonlocal length
        piece = piece.strip()
        if not piece:
            return True
        if length + len(piece) + 2 > max_chars:
            remaining = max_chars - length
            if remaining > 0:
                pieces.append(piece[:remaining] + "\n...")
            return False
        pieces.append(piece)
        length += len(piece) + 2
        return True

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for key in ("reasoning", "reasoning_content", "thinking"):
            value = msg.get(key)
            if isinstance(value, str) and value.strip() and not add(value):
                break
        details = msg.get("reasoning_details")
        if isinstance(details, list):
            for item in details:
                if not isinstance(item, dict):
                    continue
                summary = item.get("summary")
                if isinstance(summary, list):
                    for part in summary:
                        if (
                            isinstance(part, dict)
                            and isinstance(part.get("text"), str)
                            and not add(part["text"])
                        ):
                            break
                if isinstance(item.get("text"), str) and not add(item["text"]):
                    break
        anthropic_content = msg.get("_anthropic_content")
        if isinstance(anthropic_content, list):
            for block in anthropic_content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "thinking"
                    and isinstance(block.get("thinking"), str)
                    and not add(block["thinking"])
                ):
                    break
    return "\n\n".join(pieces)


def _model_label(slot: str) -> str:
    """Use a readable catalog label without exposing provider internals."""
    return slot.replace("_", ".", 1)


def _orchestration_trace(run: FusionRun, include_reasoning: bool = False) -> str:
    """Build Fusion's public, chronological orchestration trace."""
    main_slot = getattr(run, "main_slot", "gpt-5_6-sol")
    sidekick_slot = getattr(run, "sidekick_slot", "gpt-5_6-luna")
    lines = [f"Fusion · Planning · {_model_label(main_slot)}"]
    if run.plan:
        lines.extend(["", "Plan:", run.plan])
    if run.sidekick_brief:
        lines.extend([
            "",
            f"Delegated to sidekick · {_model_label(sidekick_slot)}",
            "",
            "Brief:",
            run.sidekick_brief,
        ])
    tool_count = sum(
        len(msg.get("tool_calls") or [])
        for msg in run.sidekick_messages
        if msg.get("role") == "assistant"
    )
    if tool_count:
        lines.extend(["", f"Fusion · Sidekick requested {tool_count} tools"])
        lines.extend(
            "Fusion · Tool result received"
            for msg in run.sidekick_messages
            if msg.get("role") == "tool"
        )
    lines.extend(["", "Fusion · Reviewing sidekick report"])
    if run.status == "completed":
        lines.append("Fusion · Accepted")
    elif run.status == "error":
        lines.append("Fusion · Orchestration failed")
    if include_reasoning:
        provider = _extract_reasoning_trace(run.main_messages + run.sidekick_messages)
        if provider:
            lines.extend(["", "Provider summary:", provider])
    return "\n".join(lines)


class FusionCoordinator:
    """Thin wrapper that calls catalog-bound worker slots."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or _fusion_config()
        self.main_slot = self.config.main_slot()
        self.sidekick_slot = self.config.sidekick_slot()
        self.max_follow_ups = self.config.max_follow_ups()
        self.context_window = self.config.context_window()
        self.max_output_tokens = self.config.max_output_tokens()

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        """Approximate token count for a single chat message.

        This is a fast, dependency-free estimate (roughly 4 characters per
        token) used for context-window trimming before a provider call.
        """
        text = ""
        content = message.get("content")
        if isinstance(content, str):
            text += content
        elif content is not None:
            text += json.dumps(content)
        for tc in message.get("tool_calls", []):
            fn = tc.get("function", {})
            text += str(fn.get("name", ""))
            text += str(fn.get("arguments", ""))
        for key in ("tool_call_id", "name"):
            value = message.get(key)
            if value is not None:
                text += str(value)
        return max(1, len(text) // 4)

    def _token_sum(self, messages: list[dict[str, Any]]) -> int:
        return sum(self._estimate_message_tokens(message) for message in messages)

    def _message_groups(
        self,
        messages: list[dict[str, Any]],
    ) -> list[list[dict[str, Any]]]:
        """Group assistant tool_calls with all matching tool results.

        Each group is an atomic conversational unit that must be kept or
        dropped together so providers never receive an orphaned tool result.
        """
        groups: list[list[dict[str, Any]]] = []
        i = 0
        while i < len(messages):
            message = messages[i]
            if message.get("role") == "assistant" and message.get("tool_calls"):
                group = [message]
                tool_ids = {str(tc.get("id", "")) for tc in message["tool_calls"]}
                i += 1
                while (
                    i < len(messages)
                    and messages[i].get("role") == "tool"
                    and str(messages[i].get("tool_call_id", "")) in tool_ids
                ):
                    group.append(messages[i])
                    i += 1
                groups.append(group)
            else:
                groups.append([message])
                i += 1
        return groups

    def _prune_tool_messages(
        self, messages: list[dict[str, Any]], *, max_chars: int, head_chars: int, tail_chars: int
    ) -> list[dict[str, Any]]:
        pruned: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
                pruned.append(
                    {
                        **msg,
                        "content": utils.prune_tool_result(
                            msg["content"],
                            max_chars=max_chars,
                            head_chars=head_chars,
                            tail_chars=tail_chars,
                        ),
                    }
                )
            else:
                pruned.append(msg)
        return pruned

    def _prune_for_budget(
        self, messages: list[dict[str, Any]], max_input_tokens: int
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages
        pruned = self._prune_tool_messages(
            messages, max_chars=8192, head_chars=4096, tail_chars=1024
        )
        if self._token_sum(pruned) <= max_input_tokens:
            return pruned
        return self._prune_tool_messages(
            pruned, max_chars=2048, head_chars=1024, tail_chars=256
        )

    def _drop_old_groups(
        self, messages: list[dict[str, Any]], max_input_tokens: int
    ) -> list[dict[str, Any]]:
        if not messages:
            return messages
        estimates = [self._estimate_message_tokens(message) for message in messages]
        trimmed = [messages[0]]
        budget = max_input_tokens - estimates[0]
        tail: list[dict[str, Any]] = []
        for group in reversed(self._message_groups(messages[1:])):
            group_tokens = sum(self._estimate_message_tokens(item) for item in group)
            if group_tokens > budget:
                break
            tail.extend(reversed(group))
            budget -= group_tokens
        trimmed.extend(reversed(tail))
        return trimmed

    def _flatten_groups(self, groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for group in groups:
            flattened.extend(group)
        return flattened

    def _compact_replay(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_input_tokens: int,
    ) -> list[dict[str, Any]] | None:
        """Summarize the middle of history on the same slot, replaying the prefix.

        The summarizer request is system + history + a trailing instruction so
        the provider cache stays warm up to that instruction. Stored history
        keeps the original system message and a verbatim tail.
        """
        if len(messages) < 3:
            return None
        system = messages[0]
        groups = self._message_groups(messages[1:])
        if len(groups) < 2:
            return None
        instruction = {"role": "user", "content": COMPACTION_INSTRUCTION}
        system_tokens = self._estimate_message_tokens(system)
        instruction_tokens = self._estimate_message_tokens(instruction)
        min_middle_tokens = self._token_sum(groups[0])
        max_tail = min(
            max_input_tokens - system_tokens - instruction_tokens - min_middle_tokens,
            max_input_tokens - system_tokens - 64,
        )
        if max_tail <= 0:
            return None
        body_tokens = sum(self._token_sum(group) for group in groups)
        tail_target = max(1, min(int(body_tokens * COMPACTION_RETAIN_RATIO), max_tail))
        tail_groups: list[list[dict[str, Any]]] = []
        tail_tokens = 0
        while len(groups) > 1:
            group = groups[-1]
            group_tokens = self._token_sum(group)
            if tail_groups and tail_tokens + group_tokens > tail_target:
                break
            if not tail_groups and group_tokens > tail_target:
                return None
            groups.pop()
            tail_groups.insert(0, group)
            tail_tokens += group_tokens
            if tail_tokens >= tail_target:
                break
        if not groups or not tail_groups:
            return None

        def replay(middle: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
            return [
                system,
                *self._flatten_groups(middle),
                *self._flatten_groups(tail_groups),
                instruction,
            ]

        while groups and self._token_sum(replay(groups)) > max_input_tokens:
            groups.pop(0)
        if not groups:
            return None

        run = getattr(serve_config._history_context, "active_run", None)
        previous_choice = getattr(run, "active_tool_choice", None) if run is not None else None
        if run is not None:
            run.active_tool_choice = "none"
        try:
            data = providers._provider_response(
                slot,
                replay(groups),
                min(COMPACTION_MAX_OUTPUT_TOKENS, self._output_tokens_for(slot)),
                0.7,
                tools,
            )
        except (RuntimeError, TypeError, KeyError, IndexError):
            return None
        finally:
            if run is not None:
                run.active_tool_choice = previous_choice

        message = (data.get("choices") or [{}])[0].get("message") or {}
        if message.get("tool_calls"):
            return None
        text = str(message.get("content") or "").strip()
        if not text:
            return None
        summary = text
        match = re.search(
            r"<compacted-summary>\s*([\s\S]*?)\s*</compacted-summary>", text, re.IGNORECASE
        )
        if match:
            summary = match.group(1).strip() or text
        return [
            system,
            {
                "role": "user",
                "content": f"<compacted-summary>\n{summary}\n</compacted-summary>",
            },
            *self._flatten_groups(tail_groups),
        ]

    def _fit_messages(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_input_tokens: int,
    ) -> list[dict[str, Any]]:
        pruned = self._prune_for_budget(messages, max_input_tokens)
        if self._token_sum(pruned) <= max_input_tokens:
            return pruned
        compacted = self._compact_replay(slot, pruned, tools, max_input_tokens)
        if compacted is not None and self._token_sum(compacted) <= max_input_tokens:
            return compacted
        return self._drop_old_groups(compacted or pruned, max_input_tokens)

    def _trim_messages(
        self,
        messages: list[dict[str, Any]],
        max_input_tokens: int,
    ) -> list[dict[str, Any]]:
        """Fit history without a model call: prune tool bodies, then drop groups."""
        pruned = self._prune_for_budget(messages, max_input_tokens)
        if self._token_sum(pruned) <= max_input_tokens:
            return pruned
        return self._drop_old_groups(pruned, max_input_tokens)

    def _output_tokens_for(self, slot: str) -> int:
        """Return the model-specific output token cap from the catalog."""
        try:
            resolved = providers._resolve_model_spec(slot)
        except (RuntimeError, ValueError):
            return 4096
        return resolved.max_tokens or 4096

    def _call_worker(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call a worker slot and return (assistant_message, usage).

        The returned assistant message preserves provider-specific metadata such
        as ``reasoning``, ``reasoning_details``, ``_anthropic_content`` and
        ``_anthropic_tool_ids`` so that later turns can replay required state.
        """
        model_cap = self._output_tokens_for(slot)
        output_tokens = min(self.max_output_tokens, model_cap)
        input_budget = max(0, self.context_window - output_tokens)
        fitted = self._fit_messages(slot, messages, tools, input_budget)
        data = providers._provider_response(slot, fitted, output_tokens, 0.7, tools)
        msg = dict(data["choices"][0]["message"])
        msg.setdefault("role", "assistant")
        msg["content"] = str(msg.get("content") or "")
        tcs = msg.get("tool_calls") or []
        calls: list[dict[str, Any]] = []
        for tc in tcs:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments") or ""
            if not isinstance(raw_args, str):
                raw_args = json.dumps(raw_args)
            call = {
                "id": str(tc.get("id", f"call_{len(calls)}")),
                "type": "function",
                "function": {"name": str(fn.get("name", "")), "arguments": raw_args},
            }
            calls.append(call)
        if calls:
            msg["tool_calls"] = calls
        else:
            msg.pop("tool_calls", None)
        usage = data.get("usage") or {}
        return msg, usage


class FusionRun(NativeRun):
    """A resumable lead/sidekick run whose roles are bound by the catalog."""

    def __init__(
        self,
        run_id: str,
        brief: str = "",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(run_id)
        self.kind = "fusion"
        self.brief = brief
        coordinator = FusionCoordinator()
        # Keep the slots with the run so status/trace responses remain stable
        # if the catalog is changed while a run is in progress.
        self.main_slot = coordinator.main_slot
        self.sidekick_slot = coordinator.sidekick_slot
        self.tools = utils._convert_tools(tools)
        if messages:
            self.main_messages: list[dict[str, Any]] = [
                {"role": "system", "content": MAIN_PREAMBLE},
                *messages,
            ]
        else:
            self.main_messages = [
                {"role": "system", "content": MAIN_PREAMBLE},
                {"role": "user", "content": brief},
            ]
        self.sidekick_messages: list[dict[str, Any]] = [
            {"role": "system", "content": SIDEKICK_PREAMBLE},
        ]
        self.active_role: str = "main"
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.planning_tool_rounds = 0
        self.repeat_guard = utils.RepeatToolGuard()
        self.report: str | None = None
        self.plan: str = ""
        self.sidekick_brief: str = ""
        self.error: str | None = None
        self.follow_up_count = 0
        self.status = "main_planning"

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Ensure new fields are present for forward compatibility.
        for key, default in {
            "plan": "",
            "sidekick_brief": "",
            "follow_up_count": 0,
            "planning_tool_rounds": 0,
            "active_role": "main",
            "main_slot": "gpt-5_6-sol",
            "sidekick_slot": "gpt-5_6-luna",
        }.items():
            if not hasattr(self, key):
                setattr(self, key, default)
        if not hasattr(self, "repeat_guard"):
            self.repeat_guard = utils.RepeatToolGuard()

    def _append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        target = self.main_messages if self.active_role == "main" else self.sidekick_messages
        for result in tool_results:
            raw_content = str(result.get("content", ""))
            target.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": raw_content[:utils.RUN_MAX_MSG_BYTES],
                    "is_error": bool(result.get("is_error", False)),
                }
            )
            # Advisory repeat-tool check
            call_info = next(
                (
                    tc
                    for tc in self.pending_tool_calls
                    if tc.get("id") == result.get("tool_call_id")
                ),
                None,
            )
            if call_info and isinstance(call_info.get("function"), dict):
                fn = call_info["function"]
                reminder = self.repeat_guard.observe(fn.get("name", ""), fn.get("arguments", "{}"))
                if reminder:
                    target.append(
                        {"role": "user", "content": utils.system_reminder(reminder)}
                    )

    def _call_main(
        self,
        coordinator: FusionCoordinator,
        prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        if prompt is not None:
            self.main_messages.append({"role": "user", "content": prompt})
        message, usage = coordinator._call_worker(
            coordinator.main_slot, self.main_messages, tools
        )
        self.main_messages.append(message)
        text = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if getattr(serve_config._history_context, "active_run", None) is not self:
            self.add_usage(usage, model=coordinator.main_slot)
        self.record_activity(
            "main_turn",
            role="main",
            model=coordinator.main_slot,
            status="completed",
            summary=text[:200] if text else "tool-calls",
        )
        return text, calls, usage

    def _call_sidekick(
        self,
        coordinator: FusionCoordinator,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        message, usage = coordinator._call_worker(
            coordinator.sidekick_slot, self.sidekick_messages, self.tools
        )
        self.sidekick_messages.append(message)
        text = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if getattr(serve_config._history_context, "active_run", None) is not self:
            self.add_usage(usage, model=coordinator.sidekick_slot)
        self.record_activity(
            "sidekick_turn",
            role="sidekick",
            model=coordinator.sidekick_slot,
            status="completed",
            summary=text[:200] if text else "tool-calls",
        )
        return text, calls, usage

    def _parse_main_plan(self, text: str) -> tuple[str, str]:
        plan_match = re.search(r"PLAN:(.*?)(?:BRIEF:|$)", text, re.DOTALL)
        brief_match = re.search(r"BRIEF:(.*)", text, re.DOTALL)
        if not plan_match or not brief_match:
            raise ValueError("main did not produce both PLAN and BRIEF sections")
        plan = plan_match.group(1).strip()
        sidekick_brief = brief_match.group(1).strip()
        if not plan or not sidekick_brief:
            raise ValueError("main produced empty PLAN or BRIEF")
        return plan, sidekick_brief

    def _parse_main_review(self, text: str) -> tuple[bool, str]:
        stripped = text.strip()
        if stripped.upper().startswith("ACCEPT") or re.search(
            r"^\s*ACCEPT\b", stripped, re.IGNORECASE
        ):
            return True, ""
        follow_match = re.search(r"\bFOLLOW_UP:\s*(.*)", stripped, re.DOTALL | re.IGNORECASE)
        if follow_match:
            feedback = follow_match.group(1).strip()
            if feedback:
                return False, feedback
        if stripped.upper().startswith("REJECT"):
            return False, stripped
        raise ValueError(
            "main review did not output a valid decision "
            f"('ACCEPT' or 'FOLLOW_UP: <feedback>'), got: {stripped[:120]!r}"
        )

    def _validate_tool_results(self, tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        expected_ids = {str(tc.get("id")) for tc in self.pending_tool_calls if tc.get("id")}
        validated = utils._validate_tool_results(tool_results, expected_ids)
        id_order = {
            tc_id: idx
            for idx, tc_id in enumerate(
                str(tc.get("id")) for tc in self.pending_tool_calls if tc.get("id")
            )
        }
        validated.sort(key=lambda r: id_order.get(str(r.get("tool_call_id")), 999))
        return validated

    def _error_event(self, exc: Exception, request_id: str | None) -> dict[str, Any]:
        self.status = "error"
        self.error = str(exc)
        self.record_activity(
            "error",
            role="fusion",
            model="",
            status="error",
            detail=self.error,
        )
        event = {
            "run_id": self.run_id,
            "status": "error",
            "report": self.error,
            "pending_tool_calls": None,
            "usage": self.usage,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def _ok_event(self, request_id: str | None) -> dict[str, Any]:
        event = {
            "run_id": self.run_id,
            "status": self.status,
            "report": self.report if self.status == "completed" else None,
            "pending_tool_calls": (
                self.pending_tool_calls if self.status == "awaiting_tools" else None
            ),
            "usage": self.usage,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def advance(
        self,
        tool_results: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        message: str | None = None,
        coordinator: FusionCoordinator | None = None,
    ) -> dict[str, Any]:
        """Run the Fusion state machine until suspension, completion, or error."""
        coordinator = coordinator or FusionCoordinator()
        tool_results = tool_results or []
        try:
            return self._advance(tool_results, request_id, message, coordinator)
        except Exception as exc:  # noqa: BLE001 - catch-all guard for worker/state errors
            return self._error_event(exc, request_id)

    def _summarize_sidekick_tool_history(self) -> str:
        lines: list[str] = []
        total_calls = 0
        error_count = 0
        for msg in self.sidekick_messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for call in msg["tool_calls"]:
                    total_calls += 1
                    fn = call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "")
                    lines.append(f"- Tool call: {name}({args[:120]})")
            elif msg.get("role") == "tool":
                content = str(msg.get("content", ""))
                is_err = bool(msg.get("is_error"))
                if is_err:
                    error_count += 1
                status = "ERROR" if is_err else "OK"
                lines.append(f"  Result [{status}]: {content[:200]}")
        if not lines:
            return "None (sidekick used no tools)"
        summary_header = f"Total tool invocations: {total_calls} (Errors: {error_count})\n"
        return summary_header + "\n".join(lines)

    def _advance(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        if self.cancelled:
            raise RuntimeError("run is cancelled")

        # Apply client tool results only when awaiting tools.
        if self.status == "awaiting_tools":
            validated = self._validate_tool_results(tool_results)
            self._append_tool_results(validated)
            self.pending_tool_calls = []
            if self.active_role == "main":
                self.status = "main_planning" if not self.plan else "main_review"
            else:
                self.status = "sidekick_pending"
        elif tool_results:
            raise ValueError("tool_results are only valid when status is awaiting_tools")

        # Main planning on a fresh run or resumed planning.
        if self.status == "main_planning":
            self.active_role = "main"
            available_tools = self.tools if (self.planning_tool_rounds < 2 and self.tools) else None
            main_text, main_calls, _ = self._call_main(
                coordinator, tools=available_tools
            )

            # The main model may only call tools it was given, and only during the
            # first two planning rounds. After that (or if it calls an unknown tool)
            # it must produce the PLAN/BRIEF text.
            if main_calls:
                allowed_names = {
                    t.get("function", {}).get("name")
                    for t in self.tools
                    if t.get("function")
                }
                unknown = [
                    c
                    for c in main_calls
                    if c.get("function", {}).get("name") not in allowed_names
                ]
                if unknown or self.planning_tool_rounds >= 2:
                    if unknown:
                        unknown_names = {
                            str(c.get("function", {}).get("name", "")) for c in unknown
                        }
                        names = ", ".join(sorted(unknown_names))
                        reminder = PLAN_UNKNOWN_TOOL_PROMPT.format(names=names)
                    else:
                        reminder = PLAN_TOOL_BUDGET_PROMPT
                    main_text, main_calls, _ = self._call_main(
                        coordinator, prompt=reminder, tools=None
                    )
                    if main_calls:
                        call_names = {
                            str(c.get("function", {}).get("name", "")) for c in main_calls
                        }
                        names = ", ".join(sorted(call_names))
                        raise ValueError(
                            f"main kept calling tools when a plan was required: {names}"
                        )
                else:
                    self.planning_tool_rounds += 1
                    self.pending_tool_calls = main_calls
                    self.status = "awaiting_tools"
                    return self._ok_event(request_id)

            # Parse the plan, with one retry on malformed output.
            try:
                self.plan, self.sidekick_brief = self._parse_main_plan(main_text)
            except ValueError:
                main_text, main_calls, _ = self._call_main(
                    coordinator, prompt=PLAN_REMINDER_PROMPT, tools=None
                )
                if main_calls:
                    raise ValueError("main called tools instead of producing a plan") from None
                self.plan, self.sidekick_brief = self._parse_main_plan(main_text)
            self.sidekick_messages.append(
                {"role": "user", "content": self.sidekick_brief}
            )
            self.status = "sidekick_pending"

        # Bounded sidekick-main loop.
        max_iterations = max(1, coordinator.max_follow_ups + 1)
        for _ in range(max_iterations * 4):  # generous step ceiling
            if self.status == "sidekick_pending":
                self.active_role = "sidekick"
                sidekick_text, sidekick_calls, _ = self._call_sidekick(coordinator)
                if sidekick_calls:
                    self.pending_tool_calls = sidekick_calls
                    self.status = "awaiting_tools"
                    break
                # Sidekick produced a report; ask the main to review.
                self.status = "main_review"
                self.active_role = "main"
                tool_summary = self._summarize_sidekick_tool_history()
                review_prompt = (
                    f"{REVIEW_PROMPT}"
                    f"Tool Activity by Sidekick:\n{tool_summary}\n\n"
                    f"Report:\n{sidekick_text}"
                )
                review_text, review_calls, _ = self._call_main(
                    coordinator, review_prompt, tools=self.tools or None
                )
                if review_calls:
                    self.pending_tool_calls = review_calls
                    self.status = "awaiting_tools"
                    break
                accepted, feedback = self._parse_main_review(review_text)
                if accepted:
                    self.report = sidekick_text
                    self.status = "completed"
                    break
                # Main requested a sidekick follow-up.
                self.follow_up_count += 1
                if self.follow_up_count > coordinator.max_follow_ups:
                    self.report = sidekick_text
                    self.status = "completed"
                    break
                self.sidekick_messages.append({"role": "user", "content": feedback})
                self.record_activity(
                    "follow_up",
                    role="main",
                    model=coordinator.main_slot,
                    status="completed",
                    summary=feedback[:200],
                )
                self.status = "sidekick_pending"
                continue
            if self.status == "main_review":
                self.active_role = "main"
                review_text, review_calls, _ = self._call_main(
                    coordinator, tools=self.tools or None
                )
                if review_calls:
                    self.pending_tool_calls = review_calls
                    self.status = "awaiting_tools"
                    break
                accepted, feedback = self._parse_main_review(review_text)
                if accepted:
                    last_sidekick_text = self.sidekick_messages[-1].get("content", "")
                    self.report = self.report or last_sidekick_text
                    self.status = "completed"
                    break
                self.follow_up_count += 1
                if self.follow_up_count > coordinator.max_follow_ups:
                    last_sidekick_text = self.sidekick_messages[-1].get("content", "")
                    self.report = self.report or last_sidekick_text
                    self.status = "completed"
                    break
                self.sidekick_messages.append({"role": "user", "content": feedback})
                self.status = "sidekick_pending"
                continue
            if self.status in ("completed", "awaiting_tools", "error"):
                break

        return self._ok_event(request_id)

    def advance_idempotent(
        self,
        tool_results: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        message: str | None = None,
        coordinator: FusionCoordinator | None = None,
    ) -> dict[str, Any]:
        if request_id is None:
            return self.advance(tool_results, request_id, message, coordinator)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cast(dict[str, Any], cached)
            event = cast(
                dict[str, Any],
                self.advance(tool_results, request_id, message, coordinator),
            )
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event


def _put_run(run: NativeRun) -> None:
    if RUN_STORE == "redis":
        _redis_put(run)
    elif RUN_STORE == "file":
        _file_put(run)
    else:
        serve_config._runs[run.run_id] = run


def _run_lock(run_id: str) -> Any:
    if RUN_STORE == "redis":
        return _redis_run_lock(run_id)
    if RUN_STORE == "file":
        return _file_run_lock(run_id)
    return nullcontext()


def advance_fusion_run(
    run_id: str,
    request_id: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Load, advance, and save a FusionRun with the same locking as advance_run."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        run.in_flight += 1
        _put_run(run)
        try:
            serve_config._history_context.active_run = run
            coordinator = FusionCoordinator()
            event = run.advance_idempotent(tool_results, request_id, message, coordinator)
            if event.get("status") in ("completed", "error"):
                # Persist learning record only for terminal states.
                pass
            return event
        finally:
            serve_config._history_context.active_run = None
            run.in_flight -= 1
            run.touch()
            _put_run(run)


def create_fusion_run(
    brief: str,
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
) -> FusionRun:
    run_id = uuid.uuid4().hex
    run = FusionRun(run_id, brief, tools, messages=messages)
    _put_run(run)
    return run


def fusion_run_status(run_id: str) -> dict[str, Any]:
    """Return the current status of a FusionRun without advancing it."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        return {
            "run_id": run.run_id,
            "status": run.status,
            "report": run.report if run.status == "completed" else None,
            "pending_tool_calls": (
                run.pending_tool_calls if run.status == "awaiting_tools" else None
            ),
            "usage": run.usage,
            "activity": run._activity,
        }
