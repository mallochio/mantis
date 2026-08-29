"""mantis-fusion: catalog-driven lead/sidekick resumable orchestration."""

from __future__ import annotations

import json
import os
import re
import threading
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, cast

import model_catalog
import providers
import serve_config
import utils
from fusion_budget import FusionBudgetGuard
from fusion_router import FusionRouter
from fusion_types import (
    FusionPlan,
    FusionRoutingConfig,
    FusionRunBudget,
    FusionSidekickAssignment,
    FusionToolOptions,
    FusionWorkerProfile,
)

from runs import (
    RUN_STORE,
    NativeRun,
    _file_put,
    _file_run_lock,
    _redis_put,
    _redis_run_lock,
    _write_learning_record,
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
    "You are the lead engineer on a software task. Plan the work, delegate execution to "
    "a sidekick, and review the sidekick's output. "
    "When you can fully resolve the user request without code changes or tool use "
    "(clarifications, trivia, nothing-to-do, or answers already in context), reply with "
    "exactly one section: 'ANSWER:' followed by the user-facing reply and nothing else. "
    "Otherwise respond with exactly two sections: 'PLAN:' containing the high-level plan, "
    "and 'BRIEF:' containing a self-contained brief for the sidekick. The brief must state "
    "the goal, the files to touch, hard constraints and edge cases, and the exact "
    "verification commands to run; require the sidekick to report the changes made, "
    "the commands it ran, and their results. When reviewing a sidekick report, reply "
    "exactly 'ACCEPT' if the report is satisfactory. Otherwise reply 'FOLLOW_UP:' "
    "followed by concise, specific feedback. After an escalation from the sidekick, "
    "reply with 'ANSWER:' or a new 'PLAN:' and 'BRIEF:'."
)

SIDEKICK_PREAMBLE = (
    "You are a fast, cheap coding sidekick. Implement, test, and lint according to "
    "the brief you are given. Treat 'goal:' as the stable task anchor and 'latest_user:' "
    "as the current narrowing ask; obey 'brief:' for what to do. Do not redesign the "
    "work. If the brief is impossible, needs frontier judgment, or you are over budget, "
    "reply with exactly 'ESCALATE_TO_MAIN:' followed by a concise reason and stop. "
    "You may use the provided tools. When finished successfully, return a concise final "
    "report listing the changes made, the verification commands you ran with their real "
    "output, and any remaining issues."
)

FUSION_BRIEF_OPEN = "<fusion-brief>"
FUSION_BRIEF_CLOSE = "</fusion-brief>"
FUSION_FOLLOW_UP_OPEN = "<fusion-follow-up>"
FUSION_FOLLOW_UP_CLOSE = "</fusion-follow-up>"
DELEGATION_MODES = frozenset({"available", "forced"})


def _format_fusion_brief(goal: str, latest_user: str, plan: str, brief: str) -> str:
    return (
        f"{FUSION_BRIEF_OPEN}\n"
        f"goal: {goal}\n"
        f"latest_user: {latest_user}\n"
        f"plan: {plan}\n"
        f"brief: {brief}\n"
        f"{FUSION_BRIEF_CLOSE}"
    )


def _format_fusion_follow_up(feedback: str) -> str:
    return f"{FUSION_FOLLOW_UP_OPEN}\n{feedback}\n{FUSION_FOLLOW_UP_CLOSE}"


def _user_message_texts(messages: list[dict[str, Any]] | None, brief: str) -> tuple[str, str]:
    """Return (first_user, last_user) texts for stable goal and latest_user."""
    users: list[str] = []
    for message in messages or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            users.append(content.strip())
        elif content is not None:
            text = str(content).strip()
            if text:
                users.append(text)
    if users:
        return users[0], users[-1]
    brief_text = brief.strip() or brief
    return brief_text, brief_text


REVIEW_PROMPT = (
    "The sidekick produced the following report. Review it. If it is satisfactory, "
    "reply exactly 'ACCEPT'. Otherwise reply 'FOLLOW_UP:' followed by concise feedback "
    "so the sidekick can revise.\n\nReport:\n"
)

ESCALATE_PROMPT = (
    "The sidekick escalated back to you. Reply with exactly 'ANSWER:' and a "
    "user-facing reply, or with 'PLAN:' and 'BRIEF:' to redelegate. "
    "Do not call tools.\n\nEscalation:\n"
)

PLAN_REMINDER_PROMPT = (
    "Your previous response did not contain the required sections. "
    "Reply with exactly two sections and nothing else: "
    "'PLAN:' containing the high-level plan, and "
    "'BRIEF:' containing a self-contained brief for the sidekick. "
    "Do not include commentary, markdown, or tool calls."
)

AVAILABLE_REMINDER_PROMPT = (
    "Your previous response was not valid. Reply with exactly 'ANSWER:' followed by "
    "a user-facing reply, or with exactly two sections 'PLAN:' and 'BRIEF:'. "
    "Do not include commentary, markdown, or tool calls."
)

FORCED_ANSWER_REMINDER_PROMPT = (
    "Direct ANSWER is not allowed for this forced-delegation run. "
    "Reply with exactly two sections and nothing else: "
    "'PLAN:' containing the high-level plan, and "
    "'BRIEF:' containing a self-contained brief for the sidekick."
)

PLAN_TOOL_BUDGET_PROMPT = (
    "You have already used the allowed planning tool budget. "
    "Stop calling tools and reply with exactly two sections: "
    "'PLAN:' and 'BRIEF:'. No other text or tool calls."
)

SIDEKICK_TOOL_BUDGET_PROMPT = (
    "You have used the allowed sidekick tool budget. "
    "Stop calling tools. Either finish with a concise final report, or reply with "
    "exactly 'ESCALATE_TO_MAIN:' followed by a concise reason."
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

PLAN_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "fusion_plan",
        "schema": {
            "type": "object",
            "required": ["complexity", "main_task", "sidekick_assignments"],
            "properties": {
                "complexity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Estimated difficulty of the user request.",
                },
                "main_task": {
                    "type": "string",
                    "description": "Concise summary of what the main engineer will verify.",
                },
                "sidekick_assignments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["task"],
                        "properties": {
                            "task": {"type": "string"},
                            "profile": {"type": "string"},
                        },
                    },
                },
                "verification_commands": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "strict": True,
    },
}


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

    def main_tools(self) -> str:
        raw = str(self._load().get("main_tools") or "none").strip().lower()
        return raw if raw in {"none", "plan", "review", "plan+review"} else "none"

    def max_follow_ups(self) -> int:
        raw = self._load().get("max_follow_ups") or "3"
        try:
            return int(raw)
        except (TypeError, ValueError):
            return 3

    def sidekick_max_tool_rounds(self) -> int:
        raw = self._load().get("sidekick_max_tool_rounds") or "16"
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 16
        return max(1, value)

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

    def main_routing(self) -> FusionRoutingConfig:
        return FusionRoutingConfig.model_validate(
            {k: v for k, v in (self._load() or {}).items() if k in FusionRoutingConfig.model_fields}
        )

    def main_router(self) -> FusionRouter:
        return FusionRouter.from_config(self.main_routing(), "main")

    def sidekick_router(self) -> FusionRouter:
        return FusionRouter.from_config(self.main_routing(), "sidekick")

    def worker_profiles(self) -> list[FusionWorkerProfile]:
        raw = self._load().get("worker_profiles") or []
        if not isinstance(raw, list):
            return []
        return [FusionWorkerProfile.model_validate(item) for item in raw]

    def default_budget(self) -> FusionRunBudget:
        return FusionRunBudget.model_validate(self._load().get("budget") or {})

    def tool_options(self) -> FusionToolOptions:
        return FusionToolOptions.model_validate(self._load().get("tool_options") or {})


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
        lines.extend(
            [
                "",
                f"Delegated to sidekick · {_model_label(sidekick_slot)}",
                "",
                "Brief:",
                run.sidekick_brief,
            ]
        )
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
        self.main_router = self.config.main_router()
        self.sidekick_router = self.config.sidekick_router()
        self.main_slot = self.select("main", 0, 0)
        self.sidekick_slot = self.select("sidekick", 0, 0)
        self.max_follow_ups = self.config.max_follow_ups()
        self.sidekick_max_tool_rounds = self.config.sidekick_max_tool_rounds()
        self.context_window = self.config.context_window()
        self.max_output_tokens = self.config.max_output_tokens()
        self.worker_profiles = {p.name: p for p in self.config.worker_profiles()}
        self.default_tool_options = self.config.tool_options()
        self.default_budget = self.config.default_budget()

    def select(self, role: str, turn_index: int, escalation_count: int) -> str:
        router = self.main_router if role == "main" else self.sidekick_router
        return router.select(turn_index, escalation_count)

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
        return self._prune_tool_messages(pruned, max_chars=2048, head_chars=1024, tail_chars=256)

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


class SidekickLane:
    """One resumable sidekick worker lane for the structured Fusion path."""

    def __init__(
        self,
        lane_id: str,
        run: FusionRun,
        coordinator: FusionCoordinator,
        assignment: FusionSidekickAssignment,
        profile: FusionWorkerProfile,
    ) -> None:
        self.lane_id = lane_id
        self.run = run
        self.coordinator = coordinator
        self.assignment = assignment
        self.profile = profile
        self.tool_rounds = 0
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.report: str | None = None
        self.error: str | None = None
        self.complete = False
        self.messages = self._build_messages()

    def _build_messages(self) -> list[dict[str, Any]]:
        system = SIDEKICK_PREAMBLE
        if self.profile.instructions:
            system += "\n\n" + self.profile.instructions
        plan = self.run.structured_plan.main_task if self.run.structured_plan else ""
        brief = _format_fusion_brief(
            self.run.goal,
            self.run.latest_user,
            plan,
            self.assignment.task,
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": brief},
        ]

    def _tools_for_lane(self) -> list[dict[str, Any]] | None:
        tools = list(self.run.tools or [])
        if self.profile.tools:
            names = set(self.profile.tools)
            tools = [t for t in tools if str(t.get("function", {}).get("name", "")) in names]
        return tools or None

    def _select_slot(self) -> str:
        if self.profile.model:
            router = FusionRouter(
                FusionRoutingConfig(sidekick=self.profile.model),
                "sidekick",
                self.coordinator.sidekick_router.base_route,
            )
            return router.select(self.run.follow_up_count, self.run.follow_up_count)
        return self.run.sidekick_router.select(self.run.follow_up_count, self.run.follow_up_count)

    def step(self, tool_results: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.error:
            return self._status()
        if self.complete:
            return self._status(status="completed")
        if tool_results:
            self._append_tool_results(tool_results)
            self.pending_tool_calls = []
        if self.tool_rounds >= self.coordinator.sidekick_max_tool_rounds:
            self.report = "ESCALATE_TO_MAIN: sidekick exceeded tool budget"
            self.complete = True
            return self._status(status="completed", report=self.report)

        slot = self._select_slot()
        tools = self._tools_for_lane()
        message, usage = self.coordinator._call_worker(slot, self.messages, tools)
        if self.run.budget is not None:
            self.run.budget.consume_tokens(usage)
            self.run.budget.consume_turn()
        self.run.add_usage(usage, model=slot)
        self.run.record_activity(
            "sidekick_lane_turn",
            lane=self.lane_id,
            role="sidekick",
            model=slot,
            status="completed",
        )
        providers._emit_progress(
            {
                "type": "task.completed",
                "role": "sidekick",
                "lane": self.lane_id,
                "model": slot,
                "task": self.assignment.task,
            }
        )

        text = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        self.messages.append(message)
        if calls:
            self.tool_rounds += 1
            for call in calls:
                call["id"] = f"{self.lane_id}:{call['id']}"
            self.pending_tool_calls = calls
            return self._status(status="awaiting_tools", pending=calls)

        escalate = self._parse_escalate(text)
        if escalate is not None:
            self.report = escalate
            self.complete = True
            return self._status(status="completed", report=escalate)

        self.report = text
        self.complete = True
        return self._status(status="completed", report=text)

    def _parse_escalate(self, text: str) -> str | None:
        match = re.match(
            r"^\s*ESCALATE_TO_MAIN:\s*(.*)\s*$",
            text.strip(),
            re.DOTALL | re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).strip() or "sidekick requested escalation"

    def _append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = {"asst": {"tool_calls": [dict(call) for call in self.pending_tool_calls]}}
        self.run.record_tool_results(pending, tool_results)
        for result in tool_results:
            raw_content = str(result.get("content", ""))
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": raw_content[: utils.RUN_MAX_MSG_BYTES],
                    "is_error": bool(result.get("is_error", False)),
                }
            )

    def _status(
        self,
        status: str | None = None,
        pending: list[dict[str, Any]] | None = None,
        report: str | None = None,
    ) -> dict[str, Any]:
        if status is None:
            status = "error" if self.error else "completed" if self.complete else "in_progress"
        return {
            "lane_id": self.lane_id,
            "status": status,
            "pending_tool_calls": pending or [],
            "report": report or self.report,
            "error": self.error,
        }

    def apply_follow_up(self, feedback: str) -> None:
        """Reset a completed lane so it can revise with feedback."""
        if not self.complete or self.error:
            return
        self.complete = False
        self.report = None
        self.messages.append({"role": "user", "content": _format_fusion_follow_up(feedback)})


class FusionRun(NativeRun):
    """A resumable lead/sidekick run whose roles are bound by the catalog."""

    def __init__(
        self,
        run_id: str,
        brief: str = "",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        delegation_mode: str = "forced",
        worker_profiles: list[FusionWorkerProfile] | None = None,
        budget: FusionRunBudget | None = None,
        tool_options: FusionToolOptions | None = None,
    ) -> None:
        super().__init__(run_id)
        self.kind = "fusion"
        self.brief = brief
        mode = delegation_mode if delegation_mode in DELEGATION_MODES else "forced"
        self.delegation_mode = mode
        self.worker_profiles = {p.name: p for p in (worker_profiles or [])}
        self.budget = FusionBudgetGuard(**budget.model_dump()) if budget is not None else None
        self.tool_options = tool_options or FusionToolOptions()
        goal, latest_user = _user_message_texts(messages, brief)
        self.goal = goal
        self.latest_user = latest_user
        # Feeds the learning record's task/task_hash (redacted by runs.py).
        self.query = goal
        coordinator = FusionCoordinator()
        # Keep the slots with the run so status/trace responses remain stable
        # if the catalog is changed while a run is in progress.
        self.main_slot = coordinator.main_slot
        self.sidekick_slot = coordinator.sidekick_slot
        self.main_router = coordinator.main_router
        self.sidekick_router = coordinator.sidekick_router
        self.slot_models = [self.main_slot, self.sidekick_slot]
        policy = coordinator.config.main_tools()
        self.main_tools_policy = frozenset(
            {"plan", "review"} if policy == "plan+review" else {policy} - {"none"}
        )
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
        self.sidekick_tool_rounds = 0
        self.repeat_guard = utils.RepeatToolGuard()
        self.report: str | None = None
        self.plan: str = ""
        self.sidekick_brief: str = ""
        self.error: str | None = None
        self.follow_up_count = 0
        self.follow_up_capped = False
        self.completed_via: str = ""
        self._resume_allows_answer = False
        self.turns: list[dict[str, Any]] = []
        self.structured = (
            bool(self.worker_profiles)
            or bool(self.tool_options.enabled)
            or self.budget is not None
            or bool(coordinator.worker_profiles)
        )
        self.structured_plan: FusionPlan | None = None
        self.sidekick_lanes: list[SidekickLane] = []
        self.sidekick_reports: list[str] = []
        self.status = "main_planning"

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Ensure new fields are present for forward compatibility.
        for key, default in {
            "plan": "",
            "sidekick_brief": "",
            "follow_up_count": 0,
            "follow_up_capped": False,
            # Old pickles predate the config gate and allowed lead tools in
            # both phases; keep that behavior for restored runs.
            "main_tools_policy": frozenset({"plan", "review"}),
            "query": "",
            "slot_models": [],
            "turns": [],
            "planning_tool_rounds": 0,
            "sidekick_tool_rounds": 0,
            "active_role": "main",
            "main_slot": "gpt-5_6-sol",
            "sidekick_slot": "gpt-5_6-luna",
            "main_router": None,
            "sidekick_router": None,
            "delegation_mode": "forced",
            "goal": getattr(self, "brief", "") or "",
            "latest_user": getattr(self, "brief", "") or "",
            "completed_via": "",
            "_resume_allows_answer": False,
            "worker_profiles": {},
            "budget": None,
            "tool_options": FusionToolOptions(),
            "structured": False,
            "structured_plan": None,
            "sidekick_lanes": [],
            "sidekick_reports": [],
        }.items():
            if not hasattr(self, key):
                setattr(self, key, default)
        if not hasattr(self, "repeat_guard"):
            self.repeat_guard = utils.RepeatToolGuard()

    def _append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        pending = {"asst": {"tool_calls": [dict(call) for call in self.pending_tool_calls]}}
        self.record_tool_results(pending, tool_results)
        target = self.main_messages if self.active_role == "main" else self.sidekick_messages
        for result in tool_results:
            raw_content = str(result.get("content", ""))
            target.append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": raw_content[: utils.RUN_MAX_MSG_BYTES],
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
                    target.append({"role": "user", "content": utils.system_reminder(reminder)})

    def _call_lane(
        self,
        coordinator: FusionCoordinator,
        role: str,
        prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        *,
        turn_index: int = 0,
        escalation_count: int | None = None,
        slot_override: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Call one lane's worker and record usage/activity for the turn."""
        esc = self.follow_up_count if escalation_count is None else escalation_count
        if slot_override is not None:
            slot = slot_override
        elif role == "main":
            slot = (
                self.main_router.select(turn_index, esc)
                if self.main_router is not None
                else self.main_slot
            )
        else:
            slot = (
                self.sidekick_router.select(turn_index, esc)
                if self.sidekick_router is not None
                else self.sidekick_slot
            )
        messages = self.main_messages if role == "main" else self.sidekick_messages
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        self.cache_namespace = providers._prompt_cache_namespace(messages, self.tools or None)
        if self.budget is not None:
            self.budget.check_timeout()
        message, usage = coordinator._call_worker(slot, messages, tools)
        messages.append(message)
        if self.budget is not None:
            self.budget.consume_tokens(usage)
            self.budget.consume_turn()
        text = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if getattr(serve_config._history_context, "active_run", None) is not self:
            self.add_usage(usage, model=slot)
        self.record_activity(
            f"{role}_turn",
            role=role,
            model=slot,
            status="completed",
            summary=text[:200] if text else "tool-calls",
        )
        self.turns.append({"role": role, "model_name": slot})
        providers._emit_progress(
            {
                "type": f"{role}_turn",
                "role": role,
                "model": slot,
                "status": "completed",
                "summary": text[:200] if text else "tool-calls",
            }
        )
        return text, calls, usage

    def _parse_main_answer(self, text: str) -> str | None:
        match = re.match(r"^\s*ANSWER:\s*(.*)\s*$", text.strip(), re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        answer = match.group(1).strip()
        return answer or None

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

    def _parse_sidekick_escalate(self, text: str) -> str | None:
        match = re.match(
            r"^\s*ESCALATE_TO_MAIN:\s*(.*)\s*$", text.strip(), re.DOTALL | re.IGNORECASE
        )
        if not match:
            return None
        reason = match.group(1).strip()
        return reason or "sidekick requested escalation"

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

    def _queue_sidekick_brief(self) -> None:
        self.sidekick_messages.append(
            {
                "role": "user",
                "content": _format_fusion_brief(
                    self.goal, self.latest_user, self.plan, self.sidekick_brief
                ),
            }
        )
        self.sidekick_tool_rounds = 0
        self.status = "sidekick_pending"

    def _complete_answer(self, answer: str) -> None:
        self.report = answer
        self.completed_via = "answer"
        self.status = "completed"

    def _handle_main_planning_text(
        self,
        coordinator: FusionCoordinator,
        main_text: str,
        *,
        allow_retry: bool,
        allow_answer: bool | None = None,
    ) -> None:
        """Parse ANSWER or PLAN/BRIEF from main; may issue one reminder retry."""
        answer_ok = self.delegation_mode == "available" if allow_answer is None else allow_answer
        answer = self._parse_main_answer(main_text)
        if answer is not None:
            if not answer_ok:
                if not allow_retry:
                    raise ValueError("main emitted ANSWER on a forced-delegation run")
                main_text, main_calls, _ = self._call_lane(
                    coordinator, "main", prompt=FORCED_ANSWER_REMINDER_PROMPT, tools=None
                )
                if main_calls:
                    raise ValueError("main called tools instead of producing a plan")
                self._handle_main_planning_text(
                    coordinator,
                    main_text,
                    allow_retry=False,
                    allow_answer=False,
                )
                return
            self._complete_answer(answer)
            return

        try:
            self.plan, self.sidekick_brief = self._parse_main_plan(main_text)
        except ValueError:
            if not allow_retry:
                raise
            reminder = AVAILABLE_REMINDER_PROMPT if answer_ok else PLAN_REMINDER_PROMPT
            main_text, main_calls, _ = self._call_lane(
                coordinator, "main", prompt=reminder, tools=None
            )
            if main_calls:
                raise ValueError("main called tools instead of producing a plan") from None
            self._handle_main_planning_text(
                coordinator,
                main_text,
                allow_retry=False,
                allow_answer=answer_ok,
            )
            return
        self._queue_sidekick_brief()

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
            "usage_models": self.usage_models,
            "cost": providers._usage_cost(self.usage_models),
            "follow_up_count": self.follow_up_count,
            "follow_up_capped": self.follow_up_capped,
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
            "usage_models": self.usage_models,
            "cost": providers._usage_cost(self.usage_models),
            "follow_up_count": self.follow_up_count,
            "follow_up_capped": self.follow_up_capped,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def _resolve_profile(
        self,
        name: str | None,
        coordinator: FusionCoordinator,
    ) -> FusionWorkerProfile:
        """Return a worker profile by name, falling back to catalog or skills."""
        if not name:
            return FusionWorkerProfile(name="default")
        if name in self.worker_profiles:
            return self.worker_profiles[name]
        if name in coordinator.worker_profiles:
            return coordinator.worker_profiles[name]
        skill = utils._load_skill_profile(name)
        if skill is not None:
            return skill
        return FusionWorkerProfile(name=name)

    def _filter_tools(self) -> list[dict[str, Any]]:
        """Apply tool_options and profile filtering to the tool list."""
        tools = utils._filter_tools_by_options(self.tools, self.tool_options)
        return tools

    def _parse_plan_text(self, text: str) -> FusionPlan | None:
        """Parse a structured FusionPlan from JSON or legacy PLAN:/BRIEF: text."""
        with suppress(ValueError, json.JSONDecodeError):
            return FusionPlan.model_validate_json(text)
        answer = self._parse_main_answer(text)
        if answer is not None:
            return FusionPlan(
                complexity=0.0,
                main_task=answer,
                sidekick_assignments=[],
            )
        try:
            plan, brief = self._parse_main_plan(text)
            return FusionPlan(
                complexity=0.5,
                main_task=plan,
                sidekick_assignments=[FusionSidekickAssignment(task=brief)],
            )
        except ValueError:
            return None

    def _distribute_tool_results(
        self, tool_results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Map each tool result to the sidekick lane it belongs to."""
        by_lane: dict[str, list[dict[str, Any]]] = {}
        for result in tool_results:
            tc_id = str(result.get("tool_call_id", ""))
            prefix = tc_id.split(":", 1)[0] if ":" in tc_id else ""
            by_lane.setdefault(prefix, []).append(result)
        return by_lane

    def _advance(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        if self.structured:
            return self._advance_structured(tool_results, request_id, message, coordinator)
        return self._advance_legacy(tool_results, request_id, message, coordinator)

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

    def _advance_legacy(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        if self.cancelled:
            raise RuntimeError("run is cancelled")

        if message is not None and tool_results:
            raise ValueError("cannot combine message follow-up with tool_results")

        if message is not None:
            if self.status == "awaiting_tools":
                raise ValueError("message follow-up is invalid while awaiting_tools")
            if self.status == "error":
                raise ValueError("cannot follow up an errored fusion run")
            self.latest_user = message
            self.main_messages.append({"role": "user", "content": message})
            self.pending_tool_calls = []
            self.planning_tool_rounds = 0
            self.status = "main_planning"
            self._resume_allows_answer = True

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

        # Main planning on a fresh run, textual resume, or resumed planning.
        if self.status == "main_planning":
            self.active_role = "main"
            available_tools = (
                self.tools
                if "plan" in self.main_tools_policy and self.planning_tool_rounds < 2 and self.tools
                else None
            )
            main_text, main_calls, _ = self._call_lane(coordinator, "main", tools=available_tools)

            # The main model may only call tools it was actually offered ("plan"
            # policy, max two rounds). Stray calls when no tools were offered,
            # unknown tools, or budget exhaustion force it back to text.
            if main_calls:
                allowed_names = {
                    t.get("function", {}).get("name") for t in self.tools if t.get("function")
                }
                unknown = [
                    c for c in main_calls if c.get("function", {}).get("name") not in allowed_names
                ]
                if unknown or available_tools is None or self.planning_tool_rounds >= 2:
                    if unknown:
                        unknown_names = {
                            str(c.get("function", {}).get("name", "")) for c in unknown
                        }
                        names = ", ".join(sorted(unknown_names))
                        reminder = PLAN_UNKNOWN_TOOL_PROMPT.format(names=names)
                    else:
                        reminder = PLAN_TOOL_BUDGET_PROMPT
                    main_text, main_calls, _ = self._call_lane(
                        coordinator, "main", prompt=reminder, tools=None
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

            self._handle_main_planning_text(
                coordinator,
                main_text,
                allow_retry=True,
                allow_answer=True if self._resume_allows_answer else None,
            )
            self._resume_allows_answer = False
            if self.status == "completed":
                return self._ok_event(request_id)

        # Bounded sidekick-main loop.
        max_iterations = max(1, coordinator.max_follow_ups + 1)
        for _ in range(max_iterations * 4):  # generous step ceiling
            if self.status == "sidekick_pending":
                self.active_role = "sidekick"
                at_tool_budget = self.sidekick_tool_rounds >= coordinator.sidekick_max_tool_rounds
                tools_allowed = None if at_tool_budget else self.tools
                sidekick_text, sidekick_calls, _ = self._call_lane(
                    coordinator, "sidekick", tools=tools_allowed
                )
                if sidekick_calls:
                    if at_tool_budget:
                        sidekick_text, sidekick_calls, _ = self._call_lane(
                            coordinator,
                            "sidekick",
                            prompt=SIDEKICK_TOOL_BUDGET_PROMPT,
                            tools=None,
                        )
                        if sidekick_calls:
                            sidekick_text = (
                                "ESCALATE_TO_MAIN: sidekick kept calling tools "
                                "after the tool budget was exhausted"
                            )
                            sidekick_calls = []
                    else:
                        self.sidekick_tool_rounds += 1
                        self.pending_tool_calls = sidekick_calls
                        self.status = "awaiting_tools"
                        break

                escalate_reason = self._parse_sidekick_escalate(sidekick_text)
                if escalate_reason is not None:
                    if self.follow_up_count >= coordinator.max_follow_ups:
                        self.follow_up_capped = True
                        self.report = self.report or sidekick_text
                        self.completed_via = "capped"
                        self.status = "completed"
                        break
                    self.follow_up_count += 1
                    self.main_messages.append(
                        {
                            "role": "user",
                            "content": f"{ESCALATE_PROMPT}{escalate_reason}",
                        }
                    )
                    self.active_role = "main"
                    esc_text, esc_calls, _ = self._call_lane(coordinator, "main", tools=None)
                    if esc_calls:
                        raise ValueError("main called tools during escalation handling")
                    self._handle_main_planning_text(
                        coordinator,
                        esc_text,
                        allow_retry=True,
                        allow_answer=True,
                    )
                    if self.status == "completed":
                        break
                    continue

                # Queue the report for the single review state. Keeping the prompt in
                # main_messages also lets a tool-assisted review resume without a
                # separate first-review branch.
                remaining = coordinator.max_follow_ups - self.follow_up_count
                review_prompt = (
                    f"{REVIEW_PROMPT}"
                    f"Tool Activity by Sidekick:\n{self._summarize_sidekick_tool_history()}\n\n"
                    f"Report:\n{sidekick_text}\n\n"
                    f"Follow-up budget remaining: {remaining} of {coordinator.max_follow_ups}."
                )
                self.main_messages.append({"role": "user", "content": review_prompt})
                self.status = "main_review"
                continue
            if self.status == "main_review":
                self.active_role = "main"
                review_text, review_calls, _ = self._call_lane(
                    coordinator,
                    "main",
                    tools=self.tools if "review" in self.main_tools_policy else None,
                )
                if review_calls:
                    self.pending_tool_calls = review_calls
                    self.status = "awaiting_tools"
                    break
                accepted, feedback = self._parse_main_review(review_text)
                if accepted:
                    last_sidekick_text = self.sidekick_messages[-1].get("content", "")
                    self.report = self.report or last_sidekick_text
                    self.completed_via = "accept"
                    self.status = "completed"
                    break
                if self.follow_up_count >= coordinator.max_follow_ups:
                    self.follow_up_capped = True
                    last_sidekick_text = self.sidekick_messages[-1].get("content", "")
                    self.report = self.report or last_sidekick_text
                    self.completed_via = "capped"
                    self.status = "completed"
                    break
                self.follow_up_count += 1
                self.sidekick_messages.append(
                    {"role": "user", "content": _format_fusion_follow_up(feedback)}
                )
                self.status = "sidekick_pending"
                continue
            if self.status in ("completed", "awaiting_tools", "error"):
                break

        return self._ok_event(request_id)

    def _advance_structured(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        """Run the structured main/sidekick swarm state machine."""
        if self.cancelled:
            raise RuntimeError("run is cancelled")

        if message is not None and tool_results:
            raise ValueError("cannot combine message follow-up with tool_results")

        if message is not None:
            if self.status == "awaiting_tools":
                raise ValueError("message follow-up is invalid while awaiting_tools")
            if self.status == "error":
                raise ValueError("cannot follow up an errored fusion run")
            self.latest_user = message
            self.main_messages.append({"role": "user", "content": message})
            self.structured_plan = None
            self.sidekick_lanes = []
            self.sidekick_reports = []
            self.pending_tool_calls = []
            self.follow_up_count = 0
            self.status = "main_planning"
            self._resume_allows_answer = True

        if self.budget is not None:
            self.budget.check_timeout()

        # Apply client tool results to the lanes that requested them.
        if self.status == "awaiting_tools":
            by_lane = self._distribute_tool_results(tool_results)
            for lane in self.sidekick_lanes:
                lane_tool_results = by_lane.get(lane.lane_id, [])
                if lane_tool_results:
                    lane.step(lane_tool_results)
            # Also allow main-lane tool results if main is reviewing with tools.
            if self.active_role == "main" and self.pending_tool_calls:
                main_results = by_lane.get("main", []) + by_lane.get("", [])
                validated = self._validate_tool_results(main_results)
                self._append_tool_results(validated)
                self.pending_tool_calls = []
                self.status = "main_review"
            else:
                self.pending_tool_calls = []
                self.status = "sidekick_pending"
        elif tool_results:
            raise ValueError("tool_results are only valid when status is awaiting_tools")

        # Main planning with a structured JSON plan.
        if self.status == "main_planning":
            self.active_role = "main"
            self.active_response_format = PLAN_RESPONSE_FORMAT
            try:
                main_text, main_calls, _ = self._call_lane(coordinator, "main", tools=None)
            finally:
                self.active_response_format = None

            if main_calls:
                raise ValueError("main emitted tool calls in structured planning mode")

            plan = self._parse_plan_text(main_text)
            if plan is None:
                raise ValueError("main did not produce a valid FusionPlan")
            self.structured_plan = plan

            providers._emit_progress(
                {
                    "type": "plan.completed",
                    "role": "main",
                    "complexity": plan.complexity,
                    "assignments": len(plan.sidekick_assignments),
                }
            )

            if not plan.sidekick_assignments:
                self.report = plan.main_task
                self.completed_via = "answer"
                self.status = "completed"
                return self._ok_event(request_id)

            self.tools = self._filter_tools()
            for i, assignment in enumerate(plan.sidekick_assignments):
                profile = self._resolve_profile(assignment.profile, coordinator)
                lane = SidekickLane(
                    f"lane{i}",
                    self,
                    coordinator,
                    assignment,
                    profile,
                )
                self.sidekick_lanes.append(lane)
            self.status = "sidekick_pending"

        max_iterations = max(1, coordinator.max_follow_ups + 1)
        for _ in range(max_iterations * 4):
            if self.budget is not None:
                self.budget.check_timeout()

            if self.status == "sidekick_pending":
                self.active_role = "sidekick"
                all_pending: list[dict[str, Any]] = []
                runnable = [
                    lane
                    for lane in self.sidekick_lanes
                    if not lane.complete and not lane.error and not lane.pending_tool_calls
                ]
                # ponytail: one pool per batch; keep it until profiling justifies a persistent pool.
                # _history_context is thread-local, so hand the sink to lane
                # threads or their progress events vanish.
                sink = getattr(serve_config._history_context, "event_sink", None)

                def _step(lane: SidekickLane, event_sink: Any = sink) -> dict[str, Any]:
                    if event_sink is not None:
                        serve_config._history_context.event_sink = event_sink
                    return lane.step()

                with ThreadPoolExecutor(max_workers=max(1, len(runnable))) as executor:
                    statuses = list(executor.map(_step, runnable))
                for lane, status in zip(runnable, statuses, strict=True):
                    if status["status"] == "awaiting_tools":
                        all_pending.extend(lane.pending_tool_calls)

                if all_pending:
                    self.pending_tool_calls = all_pending
                    self.status = "awaiting_tools"
                    return self._ok_event(request_id)

                self.sidekick_reports = [
                    lane.report for lane in self.sidekick_lanes if lane.complete and not lane.error
                ]
                self.sidekick_messages = [
                    {"role": "user", "content": self.structured_plan.main_task}
                ]
                self.status = "main_review"
                continue

            if self.status == "main_review":
                self.active_role = "main"
                reports = "\n\n".join(
                    f"Report {i}:\n{report}"
                    for i, report in enumerate(self.sidekick_reports)
                    if report
                )
                review_prompt = f"{REVIEW_PROMPT}{reports}"
                review_text, review_calls, _ = self._call_lane(
                    coordinator,
                    "main",
                    prompt=review_prompt,
                    tools=self.tools if "review" in self.main_tools_policy else None,
                )

                if review_calls:
                    for call in review_calls:
                        call["id"] = f"main:{call['id']}"
                    self.pending_tool_calls = review_calls
                    self.status = "awaiting_tools"
                    return self._ok_event(request_id)

                accepted, feedback = self._parse_main_review(review_text)
                if accepted:
                    self.report = next((r for r in self.sidekick_reports if r), review_text)
                    self.completed_via = "accept"
                    self.status = "completed"
                    return self._ok_event(request_id)

                if self.follow_up_count >= coordinator.max_follow_ups:
                    self.follow_up_capped = True
                    self.report = next((r for r in self.sidekick_reports if r), review_text)
                    self.completed_via = "capped"
                    self.status = "completed"
                    return self._ok_event(request_id)

                self.follow_up_count += 1
                for lane in self.sidekick_lanes:
                    lane.apply_follow_up(feedback)
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
                status = str(event["status"])
                if status == "error":
                    terminated_by = "fusion_error"
                elif run.follow_up_capped:
                    terminated_by = "fusion_capped"
                elif run.completed_via == "answer":
                    terminated_by = "fusion_answer"
                else:
                    terminated_by = "fusion_accept"
                _write_learning_record(
                    run,
                    {
                        "type": "error" if status == "error" else "completed",
                        "terminated_by": terminated_by,
                        "error": run.error,
                    },
                )
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
    delegation_mode: str = "forced",
    worker_profiles: list[FusionWorkerProfile] | None = None,
    budget: FusionRunBudget | None = None,
    tool_options: FusionToolOptions | None = None,
) -> FusionRun:
    run_id = uuid.uuid4().hex
    run = FusionRun(
        run_id,
        brief,
        tools,
        messages=messages,
        delegation_mode=delegation_mode,
        worker_profiles=worker_profiles,
        budget=budget,
        tool_options=tool_options,
    )
    _put_run(run)
    return run


def try_get_fusion_run(run_id: str) -> FusionRun | None:
    """Return a non-error FusionRun if present; otherwise None."""
    try:
        run = get_run(run_id)
    except KeyError:
        return None
    if not isinstance(run, FusionRun):
        return None
    if run.status == "error":
        return None
    return run


def fusion_run_status(run_id: str) -> dict[str, Any]:
    """Return the current status of a FusionRun without advancing it."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        return run._ok_event(None)
