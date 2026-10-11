"""mantis-fusion: catalog-driven lead/sidekick resumable orchestration."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from typing import Any, cast

import direct_completion
import fusion_background
import fusion_history
import fusion_lead
import fusion_limits
import fusion_provider_hooks as hooks
import litellm
import providers
import serve_config
import utils
from fusion_budget import FusionBudgetExceededError, FusionBudgetGuard
from fusion_config import FusionConfig
from fusion_prompts import sidekick_system_prompt
from fusion_router import FusionRouter
from fusion_types import FusionRoutingConfig, FusionRunBudget, FusionToolOptions
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
    "FusionConfig",
    "FusionCoordinator",
    "FusionRouter",
    "FusionRoutingConfig",
    "FusionRun",
    "FUSION_STATUS",
    "advance_fusion_run",
]

FUSION_STATUS = frozenset({"main_planning", "awaiting_tools", "completed", "error"})


@contextmanager
def _fusion_route_session(session_id: str):
    """Bind a stable Switchyard identity to the current Fusion lane thread."""
    context = serve_config._history_context
    previous = getattr(context, "fusion_route_session_id", None)
    context.fusion_route_session_id = session_id
    try:
        yield
    finally:
        if previous is None:
            with suppress(AttributeError):
                del context.fusion_route_session_id
        else:
            context.fusion_route_session_id = previous


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


def _trailing_error_streak(messages: list[dict[str, Any]]) -> int:
    """Count consecutive failed tool results at the tail of a lane transcript."""
    streak = 0
    for message in reversed(messages):
        if message.get("role") != "tool":
            continue
        if not message.get("is_error"):
            break
        streak += 1
    return streak


TERMINAL_SYNTHESIS_PROMPT = (
    "The Fusion run guard is exhausted; no further sidekick work or review is "
    "possible. Write the final user-facing answer now from "
    "the material below: give the best verified result available, and state "
    "clearly what remains unfinished or unverified. "
    "Return a plain assistant answer.\n"
)


_MAIN_TRACE_MAX_CHARS = 8192


_FUSION_CONFIG = FusionConfig()


def _fusion_config() -> FusionConfig:
    """Return the global FusionConfig, reloaded on first call in a new process."""
    return _FUSION_CONFIG


def _return_reasoning_by_default() -> bool:
    return os.environ.get("MANTIS_FUSION_RETURN_REASONING", "0").lower() in ("1", "true", "yes")


def _extract_reasoning_trace(
    messages: list[dict[str, Any]],
    max_chars: int = 8192,
) -> str:
    """Return safe textual provider summaries, excluding replay-only metadata.

    Identical pieces are emitted once: providers can mirror the same summary
    under ``reasoning``, ``reasoning_details`` and ``_anthropic_content``.
    """
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
    return "\n\n".join(dict.fromkeys(pieces))


def _model_label(slot: str) -> str:
    """Use a readable catalog label without exposing provider internals."""
    return slot.replace("_", ".", 1)


def _sidekick_trace_delta(run: FusionRun) -> str:
    """Return the sidekick reasoning added since the previous trace.

    The cursor is persisted on the run so each response carries new thinking
    rather than replaying the accumulated history. The brief is emitted once
    as a fallback when the worker produced no reasoning.
    """
    cursor = run.trace_cursors.get("sidekick", 0)
    if cursor > len(run.sidekick_messages):
        cursor = 0
    reasoning = _extract_reasoning_trace(run.sidekick_messages[cursor:])
    run.trace_cursors["sidekick"] = len(run.sidekick_messages)
    if reasoning:
        return reasoning
    return ""


def _orchestration_positions(run: FusionRun) -> dict[str, int]:
    cursor = run.trace_cursors.get("orchestration")
    if not isinstance(cursor, dict):
        cursor = {"main": 0, "sidekick": 0, "preamble": 0}
        run.trace_cursors["orchestration"] = cursor
    for lane in ("main", "sidekick"):
        if not isinstance(cursor.get(lane), int) or cursor[lane] < 0:
            cursor[lane] = 0
    if not isinstance(cursor.get("preamble"), int) or cursor["preamble"] < 0:
        cursor["preamble"] = 0
    return cursor


def _orchestration_provider_delta(run: FusionRun) -> str:
    cursor = _orchestration_positions(run)
    main_start = min(cursor["main"], len(run.main_messages))
    sidekick_start = min(cursor["sidekick"], len(run.sidekick_messages))
    provider = _extract_reasoning_trace(
        run.main_messages[main_start:] + run.sidekick_messages[sidekick_start:]
    )
    cursor["main"] = len(run.main_messages)
    cursor["sidekick"] = len(run.sidekick_messages)
    return provider


def _run_guard_remaining(run: FusionRun) -> dict[str, Any]:
    """Return the run guard's remaining turn, token, and timeout budget."""
    guard = run.budget
    return {
        "turns": (
            max(0, guard.max_turns - guard.turns)
            if guard is not None and guard.max_turns is not None
            else None
        ),
        "tokens": (
            max(0, guard.max_tokens - guard.tokens)
            if guard is not None and guard.max_tokens is not None
            else None
        ),
        "timeout_ms": guard.timeout_ms if guard is not None else None,
    }


def _role_usage_totals(role_usage: dict[str, dict[str, dict[str, int]]]) -> tuple[dict, dict]:
    """Flatten per-role usage into run totals and per-model token counts."""
    prompt = completion = cached = 0
    models: dict[str, dict[str, int]] = {}
    for per_model in role_usage.values():
        for slot, tokens in per_model.items():
            label = _model_label(slot)
            merged = models.setdefault(label, {"prompt_tokens": 0, "completion_tokens": 0})
            merged["prompt_tokens"] += int(tokens.get("prompt_tokens") or 0)
            merged["completion_tokens"] += int(tokens.get("completion_tokens") or 0)
            prompt += int(tokens.get("prompt_tokens") or 0)
            completion += int(tokens.get("completion_tokens") or 0)
            cached += int(tokens.get("cached_tokens") or 0)
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    return usage, models


def _merge_native_usage(metadata: dict[str, Any], models: dict[str, Any]) -> dict[str, Any]:
    """Fold serve-layer per-model usage into run metadata when lanes recorded none."""
    prompt = completion = cached = 0
    for tokens in models.values():
        if not isinstance(tokens, dict):
            continue
        prompt += int(tokens.get("prompt_tokens") or 0)
        completion += int(tokens.get("completion_tokens") or 0)
        cached += int(tokens.get("cached_tokens") or 0)
    metadata = {**metadata}
    metadata["usage"] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    metadata["usage_models"] = models
    return metadata


def _orchestration_metadata(run: FusionRun) -> dict[str, Any]:
    tool_count = sum(
        len(msg.get("tool_calls") or [])
        for msg in run.sidekick_messages
        if msg.get("role") == "assistant"
    )
    usage, usage_models = _role_usage_totals(run.role_usage)
    return {
        "usage": usage,
        "usage_models": usage_models,
        "run_id": run.run_id,
        "status": run.status,
        "main_model": _model_label(getattr(run, "main_slot", "gpt-5_6-sol")),
        "sidekick_model": _model_label(getattr(run, "sidekick_slot", "gpt-5_6-luna")),
        "tool_count": tool_count,
        "tool_results": sum(msg.get("role") == "tool" for msg in run.sidekick_messages),
        "fusion_savings_usd": run._fusion_savings_usd(),
        "handoff_history": list(getattr(run, "handoff_history", [])),
        "run_guard_remaining": _run_guard_remaining(run),
        "review": (
            "accepted"
            if run.status == "completed"
            else "failed"
            if run.status == "error"
            else "pending"
        ),
    }


def _main_trace_delta(run: FusionRun) -> str:
    """Return the main lane's own reasoning and tool activity since the last trace."""
    cursor = run.trace_cursors.get("main", 0)
    if not isinstance(cursor, int) or cursor < 0 or cursor > len(run.main_messages):
        cursor = 0
    delta = fusion_lead.visible_lead_messages(run.main_messages[cursor:])
    run.trace_cursors["main"] = len(run.main_messages)
    parts: list[str] = []
    length = 0

    def add(piece: str) -> bool:
        nonlocal length
        piece = piece.strip()
        if not piece:
            return True
        if length + len(piece) + 2 > _MAIN_TRACE_MAX_CHARS:
            remaining = _MAIN_TRACE_MAX_CHARS - length
            if remaining > 0:
                parts.append(piece[:remaining] + "\n...")
            return False
        parts.append(piece)
        length += len(piece) + 2
        return True

    for message in delta:
        role = message.get("role")
        if role == "assistant":
            reasoning = _extract_reasoning_trace([message])
            if reasoning:
                add(reasoning)
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                arguments = str(function.get("arguments") or "")
                add(f"Tool call: {name} {arguments}".strip())
        elif role == "tool":
            content = utils.prune_tool_result(
                str(message.get("content") or ""),
                max_chars=2048,
                head_chars=1536,
                tail_chars=384,
            )
            add(f"Tool result ({message.get('tool_call_id', '')}):\n{content}")
    return "\n\n".join(parts)


def _orchestration_trace(
    run: FusionRun,
    include_reasoning: bool = False,
    trace_scope: str = "orchestration",
) -> str:
    """Return only reasoning not sent in an earlier response for this run."""
    if trace_scope == "main":
        return _main_trace_delta(run)
    if trace_scope == "sidekick":
        return _sidekick_trace_delta(run)
    parts = [part for part in (_orchestration_provider_delta(run),) if part]
    if include_reasoning and not parts:
        return ""
    return "\n\n".join(parts)


class FusionCoordinator:
    """Thin wrapper that calls catalog-bound worker slots."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or _fusion_config()
        self.main_router = self.config.main_router()
        self.sidekick_router = self.config.sidekick_router()
        self.main_slot = self.select("main")
        self.sidekick_slot = self.select("sidekick")
        self.sidekick_max_tool_rounds = self.config.sidekick_max_tool_rounds()
        self.sidekick_max_consecutive_errors = self.config.sidekick_max_consecutive_errors()
        self.run_max_turns = self.config.run_max_turns()
        self.context_window = self.config.context_window()
        self.max_output_tokens = self.config.max_output_tokens()
        self.default_tool_options = self.config.tool_options()
        self.default_budget = self.config.default_budget()

    def select(self, role: str, escalation_count: int = 0) -> str:
        router = self.main_router if role == "main" else self.sidekick_router
        return cast(str, router.select(escalation_count))

    # JSON includes replay-only reasoning metadata and tool schemas that
    # LiteLLM's chat-message counter intentionally ignores. Tokenize the full
    # request shape so the context guard measures what Mantis actually stores
    # and replays, without maintaining another tokenizer implementation here.
    def _token_sum(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        if not messages:
            return 0
        payload = json.dumps(
            {"messages": messages, "tools": tools or []},
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        try:
            counted = litellm.token_counter(model="gpt-4o", text=payload)
            return max(1, int(counted))
        except Exception:  # noqa: BLE001 - counting must never break a call
            return max(1, len(payload) // 4)

    def _tool_tokens(self, tools: list[dict[str, Any]] | None) -> int:
        if not tools:
            return 0
        empty = [{"role": "user", "content": ""}]
        return max(0, self._token_sum(empty, tools) - self._token_sum(empty))

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
        """Drop the oldest groups first, preserving tool pairing and live prompts.

        The latest group is always kept (even over budget) so the newest
        review/follow-up contract reaches the provider; its oversized tool
        bodies are pruned instead.
        """
        if not messages:
            return messages
        trimmed = [messages[0]]
        budget = max_input_tokens - self._token_sum(trimmed)
        groups = self._message_groups(messages[1:])
        kept: list[list[dict[str, Any]]] = []
        for group in reversed(groups):
            group_tokens = self._token_sum(group)
            if group_tokens > budget:
                if not kept:
                    # Never drop the live contract: prune its tool bodies first.
                    kept = [
                        self._prune_tool_messages(
                            group, max_chars=2048, head_chars=1024, tail_chars=256
                        )
                    ]
                break
            kept.insert(0, group)
            budget -= group_tokens
        for group in kept:
            trimmed.extend(group)
        return trimmed

    def _fit_messages(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        max_input_tokens: int,
    ) -> list[dict[str, Any]]:
        """Fit history for one provider call without losing the live contract.

        Tool-pairing is preserved and the newest review/follow-up
        prompt always survives fitting so the live contract reaches the model.
        """
        if self._token_sum(messages, tools) <= max_input_tokens:
            return messages
        cleared = providers._clear_thinking(messages, keep=2)
        pruned = self._prune_for_budget(cleared, max_input_tokens)
        if self._token_sum(pruned, tools) <= max_input_tokens:
            return pruned
        return self._drop_old_groups(pruned, max_input_tokens - self._tool_tokens(tools))

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
        """Return the slot's authoritative output cap capped by request and config."""
        limits = fusion_limits.resolve_model_limits(slot)
        output = limits.max_output_tokens
        caps = [
            value
            for value in (self.max_output_tokens, fusion_history.requested_output(self, slot))
            if value
        ]
        return min(output, *caps) if caps else output

    def _upstream_model_for(self, slot: str) -> str:
        """Return the upstream model name a slot resolves to, or the slot itself."""
        try:
            return cast(str, providers._resolve_model_spec(slot).model)
        except (RuntimeError, ValueError, AttributeError):
            return slot

    def _context_window_for(self, slot: str) -> int:
        """Return the slot's authoritative window capped by any explicit budget."""
        limits = fusion_limits.resolve_model_limits(slot)
        window = limits.context_window
        return min(window, self.context_window) if self.context_window else window

    def _call_worker(
        self,
        slot: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        on_delta: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Call a worker slot and return (assistant_message, usage).

        The returned assistant message preserves provider-specific metadata such
        as ``reasoning``, ``reasoning_details``, ``_anthropic_content`` and
        ``_anthropic_tool_ids`` so that later turns can replay required state.
        """
        routed = slot == "mantis/base"
        serve_config._history_context.fusion_compacted = False
        serve_config._history_context.fusion_selected_model = None
        if routed:
            # The virtual route has no single context window or output cap.
            # Switchyard applies the selected target's catalog-owned limits.
            fitted = messages
            output_tokens = None
        else:
            output_tokens = self._output_tokens_for(slot)
            input_budget = max(0, self._context_window_for(slot) - output_tokens)
            before_tokens = self._token_sum(messages)
            fitted = self._fit_messages(slot, messages, tools, input_budget)
            if fitted is not messages and self._token_sum(fitted) < before_tokens:
                # Persist the fitted history so trimming is not recomputed (and tool
                # bodies are not re-sent in full) on every later provider call.
                after_tokens = self._token_sum(fitted)
                messages[:] = fitted
                serve_config._history_context.fusion_compacted = True
                providers._emit_progress(
                    {
                        "type": "fusion_context_trimmed",
                        "model": slot,
                        "input_budget": input_budget,
                        "tokens_before": before_tokens,
                        "tokens_after": after_tokens,
                        "tokens_dropped": before_tokens - after_tokens,
                    }
                )
        run = getattr(serve_config._history_context, "active_run", None)
        budget = getattr(run, "budget", None)
        timeout_s = budget.remaining_timeout_s() if budget is not None else None
        selected_model: str | None = None
        safe_tokens = output_tokens if output_tokens is not None else 4096
        if routed:
            result = direct_completion._routed_result(
                fitted,
                0.7,
                timeout_s,
                getattr(serve_config._history_context, "fusion_route_session_id", None),
                tools,
            )
            data = result.body
            selected_model = result.selected_model
        else:
            data = hooks.call_provider(
                providers._provider_response, slot, fitted, safe_tokens, tools, timeout_s, on_delta
            )
        msg = dict(data["choices"][0]["message"])
        msg.setdefault("role", "assistant")
        msg["content"] = str(msg.get("content") or "")
        actual_model = selected_model or self._upstream_model_for(slot)
        if routed:
            serve_config._history_context.fusion_selected_model = actual_model
        else:
            msg["_mantis_model"] = actual_model
        tcs = msg.get("tool_calls") or []
        if routed:
            if tcs:
                msg["tool_calls"] = tcs
            else:
                msg.pop("tool_calls", None)
            return msg, data.get("usage") or {}
        calls = hooks.normalize_tool_calls(tcs)
        if calls:
            msg["tool_calls"] = calls
        else:
            msg.pop("tool_calls", None)
        usage = data.get("usage") or {}
        return msg, usage


class FusionRun(NativeRun):
    """A resumable lead/sidekick run whose roles are bound by the catalog."""

    handoffs: fusion_lead.HandoffManager | None = None
    fusion_session_id: str | None = None
    fusion_lead_prompt_base: str | None = None
    fusion_public_checkpoint: int | None = None

    def __init__(
        self,
        run_id: str,
        brief: str = "",
        tools: list[dict[str, Any]] | None = None,
        messages: list[dict[str, Any]] | None = None,
        budget: FusionRunBudget | None = None,
        tool_options: FusionToolOptions | None = None,
    ) -> None:
        super().__init__(run_id)
        self.kind = "fusion"
        self.brief = brief
        self.tool_options = tool_options or FusionToolOptions()
        goal, latest_user = _user_message_texts(messages, brief)
        self.goal = goal
        self.latest_user = latest_user
        # Feeds the learning record's task/task_hash (redacted by runs.py).
        self.query = goal
        coordinator = FusionCoordinator()
        budget_values = budget.model_dump() if budget is not None else {}
        budget_values.setdefault("max_turns", coordinator.run_max_turns)
        self.budget: FusionBudgetGuard | None = FusionBudgetGuard(**budget_values)
        # Keep the slots with the run so status/trace responses remain stable
        # if the catalog is changed while a run is in progress.
        self.main_slot = coordinator.main_slot
        self.sidekick_slot = coordinator.sidekick_slot
        self.sidekick_max_tool_rounds = coordinator.sidekick_max_tool_rounds
        self.sidekick_max_consecutive_errors = coordinator.sidekick_max_consecutive_errors
        self.run_max_turns = coordinator.run_max_turns
        self.main_router = coordinator.main_router
        self.sidekick_router = coordinator.sidekick_router
        self.main_compaction_slot: str | None = None
        self.sidekick_compaction_slot: str | None = None
        self.slot_models = [self.main_slot, self.sidekick_slot]
        self.tools = utils._convert_tools(tools)
        preamble = ""
        if messages:
            self.main_messages: list[dict[str, Any]] = [
                {"role": "system", "content": preamble},
                *messages,
            ]
        else:
            self.main_messages = [
                {"role": "system", "content": preamble},
                {"role": "user", "content": brief},
            ]
        self.sidekick_messages: list[dict[str, Any]] = [
            {"role": "system", "content": sidekick_system_prompt()},
        ]
        self.active_role: str = "main"
        self.pending_tool_calls: list[dict[str, Any]] = []
        self.sidekick_tool_rounds = 0
        self.subtask_report: str | None = None
        self.repeat_guard = utils.RepeatToolGuard()
        self.report: str | None = None
        self.error: str | None = None
        self.follow_up_count = 0
        self.follow_up_capped = False
        self.completed_via: str = ""
        self.turns: list[dict[str, Any]] = []
        self.role_usage: dict[str, dict[str, dict[str, int]]] = {}
        self.sidekick_context_cache: dict[str, list[dict[str, Any]]] = {}
        self.handoff_history: list[dict[str, Any]] = []
        self.pending_user_messages: list[str] = []
        self.trace_cursors: dict[str, Any] = {}
        self.status = "main_planning"

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        # Ensure new fields are present for forward compatibility.
        for field in ("goal", "latest_user"):
            if not hasattr(self, field):
                setattr(self, field, getattr(self, "brief", "") or "")
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
        escalation_count: int | None = None,
        slot_override: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Call one lane's worker and record usage/activity for the turn."""
        esc = self.follow_up_count if escalation_count is None else escalation_count
        if slot_override is not None:
            slot = slot_override
        elif role == "main":
            slot = self.main_compaction_slot or (
                self.main_router.select(esc) if self.main_router is not None else self.main_slot
            )
        else:
            slot = self.sidekick_compaction_slot or (
                self.sidekick_router.select(esc)
                if self.sidekick_router is not None
                else self.sidekick_slot
            )
        self._record_selected_slot(role, slot)
        messages = self.main_messages if role == "main" else self.sidekick_messages
        if prompt is not None:
            messages.append({"role": "user", "content": prompt})
        self.cache_namespace = providers._prompt_cache_namespace(
            messages, self.tools or None, coordinator._upstream_model_for(slot)
        )
        if self.budget is not None:
            self.budget.check_timeout()
        with _fusion_route_session(f"{self.run_id}:{role}"):
            message, usage = hooks.call_lane(self, coordinator, role, slot, messages, tools)
        selected = getattr(serve_config._history_context, "fusion_selected_model", None)
        self._record_selected_slot(
            role, str(selected or message.get("_mantis_model") or slot), primary=False
        )
        if getattr(serve_config._history_context, "fusion_compacted", False):
            self._reroute_after_compaction(
                role,
                slot,
                cache_warm=self._has_cache_hits(usage),
            )
        messages.append(message)
        if self.budget is not None:
            self.budget.consume_tokens(usage)
            self.budget.consume_turn()
        text = str(message.get("content") or "")
        calls = message.get("tool_calls") or []
        if getattr(serve_config._history_context, "active_run", None) is not self:
            self.add_usage(usage, model=slot)
        self._add_role_usage(role, slot, usage)
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

    @staticmethod
    def _has_cache_hits(usage: dict[str, Any]) -> bool:
        """Return True when the usage dict reports prompt-cache hits."""
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict):
            cached = details.get("cached_tokens", 0)
            return isinstance(cached, int) and cached > 0
        return False

    def _add_role_usage(self, role: str, slot: str, usage: dict[str, Any]) -> None:
        """Accumulate per-role, per-model token usage for cost reporting."""
        if not isinstance(usage, dict):
            return
        models = self.role_usage.setdefault(role, {})
        tokens = models.setdefault(
            slot, {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        )
        for key in ("prompt_tokens", "completion_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                tokens[key] += value
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if isinstance(cached, int) and cached >= 0:
            tokens["cached_tokens"] += cached

    def _fusion_savings_usd(self) -> float | None:
        """Estimate savings: sidekick tokens priced at the main slot's rates."""
        sidekick = self.role_usage.get("sidekick") or {}
        if not sidekick:
            return None
        prices = providers._price_map()
        main_key = providers._price_key(self.main_slot, prices)
        if main_key is None:
            return None
        main_input, main_output = prices[main_key]
        actual = 0.0
        counterfactual = 0.0
        for model, tokens in sidekick.items():
            cost = providers._model_cost(model, tokens, prices)
            if cost is None:
                return None
            actual += cost
            counterfactual += (
                int(tokens.get("prompt_tokens", 0)) * main_input
                + int(tokens.get("completion_tokens", 0)) * main_output
            )
        return round(counterfactual - actual, 6)

    def _reroute_after_compaction(self, role: str, previous: str, cache_warm: bool = False) -> None:
        router = self.main_router if role == "main" else self.sidekick_router
        if router is None:
            return
        complexity = 1.0
        lane_messages = self.main_messages if role == "main" else self.sidekick_messages
        failures = max(self.follow_up_count, _trailing_error_streak(lane_messages))
        selected = router.select_at_compaction(
            complexity,
            previous,
            failures,
            cache_warm=cache_warm,
        )
        if selected == previous:
            return
        if role == "main":
            self.main_compaction_slot = selected
            self.main_slot = selected
        else:
            self.sidekick_compaction_slot = selected
            self.sidekick_slot = selected
        if selected not in self.slot_models:
            self.slot_models.append(selected)
        self.record_activity(
            "fusion_reroute",
            role=role,
            model=selected,
            summary=f"Rerouted {role} after context compaction",
        )

    def _complete_answer(self, answer: str) -> None:
        self.report = answer
        self.completed_via = "answer"
        self._record_handoff("answer", summary=answer[:200])
        self.status = "completed"

    def _synthesize_final_answer(
        self,
        coordinator: FusionCoordinator,
        *,
        context: str,
        fallback: str,
    ) -> str:
        """Ask the main lane for the terminal user-facing answer.

        The run must end with a synthesized reply, not raw worker protocol
        text. Any failure falls back to the best available report text.
        """
        self.main_messages.append({"role": "user", "content": context})
        try:
            text, calls, _ = self._call_lane(coordinator, "main", tools=None)
            answer = text.strip() if not calls else None
        except Exception:  # noqa: BLE001 - terminal synthesis must not lose the run
            return fallback
        return answer or fallback

    def _synthesis_context(self, label: str, material: str) -> str:
        """Describe a terminal run guard without legacy protocol metadata."""
        return f"{TERMINAL_SYNTHESIS_PROMPT}\n\nGoal:\n{self.goal}\n\n{label}:\n{material}"

    def _record_selected_slot(self, role: str, slot: str, *, primary: bool = True) -> None:
        """Keep the reported slot in step with the slot that actually ran."""
        if primary:
            if role == "main":
                self.main_slot = slot
            else:
                self.sidekick_slot = slot
        if slot not in self.slot_models:
            self.slot_models.append(slot)

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
        return cast(list[dict[str, Any]], validated)

    def _error_event(self, exc: Exception, request_id: str | None) -> dict[str, Any]:
        """Close the run on an unexpected failure and return its error event."""
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
            "fusion_savings_usd": self._fusion_savings_usd(),
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
            "fusion_savings_usd": self._fusion_savings_usd(),
            "follow_up_count": self.follow_up_count,
            "follow_up_capped": self.follow_up_capped,
            "activity": self._activity,
        }
        if request_id is not None:
            self.request_events[request_id] = event
        return event

    def _advance(
        self,
        tool_results: list[dict[str, Any]],
        request_id: str | None,
        message: str | None,
        coordinator: FusionCoordinator,
    ) -> dict[str, Any]:
        if self.handoffs is None:
            fusion_lead.prepare_run(self)
        return fusion_lead.advance(self, tool_results, request_id, message, coordinator)

    def _finish_run_budget(
        self,
        coordinator: FusionCoordinator,
        error: Exception,
        request_id: str | None,
    ) -> dict[str, Any]:
        fallback = self.report or self.subtask_report or "Fusion stopped before a final report."
        previous_budget = self.budget
        self.budget = None
        try:
            self.report = self._synthesize_final_answer(
                coordinator,
                context=self._synthesis_context("Run budget exhausted", str(error)),
                fallback=fallback,
            )
        finally:
            self.budget = previous_budget
        self.completed_via = "budget"
        self.follow_up_capped = True
        self.status = "completed"
        return self._ok_event(request_id)

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
            if self.status == "awaiting_tools" and not tool_results and not self.cancelled:
                # A live run owns the session: queue any instruction and
                # re-emit the outstanding calls instead of forking a new
                # run or erroring on an empty retry.
                if message is not None:
                    self.pending_user_messages.append(message)
                return self._ok_event(request_id)
            return self._advance(tool_results, request_id, message, coordinator)
        except FusionBudgetExceededError as exc:
            return self._finish_run_budget(coordinator, exc, request_id)
        except Exception as exc:  # noqa: BLE001 - catch-all guard for worker/state errors
            return self._error_event(exc, request_id)

    def _snapshot_sidekick_contexts(
        self,
        coordinator: FusionCoordinator | None,
    ) -> None:
        """Cache healthy sidekick transcripts for the next user turn's lanes."""
        cache: dict[str, list[dict[str, Any]]] = {}
        hooks.cache_sidekick_history(self, coordinator, cache)
        self.sidekick_context_cache = cache

    def _record_handoff(self, kind: str, **fields: Any) -> None:
        """Append one bounded entry to the structured handoff history."""
        entry = {"kind": kind, "follow_up": self.follow_up_count, **fields}
        self.handoff_history.append(entry)
        del self.handoff_history[:-50]

    def advance_idempotent(
        self,
        tool_results: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        message: str | None = None,
        coordinator: FusionCoordinator | None = None,
        client_history: list[dict[str, Any]] | None = None,
        client_tool_ids: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if request_id is None:
            with self.request_lock:
                self._rebase_client_history(client_history, client_tool_ids)
                return self.advance(tool_results, request_id, message, coordinator)
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            raise ValueError("request_id must be a string of at most 128 characters")
        with self.request_lock:
            cached = self.request_events.get(request_id)
            if cached is not None:
                return cast(dict[str, Any], cached)
            self._rebase_client_history(client_history, client_tool_ids)
            event = cast(
                dict[str, Any],
                self.advance(tool_results, request_id, message, coordinator),
            )
            self.request_events[request_id] = event
            while len(self.request_events) > 64:
                self.request_events.pop(next(iter(self.request_events)))
            return event

    def _rebase_client_history(
        self,
        client_history: list[dict[str, Any]] | None,
        client_tool_ids: dict[str, str] | None,
    ) -> None:
        """Apply the harness's authoritative transcript when one is supplied."""
        if client_history is not None:
            fusion_history.rebase(self, client_history, client_tool_ids or {})


def _put_run(run: NativeRun) -> None:
    if isinstance(run, FusionRun):
        fusion_background.sync_run(run)
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


def _record_fusion_completion(run: FusionRun, status: str) -> None:
    """Write the terminal learning record for a finished Fusion run."""
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


def advance_fusion_run(
    run_id: str,
    request_id: str | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    message: str | None = None,
    client_history: list[dict[str, Any]] | None = None,
    client_tool_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Load, advance, and save a FusionRun with the same locking as advance_run."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        run.in_flight += 1
        _put_run(run)
        try:
            return _advance_locked_run(
                run, request_id, tool_results, message, client_history, client_tool_ids
            )
        finally:
            serve_config._history_context.active_run = None
            run.in_flight -= 1
            run.touch()
            _put_run(run)


def _advance_locked_run(
    run: FusionRun,
    request_id: str | None,
    tool_results: list[dict[str, Any]] | None,
    message: str | None,
    client_history: list[dict[str, Any]] | None,
    client_tool_ids: dict[str, str] | None,
) -> dict[str, Any]:
    """Advance an in-flight loaded run and record its terminal learning event."""
    serve_config._history_context.active_run = run
    coordinator = FusionCoordinator()
    event = run.advance_idempotent(
        tool_results,
        request_id,
        message,
        coordinator,
        client_history=client_history,
        client_tool_ids=client_tool_ids,
    )
    if event.get("status") in ("completed", "error"):
        _record_fusion_completion(run, str(event["status"]))
    return event


def create_fusion_run(
    brief: str,
    tools: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    budget: FusionRunBudget | None = None,
    tool_options: FusionToolOptions | None = None,
) -> FusionRun:
    run_id = uuid.uuid4().hex
    run = FusionRun(
        run_id,
        brief,
        tools,
        messages=messages,
        budget=budget,
        tool_options=tool_options,
    )
    _put_run(run)
    return run


def refresh_fusion_run_tools(run: FusionRun, tools: list[dict[str, Any]]) -> None:
    """Replace live schemas under the run lock and persist the matching delegation aliases."""
    with _run_lock(run.run_id):
        target, persisted = run, False
        with suppress(KeyError):
            stored = get_run(run.run_id)
            if isinstance(stored, FusionRun):
                target, persisted = stored, True
        fusion_lead.set_tool_schemas(target, tools)
        if persisted:
            _put_run(target)
        if target is not run:
            fusion_lead.copy_refreshed_tool_state(run, target)


def try_get_fusion_run(run_id: str) -> FusionRun | None:
    """Return a live FusionRun eligible for non-tool continuation."""
    try:
        run = get_run(run_id)
    except KeyError:
        return None
    if not isinstance(run, FusionRun):
        return None
    if run.status in {"completed", "error"} or run.cancelled or run.finished:
        return None
    return run


def fusion_run_status(run_id: str) -> dict[str, Any]:
    """Return the current status of a FusionRun without advancing it."""
    with _run_lock(run_id):
        run = get_run(run_id)
        if not isinstance(run, FusionRun):
            raise TypeError(f"run {run_id} is not a FusionRun")
        return run._ok_event(None)
