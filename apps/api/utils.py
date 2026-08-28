#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: OpenAI-compatible serving layer for the OpenFugu TRINITY + Conductor coordinators.
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

A client POSTs to /v1/chat/completions as if calling one model; internally the
requested coordinator ("trinity" or "conductor") runs the full loop. The
model field in the request selects the coordinator.

stdlib http.server only — no FastAPI/uvicorn.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, cast

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# When running from the repo's orchestration runtime, also find upstream OpenFugu.
if not (_HERE / "mini.py").exists():
    _OPENFUGU = _HERE.parent / "OpenFugu" / "openfugu"
    if _OPENFUGU.exists():
        sys.path.insert(0, str(_OPENFUGU))

import providers
import serve_config
from mini import (
    DEFAULT_SLOT_LABELS,
)
from serve_config import (
    _INTERNAL_TOOL_ID,
    _PUBLIC_RUN_TOKEN_LENGTH,
    _PUBLIC_TOOL_PREFIX,
    MODEL_MODES,
    RUN_MAX_MSG_BYTES,
)

import runs


def _split_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Return the text query and history; multimodal content stays in the run."""
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    if last_user_idx == -1:
        return "", messages
    return _message_text(messages[last_user_idx].get("content")), messages[:last_user_idx]


def _json_object(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw or b"{}")
    if not isinstance(value, dict):
        raise TypeError("request body must be a JSON object")
    return value


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return "" if content is None else str(content)


def _with_images(text: str, content: Any) -> Any:
    if not isinstance(content, list):
        return text
    images = [
        part for part in content if isinstance(part, dict) and part.get("type") == "image_url"
    ]
    return [{"type": "text", "text": text}, *images] if images else text


def _last_user_content(messages: list[dict[str, Any]]) -> Any:
    return next(
        (message.get("content") for message in reversed(messages) if message.get("role") == "user"),
        "",
    )


def _mode_for_model(model: Any) -> str:
    if not isinstance(model, str) or model not in MODEL_MODES:
        raise ValueError(f"unknown model: {model}")
    return MODEL_MODES[model]


def _public_tool_id(run_id: str, internal_id: str) -> str:
    run_token = base64.urlsafe_b64encode(bytes.fromhex(run_id)).decode().rstrip("=")
    return f"{_PUBLIC_TOOL_PREFIX}{run_token}_{internal_id}"


def _parse_public_tool_id(tool_id: Any) -> tuple[str, str] | None:
    if not isinstance(tool_id, str) or not tool_id.startswith(_PUBLIC_TOOL_PREFIX):
        return None
    offset = len(_PUBLIC_TOOL_PREFIX)
    run_token = tool_id[offset : offset + _PUBLIC_RUN_TOKEN_LENGTH]
    internal_id = tool_id[offset + _PUBLIC_RUN_TOKEN_LENGTH + 1 :]
    if tool_id[offset + _PUBLIC_RUN_TOKEN_LENGTH :][:1] != "_":
        return None
    if not _INTERNAL_TOOL_ID.fullmatch(internal_id):
        return None
    try:
        run_bytes = base64.urlsafe_b64decode(run_token + "==")
    except (ValueError, UnicodeError):
        return None
    if len(run_bytes) != 16:
        return None
    canonical = base64.urlsafe_b64encode(run_bytes).decode().rstrip("=")
    return (run_bytes.hex(), internal_id) if canonical == run_token else None


def _continuation(messages: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """Extract a Mantis continuation from trailing standard OpenAI tool messages."""
    if not isinstance(messages, list):
        return None
    trailing: list[dict[str, Any]] = []
    i = len(messages) - 1
    while i >= 0 and isinstance(messages[i], dict) and messages[i].get("role") == "tool":
        trailing.append(messages[i])
        i -= 1
    if not trailing or i < 0:
        return None
    assistant = messages[i]
    if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
        return None
    advertised = {
        call.get("id") for call in assistant.get("tool_calls", []) if isinstance(call, dict)
    }
    results: list[dict[str, Any]] = []
    run_id: str | None = None
    for message in reversed(trailing):
        public_id = message.get("tool_call_id")
        parsed = _parse_public_tool_id(public_id)
        if parsed is None or public_id not in advertised:
            raise ValueError("tool result does not belong to a Mantis tool call")
        current_run, internal_id = parsed
        if run_id is not None and current_run != run_id:
            raise ValueError("tool results span multiple Mantis runs")
        run_id = current_run
        results.append(
            {
                "tool_call_id": internal_id,
                "content": _message_text(message.get("content"))[:RUN_MAX_MSG_BYTES],
                "is_error": False,
            }
        )
    return (cast(str, run_id), results)


def _advance_to_boundary(run_id: str, tool_results: Any = None) -> dict[str, Any]:
    event = runs.advance_run(run_id, tool_results)
    for _ in range(128):
        if event.get("type") != "step_complete":
            return event
        event = runs.advance_run(run_id, None)
    return {"type": "error", "error": "orchestration exceeded 128 internal steps"}


def _run_trace(run: Any) -> dict[str, Any]:
    steps = list(getattr(run, "turns", getattr(run, "steps", [])))
    return {
        "mode": run.kind,
        "terminated_by": run.terminated_by,
        "steps": [
            {
                key: step.get(key)
                for key in ("turn", "role", "agent_id", "model_name")
                if step.get(key) is not None
            }
            for step in steps
        ],
    }


def _request_usage(messages: list[dict[str, Any]], completion_text: str) -> dict[str, int]:
    """Estimate per-request usage from the client's messages and this response.

    OpenAI clients (including the prime-agent harness) track their own context
    size from the response usage and compact when it approaches the window. The
    run's accumulated usage spans every internal orchestrator call and tool
    round, so reporting it would make the client see the context grow by the
    full orchestration cost each round and compact repeatedly. Report only the
    current request's context instead, using the same chars/4 estimate the
    client itself applies to messages without usage.
    """
    prompt_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        prompt_chars += len(_message_text(message.get("content")))
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            prompt_chars += len(str(function.get("name", ""))) + len(arguments)
    prompt_tokens = max(1, (prompt_chars + 3) // 4)
    completion_tokens = max(1, (len(completion_text) + 3) // 4)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _completion_response(
    model: str,
    messages: list[dict[str, Any]],
    run: Any,
    event: dict[str, Any],
    details: str = "none",
) -> dict[str, Any]:
    completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if event.get("type") == "tool_calls":
        message["tool_calls"] = [
            _openai_tool_call(
                str(call.get("name", "")),
                _public_tool_id(run.run_id, str(call.get("id", ""))),
                call.get("arguments") if isinstance(call.get("arguments"), dict) else {},
            )
            for call in event.get("tool_calls", [])
        ]
        finish_reason = "tool_calls"
        completion_text = "".join(call["function"]["arguments"] for call in message["tool_calls"])
    else:
        message["content"] = str(event.get("text", ""))
        message.update(getattr(run, "response_metadata", {}))
        finish_reason = "stop"
        completion_text = message["content"]
    usage: dict[str, Any] = _request_usage(messages, completion_text)
    cost = providers._usage_cost(getattr(run, "usage_models", {}))
    if cost is not None:
        usage["cost"] = cost
    body: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": usage,
    }
    if details != "none":
        body["mantis"] = providers._run_mantis_details(run, details)
    return body


def _parse_args() -> argparse.Namespace:
    if serve_config._args is not None:
        return serve_config._args
    ap = argparse.ArgumentParser(description="Serve Mantis as one OpenAI-compatible model.")
    ap.add_argument(
        "--model",
        default=os.environ.get("MANTIS_MODEL", "Qwen/Qwen3-0.6B"),
        help="Qwen3-0.6B dir or HF id",
    )
    ap.add_argument(
        "--vector",
        default=os.environ.get("MANTIS_VECTOR", "model_iter_60.npy"),
        help="base vector (19456) — SVF + head",
    )
    ap.add_argument(
        "--head",
        default=os.environ.get("MANTIS_HEAD"),
        help="optional trained head-only vector/safetensors; overrides the "
        "head from --vector after SVF is applied",
    )
    default_workers = os.environ.get("MANTIS_WORKER_MODELS") or os.environ.get(
        "MANTIS_WORKER_MODEL"
    )
    ap.add_argument(
        "--slot-models",
        metavar="CSV",
        default=default_workers,
        help="provider/model[|reasoning_effort] worker specs (CSV); also MANTIS_WORKER_MODELS",
    )
    ap.add_argument(
        "--local-models",
        metavar="CSV",
        default=os.environ.get("MANTIS_LOCAL_MODELS"),
        help="local HF worker model paths (CSV). "
        "Optional 'path@device' per entry; also MANTIS_LOCAL_MODELS",
    )
    ap.add_argument(
        "--host",
        default=os.environ.get("MANTIS_HOST", "0.0.0.0"),  # noqa: S104
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MANTIS_PORT", "8088")),
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=int(os.environ.get("MANTIS_MAX_TURNS", "5")),
    )
    serve_config._args, _ = (
        ap.parse_known_args()
    )  # ignore uvicorn's own argv (api:app --app-dir ...)
    return serve_config._args


def _convert_tools(tools: Any) -> list[dict[str, Any]]:
    """Validate, copy, and freeze OpenAI function tools in name order.

    Lexicographic order is locale-independent so a client reshuffle cannot
    bust the provider prompt-cache prefix.
    """
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict) or not function.get("name"):
            continue
        out.append({"type": "function", "function": dict(function)})
    out.sort(key=lambda tool: str(tool["function"]["name"]))
    return out


def system_reminder(text: str) -> str:
    """Wrap a harness notice as user-visible text that must not enter system."""
    return f"<system-reminder>\n{text}\n</system-reminder>"


def _model_completion(
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]]]:
    """Call the selected provider directly; return (text, tool_calls)."""
    data = providers._provider_response(model, messages, 4096, 0.7, tools)
    msg = data["choices"][0]["message"]
    text = str(msg.get("content") or "")
    tcs = msg.get("tool_calls") or []
    calls: list[dict[str, Any]] = []
    for tc in tcs:
        fn = tc.get("function", {})
        raw_arguments = fn.get("arguments")
        try:
            args = json.loads(raw_arguments) if raw_arguments else {}
        except (json.JSONDecodeError, ValueError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {}
        call = {"name": str(fn.get("name")), "arguments": args}
        provider_ids = msg.get("_anthropic_tool_ids")
        if isinstance(provider_ids, dict) and isinstance(tc.get("id"), str):
            call["_anthropic_tool_id"] = provider_ids.get(tc["id"], tc["id"])
        calls.append(call)
    metadata = {key: msg[key] for key in ("reasoning_details", "_anthropic_content") if key in msg}
    if calls and metadata:
        calls[0]["_assistant_metadata"] = metadata
    return text, calls


def _openai_tool_call(name: str, _id: str, arguments: dict) -> dict[str, Any]:
    return {
        "id": _id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def prune_tool_result(
    content: str,
    max_chars: int = 16000,
    head_chars: int = 8000,
    tail_chars: int = 2000,
) -> str:
    """Replay-safe head/tail pruning for large tool results.

    Follows DeepSeek Harness compaction-tool-result-pruner pattern: preserves
    the initial output context (head) and final status/summary (tail) while
    collapsing middle bytes with an explicit omitted-length marker.
    """
    if not isinstance(content, str) or len(content) <= max_chars:
        return content
    omitted = len(content) - head_chars - tail_chars
    if omitted <= 0:
        return content
    head = content[:head_chars]
    tail = content[-tail_chars:] if tail_chars > 0 else ""
    marker = f"\n\n[... Omitted {omitted} characters of tool output for context efficiency ...]\n\n"
    return head + marker + tail


class RepeatToolGuard:
    """Advisory loop-breaker for repetitive tool invocations.

    Inspired by DeepSeek Harness dsh-repeat-tool-reminder. Tracks consecutive
    identical tool calls using canonicalized JSON arguments and returns
    escalating non-blocking system reminders at specified thresholds.
    """

    def __init__(self, thresholds: tuple[int, ...] = (3, 5)) -> None:
        self.thresholds = thresholds
        self.last_tool: str | None = None
        self.last_canonical_args: str | None = None
        self.consecutive_count: int = 0

    @staticmethod
    def canonicalize_args(args: Any) -> str:
        if isinstance(args, str):
            try:
                parsed = json.loads(args)
                return json.dumps(parsed, sort_keys=True, separators=(",", ":"))
            except (json.JSONDecodeError, TypeError):
                return args.strip()
        if isinstance(args, dict):
            try:
                return json.dumps(args, sort_keys=True, separators=(",", ":"))
            except TypeError:
                return str(args)
        return str(args)

    def observe(self, tool_name: str, args: Any) -> str | None:
        """Record a tool invocation and return an advisory reminder if a threshold is hit."""
        canonical = self.canonicalize_args(args)
        if tool_name == self.last_tool and canonical == self.last_canonical_args:
            self.consecutive_count += 1
        else:
            self.last_tool = tool_name
            self.last_canonical_args = canonical
            self.consecutive_count = 1

        for thresh in self.thresholds:
            if self.consecutive_count == thresh:
                preview = canonical[:120]
                msg = (
                    f"[Advisory Notice: Tool '{tool_name}' has been called "
                    f"{self.consecutive_count} times consecutively with identical "
                    f"arguments ({preview}). If the tool result is unchanged or not making "
                    f"progress, adjust parameters, try an alternate approach, or conclude "
                    f"your response.]"
                )
                return msg
        return None

    def reset(self) -> None:
        self.last_tool = None
        self.last_canonical_args = None
        self.consecutive_count = 0


def _validate_tool_results(tool_results: Any, expected_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(tool_results, list):
        raise TypeError("tool_results must be a list")
    if not all(isinstance(result, dict) for result in tool_results):
        raise ValueError("each tool result must be an object")
    results = cast(list[dict[str, Any]], tool_results)
    ids = [result.get("tool_call_id") for result in results]
    if not all(isinstance(tool_id, str) and tool_id for tool_id in ids):
        raise ValueError("each tool result requires a tool_call_id")
    string_ids = cast(list[str], ids)
    if len(string_ids) != len(set(string_ids)) or set(string_ids) != expected_ids:
        raise ValueError(
            f"tool result id mismatch: expected {sorted(expected_ids)} got {sorted(string_ids)}"
        )
    return results


def _configured_slot_models(override: Any = None) -> list[str]:
    value = override
    if value is None:
        configured = (
            getattr(serve_config._args, "slot_models", None)
            if serve_config._args is not None
            else None
        )
        configured = (
            configured
            or os.environ.get("MANTIS_WORKER_MODELS")
            or os.environ.get("MANTIS_WORKER_MODEL")
        )
        value = configured.split(",") if configured else list(DEFAULT_SLOT_LABELS)
    if not isinstance(value, list):
        raise TypeError("slot_models must be a non-empty list of model names")
    models = [model.strip() for model in value if isinstance(model, str) and model.strip()]
    if len(models) != len(value) or not models:
        raise ValueError("slot_models must be a non-empty list of model names")
    return models


# Capability bundles map to common tool-name prefixes or tags.
_BUNDLE_TOOLS: dict[str, set[str]] = {
    "files": {"list_files", "read_file", "search_files", "write_file", "edit_file"},
    "shell": {"bash", "shell", "sh", "execute_command"},
    "code": {"execute_code", "ipython", "python", "exec", "run_code"},
    "browser": {"search_web", "fetch_web_page", "browser_agent", "computer_use"},
}


def _filter_tools_by_options(tools: list[dict[str, Any]], options: Any) -> list[dict[str, Any]]:
    """Return the subset of tools allowed by the given tool options."""
    if not options:
        return list(tools or [])
    enabled = getattr(options, "enabled", None)
    if not enabled:
        return list(tools or [])
    allowed_names: set[str] = set()
    for capability in enabled:
        if capability in _BUNDLE_TOOLS:
            allowed_names.update(_BUNDLE_TOOLS[capability])
        else:
            allowed_names.add(str(capability))
    return [t for t in (tools or []) if str(t.get("function", {}).get("name", "")) in allowed_names]


def _load_skill_profile(name: str) -> Any | None:
    """Load a Mantis skill as a Fusion worker profile, if present."""
    for root in (
        Path.home() / ".agents" / "skills",
        Path.home() / ".devin" / "skills",
        Path(__file__).resolve().parent.parent.parent / ".agents" / "skills",
    ):
        path = root / name / "SKILL.md"
        if not path.is_file():
            continue
        try:
            text = path.read_text()
            lines = text.splitlines()
            title = lines[0].lstrip("# ").strip() if lines else name
            description = ""
            instructions = ""
            for i, line in enumerate(lines[1:], start=2):
                if line.lower().startswith("## description"):
                    description = "\n".join(
                        ln.strip() for ln in lines[i:] if ln.strip() and not ln.startswith("##")
                    ).split("\n\n")[0]
                elif line.lower().startswith("## instructions") or line.lower().startswith(
                    "## prompt"
                ):
                    instructions = "\n".join(
                        ln.strip() for ln in lines[i:] if ln.strip() and not ln.startswith("##")
                    ).split("\n\n")[0]
            from fusion_types import FusionWorkerProfile

            return FusionWorkerProfile(
                name=name,
                description=description or title,
                instructions=instructions or text[:4000],
            )
        except OSError:
            continue
    return None


__all__ = [k for k in globals() if not k.startswith("__")]
