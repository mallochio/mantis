"""Native Anthropic Messages conversion for signed-thinking providers."""

from __future__ import annotations

import json
from typing import Any

_VERSION = "2023-06-01"


def _text(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"type": "text", "text": str(content or "")}]
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            blocks.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image_url":
            image = part.get("image_url") or {}
            url = image.get("url") if isinstance(image, dict) else None
            if isinstance(url, str) and url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _tool_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        raw = function.get("arguments", "{}")
        try:
            input_ = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            input_ = {}
        blocks.append(
            {
                "type": "tool_use",
                "id": str(call.get("id", "")),
                "name": str(function.get("name", "")),
                "input": input_ if isinstance(input_, dict) else {},
            }
        )
    return blocks


def _replayable_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    return [
        block
        for block in content
        if isinstance(block, dict)
        and (
            block.get("type") != "thinking"
            or (isinstance(block.get("thinking"), str) and bool(block["thinking"].strip()))
        )
    ]


def chat_to_anthropic(
    messages: list[dict[str, Any]],
) -> tuple[str | list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Convert canonical Mantis history while preserving native assistant blocks."""
    system: list[dict[str, Any]] = []
    converted: list[dict[str, Any]] = []
    tool_ids: dict[str, str] = {}
    for message in messages:
        role = message.get("role")
        if role in {"system", "developer"}:
            system.extend(_text(message.get("content")))
            continue
        if role == "assistant":
            raw = message.get("_anthropic_content")
            content = (
                _replayable_content(raw)
                if isinstance(raw, list)
                else _text(message.get("content")) + _tool_blocks(message)
            )
            mapping = message.get("_anthropic_tool_ids")
            if isinstance(mapping, dict):
                tool_ids.update({str(key): str(value) for key, value in mapping.items()})
            converted.append({"role": "assistant", "content": content})
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id", ""))
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_ids.get(call_id, call_id),
                            "content": str(message.get("content", "")),
                            "is_error": bool(message.get("is_error", False)),
                        }
                    ],
                }
            )
            continue
        if role == "user":
            converted.append({"role": "user", "content": _text(message.get("content"))})
    return (system or None), converted


def _tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        converted.append(
            {
                "name": str(function.get("name", "")),
                "description": str(function.get("description", "")),
                "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


def build_anthropic_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    effort: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
) -> dict[str, Any]:
    system, history = chat_to_anthropic(messages)
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "messages": history}
    if system:
        body["system"] = system
    if effort and effort != "none":
        # Bifrost maps an enabled thinking request to the selected Bedrock model.
        # Budget by effort level so low/medium do not always pay for the full
        # 16k reasoning allowance.
        budgets = {
            "minimal": 1024,
            "low": 2048,
            "medium": 8192,
            "high": 16384,
            "xhigh": 16384,
            "max": 16384,
        }
        budget = budgets.get(effort, 8192)
        body["thinking"] = {
            "type": "enabled",
            "budget_tokens": min(budget, max_tokens - 1, 16384),
        }
    native_tools = _tools(tools)
    if native_tools:
        body["tools"] = native_tools
    if tool_choice == "auto":
        body["tool_choice"] = {"type": "auto"}
    elif tool_choice == "required":
        body["tool_choice"] = {"type": "any"}
    elif tool_choice == "none":
        body["tool_choice"] = {"type": "none"}
    elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        function = tool_choice.get("function") or {}
        body["tool_choice"] = {"type": "tool", "name": function.get("name")}
    return body


def _mark_anthropic_block(block: dict[str, Any]) -> None:
    if "cache_control" not in block:
        block["cache_control"] = {"type": "ephemeral"}


def apply_anthropic_prompt_cache(body: dict[str, Any]) -> dict[str, Any]:
    """Mark system, tools, and the history prefix for Anthropic prompt cache.

    Breakpoints follow Anthropic's 4-breakpoint budget: last system block, last
    tool, and the last content block of the message before the latest turn.
    The input dict is copied; nested lists/dicts that are marked are copied.
    """
    out = dict(body)
    system = out.get("system")
    if isinstance(system, list) and system:
        blocks = [dict(block) if isinstance(block, dict) else block for block in system]
        if isinstance(blocks[-1], dict):
            _mark_anthropic_block(blocks[-1])
        out["system"] = blocks
    tools = out.get("tools")
    if isinstance(tools, list) and tools and isinstance(tools[-1], dict):
        marked_tools = [dict(tool) if isinstance(tool, dict) else tool for tool in tools]
        if isinstance(marked_tools[-1], dict):
            _mark_anthropic_block(marked_tools[-1])
        out["tools"] = marked_tools
    history = out.get("messages")
    if isinstance(history, list) and len(history) >= 2:
        messages = [dict(message) for message in history]
        prior = messages[-2]
        content = prior.get("content")
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            blocks = [dict(block) if isinstance(block, dict) else block for block in content]
            if isinstance(blocks[-1], dict):
                _mark_anthropic_block(blocks[-1])
            prior["content"] = blocks
            messages[-2] = prior
            out["messages"] = messages
    return out


def anthropic_to_chat(response: dict[str, Any]) -> dict[str, Any]:
    content = response.get("content") or []
    text = "".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )
    calls = []
    tool_ids: dict[str, str] = {}
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            raw_id = str(block.get("id", ""))
            calls.append(
                {
                    "id": raw_id,
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
            tool_ids[raw_id] = raw_id
    message: dict[str, Any] = {
        "role": "assistant",
        "content": text,
        "_anthropic_content": _replayable_content(content),
    }
    if calls:
        message["tool_calls"] = calls
        message["_anthropic_tool_ids"] = tool_ids
    usage = response.get("usage") or {}
    prompt_tokens = usage.get("input_tokens", 0)
    completion_tokens = usage.get("output_tokens", 0)
    # Anthropic reports cache reads as cache_read_input_tokens; surface it in the
    # OpenAI-compatible prompt_tokens_details.cached_tokens field so Mantis cost
    # accounting and cache-hit metrics work for Claude on Vertex / Bedrock.
    cached = usage.get("cache_read_input_tokens")
    usage_out: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if isinstance(cached, int):
        usage_out["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"choices": [{"message": message}], "usage": usage_out}


def assemble_anthropic_stream(events: Any) -> dict[str, Any]:
    """Collect Anthropic SSE events into a canonical, replay-safe message."""
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    for event in events:
        if not event:
            continue
        if event.get("type") == "error":
            error = event.get("error") or {}
            raise RuntimeError(f"provider stream error: {error.get('message', error)}")
        if event.get("type") == "content_block_start":
            index = event.get("index")
            block = event.get("content_block")
            if isinstance(index, int) and isinstance(block, dict):
                blocks[index] = dict(block)
        elif event.get("type") == "content_block_delta":
            block = blocks.get(event.get("index"))
            delta = event.get("delta") or {}
            if block is not None and delta.get("type") in {"text_delta", "thinking_delta"}:
                key = "text" if delta.get("type") == "text_delta" else "thinking"
                block[key] = str(block.get(key, "")) + str(delta.get(key, ""))
            elif block is not None and delta.get("type") == "input_json_delta":
                block["_partial_json"] = str(block.get("_partial_json", "")) + str(
                    delta.get("partial_json", "")
                )
        elif event.get("type") == "message_delta":
            usage = event.get("usage") or usage
    content = list(dict(sorted(blocks.items())).values())
    for block in content:
        raw = block.pop("_partial_json", "")
        if raw:
            try:
                block["input"] = json.loads(raw)
            except json.JSONDecodeError:
                block["input"] = {}
    return anthropic_to_chat({"content": content, "usage": usage})


def anthropic_headers(key: str) -> dict[str, str]:
    return {
        "x-api-key": key,
        "anthropic-version": _VERSION,
        "Content-Type": "application/json",
        "User-Agent": "Mantis",
    }
