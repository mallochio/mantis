"""Wire-protocol adapters for hosted model providers."""

from __future__ import annotations

import json
from typing import Any


def uses_responses_api(provider: str, model: str) -> bool:
    """Use OpenRouter's native Responses API for OpenAI-family models."""
    return provider == "openrouter" and model.startswith("openai/")


def _text_part(part: dict[str, Any], assistant: bool) -> dict[str, Any] | None:
    kind = part.get("type")
    if kind in {"text", "input_text", "output_text"}:
        return {
            "type": "output_text" if assistant else "input_text",
            "text": str(part.get("text", "")),
        }
    if kind != "image_url" or assistant:
        return None
    image = part.get("image_url")
    if isinstance(image, str):
        return {"type": "input_image", "image_url": image}
    if isinstance(image, dict) and image.get("url"):
        output = {"type": "input_image", "image_url": image["url"]}
        if image.get("detail"):
            output["detail"] = image["detail"]
        return output
    return None


def _message_content(content: Any, assistant: bool) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [
            item
            for part in content
            if isinstance(part, dict)
            if (item := _text_part(part, assistant))
        ]
    if content is None:
        return []
    return [{"type": "output_text" if assistant else "input_text", "text": str(content)}]


def chat_messages_to_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate stateless Chat Completions history to Responses input items."""
    items: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        role = message.get("role")
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id", "")),
                    "output": str(message.get("content") or ""),
                }
            )
            continue
        assistant = role == "assistant"
        if assistant:
            items.extend(
                detail
                for detail in message.get("reasoning_details") or []
                if isinstance(detail, dict) and detail.get("type") == "reasoning"
            )
        content = _message_content(message.get("content"), assistant)
        if content or not assistant:
            item = {"type": "message", "role": role, "content": content}
            if assistant:
                item.update({"id": f"msg_m{message_index:08x}", "status": "completed"})
            items.append(item)
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            call_id = str(call.get("id", ""))
            arguments = function.get("arguments", "")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            items.append(
                {
                    "type": "function_call",
                    "id": f"fc_{call_id}",
                    "call_id": call_id,
                    "name": str(function.get("name", "")),
                    "arguments": arguments,
                }
            )
    return items


def _responses_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for tool in tools or []:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict):
            continue
        output.append({"type": "function", **function})
    return output


def _responses_tool_choice(choice: Any) -> Any:
    if not isinstance(choice, dict):
        return choice
    function = choice.get("function")
    if choice.get("type") == "function" and isinstance(function, dict):
        return {"type": "function", "name": function.get("name")}
    return choice


def _responses_text_format(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    if response_format.get("type") != "json_schema":
        return dict(response_format)
    schema = response_format.get("json_schema") or {}
    return {"type": "json_schema", **schema}


def build_responses_body(
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float | None,
    effort: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    response_format: dict[str, Any] | None,
    controls: dict[str, Any],
) -> dict[str, Any]:
    """Build one stateless OpenRouter Responses API request."""
    body: dict[str, Any] = {
        "model": model,
        "input": chat_messages_to_input(messages),
        "max_output_tokens": controls.get("max_tokens", max_tokens),
    }
    if temperature is not None:
        body["temperature"] = temperature
    function_tools = _responses_tools(tools)
    web_options = controls.get("web_search_options")
    if isinstance(web_options, dict):
        function_tools.append({"type": "openrouter:web_search", "parameters": web_options})
    if function_tools:
        body["tools"] = function_tools
    if tool_choice is not None:
        body["tool_choice"] = _responses_tool_choice(tool_choice)
    text_format = _responses_text_format(response_format)
    if text_format:
        body["text"] = {"format": text_format}
    reasoning = controls.get("reasoning")
    reasoning_effort = controls.get("reasoning_effort", effort)
    if isinstance(reasoning, dict):
        body["reasoning"] = reasoning
    elif reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _normalized_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    output: dict[str, Any] = {
        "prompt_tokens": usage.get("input_tokens", 0),
        "completion_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }
    if isinstance(usage.get("input_tokens_details"), dict):
        output["prompt_tokens_details"] = usage["input_tokens_details"]
    if isinstance(usage.get("output_tokens_details"), dict):
        output["completion_tokens_details"] = usage["output_tokens_details"]
    return output


def responses_to_chat(response: dict[str, Any]) -> dict[str, Any]:
    """Convert a completed Responses object to Mantis's canonical Chat shape."""
    if response.get("status") == "failed":
        error = response.get("error") or "unknown error"
        detail = error.get("message") if isinstance(error, dict) else error
        raise RuntimeError(f"provider response failed: {detail}")
    message: dict[str, Any] = {"role": "assistant", "content": None}
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    reasoning: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for item in response.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            calls.append(
                {
                    "id": item.get("call_id") or item.get("id"),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", ""),
                    },
                }
            )
        elif item.get("type") == "reasoning":
            reasoning.append(item)
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "refusal"}:
                    text.append(str(part.get("text") or part.get("refusal") or ""))
                annotations.extend(a for a in part.get("annotations") or [] if isinstance(a, dict))
    if text:
        message["content"] = "".join(text)
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_details"] = reasoning
        summaries = [
            str(part.get("text", ""))
            for item in reasoning
            for part in item.get("summary") or []
            if isinstance(part, dict) and part.get("text")
        ]
        if summaries:
            message["reasoning"] = "".join(summaries)
    if annotations:
        message["annotations"] = annotations
    return {"choices": [{"message": message}], "usage": _normalized_usage(response.get("usage"))}


def assemble_responses_stream(events: Any) -> dict[str, Any]:
    """Reassemble Responses SSE events and return canonical Chat data."""
    items: dict[int, dict[str, Any]] = {}
    text: list[str] = []
    usage: Any = None
    for event in events:
        if not event:
            continue
        event_type = event.get("type")
        if event.get("error") or event_type in {"error", "response.failed"}:
            error = event.get("error") or event.get("response", {}).get("error") or event
            detail = error.get("message") if isinstance(error, dict) else error
            raise RuntimeError(f"provider stream error: {detail}")
        if event_type in {"response.done", "response.completed"} and isinstance(
            event.get("response"), dict
        ):
            return responses_to_chat(event["response"])
        index = event.get("output_index")
        item = event.get("item")
        if (
            isinstance(index, int)
            and isinstance(item, dict)
            and event_type
            in {
                "response.output_item.added",
                "response.output_item.done",
            }
        ):
            items[index] = item
        if event_type in {"response.output_text.delta", "response.content_part.delta"}:
            text.append(str(event.get("delta", "")))
        if event_type == "response.function_call_arguments.delta" and isinstance(index, int):
            current = items.setdefault(index, {"type": "function_call"})
            current["arguments"] = str(current.get("arguments", "")) + str(event.get("delta", ""))
        if event_type == "response.function_call_arguments.done" and isinstance(index, int):
            items.setdefault(index, {"type": "function_call"})["arguments"] = event.get(
                "arguments", ""
            )
        if isinstance(event.get("response"), dict):
            usage = event["response"].get("usage") or usage
    output = list(dict(sorted(items.items())).values())
    if text and not any(item.get("type") == "message" for item in output):
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "".join(text)}],
            }
        )
    return responses_to_chat({"output": output, "usage": usage})
