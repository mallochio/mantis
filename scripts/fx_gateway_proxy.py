"""Loopback AI Gateway shim so official fx can call Mantis Base.

fx 0.0.6 speaks the Vercel AI Gateway LanguageModelV2 protocol, not OpenAI
chat completions. This process accepts that protocol on loopback and forwards
an OpenAI chat body to the local Mantis API (which proxies Switchyard).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Literal
from urllib.parse import urlsplit

MODEL_ID = "mantis/base"
DEFAULT_MANTIS_URL = "http://127.0.0.1:8088/v1/chat/completions"
DEFAULT_BIND = "127.0.0.1"
DEFAULT_PORT = 8787
SESSION_HEADER = "x-switchyard-session-id"


class ProxyError(RuntimeError):
    """Raised when the AI SDK body cannot be translated or Mantis fails."""


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _part_text(part: Mapping[str, Any]) -> str:
    part_type = part.get("type")
    if part_type in (None, "text"):
        text = part.get("text")
        if isinstance(text, str):
            return text
    output = part.get("output")
    if isinstance(output, str):
        return output
    result = part.get("result")
    if result is not None:
        return _json_text(result)
    content = part.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            _part_text(item) if isinstance(item, dict) else _json_text(item) for item in content
        )
    return ""


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                chunks.append(_part_text(item))
        return "".join(chunks)
    return _json_text(content)


def _tool_calls_from_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool-call":
            continue
        call_id = item.get("toolCallId") or item.get("id")
        name = item.get("toolName") or item.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        arguments = item.get("input", item.get("args", {}))
        calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    return calls


def _tool_result_messages(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if not isinstance(content, list):
        tool_call_id = message.get("toolCallId") or message.get("tool_call_id")
        if isinstance(tool_call_id, str):
            return [
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": _content_text(content) or _content_text(message.get("result")),
                }
            ]
        return []
    results: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "tool-result":
            continue
        call_id = item.get("toolCallId") or item.get("id")
        if not isinstance(call_id, str):
            continue
        results.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _part_text(item) or "",
            }
        )
    return results


def openai_messages(prompt: Any) -> list[dict[str, Any]]:
    """Translate an AI SDK LanguageModelV2 prompt into OpenAI chat messages."""
    if not isinstance(prompt, list):
        raise ProxyError("gateway prompt must be an array")
    messages: list[dict[str, Any]] = []
    for raw in prompt:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        content = raw.get("content")
        match role:
            case "system" | "developer":
                text = _content_text(content)
                if text:
                    messages.append({"role": "system", "content": text})
            case "user":
                tool_results = _tool_result_messages(raw)
                if tool_results:
                    messages.extend(tool_results)
                    leftover = ""
                    if isinstance(content, list):
                        leftover = "".join(
                            _part_text(item)
                            if isinstance(item, dict) and item.get("type") != "tool-result"
                            else (item if isinstance(item, str) else "")
                            for item in content
                        )
                    if leftover.strip():
                        messages.append({"role": "user", "content": leftover})
                else:
                    messages.append({"role": "user", "content": _content_text(content)})
            case "assistant":
                message: dict[str, Any] = {"role": "assistant"}
                text = _content_text(
                    [
                        item
                        for item in content
                        if not (isinstance(item, dict) and item.get("type") == "tool-call")
                    ]
                    if isinstance(content, list)
                    else content
                )
                if text:
                    message["content"] = text
                tool_calls = _tool_calls_from_content(content)
                if not tool_calls and isinstance(raw.get("tool_calls"), list):
                    tool_calls = [
                        call for call in raw["tool_calls"] if isinstance(call, dict)
                    ]
                if tool_calls:
                    message["tool_calls"] = tool_calls
                    message.setdefault("content", None)
                if "content" in message or "tool_calls" in message:
                    messages.append(message)
            case "tool":
                results = _tool_result_messages(raw)
                if results:
                    messages.extend(results)
                else:
                    call_id = raw.get("toolCallId") or raw.get("tool_call_id")
                    if isinstance(call_id, str):
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": _content_text(content),
                            }
                        )
            case _:
                unknown = role if isinstance(role, str) else type(role).__name__
                raise ProxyError(f"unsupported prompt role: {unknown}")
    if not messages:
        raise ProxyError("gateway prompt produced no OpenAI messages")
    return messages


def openai_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list) or not tools:
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = tool.get("inputSchema") or tool.get("parameters") or {"type": "object"}
        converted.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": parameters,
                },
            }
        )
    return converted or None


def openai_chat_body(payload: Mapping[str, Any], model: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": openai_messages(payload.get("prompt")),
        "stream": False,
    }
    tools = openai_tools(payload.get("tools"))
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    max_tokens = payload.get("maxOutputTokens") or payload.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        body["max_tokens"] = max_tokens
    return body


def _usage_event(usage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    cached = None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        cached = details.get("cached_tokens")
    event: dict[str, Any] = {
        "inputTokens": {"total": int(prompt or 0)},
        "outputTokens": {"total": int(completion or 0)},
    }
    if isinstance(cached, int) and cached > 0:
        event["inputTokens"]["cacheRead"] = cached
    return event


def sse_events_from_chat(response: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProxyError("mantis response missing choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProxyError("mantis response missing message")
    events: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        events.append({"type": "text-delta", "id": "answer_1", "delta": content})
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            call_id = call.get("id")
            if not isinstance(name, str) or not isinstance(call_id, str):
                continue
            arguments = function.get("arguments") or "{}"
            parsed: Any = arguments
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed = arguments
            events.append(
                {
                    "type": "tool-call",
                    "toolCallId": call_id,
                    "toolName": name,
                    "input": parsed,
                }
            )
    finish_reason = choices[0].get("finish_reason") or "stop"
    unified: Literal["stop", "tool-calls"]
    match finish_reason:
        case "tool_calls" | "tool-calls":
            unified = "tool-calls"
        case _:
            unified = "stop"
    finish: dict[str, Any] = {
        "type": "finish",
        "finishReason": {"unified": unified, "raw": finish_reason},
    }
    usage = _usage_event(response.get("usage") if isinstance(response.get("usage"), dict) else None)
    if usage is not None:
        finish["usage"] = usage
    events.append(finish)
    return events


def encode_sse(events: list[dict[str, Any]]) -> bytes:
    chunks = [f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events]
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def models_payload(model: str) -> dict[str, Any]:
    return {"data": [{"id": model, "type": "language", "tags": ["tool-use"]}]}


def post_json(
    url: str,
    body: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> tuple[dict[str, Any], Mapping[str, str]]:
    request = urllib.request.Request(  # noqa: S310
        _require_loopback_http(url),
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
            return payload, {key.lower(): value for key, value in response.headers.items()}
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ProxyError(f"mantis HTTP {error.code}: {detail}") from error


def _loopback_http(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def _require_loopback_http(url: str) -> str:
    if not _loopback_http(url):
        raise ProxyError("mantis URL must be loopback http")
    return url


def _gateway_object(raw: bytes) -> dict[str, Any]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ProxyError("gateway body must be a JSON object")
    return payload


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "fx-mantis-gateway/1.0"

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def _write(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path in {"/health", "/ready"}:
            self._write(200, b'{"ok":true}', "application/json")
            return
        if path == "/coding-agent/v1/models":
            payload = models_payload(self.server.model_id)
            self._write(200, json.dumps(payload).encode("utf-8"), "application/json")
            return
        self._write(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path not in {"/v3/ai/language-model", "/chat"}:
            self._write(404, b'{"error":"not found"}', "application/json")
            return
        length = int(self.headers.get("content-length") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = _gateway_object(raw)
            body = openai_chat_body(payload, self.server.model_id)
            headers = {
                "authorization": f"Bearer {self.server.api_key}",
                SESSION_HEADER: self.server.session_id,
            }
            response, upstream_headers = post_json(
                self.server.mantis_url, body, headers, self.server.timeout_s
            )
            selected = (
                upstream_headers.get("x-route-model")
                or upstream_headers.get("x-model-router-selected-model")
                or ""
            )
            record = {
                "selected_model": selected,
                "prompt_tokens": (response.get("usage") or {}).get("prompt_tokens")
                if isinstance(response.get("usage"), dict)
                else None,
                "completion_tokens": (response.get("usage") or {}).get("completion_tokens")
                if isinstance(response.get("usage"), dict)
                else None,
                "cached_tokens": (
                    (response.get("usage") or {}).get("prompt_tokens_details") or {}
                ).get("cached_tokens")
                if isinstance(response.get("usage"), dict)
                else None,
                "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason")
                if isinstance(response.get("choices"), list) and response["choices"]
                else None,
                "tool_calls": len(
                    ((response.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
                    or []
                )
                if isinstance(response.get("choices"), list) and response["choices"]
                else 0,
            }
            self.server.hop_log.append(record)
            if self.server.hop_log_path:
                with open(self.server.hop_log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            sse = encode_sse(sse_events_from_chat(response))
        except (ProxyError, json.JSONDecodeError, KeyError, TypeError) as error:
            self._write(
                502,
                json.dumps({"error": {"message": str(error)}}).encode("utf-8"),
                "application/json",
            )
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("x-route-model", selected)
        self.send_header("content-length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


class GatewayServer(ThreadingHTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        mantis_url: str,
        api_key: str,
        model_id: str,
        session_id: str,
        timeout_s: float,
        hop_log_path: str | None,
    ) -> None:
        if not _loopback_http(f"http://{host}:{port}/"):
            raise ProxyError("fx gateway proxy must bind loopback")
        super().__init__((host, port), GatewayHandler)
        self.mantis_url = mantis_url
        self.api_key = api_key
        self.model_id = model_id
        self.session_id = session_id
        self.timeout_s = timeout_s
        self.hop_log: list[dict[str, Any]] = []
        self.hop_log_path = hop_log_path


def serve() -> None:
    host = os.environ.get("FX_GATEWAY_BIND", DEFAULT_BIND)
    port = int(os.environ.get("FX_GATEWAY_PORT", str(DEFAULT_PORT)))
    api_key = os.environ.get("MANTIS_API_KEY")
    if not api_key:
        raise SystemExit("MANTIS_API_KEY is required")
    server = GatewayServer(
        host,
        port,
        mantis_url=os.environ.get("MANTIS_CHAT_URL", DEFAULT_MANTIS_URL),
        api_key=api_key,
        model_id=os.environ.get("FX_MODEL", MODEL_ID),
        session_id=os.environ.get("MANTIS_FX_SESSION_ID", "fx-base-smoke"),
        timeout_s=float(os.environ.get("MANTIS_ROUTER_TIMEOUT_S", "300")),
        hop_log_path=os.environ.get("FX_HOP_LOG"),
    )
    print(f"fx gateway proxy on http://{host}:{port}", flush=True)
    print(f"chat {server.mantis_url} model={server.model_id}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    # Allow importing conversion helpers without starting the server.
    threading.current_thread().name = "fx-gateway-proxy"
    serve()
