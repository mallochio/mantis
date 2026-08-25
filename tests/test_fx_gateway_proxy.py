"""AI Gateway → OpenAI translation used to drive fx against Mantis Base."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fx_gateway_proxy import (
    GatewayServer,
    ProxyError,
    encode_sse,
    models_payload,
    openai_chat_body,
    openai_messages,
    openai_tools,
    post_json,
    serve,
    sse_events_from_chat,
)


def test_prompt_and_tools_become_openai_chat_body():
    body = openai_chat_body(
        {
            "prompt": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": [{"type": "text", "text": "list files"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool-call",
                            "toolCallId": "call_1",
                            "toolName": "bash",
                            "input": {"command": "ls"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "call_1",
                            "result": "a.py\n",
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "bash",
                    "description": "run a command",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                }
            ],
            "maxOutputTokens": 2048,
        },
        "mantis/base",
    )
    assert body["model"] == "mantis/base"
    assert body["stream"] is False
    assert body["max_tokens"] == 2048
    assert openai_chat_body({"prompt": [{"role": "user", "content": "hi"}]}, "mantis/base")[
        "max_tokens"
    ] == 4096
    assert body["messages"][0] == {"role": "system", "content": "You are a coding agent."}
    assert body["messages"][1] == {"role": "user", "content": "list files"}
    assert body["messages"][2]["tool_calls"][0]["function"]["name"] == "bash"
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "a.py\n"}
    assert body["tools"][0]["function"]["name"] == "bash"


def test_user_tool_results_flatten_into_tool_messages():
    messages = openai_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "tool-result", "toolCallId": "t1", "result": "boom"},
                    {"type": "text", "text": "try again"},
                ],
            }
        ]
    )
    assert messages[0] == {"role": "tool", "tool_call_id": "t1", "content": "boom"}
    assert messages[1] == {"role": "user", "content": "try again"}


def test_structured_failed_command_is_visible_to_stage_router():
    messages = openai_messages(
        [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "term_1",
                        "toolName": "terminal",
                        "output": {
                            "type": "json",
                            "value": {
                                "exit_code": 1,
                                "stdout": "",
                                "stderr": "",
                                "stdout_bytes": 0,
                            },
                        },
                    }
                ],
            }
        ]
    )
    content = messages[0]["content"]
    assert "exit_code" in content
    assert "AssertionError" in content


def test_pytest_assertion_output_is_not_double_tagged():
    messages = openai_messages(
        [
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool-result",
                        "toolCallId": "term_2",
                        "result": "AssertionError: lists differ\n",
                    }
                ],
            }
        ]
    )
    assert messages[0]["content"].count("AssertionError") == 1


def test_sse_falls_back_to_reasoning_when_content_is_empty():
    events = sse_events_from_chat(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": None, "reasoning": "pong"},
                }
            ]
        }
    )
    assert events[0] == {"type": "text-delta", "id": "answer_1", "delta": "pong"}


def test_sse_emits_text_then_tool_call():
    events = sse_events_from_chat(
        {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": "I'll inspect the repo.",
                        "tool_calls": [
                            {
                                "id": "call_9",
                                "function": {
                                    "name": "read_file",
                                    "arguments": json.dumps({"path": "a.py"}),
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 11,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 3},
            },
        }
    )
    assert events[0]["type"] == "text-delta"
    assert events[1]["toolName"] == "read_file"
    assert events[1]["input"] == {"path": "a.py"}
    assert events[2]["finishReason"]["unified"] == "tool-calls"
    assert events[2]["usage"]["inputTokens"]["cacheRead"] == 3
    encoded = encode_sse(events).decode("utf-8")
    assert encoded.endswith("data: [DONE]\n\n")


def test_openai_tools_skips_nameless_entries():
    assert openai_tools([{"description": "nope"}]) is None
    assert openai_tools(None) is None


def test_empty_prompt_and_malformed_upstream_are_rejected():
    with pytest.raises(ProxyError, match="no OpenAI messages"):
        openai_messages([])
    with pytest.raises(ProxyError, match="missing choices"):
        sse_events_from_chat({"choices": []})
    with pytest.raises(ProxyError, match="bind loopback"):
        GatewayServer(
            "example.test",
            80,
            mantis_url="http://127.0.0.1:8088/v1/chat/completions",
            api_key="k",
            model_id="mantis/base",
            session_id="s",
            timeout_s=1,
            hop_log_path=None,
        )


def test_tool_fallbacks_and_unparsed_arguments():
    messages = openai_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "c2",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "toolCallId": "c2", "content": "out"},
        ]
    )
    assert messages[0]["tool_calls"][0]["id"] == "c2"
    assert messages[1]["tool_call_id"] == "c2"
    events = sse_events_from_chat(
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "tool_calls": [
                            {"id": "x", "function": {"name": "bash", "arguments": "not-json"}},
                            "skip-me",
                        ]
                    },
                }
            ]
        }
    )
    assert events[0]["input"] == "not-json"
    assert openai_tools(["nope", {"name": "ok"}])[0]["function"]["name"] == "ok"


def test_serve_requires_api_key(monkeypatch):
    monkeypatch.delenv("MANTIS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="MANTIS_API_KEY"):
        serve()


def test_post_json_reads_success_and_error_bodies():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("content-length") or "0")
            self.rfile.read(length)
            if self.path == "/ok":
                body = b'{"ok":true}'
                self.send_response(200)
            else:
                body = b'{"error":"nope"}'
                self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("x-route-model", "stealth/ox-alpha")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    try:
        port = upstream.server_address[1]
        payload, headers = post_json(
            f"http://127.0.0.1:{port}/ok",
            {"hi": 1},
            {"authorization": "Bearer k"},
            5,
        )
        assert payload == {"ok": True}
        assert headers["x-route-model"] == "stealth/ox-alpha"
        with pytest.raises(ProxyError, match="mantis HTTP 502"):
            post_json(f"http://127.0.0.1:{port}/bad", {}, {}, 5)
        with pytest.raises(ProxyError, match="loopback"):
            post_json("https://example.test/v1", {}, {}, 5)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_loopback_http_server_lists_models_and_proxies(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_post(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout
        return (
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
            {"x-route-model": "openrouter/free"},
        )

    monkeypatch.setattr("fx_gateway_proxy.post_json", fake_post)
    log_path = tmp_path / "hops.jsonl"
    server = GatewayServer(
        "127.0.0.1",
        0,
        mantis_url="http://127.0.0.1:8088/v1/chat/completions",
        api_key="test-key",
        model_id="mantis/base",
        session_id="sess-1",
        timeout_s=5,
        hop_log_path=str(log_path),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        assert json.loads(conn.getresponse().read().decode()) == {"ok": True}
        conn.request("GET", "/coding-agent/v1/models")
        listed = json.loads(conn.getresponse().read().decode())
        assert listed == models_payload("mantis/base")
        conn.request("GET", "/missing")
        assert conn.getresponse().status == 404
        conn.request("POST", "/missing", body=b"{}", headers={"content-type": "application/json"})
        assert conn.getresponse().status == 404
        conn.request(
            "POST",
            "/v3/ai/language-model",
            body=b"not-json",
            headers={"content-type": "application/json"},
        )
        assert conn.getresponse().status == 502
        conn.request(
            "POST",
            "/v3/ai/language-model",
            body=json.dumps({"prompt": [{"role": "user", "content": "hi"}]}),
            headers={"content-type": "application/json"},
        )
        response = conn.getresponse()
        assert response.status == 200
        assert response.getheader("x-route-model") == "openrouter/free"
        body = response.read().decode()
        assert "ok" in body
        assert captured["body"]["messages"][0]["content"] == "hi"
        assert captured["headers"]["x-switchyard-session-id"] == "sess-1"
        hop = json.loads(log_path.read_text().splitlines()[0])
        assert hop["selected_model"] == "openrouter/free"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
