"""End-to-end contracts for harness-neutral mantis/base tool loops."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import api
import base_proxy
import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from switchyard_config import load_switchyard_route, render_switchyard_toml

TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}
TOOL_CALL = {
    "id": "call_readme",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
}


@pytest.fixture
def routed_client(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    trajectories: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        trajectories.append(payload)
        if payload["messages"][-1]["role"] == "tool":
            message = {"role": "assistant", "content": "README inspected"}
            finish_reason = "stop"
        else:
            message = {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(trajectories)}",
                "object": "chat.completion",
                "created": 1,
                "model": "mantis/base",
                "choices": [
                    {"index": 0, "message": message, "finish_reason": finish_reason}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: "provider/routed"},
        )

    monkeypatch.setattr(
        api,
        "_router_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return TestClient(api.app), trajectories


def _raw_tool_loop(client: TestClient) -> None:
    headers = {"Authorization": "Bearer test-key"}
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect README"}]
    first = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "mantis/base", "messages": messages, "tools": [TOOL]},
    )
    assert first.status_code == 200
    call = first.json()["choices"][0]["message"]["tool_calls"][0]
    assert call == TOOL_CALL
    messages.extend(
        [
            first.json()["choices"][0]["message"],
            {"role": "tool", "tool_call_id": call["id"], "content": "fake README"},
        ]
    )
    final = client.post(
        "/v1/chat/completions",
        headers=headers,
        json={"model": "mantis/base", "messages": messages, "tools": [TOOL]},
    )
    assert final.json()["choices"][0]["message"]["content"] == "README inspected"


def _sdk_tool_loop(client: TestClient) -> None:
    sdk = OpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect README"}]
    first = sdk.chat.completions.create(model="mantis/base", messages=messages, tools=[TOOL])
    call = first.choices[0].message.tool_calls[0]
    assert call.id == TOOL_CALL["id"]
    messages.extend(
        [
            first.choices[0].message.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": call.id, "content": "fake README"},
        ]
    )
    final = sdk.chat.completions.create(model="mantis/base", messages=messages, tools=[TOOL])
    assert final.choices[0].message.content == "README inspected"


def test_raw_http_harness_executes_its_own_tool(routed_client):
    client, trajectories = routed_client
    _raw_tool_loop(client)
    assert [turn["messages"][-1]["role"] for turn in trajectories] == ["user", "tool"]
    assert trajectories[1]["messages"][-1]["content"] == "fake README"


def test_openai_sdk_harness_has_equivalent_tool_trajectory(routed_client):
    client, trajectories = routed_client
    _raw_tool_loop(client)
    raw = list(trajectories)
    trajectories.clear()
    _sdk_tool_loop(client)
    sdk = list(trajectories)

    def normalized(turns):
        return [{"messages": turn["messages"], "tools": turn["tools"]} for turn in turns]

    assert normalized(sdk) == normalized(raw)


def test_openai_sdk_stream_reconstructs_tool_call_and_continues(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")
    trajectories: list[dict[str, Any]] = []

    def sse(data: dict[str, Any]) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        trajectories.append(payload)
        if payload["messages"][-1]["role"] == "tool":
            chunks = [
                sse(
                    {
                        "id": "chatcmpl-2",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "mantis/base",
                        "choices": [{"index": 0, "delta": {"content": "README inspected"}}],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        else:
            chunks = [
                sse(
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "mantis/base",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_readme",
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "arguments": '{"path":',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ),
                sse(
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "mantis/base",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {"index": 0, "function": {"arguments": '"README.md"}'}}
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                    }
                ),
                b"data: [DONE]\n\n",
            ]
        return httpx.Response(
            200,
            content=b"".join(chunks),
            headers={"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        api,
        "_router_client",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client = TestClient(api.app)
    sdk = OpenAI(api_key="test-key", base_url="http://testserver/v1", http_client=client)
    messages: list[dict[str, Any]] = [{"role": "user", "content": "inspect README"}]
    stream = sdk.chat.completions.create(
        model="mantis/base", messages=messages, tools=[TOOL], stream=True
    )
    tool_id = ""
    tool_name = ""
    arguments = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.tool_calls:
            tool = delta.tool_calls[0]
            tool_id += tool.id or ""
            tool_name += tool.function.name or ""
            arguments += tool.function.arguments or ""
    messages.extend(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": arguments},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": tool_id, "content": "fake README"},
        ]
    )
    final = sdk.chat.completions.create(
        model="mantis/base", messages=messages, tools=[TOOL], stream=True
    )

    assert tool_id == "call_readme"
    assert json.loads(arguments) == {"path": "README.md"}
    assert "".join(chunk.choices[0].delta.content or "" for chunk in final) == "README inspected"
    assert trajectories[1]["messages"][-2]["tool_calls"][0]["id"] == "call_readme"
    assert trajectories[1]["messages"][-1]["tool_call_id"] == "call_readme"


def test_catalog_model_swap_changes_route_without_changing_client_request(tmp_path, monkeypatch):
    source = Path("config/catalog.toml").read_text()
    first_path = tmp_path / "first.toml"
    second_path = tmp_path / "second.toml"
    first_path.write_text(source)
    second_path.write_text(source.replace("zai-org/GLM-5.3", "vendor/replacement-efficient", 1))

    def efficient_model(path: Path) -> str:
        rendered = tomllib.loads(render_switchyard_toml(load_switchyard_route(path)))
        return str(rendered["targets"]["efficient"]["id"])

    selected = [efficient_model(first_path), efficient_model(second_path)]
    request_body = {
        "model": "mantis/base",
        "messages": [{"role": "user", "content": "same request"}],
        "tools": [TOOL],
    }
    seen: list[dict[str, Any]] = []
    headers: list[str] = []
    client = TestClient(api.app)
    monkeypatch.setenv("MANTIS_API_KEY", "test-key")

    for model in selected:
        def handler(request: httpx.Request, selected_model: str = model) -> httpx.Response:
            seen.append(json.loads(request.content.decode()))
            return httpx.Response(
                200,
                json={"id": "c", "choices": [{"message": {"content": "ok"}}]},
                headers={base_proxy.SWITCHYARD_SELECTED_MODEL_HEADER: selected_model},
            )

        monkeypatch.setattr(
            api,
            "_router_client",
            lambda: httpx.Client(transport=httpx.MockTransport(handler)),
        )
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json=request_body,
        )
        assert response.status_code == 200
        headers.append(response.headers["x-route-model"])

    assert seen[0] == seen[1]
    assert headers == selected
    assert headers[0] != headers[1]
