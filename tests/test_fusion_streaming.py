from __future__ import annotations

import json

import api
import fusion
import fusion_harness
import fusion_lead
import fusion_stream
import providers
import serve_config


def _provider_chunks():
    first_delta = {
        "content": "answer",
        "reasoning": "thinking",
        "tool_calls": [
            {
                "index": 0,
                "id": "call-1",
                "function": {"name": "bash", "arguments": "{"},
            }
        ],
    }
    final_delta = {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}
    return [
        {"choices": [{"delta": first_delta, "finish_reason": None}]},
        {
            "choices": [{"delta": final_delta, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
        },
    ]


def test_provider_stream_assembly_emits_content_reasoning_and_tools():
    seen = []
    result = fusion_stream.consume_provider_stream(_provider_chunks(), seen.append, lambda: None)
    message = result["choices"][0]["message"]

    assert message["content"] == "answer"
    assert message["reasoning"] == "thinking"
    assert message["tool_calls"][0]["function"]["arguments"] == "{}"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2}
    seen = [delta for delta in seen if not any(key.startswith("_mantis_turn") for key in delta)]
    assert seen[0]["content"] == "answer"
    assert seen[0]["reasoning"] == seen[0]["reasoning_content"] == "thinking"
    public = api._encode_fusion_deltas(
        fusion_stream.DeltaEncoder(), "run", seen[0], {"x-mantis-return-reasoning": "true"}
    )
    assert public[0]["reasoning"] == public[0]["reasoning_content"] == "thinking"


def test_portable_reasoning_details_emit_text_but_encrypted_details_stay_hidden():
    seen = []
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_details": [
                            {"index": 0, "type": "reasoning.text", "text": "visible"},
                            {"index": 1, "type": "reasoning.encrypted", "text": "opaque"},
                        ]
                    }
                }
            ]
        }
    ]

    fusion_stream.consume_provider_stream(chunks, seen.append, lambda: None)

    seen = [delta for delta in seen if not any(key.startswith("_mantis_turn") for key in delta)]
    assert seen[0]["reasoning"] == seen[0]["reasoning_content"] == "visible"


def test_provider_response_uses_streaming_request_when_callback_is_present(monkeypatch):
    resolved = providers.ResolvedModelSpec(
        adapter="mock-adapter",
        model="wire-model",
        effort=None,
        base_url="https://local.invalid",
        credential_env="MISSING_KEY",
        binding=None,
        protocols=None,
        slot="mock-slot",
    )
    captured = {}
    emitted = []
    monkeypatch.setattr(providers, "_runtime_bindings", lambda: None)
    monkeypatch.setattr(providers, "_failover_attempts", lambda _spec, _bindings: ["mock-slot"])
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda _spec, _bindings: resolved)
    monkeypatch.setattr(providers, "_litellm_kwargs", lambda *_args: {})
    def completion(**kwargs):
        captured.update(kwargs)
        return [{"choices": [{"delta": {"content": "live"}, "finish_reason": "stop"}]}]

    monkeypatch.setattr(providers, "_litellm_completion", completion)
    result = providers._provider_response("mock-slot", [], 100, 0.7, on_delta=emitted.append)
    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert result["choices"][0]["message"]["content"] == "live"
    assert emitted == [
        {"_mantis_turn_start": True}, {"content": "live"}, {"_mantis_turn_end": True}
    ]


def test_reasoning_details_are_replayed_only_for_the_producing_model():
    details = [{"type": "encrypted", "id": "opaque-state", "data": "unchanged"}]
    stored = {
        "role": "assistant",
        "content": "",
        "reasoning_details": details,
        "_mantis_model": "wire-model",
        "_mantis_preserve_reasoning": True,
    }

    same_model = providers._sanitize_messages([stored], "wire-model")
    other_model = providers._sanitize_messages([stored], "other-model")

    assert same_model[0]["reasoning_details"] == details
    assert "_mantis_preserve_reasoning" not in same_model[0]
    assert "reasoning_details" not in other_model[0]
    assert "_mantis_preserve_reasoning" not in other_model[0]


def test_tool_lead_replays_reasoning_details_on_its_next_turn(monkeypatch):
    details = [{"type": "encrypted", "id": "opaque-state", "data": "unchanged"}]
    replies = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": details,
            "_mantis_model": "wire-model",
            "tool_calls": [{"id": "call-1"}],
        },
        {"role": "assistant", "content": "done", "_mantis_model": "wire-model"},
    ]
    run = fusion.FusionRun(
        "reasoning-continuity", "goal", tool_options=fusion.FusionToolOptions(delegation="tool")
    )
    run.handoffs = fusion_lead.HandoffManager(None)
    coordinator = fusion.FusionCoordinator()
    monkeypatch.setattr(coordinator, "_call_worker", lambda *_args, **_kwargs: (replies.pop(0), {}))

    run._call_lane(coordinator, "main")
    run.main_messages.append({"role": "tool", "tool_call_id": "call-1", "content": "ok"})
    run._call_lane(coordinator, "main")
    replay = providers._sanitize_messages(run.main_messages, "wire-model")[-3]

    assert replay["reasoning_details"] == details
    assert "_mantis_preserve_reasoning" not in replay


def test_disconnect_stops_future_provider_deltas():
    stream = fusion_stream.FusionStream()
    with fusion_stream.bind_stream(stream):
        stream.emit_delta("run", {"content": "before"})
        stream.cancelled.set()
        stream.emit_delta("run", {"content": "after"})
        connected = serve_config._history_context.is_client_connected()

    assert connected is False
    assert stream.events.get_nowait() == ("delta", ("run", {"content": "before"}))
    assert stream.events.empty()


def test_delta_encoder_hides_delegation_calls_and_namespaces_client_ids():
    encoder = fusion_stream.DeltaEncoder()
    encode_id = api._encode_fusion_tool_call_id

    hidden = encoder.encode(
        "run",
        {"tool_calls": [{"index": 0, "id": "d1", "function": {"name": "sidekick"}}]},
        encode_id,
    )
    visible = encoder.encode(
        "run",
        {
            "tool_calls": [
                {
                    "index": 1,
                    "id": "c1",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ]
        },
        encode_id,
    )

    assert hidden == []
    encoded_id = visible[0]["tool_calls"][0]["id"]
    assert api._decode_fusion_tool_call_id(encoded_id) == ("run", "lead:c1")


def test_effective_delegation_name_filter_preserves_client_sidekick_tool():
    schema = {"type": "function", "function": {"name": "sidekick"}}
    names = fusion_harness.effective_delegation_names([schema])
    encoder = fusion_stream.DeltaEncoder(names.values())
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "work"}],
        tools=[schema],
        stream=True,
        tool_options={"delegation": "tool"},
    )
    assert api._fusion_hidden_tool_names(request, {}) == set(names.values())
    internal = {
        "index": 0,
        "id": "delegate-1",
        "function": {"name": names["sidekick"], "arguments": "{}"},
    }
    client_tool = {
        "index": 1,
        "id": "client-1",
        "function": {"name": "sidekick", "arguments": "{}"},
    }
    hidden = encoder.encode("run", {"tool_calls": [internal]}, api._encode_fusion_tool_call_id)
    client = encoder.encode("run", {"tool_calls": [client_tool]}, api._encode_fusion_tool_call_id)
    call = client[0]["tool_calls"][0]

    assert hidden == []
    assert call["function"]["name"] == "sidekick"
    assert api._decode_fusion_tool_call_id(call["id"]) == ("run", "lead:client-1")


def test_final_stream_chunk_adds_unstreamed_client_tool_calls():
    encoder = fusion_stream.DeltaEncoder()
    encoder.encode(
        "run",
        {
            "tool_calls": [
                {"index": 0, "id": "lead-1", "function": {"name": "bash", "arguments": "{}"}}
            ]
        },
        api._encode_fusion_tool_call_id,
    )
    event = {
        "run_id": "run",
        "pending_tool_calls": [
            {"id": "lead:lead-1", "function": {"name": "bash", "arguments": "{}"}},
            {"id": "sidekick:side-1", "function": {"name": "read_file", "arguments": "{}"}},
        ],
    }

    missing = api._missing_fusion_tool_calls(event, "run", encoder)

    assert len(missing) == 1
    assert api._decode_fusion_tool_call_id(missing[0]["id"]) == ("run", "sidekick:side-1")
    assert missing[0]["function"]["name"] == "read_file"


def test_comment_and_empty_delta_keepalives_are_valid_sse():
    comment = api._fusion_stream_keepalive()
    encoded = api._fusion_stream_chunk({"id": "request"}, {}, None)
    payload = json.loads(encoded.decode().removeprefix("data: ").strip())

    assert comment == b": keep-alive\n\n"
    assert payload["choices"][0]["delta"] == {}


def _live_response(_request, _request_id, **_kwargs):
    callback = serve_config._history_context.fusion_stream_callback
    callback("run", {"content": "hello"})
    return {
        "status": "completed",
        "run_id": "run",
        "report": "hello",
        "mantis_event": {"trace_scope": "main"},
    }


def test_live_fusion_stream_emits_deltas_and_final_event(monkeypatch):
    request = api.ChatRequest(
        model="mantis/fusion",
        messages=[{"role": "user", "content": "say hello"}],
        stream=True,
    )
    monkeypatch.setattr(api, "_capacity", _NoopCapacity())

    monkeypatch.setattr(api, "_run_fusion_chat", _live_response)
    chunks = list(api._iter_fusion_chat_stream(request, "request", {}))
    payloads = [
        json.loads(line[6:])
        for chunk in chunks
        for line in chunk.decode().splitlines()
        if line.startswith("data: {")
    ]

    assert any(item["choices"][0]["delta"].get("content") == "hello" for item in payloads)
    final = next(item for item in payloads if item["choices"][0]["finish_reason"] == "stop")
    assert final["mantis_event"] == {"trace_scope": "main"}
    assert chunks[-1] == b"data: [DONE]\n\n"


class _NoopCapacity:
    def release(self):
        return None
