from __future__ import annotations

import json

import api
import fusion
import fusion_stream
import serve_config


def encode_id(run_id, call_id):
    return f"{run_id}:{call_id}"


def test_fragmented_client_arguments_are_all_emitted():
    encoder = fusion_stream.DeltaEncoder()
    fragments = [
        {"index": 0, "id": "c1", "function": {"name": "bash", "arguments": "{"}},
        {"index": 0, "function": {"arguments": '"command":"ls"}'}},
    ]
    emitted = [
        item
        for fragment in fragments
        for item in encoder.encode("run", {"tool_calls": [fragment]}, encode_id)
    ]
    calls = [item["tool_calls"][0] for item in emitted]
    assert "".join(call["function"].get("arguments", "") for call in calls) == '{"command":"ls"}'
    assert calls[0]["id"] == "run:lead:c1"
    assert calls[1]["index"] == calls[0]["index"]


def test_fragmented_hidden_names_and_arguments_stay_private():
    encoder = fusion_stream.DeltaEncoder()
    fragments = [
        {"index": 0, "id": "d1", "function": {"name": "side"}},
        {"index": 0, "function": {"name": "kick", "arguments": '{"message":'}},
        {"index": 0, "function": {"arguments": '"private brief"}'}},
    ]
    emitted = [
        item
        for fragment in fragments
        for item in encoder.encode("run", {"tool_calls": [fragment]}, encode_id)
    ]
    emitted.extend(encoder.encode("run", {"_mantis_turn_end": True}, encode_id))
    assert emitted == []
    assert encoder.emitted_call_ids == set()


def test_client_name_prefixing_hidden_tool_is_flushed_at_turn_end():
    encoder = fusion_stream.DeltaEncoder()
    fragment = {"index": 0, "id": "c1", "function": {"name": "side", "arguments": "{}"}}
    assert encoder.encode("run", {"tool_calls": [fragment]}, encode_id) == []
    emitted = encoder.encode("run", {"_mantis_turn_end": True}, encode_id)
    assert emitted[0]["tool_calls"][0]["function"] == {"name": "side", "arguments": "{}"}


def provider_reply(name, call_id, arguments):
    first_call = {
        "index": 0,
        "id": call_id,
        "function": {"name": name, "arguments": arguments[:1]},
    }
    next_call = {"index": 0, "function": {"arguments": arguments[1:]}}
    return [
        {"choices": [{"delta": {"tool_calls": [first_call]}}]},
        {
            "choices": [{
                "delta": {"tool_calls": [next_call]},
                "finish_reason": "tool_calls",
            }]
        },
    ]

class Capacity:
    def release(self):
        return None


def _install_handoff_stream(monkeypatch):
    run = fusion.FusionRun(
        "stream-handoff",
        "inspect",
        tools=[
            {"type": "function", "function": {"name": "bash"}},
        ],
        tool_options=fusion.FusionToolOptions(delegation="tool"),
    )
    turns = [
        provider_reply("sidekick", "delegate", '{"message":"private brief"}'),
        provider_reply("bash", "client", '{"command":"ls"}'),
    ]

    def worker(self, slot, messages, tools, on_delta=None):
        if on_delta is None:
            return {"role": "assistant", "content": "private report"}, {}
        result = fusion_stream.consume_provider_stream(turns.pop(0), on_delta, lambda: None)
        return result["choices"][0]["message"], result["usage"]

    monkeypatch.setattr(fusion.FusionCoordinator, "_call_worker", worker)
    monkeypatch.setattr(api, "_run_fusion_chat", lambda *_args, **_kwargs: run.advance())
    monkeypatch.setattr(api, "_capacity", Capacity())
    return run


def test_live_stream_resets_provider_indexes_after_internal_handoff(monkeypatch):
    run = _install_handoff_stream(monkeypatch)
    request = api.ChatRequest(
        model="mantis/fusion", stream=True, messages=[{"role": "user", "content": "inspect"}]
    )
    raw = b"".join(api._iter_fusion_chat_stream(request, "req", {})).decode()
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    calls = [
        call
        for chunk in chunks
        for choice in chunk["choices"]
        for call in choice["delta"].get("tool_calls", [])
    ]
    assert "private brief" not in raw and "private report" not in raw
    assert calls[0]["function"]["name"] == "bash"
    assert api._decode_fusion_tool_call_id(calls[0]["id"]) == (run.run_id, "lead:client")
    assert "".join(call["function"].get("arguments", "") for call in calls) == '{"command":"ls"}'
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    assert all("_mantis_turn" not in json.dumps(chunk) for chunk in chunks)


def test_response_boundaries_preserve_public_indexes_and_ids():
    encoder = fusion_stream.DeltaEncoder()
    seen = []
    for call_id in ("first", "second"):
        chunks = provider_reply("bash", call_id, "{}")
        fusion_stream.consume_provider_stream(
            chunks, lambda delta: seen.extend(encoder.encode("run", delta, encode_id)), lambda: None
        )
    calls = [item["tool_calls"][0] for item in seen]
    assert [call["index"] for call in calls] == [0, 0, 1, 1]
    assert encoder.emitted_call_ids == {"lead:first", "lead:second"}
    assert len(encoder.calls) == 2


def test_visibility_is_bound_separately_without_consuming_preview_twice():
    encoder = fusion_stream.DeltaEncoder()
    stream = fusion_stream.FusionStream()

    def visibility(delta):
        return bool(encoder.encode("", delta, encode_id))

    hidden = {
        "tool_calls": [
            {
                "index": 0,
                "id": "d1",
                "function": {"name": "sidekick", "arguments": '{"message":"private"}'},
            }
        ]
    }
    with fusion_stream.bind_stream(stream, visibility):
        visible = serve_config._history_context.fusion_stream_is_visible
        assert visible({"_mantis_turn_start": True}) is False
        assert visible(hidden) is False
        assert visible({"content": "public"}) is True
        stream.emit_delta("run", hidden)
        stream.emit_delta("run", {"content": "public"})
    assert stream.events.qsize() == 2
    assert not hasattr(serve_config._history_context, "fusion_stream_is_visible")
    stream.cancelled.set()
    stream.emit_delta("run", {"content": "disconnected"})
    assert stream.events.qsize() == 2


def test_live_stream_keeps_nonstream_final_after_prior_commentary(monkeypatch):
    def compute(*_args, **_kwargs):
        callback = serve_config._history_context.fusion_stream_callback
        callback("run", {"_mantis_turn_start": True})
        callback("run", {"content": "I will inspect. "})
        callback("run", {"_mantis_turn_end": True})
        callback("run", {"_mantis_turn_start": True})
        callback("run", {"_mantis_turn_end": True})
        return {"status": "completed", "run_id": "run", "report": "Final answer."}

    monkeypatch.setattr(api, "_run_fusion_chat", compute)
    monkeypatch.setattr(api, "_capacity", Capacity())
    request = api.ChatRequest(
        model="mantis/fusion", stream=True, messages=[{"role": "user", "content": "inspect"}]
    )
    raw = b"".join(api._iter_fusion_chat_stream(request, "req", {})).decode()
    chunks = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
    content = "".join(
        choice["delta"].get("content", "") for chunk in chunks for choice in chunk["choices"]
    )
    assert content == "I will inspect. Final answer."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "_mantis_turn" not in raw
