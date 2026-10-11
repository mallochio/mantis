from __future__ import annotations

import fusion
import fusion_lead
import fusion_stream
import providers
import pytest
import serve_config


@pytest.fixture
def worker(monkeypatch):
    resolved = providers.ResolvedModelSpec(
        "mock", "wire-model", None, "https://local.invalid", "MISSING_KEY", None, None, "slot"
    )
    monkeypatch.setattr(providers, "_runtime_bindings", lambda: None)
    monkeypatch.setattr(providers, "_failover_attempts", lambda *_args: ["slot", "backup"])
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda *_args: resolved)
    monkeypatch.setattr(providers, "_litellm_kwargs", lambda *_args: {})
    monkeypatch.setattr(providers, "_FAILOVER_DELAY", 0)
    return resolved


def _failed_stream(delta):
    if delta is not None:
        yield {"choices": [{"delta": delta}]}
    raise providers.APIConnectionError(message="lost stream", llm_provider="mock", model="wire")


def _completion(monkeypatch, delta):
    attempts = []

    def complete(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            return _failed_stream(delta)
        return [{"choices": [{"delta": {"content": "recovered"}, "finish_reason": "stop"}]}]

    monkeypatch.setattr(providers, "_litellm_completion", complete)
    return attempts


@pytest.mark.parametrize("delta", [None, {}, {"reasoning": "not requested"}, {
    "tool_calls": [{"index": 0, "id": "d1", "function": {"name": "sidekick"}}]
}])
def test_failure_before_public_output_allows_failover(monkeypatch, worker, delta):
    attempts = _completion(monkeypatch, delta)
    emitted = []

    monkeypatch.setattr(
        serve_config._history_context, "fusion_stream_is_visible",
        lambda _delta: False, raising=False,
    )
    response = providers._provider_response("slot", [], 100, 0.7, on_delta=emitted.append)

    assert len(attempts) == 2
    assert response["choices"][0]["message"]["content"] == "recovered"


@pytest.mark.parametrize("delta", [{"content": "partial"}, {"reasoning": "requested"}, {
    "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "bash"}}]
}])
def test_failure_after_public_output_disables_failover(monkeypatch, worker, delta):
    attempts = _completion(monkeypatch, delta)
    emitted = []

    monkeypatch.setattr(
        serve_config._history_context, "fusion_stream_is_visible",
        lambda fragment: not fragment.get("_mantis_turn_start", False), raising=False,
    )
    with pytest.raises(RuntimeError, match="client-visible output; retry disabled"):
        providers._provider_response("slot", [], 100, 0.7, on_delta=emitted.append)

    assert len(attempts) == 1
    assert len([fragment for fragment in emitted if not fragment.get("_mantis_turn_start")]) == 1


def test_none_return_callback_still_guards_public_text(monkeypatch, worker):
    attempts = _completion(monkeypatch, {"content": "partial"})
    emitted = []

    with pytest.raises(RuntimeError, match="client-visible output; retry disabled"):
        providers._provider_response("slot", [], 100, 0.7, on_delta=emitted.append)

    assert len(attempts) == 1
    assert emitted[-1] == {"content": "partial"}


def test_routed_nonstream_lead_emits_turn_boundaries(monkeypatch):
    run = fusion.FusionRun("routed-boundary", "goal")
    run.handoffs = fusion_lead.HandoffManager(None)
    coordinator = fusion.FusionCoordinator()
    emitted = []
    monkeypatch.setattr(
        serve_config._history_context, "fusion_stream_callback",
        lambda _run_id, delta: emitted.append(delta), raising=False,
    )
    monkeypatch.setattr(
        coordinator, "_call_worker", lambda *_args, **_kwargs: ({"content": "final"}, {})
    )

    text, calls, usage = run._call_lane(coordinator, "main", slot_override="mantis/base")

    assert text == "final"
    assert emitted == [{"_mantis_turn_start": True}, {"_mantis_turn_end": True}]


def test_visibility_predicate_runs_once_before_queue_callback(monkeypatch, worker):
    timeline = []
    attempts = _completion(monkeypatch, {"content": "partial"})

    def visible(delta):
        timeline.append(("preview", delta))
        return bool(delta.get("content"))

    monkeypatch.setattr(
        serve_config._history_context, "fusion_stream_is_visible", visible, raising=False
    )
    with pytest.raises(RuntimeError, match="client-visible output; retry disabled"):
        providers._provider_response(
            "slot", [], 100, 0.7, on_delta=lambda delta: timeline.append(("queue", delta))
        )

    assert len(attempts) == 1
    assert [kind for kind, _delta in timeline] == ["preview", "queue", "preview", "queue"]
    assert timeline[-2][1] == timeline[-1][1] == {"content": "partial"}


def test_end_boundary_flush_is_classified_as_public_output(monkeypatch, worker):
    attempts = []
    emitted = []
    encoder = fusion_stream.DeltaEncoder()

    def complete(**kwargs):
        attempts.append(kwargs)
        return [{"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": "c1", "function": {"name": "side"}}
        ]}, "finish_reason": "tool_calls"}]}]

    def queue(delta):
        emitted.append(delta)
        if delta.get("_mantis_turn_end"):
            raise providers.APIConnectionError(
                message="lost queue", llm_provider="mock", model="wire"
            )

    monkeypatch.setattr(providers, "_litellm_completion", complete)
    monkeypatch.setattr(
        serve_config._history_context, "fusion_stream_is_visible",
        lambda delta: bool(encoder.encode("run", delta, lambda _run, call: call)), raising=False,
    )
    with pytest.raises(RuntimeError, match="client-visible output; retry disabled"):
        providers._provider_response("slot", [], 100, 0.7, on_delta=queue)

    assert len(attempts) == 1
    assert emitted[-1] == {"_mantis_turn_end": True}
