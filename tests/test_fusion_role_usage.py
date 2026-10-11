from __future__ import annotations

import fusion
import pytest
import serve_config


@pytest.mark.parametrize("role", ["main", "sidekick"])
def test_active_request_records_role_usage_without_doubling_totals(monkeypatch, role):
    run = fusion.FusionRun(f"role-usage-{role}", "goal")
    coordinator = fusion.FusionCoordinator()
    usage = {"prompt_tokens": 37, "completion_tokens": 8, "total_tokens": 45}
    usage["prompt_tokens_details"] = {"cached_tokens": 12}

    def provider_worker(slot, _messages, _tools):
        assert serve_config._history_context.active_run is run
        run.add_usage(usage, model=slot)
        return {"content": "ok", "_mantis_model": "wire-model"}, usage

    def advance(_results, _request_id, _message, active_coordinator):
        run._call_lane(active_coordinator, role, slot_override="test-slot")
        return {"status": "main_review"}

    monkeypatch.setattr(fusion, "get_run", lambda _run_id: run)
    monkeypatch.setattr(fusion, "_put_run", lambda _run: None)
    monkeypatch.setattr(fusion, "FusionCoordinator", lambda: coordinator)
    monkeypatch.setattr(coordinator, "_call_worker", provider_worker)
    monkeypatch.setattr(run, "advance_idempotent", advance)

    fusion.advance_fusion_run(run.run_id, request_id="request")

    assert run.usage == usage
    assert run.role_usage[role]["test-slot"] == {
        "prompt_tokens": 37, "completion_tokens": 8, "cached_tokens": 12,
    }
    assert serve_config._history_context.active_run is None


@pytest.mark.parametrize("role", ["main", "sidekick"])
def test_execution_lane_records_active_request_role_usage_once(monkeypatch, role):
    run = fusion.FusionRun(f"execution-usage-{role}", "goal")
    coordinator = fusion.FusionCoordinator()
    assignment = fusion.FusionSidekickAssignment(task="work")
    profile = fusion.FusionWorkerProfile(name="worker")
    lane = fusion.ExecutionLane("lane", run, coordinator, assignment, profile, role=role)
    usage = {"prompt_tokens": 37, "completion_tokens": 8, "total_tokens": 45}

    def provider_worker(slot, _messages, _tools):
        assert serve_config._history_context.active_run is run
        run.add_usage(usage, model=slot)
        return {"content": "done", "_mantis_model": "wire-model"}, usage

    monkeypatch.setattr(serve_config._history_context, "active_run", run, raising=False)
    monkeypatch.setattr(coordinator, "_call_worker", provider_worker)
    monkeypatch.setattr(lane, "_select_slot", lambda: "test-slot")

    result = lane.step()

    assert result["status"] == "completed"
    assert run.usage == usage
    assert run.role_usage[role]["test-slot"] == {
        "prompt_tokens": 37, "completion_tokens": 8, "cached_tokens": 0,
    }
