import json
from pathlib import Path

import numpy as np
import pytest
import torch

from apps.api import mini

FIXTURE = Path(__file__).parent / "fixtures/router_default.json"


def test_shipped_router_fixture_pins_head_only_shape():
    config = json.loads(FIXTURE.read_text())
    vector = np.zeros(config["vector_length"])
    vector[config["svf_length"] :] = 1
    assert config["mode"] == "head-only"
    assert np.all(vector[: config["svf_length"]] == 0)
    assert vector[config["svf_length"] :].reshape(config["head_shape"]).shape == (10, 1024)


def test_svf_consumption_and_stub_decision():
    router = mini.FuguRouter.__new__(mini.FuguRouter)
    router.torch = torch
    router.model = type(
        "Model",
        (),
        {
            "state_dict": lambda self: {
                f"projection_{index}": torch.ones(1024, 1024) for index in range(9)
            }
        },
    )()
    router._apply_svf(np.zeros(mini.SVF_LEN))
    assert len(router.svf_keys) == 9
    router.head = torch.ones(mini.HEAD_ROWS, mini.HIDDEN)
    router._hidden = lambda messages: torch.ones(mini.HIDDEN)
    decision = router.route([{"role": "user", "content": "hello"}], sample=False)
    assert decision["agent_id"] == 0
    assert decision["role_name"] == "Worker"


def test_svf_rejects_wrong_consumption():
    router = mini.FuguRouter.__new__(mini.FuguRouter)
    router.torch = torch
    router.model = type(
        "Model", (), {"state_dict": lambda self: {"projection": torch.ones(8, 8)}}
    )()
    with pytest.raises(RuntimeError, match="expected 9216"):
        router._apply_svf(np.zeros(mini.SVF_LEN))


def test_mini_coordination_plumbing_without_weights():
    class StubRouter:
        def route(self, messages, sample=False, agent_mask=None):
            return {"agent_id": 0, "role_id": 0, "role_name": "Worker"}

    result = mini.Coordinator(StubRouter(), mini.MockWorker(), sample=False).run("solve it")
    assert result.final and result.terminated_by in {"max_turns", "verifier_accept"}
    assert mini.FuguRouter.format_transcript([{"role": "user", "content": "x"}]) == "user: x"
    assert mini.Coordinator._extract_thought("<think>idea</think>") == "idea"
    assert mini.Coordinator(StubRouter(), mini.MockWorker())._parse_verification("ACCEPT: yes")


def test_ultra_offline_executor_and_validation():
    from apps.api import ultra

    assert ultra.self_test() == 0
    executor = ultra.ConductorExecutor(ultra.MockWorker())
    with pytest.raises(ValueError):
        executor.validate([0], [], [[]])
    with pytest.raises(ValueError, match="out-of-range"):
        executor.validate([7], ["task"], [[]])
    with pytest.raises(TypeError, match="non-integer"):
        executor.validate(["0"], ["task"], [[]])


def test_parse_workflow_accepts_json():
    from apps.api import ultra

    payload = json.dumps(
        {
            "model_id": [0, 1],
            "subtasks": ["plan", "implement"],
            "access_list": [[], [0]],
        }
    )
    assert ultra.parse_workflow(payload) == ([0, 1], ["plan", "implement"], [[], [0]])
    assert ultra.parse_workflow(f"```json\n{payload}\n```") == (
        [0, 1],
        ["plan", "implement"],
        [[], [0]],
    )


def test_planner_messages_prefill_and_repair():
    from apps.api import ultra
    from apps.api import runs as serve

    run = serve.ConductorRun(
        "planner",
        [{"role": "user", "content": "task"}],
        [],
        slot_models=["a", "b"],
    )
    first = run._planner_messages()
    assert first[-1] == {"role": "assistant", "content": ultra.PLANNER_PREFILL}
    repair = run._planner_messages(repair=("bad", "missing lists"))
    assert repair[2]["role"] == "assistant"
    assert "missing lists" in repair[-1]["content"]


def test_planner_repair_retries_once(monkeypatch):
    from apps.api import runs as serve

    run = serve.ConductorRun(
        "retry",
        [{"role": "user", "content": "task"}],
        [],
        slot_models=["worker"],
    )
    calls = {"n": 0}

    def fake_run_model(role, model, messages):
        calls["n"] += 1
        if calls["n"] == 1:
            run._last_planner_text = "not a workflow"
            return {"type": "error", "error": "Conductor emitted an invalid workflow: missing lists"}
        run._workflow = ([0], ["task"], [[]])
        return {"type": "step_complete", "role": "Planner", "turn": 0}

    monkeypatch.setattr(run, "_run_model", fake_run_model)
    event = run._run_planner()
    assert calls["n"] == 2
    assert event["type"] == "step_complete"


def test_planner_keeps_query_and_pool_out_of_system():
    from apps.api import ultra

    first = ultra.conductor_prompt("SECRET_QUERY", ["luna-slot", "sol-slot"])
    assert first[0]["role"] == "system"
    assert first[0]["content"] == ultra.PLANNER_SYSTEM
    assert "SECRET_QUERY" not in first[0]["content"]
    assert "luna-slot" not in first[0]["content"]
    assert "SECRET_QUERY" in first[1]["content"]
    assert "luna-slot" in first[1]["content"]
    repair = ultra.planner_repair_messages(
        "SECRET_QUERY", ["luna-slot", "sol-slot"], "bad", "missing lists"
    )
    assert repair[0] == first[0]
    assert repair[1] == first[1]
    assert repair[2]["role"] == "assistant"
