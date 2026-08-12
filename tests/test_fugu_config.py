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
