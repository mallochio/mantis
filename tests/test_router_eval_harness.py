"""Hermetic tests for the graded router evaluation harness."""

import importlib.util
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


harness = _load("router_eval", ROOT / "eval" / "router_eval.py")
metrics = _load("route_metrics", ROOT / "eval" / "route_metrics.py")
miniswe = _load("miniswe_config", ROOT / "eval" / "miniswe_config.py")


def _instance() -> dict:
    return {
        "instance_id": "demo-1",
        "problem_statement": "implement a parser",
        "docker_image": "demo:latest",
        "test_cmd": "pytest tests/test_demo.py",
        "FAIL_TO_PASS": ["tests/test_demo.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
    }


def _patch_modules(monkeypatch, *, calls: int = 1):
    import minisweagent.agents
    import minisweagent.config
    import minisweagent.environments
    import minisweagent.models

    class FakeEnvironment:
        def cleanup(self):
            pass

    class FakeModel:
        config = types.SimpleNamespace(model_name="cheap")

        def query(self, messages, **kwargs):
            return {"extra": {"response": {"usage": {"cost": 0.6}}}}

    class FakeAgent:
        def __init__(self, model):
            self.model = model
            self.messages = []

        def run(self, task):
            for _ in range(calls):
                self.messages.append(self.model.query([]))
            return {"submission": "diff --git a/a b/a\n"}

        def serialize(self):
            return {"messages": self.messages}

    monkeypatch.setattr(minisweagent.config, "get_config_from_spec", lambda path: {
        "model": {}, "agent": {}, "environment": {},
    })
    monkeypatch.setattr(
        minisweagent.environments, "get_environment", lambda config: FakeEnvironment()
    )
    monkeypatch.setattr(minisweagent.models, "get_model", lambda config: FakeModel())
    monkeypatch.setattr(
        minisweagent.agents, "get_agent",
        lambda model, env, config, default_type: FakeAgent(model),
    )


def test_dry_run_separates_caps_and_price_table():
    output = harness.dry_run(
        {"dataset": "synthetic", "dataset_revision": "test", "instance_count": 1},
        ["cheap-only", "mantis-direct", "trinity"],
        {"cheap-only": 2.0, "mantis-direct": 3.0, "trinity": 4.0},
        Path("eval/model_prices.json"),
    )
    assert "projected total: $9.00" in output
    assert "separate from caps" in output
    assert "model calls: 0 (dry-run)" in output


def test_cost_priority_and_unknown_fallback():
    prices = {"cheap": {"input_per_token": 2.0, "output_per_token": 3.0}}
    assert harness.usage_cost({"cost": 0.4}, "cheap", prices) == (0.4, "usage.cost")
    assert harness.usage_cost(
        {"prompt_tokens": 10, "completion_tokens": 5}, "cheap", prices
    ) == (35.0, "token_counts_x_price_table")
    assert harness.usage_cost({}, "missing", prices) == (None, "unknown")


def test_miniswe_local_model_configuration_is_explicit():
    config = miniswe.build_local_model_config("http://127.0.0.1:8088/v1", "mantis-trinity")
    assert config["model_kwargs"] == {
        "custom_llm_provider": "openai",
        "api_base": "http://127.0.0.1:8088/v1",
    }
    assert config["model_registry"]["mantis-trinity"]["litellm_provider"] == "openai"


def test_grader_rejects_empty_and_runs_fresh_container(monkeypatch):
    calls = []
    monkeypatch.setattr(
        harness, "_run_test_command",
        lambda image, patch, command, test_patch="", install="", timeout=300: (
            calls.append((image, patch, command)) or (True, "PASS")
        ),
    )
    assert harness.grade_patch(_instance(), "")["resolved"] is False
    assert harness.grade_patch(_instance(), "diff --git a/a b/a\n")["resolved"] is True
    assert "test_fix" in calls[0][2] and "test_existing" in calls[0][2]


def test_agent_trajectory_patch_and_grading(monkeypatch):
    _patch_modules(monkeypatch)
    monkeypatch.setattr(
        harness, "grade_patch",
        lambda instance, patch: {"resolved": True, "grader_output": "PASS"},
    )
    row = harness.run_mini_agent(
        _instance(), arm="cheap-only", endpoint="http://fake/v1/chat/completions",
        tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
        prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
        ledger=harness.CostLedger(total_limit=2.0, instance_limit=2.0),
        rng=harness.random.Random(1), step_limit=2, output_token_limit=32,
        frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
    )
    assert row["resolved"] is True
    assert row["model_patch"].startswith("diff --git")
    assert row["cost_usd"] == 0.6


def test_agent_stops_on_per_instance_budget(monkeypatch):
    _patch_modules(monkeypatch, calls=2)
    monkeypatch.setattr(
        harness, "grade_patch",
        lambda instance, patch: {"resolved": True, "grader_output": "PASS"},
    )
    row = harness.run_mini_agent(
        _instance(), arm="cheap-only", endpoint="http://fake/v1/chat/completions",
        tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
        prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
        ledger=harness.CostLedger(total_limit=5.0, instance_limit=1.0),
        rng=harness.random.Random(1), step_limit=4, output_token_limit=32,
        frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
    )
    assert row["aborted"] is True
    assert row["resolved"] is False
