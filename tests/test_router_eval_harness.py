"""Hermetic tests for the graded router evaluation harness."""

import importlib.util
import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

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
        lambda image, patch, command, test_patch="", install="", timeout=300, **kwargs: (
            calls.append((image, patch, command)) or (True, "PASS")
        ),
    )
    assert harness.grade_patch(_instance(), "")["resolved"] is False
    assert harness.grade_patch(_instance(), "diff --git a/a b/a\n")["resolved"] is True
    assert {call[2].split("::")[-1] for call in calls} == {
        "test_fix", "test_existing",
    }


def test_grader_rejects_test_tampering(monkeypatch):
    seen = []

    def run(*, reset_paths, **kwargs):
        seen.append(reset_paths)
        return False, "test files reset before official patch"

    monkeypatch.setattr(harness, "_run_test_command", run)
    instance = {**_instance(), "test_patch": "+++ b/tests/test_demo.py\n"}
    result = harness.grade_patch(instance, "diff --git a/tests/test_demo.py b/tests/test_demo.py\n")
    assert result["resolved"] is False
    assert seen and seen[0] == ["tests/test_demo.py"]


def test_cost_ledger_isolates_arm_caps_and_global_budget():
    ledger = harness.CostLedger(total_limit=2.0, arm_limit=1.0)
    ledger.record("item", "cheap-only", 0.6, 1.0)
    ledger.record("item", "expensive-only", 0.8, 1.0)
    assert ledger.pair_cost("item", "cheap-only") == 0.6
    assert ledger.pair_cost("item", "expensive-only") == 0.8
    with pytest.raises(harness.ArmBudgetExceeded):
        ledger.record("item", "cheap-only", 0.5, 1.0)
    with pytest.raises(harness.BudgetAbort):
        ledger.record("item", "middle-only", 0.6, 1.0)


def test_direct_route_frequency_uses_request_headers():
    row = {
        "instance_id": "item",
        "route_trace": [
            {"route_headers": {"x-route-model": "cheap-model"}},
            {"route_headers": {"x-route-decision": "middle"}},
        ],
    }
    assert harness.routed_request_tiers(
        row,
        {"cheap": "cheap-model", "middle": "middle-model", "expensive": "expensive-model"},
    ) == ["cheap", "middle"]


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
        ledger=harness.CostLedger(total_limit=2.0, arm_limit=2.0),
        arm_cap=2.0,
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
        ledger=harness.CostLedger(total_limit=5.0, arm_limit=1.0),
        arm_cap=1.0,
        rng=harness.random.Random(1), step_limit=4, output_token_limit=32,
        frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
    )
    assert row["aborted"] is True
    assert row["resolved"] is False


def test_agent_model_reaches_loopback_openai_endpoint(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            requests.append((self.path, self.headers, json.loads(self.rfile.read(length))))
            body = {
                "id": "fake",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "bash", "arguments": '{"command":"echo ok"}'},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "cost": 0.12},
            }
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("x-route-decision", "cheap")
            self.send_header("x-route-model", "deepseek-v4-flash")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import minisweagent.agents
        import minisweagent.config
        import minisweagent.environments

        class FakeEnvironment:
            def cleanup(self):
                pass

        class FakeAgent:
            def __init__(self, model):
                self.model = model
                self.messages = []

            def run(self, task):
                self.messages.append(self.model.query([]))
                return {"submission": "diff --git a/a b/a\n"}

            def serialize(self):
                return {"messages": self.messages}

        monkeypatch.setattr(
            minisweagent.config, "get_config_from_spec",
            lambda path: {"model": {}, "agent": {}, "environment": {}},
        )
        monkeypatch.setattr(
            minisweagent.environments, "get_environment",
            lambda config: FakeEnvironment(),
        )
        monkeypatch.setattr(
            minisweagent.agents, "get_agent",
            lambda model, env, config, default_type: FakeAgent(model),
        )
        monkeypatch.setattr(
            harness, "grade_patch",
            lambda instance, patch: {"resolved": True, "grader_output": "PASS"},
        )
        row = harness.run_mini_agent(
            _instance(),
            arm="cheap-only",
            endpoint=f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
            prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
            ledger=harness.CostLedger(total_limit=2.0, arm_limit=2.0),
            arm_cap=2.0,
            rng=harness.random.Random(1),
            step_limit=2,
            output_token_limit=32,
            frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
        )
        assert row["resolved"] is True
        assert row["cost_usd"] == 0.12
        assert row["route_trace"][0]["route_headers"]["x-route-decision"] == "cheap"
        assert requests and requests[0][2]["model"] == "cheap"
        assert requests[0][1]["X-Route-Session"].startswith("router-eval-")
    finally:
        server.shutdown()
