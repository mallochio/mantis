"""Hermetic tests for the graded router evaluation harness."""

import importlib.util
import json
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


harness = _load("router_eval", ROOT / "eval" / "router_eval.py")


def _instance(**overrides) -> dict:
    inst = {
        "instance_id": "demo-1",
        "repo": "demo/demo",
        "base_commit": "HEAD",
        "problem_statement": "implement a parser",
        "docker_image": "demo:latest",
        "test_cmd": "pytest tests/test_demo.py",
        "FAIL_TO_PASS": ["tests/test_demo.py::test_fix"],
        "PASS_TO_PASS": ["tests/test_demo.py::test_existing"],
    }
    inst.update(overrides)
    return inst


def _upstream_server(route_headers: dict | None = None):
    seen: list[tuple[str, dict, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen.append(
                (self.path, dict(self.headers), json.loads(self.rfile.read(length)))
            )
            body = {
                "id": "fake",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.42},
            }
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            for key, value in (route_headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def _make_worktree(root: Path, instance_id: str = "demo-1") -> tuple[Path, str]:
    workdir = root / instance_id
    workdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
    subprocess.run(["git", "config", "user.email", "eval@test"], cwd=workdir, check=True)
    subprocess.run(["git", "config", "user.name", "eval"], cwd=workdir, check=True)
    (workdir / "solve.py").write_text("original\n")
    subprocess.run(["git", "add", "solve.py"], cwd=workdir, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=workdir, check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workdir, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return workdir, head


def _fake_pi(tmp_path: Path) -> Path:
    script = tmp_path / "fake-pi.sh"
    script.write_text(
        "#!/bin/sh\n"
        'if [ -n "$EVAL_PROXY_BASE_URL" ]; then\n'
        '  curl -s -X POST "$EVAL_PROXY_BASE_URL/chat/completions" '
        "-H 'Content-Type: application/json' "
        "-d '{\"model\":\"cheap\",\"messages\":[]}' >/dev/null 2>&1\n"
        "fi\n"
        "echo '{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\","
        "\"content\":[],\"usage\":{\"input\":10,\"output\":5,\"totalTokens\":15}}}'\n"
        "echo '{\"type\":\"tool_execution_end\",\"toolName\":\"bash\","
        "\"args\":{},\"result\":\"ok\",\"isError\":false}'\n"
        "echo '{\"type\":\"agent_end\",\"messages\":[]}'\n"
        "echo '{\"type\":\"agent_settled\"}'\n"
        "printf 'def patched():\\n    return 1\\n' >> solve.py\n"
        "exit 0\n"
    )
    script.chmod(0o755)
    return script


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


def test_usage_from_sse_extracts_last_usage():
    stream = (
        'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"b"}}],"usage":'
        '{"prompt_tokens":10,"completion_tokens":5,"cost":0.12}}\n\n'
        "data: [DONE]\n\n"
    )
    assert harness._usage_from_sse(stream) == {
        "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.12,
    }


def test_parse_pi_events_extracts_trajectory():
    stdout = (
        '{"type":"message_end","message":{"role":"assistant","content":[],'
        '"usage":{"input":10},"stopReason":"stop"}}\n'
        '{"type":"tool_execution_end","toolName":"bash","args":{"command":"ls"},'
        '"result":"ok","isError":false}\n'
        '{"type":"agent_end","messages":[]}\n'
    )
    trajectory = harness._parse_pi_events(stdout)
    assert trajectory[0]["role"] == "assistant"
    assert trajectory[0]["usage"]["input"] == 10
    assert trajectory[1]["tool"] == "bash"
    assert len(trajectory) == 2


def test_provider_extension_points_at_proxy(tmp_path):
    ext = tmp_path / "eval-provider.ts"
    harness._render_provider_extension(
        ["cheap", "mantis", "mantis-trinity"],
        "http://127.0.0.1:1234/v1",
        "sk-test", "router-eval-sess", 512, ext,
    )
    text = ext.read_text()
    assert 'baseUrl: "http://127.0.0.1:1234/v1"' in text
    assert 'id: "mantis"' in text
    assert 'id: "mantis-trinity"' in text
    assert "maxTokens: 512" in text
    assert '"X-Route-Session": "router-eval-sess"' in text


def test_proxy_forwards_and_records_route_headers():
    upstream, seen = _upstream_server(
        route_headers={"x-route-decision": "cheap", "x-route-model": "cheap"}
    )
    ledger = harness.CostLedger(total_limit=2.0, arm_limit=2.0)
    proxy = harness._RouteRecordingProxy(
        upstream=f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions",
        api_key="sk-test", ledger=ledger, instance_id="demo-1", arm="cheap-only",
        cap=2.0,
        prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
        model="cheap", session="router-eval-test",
    )
    try:
        resp = requests.post(
            f"{proxy.base_url()}/chat/completions",
            json={"model": "cheap", "messages": []},
            headers={"X-Route-Session": "router-eval-test"},
            timeout=30,
        )
        assert resp.status_code == 200
        assert proxy.records[0]["route_headers"]["x-route-decision"] == "cheap"
        assert proxy.records[0]["cost"] == 0.42
        assert proxy.records[0]["cost_method"] == "usage.cost"
        assert ledger.pair_cost("demo-1", "cheap-only") == 0.42
        assert seen and seen[0][1]["X-Route-Session"] == "router-eval-test"
    finally:
        proxy.close()
        upstream.shutdown()


def test_proxy_aborts_on_arm_cap():
    ledger = harness.CostLedger(total_limit=5.0, arm_limit=1.0)
    proxy = harness._RouteRecordingProxy(
        upstream="http://127.0.0.1:1/v1/chat/completions",
        api_key=None, ledger=ledger, instance_id="demo-1", arm="cheap-only",
        cap=1.0, prices={}, model="cheap", session="router-eval-test",
    )
    try:
        ledger.pair_costs[("demo-1", "cheap-only")] = 1.0
        resp = requests.post(f"{proxy.base_url()}/chat/completions", json={}, timeout=30)
        assert resp.status_code == 429
        assert proxy.exceeded is True
    finally:
        proxy.close()


def test_run_pi_agent_end_to_end(tmp_path, monkeypatch):
    worktrees = tmp_path / "worktrees"
    _, head = _make_worktree(worktrees)
    upstream, seen = _upstream_server(
        route_headers={"x-route-decision": "cheap", "x-route-model": "cheap"}
    )
    monkeypatch.setattr(
        harness, "grade_patch",
        lambda instance, patch: {"resolved": True, "grader_output": "PASS"},
    )
    fake = _fake_pi(tmp_path)
    try:
        row = harness.run_pi_agent(
            _instance(base_commit=head),
            arm="cheap-only",
            endpoint=f"http://127.0.0.1:{upstream.server_port}/v1/chat/completions",
            tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
            prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
            ledger=harness.CostLedger(total_limit=2.0, arm_limit=2.0),
            arm_cap=2.0,
            rng=harness.random.Random(1), timeout=30, output_token_limit=32,
            frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
            worktrees_root=worktrees, pi_executable=(str(fake),),
        )
    finally:
        upstream.shutdown()
    assert row["resolved"] is True
    assert row["model_patch"].startswith("diff --git")
    assert "solve.py" in row["model_patch"]
    assert row["cost_usd"] == 0.42
    assert row["cost_method"] == "usage.cost"
    assert row["route_trace"][0]["route_headers"]["x-route-decision"] == "cheap"
    assert seen and seen[0][1]["X-Route-Session"].startswith("router-eval-")


def test_run_pi_agent_aborts_on_instance_budget(tmp_path):
    worktrees = tmp_path / "worktrees"
    _, head = _make_worktree(worktrees)
    ledger = harness.CostLedger(total_limit=5.0, arm_limit=1.0)
    ledger.pair_costs[("demo-1", "cheap-only")] = 1.0
    fake = _fake_pi(tmp_path)
    row = harness.run_pi_agent(
        _instance(base_commit=head),
        arm="cheap-only",
        endpoint="http://127.0.0.1:1/v1/chat/completions",
        tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
        prices={"cheap": {"input_per_token": 1.0, "output_per_token": 1.0}},
        ledger=ledger, arm_cap=1.0,
        rng=harness.random.Random(1), timeout=30, output_token_limit=32,
        frequencies={"cheap": 1.0, "middle": 0.0, "expensive": 0.0},
        worktrees_root=worktrees, pi_executable=(str(fake),),
    )
    assert row["aborted"] is True
    assert row["abort_scope"] == "instance_arm"
    assert row["resolved"] is False
