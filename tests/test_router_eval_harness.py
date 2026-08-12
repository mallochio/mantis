"""Hermetic tests for the router evaluation harness."""

import importlib.util
import json
from pathlib import Path

import httpx

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


def _manifest() -> dict:
    return {
        "dataset": "synthetic",
        "dataset_revision": "test",
        "instance_count": 1,
        "instances": [
            {
                "instance_id": "demo-1",
                "problem_statement": "implement a parser",
                "repo": "demo/repo",
                "base_commit": "abc",
            }
        ],
    }


def test_dry_run_is_free_and_projects_caps():
    output = harness.dry_run(_manifest(), ["cheap-only", "mantis-direct", "trinity"])
    assert "projected total: $2.35" in output
    assert "model calls: 0 (dry-run)" in output


def test_miniswe_local_model_configuration_is_explicit():
    config = miniswe.build_local_model_config("http://127.0.0.1:8088/v1", "mantis-trinity")
    assert config["model_kwargs"] == {
        "custom_llm_provider": "openai",
        "api_base": "http://127.0.0.1:8088/v1",
    }
    assert config["model_registry"]["mantis-trinity"]["litellm_provider"] == "openai"


def test_fake_openai_server_captures_session_headers_and_usage_cost():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={
                "x-route-decision": "middle",
                "x-route-reason": "supra",
                "x-route-model": "gpt-5.6-terra",
                "x-route-sticky": "true",
                "x-route-fallback": "false",
            },
            json={
                "choices": [{"message": {"content": "fixed"}}],
                "usage": {"cost": 0.123, "prompt_tokens": 3, "completion_tokens": 4},
            },
        )

    row = harness.run_instance(
        _manifest()["instances"][0],
        arm="mantis-direct",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        endpoint="http://fake/v1/chat/completions",
        tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
        prices={"mantis": 99.0},
        rng=harness.random.Random(1),
        max_steps=4,
        max_output_tokens=12,
    )
    assert row["resolved"] is True
    assert row["cost_usd"] == 0.123
    assert row["cost_method"] == "usage.cost"
    assert row["cost_methods"] == ["usage.cost"]
    assert row["routing_decisions"][0]["headers"]["x-route-decision"] == "middle"
    assert requests[0].headers["x-route-session"].startswith("router-eval-")


def test_fake_server_falls_back_to_token_count_pricing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "fixed"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    row = harness.run_instance(
        _manifest()["instances"][0],
        arm="cheap-only",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        endpoint="http://fake/v1/chat/completions",
        tier_models={"cheap": "cheap", "middle": "middle", "expensive": "expensive"},
        prices={"cheap": 2.0},
        rng=harness.random.Random(1),
        max_steps=1,
        max_output_tokens=12,
    )
    assert row["cost_usd"] == 0.03
    assert row["cost_method"] == "token_counts_x_catalog_prices"


def test_budget_abort_preserves_marker(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()))
    output = tmp_path / "results.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "router_eval.py",
            "--manifest",
            str(manifest_path),
            "--arms",
            "cheap-only,middle-only",
            "--budget-usd",
            "0",
            "--output",
            str(output),
            "--endpoint",
            "http://fake",
        ],
    )
    # The first row is never sent because the ceiling is already exhausted.
    harness.main()
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[-1]["metadata"]["aborted_on_budget"] is True


def test_oracle_accuracy_regret_and_interpolation():
    rows = [
        {
            "instance_id": "a",
            "arm": "cheap-only",
            "tier": "cheap",
            "resolved": True,
            "quality": 1,
            "cost_usd": 1,
        },
        {
            "instance_id": "a",
            "arm": "middle-only",
            "tier": "middle",
            "resolved": True,
            "quality": 1,
            "cost_usd": 2,
        },
        {
            "instance_id": "a",
            "arm": "expensive-only",
            "tier": "expensive",
            "resolved": True,
            "quality": 1,
            "cost_usd": 4,
        },
        {
            "instance_id": "a",
            "arm": "mantis-direct",
            "chosen_tier": "cheap",
            "resolved": True,
            "quality": 1,
            "cost_usd": 2,
        },
        {
            "instance_id": "b",
            "arm": "cheap-only",
            "tier": "cheap",
            "resolved": False,
            "quality": 0,
            "cost_usd": 1,
        },
        {
            "instance_id": "b",
            "arm": "middle-only",
            "tier": "middle",
            "resolved": True,
            "quality": 1,
            "cost_usd": 2,
        },
        {
            "instance_id": "b",
            "arm": "expensive-only",
            "tier": "expensive",
            "resolved": True,
            "quality": 1,
            "cost_usd": 4,
        },
        {
            "instance_id": "b",
            "arm": "mantis-direct",
            "chosen_tier": "cheap",
            "resolved": True,
            "quality": 0,
            "cost_usd": 1,
        },
    ]
    result = metrics.compute_metrics(rows)
    assert result["oracle"] == {"a": "cheap", "b": "middle"}
    assert result["accuracy"]["mantis-direct"]["accuracy"] == 0.5
    assert result["regret"]["mantis-direct"]["under_routing"]["quality_lost"] == 1
    assert result["regret"]["mantis-direct"]["under_routing"]["dollars_wasted"] == 1
    assert result["cheap_expensive_interpolation"]["beats_interpolation"] is False


def test_oracle_degenerate_cases_and_confusion_matrix():
    rows = [
        {
            "instance_id": "a",
            "arm": "cheap-only",
            "tier": "cheap",
            "resolved": True,
            "quality": 1,
            "cost_usd": 1,
        },
        {
            "instance_id": "a",
            "arm": "mantis-direct",
            "chosen_tier": "cheap",
            "resolved": True,
            "quality": 1,
            "cost_usd": 1,
        },
        {
            "instance_id": "b",
            "arm": "cheap-only",
            "tier": "cheap",
            "resolved": False,
            "quality": 0,
            "cost_usd": 1,
        },
        {
            "instance_id": "b",
            "arm": "mantis-direct",
            "chosen_tier": "expensive",
            "resolved": True,
            "quality": 1,
            "cost_usd": 3,
        },
    ]
    assert metrics.oracle_labels(rows) == {"a": "cheap", "b": None}
    matrix = metrics.complexity_confusion(
        [
            {"complexity": "1", "oracle_tier": "cheap"},
            {"complexity": "3", "oracle_tier": "middle"},
        ],
        targets=("cheap", "cheap", "middle", "middle", "expensive"),
    )
    assert matrix == {"1": {"cheap": 1}, "3": {"middle": 1}}
