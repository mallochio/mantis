"""Unit checks for Zen experiment additions to eval/router_eval.py.

All pure bookkeeping: shadow pricing, fixed-arm parsing, selected-cost
semantics. No model calls, no worktrees, no proxies.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_module(name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


router_eval = _load_module("router_eval")
route_metrics = _load_module("route_metrics")

SNAPSHOT = {
    "opencode-zen/deepseek-v4-flash-free": {
        "input_per_token": 0.00000014,
        "output_per_token": 0.00000028,
        "cached_input_per_token": 0.0000000028,
        "available": True,
    },
    "opencode-zen/ling-3.0-tiny-free": {
        "input_per_token": None,
        "output_per_token": None,
        "cached_input_per_token": None,
        "available": False,
    },
    "opencode-go/deepseek-v4-flash": {
        "input_per_token": 0.00000007,
        "output_per_token": 0.00000014,
        "cached_input_per_token": 0.0000000014,
        "available": True,
    },
}
ZEN = "opencode-zen/deepseek-v4-flash-free"


def test_zero_reported_cost_still_has_shadow_cost():
    usage = {"cost": 0.0, "prompt_tokens": 10_000, "completion_tokens": 1_000}
    shadow, method = router_eval.shadow_cost(usage, ZEN, SNAPSHOT)
    assert method == "shadow.tokens_x_snapshot"
    assert shadow == pytest.approx(10_000 * 0.00000014 + 1_000 * 0.00000028)
    assert shadow > 0


def test_actual_and_shadow_costs_stay_distinct():
    usage = {"cost": 0.0, "prompt_tokens": 10_000, "completion_tokens": 1_000}
    costs = router_eval.eval_costs(usage, ZEN, {}, SNAPSHOT, "shadow")
    assert costs["actual_cost_usd"] == 0.0
    assert costs["actual_cost_method"] == "usage.cost"
    assert costs["shadow_cost_usd"] > 0
    assert costs["cost"] == costs["shadow_cost_usd"]


def test_cached_token_pricing_is_included():
    base = {"prompt_tokens": 10_000, "completion_tokens": 1_000}
    no_cache, _ = router_eval.shadow_cost(base, ZEN, SNAPSHOT)
    with_cache = dict(base, prompt_cache_hit_tokens=4_000)
    cached, _ = router_eval.shadow_cost(with_cache, ZEN, SNAPSHOT)
    assert cached == pytest.approx(no_cache + 4_000 * 0.0000000028)


def test_missing_price_or_usage_is_unknown_not_zero():
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000}
    cost, method = router_eval.shadow_cost(usage, "opencode-zen/hy3-free", SNAPSHOT)
    assert (cost, method) == (None, "shadow.missing_price")
    cost, method = router_eval.shadow_cost(usage, "opencode-zen/ling-3.0-tiny-free", SNAPSHOT)
    assert (cost, method) == (None, "shadow.unavailable")
    cost, method = router_eval.shadow_cost({"prompt_tokens": 10}, ZEN, SNAPSHOT)
    assert (cost, method) == (None, "shadow.incomplete_usage")
    cost, method = router_eval.shadow_cost({}, ZEN, SNAPSHOT)
    assert (cost, method) == (None, "shadow.incomplete_usage")


def test_exact_model_lookup_rejects_basename_ambiguity():
    # "deepseek-v4-flash" basename exists under both opencode-zen (as -free)
    # and opencode-go (paid); an unqualified or differently-qualified ID must
    # never match the wrong row.
    usage = {"prompt_tokens": 10_000, "completion_tokens": 1_000}
    cost, method = router_eval.shadow_cost(usage, "deepseek-v4-flash-free", SNAPSHOT)
    assert (cost, method) == (None, "shadow.missing_price")
    cost, method = router_eval.shadow_cost(usage, "opencode-go/deepseek-v4-flash", SNAPSHOT)
    assert method == "shadow.tokens_x_snapshot"
    assert cost == pytest.approx(
        10_000 * 0.00000007 + 1_000 * 0.00000014
    )
    zen_cost, _ = router_eval.shadow_cost(usage, ZEN, SNAPSHOT)
    assert cost != zen_cost


def test_route_metrics_consume_selected_evaluation_cost():
    rows = [
        {"instance_id": "i1", "arm": "zen-flash", "cost_usd": 0.5, "resolved": True},
        {"instance_id": "i1", "arm": "mantis-direct", "cost_usd": 0.25, "resolved": True},
    ]
    assert route_metrics._cost_sum(rows) == pytest.approx(0.75)
    assert route_metrics._cost_sum([{**rows[0], "cost_usd": None}]) is None


def test_pair_cost_is_none_when_any_request_cost_unknown():
    records = [
        {"cost": 0.01, "cost_method": "shadow.tokens_x_snapshot"},
        {"cost": None, "cost_method": "shadow.incomplete_usage"},
    ]
    assert router_eval._sum_cost(records, "cost") is None
    assert router_eval._sum_cost(records, "cost") != 0.0
    assert router_eval._sum_cost([records[0]], "cost") == pytest.approx(0.01)


def test_fixed_arms_require_provider_qualified_models():
    with pytest.raises(ValueError, match="provider-qualified"):
        router_eval.parse_fixed_models(["zen-flash=deepseek-v4-flash-free"])
    with pytest.raises(ValueError, match="collides"):
        router_eval.parse_fixed_models(["cheap-only=opencode-zen/deepseek-v4-flash-free"])

    fixed = router_eval.parse_fixed_models([
        "low=opencode-zen/hy3-free",
        "high=opencode-go/deepseek-v4-flash",
    ])
    assert fixed == {
        "low": "opencode-zen/hy3-free",
        "high": "opencode-go/deepseek-v4-flash",
    }
    model, tier = router_eval._model_for_arm(
        "high", "p", {}, router_eval.random.Random(1), {}, fixed
    )
    assert model == "opencode-go/deepseek-v4-flash"
    assert tier is None


def test_committed_zen_snapshot_shape_and_values():
    snapshot = router_eval.load_shadow_prices(router_eval.DEFAULT_SHADOW_PRICES)
    assert set(snapshot) == {
        "opencode-zen/deepseek-v4-flash-free",
        "opencode-zen/mimo-v2.5-free",
        "opencode-zen/hy3-free",
        "opencode-zen/ling-3.0-tiny-free",
        "opencode-zen/nemotron-3-ultra-free",
        "opencode-zen/nemotron-3.5-lightning-free",
        "opencode-zen/laguna-s-2.1-free",
    }
    for model, entry in snapshot.items():
        if entry["available"]:
            for key in ("input_per_token", "output_per_token", "cached_input_per_token"):
                assert entry[key] and entry[key] > 0, f"{model}.{key}"
        else:
            # unavailable models must fail lookup, never price at $0
            cost, method = router_eval.shadow_cost(
                {"prompt_tokens": 1, "completion_tokens": 1}, model, snapshot
            )
            assert (cost, method) == (None, "shadow.unavailable")
