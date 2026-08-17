import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import server

LABELS = Path(__file__).parents[3] / "eval" / "routing_quality_labels.json"
ROUTE_METRICS = Path(__file__).parents[3] / "eval" / "route_metrics.py"
_SPEC = importlib.util.spec_from_file_location("route_metrics", ROUTE_METRICS)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
replay_complexity = _MODULE.replay_complexity


def _load_policy_contract():
    data = json.loads(LABELS.read_text())
    assert data["provenance"]["status"] == "placeholder-synthetic"
    policy = data["routing_policy"]
    mapping = policy["complexity_targets"]
    fingerprint = hashlib.sha256(json.dumps(mapping, separators=(",", ":")).encode()).hexdigest()
    assert fingerprint == policy["mapping_fingerprint"]
    assert len(mapping) == policy["complexity_levels"] == 5
    assert policy["invalid_complexity_target"] == server.SUPRA_INVALID_TARGET
    assert tuple(server.SUPRA_TARGETS) == tuple(mapping)
    assert server._parse_supra_complexity("Domain: coding | Complexity: 3") == 3
    prompts = [
        {"instance_id": instance_id, "prompt": item["prompt"]}
        for instance_id, item in data["labels"].items()
    ]
    replay = replay_complexity(
        prompts,
        lambda prompt: (int(prompt[-1]), server._decide_uncached(prompt)[0]),
        {instance_id: item["oracle_tier"] for instance_id, item in data["labels"].items()},
    )
    rows = [
        (row["decision"], row["oracle_tier"])
        for row in replay["rows"]
    ]
    accuracy = sum(actual == expected for actual, expected in rows) / len(rows)
    assert accuracy >= data["thresholds"]["accuracy_min"]
    assert sum(actual != expected for actual, expected in rows) <= data["thresholds"][
        "under_routing_regret_max"
    ]
    status = data["provenance"]["status"]
    print(f"routing policy contract labels={status} accuracy={accuracy:.3f}")
    return status, accuracy


def test_routing_policy_contract_uses_placeholder_labels_and_real_mapping(monkeypatch):
    monkeypatch.setattr(
        server, "SUPRA_TARGETS", ("cheap", "cheap", "cheap", "middle", "expensive")
    )
    monkeypatch.setattr(
        server,
        "_supra_complexity",
        lambda prompt: (
            server._parse_supra_complexity(f"Complexity: {prompt[-1]}"),
            0,
        ),
    )
    status, accuracy = _load_policy_contract()
    assert status == "placeholder-synthetic"
    assert accuracy == 1.0


def test_routing_policy_contract_fails_on_mapping_perturbation(monkeypatch):
    monkeypatch.setattr(
        server, "SUPRA_TARGETS", ("cheap", "cheap", "middle", "middle", "expensive")
    )
    with pytest.raises(AssertionError):
        _load_policy_contract()
