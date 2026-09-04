"""Unit tests for scripts/retrain_conductor.py.

These tests cover pool parsing, reward logic, and prompt construction without
spinning up GRPO training or GPU inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Import the module under test from the scripts package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import retrain_conductor as rc  # noqa: E402
import retrain_router_pool as rp  # noqa: E402
from ultra import CANNED, MockWorker  # noqa: E402


# ---------------------------------------------------------------------------
# Pool-spec reuse
# ---------------------------------------------------------------------------
def test_pool_utils_are_reused_from_router_retrain():
    """retrain_conductor must import split_model_spec and normalize_model_id
    from retrain_router_pool, not duplicate them."""
    assert rc.split_model_spec is rp.split_model_spec
    assert rc.normalize_model_id is rp.normalize_model_id


@pytest.mark.parametrize(
    "spec,model,effort",
    [
        ("openai/gpt-5.6-sol|medium", "openai/gpt-5.6-sol", "medium"),
        ("deepseek/deepseek-v4-flash|none", "deepseek/deepseek-v4-flash", "none"),
        ("anthropic/claude-opus-5", "anthropic/claude-opus-5", ""),
    ],
)
def test_split_model_spec_matches_router(spec, model, effort):
    m, e = rc.split_model_spec(spec)
    assert m == model
    if effort:
        assert e == effort
    else:
        assert e is None


# ---------------------------------------------------------------------------
# Prompt / dataset construction
# ---------------------------------------------------------------------------
def _mock_tokenizer():
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "</s>"

    def _apply_chat_template(msgs, tokenize=False, continue_final_message=False, **kwargs):
        return "\n".join(f"{m['role']}: {m['content']}" for m in msgs)

    tok.apply_chat_template = _apply_chat_template
    return tok


def test_build_prompt_contains_conductor_lists():
    tok = _mock_tokenizer()
    labels = [f"model{i}" for i in range(rc.N_AGENTS)]
    prompt = rc._build_prompt("sum 2+2", tok, labels, 1024)
    assert "model_id" in prompt
    assert "USER QUESTION: sum 2+2" in prompt
    assert "Plan:" in prompt


def test_build_dataset_carries_expected_column():
    pytest.importorskip("datasets")
    tok = _mock_tokenizer()
    records = [
        {"task": "a", "expected": "solve_a.sh"},
        {"task": "b", "expected": "solve_b.sh"},
    ]
    ds = rc._build_dataset(records, tok, ["m"] * rc.N_AGENTS, 1024)
    assert ds["expected"] == ["solve_a.sh", "solve_b.sh"]
    assert len(ds["prompt"]) == 2


# ---------------------------------------------------------------------------
# Reward parsing
# ---------------------------------------------------------------------------
def test_format_reward_perfect_canned_workflow():
    assert rc._format_reward_one(CANNED) == 1.0


def test_action_reward_perfect_canned_workflow():
    labels = [f"slot{i}" for i in range(rc.N_AGENTS)]
    assert rc._action_reward_one(CANNED, labels) == 1.0


def test_action_reward_rejects_forward_reference():
    bad = (
        "Plan:\n"
        "model_id: [0, 1]\n"
        'subtasks: ["a", "b"]\n'
        "access_list: [[], [1]]\n"
    )
    assert rc._action_reward_one(bad, ["m"] * rc.N_AGENTS) == 0.0


def test_format_reward_rejects_unequal_lists():
    bad = (
        "Plan:\n"
        "model_id: [0, 1]\n"
        'subtasks: ["a"]\n'
        "access_list: [[], []]\n"
    )
    assert rc._format_reward_one(bad) == 0.0


def test_format_reward_rejects_too_many_steps():
    bad = (
        "Plan:\n"
        "model_id: [0, 1, 2, 3, 4, 5]\n"
        'subtasks: ["a", "b", "c", "d", "e", "f"]\n'
        "access_list: [[], [], [], [], [], []]\n"
    )
    assert rc._format_reward_one(bad) == 0.0


# ---------------------------------------------------------------------------
# Outcome reward with MockWorker
# ---------------------------------------------------------------------------
def test_outcome_reward_matches_mock_worker_output():
    worker = MockWorker()
    completion = (
        "Plan:\n"
        "model_id: [0]\n"
        'subtasks: ["say hello"]\n'
        "access_list: [[]]\n"
    )
    expected = "[agent 0] result for: say hello"
    score = rc._outcome_reward_one(completion, expected, worker, ["m"] * rc.N_AGENTS)
    assert score == pytest.approx(1.0)


def test_outcome_reward_with_toolscale_actions():
    worker = MockWorker()
    gold = [{"name": "get_forecast", "arguments": {"id": "1"}}]
    completion = (
        "Plan:\n"
        "model_id: [0]\n"
        'subtasks: ["fetch forecast"]\n'
        "access_list: [[]]\n"
    )
    # MockWorker returns a generic string, not a tool-call plan, so the outcome
    # reward falls back to a low string-similarity score.
    score = rc._outcome_reward_one(completion, gold, worker, ["m"] * rc.N_AGENTS)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Reward function closures
# ---------------------------------------------------------------------------
def test_make_reward_functions_return_expected_scores():
    worker = MockWorker()
    labels = ["m"] * rc.N_AGENTS
    format_fn, action_fn, outcome_fn = rc.make_reward_functions(worker, labels)

    completions = [CANNED, "not a workflow"]
    assert format_fn(completions) == [1.0, 0.0]
    assert action_fn(completions) == [1.0, 0.0]

    expected = ["[agent 0] result for: Devise an algorithm for the task"]
    scores = outcome_fn([CANNED], expected=expected)
    assert 0.0 <= scores[0] <= 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def test_load_records_dispatches_to_terminalbench(monkeypatch):
    called = {}

    def _fake_terminal(dataset, limit, seed, val_frac, cache_dir=None):
        called["terminal"] = (dataset, limit, seed, val_frac)
        return ([{"task": "t", "expected": "e"}], [])

    def _fake_toolscale(limit, seed, val_frac):
        called["toolscale"] = (limit, seed, val_frac)
        return ([{"task": "t2", "expected": "e2"}], [])

    monkeypatch.setattr(rc, "load_terminalbench_tasks", _fake_terminal)
    monkeypatch.setattr(rc, "load_toolscale_tasks", _fake_toolscale)

    records = rc._load_records("s3://bucket/terminal-bench-2.1/", 8, 123)
    assert records == [{"task": "t", "expected": "e"}]
    assert called["terminal"] == ("s3://bucket/terminal-bench-2.1/", 8, 123, 0.0)
    assert "toolscale" not in called


def test_load_records_dispatches_to_toolscale(monkeypatch):
    called = {}

    def _fake_terminal(dataset, limit, seed, val_frac, cache_dir=None):
        called["terminal"] = True
        return ([{"task": "t", "expected": "e"}], [])

    def _fake_toolscale(limit, seed, val_frac):
        called["toolscale"] = (limit, seed, val_frac)
        return ([{"task": "t2", "expected": json.dumps([{"name": "x"}])}], [])

    monkeypatch.setattr(rc, "load_terminalbench_tasks", _fake_terminal)
    monkeypatch.setattr(rc, "load_toolscale_tasks", _fake_toolscale)

    records = rc._load_records("nvidia/ToolScale", 5, 42)
    assert records == [{"task": "t2", "expected": json.dumps([{"name": "x"}])}]
    assert called["toolscale"] == (5, 42, 0.0)
    assert "terminal" not in called


# ---------------------------------------------------------------------------
# Real checkpoint / manifest helpers
# ---------------------------------------------------------------------------
def test_is_real_checkpoint_recognizes_default_id():
    assert rc._is_real_checkpoint("di-zhang-fdu/openfugu-conductor-3b") is True
    assert rc._is_real_checkpoint("/data/checkpoints/openfugu-conductor-3b") is True
    assert rc._is_real_checkpoint("HuggingFaceTB/SmolLM2-135M-Instruct") is False


def test_real_checkpoint_smoke_rejects_cpu(monkeypatch):
    """--real-checkpoint-smoke must fail fast on CPU/MPS."""
    called = {}

    def _fake_load_records(dataset, limit, seed):
        called["loaded"] = True
        return [{"task": "t", "expected": "e"}]

    monkeypatch.setattr(rc, "_load_records", _fake_load_records)
    monkeypatch.setattr(rc.sys, "argv", [
        "retrain_conductor.py",
        "--pool",
        ",".join(["openai/gpt-5.6-sol|none"] * rc.N_AGENTS),
        "--real-checkpoint-smoke",
        "--base",
        rc.DEFAULT_BASE,
    ])

    # Simulate CPU even if tests run on GPU.
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit):
        rc.main()
    assert called.get("loaded") is True
