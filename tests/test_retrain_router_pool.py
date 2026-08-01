"""Unit tests for scripts/retrain_router_pool.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import retrain_router_pool as rp
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeBoto3:
    """Return canned S3 object metadata and writes files into the local dir."""

    def __init__(self, prefix: str, files: dict[str, str]):
        self.prefix = prefix.rstrip("/")
        self.files = {f"{self.prefix}/{k}": v for k, v in files.items()}

    def client(self, name: str) -> _FakeClient:
        return _FakeClient(self.files)


class _FakeClient:
    def __init__(self, files: dict[str, str]):
        self._files = files

    def get_paginator(self, name: str):
        return _FakePaginator(self._files)


class _FakePaginator:
    def __init__(self, files: dict[str, str]):
        self._files = files

    def paginate(self, **kwargs):
        keys = [k for k in self._files if k.startswith(kwargs.get("Prefix", ""))]
        yield {"Contents": [{"Key": k} for k in keys]}


class _FakeDownloader:
    def __init__(self, files: dict[str, str]):
        self._files = files

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        Path(dest).write_text(self._files[key])


def _make_boto3_module(files: dict[str, str], prefix: str) -> Any:
    """Build a minimal boto3 module substitute."""
    mod = MagicMock()
    mod.client = lambda name: _FakeClientForBucket(files, prefix)
    return mod


class _FakeClientForBucket:
    def __init__(self, files: dict[str, str], prefix: str):
        self._files = files
        self._prefix = prefix

    def get_paginator(self, name: str):
        return self

    def paginate(self, **kwargs):
        keys = [k for k in self._files if k.startswith(kwargs.get("Prefix", ""))]
        yield {"Contents": [{"Key": k} for k in keys]}

    def download_file(self, bucket: str, key: str, dest: str) -> None:
        Path(dest).write_text(self._files[key])


# ---------------------------------------------------------------------------
# Model spec parsing
# ---------------------------------------------------------------------------
def test_split_model_spec_with_effort():
    assert rp.split_model_spec("openai/gpt-5.6-sol|medium") == ("openai/gpt-5.6-sol", "medium")


def test_split_model_spec_without_effort():
    assert rp.split_model_spec("anthropic/claude-sonnet-5 ") == ("anthropic/claude-sonnet-5", None)


def test_split_model_spec_empty_effort():
    assert rp.split_model_spec("anthropic/claude-sonnet-5|") == ("anthropic/claude-sonnet-5", None)


def test_normalize_full_openrouter_id():
    assert rp.normalize_model_id("openrouter/anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_normalize_known_alias():
    assert rp.normalize_model_id("claude-sonnet-5") == "anthropic/claude-sonnet-5"
    assert rp.normalize_model_id("gpt-5.6-luna") == "openai/gpt-5.6-luna"
    assert rp.normalize_model_id("deepseek-v4-flash") == "deepseek/deepseek-v4-flash"
    assert rp.normalize_model_id("glm-5.2") == "z-ai/glm-5.2"


def test_normalize_unsupported_raises():
    with pytest.raises(ValueError):
        rp.normalize_model_id("unknown-model")


def test_is_reasoning_model():
    assert rp._is_reasoning_model("claude-sonnet-5")
    assert rp._is_reasoning_model("openai/gpt-5.6-terra")
    assert not rp._is_reasoning_model("deepseek/deepseek-v4-flash")
    assert not rp._is_reasoning_model("z-ai/glm-5.2")


# ---------------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------------
def test_openrouter_worker_reasoning(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_completion(**kw):
        captured.update(kw)
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    fake_litellm = MagicMock(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    worker = rp.OpenRouterWorker(["claude-sonnet-5|medium"])
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)

    assert result == "ok"
    assert captured["model"] == "openai/anthropic/claude-sonnet-5"
    assert captured["reasoning_effort"] == "medium"
    assert captured["custom_llm_provider"] == "openai"
    assert "temperature" not in captured


def test_openrouter_worker_non_reasoning(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_completion(**kw):
        captured.update(kw)
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    fake_litellm = MagicMock(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    worker = rp.OpenRouterWorker(["deepseek-v4-flash|none"])
    result = worker("Worker", [{"role": "user", "content": "hi"}], 0)

    assert result == "ok"
    assert captured["temperature"] == 0.2
    assert captured["reasoning_effort"] == "none"


def test_openrouter_worker_failure(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("api down")

    fake_litellm = MagicMock(completion=_boom)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    worker = rp.OpenRouterWorker(["glm-5.2"])
    assert worker("Worker", [{"role": "user", "content": "hi"}], 0) == ""


# ---------------------------------------------------------------------------
# Messages and reward
# ---------------------------------------------------------------------------
def test_worker_messages_default_system():
    msgs = rp.worker_messages("solve")
    assert msgs[0]["role"] == "system"
    assert "tool-use planner" in msgs[0]["content"]
    assert msgs[1]["content"] == "solve"


def test_worker_messages_custom_system():
    msgs = rp.worker_messages("task", "custom system")
    assert msgs[0]["content"] == "custom system"


def test_reward_for_str():
    assert rp.reward_for("echo hi", "echo hi") == pytest.approx(1.0)
    assert rp.reward_for("abc", "xyz") < 1.0


def test_reward_for_empty_str():
    assert rp.reward_for("", "foo") == 0.0
    assert rp.reward_for("foo", "") == 0.0


def test_reward_for_toolscale():
    completion = "<answer>[{'name': 'a', 'arguments': {'x': 1}}]</answer>"
    gold = [{"name": "a", "arguments": {"x": 1}}]
    assert rp.reward_for(completion, gold) > 0.0


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def test_load_terminalbench_local(tmp_path):
    task_dir = tmp_path / "task-1"
    (task_dir / "solution").mkdir(parents=True)
    (task_dir / "instruction.md").write_text("write hello")
    (task_dir / "solution" / "solve.sh").write_text("echo hello")

    train, val = rp.load_terminalbench_tasks(str(tmp_path), limit=1, seed=1, val_frac=0.0)
    assert len(train) == 1
    assert train[0]["task"] == "write hello"
    assert train[0]["expected"] == "echo hello"
    assert train[0]["system"] == rp.TERMINAL_SYSTEM


def test_load_terminalbench_s3(monkeypatch, tmp_path):
    prefix = "terminal-bench-2.1"
    files = {
        f"{prefix}/task-a/instruction.md": "task a",
        f"{prefix}/task-a/solution/solve.sh": "solve a",
    }

    def _fake_client(name: str) -> Any:
        return _FakeClientForBucket(files, prefix)

    fake_boto3 = MagicMock()
    fake_boto3.client = _fake_client
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    train, val = rp.load_terminalbench_tasks(
        f"s3://external/{prefix}/", limit=10, val_frac=0.0, cache_dir=str(tmp_path / "cache")
    )
    assert len(train) == 1
    assert train[0]["task"] == "task a"


def test_load_toolscale_tasks(monkeypatch):
    class _FakeDS:
        def __init__(self, rows):
            self._rows = rows

        def shuffle(self, seed: int):
            return self

        def __iter__(self):
            return iter(self._rows)

    rows = [
        {
            "user_scenario": {"instructions": {"task_instructions": "do X"}},
            "evaluation_criteria": {"actions": [{"name": "tool", "arguments": {}}]},
        }
    ]
    monkeypatch.setattr(rp, "load_dataset", lambda *args, **kwargs: _FakeDS(rows))
    train, val = rp.load_toolscale_tasks(limit=1)
    assert len(train) == 1
    assert train[0]["task"] == "do X"
    assert train[0]["expected"] == [{"name": "tool", "arguments": {}}]


# ---------------------------------------------------------------------------
# Hidden-state extraction and training
# ---------------------------------------------------------------------------
def test_extract_hidden_states():
    router = MagicMock()
    router.model.eval = lambda: None
    router._hidden = lambda msgs: torch.tensor([1.0, 2.0, 3.0])
    tasks = ["a", "b"]
    X = rp.extract_hidden_states(router, tasks)
    assert X.shape == (2, 3)


def test_train_head(tmp_path):
    n = 4
    X = torch.randn(n, rp.HIDDEN)
    y_worker = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    y_role = torch.tensor([2, 2, 1, 1], dtype=torch.long)
    head0 = torch.randn(rp.HEAD_ROWS, rp.HIDDEN)
    weight, best = rp.train_head(X, y_worker, y_role, head0, epochs=2, device="cpu")
    assert weight.shape == (rp.HEAD_ROWS, rp.HIDDEN)
    assert 0.0 <= best <= 1.0


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------
def test_main_smoke(monkeypatch, tmp_path):
    # TerminalBench local fixture
    base = tmp_path / "tb"
    for name, task, solve in [
        ("task-1", "print one", "echo one"),
        ("task-2", "print two", "echo two"),
    ]:
        d = base / name
        (d / "solution").mkdir(parents=True)
        (d / "instruction.md").write_text(task)
        (d / "solution" / "solve.sh").write_text(solve)

    # Fake vector file
    vec = tmp_path / "vec.npy"
    np.save(vec, np.zeros(rp.VEC_LEN))

    class FakeWorker:
        def __init__(self, pool):
            self.pool = pool

        def __call__(self, role, messages, agent_id):
            return f"reply-{agent_id}"

    class FakeRouter:
        head = torch.zeros(rp.HEAD_ROWS, rp.HIDDEN)
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def route(self, messages, sample=False):
            return {"agent_id": 0, "role_id": 1}

    monkeypatch.setattr(rp, "OpenRouterWorker", FakeWorker)
    monkeypatch.setattr(rp, "FuguRouter", FakeRouter)
    monkeypatch.setattr(rp, "extract_hidden_states", lambda router, tasks, batch_size=8: torch.zeros(len(tasks), rp.HIDDEN))
    monkeypatch.setattr(
        rp,
        "train_head",
        lambda X, yw, yr, h0, **kwargs: (torch.zeros(rp.HEAD_ROWS, rp.HIDDEN), 0.75),
    )
    monkeypatch.setattr(rp, "tqdm", lambda x, **kw: x)

    out_dir = tmp_path / "out"
    argv = [
        "--pool",
        "anthropic/claude-sonnet-5|medium,anthropic/claude-opus-5|medium,"
        "openai/gpt-5.6-sol|medium,openai/gpt-5.6-luna|max,"
        "openai/gpt-5.6-terra|xhigh,deepseek/deepseek-v4-flash|none,"
        "z-ai/glm-5.2|none",
        "--dataset",
        str(base),
        "--limit",
        "2",
        "--epochs",
        "1",
        "--fugu-vector",
        str(vec),
        "--output-dir",
        str(out_dir),
        "--val-frac",
        "0.5",
    ]
    rp.main(argv)

    assert (out_dir / "report.json").exists()
    report = json.loads((out_dir / "report.json").read_text())
    assert report["train_size"] == 1
    assert report["val_size"] == 1
    assert report["vector"].endswith("model_iter_60.npy")
    assert (out_dir / "train_labels.jsonl").exists()


def test_normalize_model_id_with_slash():
    assert rp.normalize_model_id("anthropic/claude-sonnet-5") == "anthropic/claude-sonnet-5"


def test_normalize_unsupported_alias():
    with pytest.raises(ValueError):
        rp.normalize_model_id("not-a-real-model")


def test_openrouter_worker_api_base(monkeypatch):
    captured: dict[str, Any] = {}

    def _fake_completion(**kw):
        captured.update(kw)
        return MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])

    fake_litellm = MagicMock(completion=_fake_completion)
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")

    worker = rp.OpenRouterWorker(["claude-sonnet-5|medium"], api_base="http://custom/")
    worker("Worker", [{"role": "user", "content": "hi"}], 0)
    assert captured["api_base"] == "http://custom/"


def test_openrouter_worker_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "litellm", MagicMock())
    with pytest.raises(ValueError):
        rp.OpenRouterWorker(["claude-sonnet-5"])


def test_sync_s3_to_local_invalid_uri(tmp_path):
    with pytest.raises(ValueError):
        rp._sync_s3_to_local("not-s3://bucket/path", tmp_path)


def test_sync_s3_to_local_skips_non_matching_keys(monkeypatch, tmp_path):
    prefix = "tb"
    files = {
        f"{prefix}/task-a/instruction.md": "task a",
        f"{prefix}/task-a/solution/solve.sh": "solve a",
        f"{prefix}/README.md": "ignore me",
    }
    monkeypatch.setitem(sys.modules, "boto3", _make_boto3_module(files, prefix))
    rp._sync_s3_to_local(f"s3://external/{prefix}/", tmp_path)
    assert (tmp_path / "task-a" / "instruction.md").exists()
    assert not (tmp_path / "README.md").exists()


def test_load_terminalbench_skips_missing_solve(tmp_path):
    d = tmp_path / "task-1"
    d.mkdir()
    (d / "instruction.md").write_text("task")
    train, val = rp.load_terminalbench_tasks(str(tmp_path), limit=10, val_frac=0.0)
    assert train == []


def test_load_terminalbench_with_limit(tmp_path):
    for i in range(3):
        d = tmp_path / f"task-{i}"
        (d / "solution").mkdir(parents=True)
        (d / "instruction.md").write_text(f"task {i}")
        (d / "solution" / "solve.sh").write_text(f"solve {i}")
    train, val = rp.load_terminalbench_tasks(str(tmp_path), limit=2, val_frac=0.0)
    assert len(train) == 2


def test_load_toolscale_with_limit(monkeypatch):
    class _FakeDS:
        def __init__(self, rows):
            self._rows = rows

        def shuffle(self, seed: int):
            return self

        def __iter__(self):
            return iter(self._rows)

    rows = [
        {
            "user_scenario": {"instructions": {"task_instructions": f"do {i}"}},
            "evaluation_criteria": {"actions": [{"name": "tool", "arguments": {}}]},
        }
        for i in range(3)
    ]
    monkeypatch.setattr(rp, "load_dataset", lambda *a, **kw: _FakeDS(rows))
    train, val = rp.load_toolscale_tasks(limit=2)
    assert len(train) == 2


def test_reward_for_toolscale_none():
    gold = [{"name": "tool", "arguments": {}}]
    assert rp.reward_for("no answer tags", gold) == 0.0


def test_train_head_no_validation():
    X = torch.randn(1, rp.HIDDEN)
    y_worker = torch.tensor([0], dtype=torch.long)
    y_role = torch.tensor([2], dtype=torch.long)
    head0 = torch.randn(rp.HEAD_ROWS, rp.HIDDEN)
    weight, best = rp.train_head(X, y_worker, y_role, head0, epochs=2, device="cpu")
    assert weight.shape == (rp.HEAD_ROWS, rp.HIDDEN)


def test_main_no_pool():
    with pytest.raises(SystemExit):
        rp.main([])


def test_main_toolscale(monkeypatch, tmp_path):
    vec = tmp_path / "vec.npy"
    np.save(vec, np.zeros(rp.VEC_LEN))

    class FakeWorker:
        def __init__(self, pool):
            self.pool = pool

        def __call__(self, role, messages, agent_id):
            return f"reply-{agent_id}"

    class FakeRouter:
        head = torch.zeros(rp.HEAD_ROWS, rp.HIDDEN)
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def route(self, messages, sample=False):
            return {"agent_id": 0, "role_id": 1}

    monkeypatch.setattr(rp, "OpenRouterWorker", FakeWorker)
    monkeypatch.setattr(rp, "FuguRouter", FakeRouter)
    monkeypatch.setattr(
        rp, "extract_hidden_states",
        lambda router, tasks, batch_size=8: torch.zeros(len(tasks), rp.HIDDEN),
    )
    monkeypatch.setattr(
        rp, "train_head",
        lambda X, yw, yr, h0, **kwargs: (torch.zeros(rp.HEAD_ROWS, rp.HIDDEN), 0.5),
    )
    monkeypatch.setattr(rp, "tqdm", lambda x, **kw: x)
    monkeypatch.setattr(
        rp,
        "load_toolscale_tasks",
        lambda limit, seed=42, val_frac=0.1: (
            [{"task": "do X", "expected": [{"name": "tool"}], "system": None}],
            [{"task": "do Y", "expected": [{"name": "tool"}], "system": None}],
        ),
    )

    out_dir = tmp_path / "out"
    argv = [
        "--pool",
        "anthropic/claude-sonnet-5,anthropic/claude-opus-5,"
        "openai/gpt-5.6-sol,openai/gpt-5.6-luna,"
        "openai/gpt-5.6-terra,deepseek/deepseek-v4-flash,"
        "z-ai/glm-5.2",
        "--dataset",
        "nvidia/ToolScale",
        "--limit",
        "1",
        "--epochs",
        "1",
        "--fugu-vector",
        str(vec),
        "--output-dir",
        str(out_dir),
        "--val-frac",
        "0.5",
    ]
    rp.main(argv)

    assert (out_dir / "report.json").exists()
    report = json.loads((out_dir / "report.json").read_text())
    assert report["dataset"] == "nvidia/ToolScale"


def test_main_uses_cache(monkeypatch, tmp_path):
    vec = tmp_path / "vec.npy"
    np.save(vec, np.zeros(rp.VEC_LEN))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cache_path = out_dir / "worker_outputs.jsonl"
    cache_lines = [
        json.dumps({"task": f"task-{t}", "model": f"m{i}", "completion": "cached"})
        for t in ("a", "b")
        for i in range(7)
    ]
    cache_path.write_text("\n".join(cache_lines) + "\n")

    class FakeWorker:
        def __init__(self, pool):
            self.pool = pool

        def __call__(self, role, messages, agent_id):
            raise AssertionError("worker should not be called when cache hit")

    class FakeRouter:
        head = torch.zeros(rp.HEAD_ROWS, rp.HIDDEN)
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def route(self, messages, sample=False):
            return {"agent_id": 0, "role_id": 1}

    monkeypatch.setattr(rp, "OpenRouterWorker", FakeWorker)
    monkeypatch.setattr(rp, "FuguRouter", FakeRouter)
    monkeypatch.setattr(
        rp, "extract_hidden_states",
        lambda router, tasks, batch_size=8: torch.zeros(len(tasks), rp.HIDDEN),
    )
    monkeypatch.setattr(
        rp, "train_head",
        lambda X, yw, yr, h0, **kwargs: (torch.zeros(rp.HEAD_ROWS, rp.HIDDEN), 0.5),
    )
    monkeypatch.setattr(rp, "tqdm", lambda x, **kw: x)
    monkeypatch.setattr(
        rp,
        "load_terminalbench_tasks",
        lambda *a, **kw: (
            [
                {"task": "task-a", "expected": "echo a", "system": None, "name": "task-a"},
                {"task": "task-b", "expected": "echo b", "system": None, "name": "task-b"},
            ],
            [],
        ),
    )

    pool = "m0,m1,m2,m3,m4,m5,m6"
    argv = [
        "--pool", pool,
        "--dataset", "terminal",
        "--limit", "2",
        "--epochs", "1",
        "--fugu-vector", str(vec),
        "--output-dir", str(out_dir),
        "--val-frac", "0.0",
    ]
    rp.main(argv)
    report = json.loads((out_dir / "report.json").read_text())
    assert report["train_size"] == 2


def test_extract_hidden_states_numpy():
    router = MagicMock()
    router.model.eval = lambda: None
    router._hidden = lambda msgs: np.array([1.0, 2.0, 3.0])
    tasks = ["a"]
    X = rp.extract_hidden_states(router, tasks)
    assert X.shape == (1, 3)
    assert isinstance(X, torch.Tensor)


def test_train_head_auto_device():
    X = torch.randn(2, rp.HIDDEN)
    y_worker = torch.tensor([0, 1], dtype=torch.long)
    y_role = torch.tensor([2, 1], dtype=torch.long)
    head0 = torch.randn(rp.HEAD_ROWS, rp.HIDDEN)
    weight, best = rp.train_head(X, y_worker, y_role, head0, epochs=1)
    assert weight.shape == (rp.HEAD_ROWS, rp.HIDDEN)


def test_main_pad_pool(monkeypatch, tmp_path):
    vec = tmp_path / "vec.npy"
    np.save(vec, np.zeros(rp.VEC_LEN))

    class FakeWorker:
        def __init__(self, pool):
            self.pool = pool

        def __call__(self, role, messages, agent_id):
            return f"reply-{agent_id}"

    class FakeRouter:
        head = torch.zeros(rp.HEAD_ROWS, rp.HIDDEN)
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def route(self, messages, sample=False):
            return {"agent_id": 0, "role_id": 1}

    monkeypatch.setattr(rp, "OpenRouterWorker", FakeWorker)
    monkeypatch.setattr(rp, "FuguRouter", FakeRouter)
    monkeypatch.setattr(
        rp, "extract_hidden_states",
        lambda router, tasks, batch_size=8: torch.zeros(len(tasks), rp.HIDDEN),
    )
    monkeypatch.setattr(
        rp, "train_head",
        lambda X, yw, yr, h0, **kwargs: (torch.zeros(rp.HEAD_ROWS, rp.HIDDEN), 0.5),
    )
    monkeypatch.setattr(rp, "tqdm", lambda x, **kw: x)

    dataset_dir = tmp_path / "terminal"
    (dataset_dir / "task0" / "solution").mkdir(parents=True)
    (dataset_dir / "task0" / "instruction.md").write_text("do x")
    (dataset_dir / "task0" / "solution" / "solve.sh").write_text("echo x")

    out_dir = tmp_path / "out"
    argv = [
        "--pool", "m0,m1",
        "--dataset", str(dataset_dir),
        "--limit", "1",
        "--epochs", "1",
        "--fugu-vector", str(vec),
        "--output-dir", str(out_dir),
        "--val-frac", "0.0",
    ]
    rp.main(argv)
    report = json.loads((out_dir / "report.json").read_text())
    assert report["train_size"] == 1


def test_load_cost_table_skips_comments(tmp_path):
    p = tmp_path / "costs.json"
    p.write_text(json.dumps({
        "_note": "comment",
        "model/a": 0.01,
        "model/b": 0.02,
    }))
    costs = rp._load_cost_table(p)
    assert costs == {"model/a": 0.01, "model/b": 0.02}


def test_pick_best_worker_quality():
    pool = ["expensive", "cheap"]
    scores = [0.1, 0.9]
    costs = {"expensive": 10.0, "cheap": 0.01}
    # quality mode picks highest score regardless of cost
    assert rp._pick_best_worker(scores, pool, "quality", costs) == 1


def test_pick_best_worker_cost_equal_scores():
    pool = ["openai/gpt-5.6-luna", "anthropic/claude-opus-5"]
    scores = [0.5, 0.5]
    costs = {
        "openai/gpt-5.6-luna": 0.0008,
        "anthropic/claude-opus-5": 0.035,
    }
    # equal scores -> cheaper model wins
    assert rp._pick_best_worker(scores, pool, "cost", costs) == 0


def test_pick_best_worker_quality_2x_score_wins_despite_price():
    pool = ["openai/gpt-5.6-luna", "anthropic/claude-opus-5"]
    # worker 1 is 2x better but 40x more expensive
    scores = [0.5, 1.0]
    costs = {
        "openai/gpt-5.6-luna": 0.0008,
        "anthropic/claude-opus-5": 0.035,
    }
    # quality mode ignores price
    assert rp._pick_best_worker(scores, pool, "quality", costs) == 1


def test_main_cost_mode(monkeypatch, tmp_path):
    vec = tmp_path / "vec.npy"
    np.save(vec, np.zeros(rp.VEC_LEN))

    class FakeWorker:
        def __init__(self, pool):
            self.pool = pool

        def __call__(self, role, messages, agent_id):
            return f"reply-{agent_id}"

    class FakeRouter:
        head = torch.zeros(rp.HEAD_ROWS, rp.HIDDEN)
        device = "cpu"

        def __init__(self, *args, **kwargs):
            pass

        def route(self, messages, sample=False):
            return {"agent_id": 0, "role_id": 1}

    monkeypatch.setattr(rp, "OpenRouterWorker", FakeWorker)
    monkeypatch.setattr(rp, "FuguRouter", FakeRouter)
    monkeypatch.setattr(
        rp, "extract_hidden_states",
        lambda router, tasks, batch_size=8: torch.zeros(len(tasks), rp.HIDDEN),
    )
    monkeypatch.setattr(
        rp, "train_head",
        lambda X, yw, yr, h0, **kwargs: (torch.zeros(rp.HEAD_ROWS, rp.HIDDEN), 0.5),
    )
    monkeypatch.setattr(rp, "tqdm", lambda x, **kw: x)

    cost_table = tmp_path / "costs.json"
    cost_table.write_text(json.dumps({
        "openai/m0": 1.0,
        "openai/m1": 0.01,
    }))

    dataset_dir = tmp_path / "terminal"
    (dataset_dir / "task0" / "solution").mkdir(parents=True)
    (dataset_dir / "task0" / "instruction.md").write_text("do x")
    (dataset_dir / "task0" / "solution" / "solve.sh").write_text("echo x")

    out_dir = tmp_path / "out"
    argv = [
        "--pool", "openai/m0,openai/m1",
        "--dataset", str(dataset_dir),
        "--label-mode", "cost",
        "--cost-table", str(cost_table),
        "--limit", "1",
        "--epochs", "1",
        "--fugu-vector", str(vec),
        "--output-dir", str(out_dir),
        "--val-frac", "0.0",
    ]
    rp.main(argv)
    report = json.loads((out_dir / "report.json").read_text())
    assert report["label_mode"] == "cost"
    assert report["cost_table"] == str(cost_table)
