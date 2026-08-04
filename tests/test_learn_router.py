from __future__ import annotations

import json
from types import SimpleNamespace

import learn_router as learn
import numpy as np
import torch


def _record(task: str, label: int, pool: list[str], timestamp: int = 1) -> dict:
    return {
        "schema_version": 1,
        "timestamp": timestamp,
        "task": task,
        "task_hash": f"{timestamp:08x}" + "0" * 56,
        "pool": pool,
        "trainable": True,
        "label_worker": label,
    }


def test_load_records_filters_and_deduplicates(tmp_path):
    pool = [f"m{i}" for i in range(learn.N_AGENTS)]
    old = _record("task", 1, pool, 1)
    new = {**old, "timestamp": 2, "label_worker": 2}
    invalid = {**old, "task_hash": "bad", "trainable": False}
    (tmp_path / "runs-a.jsonl").write_text(
        "not-json\n" + "\n".join(json.dumps(row) for row in (old, invalid, new))
    )
    (tmp_path / "runs-b.jsonl").write_text(json.dumps({**old, "pool": ["other"]}))
    rows = learn.load_records(tmp_path, pool)
    assert len(rows) == 1 and rows[0]["label_worker"] == 2
    assert learn.dataset_digest(rows) == learn.dataset_digest(rows)


def test_split_accuracy_and_training():
    records = [
        {"task_hash": f"{i:08x}" + "0" * 56, "label_worker": i % learn.N_AGENTS} for i in range(10)
    ]
    train, val = learn.split_records(records)
    assert train and val and len(train) + len(val) == len(records)

    features = torch.zeros(4, learn.HIDDEN)
    features[:, 0] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    labels = torch.tensor([0, 1, 0, 1])
    weight = torch.zeros(learn.HEAD_ROWS, learn.HIDDEN)
    weight[0, 0], weight[1, 0] = 1, -1
    assert learn.accuracy(weight, features, labels) == 1.0
    candidate = learn.train_candidate(
        features, labels, torch.zeros_like(weight), 2, 0.01, 0.01, "cpu"
    )
    assert candidate.shape == weight.shape


def test_write_candidate_and_promote(tmp_path):
    source = tmp_path / "source.npy"
    np.save(source, np.zeros(learn.VEC_LEN))
    weight = torch.ones(learn.HEAD_ROWS, learn.HIDDEN)
    output = learn.write_candidate(tmp_path, source, weight, {"ok": True}, promote=True)
    assert (output / "model_iter_60.npy").exists()
    assert (tmp_path / "promoted" / "router_head.npy").exists()
    assert json.loads((tmp_path / "promoted" / "report.json").read_text()) == {"ok": True}


def test_train_once_waiting_unchanged_and_promotes(tmp_path, monkeypatch):
    pool = [f"m{i}" for i in range(learn.N_AGENTS)]
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    monkeypatch.setenv("MANTIS_WORKER_MODELS", ",".join(pool))
    assert learn.train_once(SimpleNamespace(min_runs=2))["status"] == "waiting"

    records = [_record(f"task-{i}", i % learn.N_AGENTS, pool, i) for i in range(1, 11)]
    monkeypatch.setattr(learn, "load_records", lambda *_args: records)
    vector = tmp_path / "vector.npy"
    np.save(vector, np.zeros(learn.VEC_LEN))
    monkeypatch.setenv("MANTIS_VECTOR", str(vector))

    class FakeRouter:
        def __init__(self, *_args, **_kwargs):
            self.head = torch.zeros(learn.HEAD_ROWS, learn.HIDDEN)
            self.device = torch.device("cpu")

    monkeypatch.setattr(learn, "FuguRouter", FakeRouter)
    monkeypatch.setattr(
        learn, "hidden_states", lambda _router, rows: torch.zeros(len(rows), learn.HIDDEN)
    )
    monkeypatch.setattr(learn, "train_candidate", lambda _x, _y, initial, *_a: initial + 1)
    scores = iter([0.1, 0.8])
    monkeypatch.setattr(learn, "accuracy", lambda *_a: next(scores))
    args = SimpleNamespace(
        min_runs=2,
        min_validation=1,
        min_improvement=0.02,
        epochs=1,
        lr=0.01,
        l2=0.01,
        promote=True,
    )
    result = learn.train_once(args)
    assert result["status"] == "promoted"
    assert (tmp_path / "promoted" / "model_iter_60.npy").exists()
    assert learn.train_once(args)["status"] == "unchanged"


def test_parse_args_and_main_lock(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MANTIS_LEARNING_DIR", str(tmp_path))
    args = learn.parse_args(["--min-runs", "3", "--epochs", "2"])
    assert args.min_runs == 3 and args.epochs == 2
    monkeypatch.setattr(learn, "train_once", lambda _args: {"status": "waiting"})
    learn.main(["--min-runs", "3"])
    assert '"status": "waiting"' in capsys.readouterr().out
    lock = tmp_path / "trainer.lock"
    lock.write_text("busy")
    learn.main([])
    assert "already running" in capsys.readouterr().out
