#!/usr/bin/env python3
"""Train a guarded TRINITY router candidate from opt-in local run telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT / "OpenFugu" / "openfugu", ROOT / "openfugu"):
    if candidate.exists():
        sys.path.append(str(candidate))
        break

from mini import HEAD_ROWS, HIDDEN, N_AGENTS, ROUTER_SYSTEM_PROMPT, SVF_LEN, VEC_LEN, FuguRouter


def learning_dir() -> Path:
    return Path(
        os.path.expanduser(os.environ.get("MANTIS_LEARNING_DIR", "~/.local/share/mantis/learning"))
    )


def worker_pool() -> list[str]:
    raw = os.environ.get("MANTIS_WORKER_MODELS", "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_records(root: Path, pool: list[str]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("runs-*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            label = record.get("label_worker")
            if (
                record.get("schema_version") != 1
                or not record.get("trainable")
                or record.get("pool") != pool
                or not isinstance(record.get("task"), str)
                or not isinstance(label, int)
                or not 0 <= label < N_AGENTS
            ):
                continue
            key = str(
                record.get("task_hash") or hashlib.sha256(record["task"].encode()).hexdigest()
            )
            if key not in latest or record.get("timestamp", 0) >= latest[key].get("timestamp", 0):
                latest[key] = record
    return sorted(latest.values(), key=lambda item: (item.get("timestamp", 0), item["task_hash"]))


def dataset_digest(records: list[dict[str, Any]]) -> str:
    payload = "\n".join(f"{row['task_hash']}:{row['label_worker']}" for row in records)
    return hashlib.sha256(payload.encode()).hexdigest()


def split_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for row in records:
        bucket = int(str(row["task_hash"])[:8], 16) % 5
        (val if bucket == 0 else train).append(row)
    return train, val


def hidden_states(router: FuguRouter, records: list[dict[str, Any]]) -> torch.Tensor:
    values = []
    router.model.eval()
    for row in records:
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": row["task"]},
        ]
        with torch.no_grad():
            hidden = router._hidden(messages)
            if isinstance(hidden, np.ndarray):
                hidden = torch.from_numpy(hidden)
            values.append(hidden.detach().cpu().float())
    return torch.stack(values)


def accuracy(weight: torch.Tensor, features: torch.Tensor, labels: torch.Tensor) -> float:
    if len(labels) == 0:
        return 0.0
    with torch.no_grad():
        predictions = F.linear(features, weight)[:, :N_AGENTS].argmax(dim=1)
    return float((predictions == labels).float().mean())


def train_candidate(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial: torch.Tensor,
    epochs: int,
    lr: float,
    l2: float,
    device: str,
) -> torch.Tensor:
    layer = nn.Linear(HIDDEN, HEAD_ROWS, bias=False).to(device)
    initial_device = initial.to(device)
    with torch.no_grad():
        layer.weight.copy_(initial_device)
    x = features.to(device)
    y = labels.to(device)
    optimizer = torch.optim.Adam(layer.parameters(), lr=lr)
    for _ in range(epochs):
        logits = layer(x)[:, :N_AGENTS]
        loss = F.cross_entropy(logits, y) + l2 * torch.mean((layer.weight - initial_device) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return layer.weight.detach().cpu()


def write_candidate(
    root: Path,
    source_vector: Path,
    weight: torch.Tensor,
    report: dict[str, Any],
    promote: bool,
) -> Path:
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{socket.gethostname()}"
    output = root / "candidates" / stamp
    output.mkdir(parents=True, exist_ok=False)
    vector = np.load(source_vector)
    head = weight.numpy().astype(np.float64).reshape(-1)
    merged = np.concatenate([vector[:SVF_LEN], head]).astype(np.float64)
    if merged.shape != (VEC_LEN,):
        raise ValueError(f"invalid candidate vector shape: {merged.shape}")
    np.save(output / "router_head.npy", head)
    np.save(output / "model_iter_60.npy", merged)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if promote:
        promoted = root / "promoted"
        promoted.mkdir(parents=True, exist_ok=True)
        for name in ("router_head.npy", "model_iter_60.npy", "report.json"):
            source = output / name
            temporary = promoted / f".{name}.tmp"
            temporary.write_bytes(source.read_bytes())
            temporary.replace(promoted / name)
    return output


def train_once(args: argparse.Namespace) -> dict[str, Any]:
    root = learning_dir()
    pool = worker_pool()
    if len(pool) != N_AGENTS:
        return {"status": "waiting", "reason": f"expected {N_AGENTS} worker models", "records": 0}
    records = load_records(root, pool)
    if len(records) < args.min_runs:
        return {
            "status": "waiting",
            "reason": "not enough high-confidence runs",
            "records": len(records),
        }
    train, val = split_records(records)
    if len(train) < 2 or len(val) < args.min_validation:
        return {"status": "waiting", "reason": "not enough held-out runs", "records": len(records)}
    digest = dataset_digest(records)
    state_path = root / "state.json"
    if state_path.exists():
        try:
            if json.loads(state_path.read_text()).get("dataset_digest") == digest:
                return {"status": "unchanged", "records": len(records)}
        except (json.JSONDecodeError, OSError):
            pass

    model = os.environ.get("MANTIS_MODEL", "Qwen/Qwen3-0.6B")
    vector = Path(
        os.path.expanduser(
            os.environ.get("MANTIS_VECTOR", str(ROOT / "artifacts/model_iter_60.npy"))
        )
    )
    device = os.environ.get("MANTIS_LEARNING_DEVICE", "cpu")
    router = FuguRouter(model, str(vector), device=device, seed=0)
    all_features = hidden_states(router, train + val)
    train_features, val_features = all_features[: len(train)], all_features[len(train) :]
    train_labels = torch.tensor([row["label_worker"] for row in train], dtype=torch.long)
    val_labels = torch.tensor([row["label_worker"] for row in val], dtype=torch.long)
    initial = router.head.detach().cpu().float()
    baseline = accuracy(initial, val_features, val_labels)
    candidate = train_candidate(
        train_features, train_labels, initial, args.epochs, args.lr, args.l2, device
    )
    candidate_accuracy = accuracy(candidate, val_features, val_labels)
    promoted = candidate_accuracy >= baseline + args.min_improvement
    report = {
        "dataset_digest": digest,
        "records": len(records),
        "train_records": len(train),
        "validation_records": len(val),
        "baseline_accuracy": baseline,
        "candidate_accuracy": candidate_accuracy,
        "min_improvement": args.min_improvement,
        "promoted": promoted,
        "pool": pool,
    }
    output = write_candidate(root, vector, candidate, report, promote=promoted and args.promote)
    report["output"] = str(output)
    root.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"status": "promoted" if promoted and args.promote else "candidate", **report}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="recheck periodically")
    parser.add_argument(
        "--promote", action="store_true", help="promote only candidates that pass the gate"
    )
    parser.add_argument(
        "--min-runs", type=int, default=int(os.environ.get("MANTIS_LEARNING_MIN_RUNS", "50"))
    )
    parser.add_argument("--min-validation", type=int, default=5)
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=float(os.environ.get("MANTIS_LEARNING_MIN_IMPROVEMENT", "0.02")),
    )
    parser.add_argument(
        "--epochs", type=int, default=int(os.environ.get("MANTIS_LEARNING_EPOCHS", "20"))
    )
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--l2", type=float, default=0.02)
    parser.add_argument(
        "--interval", type=int, default=int(os.environ.get("MANTIS_LEARNING_INTERVAL", "300"))
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    lock = learning_dir() / "trainer.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    stale_after = int(os.environ.get("MANTIS_LEARNING_LOCK_TTL", "14400"))
    if lock.exists() and time.time() - lock.stat().st_mtime > stale_after:
        lock.unlink(missing_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(descriptor, f"{socket.gethostname()}:{os.getpid()}".encode())
        os.close(descriptor)
    except FileExistsError:
        print("[mantis-learning] trainer already running", flush=True)
        return
    try:
        while True:
            lock.touch()
            try:
                print(
                    f"[mantis-learning] {json.dumps(train_once(args), sort_keys=True)}", flush=True
                )
            except Exception as error:  # noqa: BLE001
                print(f"[mantis-learning] training skipped: {error}", flush=True)
            if not args.watch:
                break
            time.sleep(max(10, args.interval))
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
