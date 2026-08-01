#!/usr/bin/env python3
"""Retrain the OpenFugu TRINITY router head for a new 7-slot worker pool.

Usage (local):
  export OPENROUTER_API_KEY=sk-or-v1-...
  python scripts/retrain_router_pool.py \
    --pool "<7 model specs separated by commas, optional |reasoning_effort>" \
    --output-dir ./outputs/router_retrain

  Example pool: anthropic/claude-sonnet-5|medium, anthropic/claude-opus-5|medium,
  openai/gpt-5.6-sol|medium, openai/gpt-5.6-luna|max, openai/gpt-5.6-terra|xhigh,
  deepseek/deepseek-v4-flash|none, z-ai/glm-5.2|none

Usage (SkyPilot):
  sky launch launch/sky/retrain_fugu_router.yaml

What it does:
  1. Downloads a task dataset (default nvidia/ToolScale) and a small validation split.
  2. Calls each worker in the pool for each task through OpenRouter (LiteLLM
     OpenAI-compatible endpoint). Responses are scored against the expected
     tool-call plan from ToolScale; the highest-scoring worker becomes the
     gold worker label for that task.
  3. Runs the Qwen3-0.6B TRINITY backbone to extract penultimate-token hidden
     states for each task.
  4. Fine-tunes the 10x1024 linear head (7 worker logits + 3 role logits) with
     cross-entropy on the gold worker/role labels while regularizing toward the
     original head (L2).
  5. Writes a new model_iter_60.npy (SVF offsets + trained head) and a
     router_head.npy (head only) to the output directory.

Environment:
  OPENROUTER_API_KEY  required for worker calls
  HF_TOKEN            optional, avoids HF rate limits / gates Qwen3-0.6B
  FUGU_MODEL          Qwen3-0.6B dir or HF id (default Qwen/Qwen3-0.6B)
  FUGU_VECTOR         existing TRINITY vector (default ./artifacts/model_iter_60.npy)
  RETRAIN_WORKER_MODELS  optional override of --pool. Each entry can append
                         '|reasoning_effort' (e.g. 'openai/gpt-5.6-terra|xhigh').
  RETRAIN_LIMIT       optional override of --limit
  RETRAIN_EPOCHS      optional override of --epochs
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch import nn
from tqdm.auto import tqdm

# Make OpenFugu internals importable when this script lives in fugu-local/scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenFugu"))

from openfugu.mini import (
    HEAD_ROWS,
    HIDDEN,
    N_AGENTS,
    ROUTER_SYSTEM_PROMPT,
    SVF_LEN,
    VEC_LEN,
    FuguRouter,
)
from train.toolscale_data import SYSTEM, _parse_plan, _score

# ---------------------------------------------------------------------------
# OpenRouter worker wrapper (explicit OpenAI-compatible provider)
# ---------------------------------------------------------------------------

KNOWN_PREFIXES = {
    "claude-": "anthropic/",
    "gpt-": "openai/",
    "gemini-": "google/",
    "deepseek-": "deepseek/",
    "minimax-": "minimax/",
    "glm-": "z-ai/",
    "kimi-": "moonshotai/",
    "qwen": "qwen/",
    "mimo-": "xiaomi/",
    "gemma-": "google/",
}


def split_model_spec(spec: str):
    """Parse 'model_id|reasoning_effort' into (model_id, effort).

    Full OpenRouter ids like 'openai/gpt-5.6-sol|medium' are supported.
    Effort is optional; None means no reasoning_effort is sent.
    """
    spec = spec.strip()
    effort: str | None = None
    if "|" in spec:
        spec, effort = spec.rsplit("|", 1)
        effort = effort.strip() or None
    model = spec.strip()
    return model, effort


def normalize_model_id(model: str) -> str:
    """Turn an alias or openrouter/... id into a full OpenRouter model id."""
    model = model.strip()
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    if "/" in model:
        return model
    for prefix, provider in KNOWN_PREFIXES.items():
        if model.lower().startswith(prefix):
            return provider + model
    raise ValueError(
        f"Could not infer OpenRouter provider for '{model}'. "
        f"Pass a full id like 'anthropic/claude-sonnet-5' or 'openrouter/...'."
    )


def _is_reasoning_model(model: str) -> bool:
    """Heuristic for models configured with reasoning_effort != none.

    OpenRouter/LiteLLM reject temperature != 1 for these reasoning models.
    """
    return "claude-" in model or "gpt-5.6-" in model


class OpenRouterWorker:
    """Call a heterogeneous pool through OpenRouter with litellm."""
    def __init__(self, models: list[str], api_key: str | None = None,
                 api_base: str = "https://openrouter.ai/api/v1",
                 max_tokens: int = 1024, temperature: float = 0.2,
                 timeout: int = 120):
        import litellm
        self.litellm = litellm
        specs = [split_model_spec(m) for m in models]
        self.models = [normalize_model_id(m) for m, _ in specs]
        self.efforts = [e for _, e in specs]
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, role: str, messages: list[dict], agent_id: int) -> str:
        model = self.models[agent_id % len(self.models)]
        effort = self.efforts[agent_id % len(self.efforts)]
        kw = {
            "model": f"openai/{model}",
            "messages": messages,
            "max_tokens": self.max_tokens,
            "custom_llm_provider": "openai",
            "timeout": self.timeout,
            # reasoning_effort is an extra param for LiteLLM's openai provider;
            # tell LiteLLM to allow it through to OpenRouter.
            "allowed_openai_params": ["reasoning_effort"],
        }
        # Reasoning models (claude-*, gpt-5.6-*) require temperature=1 or omitted.
        if not _is_reasoning_model(model):
            kw["temperature"] = self.temperature
        if effort:
            kw["reasoning_effort"] = effort
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        try:
            r = self.litellm.completion(**kw)
            return str(r.choices[0].message.content or "")
        except Exception as e:  # noqa: BLE001
            print(f"[worker] call failed for {model}: {e}", flush=True)
            return ""


# ---------------------------------------------------------------------------
# Dataset and reward helpers
# ---------------------------------------------------------------------------

TERMINAL_SYSTEM = (
    "You are an autonomous terminal agent. Given a task instruction, output the "
    "shell commands and minimal explanation needed to solve it. Be concise."
)


TERMINAL_PATTERNS = ("/instruction.md", "/task.toml", "/solution/solve.sh")


def _sync_s3_to_local(s3_uri: str, local_dir: Path):
    """Mirror only the small TerminalBench metadata files from S3."""
    import boto3
    m = re.match(r"s3://([^/]+)/?(.*)", s3_uri)
    if not m:
        raise ValueError(f"invalid S3 URI: {s3_uri}")
    bucket, prefix = m.group(1), m.group(2).rstrip("/")
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    local_dir.mkdir(parents=True, exist_ok=True)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not any(key.endswith(s) for s in TERMINAL_PATTERNS):
                continue
            rel = Path(key).relative_to(prefix)
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))


def load_terminalbench_tasks(
    dataset: str = "zai-org/terminal-bench-2-verified",
    limit: int | None = None,
    seed: int = 42,
    val_frac: float = 0.1,
    cache_dir: str | Path | None = None,
):
    """Load TerminalBench 2.1 task prompts and reference solution scripts.

    Only the small metadata files are downloaded (instruction.md, task.toml,
    solution/solve.sh); the heavy environment tarballs are skipped.
    """
    local_dir = Path(cache_dir or os.path.expanduser("~/.cache/terminal-bench-2.1"))

    if dataset.startswith("s3://"):
        _sync_s3_to_local(dataset, local_dir)
        local = local_dir
    elif Path(dataset).exists():
        local = Path(dataset)
    else:
        local = Path(
            snapshot_download(
                dataset,
                repo_type="dataset",
                local_dir=str(local_dir),
                allow_patterns=[f"*{s}" for s in TERMINAL_PATTERNS],
            )
        )

    records = []
    for instr_path in sorted(local.glob("*/instruction.md")):
        task_dir = instr_path.parent
        solve_path = task_dir / "solution" / "solve.sh"
        if not solve_path.exists():
            continue
        instruction = instr_path.read_text().strip()
        reference = solve_path.read_text().strip()
        records.append({
            "task": instruction,
            "expected": reference,
            "system": TERMINAL_SYSTEM,
            "name": task_dir.name,
        })

    rng = np.random.default_rng(seed)
    rng.shuffle(records)  # type: ignore[arg-type]
    if limit:
        records = records[:limit]
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]


def load_toolscale_tasks(limit: int, seed: int = 42, val_frac: float = 0.1):
    ds = load_dataset("nvidia/ToolScale", split="train")
    ds = ds.shuffle(seed=seed)

    records = []
    for row in ds:
        us = row.get("user_scenario") or {}
        instr = us.get("instructions") or {}
        task = instr.get("task_instructions") or instr.get("reason_for_call") or ""
        ec = row.get("evaluation_criteria") or {}
        expected = ec.get("actions") or []
        if task and expected:
            records.append({"task": task, "expected": list(expected), "system": SYSTEM})
        if limit and len(records) >= limit:
            break

    rng = np.random.default_rng(seed)
    rng.shuffle(records)  # type: ignore[arg-type]
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]


def worker_messages(task: str, system: str | None = None) -> list[dict]:
    # Anthropic models on OpenRouter do not allow an assistant-message prefill,
    # so we end with a user message and rely on the system prompt for the format.
    return [
        {"role": "system", "content": system or SYSTEM},
        {"role": "user", "content": task},
    ]


def reward_for(completion: str, gold: str | list) -> float:
    # TerminalBench gold is a reference shell script; use a cheap, deterministic
    # similarity as a routing reward. ToolScale gold is a list of tool-call dicts.
    if isinstance(gold, str):
        if not completion or not gold:
            return 0.0
        return float(difflib.SequenceMatcher(None, completion, gold).ratio())
    pred = _parse_plan(completion)
    if pred is None:
        return 0.0
    return float(_score(pred, gold))


# ---------------------------------------------------------------------------
# Hidden-state extraction
# ---------------------------------------------------------------------------

def extract_hidden_states(router: FuguRouter, tasks: list[str], batch_size: int = 8):
    """Return (H,)-shaped hidden-state tensors for each task, no gradient."""
    router.model.eval()
    hidden = []
    for task in tqdm(tasks, desc="hidden states"):
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": task},
        ]
        with torch.no_grad():
            h = router._hidden(msgs)
            if isinstance(h, np.ndarray):
                h = torch.from_numpy(h)
            hidden.append(h.detach().cpu().float())
    return torch.stack(hidden, dim=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_head(X: torch.Tensor, y_worker: torch.Tensor, y_role: torch.Tensor,
               head0: torch.Tensor, epochs: int = 30, lr: float = 1e-3,
               alpha: float = 0.1, l2_lambda: float = 0.01,
               device: str | None = None) -> tuple[torch.Tensor, float]:
    """Fine-tune the shared 10x1024 head."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    X = X.to(device)
    y_worker = y_worker.to(device)
    y_role = y_role.to(device)
    head0 = head0.to(device)

    head = nn.Linear(HIDDEN, HEAD_ROWS, bias=False).to(device)
    with torch.no_grad():
        head.weight.copy_(head0)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    n_total = len(y_worker)
    n_train = max(1, int(0.9 * n_total))
    indices = torch.randperm(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    best_val = -1.0
    best_weight = head.weight.detach().cpu().clone()
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        for i in range(0, len(train_idx), 32):
            idx = train_idx[perm[i:i + 32]]
            logits = head(X[idx])
            loss_w = F.cross_entropy(logits[:, :N_AGENTS], y_worker[idx])
            loss_r = F.cross_entropy(logits[:, N_AGENTS:], y_role[idx])
            loss = loss_w + alpha * loss_r + l2_lambda * torch.mean((head.weight - head0) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        head.eval()
        with torch.no_grad():
            if len(val_idx) > 0:
                logits = head(X[val_idx])
                acc_w = (logits[:, :N_AGENTS].argmax(dim=1) == y_worker[val_idx]).float().mean()
                acc_r = (logits[:, N_AGENTS:].argmax(dim=1) == y_role[val_idx]).float().mean()
            else:
                logits = head(X[train_idx])
                acc_w = (logits[:, :N_AGENTS].argmax(dim=1) == y_worker[train_idx]).float().mean()
                acc_r = (logits[:, N_AGENTS:].argmax(dim=1) == y_role[train_idx]).float().mean()
            acc_w = acc_w.item()
            acc_r = acc_r.item()
        print(f"[epoch {epoch + 1}/{epochs}] loss={epoch_loss:.4f} "
              f"val_worker_acc={acc_w:.3f} val_role_acc={acc_r:.3f}", flush=True)
        if acc_w > best_val:
            best_val = acc_w
            best_weight = head.weight.detach().cpu().clone()

    return best_weight, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_retrain_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain TRINITY router head on a new worker pool")
    ap.add_argument("--pool", default=os.environ.get("RETRAIN_WORKER_MODELS"), required=False,
                    help="Comma-separated OpenRouter worker model ids. "
                         "Append '|reasoning_effort' per model, e.g. openai/gpt-5.6-terra|xhigh")
    ap.add_argument(
        "--output-dir",
        default=os.environ.get("RETRAIN_OUTPUT_DIR", "outputs/router_retrain"),
    )
    ap.add_argument("--dataset", default="nvidia/ToolScale",
                    help="Dataset to train on. 'nvidia/ToolScale' (default) or a "
                         "TerminalBench 2.1 HF repo such as 'zai-org/terminal-bench-2-verified'.")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("RETRAIN_LIMIT", "200")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("RETRAIN_EPOCHS", "30")))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.1, help="role-loss weight")
    ap.add_argument("--l2", type=float, default=0.01, help="L2 regularization toward original head")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default="worker_outputs.jsonl",
                    help="Cache worker responses to avoid re-calling the API")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fugu-model", default=os.environ.get("FUGU_MODEL", "Qwen/Qwen3-0.6B"))
    default_vec = str(REPO_ROOT / "artifacts" / "model_iter_60.npy")
    ap.add_argument("--fugu-vector", default=os.environ.get("FUGU_VECTOR", default_vec))
    args = ap.parse_args(argv)

    if not args.pool:
        ap.error("--pool or RETRAIN_WORKER_MODELS is required")
    return args


def _prepare_pool(pool_csv: str) -> list[str]:
    pool = [m.strip() for m in pool_csv.split(",") if m.strip()]
    if len(pool) != N_AGENTS:
        print(
            f"[warn] pool has {len(pool)} models; TRINITY expects {N_AGENTS}. "
            "Padding with the last model.",
            flush=True,
        )
        while len(pool) < N_AGENTS:
            pool.append(pool[-1])
    print(f"[retrain] worker pool: {pool}", flush=True)
    return pool


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def _load_retrain_data(args: argparse.Namespace):
    print(f"[retrain] loading up to {args.limit} rows from {args.dataset}...", flush=True)
    if "terminal" in args.dataset.lower():
        train_ds, val_ds = load_terminalbench_tasks(
            args.dataset, args.limit, args.seed, args.val_frac
        )
    else:
        train_ds, val_ds = load_toolscale_tasks(args.limit, args.seed, args.val_frac)
    print(f"[retrain] train={len(train_ds)} val={len(val_ds)}", flush=True)
    return train_ds, val_ds


def _load_worker_cache(cache_path: Path) -> dict[tuple[str, str], dict]:
    cache: dict[tuple[str, str], dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cache[(rec["task"], rec["model"])] = rec
        print(f"[retrain] loaded {len(cache)} cached worker responses", flush=True)
    return cache


def _write_worker_cache(cache_path: Path, cache: dict[tuple[str, str], dict]) -> None:
    with open(cache_path, "w") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in cache.values())


def _score_one_worker(
    worker: OpenRouterWorker,
    task: str,
    system: str | None,
    agent_id: int,
    model: str,
    cache: dict[tuple[str, str], dict],
) -> str:
    key = (task, model)
    if key in cache:
        return str(cache[key]["completion"])
    completion = worker("Worker", worker_messages(task, system), agent_id)
    cache[key] = {"task": task, "model": model, "completion": completion}
    return completion


def _score_worker_pool(
    worker: OpenRouterWorker,
    pool: list[str],
    train_ds: list[dict],
    cache_path: Path,
) -> list[dict]:
    cache = _load_worker_cache(cache_path)
    results = []
    for row in tqdm(train_ds, desc="scoring workers"):
        task = row["task"]
        gold = row["expected"]
        system = row.get("system")
        scores = [
            reward_for(_score_one_worker(worker, task, system, agent_id, model, cache), gold)
            for agent_id, model in enumerate(pool)
        ]
        best = int(np.argmax(scores))
        results.append({"task": task, "gold_worker": best, "scores": scores, "gold": gold})
    _write_worker_cache(cache_path, cache)
    return results


def _write_labels(out_dir: Path, results: list[dict]) -> None:
    with open(out_dir / "train_labels.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in results)


def _teacher_role_labels(router: FuguRouter, tasks: list[str]) -> torch.Tensor:
    y_role = torch.empty(len(tasks), dtype=torch.long)
    for i, task in enumerate(tqdm(tasks, desc="role labels")):
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": task},
        ]
        y_role[i] = router.route(msgs, sample=False)["role_id"]
    return y_role


def _train_router_head(
    router: FuguRouter,
    tasks: list[str],
    results: list[dict],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, float]:
    print("[retrain] extracting hidden states and teacher role labels...", flush=True)
    X = extract_hidden_states(router, tasks)
    y_worker = torch.tensor([r["gold_worker"] for r in results], dtype=torch.long)
    y_role = _teacher_role_labels(router, tasks)
    head0 = router.head.detach().cpu().clone()
    print("[retrain] training head...", flush=True)
    return train_head(
        X, y_worker, y_role, head0,
        epochs=args.epochs, lr=args.lr, alpha=args.alpha, l2_lambda=args.l2,
        device=str(router.device),
    )


def _write_trained_vector(
    trained_weight: torch.Tensor,
    original_vec: np.ndarray,
    out_dir: Path,
) -> tuple[Path, Path]:
    svf = original_vec[:SVF_LEN]
    new_head = trained_weight.numpy().astype(np.float64).reshape(-1)
    new_vec = np.concatenate([svf, new_head]).astype(np.float64)
    assert new_vec.shape == (VEC_LEN,), new_vec.shape
    vec_path = out_dir / "model_iter_60.npy"
    head_path = out_dir / "router_head.npy"
    np.save(vec_path, new_vec)
    np.save(head_path, new_head)
    return vec_path, head_path


def _validate_router(
    router_val: FuguRouter,
    val_rows: list[dict],
    worker: OpenRouterWorker,
) -> float:
    print("[retrain] running quick validation on held-out tasks...", flush=True)
    val_hits = 0.0
    for row in val_rows:
        task, gold, system = row["task"], row["expected"], row.get("system")
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": task},
        ]
        pred_worker = router_val.route(msgs, sample=False)["agent_id"]
        completion = worker("Worker", worker_messages(task, system), pred_worker)
        val_hits += reward_for(completion, gold)
    return val_hits / max(1, len(val_rows))


def _build_report(
    args: argparse.Namespace,
    pool: list[str],
    train_size: int,
    val_size: int,
    best_val_acc: float,
    avg_val_reward: float,
    out_dir: Path,
    vec_path: Path,
    head_path: Path,
) -> dict:
    return {
        "pool": pool,
        "dataset": args.dataset,
        "limit": args.limit,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_worker_acc": best_val_acc,
        "avg_val_reward": avg_val_reward,
        "output": str(out_dir),
        "vector": str(vec_path),
        "head": str(head_path),
    }


def main(argv=None) -> None:
    args = _parse_retrain_args(argv)
    pool = _prepare_pool(args.pool)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds = _load_retrain_data(args)
    cache_path = out_dir / args.cache
    worker = OpenRouterWorker(pool)
    results = _score_worker_pool(worker, pool, train_ds, cache_path)
    _write_labels(out_dir, results)

    device = _resolve_device(args.device)
    print(f"[retrain] loading TRINITY backbone {args.fugu_model} on {device}...", flush=True)
    router = FuguRouter(args.fugu_model, args.fugu_vector, device=device, seed=args.seed)
    tasks = [r["task"] for r in results]
    trained_weight, best_val_acc = _train_router_head(router, tasks, results, args)

    original_vec = np.load(args.fugu_vector).astype(np.float64)
    vec_path, head_path = _write_trained_vector(trained_weight, original_vec, out_dir)

    router_val = FuguRouter(args.fugu_model, str(vec_path), device=device, seed=args.seed)
    avg_val_reward = _validate_router(router_val, val_ds, worker)

    report = _build_report(
        args, pool, len(results), len(val_ds), best_val_acc, avg_val_reward,
        out_dir, vec_path, head_path,
    )
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[retrain] done. Report:", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
