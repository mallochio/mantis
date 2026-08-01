#!/usr/bin/env python3
"""Retrain the OpenFugu TRINITY router head for a new 7-slot worker pool.

Usage (local):
  export OPENROUTER_API_KEY=sk-or-v1-...
  python scripts/retrain_router_pool.py \
    --pool "anthropic/claude-haiku-4.5,anthropic/claude-sonnet-5,anthropic/claude-sonnet-4.6,anthropic/claude-sonnet-4.5,anthropic/claude-opus-4.8,anthropic/claude-opus-5,anthropic/claude-fable-5" \
    --output-dir ./outputs/router_retrain

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
  RETRAIN_WORKER_MODELS  optional override of --pool
  RETRAIN_LIMIT       optional override of --limit
  RETRAIN_EPOCHS      optional override of --epochs
"""
from __future__ import annotations
import argparse, json, os, sys, time, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from datasets import load_dataset
from tqdm.auto import tqdm

# Make OpenFugu internals importable when this script lives in fugu-local/scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "OpenFugu"))

from openfugu.mini import (
    FuguRouter, N_AGENTS, N_ROLES, HEAD_ROWS, HIDDEN, SVF_LEN, VEC_LEN,
    ROUTER_SYSTEM_PROMPT, DEFAULT_SLOT_LABELS,
)
from train.toolscale_data import _parse_plan, _score, SYSTEM


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


class OpenRouterWorker:
    """Call a heterogeneous pool through OpenRouter with litellm."""
    def __init__(self, models: list[str], api_key: str | None = None,
                 api_base: str = "https://openrouter.ai/api/v1",
                 max_tokens: int = 1024, temperature: float = 0.2,
                 timeout: int = 120):
        import litellm
        self.litellm = litellm
        self.models = [normalize_model_id(m) for m in models]
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, role: str, messages: list[dict], agent_id: int) -> str:
        model = self.models[agent_id % len(self.models)]
        kw = dict(
            model=model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            custom_llm_provider="openai",
            timeout=self.timeout,
        )
        if self.api_key:
            kw["api_key"] = self.api_key
        if self.api_base:
            kw["api_base"] = self.api_base
        try:
            r = self.litellm.completion(**kw)
            return r.choices[0].message.content or ""
        except Exception as e:
            print(f"[worker] call failed for {model}: {e}", flush=True)
            return ""


# ---------------------------------------------------------------------------
# Dataset and reward helpers
# ---------------------------------------------------------------------------

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
            records.append({"task": task, "expected": list(expected)})
        if limit and len(records) >= limit:
            break

    rng = np.random.default_rng(seed)
    rng.shuffle(records)
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]


def worker_messages(task: str) -> list[dict]:
    # Anthropic models on OpenRouter do not allow an assistant-message prefill,
    # so we end with a user message and rely on the system prompt for the format.
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]


def reward_for(completion: str, gold: list[dict]) -> float:
    pred = _parse_plan(completion)
    if pred is None:
        return 0.0
    return _score(pred, gold)


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
               device: str | None = None):
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
    best_weight = None
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
                acc_w = (logits[:, :N_AGENTS].argmax(dim=1) == y_worker[val_idx]).float().mean().item()
                acc_r = (logits[:, N_AGENTS:].argmax(dim=1) == y_role[val_idx]).float().mean().item()
            else:
                logits = head(X[train_idx])
                acc_w = (logits[:, :N_AGENTS].argmax(dim=1) == y_worker[train_idx]).float().mean().item()
                acc_r = (logits[:, N_AGENTS:].argmax(dim=1) == y_role[train_idx]).float().mean().item()
        print(f"[epoch {epoch + 1}/{epochs}] loss={epoch_loss:.4f} "
              f"val_worker_acc={acc_w:.3f} val_role_acc={acc_r:.3f}", flush=True)
        if acc_w > best_val:
            best_val = acc_w
            best_weight = head.weight.detach().cpu().clone()

    return best_weight, best_val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Retrain TRINITY router head on a new worker pool")
    ap.add_argument("--pool", default=os.environ.get("RETRAIN_WORKER_MODELS"), required=False,
                    help="Comma-separated OpenRouter worker model ids")
    ap.add_argument("--output-dir", default=os.environ.get("RETRAIN_OUTPUT_DIR", "outputs/router_retrain"))
    ap.add_argument("--dataset", default="nvidia/ToolScale")
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
    ap.add_argument("--fugu-vector", default=os.environ.get("FUGU_VECTOR", str(REPO_ROOT / "artifacts" / "model_iter_60.npy")))
    args = ap.parse_args(argv)

    if not args.pool:
        ap.error("--pool or RETRAIN_WORKER_MODELS is required")
    pool = [m.strip() for m in args.pool.split(",") if m.strip()]
    if len(pool) != N_AGENTS:
        print(f"[warn] pool has {len(pool)} models; TRINITY expects {N_AGENTS}. Padding with the last model.", flush=True)
        while len(pool) < N_AGENTS:
            pool.append(pool[-1])
    print(f"[retrain] worker pool: {pool}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load data ----------------------------------------------------------
    print(f"[retrain] loading up to {args.limit} rows from {args.dataset}...", flush=True)
    train_ds, val_ds = load_toolscale_tasks(args.limit, args.seed, args.val_frac)
    tasks = [row["task"] for row in train_ds]
    val_tasks = [row["task"] for row in val_ds]
    val_golds = [row["expected"] for row in val_ds]
    print(f"[retrain] train={len(tasks)} val={len(val_tasks)}", flush=True)

    # --- collect worker rewards --------------------------------------------
    cache_path = out_dir / args.cache
    worker = OpenRouterWorker(pool)

    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                rec = json.loads(line)
                cache[(rec["task"], rec["model"])] = rec
        print(f"[retrain] loaded {len(cache)} cached worker responses", flush=True)

    results = []
    for idx, row in enumerate(tqdm(train_ds, desc="scoring workers")):
        task = row["task"]
        gold = row["expected"]
        scores = []
        for agent_id, model in enumerate(pool):
            key = (task, model)
            if key in cache:
                completion = cache[key]["completion"]
            else:
                completion = worker("Worker", worker_messages(task), agent_id)
                cache[key] = {"task": task, "model": model, "completion": completion}
            scores.append(reward_for(completion, gold))
        best = int(np.argmax(scores))
        results.append({"task": task, "gold_worker": best, "scores": scores, "gold": gold})

    with open(cache_path, "w") as f:
        for rec in cache.values():
            f.write(json.dumps(rec) + "\n")

    # Save human-readable labels
    with open(out_dir / "train_labels.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # --- load router backbone ----------------------------------------------
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[retrain] loading TRINITY backbone {args.fugu_model} on {device}...", flush=True)
    router = FuguRouter(args.fugu_model, args.fugu_vector, device=device, seed=args.seed)

    # --- teacher role labels from current head -------------------------------
    print("[retrain] extracting hidden states and teacher role labels...", flush=True)
    X = extract_hidden_states(router, tasks)
    y_worker = torch.tensor([r["gold_worker"] for r in results], dtype=torch.long)
    y_role = torch.empty_like(y_worker)
    for i, task in enumerate(tqdm(tasks, desc="role labels")):
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": task},
        ]
        r = router.route(msgs, sample=False)
        y_role[i] = r["role_id"]

    head0 = router.head.detach().cpu().clone()

    # --- train ---------------------------------------------------------------
    print("[retrain] training head...", flush=True)
    trained_weight, best_val_acc = train_head(
        X, y_worker, y_role, head0,
        epochs=args.epochs, lr=args.lr, alpha=args.alpha, l2_lambda=args.l2, device=device,
    )

    # --- assemble new vector -------------------------------------------------
    original_vec = np.load(args.fugu_vector).astype(np.float64)
    svf = original_vec[:SVF_LEN]
    new_head = trained_weight.numpy().astype(np.float64).reshape(-1)
    new_vec = np.concatenate([svf, new_head]).astype(np.float64)
    assert new_vec.shape == (VEC_LEN,), new_vec.shape

    vec_path = out_dir / "model_iter_60.npy"
    head_path = out_dir / "router_head.npy"
    np.save(vec_path, new_vec)
    np.save(head_path, new_head)

    # --- quick validation ----------------------------------------------------
    print("[retrain] running quick validation on held-out tasks...", flush=True)
    router_val = FuguRouter(args.fugu_model, str(vec_path), device=device, seed=args.seed)
    val_hits = 0
    for task, gold in zip(val_tasks, val_golds):
        msgs = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT.format(num_agents=N_AGENTS)},
            {"role": "user", "content": task},
        ]
        # greedy route
        pred_worker = router_val.route(msgs, sample=False)["agent_id"]
        completion = worker("Worker", worker_messages(task), pred_worker)
        score = reward_for(completion, gold)
        val_hits += score
    avg_val_reward = val_hits / max(1, len(val_tasks))

    report = {
        "pool": pool,
        "dataset": args.dataset,
        "limit": args.limit,
        "train_size": len(tasks),
        "val_size": len(val_tasks),
        "best_val_worker_acc": best_val_acc,
        "avg_val_reward": avg_val_reward,
        "output": str(out_dir),
        "vector": str(vec_path),
        "head": str(head_path),
    }
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[retrain] done. Report:", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
