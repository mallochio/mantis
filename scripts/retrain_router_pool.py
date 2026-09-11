#!/usr/bin/env python3
"""Retrain the OpenFugu TRINITY router head for a new 7-slot worker pool.

Usage (local):
  export LITELLM_API_KEY=...
  python scripts/retrain_router_pool.py \
    --pool "<7 model specs separated by commas, optional |reasoning_effort>" \
    --output-dir ./outputs/router_retrain

  Example pool: google/gemini-3.6-flash|high,openai/gpt-5.6-luna|max,
  openai/gpt-5.6-sol|medium,deepseek/deepseek-v4-flash-0731|max,
  anthropic/claude-opus-5|medium,anthropic/claude-sonnet-5|medium,
  google/gemini-3.1-pro-preview|high

Usage (SkyPilot):

What it does:
  1. Downloads a task dataset (default TerminalBench 2.1 mirror) and a small
     validation split.
  2. Calls each worker in the pool for each task through the local LiteLLM proxy. Responses
     are scored against the reference solution; the best worker becomes the gold
     worker label for that task.
  3. Runs the Qwen3-0.6B TRINITY backbone to extract penultimate-token hidden
     states for each task.
  4. Fine-tunes the 10x1024 linear head (7 worker logits + 3 role logits) with
     cross-entropy on the gold worker/role labels while regularizing toward the
     original head (L2).
  5. Writes a new model_iter_60.npy (SVF offsets + trained head) and a
     router_head.npy (head only) to the output directory.

Label modes:
  quality  - pick the worker with the highest reward score.
  cost     - pick the worker with the best reward / cost ratio.
  budgeted - pick the cheapest worker whose score is within
             --quality-tolerance of the best successful score. For binary
             outcome tasks, a worker must pass (score ~1) or the task is
             skipped.

Environment:
  LITELLM_API_KEY     required for worker calls
  LITELLM_BASE_URL     OpenAI-compatible endpoint (default http://127.0.0.1:8080/v1)
  HF_TOKEN            optional, avoids HF rate limits / gates Qwen3-0.6B
  MANTIS_MODEL          Qwen3-0.6B dir or HF id (default Qwen/Qwen3-0.6B)
  MANTIS_VECTOR         existing TRINITY vector (default ./artifacts/model_iter_60.npy)
  RETRAIN_WORKER_MODELS  optional override of --pool. Each entry can append
                         '|reasoning_effort' (e.g. 'openai/gpt-5.6-terra|xhigh').
  RETRAIN_LIMIT       optional override of --limit
  RETRAIN_EPOCHS      optional override of --epochs
  RETRAIN_LABEL_MODE  default "budgeted"
  RETRAIN_QUALITY_TOLERANCE  default 0.05
  RETRAIN_MAX_WORKER_CONCURRENCY  default 3
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import nn
from tqdm.auto import tqdm

# Make API mode internals importable when this script lives in mantis/scripts.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "apps" / "api"))

from mini import (
    HEAD_ROWS,
    HIDDEN,
    N_AGENTS,
    ROUTER_SYSTEM_PROMPT,
    SVF_LEN,
    VEC_LEN,
    FuguRouter,
)
from toolscale_data import SYSTEM, _parse_plan, _score

# ---------------------------------------------------------------------------
# LiteLLM worker wrapper
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

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


def split_model_spec(spec: str):
    """Parse 'model_id|reasoning_effort' into (model_id, effort).

    LiteLLM model ids like 'gpt-5.6-sol|medium' are supported.
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
    """Legacy normalizer for existing cost tables and compatibility tests."""
    model = model.strip()
    if model.startswith("openrouter/"):
        return model[len("openrouter/") :]
    if "/" in model:
        return model
    for prefix, provider in KNOWN_PREFIXES.items():
        if model.lower().startswith(prefix):
            return provider + model
    raise ValueError(
        f"Could not infer provider for '{model}'. Pass a full id like 'anthropic/claude-sonnet-5'."
    )


def _is_reasoning_model(model: str) -> bool:
    """Heuristic for legacy reasoning models.

    Kept for backward-compatible tests; runtime reasoning detection uses the
    configured reasoning_effort value.
    """
    return "claude-" in model or "gpt-5.6-" in model


def _error_status_code(exc: Exception) -> int | None:
    """Best-effort HTTP status code extraction from common exception types."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _is_transient_error(exc: Exception) -> bool:
    """Return True if the exception is a retryable transient failure."""
    status = _error_status_code(exc)
    if isinstance(status, int) and status in (408, 409, 429, 500, 502, 503, 504):
        return True
    name = type(exc).__name__.lower()
    if "timeout" in name or ("rate" in name and "limit" in name):
        return True
    msg = str(exc).lower()
    for token in (
        "timeout",
        "408",
        "409",
        "429",
        "502",
        "503",
        "504",
        "rate limit",
        "too many requests",
        "service unavailable",
    ):
        if token in msg:
            return True
    return False


def _error_status(exc: Exception) -> str:
    """Classify an exception into a short status string."""
    status = _error_status_code(exc)
    if isinstance(status, int):
        return f"http_{status}"
    name = type(exc).__name__.lower()
    if "timeout" in name:
        return "timeout"
    if "rate" in name and "limit" in name:
        return "rate_limit"
    return "error"


class LiteLLMWorker:
    """Call a heterogeneous worker pool through a LiteLLM proxy (Chat Completions)."""

    def __init__(
        self,
        models: list[str],
        api_key: str | None = None,
        api_base: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.2,
        timeout: int = 60,
        max_attempts: int = 2,
        retry_backoff: float = 2.0,
        max_worker_concurrency: int = 3,
        per_model_concurrency: dict[str, int] | str | None = None,
    ):
        specs = [split_model_spec(m) for m in models]
        self.models = [m.removeprefix("litellm/") for m, _ in specs]
        self.efforts = [e for _, e in specs]
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY")
        if not self.api_key:
            raise ValueError("LITELLM_API_KEY is required")
        self.api_base = (
            api_base or os.environ.get("LITELLM_BASE_URL", "http://127.0.0.1:8080/v1")
        ).rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.retry_backoff = retry_backoff
        self.max_worker_concurrency = max(1, max_worker_concurrency)
        self._session = requests.Session()
        self._global_sem = threading.Semaphore(self.max_worker_concurrency)

        parsed_limits: dict[str, int] = {}
        if isinstance(per_model_concurrency, str):
            parsed_limits = json.loads(per_model_concurrency)
        elif per_model_concurrency:
            parsed_limits = dict(per_model_concurrency)
        self._per_model_sem = []
        for i, model in enumerate(self.models):
            effort = self.efforts[i]
            is_reasoning = effort is not None and effort != "none"
            default_limit = 1 if is_reasoning else 2
            limit = parsed_limits.get(model, parsed_limits.get(f"{model}|{effort}", default_limit))
            self._per_model_sem.append(threading.Semaphore(limit))

    def _single_call(self, model: str, effort: str | None, messages: list[dict]) -> str:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        # Only set temperature when no reasoning effort is requested.
        if effort is None or effort == "none":
            body["temperature"] = self.temperature
        else:
            body["reasoning_effort"] = effort

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.api_base}/chat/completions"
        # Enforce a hard per-call ceiling: 10 s connect, then read up to self.timeout.
        resp = self._session.post(url, json=body, headers=headers, timeout=(10, self.timeout))
        resp.raise_for_status()
        data = resp.json()
        return str(data["choices"][0]["message"].get("content") or "")

    def call_with_retry(
        self,
        role: str,
        messages: list[dict],
        agent_id: int,
        task_id: int | None = None,
    ) -> tuple[str, dict]:
        """Call one worker with retries and return (completion, event_dict)."""
        model = self.models[agent_id % len(self.models)]
        effort = self.efforts[agent_id % len(self.efforts)]
        per_model_sem = self._per_model_sem[agent_id % len(self._per_model_sem)]
        event: dict[str, object] = {
            "task_id": task_id,
            "model": model,
            "agent_id": agent_id,
            "reasoning_effort": effort,
        }
        start_total = time.perf_counter()
        completion = ""
        attempts = 0
        status = "success"
        http_code: int | None = 200
        error: str | None = None
        call_elapsed = 0.0
        queue_wait = 0.0

        per_model_sem.acquire()
        try:
            self._global_sem.acquire()
            try:
                queue_wait = time.perf_counter() - start_total
                for attempt in range(1, self.max_attempts + 1):
                    attempts = attempt
                    call_start = time.perf_counter()
                    try:
                        completion = self._single_call(model, effort, messages)
                        call_elapsed = time.perf_counter() - call_start
                        status = "success"
                        http_code = 200
                        error = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        call_elapsed = time.perf_counter() - call_start
                        status = _error_status(exc)
                        http_code = getattr(exc, "status_code", None)
                        error = str(exc)
                        if attempt < self.max_attempts and _is_transient_error(exc):
                            time.sleep(self.retry_backoff)
                            continue
                        break
            finally:
                self._global_sem.release()
        finally:
            per_model_sem.release()

        elapsed = time.perf_counter() - start_total
        event.update(
            {
                "attempt": attempts,
                "status": status,
                "http_code": http_code,
                "error": error,
                "queue_wait": round(queue_wait, 6),
                "call_elapsed": round(call_elapsed, 6),
                "elapsed": round(elapsed, 6),
            }
        )
        return completion, event

    def __call__(self, role: str, messages: list[dict], agent_id: int) -> str:
        """Legacy string-only interface."""
        return self.call_with_retry(role, messages, agent_id)[0]


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
        records.append(
            {
                "task": instruction,
                "expected": reference,
                "system": TERMINAL_SYSTEM,
                "name": task_dir.name,
            }
        )

    random.Random(seed).shuffle(records)
    if limit:
        records = records[:limit]
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]


def load_toolscale_tasks(limit: int, seed: int = 42, val_frac: float = 0.1):
    # Imported lazily so API-only installs (without the `train` extra) can
    # still import this module for pool parsing and unit tests.
    from datasets import load_dataset

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

    random.Random(seed).shuffle(records)
    n_val = int(len(records) * val_frac)
    return records[n_val:], records[:n_val]


def worker_messages(task: str, system: str | None = None) -> list[dict]:
    # Anthropic-compatible models do not allow an assistant-message prefill,
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


def train_head(
    X: torch.Tensor,
    y_worker: torch.Tensor,
    y_role: torch.Tensor,
    head0: torch.Tensor,
    epochs: int = 30,
    lr: float = 1e-3,
    alpha: float = 0.1,
    l2_lambda: float = 0.01,
    device: str | None = None,
) -> tuple[torch.Tensor, float]:
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
            idx = train_idx[perm[i : i + 32]]
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
        print(
            f"[epoch {epoch + 1}/{epochs}] loss={epoch_loss:.4f} "
            f"val_worker_acc={acc_w:.3f} val_role_acc={acc_r:.3f}",
            flush=True,
        )
        if acc_w > best_val:
            best_val = acc_w
            best_weight = head.weight.detach().cpu().clone()

    return best_weight, best_val


# ---------------------------------------------------------------------------
# Worker scoring and label selection
# ---------------------------------------------------------------------------


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


def _write_events(events_path: Path, events: list[dict]) -> None:
    with open(events_path, "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)


def _score_one_worker(
    worker: LiteLLMWorker,
    task_idx: int,
    task: str,
    expected: str | list,
    system: str | None,
    agent_id: int,
    model: str,
    cache: dict[tuple[str, str], dict],
    cache_lock: threading.Lock,
) -> tuple[str, float, dict]:
    """Return (completion, score, event) for one (task, worker) pair."""
    key = (task, model)
    if key in cache:
        completion = str(cache[key]["completion"])
        score = reward_for(completion, expected)
        event = {
            "task_id": task_idx,
            "model": model,
            "agent_id": agent_id,
            "status": "cached",
            "attempt": 0,
            "score": score,
        }
        return completion, score, event

    if hasattr(worker, "call_with_retry"):
        completion, event = worker.call_with_retry(
            "Worker", worker_messages(task, system), agent_id, task_id=task_idx
        )
    else:
        # Fallback for unmocked / legacy worker interfaces in tests.
        completion = worker("Worker", worker_messages(task, system), agent_id)
        event = {
            "task_id": task_idx,
            "model": model,
            "agent_id": agent_id,
            "status": "mock",
            "attempt": 1,
        }

    score = reward_for(completion, expected)
    event["score"] = score
    print(
        f"[worker] task={task_idx} model={model} status={event.get('status', 'unknown')} "
        f"attempt={event.get('attempt', 0)} score={score:.3f} "
        f"elapsed={event.get('elapsed', 0.0):.2f}s",
        flush=True,
    )
    with cache_lock:
        cache[key] = {"task": task, "model": model, "completion": completion}
    return completion, score, event


def _model_cost(model: str, costs: dict[str, float] | None) -> float:
    if not costs:
        return float("inf")
    if model in costs:
        return costs[model]
    matches = [cost for name, cost in costs.items() if name.rsplit("/", 1)[-1] == model]
    return matches[0] if len(matches) == 1 else float("inf")


def _cheapest_eligible(
    eligible: list[tuple[int, float]],
    pool: list[str],
    costs: dict[str, float] | None,
) -> int:
    def sort_key(item: tuple[int, float]) -> tuple[float, float, int]:
        i, score = item
        model, _ = split_model_spec(pool[i])
        return (_model_cost(model.removeprefix("litellm/"), costs), -score, i)

    return min(eligible, key=sort_key)[0]


def _pick_best_worker(
    scores: list[float | None],
    pool: list[str],
    label_mode: str,
    costs: dict[str, float] | None,
) -> int:
    """Return the worker index that should be the gold label for this task.

    Returns -1 if no usable worker is present.
    """
    usable = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not usable:
        return -1
    if label_mode == "quality":
        return max(usable, key=lambda x: (x[1], -x[0]))[0]

    # cost mode
    if costs is None:
        raise ValueError("cost mode requires a cost table")
    ratios: list[tuple[int, float]] = []
    for i, s in usable:
        model, _ = split_model_spec(pool[i])
        cost = max(_model_cost(model.removeprefix("litellm/"), costs), 1e-6)
        ratios.append((i, s / cost))
    if not ratios:
        return -1
    return max(ratios, key=lambda x: (x[1], scores[x[0]], -x[0]))[0]


def _pick_budgeted_worker(
    scores: list[float | None],
    pool: list[str],
    costs: dict[str, float] | None,
    tolerance: float,
    expected: str | list,
) -> tuple[int, str | None]:
    """Budgeted label selection.

    Returns (agent_id, skip_reason). skip_reason is None when a gold worker
    was selected.
    """
    usable = [(i, s) for i, s in enumerate(scores) if s is not None]
    if len(usable) < 2:
        return -1, "insufficient_workers"

    if isinstance(expected, list):
        # Binary outcome: worker must fully pass.
        passing = [(i, s) for i, s in usable if s >= 1.0 - 1e-6]
        if not passing:
            return -1, "no_verified_worker_passed"
        return _cheapest_eligible(passing, pool, costs), None

    s_max = max(s for _, s in usable)
    eligible = [(i, s) for i, s in usable if s >= s_max - tolerance]
    return _cheapest_eligible(eligible, pool, costs), None


def _process_task_scores(
    task_idx: int,
    task: str,
    expected: str | list,
    scores: list[float | None],
    pool: list[str],
    label_mode: str,
    costs: dict[str, float] | None,
    quality_tolerance: float,
) -> dict:
    if label_mode == "budgeted":
        gold, reason = _pick_budgeted_worker(scores, pool, costs, quality_tolerance, expected)
    elif label_mode == "cost":
        gold = _pick_best_worker(scores, pool, "cost", costs)
        reason = "insufficient_workers" if gold < 0 else None
    else:
        gold = _pick_best_worker(scores, pool, "quality", costs)
        reason = "insufficient_workers" if gold < 0 else None

    if reason:
        return {
            "task_idx": task_idx,
            "task": task,
            "gold": expected,
            "scores": scores,
            "gold_worker": None,
            "skip_reason": reason,
        }
    return {
        "task_idx": task_idx,
        "task": task,
        "gold": expected,
        "scores": scores,
        "gold_worker": gold,
        "skip_reason": None,
    }


def _score_worker_pool(
    worker: LiteLLMWorker,
    pool: list[str],
    train_ds: list[dict],
    out_dir: Path,
    label_mode: str,
    costs: dict[str, float] | None,
    quality_tolerance: float,
    cache_path: Path,
) -> tuple[list[dict], list[dict], dict[str, dict[int, int]], list[dict]]:
    """Score every worker on every task and return (results, skipped, label_dist, events)."""
    cache = _load_worker_cache(cache_path)
    cache_lock = threading.Lock()
    events: list[dict] = []
    n_agents = len(pool)
    max_concurrency = getattr(worker, "max_worker_concurrency", 3)
    partial = [
        {"scores": [None] * n_agents, "completions": [""] * n_agents, "remaining": n_agents}
        for _ in train_ds
    ]
    final: list[dict | None] = [None] * len(train_ds)

    max_workers = max_concurrency + n_agents
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures: dict = {}
        for i, row in enumerate(train_ds):
            task = row["task"]
            expected = row["expected"]
            system = row.get("system")
            for agent_id, spec in enumerate(pool):
                model, _ = split_model_spec(spec)
                model_id = model.removeprefix("litellm/")
                fut = executor.submit(
                    _score_one_worker,
                    worker,
                    i,
                    task,
                    expected,
                    system,
                    agent_id,
                    model_id,
                    cache,
                    cache_lock,
                )
                futures[fut] = (i, agent_id)

        for fut in as_completed(futures):
            i, agent_id = futures[fut]
            completion, score, event = fut.result()
            events.append(event)
            partial[i]["scores"][agent_id] = score
            partial[i]["completions"][agent_id] = completion
            partial[i]["remaining"] -= 1
            if partial[i]["remaining"] == 0:
                row = train_ds[i]
                final[i] = _process_task_scores(
                    i,
                    row["task"],
                    row["expected"],
                    partial[i]["scores"],
                    pool,
                    label_mode,
                    costs,
                    quality_tolerance,
                )

    _write_worker_cache(cache_path, cache)
    _write_events(out_dir / "events.jsonl", events)

    results = [r for r in final if r is not None and r.get("gold_worker") is not None]
    skipped = [r for r in final if r is not None and r.get("gold_worker") is None]
    label_dist = _label_distribution(final, pool, costs, quality_tolerance)
    return results, skipped, label_dist, events


def _label_distribution(
    final: list[dict | None],
    pool: list[str],
    costs: dict[str, float] | None,
    quality_tolerance: float,
) -> dict[str, dict[int, int]]:
    """Compute quality/cost/budgeted label distributions on the same scored data."""
    dist: dict[str, dict[int, int]] = {"quality": {}, "cost": {}, "budgeted": {}}
    for r in final:
        if r is None or r.get("gold_worker") is None:
            continue
        scores = r["scores"]
        expected = r["gold"]
        for mode in ("quality", "cost"):
            label = _pick_best_worker(scores, pool, mode, costs)
            if label >= 0:
                dist[mode][label] = dist[mode].get(label, 0) + 1
        label, _ = _pick_budgeted_worker(scores, pool, costs, quality_tolerance, expected)
        if label >= 0:
            dist["budgeted"][label] = dist["budgeted"].get(label, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Pricing / cost helpers
# ---------------------------------------------------------------------------


def _resolve_costs(
    pool: list[str], fallback_path: str | Path, out_dir: Path
) -> tuple[dict[str, float], str]:
    """Load the explicit shadow-cost table used for LiteLLM-routed labels."""
    costs = _load_cost_table(fallback_path)
    snapshot = {"source": "cost_table", "pool": pool, "costs": costs}
    (out_dir / "pricing_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return costs, "cost_table"


def _load_cost_table(path: str | Path) -> dict[str, float]:
    """Load a cost table JSON, skipping comment keys that start with '_'."""
    with open(path) as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_retrain_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Retrain TRINITY router head on a new worker pool")
    ap.add_argument(
        "--pool",
        default=os.environ.get("RETRAIN_WORKER_MODELS"),
        required=False,
        help="Comma-separated LiteLLM model ids. "
        "Append '|reasoning_effort' per model, e.g. openai/gpt-5.6-terra|xhigh",
    )
    ap.add_argument(
        "--output-dir",
        default=os.environ.get("RETRAIN_OUTPUT_DIR", "outputs/router_retrain"),
    )
    ap.add_argument(
        "--dataset",
        default="s3://external-datasets-archive/terminal-bench-2.1/",
        help="Dataset to train on. 'nvidia/ToolScale' or a TerminalBench 2.1 source.",
    )
    ap.add_argument("--limit", type=int, default=int(os.environ.get("RETRAIN_LIMIT", "200")))
    ap.add_argument("--epochs", type=int, default=int(os.environ.get("RETRAIN_EPOCHS", "30")))
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.1, help="role-loss weight")
    ap.add_argument("--l2", type=float, default=0.01, help="L2 regularization toward original head")
    default_cost_table = str(REPO_ROOT / "config" / "worker-costs.json")
    ap.add_argument(
        "--label-mode",
        choices=["quality", "cost", "budgeted"],
        default=os.environ.get("RETRAIN_LABEL_MODE", "budgeted"),
        help="How to pick the gold worker per task (quality=argmax score, "
        "cost=argmax score/cost, budgeted=cheapest within tolerance).",
    )
    ap.add_argument(
        "--quality-tolerance",
        type=float,
        default=float(os.environ.get("RETRAIN_QUALITY_TOLERANCE", "0.05")),
        help="Budgeted-mode tolerance below the max successful score.",
    )
    ap.add_argument(
        "--cost-table",
        default=os.environ.get("RETRAIN_COST_TABLE", default_cost_table),
        help="JSON mapping model id -> shadow USD per task call.",
    )
    ap.add_argument(
        "--max-worker-concurrency",
        type=int,
        default=int(os.environ.get("RETRAIN_MAX_WORKER_CONCURRENCY", "3")),
        help="Global cap on concurrent worker API calls.",
    )
    ap.add_argument(
        "--per-model-concurrency",
        default=os.environ.get("RETRAIN_PER_MODEL_CONCURRENCY"),
        help="JSON dict of model id -> max concurrent calls (default 1 reasoning, 2 else).",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("RETRAIN_MAX_TOKENS", "256")),
        help="Max tokens per worker completion (reduces cost/hang time for short outputs).",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("RETRAIN_TIMEOUT", "60")),
        help="Per-call read timeout in seconds (connect timeout is fixed at 10s).",
    )
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cache",
        default="worker_outputs.jsonl",
        help="Cache worker responses to avoid re-calling the API",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--fugu-model", default=os.environ.get("MANTIS_MODEL", "Qwen/Qwen3-0.6B"))
    default_vec = str(REPO_ROOT / "artifacts" / "model_iter_60.npy")
    ap.add_argument("--fugu-vector", default=os.environ.get("MANTIS_VECTOR", default_vec))
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
    is_terminal = (
        Path(args.dataset).exists()
        or "terminal" in args.dataset.lower()
        or args.dataset.startswith("s3://")
    )
    if is_terminal:
        train_ds, val_ds = load_terminalbench_tasks(
            args.dataset, args.limit, args.seed, args.val_frac
        )
    else:
        train_ds, val_ds = load_toolscale_tasks(args.limit, args.seed, args.val_frac)
    print(f"[retrain] train={len(train_ds)} val={len(val_ds)}", flush=True)
    return train_ds, val_ds


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
        X,
        y_worker,
        y_role,
        head0,
        epochs=args.epochs,
        lr=args.lr,
        alpha=args.alpha,
        l2_lambda=args.l2,
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
    worker: LiteLLMWorker,
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
    results: list[dict],
    skipped: list[dict],
    label_distribution: dict,
    pricing_source: str,
    train_size: int,
    val_size: int,
    best_val_acc: float,
    avg_val_reward: float,
    out_dir: Path,
    vec_path: Path,
    head_path: Path,
) -> dict:
    total_tasks = train_size + len(skipped)
    retention = len(results) / total_tasks if total_tasks else 0.0
    skipped_reasons = Counter(s.get("skip_reason") for s in skipped)
    report = {
        "pool": pool,
        "dataset": args.dataset,
        "limit": args.limit,
        "label_mode": args.label_mode,
        "quality_tolerance": args.quality_tolerance,
        "max_worker_concurrency": args.max_worker_concurrency,
        "train_size": len(results),
        "skipped_count": len(skipped),
        "retention": retention,
        "skipped_reasons": dict(skipped_reasons),
        "val_size": val_size,
        "best_val_worker_acc": best_val_acc,
        "avg_val_reward": avg_val_reward,
        "output": str(out_dir),
        "vector": str(vec_path),
        "head": str(head_path),
        "label_distribution": label_distribution,
        "pricing_source": pricing_source,
    }
    if args.label_mode in ("cost", "budgeted"):
        report["cost_table"] = args.cost_table
    return report


def main(argv=None) -> None:
    args = _parse_retrain_args(argv)
    pool = _prepare_pool(args.pool)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    costs, pricing_source = _resolve_costs(pool, args.cost_table, out_dir)
    train_ds, val_ds = _load_retrain_data(args)
    cache_path = out_dir / args.cache

    per_model_concurrency = None
    if args.per_model_concurrency:
        per_model_concurrency = json.loads(args.per_model_concurrency)

    worker = LiteLLMWorker(
        pool,
        max_worker_concurrency=args.max_worker_concurrency,
        per_model_concurrency=per_model_concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    results, skipped, label_distribution, events = _score_worker_pool(
        worker,
        pool,
        train_ds,
        out_dir,
        args.label_mode,
        costs,
        args.quality_tolerance,
        cache_path,
    )

    if not results:
        print("[retrain] all tasks skipped; cannot train.", flush=True)
        report = _build_report(
            args,
            pool,
            results,
            skipped,
            label_distribution,
            pricing_source,
            len(train_ds),
            len(val_ds),
            0.0,
            0.0,
            out_dir,
            out_dir / "model_iter_60.npy",
            out_dir / "router_head.npy",
        )
        with open(out_dir / "report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        return

    _write_labels(out_dir, results)
    with open(out_dir / "skipped_tasks.jsonl", "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in skipped)

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
        args,
        pool,
        results,
        skipped,
        label_distribution,
        pricing_source,
        len(train_ds),
        len(val_ds),
        best_val_acc,
        avg_val_reward,
        out_dir,
        vec_path,
        head_path,
    )
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("[retrain] done. Report:", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
