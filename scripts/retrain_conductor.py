#!/usr/bin/env python3
"""Retrain the OpenFugu Conductor (Llama-3.2-3B GRPO) on a new worker pool.

Usage (smoke test):
  export OPENROUTER_API_KEY=sk-or-v1-...
  export HF_TOKEN=hf-...
  python scripts/retrain_conductor.py \
    --pool "anthropic/claude-sonnet-5|medium,...,z-ai/glm-5.2|none" \
    --dataset s3://external-datasets-archive/terminal-bench-2.1/ \
    --steps 20 --limit 8 --base di-zhang-fdu/openfugu-conductor-3b

The pool format is identical to scripts/retrain_router_pool.py. Worker calls during
rollout DAG execution reuse retrain_router_pool.OpenRouterWorker, so the same
OpenRouter key and routing aliases apply.
"""
from __future__ import annotations

import argparse
import datetime
import difflib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

# retrain_router_pool adds OpenFugu to sys.path and exposes the pool utilities.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from retrain_router_pool import (  # noqa: E402
    OpenRouterWorker,
    load_terminalbench_tasks,
    load_toolscale_tasks,
    normalize_model_id,
    split_model_spec,
)

from openfugu.ultra import (  # noqa: E402
    MAX_STEPS,
    N_AGENTS,
    ConductorExecutor,
    MockWorker,
    conductor_prompt,
    parse_workflow,
    visible_indices,
)
from train.toolscale_data import _parse_plan, _score  # noqa: E402

DEFAULT_BASE = "di-zhang-fdu/openfugu-conductor-3b"
DEFAULT_DATASET = "s3://external-datasets-archive/terminal-bench-2.1/"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_records(dataset: str, limit: int, seed: int) -> list[Any]:
    """Load a task dataset in the same shape as the router retrain script."""
    if "toolscale" in dataset.lower():
        train, _ = load_toolscale_tasks(limit=limit, seed=seed, val_frac=0.0)
        return cast(list[Any], train)
    train, _ = load_terminalbench_tasks(
        dataset, limit=limit, seed=seed, val_frac=0.0
    )
    return cast(list[Any], train)


def _example_answer(slot_labels: list[str]) -> str:
    """One-shot example that demonstrates the required 3-list format."""
    subtask = f"implement the solution using {slot_labels[1]}"
    return (
        "Plan:\n"
        "model_id: [0, 1]\n"
        f"subtasks: ['break the problem into subtasks', '{subtask}']\n"
        "access_list: [[], [0]]"
    )


def _format_plain(msgs: list[dict[str, str]]) -> str:
    """Fallback when the tokenizer has no chat_template."""
    parts = []
    for m in msgs:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n\n".join(parts)


def _build_prompt(task: str, tokenizer: Any, slot_labels: list[str], max_length: int) -> str:
    """Chat-formatted prompt that asks the Conductor for the three-list DAG."""
    base = conductor_prompt(task, slot_labels)
    example_task = "Plan a two-step coding workflow for: implement a recursive fibonacci function."
    example_messages = [
        {"role": "user", "content": example_task},
        {"role": "assistant", "content": _example_answer(slot_labels)},
    ]
    # Prefill the assistant to nudge the model into the canonical 3-list format.
    msgs = base[:1] + example_messages + base[1:] + [{"role": "assistant", "content": "Plan:\n"}]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            continue_final_message=True,
            truncation=True,
            max_length=max_length,
        )
    else:
        prompt = _format_plain(msgs)
    return str(prompt or "")


def _build_dataset(
    records: list[dict], tokenizer: Any, slot_labels: list[str], max_prompt_length: int
) -> Any:
    """Return a HuggingFace `Dataset` with `prompt` and `expected` columns."""
    from datasets import Dataset

    prompts = [
        _build_prompt(r["task"], tokenizer, slot_labels, max_prompt_length)
        for r in records
    ]
    expected = [r["expected"] for r in records]
    tasks = [r["task"] for r in records]
    return Dataset.from_dict(
        {"prompt": prompts, "expected": expected, "task": tasks}
    )


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------
def _parse_dag(completion: str) -> Any:
    """Parse a Conductor completion into (model_id, subtasks, access_list)."""
    try:
        return parse_workflow(completion)
    except (ValueError, TypeError):
        return None


def _format_reward_one(completion: str) -> float:
    """1.0 if the completion parses into three equal-length non-empty lists."""
    parsed = _parse_dag(completion)
    if parsed is None:
        return 0.0
    model_ids, subtasks, access = parsed
    if not (model_ids and subtasks and access):
        return 0.0
    if len(model_ids) != len(subtasks) or len(subtasks) != len(access):
        return 0.0
    if not (1 <= len(model_ids) <= MAX_STEPS):
        return 0.0
    if not all(isinstance(s, str) and s.strip() for s in subtasks):
        return 0.0
    return 1.0


def _action_reward_one(completion: str, slot_labels: list[str]) -> float:
    """Fraction of steps with valid worker ids and DAG-legal access lists."""
    parsed = _parse_dag(completion)
    if parsed is None:
        return 0.0
    model_ids, subtasks, access = parsed
    n = len(slot_labels)
    if not (model_ids and subtasks and access):
        return 0.0
    if len(model_ids) != len(subtasks) or len(subtasks) != len(access):
        return 0.0
    if len(model_ids) > MAX_STEPS:
        return 0.0

    score = 0.0
    for i, raw in enumerate(model_ids):
        try:
            mid = int(raw)
        except (TypeError, ValueError):
            return 0.0
        if not (0 <= mid < n):
            return 0.0
        if not (isinstance(subtasks[i], str) and subtasks[i].strip()):
            return 0.0
        try:
            visible_indices(access, i)
        except ValueError:
            return 0.0
        score += 1.0
    return score / len(model_ids)


def _outcome_reward_one(
    completion: str, expected: Any, worker: Any, slot_labels: list[str]
) -> float:
    """Execute the parsed DAG and compare the final worker output to `expected`."""
    parsed = _parse_dag(completion)
    if parsed is None:
        return 0.0
    model_ids, subtasks, access = parsed
    if not (model_ids and subtasks and access):
        return 0.0
    if len(model_ids) != len(subtasks) or len(subtasks) != len(access):
        return 0.0
    try:
        result = ConductorExecutor(worker, slot_labels=slot_labels).execute(
            model_ids, subtasks, access
        )
        final = (result.final or "").strip()
    except (ValueError, TypeError, RuntimeError):
        return 0.0

    if not expected:
        return 0.0

    # ToolScale expected actions are stored as JSON list strings; TerminalBench
    # expected outputs are plain strings (reference solve.sh).
    if isinstance(expected, str):
        exp = expected.strip()
        if exp.startswith("["):
            try:
                gold = json.loads(exp)
                pred = _parse_plan(final)
                if pred is not None:
                    return float(_score(pred, gold))
            except (json.JSONDecodeError, TypeError):
                gold = None
            if gold is not None and final:
                return float(difflib.SequenceMatcher(None, final, exp).ratio())
        if not final:
            return 0.0
        return float(difflib.SequenceMatcher(None, final, exp).ratio())

    # Expected is already a list of action dicts.
    pred = _parse_plan(final)
    if pred is None:
        return 0.0
    return float(_score(pred, expected))


def make_reward_functions(worker: Any, slot_labels: list[str]):
    """Return the three reward functions used for verifiable Conductor GRPO."""

    def conductor_format_reward(completions: list[str], **kwargs: Any) -> list[float]:
        return [_format_reward_one(c) for c in completions]

    def conductor_action_reward(
        completions: list[str], **kwargs: Any
    ) -> list[float]:
        return [_action_reward_one(c, slot_labels) for c in completions]

    def conductor_outcome_reward(
        completions: list[str], **kwargs: Any
    ) -> list[float]:
        expected = kwargs.get("expected") or [None] * len(completions)
        # Execute independent rollout DAGs concurrently.
        max_workers = min(len(completions), 8)
        results: list[float] = [0.0] * len(completions)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    _outcome_reward_one, c, e, worker, slot_labels
                ): i
                for i, (c, e) in enumerate(zip(completions, expected, strict=False))
            }
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return results

    return [
        conductor_format_reward,
        conductor_action_reward,
        conductor_outcome_reward,
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class MetricsJSONLCallback:
    """Write trl's per-step logs to metrics.jsonl."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any):
        if logs:
            rec = {"step": state.global_step, **logs}
            with open(self.path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")


def _write_configs(
    out: Path, args: argparse.Namespace, pool_specs: list[str], loaded: int
) -> None:
    (out / "pool.json").write_text(
        json.dumps(
            {
                "pool": pool_specs,
                "base": args.base,
                "steps": args.steps,
                "limit": args.limit,
                "num_generations": args.num_generations,
                "per_device_batch": args.per_device_batch,
            },
            indent=2,
        )
    )
    (out / "dataset.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "limit": args.limit,
                "loaded": loaded,
                "seed": args.seed,
            },
            indent=2,
        )
    )


def _env_int(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO retraining for OpenFugu Conductor")
    steps = _env_int("RETRAIN_STEPS", "20")
    limit = _env_int("RETRAIN_LIMIT", "50")
    num_gen = _env_int("RETRAIN_NUM_GENERATIONS", "2")
    per_dev = _env_int("RETRAIN_PER_DEVICE_BATCH", "2")
    max_prompt = _env_int("RETRAIN_MAX_PROMPT_LENGTH", "1024")
    max_comp = _env_int("RETRAIN_MAX_COMPLETION_LENGTH", "384")
    worker_timeout = _env_int("FUGU_WORKER_TIMEOUT", "120")
    worker_max_tokens = _env_int("FUGU_WORKER_MAX_TOKENS", "512")
    mock_worker = os.environ.get("FUGU_CONDUCTOR_MOCK_WORKER") == "1"
    temperature = float(os.environ.get("RETRAIN_TEMPERATURE", "1.0"))

    parser.add_argument("--pool", required=True, help="model|effort CSV (same as router retrain)")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--base", default=os.environ.get("FUGU_BASE_MODEL", DEFAULT_BASE))
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--limit", type=int, default=limit)
    parser.add_argument("--num-generations", type=int, default=num_gen)
    parser.add_argument("--per-device-batch", type=int, default=per_dev)
    parser.add_argument("--max-prompt-length", type=int, default=max_prompt)
    parser.add_argument("--max-completion-length", type=int, default=max_comp)
    parser.add_argument("--output-dir", default=os.environ.get("RETRAIN_OUTPUT_DIR"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-timeout", type=int, default=worker_timeout)
    parser.add_argument("--worker-max-tokens", type=int, default=worker_max_tokens)
    parser.add_argument("--mock-worker", action="store_true", default=mock_worker)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--reward", default="verifiable", choices=["verifiable"])
    args = parser.parse_args()

    raw_specs = [s.strip() for s in args.pool.split(",") if s.strip()]
    if len(raw_specs) != N_AGENTS:
        raise ValueError(f"pool must contain exactly {N_AGENTS} specs, got {len(raw_specs)}")

    parsed = [split_model_spec(s) for s in raw_specs]
    models = [normalize_model_id(m) for m, _ in parsed]
    efforts = [e for _, e in parsed]
    pool_specs = [f"{m}|{e}" if e else m for m, e in zip(models, efforts, strict=True)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.output_dir or f"outputs/conductor_retrain/{timestamp}")
    out.mkdir(parents=True, exist_ok=True)

    _write_configs(out, args, pool_specs, 0)

    records = _load_records(args.dataset, args.limit, args.seed)
    if not records:
        raise ValueError("no records loaded")
    _write_configs(out, args, pool_specs, len(records))

    # Slot labels shown to the Conductor include the reasoning effort tag.
    slot_labels = pool_specs

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    model = model.to(torch.device(device))  # type: ignore[arg-type]
    model.config.use_cache = False

    ds = _build_dataset(records, tok, slot_labels, args.max_prompt_length)

    if args.mock_worker:
        worker: Any = MockWorker()
    else:
        worker = OpenRouterWorker(
            models=pool_specs,
            timeout=args.worker_timeout,
            max_tokens=args.worker_max_tokens,
        )

    reward_funcs = make_reward_functions(worker, slot_labels)

    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    class _MetricsJSONLCallback(TrainerCallback):
        def __init__(self, path: Path):
            self._writer = MetricsJSONLCallback(path)

        def on_log(self, args, state, control, logs=None, **kwargs):
            self._writer.on_log(args, state, control, logs, **kwargs)

    cfg = GRPOConfig(
        output_dir=str(out / "checkpoint"),
        per_device_train_batch_size=args.per_device_batch,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.steps,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(args.steps // 2, 1),
        report_to=[],
        use_vllm=False,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=torch.cuda.is_available(),
        temperature=args.temperature,
        beta=0.0,  # no KL — matches Fugu-Ultra report
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=reward_funcs,
        args=cfg,
        train_dataset=ds,
        callbacks=[_MetricsJSONLCallback(out / "metrics.jsonl")],
    )

    print(
        f"[retrain_conductor] starting GRPO smoke: base={args.base} "
        f"steps={args.steps} limit={args.limit} pool={pool_specs}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(out / "checkpoint"))
    print(f"[retrain_conductor] DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
