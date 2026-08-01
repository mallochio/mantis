#!/usr/bin/env python3
"""Retrain the OpenFugu Conductor (Llama-3.2-3B GRPO) on a new worker pool.

Usage (smoke test):
  export OPENROUTER_API_KEY=sk-or-v1-...
  export HF_TOKEN=hf-...
  python scripts/retrain_conductor.py \
    --pool "anthropic/claude-sonnet-5|medium,...,z-ai/glm-5.2|none" \
    --dataset s3://external-datasets-archive/terminal-bench-2.1/ \
    --steps 20 --limit 8 --base di-zhang-fdu/openfugu-conductor-3b

Real-3B one-step acceptance test:
  python scripts/retrain_conductor.py \
    --pool "..." --dataset s3://external-datasets-archive/terminal-bench-2.1/ \
    --real-checkpoint-smoke --base di-zhang-fdu/openfugu-conductor-3b

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
import subprocess
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
REAL_CHECKPOINT = DEFAULT_BASE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _gpu_info() -> tuple[str, int]:
    """Return (gpu_type, gpu_count) as strings/int, or ('none', 0) if no GPU."""
    try:
        import torch
    except ImportError:
        return "unknown", 0

    if not torch.cuda.is_available():
        return "none", 0
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else "none"
    return name, count


def _is_real_checkpoint(base: str) -> bool:
    """True if the requested base is the real OpenFugu Conductor checkpoint."""
    base_path = Path(base).name
    return (
        base == REAL_CHECKPOINT
        or base_path == REAL_CHECKPOINT
        or base.endswith("openfugu-conductor-3b")
    )


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


def _write_manifest(
    out: Path,
    args: argparse.Namespace,
    pool_specs: list[str],
    device: str,
    dtype: str,
    gpu_type: str,
    gpu_count: int,
    git_commit: str,
    real_checkpoint_loaded: bool,
    valid_for_runtime: bool,
    acceptance_passed: bool,
    checkpoint_path: str,
    run_id: str,
) -> None:
    """Write a manifest.json describing the run and its validity."""
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": git_commit,
        "base_model": args.base,
        "real_checkpoint_loaded": real_checkpoint_loaded,
        "device": device,
        "gpu_type": gpu_type,
        "gpu_count": gpu_count,
        "dtype": str(dtype),
        "dataset": args.dataset,
        "task_limit": args.limit,
        "steps": args.steps,
        "generations": args.num_generations,
        "per_device_batch": args.per_device_batch,
        "pool": pool_specs,
        "output_checkpoint_path": checkpoint_path,
        "acceptance_passed": acceptance_passed,
        "valid_for_runtime": valid_for_runtime,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _acceptance_generation(
    checkpoint_dir: Path,
    tokenizer: Any,
    slot_labels: list[str],
    prompt: str,
    max_completion_length: int,
    temperature: float,
) -> tuple[str, float]:
    """Reload the saved checkpoint and run one inference generation.

    Returns the generated text and its format reward.
    """
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(checkpoint_dir), torch_dtype=dtype, trust_remote_code=True
    )
    model = model.to(torch.device(device))  # type: ignore[arg-type]
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=max(temperature, 0.01),
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = outputs[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
    reward = _format_reward_one(completion)
    return completion, reward


def _env_int(key: str, default: str) -> int:
    return int(os.environ.get(key, default))


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO retraining for OpenFugu Conductor")

    # Defaults that can be overridden by --real-checkpoint-smoke.
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
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-generations", type=int, default=None)
    parser.add_argument("--per-device-batch", type=int, default=per_dev)
    parser.add_argument("--max-prompt-length", type=int, default=max_prompt)
    parser.add_argument("--max-completion-length", type=int, default=max_comp)
    parser.add_argument("--output-dir", default=os.environ.get("RETRAIN_OUTPUT_DIR"))
    parser.add_argument("--run-id", default=os.environ.get("RETRAIN_RUN_ID"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker-timeout", type=int, default=worker_timeout)
    parser.add_argument("--worker-max-tokens", type=int, default=worker_max_tokens)
    parser.add_argument("--mock-worker", action="store_true", default=mock_worker)
    parser.add_argument("--temperature", type=float, default=temperature)
    parser.add_argument("--reward", default="verifiable", choices=["verifiable"])
    parser.add_argument(
        "--real-checkpoint-smoke",
        action="store_true",
        help="One-step acceptance test for the real 3B checkpoint. Requires CUDA.",
    )
    args = parser.parse_args()

    # Apply mode-specific defaults; explicit CLI flags take precedence.
    if args.real_checkpoint_smoke:
        if args.steps is None:
            args.steps = 1
        if args.limit is None:
            args.limit = 1
        if args.num_generations is None:
            args.num_generations = 2
    else:
        if args.steps is None:
            args.steps = steps
        if args.limit is None:
            args.limit = limit
        if args.num_generations is None:
            args.num_generations = num_gen

    # num_generations must divide per_device_train_batch_size for GRPO grouping.
    if args.num_generations < 2:
        parser.error("--num-generations must be at least 2 for GRPO")
    if args.per_device_batch % args.num_generations != 0:
        parser.error(
            f"--per-device-batch ({args.per_device_batch}) must be a multiple of "
            f"--num-generations ({args.num_generations})"
        )

    raw_specs = [s.strip() for s in args.pool.split(",") if s.strip()]
    if len(raw_specs) != N_AGENTS:
        raise ValueError(f"pool must contain exactly {N_AGENTS} specs, got {len(raw_specs)}")

    parsed = [split_model_spec(s) for s in raw_specs]
    models = [normalize_model_id(m) for m, _ in parsed]
    efforts = [e for _, e in parsed]
    pool_specs = [f"{m}|{e}" if e else m for m, e in zip(models, efforts, strict=True)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"conductor-retrain-{timestamp}"
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

    if args.real_checkpoint_smoke and not torch.cuda.is_available():
        raise SystemExit(
            "--real-checkpoint-smoke requires a CUDA GPU. "
            "CPU/MPS is not supported for the real 3B checkpoint acceptance test."
        )

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True
    )
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
        f"[retrain_conductor] starting GRPO: base={args.base} "
        f"steps={args.steps} limit={args.limit} generations={args.num_generations} "
        f"per_device_batch={args.per_device_batch} pool={pool_specs}",
        flush=True,
    )
    trainer.train()
    trainer.save_model(str(out / "checkpoint"))
    print(f"[retrain_conductor] checkpoint saved to {out / 'checkpoint'}", flush=True)

    # Acceptance test for the real checkpoint smoke.
    gpu_type, gpu_count = _gpu_info()
    git_commit = _get_git_commit()
    real_checkpoint_loaded = _is_real_checkpoint(args.base)
    valid_for_runtime = False
    acceptance_passed = False
    checkpoint_path = str(out / "checkpoint")

    if args.real_checkpoint_smoke:
        test_prompt = _build_prompt(records[0]["task"], tok, slot_labels, args.max_prompt_length)
        completion, format_reward = _acceptance_generation(
            out / "checkpoint",
            tok,
            slot_labels,
            test_prompt,
            args.max_completion_length,
            args.temperature,
        )
        acceptance_passed = bool(format_reward > 0.0)
        print(
            f"[retrain_conductor] acceptance generation: "
            f"format_reward={format_reward} completion={completion[:200]!r}",
            flush=True,
        )
        if not acceptance_passed:
            raise RuntimeError(
                "Real-checkpoint acceptance test failed: generated completion is not a "
                f"parseable Conductor DAG. Reward={format_reward}"
            )
        valid_for_runtime = real_checkpoint_loaded and device == "cuda" and acceptance_passed
    else:
        # Non-smoke/regular run: valid only if it actually trained the real checkpoint on GPU.
        valid_for_runtime = real_checkpoint_loaded and device == "cuda"

    _write_manifest(
        out=out,
        args=args,
        pool_specs=pool_specs,
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        git_commit=git_commit,
        real_checkpoint_loaded=real_checkpoint_loaded,
        valid_for_runtime=valid_for_runtime,
        acceptance_passed=acceptance_passed,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
    )
    print(
        f"[retrain_conductor] manifest written. "
        f"real_checkpoint_loaded={real_checkpoint_loaded} "
        f"valid_for_runtime={valid_for_runtime}",
        flush=True,
    )
    print(f"[retrain_conductor] DONE: {out}", flush=True)


if __name__ == "__main__":
    main()
