#!/usr/bin/env python3
"""Comparative eval harness for mantis.

Usage:
    python3 eval/run_eval.py --config {direct,trinity,conductor-old,conductor-new,conductor-luna} \
        --fixtures eval/fixtures.jsonl --output eval/results.jsonl

Cost estimation method:
- `config/worker-costs.json` gives a per-call USD estimate for each raw
  OpenRouter model id (assumes ~2K prompt + 1K completion tokens). These
  prices were fetched from /api/v1/models by update_pool.py.
- `config/catalog.toml` maps LiteLLM worker slots to upstream model ids.
- "direct": one call at the requested model's cost.
- "trinity": parse fugu_trace (e.g. "Worker(3)->Thinker(1)->Verifier(1):...").
  Every "Role(slot)" arrow is one worker call; cost = sum(slot model cost).
- "conductor", "conductor-old", "conductor-new", "conductor-luna": fugu_trace
  is "steps:N:conductor". Cost = 1 planning call at MANTIS_CONDUCTOR_MODEL cost
  plus N step calls at the average pool worker cost. This is an upper-bound;
  the actual DAG may call multiple workers per step. If the trace cannot be
  parsed we fall back to usage.fugu_turns.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import tomllib
from pathlib import Path
from typing import Any

import requests

REPO = Path(__file__).resolve().parent.parent


def load_catalog_models(path: Path) -> tuple[list[str], str]:
    """Return worker models and the Conductor planner from the catalog."""
    with path.open("rb") as handle:
        catalog = tomllib.load(handle)
    mantis = catalog["mantis"]
    workers = mantis["workers"]
    models = [workers[slot]["upstream_model"] for slot in mantis["slot_order"]]
    cm = mantis["conductor_model"]
    # strip any provider prefix for backwards compat (litellm/, openrouter/, etc.)
    for prefix in ("litellm/", "openrouter/", "opencode-go/"):
        if cm.startswith(prefix):
            cm = cm[len(prefix) :]
            break
    return models, cm


def load_worker_costs(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def default_worker_costs_path() -> Path:
    return REPO / "config" / "worker-costs.json"


def prepare_output(path: Path, *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not append:
        path.write_text("")


def _cost_key(model: str) -> str:
    # strip any provider prefix (litellm/, openrouter/, opencode-go/)
    for prefix in ("litellm/", "openrouter/", "opencode-go/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def avg_pool_cost(costs: dict[str, float], models: list[str]) -> float:
    values = [costs[model] for model in map(_cost_key, models) if model in costs]
    return sum(values) / len(values) if values else 0.01


def model_cost(model: str, costs: dict[str, float]) -> float:
    return costs.get(_cost_key(model), 0.01)


def parse_conductor_trace(trace: str) -> int:
    m = re.search(r"steps:(\d+):conductor", trace)
    if m:
        return int(m.group(1))
    return 0


def parse_trinity_trace(trace: str) -> list[int]:
    # e.g. Worker(3)->Thinker(1)->Verifier(1):verifier_accept
    return [int(x) for x in re.findall(r"\w+\((\d+)\)", trace)]


def estimate_cost(
    config: str,
    response: dict[str, Any],
    costs: dict[str, float],
    slot_models: list[str],
    conductor_model: str = "",
) -> tuple[float, int]:
    """Return (est_cost_usd, n_worker_calls)."""
    usage = response.get("usage", {}) or {}
    trace = usage.get("fugu_trace", "")
    turns = usage.get("fugu_turns", 0) or 0

    if config == "direct":
        model = os.environ.get("DIRECT_MODEL", "deepseek-v4-flash")
        return model_cost(model, costs), 1

    pool_avg = avg_pool_cost(costs, slot_models)

    if config == "trinity":
        slots = parse_trinity_trace(trace or "")
        if not slots:
            slots = [0] * turns
        n = len(slots)
        cost = 0.0
        for sid in slots:
            model = slot_models[sid % len(slot_models)] if slot_models else ""
            cost += model_cost(model, costs)
        return cost, n

    # conductor-*
    steps = parse_conductor_trace(trace or "")
    if not steps:
        steps = turns
    plan_cost = model_cost(conductor_model, costs)
    step_cost = steps * pool_avg
    return plan_cost + step_cost, steps + 1


def post_chat(
    url: str,
    model: str,
    prompt: str,
    timeout: float,
    api_key: str | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    # Reasoning models reject temperature != 1 when reasoning effort is set.
    if "-luna" in model or "-sol" in model or "claude-" in model or "gemini-" in model:
        payload.pop("temperature", None)
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    data: dict[str, Any] = r.json()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one eval config.")
    parser.add_argument(
        "--config",
        required=True,
        choices=["direct", "trinity", "conductor-old", "conductor-new", "conductor-luna"],
    )
    parser.add_argument("--fixtures", default=str(REPO / "eval" / "fixtures.jsonl"))
    parser.add_argument("--output", default=str(REPO / "eval" / "results.jsonl"))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--litellm-url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--mantis-url", default="http://localhost:8088/v1/chat/completions")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to an existing output instead of starting a fresh run",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LITELLM_API_KEY") or os.environ.get("MANTIS_API_KEY"),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set LITELLM_API_KEY or MANTIS_API_KEY in the environment")

    costs = load_worker_costs(default_worker_costs_path())
    slot_models, conductor_model = load_catalog_models(REPO / "config" / "catalog.toml")

    fixtures: list[dict[str, Any]] = []
    with open(args.fixtures) as f:
        for line in f:
            line = line.strip()
            if line:
                fixtures.append(json.loads(line))

    out_path = Path(args.output)
    prepare_output(out_path, append=args.append)

    # Map config to endpoint/model
    if args.config == "direct":
        url = args.litellm_url
        model = os.environ.get("DIRECT_MODEL", "deepseek-v4-flash")
    else:
        url = args.mantis_url
        model = "mantis/trinity" if args.config == "trinity" else "mantis/ultra"

    for fx in fixtures:
        prompt = fx["prompt"]
        rec: dict[str, Any] = {
            "id": fx["id"],
            "config": args.config,
            "tier": fx["tier"],
            "prompt": prompt,
            "response_text": "",
            "latency_s": 0.0,
            "est_cost_usd": 0.0,
            "fugu_trace": None,
            "n_worker_calls": 0,
            "error": None,
        }
        start = time.time()
        try:
            resp = post_chat(url, model, prompt, args.timeout, api_key=args.api_key)
            rec["latency_s"] = round(time.time() - start, 3)
            rec["response_text"] = (
                resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            usage = resp.get("usage", {}) or {}
            rec["fugu_trace"] = usage.get("fugu_trace")
            rec["est_cost_usd"], rec["n_worker_calls"] = estimate_cost(
                args.config, resp, costs, slot_models, conductor_model
            )
        except requests.exceptions.Timeout:
            rec["latency_s"] = round(time.time() - start, 3)
            rec["error"] = "timeout"
        except (requests.exceptions.RequestException, ValueError, OSError) as e:
            rec["latency_s"] = round(time.time() - start, 3)
            rec["error"] = f"{type(e).__name__}: {e}"

        with open(out_path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(
            f"[{args.config}] {fx['id']} latency={rec['latency_s']} "
            f"cost={rec['est_cost_usd']:.4f} error={rec['error']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
