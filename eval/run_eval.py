#!/usr/bin/env python3
"""Comparative eval harness for fugu-local.

Usage:
    python3 eval/run_eval.py --config {direct,trinity,conductor-old,conductor-new,conductor-luna} \
        --fixtures eval/fixtures.jsonl --output eval/results.jsonl

Cost estimation method:
- `configs/worker-costs.json` gives a per-call USD estimate for each raw
  OpenRouter model id (assumes ~2K prompt + 1K completion tokens). These
  prices were fetched from /api/v1/models by update_pool.py.
- `configs/litellm.yaml` maps LiteLLM aliases to raw OpenRouter ids.
  A leading "openai/" is the LiteLLM provider prefix; stripping it yields
  the OpenRouter id used in worker-costs.json.
- "direct": one call at the requested model's cost.
- "trinity": parse fugu_trace (e.g. "Worker(3)->Thinker(1)->Verifier(1):...").
  Every "Role(slot)" arrow is one worker call; cost = sum(slot model cost).
- "conductor", "conductor-old", "conductor-new", "conductor-luna": fugu_trace
  is "steps:N:conductor". Cost = 1 planning call at FUGU_CONDUCTOR_MODEL cost
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
from pathlib import Path
from typing import Any

import requests
import yaml

REPO = Path(__file__).resolve().parent.parent


def load_litellm_alias_map(path: Path) -> dict[str, str]:
    """Map LiteLLM alias -> raw OpenRouter model id."""
    data = yaml.safe_load(path.read_text())
    alias_map: dict[str, str] = {}
    for entry in data.get("model_list", []):
        alias = entry.get("model_name")
        raw = entry.get("litellm_params", {}).get("model", "")
        if alias and raw.startswith("openai/"):
            alias_map[alias] = raw[len("openai/") :]
    return alias_map


def load_worker_costs(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {k: float(v) for k, v in data.items() if not k.startswith("_")}


def slot_models_from_env() -> list[str]:
    raw = os.environ.get("FUGU_WORKER_MODELS") or os.environ.get("FUGU_WORKER_MODEL") or ""
    return [m.strip() for m in raw.split(",") if m.strip()]


def avg_pool_cost(costs: dict[str, float], aliases: list[str], alias_map: dict[str, str]) -> float:
    total = 0.0
    count = 0
    for alias in aliases:
        raw = alias_map.get(alias, alias)
        c = costs.get(raw)
        if c is not None:
            total += c
            count += 1
    return total / count if count else 0.01


def model_cost(model_or_alias: str, costs: dict[str, float], alias_map: dict[str, str]) -> float:
    raw = alias_map.get(model_or_alias, model_or_alias)
    return costs.get(raw, 0.01)


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
    alias_map: dict[str, str],
    slot_aliases: list[str],
) -> tuple[float, int]:
    """Return (est_cost_usd, n_worker_calls)."""
    usage = response.get("usage", {}) or {}
    trace = usage.get("fugu_trace", "")
    turns = usage.get("fugu_turns", 0) or 0

    if config == "direct":
        model = os.environ.get("DIRECT_MODEL", "gpt-5.6-luna-max")
        return model_cost(model, costs, alias_map), 1

    pool_avg = avg_pool_cost(costs, slot_aliases, alias_map)

    if config == "trinity":
        slots = parse_trinity_trace(trace or "")
        if not slots:
            slots = [0] * turns
        n = len(slots)
        cost = 0.0
        for sid in slots:
            alias = slot_aliases[sid % len(slot_aliases)] if slot_aliases else ""
            cost += model_cost(alias, costs, alias_map)
        return cost, n

    # conductor-*
    steps = parse_conductor_trace(trace or "")
    if not steps:
        steps = turns
    conductor_model = os.environ.get("FUGU_CONDUCTOR_MODEL", "gpt-5.6-luna-max")
    plan_cost = model_cost(conductor_model, costs, alias_map)
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
    # LiteLLM reasoning models reject temperature!=1 when reasoning_effort set;
    # drop_params in litellm_settings handles it, but avoid conflict for direct.
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
    parser.add_argument("--litellm-url", default="http://localhost:3001/v1/chat/completions")
    parser.add_argument("--openfugu-url", default="http://localhost:8088/v1/chat/completions")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LITELLM_KEY") or os.environ.get("FUGU_API_KEY"),
    )
    args = parser.parse_args()
    if not args.api_key:
        parser.error("set LITELLM_KEY or FUGU_API_KEY in the environment")

    costs = load_worker_costs(REPO / "configs" / "worker-costs.json")
    alias_map = load_litellm_alias_map(REPO / "configs" / "litellm.yaml")
    slot_aliases = slot_models_from_env()
    if not slot_aliases:
        slot_aliases = list(alias_map.keys())[:7]

    fixtures: list[dict[str, Any]] = []
    with open(args.fixtures) as f:
        for line in f:
            line = line.strip()
            if line:
                fixtures.append(json.loads(line))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map config to endpoint/model
    if args.config == "direct":
        url = args.litellm_url
        model = os.environ.get("DIRECT_MODEL", "gpt-5.6-luna-max")
    else:
        url = args.openfugu_url
        model = "trinity" if args.config == "trinity" else "conductor"

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
                args.config, resp, costs, alias_map, slot_aliases
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
