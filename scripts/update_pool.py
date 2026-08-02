#!/usr/bin/env python3
"""Update mantis pool configuration for a new 7-slot worker pool.

Usage:
    python3 scripts/update_pool.py \
      --pool "anthropic/claude-sonnet-5|medium,..." \
      --fetch-costs --env-file .env

This updates:
- configs/litellm.yaml      (LiteLLM aliases used by the runtime orchestrator)
- configs/worker-costs.json (per-task cost estimates for cost-aware router labels)
- launch/sky/*.yaml         (RETRAIN_WORKER_MODELS defaults, optional)
- .env/.env.example         (FUGU_WORKER_MODELS alias list, optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

KNOWN_PREFIXES = {
    "claude-": "anthropic/",
    "gpt-": "openai/",
    "gemini-": "google/",
    "deepseek-": "deepseek/",
    "minimax-": "minimax/",
    "glm-": "z-ai/",
    "qwen": "qwen/",
    "mimo-": "xiaomi/",
    "gemma-": "google/",
}


def split_model_spec(spec: str) -> tuple[str, str | None]:
    """Parse 'model_id|reasoning_effort' into (model_id, effort)."""
    spec = spec.strip()
    effort: str | None = None
    if "|" in spec:
        spec, effort = spec.rsplit("|", 1)
        effort = effort.strip() or None
    return spec.strip(), effort


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
        "Pass a full id like 'anthropic/claude-sonnet-5' or 'openrouter/...'."
    )


def alias_for(full_id: str, effort: str | None) -> str:
    """Make a LiteLLM model_name alias from a full OpenRouter id."""
    base = full_id.rsplit("/", 1)[-1]
    if effort and effort != "none":
        return f"{base}-{effort}"
    return base


def parse_pool(csv: str) -> list[tuple[str, str | None]]:
    specs = [split_model_spec(s) for s in csv.split(",") if s.strip()]
    out: list[tuple[str, str | None]] = []
    for model, effort in specs:
        out.append((normalize_model_id(model), effort))
    return out


def update_litellm_config(path: Path, pool: list[tuple[str, str | None]]) -> None:
    lines = path.read_text().splitlines()
    start_marker = "# 7-slot fugu worker pool"
    start = next((i for i, line in enumerate(lines) if start_marker in line), None)
    if start is None:
        raise SystemExit(f"Could not find '{start_marker}' marker in {path}")
    # Find the first entry after the marker; then end at the next blank/top-level comment.
    first_entry = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"^  - ", lines[i])),
        None,
    )
    if first_entry is None:
        raise SystemExit(f"Could not find worker entries after '{start_marker}' in {path}")
    end = next(
        (
            i
            for i in range(first_entry + 1, len(lines))
            if not lines[i].strip() or re.match(r"^  # ", lines[i])
        ),
        len(lines),
    )
    new_lines = [
        "  # 7-slot fugu worker pool. Order becomes slot ids 0..n-1 for TRINITY/Conductor.",
        "  # Reasoning effort is baked into the LiteLLM alias so the OpenFugu workers",
        "  # only need to call the alias.",
    ]
    for full_id, effort in pool:
        alias = alias_for(full_id, effort)
        new_lines.append(f"  - model_name: {alias}")
        new_lines.append("    litellm_params:")
        new_lines.append(f"      model: openai/{full_id}")
        new_lines.append("      api_base: https://openrouter.ai/api/v1")
        new_lines.append("      api_key: os.environ/OPENROUTER_API_KEY")
        if effort:
            new_lines.append(f"      reasoning_effort: {effort}")
            new_lines.append('      allowed_openai_params: ["reasoning_effort"]')
        else:
            new_lines.append('      allowed_openai_params: ["reasoning_effort"]')
    path.write_text("\n".join(lines[:start] + new_lines + lines[end:]) + "\n")


def fetch_openrouter_costs(api_key: str | None) -> dict[str, dict[str, float]]:
    """Return OpenRouter pricing indexed by model id."""
    import urllib.error
    import urllib.request

    url = "https://openrouter.ai/api/v1/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(  # noqa: S310
            req, timeout=60,
        ) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"OpenRouter pricing fetch failed: {e.code} {e.reason}") from e

    costs: dict[str, dict[str, float]] = {}
    for entry in data.get("data", []):
        model_id = entry.get("id", "")
        pricing = entry.get("pricing", {})
        if not model_id or not pricing:
            continue
        costs[model_id] = {
            "prompt": float(pricing.get("prompt", 0.0) or 0.0),
            "completion": float(pricing.get("completion", 0.0) or 0.0),
        }
    return costs


def update_cost_table(
    path: Path, pool: list[tuple[str, str | None]], costs: dict[str, dict[str, float]] | None
) -> dict[str, float]:
    if path.exists():
        table: dict[str, Any] = json.loads(path.read_text())
    else:
        note = "Per-task cost estimates for TRINITY router retraining label weighting, not billing."
        assumptions = (
            "Prices fetched from OpenRouter /api/v1/models; "
            "per-task cost assumes 2000 input tokens + 1000 output tokens."
        )
        table = {"_note": note, "_assumptions": assumptions}
    if costs:
        for full_id, _effort in pool:
            price = costs.get(full_id)
            if not price:
                print(f"[warn] no OpenRouter pricing for {full_id}", file=sys.stderr)
                continue
            table[full_id] = round(price["prompt"] * 2000 + price["completion"] * 1000, 9)
    path.write_text(json.dumps(table, indent=2) + "\n")
    return {k: v for k, v in table.items() if not k.startswith("_")}


def update_yaml_env(path: Path, pool_csv: str) -> None:
    text = path.read_text()
    new_text = re.sub(
        r'RETRAIN_WORKER_MODELS:\s*"[^"]*"',
        f'RETRAIN_WORKER_MODELS: "{pool_csv}"',
        text,
    )
    path.write_text(new_text)


def update_env_file(path: Path, aliases: list[str]) -> None:
    aliases_csv = ",".join(aliases)
    text = path.read_text() if path.exists() else ""
    if "FUGU_WORKER_MODELS=" in text:
        text = re.sub(r"FUGU_WORKER_MODELS=.*", f"FUGU_WORKER_MODELS={aliases_csv}", text)
    else:
        text += f"\nFUGU_WORKER_MODELS={aliases_csv}\n"
    path.write_text(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", required=True, help="Comma-separated model specs: full_id|effort")
    parser.add_argument("--fetch-costs", action="store_true", help="Fetch OpenRouter prices")
    parser.add_argument("--litellm-config", default=str(REPO_ROOT / "configs" / "litellm.yaml"))
    parser.add_argument("--worker-costs", default=str(REPO_ROOT / "configs" / "worker-costs.json"))
    parser.add_argument("--env-file", help="Update FUGU_WORKER_MODELS in this .env file")
    parser.add_argument(
        "--update-yamls",
        action="store_true",
        help="Set RETRAIN_WORKER_MODELS in SkyPilot YAMLs",
    )
    router_yaml = REPO_ROOT / "launch" / "sky" / "retrain_fugu_router.yaml"
    smoke_yaml = REPO_ROOT / "launch" / "sky" / "retrain_fugu_conductor_real_3b_smoke_gcp.yaml"
    full_yaml = REPO_ROOT / "launch" / "sky" / "retrain_fugu_conductor.yaml"
    parser.add_argument("--router-yaml", default=str(router_yaml))
    parser.add_argument("--conductor-smoke-yaml", default=str(smoke_yaml))
    parser.add_argument("--conductor-full-yaml", default=str(full_yaml))
    args = parser.parse_args(argv)

    pool = parse_pool(args.pool)
    aliases = [alias_for(full_id, effort) for full_id, effort in pool]
    pool_csv = ",".join(f"{full_id}|{effort}" if effort else full_id for full_id, effort in pool)

    costs: dict[str, dict[str, float]] | None = None
    if args.fetch_costs:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        costs = fetch_openrouter_costs(api_key)

    update_litellm_config(Path(args.litellm_config), pool)
    cost_map = update_cost_table(Path(args.worker_costs), pool, costs)

    if args.update_yamls:
        update_yaml_env(Path(args.router_yaml), pool_csv)
        update_yaml_env(Path(args.conductor_smoke_yaml), pool_csv)
        update_yaml_env(Path(args.conductor_full_yaml), pool_csv)

    if args.env_file:
        update_env_file(Path(args.env_file), aliases)

    print(f"Updated LiteLLM aliases ({args.litellm_config}):")
    for alias, (full_id, effort) in zip(aliases, pool, strict=True):
        print(f"  {alias} -> openai/{full_id} (effort={effort})")
    print("\nRETRAIN_WORKER_MODELS for --env or YAML overrides:")
    print(pool_csv)
    if args.update_yamls:
        print("\nSkyPilot YAMLs updated with the new pool.")
    if args.env_file:
        print(f"\nFUGU_WORKER_MODELS updated in {args.env_file}: {','.join(aliases)}")
    if not costs:
        print("\n[note] --fetch-costs not used; cost table preserved.")
        print("      Add missing entries before a cost-mode retrain.")
    else:
        print("\nPer-task cost estimates (2000 in + 1000 out):")
        for full_id, price in cost_map.items():
            print(f"  {full_id}: ${price:.6f}")


if __name__ == "__main__":
    main()
