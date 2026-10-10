#!/usr/bin/env python3
"""Drive routed harness cases against a live server and emit run records.

Writes one summarize-compatible record per case and variant; judgment
cases are marked ``needs_review`` until a reviewer reads the transcript.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import tomllib
from pathlib import Path
from typing import Any

import httpx
from evaluate_routed_harness import VARIANTS, load_cases
from routed_ab_tools import (
    OPENROUTER_BASE_URL,
    case_observation,
    execute_tool_call,
    openrouter_headers,
    post_round,
    request_body,
)

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "eval"
DEFAULT_CATALOG = REPO / "config" / "catalog.toml"
DEFAULT_CASES = EVAL / "routed-harness-cases.jsonl"
DEFAULT_OUT = EVAL / "results-routed-ab.jsonl"
DEFAULT_TRANSCRIPTS = EVAL / "runs" / "routed-ab"


def load_catalog(path: Path) -> dict[str, Any]:
    """Return the parsed catalog TOML."""
    with path.open("rb") as handle:
        return tomllib.load(handle)


def base_target_spec(catalog: dict[str, Any], role: str) -> dict[str, str]:
    """Return the provider-qualified model and effort for one base target."""
    target = catalog["base"]["targets"][role]
    return {
        "model": f"{target['provider']}/{target['upstream_model']}",
        "upstream_model": str(target["upstream_model"]),
        "reasoning_effort": str(target.get("reasoning_effort") or "none"),
    }


def request_plan(variant: str, catalog: dict[str, Any]) -> dict[str, Any]:
    """Map one benchmark variant onto chat request fields."""
    if variant in {"efficient-only", "capable-only"}:
        spec = base_target_spec(catalog, variant.split("-")[0])
        return {
            "model": spec["upstream_model"],
            "reasoning_effort": spec["reasoning_effort"],
            "base_url": OPENROUTER_BASE_URL,
        }
    if variant == "mantis-base":
        return {"model": "mantis/base"}
    if variant == "fusion-pinned":
        return {
            "model": "mantis/fusion",
            "tool_options": {"enabled": ["workspace", "shell"], "server_execution": True},
            "worker_profiles": [
                {"name": "default", "model": str(catalog["fusion"]["main"])}
            ],
        }
    if variant == "fusion-routed":
        return {
            "model": "mantis/fusion",
            "tool_options": {"enabled": ["workspace", "shell"], "server_execution": True},
        }
    raise ValueError(f"unknown variant {variant!r}")


def auto_judge(case_id: str, content: str, tool_rounds: int) -> bool | None:
    """Return a machine verdict where the case allows one, else None."""
    if case_id == "no-tool-chat":
        return content.strip() == "ROUTING_OK" and tool_rounds == 0
    return None


def usage_numbers(event: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    """Extract run token usage and cost from the response envelope."""
    usage = event.get("usage") if isinstance(event, dict) else None
    if not isinstance(usage, dict):
        usage = response.get("usage") if isinstance(response, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    cost = event.get("cost") if isinstance(event, dict) else None
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_read_tokens": int(cached or 0),
        "provider_cost": float(cost) if isinstance(cost, (int, float)) else 0.0,
    }


def selected_models(event: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Return the models a run actually billed."""
    models = event.get("usage_models") if isinstance(event, dict) else None
    if isinstance(models, dict) and models:
        return sorted(str(key) for key in models)
    if isinstance(models, list) and models:
        return sorted(str(item) for item in models)
    return [str(plan["model"])]


def _apply_tool_results(
    message: dict[str, Any],
    workspace: Path,
    messages: list[dict[str, Any]],
) -> None:
    """Execute pending calls and append their results to the transcript."""
    messages.append({"role": "assistant", "content": "", "tool_calls": message["tool_calls"]})
    messages.extend(
        {
            "role": "tool",
            "tool_call_id": call.get("id", ""),
            "content": execute_tool_call(call, workspace),
        }
        for call in message["tool_calls"]
    )


def drive_case(
    client: httpx.Client,
    base_url: str,
    plan: dict[str, Any],
    prompt: str,
    workspace: Path,
    max_rounds: int,
    plan_auth: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one case to completion and return the observation envelope."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    body = request_body(plan, messages)
    headers: dict[str, str] = dict(plan_auth or {})
    rounds = 0
    started = time.monotonic()
    payloads: list[dict[str, Any]] = []
    while rounds < max_rounds:
        rounds += 1
        payload, headers = post_round(client, base_url, body, headers)
        payloads.append(payload)
        choices = payload.get("choices") or [{}]
        message = choices[0].get("message") or {}
        if choices[0].get("finish_reason") != "tool_calls":
            return case_observation(plan, payloads, message, rounds - 1, started)
        _apply_tool_results(message, workspace, messages)
    return case_observation(
        plan, payloads, {"content": "EXCEEDED_TOOL_BUDGET"}, max_rounds, started
    )


def build_record(variant: str, case: Any, obs: dict, plan: dict) -> dict:
    """Assemble one summarize-compatible run record."""
    event = obs["event"]
    verdict = auto_judge(case.case_id, obs["content"], obs["tool_rounds"])
    record = {
        "variant": variant,
        "case_id": case.case_id,
        "passed": bool(verdict) if verdict is not None else False,
        "needs_review": verdict is None,
        "selected_models": selected_models(event, plan),
        **usage_numbers(event, obs["response"]),
        "tool_rounds": int(obs["tool_rounds"]),
        "wall_time_s": round(float(obs["wall_time_s"]), 6),
        "report": obs["content"],
    }
    if isinstance(event, dict):
        for key in ("fusion_savings_usd", "main_model", "sidekick_model"):
            if event.get(key) is not None:
                record[key] = event[key]
    return record


def _persist_run(
    record: dict[str, Any],
    observation: dict[str, Any],
    case: Any,
    out_path: Path,
    transcripts_dir: Path,
) -> None:
    """Append one record and save its full transcript."""
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / f"{record['variant']}--{case.case_id}.json"
    transcript_path.write_text(
        json.dumps(
            {"case": case.case_id, "prompt": case.prompt, **observation},
            indent=2,
            default=str,
        )
    )


def run_variant(
    client: httpx.Client,
    base_url: str,
    variant: str,
    cases: list[Any],
    catalog: dict[str, Any],
    out_path: Path,
    transcripts_dir: Path,
    workspace_root: Path,
    max_rounds: int,
) -> int:
    """Execute every case under one variant and append records."""
    plan = request_plan(variant, catalog)
    plan_url = str(plan.get("base_url") or base_url)
    plan_auth = openrouter_headers() if plan.get("base_url") else {}
    count = 0
    for case in cases:
        workspace = workspace_root / variant / case.case_id
        workspace.mkdir(parents=True, exist_ok=True)
        observation = drive_case(
            client, plan_url, plan, case.prompt, workspace, max_rounds, plan_auth
        )
        record = build_record(variant, case, observation, plan)
        _persist_run(record, observation, case, out_path, transcripts_dir)
        count += 1
        print(f"{variant} {case.case_id}: {record['passed']} ({record['provider_cost']:.6f} USD)")
    return count


def build_parser() -> argparse.ArgumentParser:
    """Return the runner CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    base_url = os.environ.get("MANTIS_BASE_URL", "http://127.0.0.1:8000")
    parser.add_argument("--base-url", default=base_url)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--transcripts-dir", type=Path, default=DEFAULT_TRANSCRIPTS)
    parser.add_argument("--workspace-root", type=Path, default=REPO / ".scratch" / "routed-ab")
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the selected variants and write run records."""
    args = build_parser().parse_args(argv)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {', '.join(unknown)}")
    cases = load_cases(args.cases)
    catalog = load_catalog(args.catalog)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    api_key = os.environ.get("MANTIS_API_KEY", "")
    auth = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=args.timeout, headers=auth) as client:
        for variant in variants:
            total += run_variant(
                client,
                args.base_url.rstrip("/"),
                variant,
                cases,
                catalog,
                out_path,
                args.transcripts_dir,
                args.workspace_root,
                args.max_rounds,
            )
    print(f"runs={total} records={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
