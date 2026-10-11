"""Build catalog-based run plans and durable evaluation records."""

from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

from ab_evidence import tool_use_count
from ab_seeds import transcript_name
from evaluate_routed_harness import EvalCase
from routed_ab_tools import OPENROUTER_BASE_URL, priced_cost


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


def _fusion_plan(variant: str, catalog: dict[str, Any], workspace: Path | None) -> dict[str, Any]:
    """Build a fusion request using the catalog's configured worker model."""
    if variant not in {"fusion-pinned", "fusion-tool", "fusion-routed"}:
        raise ValueError(f"unknown variant {variant!r}")
    options: dict[str, Any] = {
        "enabled": ["workspace", "shell"],
        "server_execution": True,
        "delegation": "marker",
    }
    profiles = [{"name": "default", "model": str(catalog["fusion"]["main"])}]
    if variant == "fusion-tool":
        if workspace is None:
            raise ValueError("fusion-tool requires its per-run workspace path")
        options.update({"delegation": "tool", "workspace": str(workspace.resolve())})
    plan: dict[str, Any] = {"model": "mantis/fusion", "tool_options": options}
    if variant != "fusion-routed":
        plan["worker_profiles"] = profiles
    return plan


def request_plan(
    variant: str, catalog: dict[str, Any], workspace: Path | None = None
) -> dict[str, Any]:
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
    return _fusion_plan(variant, catalog, workspace)


def auto_judge(
    case_id: str, content: str, tool_rounds: int, observed_tool_uses: int | None = None
) -> bool | None:
    """Return a machine verdict only when the observed case supports one."""
    if case_id == "no-tool-chat":
        answer = content.strip().removeprefix("ANSWER:").strip()
        no_tools = tool_rounds == 0 and (observed_tool_uses is None or observed_tool_uses == 0)
        return answer == "ROUTING_OK" and no_tools
    return None


def usage_numbers(
    event: dict[str, Any],
    response: dict[str, Any],
    prices: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract run token usage and cost from the response envelope."""
    usage = event.get("usage") if isinstance(event, dict) else None
    if not isinstance(usage, dict) or not usage:
        usage = response.get("usage") if isinstance(response, dict) else {}
    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
    cost = event.get("cost") if isinstance(event, dict) else None
    models = event.get("usage_models") if isinstance(event, dict) else None
    if not isinstance(models, dict):
        models = {}
    if not isinstance(cost, (int, float)) or float(cost) == 0.0:
        cost = priced_cost(usage, models, prices or {})
        basis = "priced_estimate"
    else:
        basis = "upstream_billed"
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "cache_read_tokens": int(cached or 0),
        "provider_cost": float(cost),
        "cost_basis": basis,
    }


def selected_models(event: dict[str, Any], plan: dict[str, Any]) -> list[str]:
    """Return the models a run actually billed."""
    models = event.get("usage_models") if isinstance(event, dict) else None
    if isinstance(models, dict) and models:
        return sorted(str(key) for key in models)
    if isinstance(models, list) and models:
        return sorted(str(item) for item in models)
    return [str(plan["model"])]


def _extra_event_fields(event: dict[str, Any]) -> dict[str, Any]:
    """Return role usage and fusion-specific event fields when available."""
    fields = {}
    if event.get("usage_models"):
        fields["usage_models"] = event["usage_models"]
    for key in ("fusion_savings_usd", "main_model", "sidekick_model"):
        if event.get(key) is not None:
            fields[key] = event[key]
    return fields


def build_record(
    variant: str,
    case: EvalCase,
    obs: dict[str, Any],
    plan: dict[str, Any],
    prices: dict[str, Any] | None = None,
    seed: int = 1,
    checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one summarize-compatible run record."""
    event = obs["event"]
    verdict = auto_judge(case.case_id, obs["content"], obs["tool_rounds"], tool_use_count(obs))
    record = {
        "variant": variant,
        "case_id": case.case_id,
        "seed": seed,
        "passed": bool(verdict) if verdict is not None else False,
        "needs_review": verdict is None,
        "selected_models": selected_models(event, plan),
        **usage_numbers(event, obs["response"], prices),
        "tool_rounds": int(obs["tool_rounds"]),
        "wall_time_s": round(float(obs["wall_time_s"]), 6),
        "report": obs["content"],
    }
    record.update(
        checks or {"check_passed": bool(verdict), "check_output": "", "check_command": []}
    )
    if isinstance(event, dict):
        record.update(_extra_event_fields(event))
    return record


def _persist_run(
    record: dict[str, Any],
    observation: dict[str, Any],
    case: EvalCase,
    out_path: Path,
    transcripts_dir: Path,
) -> None:
    """Append one record and save its full transcript."""
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcripts_dir / str(
        record.get("transcript_file")
        or transcript_name(record["variant"], case.case_id, int(record["seed"]))
    )
    transcript_path.write_text(
        json.dumps(
            {"case": case.case_id, "seed": record["seed"], "prompt": case.prompt, **observation},
            indent=2,
            default=str,
        )
    )


def seed_workspace(case_id: str, workspace: Path, fixtures_root: Path) -> Path:
    """Copy one case's fixture files into its arm workspace without deleting files."""
    fixture_dir = fixtures_root / case_id
    if fixture_dir.is_dir():
        shutil.copytree(fixture_dir, workspace, dirs_exist_ok=True)
    return workspace


def load_prices(path: Path) -> dict[str, Any]:
    """Load the public model price card."""
    with path.open() as handle:
        data: dict[str, Any] = json.load(handle)
    return data
