"""Hermetic-first SWE-rebench router evaluation harness.

The command can execute against OpenAI-compatible endpoints, but tests use a
local canned server. Never invoke a real run without an explicit budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
from pathlib import Path
from typing import Any, cast

import httpx

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "eval" / "router_manifest.json"
TIERS = ("cheap", "middle", "expensive")
ARM_CAPS = {
    "cheap-only": 0.05,
    "middle-only": 0.25,
    "expensive-only": 0.85,
    "mantis-direct": 0.80,
    "heuristic": 0.80,
    "random-matched": 0.80,
    "trinity": 1.50,
}


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads(path.read_text()))
    if not manifest.get("instances") or manifest["instance_count"] != len(manifest["instances"]):
        raise ValueError("manifest instance_count does not match instances")
    return manifest


def choose_heuristic(prompt: str) -> str:
    words = len(prompt.split())
    complex_terms = ("refactor", "distributed", "concurrency", "architecture", "across")
    if words > 140 or sum(term in prompt.lower() for term in complex_terms) >= 2:
        return "expensive"
    if words > 65 or any(term in prompt.lower() for term in ("implement", "debug", "test")):
        return "middle"
    return "cheap"


def _usage_cost(data: dict[str, Any], model: str, prices: dict[str, float]) -> tuple[float, str]:
    usage = data.get("usage") or {}
    if usage.get("cost") is not None:
        return float(usage["cost"]), "usage.cost"
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    price = prices.get(model, 0.0)
    return (prompt + completion) / 1000 * price, "token_counts_x_catalog_prices"


def _headers(response: httpx.Response) -> dict[str, str]:
    names = (
        "x-route-decision",
        "x-route-reason",
        "x-route-model",
        "x-route-sticky",
        "x-route-fallback",
    )
    return {name: response.headers[name] for name in names if name in response.headers}


def _model_for_arm(arm: str, tier_models: dict[str, str]) -> tuple[str, str | None]:
    if arm in TIERS or arm.endswith("-only") or arm in {"heuristic", "random-matched"}:
        tier = arm.removesuffix("-only")
        return tier_models[tier], tier
    if arm == "trinity":
        return "mantis-trinity", None
    return "mantis", None


def run_instance(
    instance: dict[str, Any],
    *,
    arm: str,
    client: httpx.Client,
    endpoint: str,
    tier_models: dict[str, str],
    prices: dict[str, float],
    rng: random.Random,
    max_steps: int,
    max_output_tokens: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Run one instance and preserve every request's routing trace."""
    selected = arm
    if arm == "heuristic":
        selected = choose_heuristic(instance["problem_statement"])
    elif arm == "random-matched":
        selected = rng.choices(list(TIERS), weights=[0.55, 0.30, 0.15])[0]
    model, pinned = _model_for_arm(selected, tier_models)
    session = f"router-eval-{secrets.token_hex(8)}"
    decisions: list[dict[str, Any]] = []
    total_cost = 0.0
    methods: list[str] = []
    text = prompt or instance["problem_statement"]
    try:
        for step in range(max_steps):
            headers = {"content-type": "application/json"}
            if arm in {"mantis-direct", "trinity"}:
                headers["X-Route-Session"] = session
            response = client.post(
                endpoint,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": max_output_tokens,
                },
            )
            response.raise_for_status()
            data = response.json()
            cost, method = _usage_cost(data, model, prices)
            total_cost += cost
            methods.append(method)
            decisions.append(
                {
                    "step": step + 1,
                    "cost_usd": cost,
                    "cost_method": method,
                    "headers": _headers(response),
                }
            )
            text = str(
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            )
            if data.get("choices", [{}])[0].get("finish_reason", "stop") == "stop":
                break
        return {
            "instance_id": instance["instance_id"],
            "arm": arm,
            "tier": pinned or selected,
            "resolved": bool(text.strip()),
            "response_text": text,
            "cost_usd": total_cost,
            "cost_method": (methods[0] if len(set(methods)) == 1 else "mixed"),
            "cost_methods": sorted(set(methods)),
            "routing_decisions": decisions,
            "session": (session if arm in {"mantis-direct", "trinity"} else None),
        }
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        return {
            "instance_id": instance["instance_id"],
            "arm": arm,
            "tier": pinned or selected,
            "resolved": False,
            "error": str(exc),
            "cost_usd": total_cost,
            "cost_method": (methods[0] if len(set(methods)) == 1 else "mixed"),
            "cost_methods": sorted(set(methods)),
            "routing_decisions": decisions,
        }


def projected_spend(instance_count: int, arms: list[str]) -> float:
    return sum(ARM_CAPS[arm] for arm in arms) * instance_count


def dry_run(manifest: dict[str, Any], arms: list[str]) -> str:
    total = projected_spend(manifest["instance_count"], arms)
    lines = [
        f"manifest: {manifest['dataset']}@{manifest['dataset_revision']}",
        f"instances: {manifest['instance_count']}",
        f"arms: {', '.join(arms)}",
        "projected worst-case spend (per-instance caps):",
    ]
    lines.extend(f"  {arm}: ${ARM_CAPS[arm] * manifest['instance_count']:.2f}" for arm in arms)
    lines.append(f"projected total: ${total:.2f}")
    lines.append("model calls: 0 (dry-run)")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SWE-rebench router harness.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--arms",
        default="cheap-only,middle-only,expensive-only,mantis-direct,heuristic,random-matched",
    )
    parser.add_argument("--include-trinity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=250.0)
    parser.add_argument("--per-instance-cost", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=REPO / "eval" / "runs" / "router-results.jsonl"
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--mantis-endpoint", default="http://127.0.0.1:8088/v1/chat/completions")
    parser.add_argument(
        "--tier-models",
        default="cheap=deepseek-v4-flash,middle=gpt-5.6-terra,expensive=gpt-5.6-sol",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if args.include_trinity and "trinity" not in arms:
        arms.append("trinity")
    invalid = set(arms) - set(ARM_CAPS)
    if invalid:
        parser.error(f"unknown arms: {sorted(invalid)}")
    if args.dry_run:
        print(dry_run(manifest, arms))
        return

    tier_models = dict(item.split("=", 1) for item in args.tier_models.split(","))
    prices = {
        model: cap / 3_000
        for model, cap in (
            ("deepseek-v4-flash", ARM_CAPS["cheap-only"]),
            ("gpt-5.6-terra", ARM_CAPS["middle-only"]),
            ("gpt-5.6-sol", ARM_CAPS["expensive-only"]),
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    rng = random.Random(args.seed)
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "arms": arms,
        "budget_usd": args.budget_usd,
        "per_instance_cost": args.per_instance_cost,
        "max_steps": args.max_steps,
        "max_output_tokens": args.max_output_tokens,
        "seed": args.seed,
        "aborted_on_budget": False,
    }
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        with httpx.Client(timeout=300) as client:
            for instance in manifest["instances"]:
                for arm in arms:
                    if spent >= args.budget_usd:
                        metadata["aborted_on_budget"] = True
                        break
                    row = run_instance(
                        instance,
                        arm=arm,
                        client=client,
                        endpoint=(
                            args.mantis_endpoint
                            if arm in {"mantis-direct", "trinity"}
                            else args.endpoint
                        ),
                        tier_models=tier_models,
                        prices=prices,
                        rng=rng,
                        max_steps=args.max_steps,
                        max_output_tokens=args.max_output_tokens,
                    )
                    if row["cost_usd"] > args.per_instance_cost:
                        row["error"] = "per-instance cost ceiling exceeded"
                    spent += row["cost_usd"]
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                if metadata["aborted_on_budget"]:
                    break
        output.write(json.dumps({"metadata": {**metadata, "spent_usd": spent}}) + "\n")
    print(json.dumps({**metadata, "spent_usd": spent, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
