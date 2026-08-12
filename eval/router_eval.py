"""Budgeted SWE-rebench router evaluation using mini-SWE-agent and Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import secrets
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "eval" / "router_manifest.json"
DEFAULT_PRICES = REPO / "eval" / "model_prices.json"
TIERS = ("cheap", "middle", "expensive")
ARMS = (
    "cheap-only", "middle-only", "expensive-only", "mantis-direct",
    "heuristic", "random-matched", "trinity",
)
DEFAULT_CAPS = {
    "cheap-only": 0.05, "middle-only": 0.25, "expensive-only": 0.85,
    "mantis-direct": 0.80, "heuristic": 0.80, "random-matched": 0.80,
    "trinity": 1.50,
}


class BudgetAbort(RuntimeError):
    """Raised before a request that would exceed a budget."""


class CostLedger:
    def __init__(self, *, total_limit: float, instance_limit: float) -> None:
        self.total_limit = total_limit
        self.instance_limit = instance_limit
        self.total = 0.0
        self.instance = 0.0
        self.aborted = False

    def before_request(self) -> None:
        if self.total >= self.total_limit or self.instance >= self.instance_limit:
            self.aborted = True
            raise BudgetAbort("budget ceiling reached before model request")

    def record(self, cost: float | None) -> None:
        if cost is not None:
            self.total += cost
            self.instance += cost
            if self.total >= self.total_limit or self.instance >= self.instance_limit:
                self.aborted = True
                raise BudgetAbort("budget ceiling reached after model request")

    def begin_instance(self) -> None:
        self.instance = 0.0


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = cast(dict[str, Any], json.loads(path.read_text()))
    if manifest.get("instance_count") != len(manifest.get("instances", [])):
        raise ValueError("manifest instance_count does not match instances")
    return manifest


def load_prices(path: Path) -> dict[str, dict[str, float]]:
    data = cast(dict[str, Any], json.loads(path.read_text()))
    return cast(dict[str, dict[str, float]], data["models"])


def choose_heuristic(prompt: str) -> str:
    words = len(prompt.split())
    terms = ("refactor", "distributed", "concurrency", "architecture", "across")
    if words > 140 or sum(term in prompt.lower() for term in terms) >= 2:
        return "expensive"
    if words > 65 or any(term in prompt.lower() for term in ("implement", "debug", "test")):
        return "middle"
    return "cheap"


def usage_cost(
    usage: dict[str, Any], model: str, prices: dict[str, dict[str, float]]
) -> tuple[float | None, str]:
    if usage.get("cost") is not None:
        return float(usage["cost"]), "usage.cost"
    price = prices.get(model)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if price is None or prompt is None or completion is None:
        return None, "unknown"
    return (
        float(prompt) * price["input_per_token"]
        + float(completion) * price["output_per_token"],
        "token_counts_x_price_table",
    )


def _tier_for_arm(
    arm: str, prompt: str, rng: random.Random, frequencies: dict[str, float]
) -> str:
    if arm.endswith("-only"):
        return arm.removesuffix("-only")
    if arm == "heuristic":
        return choose_heuristic(prompt)
    if arm == "random-matched":
        return rng.choices(list(frequencies), weights=list(frequencies.values()))[0]
    raise ValueError(f"{arm} is not a tier-selection arm")


def _model_for_arm(
    arm: str,
    prompt: str,
    tier_models: dict[str, str],
    rng: random.Random,
    frequencies: dict[str, float],
) -> tuple[str, str | None]:
    if arm in TIERS or arm.endswith("-only") or arm in {"heuristic", "random-matched"}:
        tier = _tier_for_arm(arm, prompt, rng, frequencies)
        return tier_models[tier], tier
    return ("mantis-trinity" if arm == "trinity" else "mantis"), None


def _run_test_command(
    image: str,
    patch: str,
    command: str,
    *,
    test_patch: str = "",
    install: str = "",
    timeout: int = 300,
) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty model patch"
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
        handle.write(f"{test_patch}\n{patch}")
        patch_path = Path(handle.name)
    try:
        setup = f"{install} && " if install else ""
        shell = (
            "source /root/.bashrc && conda activate testbed && "
            f"cd /testbed && git apply /tmp/model.patch && {setup}{command}"
        )
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-v", f"{patch_path}:/tmp/model.patch:ro",
                image, "bash", "-lc", shell,
            ],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return result.returncode == 0, result.stdout + result.stderr
    finally:
        patch_path.unlink(missing_ok=True)


def grade_patch(instance: dict[str, Any], patch: str) -> dict[str, Any]:
    """Grade a patch in a fresh prebuilt image."""
    if not patch.strip():
        return {"resolved": False, "grader_output": "empty model patch"}
    tests = instance.get("FAIL_TO_PASS", []) + instance.get("PASS_TO_PASS", [])
    test_cmd = instance.get("test_cmd", "")
    if tests and test_cmd.startswith("pytest"):
        prefix = test_cmd.split(" tests", 1)[0]
        command = f"{prefix} " + " ".join(shlex.quote(test) for test in tests)
    else:
        command = test_cmd
    passed, output = _run_test_command(
        instance["docker_image"],
        patch,
        command,
        test_patch=instance.get("test_patch", ""),
        install=instance.get("install", ""),
    )
    return {"resolved": passed, "grader_output": output}


def _route_headers(response: Any) -> dict[str, str]:
    headers = getattr(response, "_response_headers", None) or {}
    wanted = (
        "x-route-decision", "x-route-reason", "x-route-model",
        "x-route-sticky", "x-route-fallback",
    )
    return {key: str(headers[key]) for key in wanted if key in headers}


def run_mini_agent(
    instance: dict[str, Any],
    *,
    arm: str,
    endpoint: str,
    tier_models: dict[str, str],
    prices: dict[str, dict[str, float]],
    ledger: CostLedger,
    rng: random.Random,
    step_limit: int,
    output_token_limit: int,
    frequencies: dict[str, float],
) -> dict[str, Any]:
    """Drive mini-SWE-agent in the instance image, then grade its submission."""
    try:
        import litellm
        from minisweagent.agents import get_agent
        from minisweagent.config import builtin_config_dir, get_config_from_spec
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model
        from minisweagent.utils.serialize import recursive_merge
    except ImportError as exc:
        raise RuntimeError("install the eval extra to run mini-SWE-agent") from exc

    model_name, selected_tier = _model_for_arm(
        arm, instance["problem_statement"], tier_models, rng, frequencies
    )
    session = f"router-eval-{secrets.token_hex(8)}"
    config = recursive_merge(
        get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench.yaml"),
        {
            "model": {
                "model_class": "litellm",
                "model_name": model_name,
                "model_kwargs": {
                    "custom_llm_provider": "openai",
                    "api_base": endpoint.removesuffix("/chat/completions"),
                    "max_tokens": output_token_limit,
                    "extra_headers": {"X-Route-Session": session},
                },
                "cost_tracking": "ignore_errors",
            },
            "agent": {
                "mode": "yolo", "step_limit": step_limit,
                "cost_limit": 0, "confirm_exit": False,
            },
            "environment": {
                "environment_class": "docker",
                "image": instance["docker_image"], "cwd": "/testbed",
            },
        },
    )

    class GuardedModel:
        """Proxy mini-SWE-agent's model while enforcing request budgets."""

        def __init__(self, wrapped: Any) -> None:
            self.wrapped = wrapped
            self.config = wrapped.config

        def __getattr__(self, name: str) -> Any:
            return getattr(self.wrapped, name)

        def query(self, messages: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
            ledger.before_request()
            result = cast(dict[str, Any], self.wrapped.query(messages, **kwargs))
            response = result.get("extra", {}).get("response", {})
            usage = response.get("usage", {}) if isinstance(response, dict) else {}
            cost, method = usage_cost(usage, model_name, prices)
            ledger.record(cost)
            result.setdefault("extra", {}).update(
                {
                    "measured_cost": cost,
                    "cost_method": method,
                    "route_headers": getattr(
                        self.wrapped, "last_route_headers", _route_headers(response)
                    ),
                }
            )
            return result

    env = get_environment(config["environment"])
    raw_model = get_model(config=config["model"])
    litellm.register_model(
        {
            model_name: {
                "model_name": model_name,
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        }
    )
    if hasattr(raw_model, "_query"):
        original_query = raw_model._query

        def traced_query(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
            response = original_query(messages, **kwargs)
            raw_model.last_route_headers = _route_headers(response)
            return response

        raw_model._query = traced_query
    model = GuardedModel(raw_model)
    agent = get_agent(model, env, config["agent"], default_type="interactive")
    try:
        info = agent.run(instance["problem_statement"])
        patch = cast(str, info.get("submission", ""))
        grade = grade_patch(instance, patch)
        return {
            "instance_id": instance["instance_id"], "arm": arm,
            "tier": selected_tier, "resolved": grade["resolved"],
            "model_patch": patch, "cost_usd": ledger.instance,
            "trajectory": cast(dict[str, Any], agent.serialize()),
            "route_trace": [
                message.get("extra", {})
                for message in agent.messages
                if message.get("extra", {}).get("route_headers") is not None
            ],
            "grader_output": grade["grader_output"],
        }
    except BudgetAbort as exc:
        return {
            "instance_id": instance["instance_id"], "arm": arm,
            "tier": selected_tier, "resolved": False,
            "aborted": True, "error": str(exc), "cost_usd": ledger.instance,
            "trajectory": cast(dict[str, Any], agent.serialize()),
        }
    finally:
        env.cleanup()


def projected_spend(instance_count: int, arms: list[str], caps: dict[str, float]) -> float:
    return sum(caps[arm] for arm in arms) * instance_count


def dry_run(
    manifest: dict[str, Any], arms: list[str], caps: dict[str, float], prices_path: Path
) -> str:
    total = projected_spend(manifest["instance_count"], arms, caps)
    return "\n".join(
        [
            f"manifest: {manifest['dataset']}@{manifest['dataset_revision']}",
            f"created_at_window: {manifest.get('created_at_window', 'unspecified')}",
            f"instances: {manifest['instance_count']}",
            f"price_table: {prices_path} (separate from caps)",
            "projected worst-case spend (caps only; no model calls):",
            *[
                f"  {arm}: ${caps[arm] * manifest['instance_count']:.2f}"
                for arm in arms
            ],
            f"projected total: ${total:.2f}",
            "model calls: 0 (dry-run)",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    parser.add_argument(
        "--arms",
        default="cheap-only,middle-only,expensive-only,mantis-direct,heuristic,random-matched",
    )
    parser.add_argument("--include-trinity", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=250.0)
    parser.add_argument("--per-instance-cost", type=float, default=5.0)
    parser.add_argument("--step-limit", type=int, default=50)
    parser.add_argument("--output-token-limit", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output", type=Path, default=REPO / "eval" / "runs" / "router-results.jsonl"
    )
    parser.add_argument(
        "--bifrost-endpoint",
        default="http://127.0.0.1:8080/v1/chat/completions",
    )
    parser.add_argument(
        "--mantis-endpoint",
        default="http://127.0.0.1:8088/v1/chat/completions",
    )
    parser.add_argument(
        "--tier-models",
        default="cheap=deepseek-v4-flash,middle=gpt-5.6-terra,expensive=gpt-5.6-sol",
    )
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    prices = load_prices(args.prices)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if args.include_trinity and "trinity" not in arms:
        arms.append("trinity")
    if set(arms) - set(ARMS):
        parser.error(f"unknown arms: {sorted(set(arms) - set(ARMS))}")
    caps = {arm: DEFAULT_CAPS[arm] for arm in arms}
    if args.dry_run:
        print(dry_run(manifest, arms, caps, args.prices))
        return

    tier_models = dict(item.split("=", 1) for item in args.tier_models.split(","))
    rng = random.Random(args.seed)
    ledger = CostLedger(total_limit=args.budget_usd, instance_limit=args.per_instance_cost)
    frequencies = {tier: 1 / len(TIERS) for tier in TIERS}
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "arms": arms, "caps": caps, "budget_usd": args.budget_usd,
        "per_instance_cost": args.per_instance_cost,
        "step_limit": args.step_limit,
        "output_token_limit": args.output_token_limit, "seed": args.seed,
        "aborted_on_budget": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        for instance in manifest["instances"]:
            ledger.begin_instance()
            if ledger.total >= ledger.total_limit:
                metadata["aborted_on_budget"] = True
                break
            rows: list[dict[str, Any]] = []
            for arm in arms:
                endpoint = (
                    args.mantis_endpoint if arm in {"mantis-direct", "trinity"}
                    else args.bifrost_endpoint
                )
                row = run_mini_agent(
                    instance, arm=arm, endpoint=endpoint, tier_models=tier_models,
                    prices=prices, ledger=ledger, rng=rng,
                    step_limit=args.step_limit, output_token_limit=args.output_token_limit,
                    frequencies=frequencies,
                )
                rows.append(row)
                if row.get("aborted"):
                    metadata["aborted_on_budget"] = True
                    break
            if metadata["aborted_on_budget"]:
                break
            for row in rows:
                output.write(json.dumps(row) + "\n")
            output.flush()
            direct_rows = [row for row in rows if row["arm"] == "mantis-direct"]
            observed = [row.get("tier") for row in direct_rows if row.get("tier") in TIERS]
            if observed:
                frequencies = {tier: observed.count(tier) / len(observed) for tier in TIERS}
        output.write(json.dumps({"metadata": {**metadata, "spent_usd": ledger.total}}) + "\n")
    print(json.dumps({**metadata, "spent_usd": ledger.total}, indent=2))


if __name__ == "__main__":
    main()
