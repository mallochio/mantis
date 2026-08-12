"""Budgeted SWE-rebench router evaluation using pi (headless) and Docker.

The agent driving each instance is `pi` in headless mode (`--mode json`,
`-p`). pi is pointed at a local header-recording proxy that forwards chat
completions to Bifrost / the Mantis gateway; the proxy records `x-route-*`
response headers, per-request usage, and enforces the arm/global budget
ceiling by refusing to forward requests once a cap is reached.

The model endpoints are OpenAI-compatible (Bifrost `:8080`, Mantis `:8088`),
so no litellm or mini-swe-agent dependency is involved.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import random
import secrets
import shlex
import subprocess
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import requests

REPO = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO / "eval" / "router_manifest.json"
DEFAULT_PRICES = REPO / "eval" / "model_prices.json"
DEFAULT_WORKTREES = REPO / "eval" / "runs" / "worktrees"
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
ROUTE_HEADER_KEYS = (
    "x-route-decision", "x-route-reason", "x-route-model",
    "x-route-sticky", "x-route-fallback",
)
PI_TOOLS = "read,bash,edit,write"


class BudgetAbort(RuntimeError):
    """Raised before a request that would exceed a budget."""


class ArmBudgetExceeded(BudgetAbort):
    """A single instance/arm reached its configured ceiling."""


class CostLedger:
    def __init__(self, *, total_limit: float, arm_limit: float) -> None:
        self.total_limit = total_limit
        self.arm_limit = arm_limit
        self.total = 0.0
        self.pair_costs: dict[tuple[str, str], float] = {}
        self.aborted = False
        self._lock = threading.Lock()

    def before_request(self, instance_id: str, arm: str, cap: float) -> None:
        with self._lock:
            pair = self.pair_costs.get((instance_id, arm), 0.0)
            if self.total >= self.total_limit:
                self.aborted = True
                raise BudgetAbort("budget ceiling reached before model request")
            if pair >= cap:
                raise ArmBudgetExceeded(f"per-arm cap reached for {instance_id}/{arm}")

    def record(self, instance_id: str, arm: str, cost: float | None, cap: float) -> None:
        if cost is None:
            return
        with self._lock:
            self.total += cost
            key = (instance_id, arm)
            self.pair_costs[key] = self.pair_costs.get(key, 0.0) + cost
            if self.total >= self.total_limit:
                self.aborted = True
                raise BudgetAbort("global budget reached after model request")
            if self.pair_costs[key] >= cap:
                raise ArmBudgetExceeded(f"per-arm cap reached for {instance_id}/{arm}")

    def pair_cost(self, instance_id: str, arm: str) -> float:
        with self._lock:
            return self.pair_costs.get((instance_id, arm), 0.0)


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


def routed_request_tiers(
    row: dict[str, Any], tier_models: dict[str, str]
) -> list[str]:
    model_to_tier = {model: tier for tier, model in tier_models.items()}
    tiers = []
    for event in row.get("route_trace", []):
        headers = event.get("route_headers", {})
        tier = model_to_tier.get(headers.get("x-route-model"))
        if tier is None and headers.get("x-route-decision") in TIERS:
            tier = headers["x-route-decision"]
        if tier in TIERS:
            tiers.append(tier)
    if not tiers:
        raise ValueError(f"mantis-direct row has no usable route trace: {row.get('instance_id')}")
    return tiers


def _run_test_command(
    image: str,
    patch: str,
    command: str,
    *,
    test_patch: str = "",
    base_commit: str = "",
    reset_paths: list[str] | None = None,
    install: str = "",
    timeout: int = 300,
) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty model patch"
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
        handle.write(patch)
        patch_path = Path(handle.name)
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as handle:
        handle.write(test_patch)
        test_path = Path(handle.name)
    try:
        setup = f"{install} && " if install else ""
        reset = ""
        if base_commit and reset_paths:
            reset = f"git checkout {shlex.quote(base_commit)} -- " + " ".join(
                shlex.quote(path) for path in reset_paths
            ) + " && "
        shell = (
            "source /root/.bashrc && conda activate testbed && "
            f"cd /testbed && git apply /tmp/model.patch && {reset}"
            "git apply /tmp/test.patch && " + f"{setup}{command}"
        )
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{patch_path}:/tmp/model.patch:ro",
                "-v", f"{test_path}:/tmp/test.patch:ro",
                image, "bash", "-lc", shell,
            ],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
        return result.returncode == 0, result.stdout + result.stderr
    finally:
        patch_path.unlink(missing_ok=True)
        test_path.unlink(missing_ok=True)


def _test_paths(test_patch: str) -> list[str]:
    return sorted({
        line[6:].split("\t", 1)[0]
        for line in test_patch.splitlines()
        if line.startswith("+++ b/")
    })


def _test_command(instance: dict[str, Any], tests: list[str]) -> str:
    tokens = shlex.split(instance.get("test_cmd", ""))
    if not tokens or tokens[0] != "pytest":
        raise ValueError("test_cmd is not a pytest command")
    split_at = next(
        (i for i, token in enumerate(tokens[1:], 1) if token.startswith("tests/")),
        None,
    )
    if split_at is None or not tests:
        raise ValueError("test_cmd has no test-root token")
    return " ".join(shlex.quote(token) for token in tokens[:split_at] + tests)


def grade_patch(instance: dict[str, Any], patch: str) -> dict[str, Any]:
    """Grade a patch in a fresh prebuilt image."""
    if not patch.strip():
        return {"resolved": False, "grader_output": "empty model patch"}
    f2p = list(instance.get("FAIL_TO_PASS", []))
    p2p = list(instance.get("PASS_TO_PASS", []))
    try:
        commands = {"fail_to_pass": _test_command(instance, f2p),
                    "pass_to_pass": _test_command(instance, p2p)}
    except ValueError as exc:
        return {"resolved": False, "grader_error": str(exc)}
    common = {
        "image": instance["docker_image"], "patch": patch,
        "test_patch": instance.get("test_patch", ""),
        "base_commit": instance.get("base_commit", ""),
        "reset_paths": _test_paths(instance.get("test_patch", "")),
        "install": instance.get("install", ""),
    }
    results = {}
    for name, command in commands.items():
        passed, output = _run_test_command(command=command, **common)
        results[name] = {"passed": passed, "output": output}
    return {
        "resolved": all(result["passed"] for result in results.values()),
        "test_results": results,
        "grader_output": "\n".join(
            cast(str, result["output"]) for result in results.values()
        ),
    }


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )


def _ensure_worktree(instance: dict[str, Any], root: Path) -> tuple[Path, Path | None, bool]:
    """Materialize the instance's repo at base_commit as a host worktree.

    Returns (worktree_path, cache_repo, created). A pre-existing worktree (as
    tests provide) is reused; otherwise the repo is cloned (blob-less) into a
    cache and a detached worktree is added at the base commit.
    """
    repo = instance["repo"]
    commit = instance["base_commit"]
    instance_id = instance["instance_id"]
    workdir = root / instance_id
    if workdir.exists():
        _git("checkout", "--force", commit, cwd=workdir)
        _git("reset", "--hard", commit, cwd=workdir)
        return workdir, None, False
    cache = root / "cache" / repo.replace("/", "__")
    if not (cache / ".git").is_dir():
        cache.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             f"https://github.com/{repo}.git", str(cache)],
            capture_output=True, text=True, check=True,
        )
    else:
        _git("-C", str(cache), "fetch", "origin")
    workdir.parent.mkdir(parents=True, exist_ok=True)
    _git("-C", str(cache), "worktree", "add", "--detach", str(workdir), commit)
    return workdir, cache, True


def _remove_worktree(workdir: Path, cache: Path) -> None:
    _git("-C", str(cache), "worktree", "remove", "--force", str(workdir))
    _git("-C", str(cache), "worktree", "prune")


def _worktree_patch(workdir: Path) -> str:
    _git("add", "-A", cwd=workdir)
    return _git("diff", "HEAD", cwd=workdir).stdout


def _usage_from_sse(text: str) -> dict[str, Any]:
    """Pull the last `usage` object from an OpenAI-style SSE stream."""
    usage: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        with contextlib.suppress(ValueError):
            event = json.loads(payload)
            if isinstance(event, dict) and event.get("usage"):
                usage = event["usage"]
    return usage


def _usage_from_response(resp: requests.Response) -> dict[str, Any]:
    if "text/event-stream" in resp.headers.get("Content-Type", ""):
        return _usage_from_sse(resp.text)
    with contextlib.suppress(ValueError):
        return resp.json().get("usage", {}) or {}
    return {}


class _ProxyHandler(BaseHTTPRequestHandler):
    """Forward chat completions to the upstream while recording route headers
    and enforcing the arm/global budget ceiling."""

    protocol_version = "HTTP/1.1"

    def __init__(self, *args: Any, proxy: _RouteRecordingProxy, **kwargs: Any) -> None:
        self.proxy = proxy
        super().__init__(*args, **kwargs)

    def log_message(self, *_args: Any) -> None:
        pass

    def do_GET(self) -> None:
        self._error(404, "eval proxy only forwards chat completions")

    def do_POST(self) -> None:
        proxy = self.proxy
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            with proxy.lock:
                proxy.ledger.before_request(proxy.instance_id, proxy.arm, proxy.cap)
        except ArmBudgetExceeded as exc:
            with proxy.lock:
                proxy.exceeded = True
            self._error(429, str(exc))
            return
        except BudgetAbort as exc:
            with proxy.lock:
                proxy.aborted = True
            self._error(429, str(exc))
            return
        headers = {"Content-Type": "application/json"}
        if proxy.api_key:
            headers["Authorization"] = f"Bearer {proxy.api_key}"
        session = self.headers.get("X-Route-Session") or proxy.session
        if session:
            headers["X-Route-Session"] = session
        url = proxy.upstream
        try:
            resp = requests.post(url, data=body, headers=headers, timeout=300)
        except requests.RequestException as exc:
            self._error(502, f"{type(exc).__name__}: {exc}")
            return
        payload = resp.content
        route = {
            key: resp.headers[key]
            for key in ROUTE_HEADER_KEYS
            if resp.headers.get(key)
        }
        usage = _usage_from_response(resp)
        cost, method = usage_cost(usage, proxy.model, proxy.prices)
        with proxy.lock:
            proxy.records.append(
                {"route_headers": route, "usage": usage, "cost": cost,
                 "cost_method": method}
            )
            try:
                proxy.ledger.record(proxy.instance_id, proxy.arm, cost, proxy.cap)
            except ArmBudgetExceeded:
                proxy.exceeded = True
            except BudgetAbort:
                proxy.aborted = True
        self.send_response(resp.status_code)
        self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
        for key in ROUTE_HEADER_KEYS:
            if resp.headers.get(key):
                self.send_header(key, resp.headers[key])
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, code: int, message: str) -> None:
        body = json.dumps({"error": {"message": message, "type": "eval_budget"}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _RouteRecordingProxy:
    def __init__(
        self,
        *,
        upstream: str,
        api_key: str | None,
        ledger: CostLedger,
        instance_id: str,
        arm: str,
        cap: float,
        prices: dict[str, dict[str, float]],
        model: str,
        session: str,
    ) -> None:
        self.upstream = upstream
        self.api_key = api_key
        self.ledger = ledger
        self.instance_id = instance_id
        self.arm = arm
        self.cap = cap
        self.prices = prices
        self.model = model
        self.session = session
        self.records: list[dict[str, Any]] = []
        self.exceeded = False
        self.aborted = False
        self.lock = threading.Lock()
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(_ProxyHandler, proxy=self)
        )
        self.port = self._server.server_port
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def _render_provider_extension(
    models: list[str], base_url: str, api_key: str, session: str,
    max_tokens: int, out_path: Path,
) -> None:
    """Write a pi extension registering an `eval` provider pointed at the proxy."""
    lines = [
        'import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";',
        "export default function (pi: ExtensionAPI) {",
        '  pi.registerProvider("eval", {',
        '    name: "Mantis eval gateway",',
        f'    baseUrl: "{base_url}",',
        f'    apiKey: "{api_key}",',
        '    api: "openai-completions",',
        f'    headers: {{ "X-Route-Session": "{session}" }},',
        "    models: [",
    ]
    for model in models:
        lines += [
            "      {",
            f'        id: "{model}",',
            f'        name: "{model}",',
            "        reasoning: false,",
            '        input: ["text"],',
            "        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },",
            "        contextWindow: 262144,",
            f"        maxTokens: {max_tokens},",
            "      },",
        ]
    lines += ["    ],", "  });", "}"]
    out_path.write_text("\n".join(lines) + "\n")
    out_path.chmod(0o600)


def _parse_pi_events(stdout: str) -> list[dict[str, Any]]:
    """Extract a compact trajectory from pi's `--mode json` event stream."""
    trajectory: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        event_type = event.get("type")
        if event_type == "message_end":
            message = event.get("message", {})
            if message.get("role") == "assistant":
                trajectory.append(
                    {
                        "role": "assistant",
                        "content": message.get("content"),
                        "usage": message.get("usage"),
                        "stop_reason": message.get("stopReason"),
                    }
                )
        elif event_type == "tool_execution_end":
            trajectory.append(
                {
                    "tool": event.get("toolName"),
                    "args": event.get("args"),
                    "result": event.get("result"),
                    "is_error": event.get("isError"),
                }
            )
    return trajectory


def run_pi_agent(
    instance: dict[str, Any],
    *,
    arm: str,
    endpoint: str,
    tier_models: dict[str, str],
    prices: dict[str, dict[str, float]],
    ledger: CostLedger,
    arm_cap: float,
    rng: random.Random,
    timeout: int,
    output_token_limit: int,
    frequencies: dict[str, float],
    worktrees_root: Path | None = None,
    pi_executable: tuple[str, ...] = ("pi",),
    keep_worktrees: bool = False,
) -> dict[str, Any]:
    """Drive `pi` headless in a host worktree, then grade its patch.

    pi is pointed at a local proxy that forwards to `endpoint` (Bifrost or the
    Mantis gateway), records `x-route-*` headers and usage, and stops
    forwarding once the arm cap or global budget is reached.
    """
    root = worktrees_root or DEFAULT_WORKTREES
    instance_id = instance["instance_id"]
    model_name, selected_tier = _model_for_arm(
        arm, instance["problem_statement"], tier_models, rng, frequencies
    )
    session = f"router-eval-{secrets.token_hex(8)}"
    api_key = os.environ.get("BIFROST_API_KEY") or os.environ.get("MANTIS_API_KEY")
    proxy = _RouteRecordingProxy(
        upstream=endpoint, api_key=api_key, ledger=ledger, instance_id=instance_id,
        arm=arm, cap=arm_cap, prices=prices, model=model_name, session=session,
    )
    workdir, cache, created = _ensure_worktree(instance, root)
    ext_path = Path(tempfile.mkdtemp(prefix="pi-eval-")) / "eval-provider.ts"
    error: str | None = None
    abort_scope: str | None = None
    patch = ""
    trajectory: list[dict[str, Any]] = []
    try:
        models = sorted(set(tier_models.values()) | {"mantis", "mantis-trinity"})
        _render_provider_extension(
            models, proxy.base_url(), api_key or "", session, output_token_limit,
            ext_path,
        )
        cmd = [
            *pi_executable,
            "-e", str(ext_path),
            "--provider", "eval",
            "--model", f"eval/{model_name}",
            "--mode", "json",
            "--no-session",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-context-files",
            "--no-approve",
            "--offline",
            "--tools", PI_TOOLS,
            "-p", instance["problem_statement"],
        ]
        result = subprocess.run(
            cmd, cwd=str(workdir), capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "EVAL_PROXY_BASE_URL": proxy.base_url()},
        )
        trajectory = _parse_pi_events(result.stdout)
        if result.returncode != 0:
            error = f"pi exited {result.returncode}: {result.stderr.strip()[:200]}"
        if proxy.exceeded:
            abort_scope = "instance_arm"
        if proxy.aborted or ledger.aborted:
            abort_scope = "global"
        if error is None and abort_scope is None:
            patch = _worktree_patch(workdir)
        grade = (
            grade_patch(instance, patch)
            if error is None and abort_scope is None
            else None
        )
        row: dict[str, Any] = {
            "instance_id": instance_id, "arm": arm,
            "tier": selected_tier, "model": model_name,
            "resolved": bool(grade and grade["resolved"]),
            "model_patch": patch,
            "cost_usd": ledger.pair_cost(instance_id, arm),
            "cost_method": proxy.records[-1]["cost_method"] if proxy.records else None,
            "trajectory": trajectory,
            "route_trace": proxy.records,
        }
        if grade is not None:
            row["grader_output"] = grade["grader_output"]
        if abort_scope is not None:
            row["aborted"] = True
            row["abort_scope"] = abort_scope
            row["error"] = error or "budget ceiling reached"
        if error is not None and abort_scope is None:
            row["error"] = error
    except subprocess.TimeoutExpired:
        return {
            "instance_id": instance_id, "arm": arm, "tier": selected_tier,
            "model": model_name, "resolved": False, "model_patch": "",
            "cost_usd": ledger.pair_cost(instance_id, arm),
            "trajectory": trajectory, "route_trace": proxy.records,
            "error": f"pi timed out after {timeout}s", "aborted": True,
            "abort_scope": "timeout",
        }
    except Exception as exc:  # noqa: BLE001 - an eval run keeps going on failure
        return {
            "instance_id": instance_id, "arm": arm, "tier": selected_tier,
            "model": model_name, "resolved": False, "model_patch": "",
            "cost_usd": ledger.pair_cost(instance_id, arm),
            "trajectory": trajectory, "route_trace": proxy.records,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        return row
    finally:
        proxy.close()
        ext_path.unlink(missing_ok=True)
        if created and not keep_worktrees and cache is not None:
            with contextlib.suppress(subprocess.CalledProcessError):
                _remove_worktree(workdir, cache)


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
    parser.add_argument(
        "--per-instance-cost", type=float, default=None,
        help="override every per-instance/arm cap (legacy alias)",
    )
    parser.add_argument(
        "--arm-cap", action="append", default=[],
        metavar="ARM=USD", help="override one arm's per-instance cap",
    )
    parser.add_argument(
        "--timeout", type=int, default=600,
        help="wall-clock seconds per (instance, arm) pi run",
    )
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
    parser.add_argument(
        "--worktrees-root", type=Path, default=DEFAULT_WORKTREES,
        help="host worktree root for per-instance checkouts (gitignored)",
    )
    parser.add_argument("--keep-worktrees", action="store_true")
    parser.add_argument("--pi-executable", default="pi")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    prices = load_prices(args.prices)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    if args.include_trinity and "trinity" not in arms:
        arms.append("trinity")
    if set(arms) - set(ARMS):
        parser.error(f"unknown arms: {sorted(set(arms) - set(ARMS))}")
    if "random-matched" in arms and "mantis-direct" not in arms:
        parser.error("random-matched requires mantis-direct for its observed distribution")
    caps = {arm: DEFAULT_CAPS[arm] for arm in arms}
    if args.per_instance_cost is not None:
        caps = dict.fromkeys(arms, args.per_instance_cost)
    for override in args.arm_cap:
        arm, value = override.split("=", 1)
        if arm not in caps:
            parser.error(f"cannot cap disabled arm: {arm}")
        caps[arm] = float(value)
    if args.dry_run:
        print(dry_run(manifest, arms, caps, args.prices))
        return

    tier_models = dict(item.split("=", 1) for item in args.tier_models.split(","))
    rng = random.Random(args.seed)
    ledger = CostLedger(total_limit=args.budget_usd, arm_limit=max(caps.values()))
    frequencies = {tier: 1 / len(TIERS) for tier in TIERS}
    pi_executable = tuple(args.pi_executable.split())
    metadata = {
        "manifest": str(args.manifest),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "arms": arms, "caps": caps, "budget_usd": args.budget_usd,
        "per_arm_caps": caps,
        "timeout": args.timeout,
        "output_token_limit": args.output_token_limit, "seed": args.seed,
        "pi_executable": args.pi_executable,
        "worktrees_root": str(args.worktrees_root),
        "aborted_on_budget": False, "excluded_instances": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        instances = manifest["instances"]
        direct_rows: dict[str, dict[str, Any]] = {}
        for instance in instances:
            if ledger.total >= ledger.total_limit:
                metadata["aborted_on_budget"] = True
                break
            row = run_pi_agent(
                instance, arm="mantis-direct", endpoint=args.mantis_endpoint,
                tier_models=tier_models, prices=prices, ledger=ledger,
                arm_cap=caps["mantis-direct"], rng=rng,
                timeout=args.timeout, output_token_limit=args.output_token_limit,
                frequencies={tier: 1 / len(TIERS) for tier in TIERS},
                worktrees_root=args.worktrees_root, pi_executable=pi_executable,
                keep_worktrees=args.keep_worktrees,
            )
            output.write(json.dumps(row) + "\n")
            output.flush()
            if row.get("abort_scope") == "instance_arm":
                metadata["excluded_instances"].append(instance["instance_id"])
                continue
            if row.get("aborted"):
                metadata["aborted_on_budget"] = True
                break
            direct_rows[instance["instance_id"]] = row
        if not metadata["aborted_on_budget"]:
            observed = [
                tier
                for row in direct_rows.values()
                for tier in routed_request_tiers(row, tier_models)
            ]
            if not observed:
                raise RuntimeError("mantis-direct produced no route decisions")
            frequencies = {tier: observed.count(tier) / len(observed) for tier in TIERS}
            remaining_arms = [arm for arm in arms if arm != "mantis-direct"]
            for instance in instances:
                instance_id = instance["instance_id"]
                if instance_id in metadata["excluded_instances"]:
                    continue
                rows = [direct_rows[instance_id]]
                for arm in remaining_arms:
                    endpoint = (
                        args.mantis_endpoint if arm == "trinity"
                        else args.bifrost_endpoint
                    )
                    row = run_pi_agent(
                        instance, arm=arm, endpoint=endpoint,
                        tier_models=tier_models, prices=prices, ledger=ledger,
                        arm_cap=caps[arm],
                        rng=rng, timeout=args.timeout,
                        output_token_limit=args.output_token_limit,
                        frequencies=frequencies,
                        worktrees_root=args.worktrees_root, pi_executable=pi_executable,
                        keep_worktrees=args.keep_worktrees,
                    )
                    rows.append(row)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    if row.get("abort_scope") == "instance_arm":
                        metadata["excluded_instances"].append(instance_id)
                        break
                    if row.get("aborted"):
                        metadata["aborted_on_budget"] = True
                        break
                if metadata["aborted_on_budget"]:
                    break
        output.write(json.dumps({"metadata": {**metadata, "spent_usd": ledger.total}}) + "\n")
    print(json.dumps({**metadata, "spent_usd": ledger.total}, indent=2))


if __name__ == "__main__":
    main()
