#!/usr/bin/env python3
"""Multi-turn Fusion session driver for the fx.sh harness."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, cast

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BRIEF = (
    "In this scratch directory, create hello.py that prints exactly HELLO-FX, "
    "run it with python3, then run python3 -m py_compile hello.py. "
    "Report whether both commands succeeded."
)
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a shell command in the project worktree",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

# Ordered by tool-use suitability; fx picks the first pair that passes probe.
FREE_MAIN_CANDIDATES = (
    "stealth/ox-alpha",
    "openrouter/free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "poolside/laguna-s-2.1:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "openai/gpt-oss-20b:free",
)
FREE_SIDEKICK_CANDIDATES = (
    # OpenRouter stealth provider currently exposes only stealth/ox-alpha (Aug 2026).
    "stealth/ox-alpha",
    "cohere/north-mini-code:free",
    "z-ai/glm-5.2:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "qwen/qwen-2.5-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-20b:free",
)
PRIVACY_POLICY_HINT = (
    "OpenRouter blocked free models for this account (privacy/guardrail policy). "
    "Adjust https://openrouter.ai/settings/privacy to allow free-model routing, "
    "then rerun ./scripts/fx.sh"
)


def _usage_cache_summary(usage: dict[str, Any]) -> dict[str, int]:
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = usage.get("prompt_cache_hit_tokens")
    if cached is None and isinstance(usage.get("prompt_tokens_details"), dict):
        cached = usage["prompt_tokens_details"].get("cached_tokens")
    cached_int = int(cached or 0)
    return {
        "prompt_tokens": prompt,
        "cached_tokens": cached_int,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def _merge_usage(totals: dict[str, int], usage: dict[str, Any]) -> None:
    part = _usage_cache_summary(usage)
    for key, value in part.items():
        totals[key] = totals.get(key, 0) + value


def _wait_for_server(url: str, token: str, timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(
                f"{url}/v1/models",
                headers={"Authorization": f"Bearer {token}"},
                timeout=5.0,
            )
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("mantis API did not become ready in time")


def _start_server(url: str, workdir: Path, token: str, catalog: Path) -> subprocess.Popen[Any]:
    port = url.rsplit(":", 1)[-1].rstrip("/")
    env = os.environ.copy()
    env["MANTIS_RUN_STORE"] = "file"
    env["MANTIS_RUN_DIR"] = str(workdir / "runs")
    env["MANTIS_API_KEY"] = token
    env["AI_ROUTING_CONFIG"] = str(catalog)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'scripts'}:{REPO_ROOT / 'apps' / 'api'}"
    env.setdefault("MANTIS_CACHE_BREAKPOINTS", "1")
    env.setdefault("MANTIS_CACHE_RETENTION", "short")
    (workdir / "runs").mkdir(parents=True, exist_ok=True)

    import model_catalog

    catalog_obj = model_catalog.load_mantis_catalog(catalog)
    if catalog_obj is not None:
        env.update(model_catalog.render_mantis_environment(catalog_obj))

    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.api.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            port,
            "--no-access-log",
        ],
        env=env,
        cwd=str(REPO_ROOT),
    )


def _execute_tool_call(workdir: Path, tool_call: dict[str, Any]) -> dict[str, Any]:
    tool_call_id = str(tool_call.get("id", ""))
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return {"tool_call_id": tool_call_id, "content": "invalid function field", "is_error": True}
    name = function.get("name", "")
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        return {"tool_call_id": tool_call_id, "content": "invalid arguments type", "is_error": True}
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError as exc:
        return {"tool_call_id": tool_call_id, "content": f"invalid arguments: {exc}", "is_error": True}
    if not isinstance(args, dict):
        return {"tool_call_id": tool_call_id, "content": "arguments must be an object", "is_error": True}
    if name != "bash":
        return {"tool_call_id": tool_call_id, "content": f"unknown tool: {name}", "is_error": True}
    command = args.get("command", "")
    if not command or not isinstance(command, str):
        return {"tool_call_id": tool_call_id, "content": "empty or invalid command", "is_error": True}
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=90.0,
        )
    except subprocess.TimeoutExpired:
        return {"tool_call_id": tool_call_id, "content": "tool timed out", "is_error": True}
    except (OSError, ValueError) as exc:
        return {"tool_call_id": tool_call_id, "content": f"exec failed: {exc}", "is_error": True}
    output = (result.stdout or "") + (result.stderr or "")
    if len(output.encode()) > 64 * 1024:
        output = output.encode()[:64 * 1024].decode(errors="ignore") + "\n[truncated]"
    return {
        "tool_call_id": tool_call_id,
        "content": output or "<no output>",
        "is_error": result.returncode != 0,
    }


def _decode_tool_call_id(encoded: str) -> tuple[str, str]:
    if not encoded.startswith("f") or "~" not in encoded:
        raise ValueError("invalid fusion tool_call_id")
    run_id, call_id = encoded[1:].split("~", 1)
    return run_id, call_id


def _chat_turn(
    client: httpx.Client,
    url: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    response = client.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "mantis/fusion",
            "messages": messages,
            "tools": tools,
            "stream": False,
        },
    )
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


def run_chat_multiturn(
    url: str,
    token: str,
    brief: str,
    follow_up: str,
    max_tool_rounds: int,
) -> dict[str, Any]:
    """Drive Fusion through chat/completions: initial brief, tool loop, follow-up user turn."""
    client = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=600.0)
    tools = [BASH_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": brief}]
    totals: dict[str, int] = {}
    turns: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="fx-chat-") as worktree:
        workdir = Path(worktree)
        for turn_index in range(max_tool_rounds + 2):
            data = _chat_turn(client, url, messages, tools)
            usage = data.get("usage") or {}
            _merge_usage(totals, usage)
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            tool_calls = message.get("tool_calls") or []
            turns.append(
                {
                    "turn": turn_index,
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": len(tool_calls),
                    "usage": _usage_cache_summary(usage),
                    "content_preview": str(message.get("content") or "")[:240],
                }
            )
            messages.append(message)
            if not tool_calls:
                break
            for call in tool_calls:
                encoded_id = str(call.get("id", ""))
                run_id, original_id = _decode_tool_call_id(encoded_id)
                result = _execute_tool_call(workdir, {**call, "id": original_id})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": encoded_id,
                        "content": result["content"],
                    }
                )
                turns[-1]["run_id"] = run_id

        # Second user turn: new fusion run, but client history carries prior context.
        messages.append({"role": "user", "content": follow_up})
        data = _chat_turn(client, url, messages, tools)
        usage = data.get("usage") or {}
        _merge_usage(totals, usage)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        turns.append(
            {
                "turn": "follow_up_user",
                "finish_reason": choice.get("finish_reason"),
                "tool_calls": len(message.get("tool_calls") or []),
                "usage": _usage_cache_summary(usage),
                "content_preview": str(message.get("content") or "")[:240],
            }
        )

    client.close()
    cache_rate = (totals["cached_tokens"] / totals["prompt_tokens"]) if totals["prompt_tokens"] else 0.0
    return {
        "mode": "chat_multiturn",
        "turns": turns,
        "usage_total": totals,
        "cache_hit_rate": round(cache_rate, 4),
    }


def run_delegate_multiturn(
    url: str,
    token: str,
    brief: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Drive Fusion native delegate/follow_up with tool execution."""
    client = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=600.0)
    tools = [BASH_TOOL]
    response = client.post(f"{url}/v1/fusion/delegate", json={"brief": brief, "tools": tools})
    response.raise_for_status()
    event = cast(dict[str, Any], response.json())
    run_id = event["run_id"]
    totals: dict[str, int] = {}
    _merge_usage(totals, event.get("usage") or {})
    iterations = 0

    with tempfile.TemporaryDirectory(prefix="fx-delegate-") as worktree:
        workdir = Path(worktree)
        while event.get("status") == "awaiting_tools" and iterations < max_iterations:
            pending = event.get("pending_tool_calls") or []
            results = [_execute_tool_call(workdir, tc) for tc in pending]
            request_id = uuid.uuid4().hex
            response = client.post(
                f"{url}/v1/fusion/follow_up/{run_id}",
                json={"request_id": request_id, "tool_results": results},
            )
            response.raise_for_status()
            event = cast(dict[str, Any], response.json())
            _merge_usage(totals, event.get("usage") or {})
            iterations += 1

    client.close()
    cache_rate = (totals["cached_tokens"] / totals["prompt_tokens"]) if totals["prompt_tokens"] else 0.0
    return {
        "mode": "delegate_multiturn",
        "run_id": run_id,
        "status": event.get("status"),
        "report": event.get("report"),
        "iterations": iterations,
        "usage_total": totals,
        "cache_hit_rate": round(cache_rate, 4),
        "activity": event.get("activity"),
    }


def probe_openrouter(api_key: str, model: str) -> dict[str, Any]:
    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/mallochio/mantis",
            "X-Title": "mantis-fx-harness",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Reply with exactly: FX-OK"}],
            "max_tokens": 16,
        },
        timeout=60.0,
    )
    body = response.json()
    if response.status_code >= 400:
        raise RuntimeError(f"OpenRouter probe failed for {model}: {body}")
    text = str(((body.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    return {"model": model, "reply": text.strip(), "usage": body.get("usage") or {}}


def _select_free_model(api_key: str, candidates: tuple[str, ...], role: str) -> str:
    errors: list[str] = []
    for model in candidates:
        try:
            probe = probe_openrouter(api_key, model)
            print(f"[fx] probe {role} {model}: {probe['reply']!r}")
            return model
        except RuntimeError as exc:
            errors.append(f"{model}: {exc}")
    raise RuntimeError(
        f"No working OpenRouter free model for {role}. Tried:\n"
        + "\n".join(errors)
        + f"\n{PRIVACY_POLICY_HINT}"
    )


def _catalog_with_models(catalog_path: Path, main_model: str, sidekick_model: str) -> Path:
    """Write a temp catalog with the selected upstream models for Fusion slots."""
    text = catalog_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out: list[str] = []
    section: str | None = None
    for line in lines:
        if line.startswith("[") and line.endswith("]"):
            section = line.strip("[]")
        if section == "mantis.workers.gpt-5_6-sol" and line.startswith("upstream_model"):
            out.append(f'upstream_model = "{main_model}"')
            continue
        if section == "mantis.workers.gpt-5_6-luna" and line.startswith("upstream_model"):
            out.append(f'upstream_model = "{sidekick_model}"')
            continue
        out.append(line)
    temp = Path(tempfile.mkdtemp(prefix="fx-catalog-")) / "catalog.toml"
    temp.write_text("\n".join(out) + "\n", encoding="utf-8")
    return temp


def main() -> None:
    parser = argparse.ArgumentParser(description="Fusion fx harness session driver")
    parser.add_argument("--url", default="http://127.0.0.1:5511")
    parser.add_argument("--token", default=os.environ.get("MANTIS_API_KEY", "sk-fx-headless"))
    parser.add_argument("--catalog", type=Path, default=REPO_ROOT / "config/catalog.fusion-openrouter-free.toml")
    parser.add_argument("--brief", default=DEFAULT_BRIEF)
    parser.add_argument(
        "--follow-up",
        default="Without rerunning everything, confirm hello.py still prints HELLO-FX.",
    )
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--managed", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--main-model", default=os.environ.get("FX_MAIN_MODEL", ""))
    parser.add_argument("--sidekick-model", default=os.environ.get("FX_SIDEKICK_MODEL", ""))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required for OpenRouter :free Fusion tests")

    import model_catalog

    base_catalog = args.catalog
    if args.main_model and args.sidekick_model:
        main_model = args.main_model
        sidekick_model = args.sidekick_model
        if not args.skip_probe:
            probe_openrouter(api_key, main_model)
            probe_openrouter(api_key, sidekick_model)
            print(f"[fx] probe main {main_model}: ok")
            print(f"[fx] probe sidekick {sidekick_model}: ok")
        catalog_path = _catalog_with_models(base_catalog, main_model, sidekick_model)
    elif args.main_model:
        main_model = args.main_model
        if not args.skip_probe:
            probe = probe_openrouter(api_key, main_model)
            print(f"[fx] probe main {main_model}: {probe['reply']!r}")
            sidekick_model = _select_free_model(api_key, FREE_SIDEKICK_CANDIDATES, "sidekick")
        else:
            catalog = model_catalog.load_mantis_catalog(base_catalog)
            if catalog is None:
                raise SystemExit(f"catalog failed to load: {base_catalog}")
            sidekick_model = catalog.bindings.workers["gpt-5_6-luna"].upstream_model
        catalog_path = _catalog_with_models(base_catalog, main_model, sidekick_model)
    elif not args.skip_probe:
        main_model = _select_free_model(api_key, FREE_MAIN_CANDIDATES, "main")
        sidekick_model = _select_free_model(api_key, FREE_SIDEKICK_CANDIDATES, "sidekick")
        catalog_path = _catalog_with_models(base_catalog, main_model, sidekick_model)
    else:
        catalog_path = base_catalog
        catalog = model_catalog.load_mantis_catalog(catalog_path)
        if catalog is None:
            raise SystemExit(f"catalog failed to load: {catalog_path}")
        main_model = catalog.bindings.workers["gpt-5_6-sol"].upstream_model
        sidekick_model = catalog.bindings.workers["gpt-5_6-luna"].upstream_model

    catalog = model_catalog.load_mantis_catalog(catalog_path)
    if catalog is None:
        raise SystemExit(f"catalog failed to load: {catalog_path}")
    print(f"[fx] catalog={catalog_path}")
    print(f"[fx] fusion main={main_model} sidekick={sidekick_model}")

    server = None
    workdir = None
    report: dict[str, Any] = {
        "catalog": str(catalog_path),
        "models": {"main": main_model, "sidekick": sidekick_model},
    }
    try:
        if args.managed:
            workdir = Path(tempfile.mkdtemp(prefix="fx-runs-"))
            server = _start_server(args.url, workdir, args.token, catalog_path)
            _wait_for_server(args.url, args.token)

        report["delegate"] = run_delegate_multiturn(
            args.url, args.token, args.brief, args.max_iterations
        )
        print(f"[fx] delegate status={report['delegate']['status']} cache_hit_rate={report['delegate']['cache_hit_rate']}")

        report["chat"] = run_chat_multiturn(
            args.url,
            args.token,
            args.brief,
            args.follow_up,
            args.max_iterations,
        )
        print(f"[fx] chat turns={len(report['chat']['turns'])} cache_hit_rate={report['chat']['cache_hit_rate']}")

        if args.output:
            args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[fx] wrote {args.output}")
        print(json.dumps(report, indent=2))
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server.kill()
        if workdir is not None:
            import shutil

            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
