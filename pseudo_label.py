#!/usr/bin/env python3
"""Pseudo-label coding-session prompts for the llm-router using a cheap model.

Reads the router training log, deduplicates prompts, and uses a small model via
OpenRouter (Gemini 3.5 Flash Lite by default) to classify each prompt by
difficulty, domain, and whether it is a coding/math/reasoning task. Outputs:

- a JSON label file for inspection
- a Supra-Router-51M-compatible text dataset ready for causal LM fine-tuning

Usage:
    python pseudo_label.py
    python pseudo_label.py ~/.local/share/mantis/router/training.jsonl -o labels.jsonl
    python pseudo_label.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
# Batch API lives under /api/beta (not /api/v1).
OPENROUTER_BATCH_BASE = "https://openrouter.ai/api/beta"
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Prompts may be very long; the tail usually carries the current intent.
DEFAULT_MAX_PROMPT_CHARS = 4_000
DOMAINS = {"Programming", "Communication", "Math", "Science", "Creative", "General", "Business"}
LABEL_SCHEMA = {
    "name": "router_label", "strict": True,
    "schema": {"type": "object", "additionalProperties": False,
        "required": ["domain", "complexity", "coding_task", "math_task", "reasoning", "route", "analysis"],
        "properties": {
            "domain": {"type": "string", "enum": sorted(DOMAINS)},
            "complexity": {"type": "integer", "minimum": 1, "maximum": 5},
            "coding_task": {"type": "boolean"}, "math_task": {"type": "boolean"},
            "reasoning": {"type": "boolean"},
            "route": {"type": "string", "enum": ["small model", "big model"]},
            "analysis": {"type": "string", "maxLength": 300},
        }},
}

# Classification prompt shared across all calls. We ask for JSON plus a free-form
# `analysis` string that mirrors the format Supra-Router-51M was trained on.
CLASSIFICATION_PROMPT = """You are a prompt router for a coding-agent system.
Analyze the user request and return ONLY a compact JSON object with these fields:

- domain: one of "Programming", "Communication", "Math", "Science", "Creative", "General", "Business"
- complexity: integer 1 (trivial) to 5 (very hard)
- coding_task: boolean
- math_task: boolean
- reasoning: boolean (multi-step reasoning, proof, architecture)
- route: "small model" or "big model" (small for cheap local models, big for capable frontier models)
- analysis: a short one-sentence explanation of the decision

Rules:
- complexity 1: greetings, single facts, one-line requests, simple conversions.
- complexity 2: short explanations, simple one-function coding tasks.
- complexity 3: standard coding/debugging/algorithms, short design discussions.
- complexity 4: refactoring, multi-file changes, concurrency, parsers, API design, integration work.
- complexity 5: distributed systems, architecture, formal reasoning, large rewrites, complex proofs.
- coding_task is true if the prompt asks for code, debugging, refactoring, tests, or software design.
- math_task is true if it requires math, statistics, numerical proof, or complex calculations.
- reasoning is true if it requires multi-step logic, proofs, or system design.
- Set complexity higher when coding_task, math_task, or reasoning is true.
- Choose "big model" when complexity is 3 or higher.

Return ONLY valid JSON, no markdown, no explanation outside the JSON."""

_thread_local = threading.local()


def _get_thread_client() -> httpx.Client:
    """Return one httpx Client per worker thread so connections are reused."""
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = httpx.Client(timeout=60)
        _thread_local.client = client
    return client


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object from text, ignoring prose/fences."""
    text = text.strip()
    # Strip fenced code block if present.
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    in_string = False
    escaping = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if escaping:
            escaping = False
            continue
        if ch == "\\":
            escaping = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    start = -1
            if depth < 0:
                depth = 0
    return None


def truncate_prompt(prompt: str, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> str:
    """Keep the tail of the prompt; recent context usually carries the intent."""
    if len(prompt) <= max_chars:
        return prompt
    return "..." + prompt[-(max_chars - 3) :]


def load_prompts(path: Path, state: set[str]) -> list[dict[str, Any]]:
    """Load and deduplicate training rows that contain a non-empty prompt."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if h in seen or h in state:
                continue
            seen.add(h)
            rows.append({"row": row, "hash": h, "prompt": prompt})
    return rows


def build_openai_messages(prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": CLASSIFICATION_PROMPT},
        {"role": "user", "content": f"Prompt to classify:\n{prompt}"},
    ]


def classify_one(
    prompt: str,
    model: str,
    delay: float = 0.0,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call OpenRouter and parse the JSON classification."""
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY not set")

    if delay:
        time.sleep(delay)

    client = _get_thread_client()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": build_openai_messages(prompt),
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {"type": "json_schema", "json_schema": LABEL_SCHEMA},
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = extract_first_json_object(raw)
            if parsed is None:
                raise ValueError(f"no JSON in response: {raw[:200]!r}")
            usage = data.get("usage", {})
            return {
                "raw": raw,
                "parsed": parsed,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    raise last_error or RuntimeError("classification failed")


def normalize_label(parsed: dict[str, Any]) -> dict[str, Any]:
    """Sanitize model output into a consistent schema."""
    domain = str(parsed.get("domain", "General")).strip()
    if domain not in DOMAINS:
        domain = "General"
    complexity = parsed.get("complexity")
    try:
        complexity = max(1, min(5, int(complexity)))
    except (TypeError, ValueError):
        complexity = 3

    def _bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.lower() in {"true", "yes", "1", "y"}
        return False

    coding_task = _bool(parsed.get("coding_task"))
    math_task = _bool(parsed.get("math_task"))
    reasoning = _bool(parsed.get("reasoning"))

    route = str(parsed.get("route", "")).strip().lower()
    if route not in {"small model", "big model"}:
        # Simple deterministic fallback: big when complexity >= 3.
        route = "big model" if complexity >= 3 else "small model"

    return {
        "domain": domain,
        "complexity": complexity,
        "coding_task": coding_task,
        "math_task": math_task,
        "reasoning": reasoning,
        "route": route,
        "analysis": " ".join(str(parsed.get("analysis", "")).split())[:300],
    }


def supra_target_text(label: dict[str, Any]) -> str:
    """Format a label the way Supra-Router-51M expects."""
    return (
        f"Domain: {label['domain']} | "
        f"Complexity: {label['complexity']} | "
        f"Math: {label['math_task']} | "
        f"Code: {label['coding_task']} | "
        f"Route: {label['route']}"
    )


def load_state(state_path: Path) -> set[str]:
    """Load hashes of already processed prompts."""
    if not state_path.exists():
        return set()
    with state_path.open("r") as f:
        return {line.strip() for line in f if line.strip()}


def _ensure_0600(path: Path) -> None:
    """Create or chmod a file so private prompt data is not world-readable."""
    if not path.exists():
        path.touch(mode=0o600)
    path.chmod(0o600)


def save_state(state_path: Path, hashes: set[str]) -> None:
    """Atomically checkpoint only records already fsynced to both outputs."""
    state_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    tmp = state_path.with_name(state_path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for value in sorted(hashes):
            handle.write(f"{value}\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, state_path)
    state_path.chmod(0o600)


def write_batch(
    output_path: Path,
    supra_output_path: Path,
    batch: list[dict[str, Any]],
) -> None:
    _ensure_0600(output_path)
    _ensure_0600(supra_output_path)
    with output_path.open("a", encoding="utf-8") as jf, supra_output_path.open(
        "a", encoding="utf-8"
    ) as sf:
        for item in batch:
            jf.write(json.dumps(item["record"]) + "\n")
            sf.write(json.dumps(item["supra"]) + "\n")
        jf.flush()
        sf.flush()
        os.fsync(jf.fileno())
        os.fsync(sf.fileno())


def build_records(item, parsed, model, elapsed_ms, raw="") -> tuple[dict, dict]:
    label = normalize_label(parsed)
    record = {
        "hash": item["hash"],
        "model": model,
        "prompt": item["prompt"],
        "truncated_prompt": truncate_prompt(item["prompt"], DEFAULT_MAX_PROMPT_CHARS),
        "truncated": len(item["prompt"]) > DEFAULT_MAX_PROMPT_CHARS,
        **label,
        "elapsed_ms": elapsed_ms,
        "raw": raw,
    }
    supra_record = {
        "input": f"Task: {truncate_prompt(item['prompt'], DEFAULT_MAX_PROMPT_CHARS)}\nAnalysis: ",
        "target": supra_target_text(label),
        "text": f"Task: {truncate_prompt(item['prompt'], DEFAULT_MAX_PROMPT_CHARS)}\nAnalysis: {supra_target_text(label)}",
        "complexity": label["complexity"],
        "route": label["route"],
    }
    return record, supra_record


def process(
    item: dict[str, Any],
    model: str,
    max_chars: int,
    delay: float,
) -> dict[str, Any] | None:
    """Classify one prompt and return records plus per-item statistics."""
    truncated = truncate_prompt(item["prompt"], max_chars)
    t0 = time.time()
    try:
        result = classify_one(truncated, model, delay=delay)
    except Exception as exc:
        print(f"ERROR classifying prompt: {exc}")
        return None
    elapsed_ms = int((time.time() - t0) * 1000)
    record, supra_record = build_records(item, result["parsed"], model, elapsed_ms, result.get("raw", ""))
    return {
        "record": record,
        "supra": supra_record,
        "hash": item["hash"],
        "prompt_tokens": result.get("prompt_tokens", 0),
        "completion_tokens": result.get("completion_tokens", 0),
        "elapsed_ms": elapsed_ms,
    }


BATCH_TIMEOUT_S = 3600


def run_batch(
    rows: list[dict[str, Any]],
    model: str,
    output_path: Path,
    supra_output_path: Path,
    state_path: Path,
    state: set[str],
) -> None:
    """Label all pending prompts via OpenRouter's inline batch API (~50% cost)."""
    if not rows:
        print("Nothing to label.")
        return
    requests = [
        {
            "custom_id": item["hash"],
            "body": {
                "messages": build_openai_messages(truncate_prompt(item["prompt"], DEFAULT_MAX_PROMPT_CHARS)),
                "temperature": 0,
                "max_tokens": 256,
                "response_format": {"type": "json_schema", "json_schema": LABEL_SCHEMA},
            },
        }
        for item in rows
    ]
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{OPENROUTER_BATCH_BASE}/batches",
            headers=headers,
            json={"endpoint": "/v1/chat/completions", "model": model, "requests": requests},
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        print(f"Submitted batch {batch_id} with {len(requests)} requests", flush=True)
        deadline = time.time() + BATCH_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(30)
            st = client.get(f"{OPENROUTER_BATCH_BASE}/batches/{batch_id}", headers=headers).json()
            status = st.get("status")
            print(f"  batch {status}: {st.get('request_counts')}", flush=True)
            if status == "completed":
                items = {item["hash"]: item for item in rows}
                ok = err = 0
                batch: list[dict] = []
                for r in st.get("results") or []:
                    item = items.get(r.get("custom_id"))
                    if item is None:
                        continue
                    if r.get("error"):
                        err += 1
                        print(f"ERROR batch item {r['custom_id']}: {r['error']}", flush=True)
                        continue
                    content = r["response"]["body"]["choices"][0]["message"]["content"]
                    parsed = extract_first_json_object(content)
                    if parsed is None:
                        err += 1
                        print(f"ERROR no JSON in {r['custom_id']}: {content[:120]!r}", flush=True)
                        continue
                    record, supra_record = build_records(item, parsed, model, 0, raw=content)
                    batch.append({"record": record, "supra": supra_record})
                    state.add(item["hash"])
                    ok += 1
                    if len(batch) >= 32:
                        write_batch(output_path, supra_output_path, batch)
                        save_state(state_path, state)
                        batch = []
                if batch:
                    write_batch(output_path, supra_output_path, batch)
                save_state(state_path, state)
                print(f"Done. Labeled: {ok}, errors: {err}", flush=True)
                return
            if status in ("failed", "expired", "cancelled", "cancelling"):
                raise RuntimeError(f"batch {batch_id} ended with status {status}")
        raise RuntimeError(f"batch {batch_id} did not finish within {BATCH_TIMEOUT_S}s")


def demo() -> None:
    """Quick self-test that exercises normalize_label and supra formatting."""
    sample = {
        "domain": "Programming",
        "complexity": 1,
        "coding_task": True,
        "math_task": False,
        "reasoning": False,
        "route": "small model",
        "analysis": "Trivial one-function coding request.",
    }
    assert normalize_label(sample) == sample
    assert "Route: small model" in supra_target_text(sample)
    hard = {"complexity": 4, "coding_task": True}
    assert normalize_label(hard)["route"] == "big model"
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudo-label router prompts with a cheap OpenRouter model."
    )
    data_dir = Path.home() / ".local/share/mantis/router"
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=data_dir / "training.jsonl",
        help="Router training log (default: ~/.local/share/mantis/router/training.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=data_dir / "pseudo-labels.jsonl",
        help="Output JSON label file",
    )
    parser.add_argument(
        "--supra-output",
        type=Path,
        default=data_dir / "supra-train.jsonl",
        help="Supra-Router-51M training dataset output",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenRouter model ID for the labeler",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_PROMPT_CHARS,
        help="Max characters of prompt to send for classification",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep before each OpenRouter request (per worker)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Label via OpenRouter's inline batch API (~50%% cost, async)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent requests (not recommended on free/low tier)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only label the first N prompts",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a local self-test without calling OpenRouter",
    )
    args = parser.parse_args()

    if args.self_test:
        demo()
        return

    input_path: Path = args.input
    output_path: Path = args.output
    supra_output_path: Path = args.supra_output

    if not API_KEY:
        parser.error("OPENROUTER_API_KEY or OPENAI_API_KEY must be set")
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    supra_output_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = output_path.with_suffix(".state")
    state = load_state(state_path)

    rows = load_prompts(input_path, state)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} new prompts from {input_path}")
    print(f"Model: {args.model}")
    print(f"Writing labels to {output_path} and Supra dataset to {supra_output_path}")

    if args.batch:
        batch_model = args.model if args.model.endswith(":batch") else f"{args.model}:batch"
        run_batch(rows, batch_model, output_path, supra_output_path, state_path, state)
        return

    stats = {"ok": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
    latencies: list[int] = []
    batch: list[dict[str, Any]] = []
    batch_size = 10

    def process_one(item: dict[str, Any]) -> dict[str, Any] | None:
        out = process(item, args.model, args.max_chars, args.delay)
        return out

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_one, item): item for item in rows}
            for future in as_completed(futures):
                try:
                    out = future.result()
                except Exception as exc:
                    print(f"ERROR in worker: {exc}")
                    stats["errors"] += 1
                    continue
                if out is None:
                    stats["errors"] += 1
                    continue
                stats["ok"] += 1
                stats["prompt_tokens"] += out["prompt_tokens"]
                stats["completion_tokens"] += out["completion_tokens"]
                latencies.append(out["elapsed_ms"])
                batch.append({"record": out["record"], "supra": out["supra"], "hash": out["hash"]})
                if len(batch) >= batch_size:
                    write_batch(output_path, supra_output_path, batch)
                    state.update(entry["hash"] for entry in batch)
                    save_state(state_path, state)
                    batch.clear()
    else:
        for item in rows:
            out = process_one(item)
            if out is None:
                stats["errors"] += 1
                continue
            stats["ok"] += 1
            stats["prompt_tokens"] += out["prompt_tokens"]
            stats["completion_tokens"] += out["completion_tokens"]
            latencies.append(out["elapsed_ms"])
            batch.append({"record": out["record"], "supra": out["supra"], "hash": out["hash"]})
            if len(batch) >= batch_size:
                write_batch(output_path, supra_output_path, batch)
                state.update(entry["hash"] for entry in batch)
                save_state(state_path, state)
                batch.clear()

    if batch:
        write_batch(output_path, supra_output_path, batch)
        state.update(entry["hash"] for entry in batch)
        save_state(state_path, state)

    print("\nDone.")
    print(f"Labeled: {stats['ok']}, errors: {stats['errors']}")
    if latencies:
        print(f"Latency: median {statistics.median(latencies)}ms, max {max(latencies)}ms")
    print(f"Tokens: prompt={stats['prompt_tokens']}, completion={stats['completion_tokens']}")
    if args.model == DEFAULT_MODEL:
        cost = stats["prompt_tokens"] * 0.30 / 1e6 + stats["completion_tokens"] * 2.50 / 1e6
        print(f"Estimated {DEFAULT_MODEL} cost (OpenRouter): ${cost:.4f}")


if __name__ == "__main__":
    main()
