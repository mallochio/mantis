#!/usr/bin/env python3
"""Pseudo-label coding-session prompts with Gemini via OpenRouter.

Reads a training log produced by llm-router, deduplicates prompts, and uses a
cheap model (Gemini 3.5 Flash Lite by default) to classify each prompt by
difficulty, domain, and whether it is a coding/math/reasoning task. Outputs:

- a JSON label file for inspection
- a Supra-Router-51M-compatible text dataset ready for causal LM fine-tuning

Usage:
    python pseudo_label_gemini.py
    python pseudo_label_gemini.py ~/.local/share/mantis/router/training.jsonl -o labels.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

DEFAULT_MODEL = "google/gemini-3.5-flash-lite"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Prompts may be very long; the tail usually carries the current intent.
DEFAULT_MAX_PROMPT_CHARS = 4_000

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
- Choose "big model" when complexity >= 3, or when coding_task is true with complexity >= 3, or when reasoning/math_task is true with complexity >= 4.

Return ONLY valid JSON, no markdown, no explanation outside the JSON."""


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a possibly fenced/marked response."""
    text = text.strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def truncate_prompt(prompt: str, max_chars: int = DEFAULT_MAX_PROMPT_CHARS) -> str:
    """Keep the tail of the prompt; recent context usually carries intent."""
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
    client: httpx.Client,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call OpenRouter and parse the JSON classification."""
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY or OPENAI_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": build_openai_messages(prompt),
        "temperature": 0,
        "max_tokens": 256,
    }

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.post(f"{OPENROUTER_BASE}/chat/completions", headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            parsed = extract_json(raw)
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
    domain = str(parsed.get("domain", "General")).strip() or "General"
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
        # Fallback to rule used during training.
        if complexity >= 3 or (coding_task and complexity >= 3) or (
            (reasoning or math_task) and complexity >= 4
        ):
            route = "big model"
        else:
            route = "small model"

    return {
        "domain": domain,
        "complexity": complexity,
        "coding_task": coding_task,
        "math_task": math_task,
        "reasoning": reasoning,
        "route": route,
        "analysis": str(parsed.get("analysis", "")).strip(),
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


def save_state(state_path: Path, hashes: set[str]) -> None:
    with state_path.open("w") as f:
        for h in sorted(hashes):
            f.write(f"{h}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pseudo-label router prompts with Gemini via OpenRouter."
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
        default=data_dir / "pseudo-labels-gemini.jsonl",
        help="Output JSON label file",
    )
    parser.add_argument(
        "--supra-output",
        type=Path,
        default=data_dir / "supra-train-gemini.jsonl",
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
        help="Seconds to sleep between requests (helps avoid rate limits)",
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
    args = parser.parse_args()

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
    if args.limit:
        rows = rows[: args.limit]

    print(f"Loaded {len(rows)} new prompts from {input_path}")
    print(f"Model: {args.model}")
    print(f"Writing labels to {output_path} and Supra dataset to {supra_output_path}")

    stats = {"ok": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0}
    latencies: list[int] = []
    new_hashes: set[str] = set()

    def process(item: dict[str, Any]) -> dict[str, Any] | None:
        prompt = truncate_prompt(item["prompt"], args.max_chars)
        t0 = time.time()
        try:
            with httpx.Client(timeout=60) as client:
                result = classify_one(prompt, args.model, client)
        except Exception as exc:
            print(f"ERROR classifying prompt: {exc}")
            stats["errors"] += 1
            return None
        elapsed_ms = int((time.time() - t0) * 1000)
        latencies.append(elapsed_ms)
        label = normalize_label(result["parsed"])
        stats["ok"] += 1
        stats["prompt_tokens"] += result.get("prompt_tokens", 0)
        stats["completion_tokens"] += result.get("completion_tokens", 0)

        record = {
            "hash": item["hash"],
            "model": args.model,
            "prompt": item["prompt"],
            "truncated": len(item["prompt"]) > args.max_chars,
            **label,
            "elapsed_ms": elapsed_ms,
            "raw": result["raw"],
        }
        supra_record = {
            "input": f"Task: {item['prompt']}\nAnalysis: ",
            "target": supra_target_text(label),
            "text": f"Task: {item['prompt']}\nAnalysis: {supra_target_text(label)}",
            "complexity": label["complexity"],
            "route": label["route"],
        }
        return {"record": record, "supra": supra_record, "hash": item["hash"]}

    # Simple sequential or thread-pooled execution.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def write_batch(batch: list[dict[str, Any]]) -> None:
        with output_path.open("a", encoding="utf-8") as jf, supra_output_path.open(
            "a", encoding="utf-8"
        ) as sf:
            for item in batch:
                jf.write(json.dumps(item["record"]) + "\n")
                sf.write(json.dumps(item["supra"]) + "\n")

    batch: list[dict[str, Any]] = []
    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process, item): item for item in rows}
            for future in as_completed(futures):
                out = future.result()
                if out:
                    batch.append(out)
                    new_hashes.add(out["hash"])
                if args.delay:
                    time.sleep(args.delay)
                if len(batch) >= 10:
                    write_batch(batch)
                    state.update(new_hashes)
                    save_state(state_path, state)
                    batch.clear()
                    new_hashes.clear()
    else:
        for item in rows:
            out = process(item)
            if out:
                batch.append(out)
                new_hashes.add(out["hash"])
            if args.delay:
                time.sleep(args.delay)
            if len(batch) >= 10:
                write_batch(batch)
                state.update(new_hashes)
                save_state(state_path, state)
                batch.clear()
                new_hashes.clear()

    if batch:
        write_batch(batch)
        state.update(new_hashes)
        save_state(state_path, state)

    print("\nDone.")
    print(f"Labeled: {stats['ok']}, errors: {stats['errors']}")
    if latencies:
        print(f"Latency: median {statistics.median(latencies)}ms, max {max(latencies)}ms")
    print(f"Tokens: prompt={stats['prompt_tokens']}, completion={stats['completion_tokens']}")
    print(
        f"Estimated 3.5 Flash Lite cost (OpenRouter): "
        f"${(stats['prompt_tokens'] * 0.30 / 1e6 + stats['completion_tokens'] * 2.50 / 1e6):.4f}"
    )


if __name__ == "__main__":
    main()
