"""Calibrated assessment of routed A/B records via the TypeSafe Jev decisions API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx
from ab_evidence import tool_records

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "eval" / "runs" / "routed-ab"
MAX_TOOL_TRANSCRIPT_CHARS = 24_000
ANSWER_CONVENTION = (
    "The orchestration system prefixes its final answers with 'ANSWER:'. "
    "The prefix is presentation, not part of the requested response."
)
QUALITY_LEVELS = [
    "Failed, evaded, or asked for clarification instead of acting.",
    "Completed with wrong scope, gaps, or unverified claims.",
    "Completed and consistent, with cited verification evidence.",
    "Completed, verified, and matches every expected check.",
]
FAILURE_MODES = {
    "none": "No failure; the task was completed.",
    "clarification_stall": "The agent asked for information instead of acting.",
    "wrong_scope": "The agent acted on the wrong file, test, or requirement.",
    "unverified_claims": "The agent claimed completion without evidence.",
    "not_attempted": "No meaningful action was taken.",
}


def load_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Return case definitions keyed by id."""
    cases = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                case = json.loads(line)
                cases[case["id"]] = case
    return cases


def compact_handoffs(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep bounded handoff task, outcome, message, report, and status details."""
    compact = []
    entries = event.get("handoff_history") or []
    for entry in entries[:20]:
        if not isinstance(entry, dict):
            continue
        item = {"kind": _bounded_text(entry.get("kind"), 200)}
        for key in ("task", "outcome", "follow_up", "message", "report", "status"):
            if entry.get(key) is not None:
                value = entry[key]
                item[key] = _bounded_text(value, 2_000) if isinstance(value, str) else value
        compact.append(item)
    return compact


def _transcript_path(record: dict[str, Any], transcripts_dir: Path) -> tuple[Path, int]:
    """Select a seed transcript, permitting old names only for seedless rows."""
    seed_value = record.get("seed")
    seed = int(seed_value) if seed_value is not None else 1
    transcript_file = record.get("transcript_file")
    if isinstance(transcript_file, str) and transcript_file:
        return transcripts_dir / Path(transcript_file).name, seed
    stem = f"{record['variant']}--{record['case_id']}"
    if seed_value is not None:
        return transcripts_dir / f"{stem}--s{seed}.json", seed
    legacy = transcripts_dir / f"{stem}.json"
    seeded = transcripts_dir / f"{stem}--s1.json"
    return (legacy if legacy.is_file() else seeded), seed


def _bounded_json(value: Any) -> str:
    """Serialize evidence within the decision service context budget."""
    text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    marker = "...[truncated]"
    if len(text) <= MAX_TOOL_TRANSCRIPT_CHARS:
        return text
    return text[: MAX_TOOL_TRANSCRIPT_CHARS - len(marker)] + marker


def _bounded_text(value: Any, limit: int = 4_000) -> str:
    """Bound independent check output while preserving its leading evidence."""
    text = str(value or "")
    marker = "...[truncated]"
    return text if len(text) <= limit else text[: limit - len(marker)] + marker


def _tool_transcript(transcript: dict[str, Any]) -> str:
    """Return bounded client and server tool evidence for independent review."""
    return _bounded_json(
        {
            "messages": transcript.get("messages"),
            "response_payloads": transcript.get("response_payloads"),
            "tool_records": tool_records(transcript),
            "server_run_details": transcript.get("server_run_details"),
        }
    )


def _agent_state(
    record: dict[str, Any],
    seed: int,
    final_answer: str,
    event: dict[str, Any],
    transcript: dict[str, Any],
) -> dict[str, Any]:
    """Assemble bounded agent status and independent transcript evidence."""
    return {
        "variant": record["variant"],
        "seed": seed,
        "final_answer": final_answer,
        "tool_rounds": record.get("tool_rounds"),
        "wall_time_s": round(record.get("wall_time_s") or 0.0, 1),
        "check_passed": record.get("check_passed"),
        "check_output": _bounded_text(record.get("check_output")),
        "check_command": record.get("check_command"),
        "evidence_rule": (
            "Recorded tool results and independent checks are evidence; claims alone are not proof."
        ),
        "handoffs": compact_handoffs(event),
        "tool_transcript": _tool_transcript(transcript),
    }


def build_state(
    record: dict[str, Any], case: dict[str, Any], transcripts_dir: Path | None = None
) -> dict[str, Any]:
    """Assemble task, independent checks, and bounded transcript evidence."""
    transcript_dir = transcripts_dir or TRANSCRIPTS
    path, seed = _transcript_path(record, transcript_dir)
    transcript = json.loads(path.read_text()) if path.is_file() else {}
    answer = transcript.get("content") or record.get("report") or record.get("content") or ""
    event = dict(transcript.get("server_run_details") or {})
    event.update(transcript.get("event") or {})
    return {
        "answer_convention": ANSWER_CONVENTION,
        "task": case["prompt"],
        "expected": case["expected"],
        "agent": _agent_state(record, seed, answer, event, transcript),
    }


def questions_for(case: dict[str, Any]) -> dict[str, Any]:
    """Build the success Noul, quality Score, and failure-mode Choice."""
    checks = " ".join(case["expected"]["checks"])
    return {
        "success": {
            "type": "noul",
            "instructions": (
                "Did the agent complete the task as described, satisfying the "
                f"expected checks ({checks})?"
            ),
            "criteria": {
                "true": "The report and history show the task completed with all checks satisfied.",
                "false": "The task was not completed, evaded, stalled, or checks failed.",
            },
        },
        "quality": {
            "type": "score",
            "instructions": "Rate the agent's completion quality for this task.",
            "criteria": QUALITY_LEVELS,
        },
        "failure_mode": {
            "type": "choice",
            "instructions": "Which failure mode best matches, if any?",
            "criteria": FAILURE_MODES,
        },
    }


def assess(client: httpx.Client, body: dict[str, Any]) -> dict[str, Any]:
    """Post one decision request and return the parsed answers payload."""
    response = client.post(DECISIONS_URL, json=body)
    response.raise_for_status()
    return cast(dict[str, Any], response.json())


def parse_args() -> argparse.Namespace:
    """Return the CLI arguments with eval-directory defaults."""
    parser = argparse.ArgumentParser()
    eval_dir = REPO / "eval"
    parser.add_argument("--records", type=Path, default=eval_dir / "results-routed-ab.jsonl")
    parser.add_argument("--cases", type=Path, default=eval_dir / "routed-harness-cases.jsonl")
    parser.add_argument("--out", type=Path, default=eval_dir / "results-jev-assessment.jsonl")
    parser.add_argument("--transcripts", type=Path, default=TRANSCRIPTS)
    return parser.parse_args()


def assessment_entry(
    client: httpx.Client,
    record: dict[str, Any],
    case: dict[str, Any],
    transcripts_dir: Path | None = None,
) -> dict[str, Any]:
    """Assess one record and return the JSONL entry for it."""
    body = {
        "model": MODEL,
        "state": build_state(record, case, transcripts_dir),
        "questions": questions_for(case),
    }
    payload = assess(client, body)
    return {
        "variant": record["variant"],
        "case_id": record["case_id"],
        "seed": int(record.get("seed", 1)),
        "provider_cost": record["provider_cost"],
        "tool_rounds": record.get("tool_rounds"),
        "answers": payload.get("answers"),
        "usage": payload.get("usage"),
    }


def main() -> None:
    """Assess every routed A/B record and write one JSONL entry per record."""
    args = parse_args()
    cases = load_cases(args.cases)
    records = [json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    api_key = os.environ["OPENROUTER_API_KEY"]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    assessed = 0
    with (
        httpx.Client(headers=headers, timeout=60) as client,
        args.out.open("w") as sink,
    ):
        for record in records:
            entry = assessment_entry(client, record, cases[record["case_id"]], args.transcripts)
            sink.write(json.dumps(entry) + "\n")
            sink.flush()
            assessed += 1
            summary = json.dumps(entry["answers"])
            print(f"{record['variant']} {record['case_id']}: {summary[:160]}")
            time.sleep(0.2)
    print(f"assessed={assessed} out={args.out}")


if __name__ == "__main__":
    main()
