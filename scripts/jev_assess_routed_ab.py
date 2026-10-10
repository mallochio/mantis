"""Calibrated assessment of routed A/B records via the TypeSafe Jev decisions API."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import httpx

DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
REPO = Path(__file__).resolve().parent.parent
TRANSCRIPTS = REPO / "eval" / "runs" / "routed-ab"
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
    """Reduce handoff history entries to kind, task, outcome, and follow-up."""
    compact = []
    for entry in event.get("handoff_history") or []:
        item = {"kind": entry.get("kind")}
        for key in ("task", "outcome", "follow_up"):
            if entry.get(key) is not None:
                item[key] = entry[key]
        compact.append(item)
    return compact


def build_state(record: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Assemble the decision state from a record, its transcript, and the case."""
    transcript_path = TRANSCRIPTS / f"{record['variant']}--{record['case_id']}.json"
    transcript = json.loads(transcript_path.read_text()) if transcript_path.is_file() else {}
    return {
        "answer_convention": ANSWER_CONVENTION,
        "task": case["prompt"],
        "expected": case["expected"],
        "agent": {
            "variant": record["variant"],
            "final_answer": transcript.get("content") or record.get("content") or "",
            "tool_rounds": record.get("tool_rounds"),
            "wall_time_s": round(record.get("wall_time_s") or 0.0, 1),
            "handoffs": compact_handoffs(transcript.get("event") or {}),
        },
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
    return parser.parse_args()


def assessment_entry(
    client: httpx.Client,
    record: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    """Assess one record and return the JSONL entry for it."""
    body = {
        "model": MODEL,
        "state": build_state(record, case),
        "questions": questions_for(case),
    }
    payload = assess(client, body)
    return {
        "variant": record["variant"],
        "case_id": record["case_id"],
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
            entry = assessment_entry(client, record, cases[record["case_id"]])
            sink.write(json.dumps(entry) + "\n")
            sink.flush()
            assessed += 1
            summary = json.dumps(entry["answers"])
            print(f"{record['variant']} {record['case_id']}: {summary[:160]}")
            time.sleep(0.2)
    print(f"assessed={assessed} out={args.out}")


if __name__ == "__main__":
    main()
