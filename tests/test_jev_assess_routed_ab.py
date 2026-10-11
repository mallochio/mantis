"""Tests for seed-aware Jev assessment transcript lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jev_assess_routed_ab as assessor


class FakeClient:
    def __init__(self):
        self.body: dict[str, Any] | None = None

    def post(self, url: str, json: dict[str, Any]) -> FakeResponse:
        self.body = json
        return FakeResponse()


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"answers": {"success": {"noul": 0.75}}}


def test_build_state_uses_seed_specific_transcript(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(assessor, "TRANSCRIPTS", tmp_path)
    record = {"variant": "fusion-tool", "case_id": "easy", "seed": 4, "tool_rounds": 2}
    transcript = {
        "content": "changed the setting",
        "event": {"handoff_history": [{"kind": "sidekick", "task": "edit", "outcome": "done"}]},
    }
    (tmp_path / "fusion-tool--easy--s4.json").write_text(json.dumps(transcript))
    state = assessor.build_state(record, {"prompt": "edit it", "expected": {"checks": ["pass"]}})
    assert state["agent"]["seed"] == 4
    assert state["agent"]["final_answer"] == "changed the setting"
    assert state["agent"]["handoffs"][0]["outcome"] == "done"


def test_assessment_entry_keeps_seed_without_network():
    client = FakeClient()
    record = {
        "variant": "fusion-tool",
        "case_id": "easy",
        "seed": 3,
        "provider_cost": 0.2,
        "tool_rounds": 1,
    }
    case = {"prompt": "make the edit", "expected": {"checks": ["test"]}}
    entry = assessor.assessment_entry(client, record, case)
    assert entry["seed"] == 3
    assert client.body["state"]["agent"]["seed"] == 3
    assert entry["answers"]["success"]["noul"] == 0.75


def test_build_state_uses_report_and_independent_check_evidence(tmp_path: Path):
    record = {
        "variant": "fusion-tool",
        "case_id": "dry-run",
        "seed": 7,
        "report": "The task is done.",
        "check_passed": False,
        "check_output": "validator rejected a missing no-effect branch",
        "check_command": ["independent dry-run acceptance validator"],
    }
    case = {"prompt": "implement it", "expected": {"checks": ["pass"]}}
    state = assessor.build_state(record, case, tmp_path)
    agent = state["agent"]
    assert agent["final_answer"] == record["report"]
    assert agent["check_passed"] is False
    assert "validator rejected" in agent["check_output"]
    assert agent["check_command"] == record["check_command"]
    assert "claims alone are not proof" in agent["evidence_rule"]


def test_seedless_rows_use_legacy_names_but_seeded_rows_do_not(tmp_path: Path):
    case = {"prompt": "task", "expected": {"checks": []}}
    legacy = {"content": "legacy answer"}
    seeded = {"content": "seed answer"}
    (tmp_path / "fusion-tool--edit.json").write_text(json.dumps(legacy))
    (tmp_path / "fusion-tool--edit--s1.json").write_text(json.dumps(seeded))
    old_row = {"variant": "fusion-tool", "case_id": "edit"}
    seed_row = {"variant": "fusion-tool", "case_id": "edit", "seed": 1}
    assert assessor.build_state(old_row, case, tmp_path)["agent"]["final_answer"] == "legacy answer"
    assert assessor.build_state(seed_row, case, tmp_path)["agent"]["final_answer"] == "seed answer"


def test_build_state_bounds_tool_transcript_and_keeps_handoff_evidence(tmp_path: Path):
    record = {"variant": "fusion-tool", "case_id": "edit", "seed": 2}
    transcript = {
        "event": {
            "handoff_history": [
                {
                    "kind": "sidekick",
                    "message": "read file",
                    "report": "done",
                    "status": "completed",
                }
            ]
        },
        "messages": [{"role": "tool", "tool_call_id": "x", "content": "y" * 80_000}],
    }
    (tmp_path / "fusion-tool--edit--s2.json").write_text(json.dumps(transcript))
    state = assessor.build_state(record, {"prompt": "task", "expected": {}}, tmp_path)
    agent = state["agent"]
    assert len(agent["tool_transcript"]) <= assessor.MAX_TOOL_TRANSCRIPT_CHARS
    assert agent["tool_transcript"].endswith("...[truncated]")
    assert agent["handoffs"] == [
        {
            "kind": "sidekick",
            "message": "read file",
            "report": "done",
            "status": "completed",
        }
    ]


def test_cli_accepts_transcripts_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["jev", "--transcripts", str(tmp_path)])
    assert assessor.parse_args().transcripts == tmp_path
