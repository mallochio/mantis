"""Tests for routed A/B seed expansion, resumption, and fixture checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import run_routed_ab as runner
from ab_seeds import run_case_checks, workspace_path
from evaluate_routed_harness import EvalCase


class FakeResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.headers: dict[str, str] = {}

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeClient:
    def __init__(self):
        self.requests: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str]) -> FakeResponse:
        self.requests.append(json)
        return FakeResponse(
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "done"}}],
                "usage": {},
            }
        )


def _case() -> EvalCase:
    return EvalCase(
        "easy-mechanical-edit",
        "easy_mechanical_edit",
        "rename the label",
        {"description": "rename", "checks": ["check settings"]},
    )


def _run(
    client: FakeClient, tmp_path: Path, completed: set[tuple[str, str, int]], seeds: int
) -> int:
    return runner.run_variant(
        client,
        "http://localhost",
        "fusion-tool",
        [_case()],
        runner.load_catalog(runner.DEFAULT_CATALOG),
        tmp_path / "rows.jsonl",
        tmp_path / "transcripts",
        tmp_path / "workspaces",
        runner.DEFAULT_FIXTURES,
        2,
        seeds=seeds,
        completed=completed,
    )


def test_seed_expansion_preserves_per_seed_workspaces_and_transcripts(tmp_path: Path):
    client = FakeClient()
    count = _run(client, tmp_path, set(), 3)
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert count == 3
    assert [row["seed"] for row in rows] == [1, 2, 3]
    for seed in range(1, 4):
        workspace = workspace_path(tmp_path / "workspaces", "fusion-tool", _case().case_id, seed)
        assert (workspace / "settings.toml").is_file()
        transcript = tmp_path / "transcripts" / f"fusion-tool--easy-mechanical-edit--s{seed}.json"
        assert transcript.is_file()
        assert client.requests[seed - 1]["tool_options"]["workspace"] == str(workspace.resolve())
        assert rows[seed - 1]["check_passed"] is False
        assert rows[seed - 1]["check_command"] == ["python check_settings.py"]


def test_resume_skips_existing_seed_without_deleting_workspace(tmp_path: Path):
    out = tmp_path / "rows.jsonl"
    row = {"variant": "fusion-tool", "case_id": _case().case_id, "seed": 2}
    out.write_text(json.dumps(row) + "\n")
    root = tmp_path / "workspaces"
    preserved = workspace_path(root, "fusion-tool", _case().case_id, 2)
    preserved.mkdir(parents=True)
    marker = preserved / "agent-state.txt"
    marker.write_text("preserve me")
    done = runner.completed_runs(out)
    client = FakeClient()
    assert _run(client, tmp_path, done, 3) == 2
    assert len(client.requests) == 2
    assert marker.read_text() == "preserve me"
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert [record.get("seed") for record in records] == [2, 1, 3]


def test_fixture_hard_check_fails_pristine_and_passes_intended_edit(tmp_path: Path):
    workspace = runner.seed_workspace(
        "easy-mechanical-edit", tmp_path / "fixture", runner.DEFAULT_FIXTURES
    )
    pristine = run_case_checks("easy-mechanical-edit", workspace, "", [], 0)
    assert pristine["check_passed"] is False
    settings = workspace / "settings.toml"
    settings.write_text(settings.read_text().replace("legacy_dispatch", "routed_dispatch"))
    edited = run_case_checks("easy-mechanical-edit", workspace, "", [], 0)
    assert edited["check_passed"] is True
    assert "CONFIG_OK" in edited["check_output"]
    assert edited["check_command"] == ["python check_settings.py"]
