"""Tests for trusted evaluation validators and transcript evidence."""

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


def _seed_workspace(case_id: str, root: Path) -> Path:
    workspace = root / case_id
    workspace.mkdir(parents=True)
    return runner.seed_workspace(case_id, workspace, runner.DEFAULT_FIXTURES)


def test_live_runner_uses_only_tool_fusion():
    catalog = runner.load_catalog(runner.DEFAULT_CATALOG)
    tool = runner.request_plan("fusion-tool", catalog, Path.cwd())
    assert "worker_profiles" not in tool
    assert "delegation" not in tool["tool_options"]
    assert "fusion-pinned" not in runner.LIVE_VARIANTS
    assert "fusion-routed" not in runner.LIVE_VARIANTS


def _prepare_force_run(
    tmp_path: Path, case: EvalCase
) -> tuple[Path, Path, Path, Path, set[tuple[str, str, int]]]:
    root = tmp_path / "workspaces"
    base = workspace_path(root, "fusion-tool", case.case_id, 1)
    base.mkdir(parents=True)
    marker = base / "old-state.txt"
    marker.write_text("preserve")
    output = tmp_path / "rows.jsonl"
    row = {"variant": "fusion-tool", "case_id": case.case_id, "seed": 1}
    output.write_text(json.dumps(row) + "\n")
    return root, base, marker, output, runner.completed_runs(output)


def _run_force(
    client: FakeClient,
    case: EvalCase,
    output: Path,
    root: Path,
    tmp_path: Path,
    done: set[tuple[str, str, int]],
) -> int:
    return runner.run_variant(
        client,
        "http://localhost",
        "fusion-tool",
        [case],
        runner.load_catalog(runner.DEFAULT_CATALOG),
        output,
        tmp_path / "transcripts",
        root,
        runner.DEFAULT_FIXTURES,
        2,
        seeds=1,
        force=True,
        completed=done,
    )


def test_force_uses_and_records_a_fresh_seed_attempt(tmp_path: Path):
    case = _case()
    root, base, marker, output, done = _prepare_force_run(tmp_path, case)
    assert _run_force(FakeClient(), case, output, root, tmp_path, done) == 1
    run = json.loads(output.read_text().splitlines()[-1])
    attempt = Path(run["workspace_path"])
    assert attempt.parent == base and attempt.name.startswith("attempt-")
    assert marker.read_text() == "preserve"
    assert run["transcript_file"].endswith(f"--{attempt.name}.json")
    assert (tmp_path / "transcripts" / run["transcript_file"]).is_file()


def test_immutable_checker_ignores_mutable_workspace_checker(tmp_path: Path):
    workspace = _seed_workspace("easy-mechanical-edit", tmp_path)
    (workspace / "check_settings.py").write_text('print("CONFIG_OK")\n')
    result = run_case_checks("easy-mechanical-edit", workspace, "", [], 0)
    assert result["check_passed"] is False
    assert "MISMATCH server" in result["check_output"]


def _enable_orchestration(workspace: Path) -> None:
    source = workspace / "mantis_router" / "orchestration.py"
    text = source.read_text()
    original = '    if op not in OPS:\n        raise ValueError(f"unsupported op {op!r}")\n'
    dry_run = original + '    if step.get("dry_run") is True:\n        return None\n'
    source.write_text(text.replace(original, dry_run))


def _enable_gateway(workspace: Path) -> None:
    gateway = workspace / "mantis_router" / "gateway.py"
    gateway.write_text(
        '"""Gateway layer: validate a request and forward it to orchestration."""\n\n'
        "from mantis_router import orchestration\n\n\n"
        "def dispatch(request: dict) -> dict:\n"
        '    """Return the response envelope for one routed request."""\n'
        '    if not isinstance(request, dict) or "op" not in request:\n'
        '        return {"status": "error", "reason": "request must be a dict with an op"}\n'
        "    result = orchestration.run_step(request)\n"
        '    if request.get("dry_run") is True:\n'
        '        return {"status": "dry_run", "result": None}\n'
        '    return {"status": "ok", "result": result}\n'
    )


def _add_dry_run_tests(workspace: Path) -> None:
    tests = workspace / "tests" / "test_router.py"
    marker = '\n\nif __name__ == "__main__":\n'
    additions = (
        "    def test_dry_run_returns_no_operation_result(self):\n"
        '        step = {"op": "upper", "value": "x", "dry_run": True}\n'
        '        self.assertIsNone(gateway.dispatch(step)["result"])\n\n'
        "    def test_dry_run_returns_its_envelope(self):\n"
        '        step = {"op": "upper", "value": "x", "dry_run": True}\n'
        '        expected = {"status": "dry_run", "result": None}\n'
        "        self.assertEqual(gateway.dispatch(step), expected)\n"
    )
    tests.write_text(tests.read_text().replace(marker, "\n\n" + additions + marker))


def _document_dry_run(workspace: Path) -> None:
    readme = workspace / "README.md"
    readme.write_text(
        readme.read_text() + "\nRequests with `dry_run: true` validate without executing.\n"
    )


def _implement_dry_run(workspace: Path) -> None:
    _enable_orchestration(workspace)
    _enable_gateway(workspace)
    _add_dry_run_tests(workspace)
    _document_dry_run(workspace)


def test_ambiguous_fixture_validator_fails_pristine_and_accepts_correct_edit(tmp_path: Path):
    workspace = _seed_workspace("ambiguous-feature", tmp_path)
    pristine = run_case_checks("ambiguous-feature", workspace, "", [], 0)
    assert pristine["check_passed"] is False
    _implement_dry_run(workspace)
    assert run_case_checks("ambiguous-feature", workspace, "", [], 0)["check_passed"] is True


def test_ambiguous_fixture_validator_detects_removed_no_effect_branch(tmp_path: Path):
    workspace = _seed_workspace("ambiguous-feature", tmp_path)
    _implement_dry_run(workspace)
    module = workspace / "mantis_router" / "orchestration.py"
    module.write_text(
        module.read_text().replace('    if step.get("dry_run") is True:\n        return None\n', "")
    )
    result = run_case_checks("ambiguous-feature", workspace, "", [], 0)
    assert result["check_passed"] is False
    assert "operation executed during dry_run" in result["check_output"]


def test_immutable_reuse_checker_ignores_mutable_workspace_checker(tmp_path: Path):
    workspace = _seed_workspace("upstream-reuse", tmp_path)
    (workspace / "check_reuse.py").write_text('print("REUSE_OK")\n')
    result = run_case_checks("upstream-reuse", workspace, "", [], 0)
    assert result["check_passed"] is False
    assert "REUSE_FAILED" in result["check_output"]
