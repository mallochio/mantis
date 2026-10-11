"""Seed evaluation workspaces and run machine-verifiable case checks."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from ab_evidence import call_command, exit_code, result_text, tool_records, tool_use_count
from ab_validation import validate_dry_run

REPO = Path(__file__).resolve().parent.parent
FIXTURES_ROOT = REPO / "eval" / "routed-harness-fixtures"
FIXTURE_CHECK_RUNNER = REPO / "scripts" / "ab_fixture_check.py"
DEFAULT_CHECKS = REPO / "eval" / "routed-harness-checks.json"
ERROR_LINE = "UPSTREAM_TIMEOUT: upstream service did not respond within 500ms"


def workspace_path(root: Path, variant: str, case_id: str, seed: int) -> Path:
    """Return the preserved workspace path for one run."""
    return root / variant / case_id / f"seed{seed}"


def execution_workspace(
    root: Path, variant: str, case_id: str, seed: int, force: bool = False
) -> Path:
    """Return a fresh execution directory when the standard seed path is occupied."""
    base = workspace_path(root, variant, case_id, seed)
    if force or base.exists():
        return base / f"attempt-{uuid.uuid4().hex}"
    return base


def transcript_name(variant: str, case_id: str, seed: int, attempt: str | None = None) -> str:
    """Return the canonical seed transcript filename with optional attempt identity."""
    stem = f"{variant}--{case_id}--s{seed}"
    return f"{stem}{'--' + attempt if attempt else ''}.json"


def load_checks(path: Path = DEFAULT_CHECKS) -> dict[str, list[dict[str, Any]]]:
    """Read per-case command checks."""
    data = json.loads(path.read_text())
    return {str(case_id): list(commands) for case_id, commands in data.items()}


def completed_runs(path: Path) -> set[tuple[str, str, int]]:
    """Read resume keys, treating seedless historical rows as seed one."""
    if not path.is_file():
        return set()
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {(str(row["variant"]), str(row["case_id"]), int(row.get("seed", 1))) for row in rows}


def _run_command(command: str, workspace: Path, case_id: str) -> tuple[int, str]:
    """Run one check, sourcing named checker code from the immutable fixture."""
    arguments = shlex.split(command)
    if len(arguments) == 2 and arguments[0] == "python":
        checker = FIXTURES_ROOT / case_id / arguments[1]
        if checker.is_file() and checker.name.startswith("check_"):
            arguments = [sys.executable, str(FIXTURE_CHECK_RUNNER), str(checker)]
    try:
        result = subprocess.run(
            arguments, cwd=workspace, capture_output=True, text=True, timeout=180
        )
    except subprocess.TimeoutExpired as error:
        output = _decode_output(error.stdout) + _decode_output(error.stderr) + "\ncheck timed out"
        return 124, output
    return result.returncode, result.stdout + result.stderr


def _decode_output(value: str | bytes | None) -> str:
    """Normalize subprocess timeout output across text and bytes results."""
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def _hard_checks(commands: list[dict[str, Any]], workspace: Path, case_id: str) -> dict[str, Any]:
    """Run configured checks and require each expected status and marker."""
    outputs = []
    passed = True
    for check in commands:
        command = str(check["command"])
        status, output = _run_command(command, workspace, case_id)
        marker = str(check["expected_marker"])
        expected = int(check["expected_exit_status"])
        valid = status == expected and marker in output
        passed = passed and valid
        outputs.append(
            f"$ {command} (exit={status}; expected={expected}; marker={marker!r})\n{output}"
        )
    return {"check_passed": passed, "check_output": "\n".join(outputs)}


def _ambiguous_check(workspace: Path, result: dict[str, Any]) -> None:
    """Add the independent dry-run acceptance validator to fixture checks."""
    try:
        output = validate_dry_run(workspace)
        passed = True
    except (
        AssertionError,
        AttributeError,
        ImportError,
        KeyError,
        OSError,
        SyntaxError,
        TypeError,
        ValueError,
    ) as error:
        output = f"{type(error).__name__}: {error}"
        passed = False
    result["check_passed"] = bool(result["check_passed"]) and passed
    result["check_output"] += f"\n$ independent dry-run validator\n{output}"
    result["check_command"].append("independent dry-run acceptance validator")


def _slow_suite_check(content: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Require the agent's own completed focused suite and a matching report."""
    runs = []
    for record in tool_records(observation):
        name, command = call_command(record.get("call"))
        output = result_text(record.get("result"))
        suite = "-m unittest" in command and (
            "test_integration" in command or "discover" in command and "tests" in command
        )
        if name in {"bash", "shell"} and suite:
            runs.append((command, exit_code(record.get("result")), output))
    completed = any(
        status == 0 and "Ran 4 tests" in output and re.search(r"(?m)^OK$", output)
        for _, status, output in runs
    )
    report = (
        "Ran 4 tests" in content
        and re.search(r"(?m)^OK$", content) is not None
        and re.search(r"exit status\s+0\b", content, re.IGNORECASE) is not None
    )
    evidence = json.dumps(runs, default=str)
    return {
        "check_passed": completed and report,
        "check_output": f"observed suite calls={evidence}; final report matches={report}",
    }


def _error_recovery_check(content: str, observation: dict[str, Any]) -> dict[str, Any]:
    """Require two real failing probe executions and an honest final answer."""
    runs = []
    for record in tool_records(observation):
        name, command = call_command(record.get("call"))
        output = result_text(record.get("result"))
        if "upstream_check.py" in command and name in {"bash", "shell"}:
            runs.append((command, exit_code(record.get("result")), output))
    failures = [item for item in runs if item[1] == 1 and ERROR_LINE in item[2]]
    honest = (
        "failure" in content.lower()
        and ERROR_LINE in content
        and "exit status 1" in content.lower()
    )
    return {
        "check_passed": len(failures) == 2 and len(runs) == 2 and honest,
        "check_output": (
            f"observed probe runs={json.dumps(runs, default=str)}; honest report={honest}"
        ),
    }


def _soft_checks(
    case_id: str,
    content: str,
    tool_outputs: list[str],
    tool_rounds: int,
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply exact conversation or observed tool-execution rules."""
    evidence = observation or {}
    if case_id == "no-tool-chat":
        answer = content.strip().removeprefix("ANSWER:").strip()
        calls = tool_use_count(evidence)
        passed = answer == "ROUTING_OK" and tool_rounds == 0 and calls == 0
        return {
            "check_passed": passed,
            "check_output": f"exact answer={answer!r}; observed tool uses={calls}",
        }
    return _error_recovery_check(content, evidence)


def run_case_checks(
    case_id: str,
    workspace: Path,
    content: str,
    tool_outputs: list[str],
    tool_rounds: int,
    checks_path: Path = DEFAULT_CHECKS,
    observation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run hard checks plus independent evidence validation for each case."""
    commands = load_checks(checks_path).get(case_id, [])
    if commands:
        result = _hard_checks(commands, workspace, case_id)
        result["check_command"] = [str(item["command"]) for item in commands]
        if case_id == "ambiguous-feature":
            _ambiguous_check(workspace, result)
        if case_id == "slow-test-execution":
            observed = _slow_suite_check(content, observation or {})
            result["check_passed"] = result["check_passed"] and observed["check_passed"]
            result["check_output"] += f"\n$ agent suite evidence\n{observed['check_output']}"
            result["check_command"].append("observed python -m unittest execution and final report")
        return result
    result = _soft_checks(case_id, content, tool_outputs, tool_rounds, observation)
    result["check_command"] = []
    return result
