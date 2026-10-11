"""Run independent acceptance checks for the dry-run fixture."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType


class _Sentinel:
    """Raise if a dry-run operation converts its input to text."""

    def __str__(self) -> str:
        """Reject operation execution during dry-run validation."""
        raise AssertionError("operation executed during dry_run")


def _fixture_modules(workspace: Path) -> tuple[ModuleType, ModuleType]:
    """Import fixture modules from a workspace without retaining them globally."""
    names = [
        name for name in sys.modules if name == "mantis_router" or name.startswith("mantis_router.")
    ]
    saved = {name: sys.modules[name] for name in names}
    for name in names:
        sys.modules.pop(name)
    paths = sys.path.copy()
    sys.path.insert(0, str(workspace.resolve()))
    try:
        gateway = importlib.import_module("mantis_router.gateway")
        orchestration = importlib.import_module("mantis_router.orchestration")
    finally:
        sys.path[:] = paths
        for name in tuple(sys.modules):
            if name == "mantis_router" or name.startswith("mantis_router."):
                sys.modules.pop(name)
        sys.modules.update(saved)
    return gateway, orchestration


def validate_dry_run(workspace: Path) -> str:
    """Check dry-run behavior, validation ordering, test additions, and documentation."""
    workspace = Path(workspace).resolve()
    gateway, orchestration = _fixture_modules(workspace)
    for op in ("upper", "count"):
        step = {"op": op, "value": _Sentinel(), "dry_run": True}
        assert orchestration.run_step(step) is None
        assert gateway.dispatch(step) == {"status": "dry_run", "result": None}
    for flag in (False, None):
        step = {"op": "upper", "value": "hello"}
        if flag is not None:
            step["dry_run"] = flag
        assert gateway.dispatch(step) == {"status": "ok", "result": "HELLO"}
    try:
        orchestration.run_step({"op": "unknown", "dry_run": True})
    except ValueError:
        pass
    else:
        raise AssertionError("dry_run bypassed operation validation")
    tests = ast.parse((workspace / "tests/test_router.py").read_text())
    functions = [node for node in ast.walk(tests) if isinstance(node, ast.FunctionDef)]
    assert sum(node.name.startswith("test_") for node in functions) > 5
    assert "dry_run" in (workspace / "README.md").read_text()
    return "DRY_RUN_OK"
