"""Run an immutable fixture checker against an edited evaluation workspace."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "eval" / "routed-harness-fixtures"


def main() -> int:
    """Execute the trusted checker source with the current directory as target."""
    source = Path(sys.argv[1]).resolve()
    if not source.is_relative_to(FIXTURES.resolve()) or not source.is_file():
        raise SystemExit("checker must be an existing fixture script")
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_path(str(source), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
