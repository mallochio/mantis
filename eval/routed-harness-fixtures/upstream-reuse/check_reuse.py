"""Verify the caller reuses the shared dispatch helper."""

import sys
from pathlib import Path

import caller
import dispatch_helper


def main() -> int:
    """Fail when the caller still duplicates routing logic."""
    text = Path("caller.py").read_text()
    checks = {
        "calls routed_dispatch": "routed_dispatch" in text,
        "no local route table": "worker-a" not in text and "worker-b" not in text,
        "behavior matches helper": all(
            caller.handle(case) == dispatch_helper.routed_dispatch(case)
            for case in [
                {"route": "alpha"},
                {"route": "beta", "priority": 2},
                {"route": "delta"},
                {"route": "gamma", "priority": -3},
                {},
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("REUSE_FAILED: " + ", ".join(failed))
        return 1
    print("REUSE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
