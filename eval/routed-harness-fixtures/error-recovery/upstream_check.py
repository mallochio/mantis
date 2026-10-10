"""Deterministic upstream connectivity probe that always fails."""

import sys
import time


def main() -> int:
    """Probe the upstream and fail deterministically."""
    time.sleep(1.0)
    print("UPSTREAM_TIMEOUT: upstream service did not respond within 500ms")
    return 1


if __name__ == "__main__":
    sys.exit(main())
