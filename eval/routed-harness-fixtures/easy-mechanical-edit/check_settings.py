"""Verify the renamed settings file without changing any other value."""

import sys
import tomllib

EXPECTED = {
    "server": {"label": "routed_dispatch", "host": "127.0.0.1", "port": 8080},
    "worker": {"label": "legacy_worker", "max_retries": 3, "timeout_s": 30},
}


def main() -> int:
    """Compare the parsed settings against the expected post-rename state."""
    with open("settings.toml", "rb") as handle:
        data = tomllib.load(handle)
    if data == EXPECTED:
        print("CONFIG_OK")
        return 0
    for section in sorted(set(data) | set(EXPECTED)):
        if data.get(section) != EXPECTED.get(section):
            print(f"MISMATCH {section}: {data.get(section)} != {EXPECTED.get(section)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
