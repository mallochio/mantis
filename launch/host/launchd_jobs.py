"""Write user LaunchAgent jobs for the local Mantis stack."""

from __future__ import annotations

import argparse
import plistlib
from pathlib import Path

LABEL_PREFIX = "com.mallochio.mantis"


def _job(label: str, script: Path, logfile: Path, env: dict[str, str]) -> dict:
    """Describe a foreground service managed directly by launchd."""
    return {
        "Label": label,
        "ProgramArguments": ["/bin/bash", str(script), "--foreground"],
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "Umask": 0o077,
        "StandardOutPath": str(logfile),
        "StandardErrorPath": str(logfile),
        "EnvironmentVariables": env,
    }


def write_jobs(repo: Path, home: Path, experimental: bool = False) -> list[Path]:
    """Install private user jobs without starting or stopping processes."""
    agents = home / ".local" / "share" / "llm-stack" / "agents"
    logs = home / ".local" / "share" / "mantis"
    for directory in (agents, logs, logs / "switchyard"):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    definitions = (
        (
            "switchyard", repo / "launch/host/lib/switchyard-local.sh",
            logs / "switchyard/server.err", {},
        ),
        (
            "api", repo / "launch/host/lib/mantis-local.sh", logs / "server.log",
            {"MANTIS_LAUNCHD_MODE": "1" if experimental else "0"},
        ),
    )
    paths = []
    for name, script, logfile, env in definitions:
        label = f"{LABEL_PREFIX}.{name}"
        path = agents / f"{label}.plist"
        path.write_bytes(plistlib.dumps(_job(label, script, logfile, env)))
        path.chmod(0o600)
        paths.append(path)
    return paths


def main() -> None:
    """Write LaunchAgents for a selected repo and user home."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["write"])
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()
    for path in write_jobs(args.repo.resolve(), args.home.resolve(), args.experimental):
        print(path)


if __name__ == "__main__":
    main()
