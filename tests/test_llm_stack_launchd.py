"""LaunchAgent structure for the local LLM stack."""

from __future__ import annotations

import plistlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest

GENERATOR = Path(__file__).resolve().parents[1] / "launch/host/launchd_jobs.py"


def _write_jobs(repo: Path, home: Path, *, experimental: bool = False) -> dict[str, dict]:
    command = [
        sys.executable, str(GENERATOR), "write", "--repo", str(repo), "--home", str(home),
    ]
    if experimental:
        command.append("--experimental")
    subprocess.run(command, capture_output=True, text=True, check=True)
    agents = home / ".local/share/llm-stack/agents"
    return {
        name: plistlib.loads((agents / f"com.mallochio.mantis.{name}.plist").read_bytes())
        for name in ("switchyard", "api")
    }


@pytest.mark.parametrize("experimental", [False, True])
def test_generated_jobs_are_foreground_launch_agents(tmp_path: Path, experimental: bool) -> None:
    repo = tmp_path / "source"
    home = tmp_path / "home"
    jobs = _write_jobs(repo, home, experimental=experimental)

    for name, script, log in (
        ("switchyard", "switchyard-local.sh", home / ".local/share/mantis/switchyard/server.err"),
        ("api", "mantis-local.sh", home / ".local/share/mantis/server.log"),
    ):
        job = jobs[name]
        assert job["Label"] == f"com.mallochio.mantis.{name}"
        assert job["ProgramArguments"] == [
            "/bin/bash", str(repo / "launch/host/lib" / script), "--foreground",
        ]
        assert job["RunAtLoad"] is True
        assert job["KeepAlive"] is True
        assert job["ThrottleInterval"] == 10
        assert job["Umask"] == 0o077
        assert job["StandardOutPath"] == str(log)
        assert job["StandardErrorPath"] == str(log)
        assert log.parent.is_dir()

    assert jobs["switchyard"]["EnvironmentVariables"] == {}
    assert jobs["api"]["EnvironmentVariables"] == {
        "MANTIS_LAUNCHD_MODE": "1" if experimental else "0",
    }


def test_normal_write_resets_api_mode_after_experimental_write(tmp_path: Path) -> None:
    repo = tmp_path / "source"
    home = tmp_path / "home"
    experimental = _write_jobs(repo, home, experimental=True)
    normal = _write_jobs(repo, home)

    assert normal["switchyard"] == experimental["switchyard"]
    assert normal["api"]["EnvironmentVariables"] == {"MANTIS_LAUNCHD_MODE": "0"}


def test_generated_jobs_have_private_permissions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    agents = home / ".local/share/llm-stack/agents"
    logs = home / ".local/share/mantis"
    directories = (agents, logs, logs / "switchyard")
    for directory in directories:
        directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        directory.chmod(0o755)
    _write_jobs(tmp_path / "source", home)

    for directory in directories:
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    for name in ("switchyard", "api"):
        job = agents / f"com.mallochio.mantis.{name}.plist"
        assert stat.S_IMODE(job.stat().st_mode) == 0o600
    assert not (home / "Library/LaunchAgents").exists()
