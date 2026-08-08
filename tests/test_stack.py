"""Tests for scripts/stack.py and scripts/stack_compose.py.

All subprocess/container calls are mocked: no real provider, container, or
network interaction (repo convention).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import httpx
import pytest
import stack
import stack_compose

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "docker-compose.yml"


def _completed(stdout: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, stdout=stdout)


# --- compose parsing -------------------------------------------------------


def test_interpolate_plain_and_default():
    env = {"MANTIS_API_KEY": "k"}
    assert stack_compose._interpolate("${MANTIS_API_KEY}", env, COMPOSE) == "k"
    assert stack_compose._interpolate("${MISSING:-fallback}", env, COMPOSE) == "fallback"
    assert stack_compose._interpolate("${MISSING}", env, COMPOSE) == ""
    assert stack_compose._interpolate("a${A}b${B:-d}c", {"A": "1"}, COMPOSE) == "a1bdc"


def test_interpolate_nested_default():
    env = {"HOME": "/Users/x"}
    value = "${MANTIS_LEARNING_HOST_DIR:-${HOME}/.local/share/mantis/learning}"
    assert (
        stack_compose._interpolate(value, env, COMPOSE) == "/Users/x/.local/share/mantis/learning"
    )


def test_interpolate_required_missing_raises():
    with pytest.raises(stack_compose.ComposeError):
        stack_compose._interpolate("${MANTIS_API_KEY:?set MANTIS_API_KEY}", {}, COMPOSE)


def test_load_spec_real_compose(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setenv("HOME", "/Users/x")
    spec = stack_compose.load_spec(COMPOSE)
    assert set(spec["services"]) == {"openfugu", "redis"}
    of = spec["services"]["openfugu"]
    assert of["container_name"] == "mantis-orchestrator"
    assert of["ports"] == ["8088:8088"]
    assert of["environment"]["MANTIS_API_KEY"] == "k"
    assert of["environment"]["MANTIS_RUN_STORE"] == "memory"
    assert "openrouter/google/gemini-3.6-flash|high" in of["environment"]["MANTIS_WORKER_MODELS"]
    assert spec["volume_names"]["hf-cache"] == "mantis_hf-cache"
    assert spec["services"]["redis"]["command"] == ["redis-server", "--appendonly", "yes"]
    assert spec["networks"] == ["mantis"]


def test_load_spec_requires_api_key(monkeypatch):
    monkeypatch.delenv("MANTIS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        stack_compose.load_spec(COMPOSE)


# --- backend detection -----------------------------------------------------


def test_detect_backend_forced(monkeypatch):
    monkeypatch.setenv("MANTIS_STACK_BACKEND", "native")
    assert stack.detect_backend() == "native"
    monkeypatch.setenv("MANTIS_STACK_BACKEND", "docker")
    assert stack.detect_backend() == "docker"


def test_detect_backend_auto_falls_back(monkeypatch):
    monkeypatch.setenv("MANTIS_STACK_BACKEND", "auto")
    monkeypatch.setattr(stack.platform, "system", lambda: "Linux")
    assert stack.detect_backend() == "docker"
    monkeypatch.setattr(stack.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(stack.shutil, "which", lambda _cmd: None)
    assert stack.detect_backend() == "docker"


def test_detect_backend_auto_native(monkeypatch):
    monkeypatch.setenv("MANTIS_STACK_BACKEND", "auto")
    monkeypatch.setattr(stack.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(stack.shutil, "which", lambda _cmd: "/usr/local/bin/container")
    monkeypatch.setattr(
        stack.subprocess, "run", lambda *a, **k: _completed("status             running\n")
    )
    assert stack.detect_backend() == "native"


# --- docker path -----------------------------------------------------------


def test_docker_args():
    base = stack.docker_args("up", ["-d"])
    assert base == ["docker", "compose", "-f", str(COMPOSE), "up", "-d"]
    assert "conductor" not in " ".join(base)
    with_redis = stack.docker_args("up", ["redis", "-d"])
    assert with_redis == [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
        "up",
        "--profile",
        "redis",
        "-d",
    ]
    with_cond = stack.docker_args("build", ["conductor"])
    assert with_cond == [
        "docker",
        "compose",
        "-f",
        str(COMPOSE),
        "-f",
        str(stack.CONDUCTOR_FILE),
        "build",
    ]


def test_main_docker_up(monkeypatch, capsys):
    monkeypatch.setattr(stack, "detect_backend", lambda: "docker")
    captured = []
    monkeypatch.setattr(stack, "_run", lambda argv: captured.append(argv))
    monkeypatch.setattr("sys.argv", ["stack.py", "up"])
    stack.main()
    assert captured == [["docker", "compose", "-f", str(COMPOSE), "up", "-d"]]


# --- native command construction -------------------------------------------


def test_mount_args(spec):
    mounts = stack._mount_args(
        spec, ["hf-cache:/root/.cache/huggingface", "./artifacts:/app/artifacts:ro"]
    )
    assert "--mount" in mounts
    joined = " ".join(mounts)
    assert "type=volume,source=mantis_hf-cache,target=/root/.cache/huggingface" in joined
    assert f"type=bind,source={REPO / 'artifacts'},target=/app/artifacts,readonly" in joined


@pytest.fixture
def spec():
    return {
        "services": {},
        "networks": ["mantis"],
        "volume_names": {"hf-cache": "mantis_hf-cache"},
    }


def test_find_ip_nested():
    data = {"status": {"networks": [{"ipv4Address": "192.168.65.9/24"}]}}
    assert stack._find_ip(data) == "192.168.65.9"
    assert stack._find_ip({"configuration": {"networks": [{"ipAddress": ""}]}}) == ""
    assert stack._find_ip({"a": {"b": []}}) == ""


def test_native_up_runs_expected_argv(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    env_files: list[str] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if "--env-file" in argv:
            with open(argv[argv.index("--env-file") + 1]) as fh:
                env_files.append(fh.read())
        if argv[:2] == ["container", "list"]:
            return _completed("")
        if argv[:2] == ["container", "image"]:
            return _completed("mantis/openfugu:local")
        if argv[:2] in (["container", "network"], ["container", "volume"]):
            return _completed("")
        if argv[:2] == ["container", "inspect"]:
            return _completed('[{"status":{"networks":[{"ipv4Address":"192.168.65.9/24"}]}}]')
        return _completed("")

    class FakeTmp:
        def __init__(self, path):
            self.name = str(path)
            self._f = path.open("w")

        def __enter__(self):
            return self._f

        def __exit__(self, *exc):
            self._f.close()

    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    monkeypatch.setattr(stack, "wait_ready", lambda: None)
    monkeypatch.setattr(
        stack.tempfile, "NamedTemporaryFile", lambda *a, **k: FakeTmp(tmp_path / "env.env")
    )
    stack.native_up(redis=True, conductor=None)
    joined = "\n".join(" ".join(c) for c in calls)
    assert "container network create mantis" in joined
    assert "container volume create mantis_hf-cache" in joined
    assert (
        "container run -d --name mantis-orchestrator --network mantis --memory 4G -p 8088:8088"
        in joined
    )
    assert "--mount type=volume,source=mantis_hf-cache,target=/root/.cache/huggingface" in joined
    assert "--env-file" in joined
    assert "MANTIS_REDIS_URL=redis://192.168.65.9:6379/0" in env_files[0]
    env_path = next(c[c.index("--env-file") + 1] for c in calls if "--env-file" in c)
    assert not os.path.exists(env_path)


def test_native_up_already_running(monkeypatch, capsys):
    def fake_run(argv, **kwargs):
        if argv[:2] == ["container", "list"]:
            return _completed("mantis-orchestrator")
        return _completed("")

    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    stack.native_up(redis=False, conductor=None)
    assert "already running" in capsys.readouterr().out


def test_native_up_builds_when_missing(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["container", "list"]:
            return _completed("")
        if argv[:2] == ["container", "image"]:
            return _completed("")
        if argv[:2] in (["container", "network"], ["container", "volume"]):
            return _completed("")
        return _completed("")

    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    monkeypatch.setattr(stack, "wait_ready", lambda: None)

    class FakeTmp:
        def __init__(self, path):
            self.name = str(path)
            self._f = path.open("w")

        def __enter__(self):
            return self._f

        def __exit__(self, *exc):
            self._f.close()

    monkeypatch.setattr(
        stack.tempfile, "NamedTemporaryFile", lambda *a, **k: FakeTmp(tmp_path / "env.env")
    )
    stack.native_up(redis=False, conductor=None)
    assert any(c[0:2] == ["container", "build"] for c in calls)


def test_native_down_stops_only_existing(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["container", "list"]:
            return _completed("mantis-orchestrator")
        return _completed("")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    stack.native_down()
    stopped = [c for c in calls if c[0:2] == ["container", "stop"]]
    deleted = [c for c in calls if c[0:2] == ["container", "delete"]]
    assert [c[-1] for c in stopped] == ["mantis-orchestrator"]
    assert [c[-1] for c in deleted] == ["mantis-orchestrator"]


def test_wait_ready_success_then_fail(monkeypatch):
    results = iter([httpx.ConnectError("down"), httpx.Response(200)])

    class FakeGet:
        def __call__(self, *a, **k):
            r = next(results)
            if isinstance(r, Exception):
                raise r
            return r

    monkeypatch.setattr(stack.httpx, "get", FakeGet())
    monkeypatch.setattr(stack.time, "sleep", lambda _s: None)
    stack.wait_ready(timeout=10)

    monkeypatch.setattr(
        stack.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )
    with pytest.raises(SystemExit):
        stack.wait_ready(timeout=0)
