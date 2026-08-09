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
    monkeypatch.setattr(stack, "wait_ready", lambda: None)
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
        "container run -d --name mantis-orchestrator --network mantis --memory 8G -p 8088:8088"
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
    monkeypatch.setattr(stack, "validate_running_container", lambda: print("already running"))
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


def test_native_down_deletes_stopped_orchestrator_and_running_redis(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv == ["container", "list"]:
            return _completed("mantis-redis")
        if argv == ["container", "list", "--all"]:
            return _completed("mantis-orchestrator\nmantis-redis")
        return _completed("")

    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    stack.native_down()
    stopped = [c[-1] for c in calls if c[0:2] == ["container", "stop"]]
    deleted = [c[-1] for c in calls if c[0:2] == ["container", "delete"]]
    assert stopped == ["mantis-redis"]
    assert deleted == ["mantis-orchestrator", "mantis-redis"]


def test_wait_ready_rejects_stale_service_metadata(monkeypatch):
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "direct")
    monkeypatch.setenv("OPENROUTER_BASE_URL", stack.DIRECT_OPENROUTER_URL)
    monkeypatch.setenv("OPENCODE_GO_ENDPOINT_URL", stack.DIRECT_OPENCODE_URL)
    expected = stack._expected_readiness_metadata()
    stale = dict(expected)
    stale["endpoint_fingerprints"] = dict(expected["endpoint_fingerprints"])
    stale["endpoint_fingerprints"]["openrouter"] = "differentpath"
    results = iter([httpx.Response(200, json=stale), httpx.Response(200, json=expected)])
    calls = []

    def fake_get(*args, **kwargs):
        calls.append((args, kwargs))
        return next(results)

    monkeypatch.setattr(stack.httpx, "get", fake_get)
    monkeypatch.setattr(stack.time, "sleep", lambda _seconds: None)
    stack.wait_ready(timeout=10)
    assert len(calls) == 2

    monkeypatch.setattr(
        stack.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )
    with pytest.raises(SystemExit, match="readiness metadata did not match"):
        stack.wait_ready(timeout=0)


def test_validate_running_container_requires_selected_profile(monkeypatch):
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "direct")
    response = httpx.Response(
        200,
        json={
            "status": "ready",
            "endpoint_profile": "cloudflare",
            "endpoint_hosts": {},
            "endpoint_fingerprints": {},
        },
    )
    monkeypatch.setattr(stack.httpx, "get", lambda *a, **k: response)
    with pytest.raises(SystemExit, match="use the restart command"):
        stack.validate_running_container()


# --- endpoint profiles ----------------------------------------------------


def test_resolve_direct_profile_defaults_and_overrides():
    direct = stack.resolve_endpoint_profile(
        "direct", {"OPENROUTER_API_KEY": "router", "OPENCODE_API_KEY": "code"}
    )
    assert direct.openrouter_url == stack.DIRECT_OPENROUTER_URL
    assert direct.opencode_url == stack.DIRECT_OPENCODE_URL
    assert direct.openrouter_key == "router"
    assert direct.opencode_key == "code"
    custom = stack.resolve_endpoint_profile(
        "direct",
        {
            "OPENROUTER_BASE_URL": "https://router.example.test/a",
            "OPENCODE_GO_ENDPOINT_URL": "https://code.example.test/b",
        },
    )
    assert custom.hosts == ("router.example.test", "code.example.test")


def test_resolve_cloudflare_profile_maps_distinct_token():
    profile = stack.resolve_endpoint_profile(
        "cloudflare",
        {
            "MANTIS_GATEWAY_API_KEY": "gateway-secret",
            "OPENROUTER_API_KEY": "provider-secret",
            "OPENROUTER_BASE_URL": stack.DIRECT_OPENROUTER_URL,
            "OPENCODE_GO_ENDPOINT_URL": stack.DIRECT_OPENCODE_URL,
        },
    )
    assert profile.openrouter_url == stack.CLOUDFLARE_GATEWAY_URL
    assert profile.opencode_url == stack.CLOUDFLARE_GATEWAY_URL
    assert profile.openrouter_key == profile.opencode_key == "gateway-secret"


def test_cloudflare_profile_uses_only_gateway_url_overrides():
    shared = stack.resolve_endpoint_profile(
        "cloudflare",
        {
            "AI_GATEWAY_API_KEY": "token",
            "MANTIS_GATEWAY_URL": "https://shared-gateway.test/v1",
            "MANTIS_GATEWAY_OPENCODE_URL": "https://code-gateway.test/v1",
        },
    )
    assert shared.openrouter_url == "https://shared-gateway.test/v1"
    assert shared.opencode_url == "https://code-gateway.test/v1"


def test_endpoint_urls_are_normalized():
    profile = stack.resolve_endpoint_profile(
        "direct",
        {
            "OPENROUTER_BASE_URL": "HTTPS://Router.Example.test:443/v1///",
            "OPENCODE_GO_ENDPOINT_URL": "https://Code.Example.test/v1/",
        },
    )
    assert profile.openrouter_url == "https://router.example.test/v1"
    assert profile.opencode_url == "https://code.example.test/v1"


@pytest.mark.parametrize(
    "url",
    [
        "https://user:pass@example.test/v1",
        "https://example.test/v1?token=value",
        "https://example.test/v1#fragment",
        "https://example.test:invalid/v1",
        "ftp://example.test/v1",
    ],
)
def test_endpoint_urls_reject_unsafe_or_invalid_values(url):
    with pytest.raises(SystemExit):
        stack.resolve_endpoint_profile("direct", {"OPENROUTER_BASE_URL": url})


def test_cloudflare_requires_token_without_disclosing_other_secrets(capsys):
    with pytest.raises(SystemExit, match="requires MANTIS_GATEWAY_API_KEY"):
        stack.resolve_endpoint_profile("cloudflare", {"OPENROUTER_API_KEY": "must-not-appear"})
    assert "must-not-appear" not in capsys.readouterr().out


def test_apply_profile_diagnostic_is_sanitized(monkeypatch, capsys):
    marker = "gateway-super-secret"
    profile = stack.resolve_endpoint_profile("cloudflare", {"AI_GATEWAY_API_KEY": marker})
    stack.apply_endpoint_profile(profile)
    output = capsys.readouterr().out
    assert marker not in output
    assert "unified-ai-gateway.siddsantham.workers.dev" in output
    assert profile.openrouter_url not in output
    assert os.environ["OPENROUTER_API_KEY"] == marker
    assert os.environ["OPENCODE_API_KEY"] == marker


def test_main_docker_restart_forces_recreate(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setenv("MANTIS_GATEWAY_API_KEY", "docker-gateway-token")
    monkeypatch.setattr(stack, "detect_backend", lambda: "docker")
    captured = []
    monkeypatch.setattr(stack, "_run", lambda argv: captured.append(argv))
    monkeypatch.setattr(stack, "wait_ready", lambda: None)
    monkeypatch.setattr("sys.argv", ["stack.py", "restart", "--endpoint-profile", "cloudflare"])
    stack.main()
    assert captured == [["docker", "compose", "-f", str(COMPOSE), "up", "-d", "--force-recreate"]]
    assert os.environ["OPENROUTER_API_KEY"] == "docker-gateway-token"


def test_main_native_restart_replaces_container(monkeypatch):
    events = []
    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "native-gateway-token")
    monkeypatch.setattr(stack, "detect_backend", lambda: "native")
    monkeypatch.setattr(stack, "_container_output", lambda argv: "mantis-redis")
    monkeypatch.setattr(stack, "native_down", lambda: events.append("down"))
    monkeypatch.setattr(
        stack,
        "native_up",
        lambda redis, conductor, memory: events.append(f"up:redis={redis}"),
    )
    monkeypatch.setattr("sys.argv", ["stack.py", "restart", "--endpoint-profile", "cloudflare"])
    stack.main()
    assert events == ["down", "up:redis=True"]
    assert os.environ["OPENCODE_API_KEY"] == "native-gateway-token"


def test_load_dotenv_defaults_preserves_exported_values(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "MANTIS_GATEWAY_API_KEY=from-file # inline comment\n"
        'OPENROUTER_BASE_URL="https://file.test/v1" # another comment\n'
        'QUOTED_VALUE="value with # literal and escaped \\"quote\\""\n'
    )
    monkeypatch.setenv("MANTIS_GATEWAY_API_KEY", "exported")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("QUOTED_VALUE", raising=False)
    stack.load_dotenv_defaults(path)
    assert os.environ["MANTIS_GATEWAY_API_KEY"] == "exported"
    assert os.environ["OPENROUTER_BASE_URL"] == "https://file.test/v1"
    assert os.environ["QUOTED_VALUE"] == 'value with # literal and escaped "quote"'
