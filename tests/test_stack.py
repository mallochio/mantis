"""Tests for scripts/stack.py and scripts/stack_compose.py.

All subprocess/container calls are mocked: no real provider, container, or
network interaction (repo convention).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import httpx
import model_catalog
import model_catalog_runtime
import pytest
import stack
import stack_compose

REPO = Path(__file__).resolve().parent.parent
COMPOSE = REPO / "docker-compose.yml"
ENTRYPOINT = REPO / "docker/openfugu/entrypoint.sh"
ABI = model_catalog.load_abi_manifest()
SLOTS = tuple(ABI.slot_order)
CONDUCTOR = ABI.conductor

AMBIENT_OVERRIDE_VARS = (
    "AI_ROUTING_CONFIG",
    "MANTIS_CATALOG_PATH",
    "MANTIS_ENDPOINT_PROFILE",
    "MANTIS_PROVIDER_BINDINGS",
    "MANTIS_WORKER_BINDINGS",
    "MANTIS_IDENTITY_CONTRACT",
    "MANTIS_PROVIDER_KEYS",
    "MANTIS_WORKER_MODELS",
    "MANTIS_CONDUCTOR_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENCODE_GO_ENDPOINT_URL",
    "AI_GATEWAY_API_KEY",
    "MANTIS_GATEWAY_URL",
    "MANTIS_GATEWAY_OPENCODE_URL",
    "CLOUDFLARE_GATEWAY_BASE_URL",
)


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Scrub ambient endpoint/catalog overrides so stack tests are hermetic.

    Interactive shells export catalog/profile variables (and direct env
    writes from earlier tests leak); stack functions read ``os.environ``
    directly, so the suite must not inherit the host environment.
    """
    for name in AMBIENT_OVERRIDE_VARS:
        monkeypatch.delenv(name, raising=False)


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
    monkeypatch.setattr(model_catalog, "load_mantis_catalog", lambda: None)
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
    monkeypatch.setattr(model_catalog, "load_mantis_catalog", lambda: None)
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
    monkeypatch.setattr(model_catalog, "load_mantis_catalog", lambda: None)
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


def _runtime_env(monkeypatch, *, profile: str = "catalog", workers=None, contract: str = "auto"):
    """Install a valid seven-slot catalog runtime environment for stack tests."""
    providers = {
        "edge": {
            "adapter": "openrouter",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }
    workers = workers or {
        slot: {
            "provider": "edge",
            "upstream_model": f"vendor/{slot}",
            "model_identity": ABI.workers[slot][0],
            "protocols": ["responses"] if slot == CONDUCTOR else ["chat_completions"],
        }
        for slot in SLOTS
    }
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", profile)
    monkeypatch.setenv("MANTIS_PROVIDER_BINDINGS", json.dumps(providers))
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", json.dumps(workers))
    monkeypatch.setenv("MANTIS_WORKER_MODELS", ",".join(SLOTS))
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", CONDUCTOR)
    if contract == "auto":
        bindings = model_catalog._runtime_bindings(providers, workers)
        contract = model_catalog_runtime._runtime_contract_hash(
            bindings, tuple(SLOTS), CONDUCTOR
        )
    if contract:
        monkeypatch.setenv("MANTIS_IDENTITY_CONTRACT", contract)
    return providers, workers


def _run_entrypoint(monkeypatch, tmp_path, env, args=("sh", "-c", "exit 0"), python_rc=0):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(f"#!/bin/sh\nexit {python_rc}\n")
    fake_python.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    # The ambient environment may carry catalog-mode variables (e.g. an
    # interactive shell exporting AI_ROUTING_CONFIG); scrub them so each
    # entrypoint case is hermetic and only the provided env applies.
    for key in (
        "AI_ROUTING_CONFIG",
        "MANTIS_CATALOG_PATH",
        "MANTIS_ENDPOINT_PROFILE",
        "MANTIS_PROVIDER_BINDINGS",
        "MANTIS_WORKER_BINDINGS",
        "MANTIS_IDENTITY_CONTRACT",
        "MANTIS_PROVIDER_KEYS",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return subprocess.run(["sh", str(ENTRYPOINT), *args], capture_output=True, text=True)



def _catalog_text(contract: str) -> str:
    """Minimal valid catalog aligned with the committed ABI manifest."""
    abi = model_catalog.load_abi_manifest()
    workers = []
    for index, slot in enumerate(abi.slot_order):
        provider = "router" if index != 3 else "code"
        protocols = '["responses"]' if slot == abi.conductor else '["chat_completions"]'
        identity, effort = abi.workers[slot]
        lines = [
            f"[mantis.workers.{slot}]",
            f'provider = "{provider}"',
            f'upstream_model = "vendor/{slot}"',
            f'model_identity = "{identity}"',
            f"protocols = {protocols}",
        ]
        if effort:
            lines.append(f'reasoning_effort = "{effort}"')
        workers.append("\n".join(lines))
    return (
        "version = 1\n\n"
        '[providers.router]\n'
        'adapter = "openrouter"\n'
        'base_url = "https://router.example.test/v1"\n'
        'credential_env = "ROUTER_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n\n'
        '[providers.code]\n'
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["chat_completions"]\n\n'
        "[mantis]\n"
        f"slot_order = {json.dumps(list(abi.slot_order))}\n"
        f'conductor = "{abi.conductor}"\n'
        f'trained_slot_contract = "{contract}"\n'
        + "\n".join(workers)
        + "\n"
    )


# --- catalog fail-closed compose path ----------------------------------------


def test_load_spec_catalog_mount_and_env_passthrough(monkeypatch):
    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setenv("HOME", "/Users/x")
    monkeypatch.delenv("AI_ROUTING_CONFIG", raising=False)
    monkeypatch.delenv("MANTIS_CATALOG_PATH", raising=False)
    spec = stack_compose.load_spec(COMPOSE)
    of = spec["services"]["openfugu"]
    assert of["environment"]["MANTIS_CATALOG_PATH"] == ""
    assert of["environment"]["AI_ROUTING_CONFIG"] == ""
    monkeypatch.setenv("MANTIS_CATALOG_PATH", "/Users/x/custom/catalog.toml")
    spec = stack_compose.load_spec(COMPOSE)
    of = spec["services"]["openfugu"]
    assert of["environment"]["MANTIS_CATALOG_PATH"] == "/Users/x/custom/catalog.toml"
    assert "/Users/x/custom/catalog.toml:/app/catalog/catalog.toml:ro" in of["volumes"]


def test_native_up_skips_catalog_mount_outside_catalog_mode(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["container", "list"]:
            return _completed("")
        if argv[:2] == ["container", "image"]:
            return _completed("mantis/openfugu:local")
        if argv[:2] in (["container", "network"], ["container", "volume"]):
            return _completed("")
        return _completed("")

    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.delenv("MANTIS_ENDPOINT_PROFILE", raising=False)
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
    joined = "\n".join(" ".join(c) for c in calls)
    assert "/app/catalog/catalog.toml" not in joined


def test_native_up_mounts_catalog_in_catalog_mode(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["container", "list"]:
            return _completed("")
        if argv[:2] == ["container", "image"]:
            return _completed("mantis/openfugu:local")
        if argv[:2] in (["container", "network"], ["container", "volume"]):
            return _completed("")
        return _completed("")

    monkeypatch.setenv("MANTIS_API_KEY", "k")
    monkeypatch.setenv("MANTIS_ENDPOINT_PROFILE", "catalog")
    catalog = tmp_path / "catalog.toml"
    catalog.write_text("version = 1\n")
    monkeypatch.setenv("MANTIS_CATALOG_PATH", str(catalog))
    monkeypatch.setattr(stack.subprocess, "run", fake_run)
    monkeypatch.setattr(stack, "wait_ready", lambda: None)

    class FakeTmp:
        written = ""

        def __init__(self, path):
            self.name = str(path)
            self._f = path.open("w")

        def __enter__(self):
            return self._f

        def __exit__(self, *exc):
            self._f.close()
            FakeTmp.written = Path(self.name).read_text()

    monkeypatch.setattr(
        stack.tempfile, "NamedTemporaryFile", lambda *a, **k: FakeTmp(tmp_path / "env.env")
    )
    stack.native_up(redis=False, conductor=None)
    joined = "\n".join(" ".join(c) for c in calls)
    # The native runtime only binds directories: the catalog file bind is
    # translated to its parent directory, and MANTIS_CATALOG_PATH points at
    # the fixed in-container file path.
    assert f"source={tmp_path},target=/app/catalog" in joined
    assert "MANTIS_CATALOG_PATH=/app/catalog/catalog.toml" in FakeTmp.written


# --- entrypoint fail-closed gate ---------------------------------------------


def test_entrypoint_fails_closed_in_catalog_mode_without_catalog(monkeypatch, tmp_path):
    result = _run_entrypoint(monkeypatch, tmp_path, {"MANTIS_ENDPOINT_PROFILE": "catalog"})
    assert result.returncode != 0
    assert "requires a mounted catalog" in result.stderr


def test_entrypoint_catalog_mode_via_catalog_env(monkeypatch, tmp_path):
    result = _run_entrypoint(monkeypatch, tmp_path, {"MANTIS_CATALOG_PATH": "/nonexistent"})
    assert result.returncode != 0
    assert "requires a mounted catalog" in result.stderr


def test_entrypoint_fails_closed_when_bindings_missing(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.toml"
    catalog.write_text("version = 1\n")
    result = _run_entrypoint(
        monkeypatch,
        tmp_path,
        {"MANTIS_ENDPOINT_PROFILE": "catalog", "MANTIS_CATALOG_PATH": str(catalog)},
    )
    assert result.returncode != 0
    assert "MANTIS_PROVIDER_BINDINGS" in result.stderr


def test_entrypoint_fails_when_catalog_validation_fails(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.toml"
    catalog.write_text("version = 1\n")
    env = {
        "MANTIS_ENDPOINT_PROFILE": "catalog",
        "MANTIS_CATALOG_PATH": str(catalog),
        "MANTIS_PROVIDER_BINDINGS": "{}",
        "MANTIS_WORKER_BINDINGS": "{}",
        "MANTIS_IDENTITY_CONTRACT": "0" * 64,
        "MANTIS_PROVIDER_KEYS": "{}",
    }
    result = _run_entrypoint(monkeypatch, tmp_path, env, python_rc=1)
    assert result.returncode != 0
    assert "catalog validation failed" in result.stderr


def test_entrypoint_valid_catalog_mode_execs_command(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.toml"
    catalog.write_text("version = 1\n")
    env = {
        "MANTIS_ENDPOINT_PROFILE": "catalog",
        "MANTIS_CATALOG_PATH": str(catalog),
        "MANTIS_PROVIDER_BINDINGS": "{}",
        "MANTIS_WORKER_BINDINGS": "{}",
        "MANTIS_IDENTITY_CONTRACT": "0" * 64,
        "MANTIS_PROVIDER_KEYS": "{}",
    }
    result = _run_entrypoint(monkeypatch, tmp_path, env, args=("sh", "-c", "echo served"))
    assert result.returncode == 0
    assert result.stdout.strip() == "served"


def test_entrypoint_passes_through_in_direct_mode(monkeypatch, tmp_path):
    result = _run_entrypoint(monkeypatch, tmp_path, {}, args=("sh", "-c", "echo served"))
    assert result.returncode == 0
    assert result.stdout.strip() == "served"


# --- full binding fingerprint readiness ---------------------------------------


def test_catalog_readiness_metadata_includes_binding_fingerprint(monkeypatch):
    _runtime_env(monkeypatch)
    expected = stack._expected_readiness_metadata()
    assert expected["endpoint_profile"] == "catalog"
    assert expected["endpoint_hosts"] == {"edge": "edge.example.test"}
    assert len(expected["binding_fingerprint"]) == 64
    assert len(expected["catalog_identity_contract"]) == 64


def test_catalog_readiness_detects_stale_bindings(monkeypatch):
    providers, workers = _runtime_env(monkeypatch)
    expected = stack._expected_readiness_metadata()
    stale = dict(expected)
    stale["binding_fingerprint"] = "0" * 64
    response = httpx.Response(200, json=stale)
    monkeypatch.setattr(stack.httpx, "get", lambda *a, **k: response)
    with pytest.raises(SystemExit, match="use the restart command"):
        stack.validate_running_container()


def test_catalog_readiness_accepts_matching_bindings(monkeypatch):
    _runtime_env(monkeypatch)
    expected = stack._expected_readiness_metadata()
    response = httpx.Response(200, json=expected)
    monkeypatch.setattr(stack.httpx, "get", lambda *a, **k: response)
    stack.validate_running_container()  # must not raise


def test_apply_catalog_pins_container_catalog_path(monkeypatch, tmp_path):
    from model_catalog import abi_contract, load_abi_manifest

    catalog = tmp_path / "catalog.toml"
    catalog.write_text("version = 1\n")
    abi = load_abi_manifest()
    catalog.write_text(_catalog_text(abi_contract(abi)))
    monkeypatch.setenv("ROUTER_KEY", "router-key")
    monkeypatch.setenv("CODE_KEY", "code-key")
    loaded = model_catalog.load_mantis_catalog(catalog)
    assert loaded is not None
    saved = dict(os.environ)
    try:
        stack.apply_catalog(loaded)
        assert os.environ["MANTIS_CATALOG_PATH"] == str(catalog)
        assert os.environ["MANTIS_ENDPOINT_PROFILE"] == "catalog"
        assert "MANTIS_PROVIDER_BINDINGS" in os.environ
    finally:
        os.environ.clear()
        os.environ.update(saved)
