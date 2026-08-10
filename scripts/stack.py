#!/usr/bin/env python3
"""Stack driver for the mantis container stack.

One spec (docker-compose.yml) drives two runtimes:
- native: Apple `container` CLI on macOS 26+ (no Docker daemon)
- docker: `docker compose` everywhere else

Override the backend with MANTIS_STACK_BACKEND=auto|native|docker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx
import model_catalog
import stack_compose
from typing_extensions import NotRequired, TypedDict

load_spec = stack_compose.load_spec

REPO = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO / "docker-compose.yml"
CONDUCTOR_FILE = REPO / "eval" / "docker-compose.conductor.yml"
IMAGE_TAG = "mantis/openfugu:local"
ORCHESTRATOR = "mantis-orchestrator"
CATALOG_MOUNT_TARGET = "/app/catalog/catalog.toml"
REDIS_NAME = "mantis-redis"
READY_URL = "http://127.0.0.1:8088/ready"
START_TIMEOUT = 90
DIRECT_OPENROUTER_URL = "https://openrouter.ai/api/v1"
DIRECT_OPENCODE_URL = "https://opencode.ai/zen/go/v1"


class ReadinessMetadata(TypedDict):
    status: str
    endpoint_profile: str
    endpoint_hosts: dict[str, str]
    endpoint_fingerprints: dict[str, str]
    catalog_identity_contract: NotRequired[str]
    binding_fingerprint: NotRequired[str]


@dataclass(frozen=True)
class EndpointProfile:
    """Resolved, container-facing provider configuration."""

    name: str
    openrouter_url: str
    opencode_url: str
    openrouter_key: str
    opencode_key: str

    @property
    def hosts(self) -> tuple[str, str]:
        return (_endpoint_host(self.openrouter_url), _endpoint_host(self.opencode_url))

    @property
    def fingerprints(self) -> tuple[str, str]:
        return (
            _endpoint_fingerprint(self.openrouter_url),
            _endpoint_fingerprint(self.opencode_url),
        )

    def environment(self) -> dict[str, str]:
        return {
            "MANTIS_ENDPOINT_PROFILE": self.name,
            "OPENROUTER_BASE_URL": self.openrouter_url,
            "OPENCODE_GO_ENDPOINT_URL": self.opencode_url,
            "OPENROUTER_API_KEY": self.openrouter_key,
            "OPENCODE_API_KEY": self.opencode_key,
        }


def _normalize_endpoint_url(url: str) -> str:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise SystemExit("provider endpoint override has an invalid port") from error
    if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
        raise SystemExit("provider endpoint overrides must be absolute HTTP(S) URLs")
    if parsed.username is not None or parsed.password is not None:
        raise SystemExit("provider endpoint overrides must not contain user information")
    if parsed.query or parsed.fragment:
        raise SystemExit("provider endpoint overrides must not contain a query or fragment")
    hostname = parsed.hostname.lower()
    host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (parsed.scheme.lower() == "http" and port == 80) or (
        parsed.scheme.lower() == "https" and port == 443
    )
    netloc = f"{host}:{port}" if port is not None and not default_port else host
    path = parsed.path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), netloc, path, "", ""))


def _endpoint_host(url: str) -> str:
    return urlsplit(_normalize_endpoint_url(url)).hostname or ""


def _endpoint_fingerprint(url: str) -> str:
    normalized = _normalize_endpoint_url(url)
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def load_dotenv_defaults(path: Path = REPO / ".env") -> None:
    """Load shell-compatible KEY=VALUE defaults; exported values keep priority."""
    if not path.exists():
        return
    for line_number, raw in enumerate(path.read_text().splitlines(), 1):
        try:
            tokens = shlex.split(raw, comments=True, posix=True)
        except ValueError as error:
            raise SystemExit(f"{path}:{line_number}: invalid quoted value") from error
        if not tokens:
            continue
        if tokens[0] == "export":
            tokens = tokens[1:]
        if len(tokens) != 1 or "=" not in tokens[0]:
            raise SystemExit(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = tokens[0].split("=", 1)
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise SystemExit(f"{path}:{line_number}: invalid environment variable name")
        os.environ.setdefault(key, value)


def resolve_endpoint_profile(name: str, env: dict[str, str] | None = None) -> EndpointProfile:
    """Resolve a profile without including credentials in diagnostics or errors."""
    source = os.environ if env is None else env
    if name == "direct":
        profile = EndpointProfile(
            name=name,
            openrouter_url=source.get("OPENROUTER_BASE_URL", DIRECT_OPENROUTER_URL),
            opencode_url=source.get("OPENCODE_GO_ENDPOINT_URL", DIRECT_OPENCODE_URL),
            openrouter_key=source.get("OPENROUTER_API_KEY", ""),
            opencode_key=source.get("OPENCODE_API_KEY", ""),
        )
    else:
        raise SystemExit(f"unknown endpoint profile: {name}")
    return EndpointProfile(
        name=profile.name,
        openrouter_url=_normalize_endpoint_url(profile.openrouter_url),
        opencode_url=_normalize_endpoint_url(profile.opencode_url),
        openrouter_key=profile.openrouter_key,
        opencode_key=profile.opencode_key,
    )


def apply_endpoint_profile(profile: EndpointProfile) -> None:
    os.environ.update(profile.environment())
    openrouter_host, opencode_host = profile.hosts
    print(
        f"endpoint profile: {profile.name} (openrouter={openrouter_host}, opencode={opencode_host})"
    )


def catalog_environment(catalog: model_catalog.MantisCatalog) -> dict[str, str]:
    """Render catalog metadata and credentials for the child process only."""
    environment = model_catalog.render_mantis_environment(catalog)
    keys = model_catalog.resolve_provider_keys(catalog)
    environment.update(
        MANTIS_ENDPOINT_PROFILE="catalog",
        MANTIS_PROVIDER_KEYS=json.dumps(keys, sort_keys=True, separators=(",", ":")),
    )
    return environment


def apply_catalog(catalog: model_catalog.MantisCatalog) -> None:
    """Select catalog bindings without writing a credential to terminal output."""
    try:
        environment = catalog_environment(catalog)
    except model_catalog.CatalogError as error:
        raise SystemExit(str(error)) from error
    os.environ.update(environment)
    # The host catalog path drives the mount source; the container always sees
    # the file at the fixed mount target (the entrypoint falls back to it).
    os.environ["MANTIS_CATALOG_PATH"] = str(catalog.path)
    providers = catalog.bindings.providers
    hosts = ", ".join(
        f"{name}={_endpoint_host(binding.base_url)}" for name, binding in sorted(providers.items())
    )
    print(f"catalog: active ({hosts})")


def detect_backend() -> str:
    """Pick the runtime: native Apple container or docker compose."""
    forced = os.environ.get("MANTIS_STACK_BACKEND", "auto")
    if forced in ("native", "docker"):
        return forced
    if platform.system() != "Darwin" or shutil.which("container") is None:
        return "docker"
    probe = subprocess.run(
        ["container", "system", "status"], capture_output=True, text=True, timeout=10
    )
    return "native" if probe.returncode == 0 and "running" in probe.stdout else "docker"


def _run(argv: list[str]) -> None:
    print("+", " ".join(argv))
    subprocess.run(argv, check=True)


def _container_output(argv: list[str]) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout


def docker_args(command: str, flags: list[str]) -> list[str]:
    """Build a docker compose argv; flags: redis, conductor, -d, -f."""
    argv = ["docker", "compose", "-f", str(COMPOSE_FILE)]
    if "conductor" in flags:
        argv += ["-f", str(CONDUCTOR_FILE)]
    argv.append(command)
    if "redis" in flags:
        argv += ["--profile", "redis"]
    return argv + [f for f in flags if f not in ("redis", "conductor")]


def _image_present(tag: str) -> bool:
    return tag in _container_output(["container", "image", "list"])


def _pull_image(image: str) -> None:
    if image not in _container_output(["container", "image", "list"]):
        _run(["container", "image", "pull", image])


def _ensure_network_and_volumes(spec: dict) -> None:
    for net in spec["networks"]:
        if net not in _container_output(["container", "network", "list"]):
            _run(["container", "network", "create", net])
    for vol in spec["volume_names"].values():
        if vol not in _container_output(["container", "volume", "list"]):
            _run(["container", "volume", "create", vol])


def _mount_args(spec: dict, volumes: list[str]) -> list[str]:
    args = []
    for vol in volumes:
        source, target = vol.split(":", 1)
        ro = target.endswith(":ro")
        target = target[:-3] if ro else target
        if source.startswith("./") or source.startswith("/"):
            path = Path(source)
            if not path.is_absolute():
                path = REPO / path
            kind = f"type=bind,source={path},target={target}"
        else:
            source = spec["volume_names"].get(source, source)
            kind = f"type=volume,source={source},target={target}"
        args += ["--mount", kind + (",readonly" if ro else "")]
    return args


def _find_ip(data) -> str:
    if isinstance(data, dict):
        for key in ("ipAddress", "ipv4Address"):
            ip = data.get(key)
            if isinstance(ip, str) and ip:
                return ip.split("/")[0]
        for value in data.values():
            found = _find_ip(value)
            if found:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_ip(item)
            if found:
                return found
    return ""


def _native_redis_ip() -> str:
    out = _container_output(["container", "inspect", REDIS_NAME])
    return _find_ip(json.loads(out))


def native_build() -> None:
    build = load_spec(COMPOSE_FILE)["services"]["openfugu"]["build"]
    _run(
        [
            "container",
            "build",
            "-f",
            str(REPO / build["dockerfile"]),
            "-t",
            IMAGE_TAG,
            str(REPO),
        ]
    )


def _native_run_redis(spec: dict) -> None:
    svc = spec["services"]["redis"]
    if REDIS_NAME in _container_output(["container", "list"]):
        return
    _pull_image(svc["image"])
    _run(
        [
            "container",
            "run",
            "-d",
            "--name",
            REDIS_NAME,
            "--network",
            "mantis",
            "--mount",
            "type=volume,source=mantis-redis,target=/data",
            svc["image"],
        ]
        + list(svc["command"])
    )


def _native_catalog_mount(
    argv: list[str], env: dict[str, str]
) -> tuple[list[str], dict[str, str]]:
    """Translate the catalog file bind into a directory bind for the native runtime.

    The Apple ``container`` runtime only binds directories (docker binds files
    fine).  The compose spec mounts the catalog file at ``CATALOG_MOUNT_TARGET``;
    native runs instead bind the catalog's parent directory at the target's
    parent directory and set ``MANTIS_CATALOG_PATH`` to the in-container file
    path so the entrypoint validates the same file.
    """
    out: list[str] = []
    for arg in argv:
        if not arg.startswith("type=bind,source="):
            out.append(arg)
            continue
        parts = arg.split(",")
        source = next((p[len("source="):] for p in parts if p.startswith("source=")), "")
        target = next((p[len("target="):] for p in parts if p.startswith("target=")), "")
        if target != CATALOG_MOUNT_TARGET or not source or os.path.isdir(source):
            out.append(arg)
            continue
        parent = os.path.dirname(source)
        if not parent or not os.path.isdir(parent):
            out.append(arg)
            continue
        rebuilt = [
            f"type=bind,source={parent},target={os.path.dirname(CATALOG_MOUNT_TARGET)}"
        ]
        rebuilt += [part for part in parts if part == "readonly"]
        out.append(",".join(rebuilt))
        env["MANTIS_CATALOG_PATH"] = CATALOG_MOUNT_TARGET
    return out, env


def native_up(redis: bool, conductor: str | None, memory: str = "8G") -> None:
    spec = load_spec(COMPOSE_FILE)
    svc = spec["services"]["openfugu"]
    _ensure_network_and_volumes(spec)
    if svc["container_name"] in _container_output(["container", "list"]):
        validate_running_container()
        return
    if not _image_present(IMAGE_TAG):
        native_build()
    env = dict(svc["environment"])
    if conductor:
        env.update(
            MANTIS_LOCAL_CONDUCTOR="/app/checkpoint",
            MANTIS_CONDUCTOR_MODEL="gpt-5.6-luna-max",
            MANTIS_CONDUCTOR_DTYPE="float32",
            MANTIS_CONDUCTOR_MAX_NEW="128",
        )
    if redis:
        _native_run_redis(spec)
        env["MANTIS_REDIS_URL"] = f"redis://{_native_redis_ip()}:6379/0"
    argv = [
        "container",
        "run",
        "-d",
        "--name",
        svc["container_name"],
        "--network",
        "mantis",
        "--memory",
        memory,
    ]
    for port in svc["ports"]:
        argv += ["-p", port]
    volumes = svc["volumes"]
    if os.environ.get("MANTIS_ENDPOINT_PROFILE", "direct") != "catalog":
        # Outside catalog mode the catalog bind mount may point at a path that
        # does not exist; a missing bind source would fail the run for nothing.
        volumes = [
            volume for volume in volumes
            if not volume.endswith(f":{CATALOG_MOUNT_TARGET}:ro")
        ]
    argv += _mount_args(spec, volumes)
    if os.environ.get("MANTIS_ENDPOINT_PROFILE", "direct") == "catalog":
        argv, env = _native_catalog_mount(argv, env)
    if conductor:
        argv += ["--mount", f"type=bind,source={conductor},target=/app/checkpoint,readonly"]
    with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as tmp:
        tmp.write("".join(f"{k}={v}\n" for k, v in env.items()))
        env_file = tmp.name
    try:
        _run(argv + ["--env-file", env_file, IMAGE_TAG])
    finally:
        os.unlink(env_file)
    wait_ready()


def _expected_readiness_metadata() -> ReadinessMetadata:
    profile = os.environ.get("MANTIS_ENDPOINT_PROFILE", "direct")
    if profile == "catalog":
        try:
            bindings = model_catalog.load_runtime_bindings()
        except model_catalog.CatalogError as error:
            raise SystemExit(str(error)) from error
        if bindings is None:
            raise SystemExit("catalog endpoint profile requires rendered provider bindings")
        urls = {name: binding.base_url for name, binding in bindings.providers.items()}
        contract = os.environ.get("MANTIS_IDENTITY_CONTRACT", "")
        if not contract:
            raise SystemExit("catalog endpoint profile requires an identity contract")
        return {
            "status": "ready",
            "endpoint_profile": profile,
            "endpoint_hosts": {name: _endpoint_host(url) for name, url in urls.items()},
            "endpoint_fingerprints": {
                name: _endpoint_fingerprint(url) for name, url in urls.items()
            },
            "catalog_identity_contract": contract,
            "binding_fingerprint": model_catalog.runtime_binding_fingerprint(),
        }
    openrouter_url = os.environ.get("OPENROUTER_BASE_URL", DIRECT_OPENROUTER_URL)
    opencode_url = os.environ.get("OPENCODE_GO_ENDPOINT_URL", DIRECT_OPENCODE_URL)
    return {
        "status": "ready",
        "endpoint_profile": profile,
        "endpoint_hosts": {
            "openrouter": _endpoint_host(openrouter_url),
            "opencode": _endpoint_host(opencode_url),
        },
        "endpoint_fingerprints": {
            "openrouter": _endpoint_fingerprint(openrouter_url),
            "opencode": _endpoint_fingerprint(opencode_url),
        },
    }


def _readiness_matches(response: httpx.Response, expected: ReadinessMetadata) -> bool:
    if response.status_code != 200:
        return False
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and all(body.get(key) == value for key, value in expected.items())


def validate_running_container() -> None:
    expected = _expected_readiness_metadata()
    try:
        matches = _readiness_matches(httpx.get(READY_URL, timeout=2), expected)
    except httpx.HTTPError:
        matches = False
    if not matches:
        raise SystemExit(
            f"{ORCHESTRATOR} is already running with different or unavailable endpoint "
            "metadata; use the restart command to replace it"
        )
    print(f"{ORCHESTRATOR} already running with the selected endpoint profile")


def wait_ready(timeout: int = START_TIMEOUT) -> None:
    expected = _expected_readiness_metadata()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _readiness_matches(httpx.get(READY_URL, timeout=2), expected):
                print("ready")
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    endpoint_hosts = expected["endpoint_hosts"]
    hosts = ", ".join(sorted(set(endpoint_hosts.values())))
    raise SystemExit(
        f"not ready within {timeout}s: expected endpoint profile "
        f"{expected['endpoint_profile']} on {hosts}; readiness metadata did not match"
    )


def native_down() -> None:
    running = _container_output(["container", "list"])
    all_containers = _container_output(["container", "list", "--all"])
    for name in (ORCHESTRATOR, REDIS_NAME):
        if name in running:
            _run(["container", "stop", "--time", "30", name])
        if name in all_containers:
            _run(["container", "delete", name])


def native_logs() -> None:
    _run(["container", "logs", "-f", ORCHESTRATOR])


def native_status() -> None:
    _run(["container", "list"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["up", "restart", "down", "build", "logs", "status"])
    parser.add_argument("--redis", action="store_true", help="also run the redis service")
    parser.add_argument(
        "--endpoint-profile",
        choices=["direct"],
        default="direct",
        help="provider endpoint and credential profile (default: direct)",
    )
    parser.add_argument(
        "--memory", default="8G", help="container memory limit (native backend; default 8G)"
    )
    parser.add_argument(
        "--conductor", metavar="DIR", help="mount a conductor checkpoint (eval override)"
    )
    args = parser.parse_args()
    if args.command in ("up", "restart"):
        load_dotenv_defaults()
        try:
            catalog = model_catalog.load_mantis_catalog()
        except model_catalog.CatalogError as error:
            raise SystemExit(str(error)) from error
        if catalog is None:
            apply_endpoint_profile(resolve_endpoint_profile(args.endpoint_profile))
        else:
            apply_catalog(catalog)
    backend = detect_backend()
    print(f"backend: {backend}")
    if backend == "docker":
        flags = []
        if args.redis:
            flags.append("redis")
        if args.conductor:
            flags.append("conductor")
        command = {
            "up": "up",
            "restart": "up",
            "down": "down",
            "build": "build",
            "logs": "logs",
            "status": "ps",
        }[args.command]
        if args.command in ("up", "restart"):
            flags.append("-d")
            if args.command == "restart":
                flags.append("--force-recreate")
        elif args.command == "logs":
            flags.append("-f")
        _run(docker_args(command, flags))
        if args.command in ("up", "restart"):
            wait_ready()
        return
    if args.command == "up":
        native_up(args.redis, args.conductor, args.memory)
    elif args.command == "restart":
        redis_was_running = REDIS_NAME in _container_output(["container", "list"])
        native_down()
        native_up(args.redis or redis_was_running, args.conductor, args.memory)
    elif args.command == "down":
        native_down()
    elif args.command == "build":
        native_build()
    elif args.command == "logs":
        native_logs()
    else:
        native_status()


if __name__ == "__main__":
    main()
