#!/usr/bin/env python3
"""Stack driver for the mantis container stack.

One spec (docker-compose.yml) drives two runtimes:
- native: Apple `container` CLI on macOS 26+ (no Docker daemon)
- docker: `docker compose` everywhere else

Override the backend with MANTIS_STACK_BACKEND=auto|native|docker.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import httpx
import stack_compose

load_spec = stack_compose.load_spec

REPO = Path(__file__).resolve().parent.parent
COMPOSE_FILE = REPO / "docker-compose.yml"
CONDUCTOR_FILE = REPO / "eval" / "docker-compose.conductor.yml"
IMAGE_TAG = "mantis/openfugu:local"
ORCHESTRATOR = "mantis-orchestrator"
REDIS_NAME = "mantis-redis"
READY_URL = "http://127.0.0.1:8088/ready"
START_TIMEOUT = 90


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


def native_up(redis: bool, conductor: str | None) -> None:
    spec = load_spec(COMPOSE_FILE)
    svc = spec["services"]["openfugu"]
    _ensure_network_and_volumes(spec)
    if svc["container_name"] in _container_output(["container", "list"]):
        print(f"{svc['container_name']} already running")
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
    ]
    for port in svc["ports"]:
        argv += ["-p", port]
    argv += _mount_args(spec, svc["volumes"])
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


def wait_ready(timeout: int = START_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(READY_URL, timeout=2).status_code == 200:
                print("ready")
                return
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise SystemExit(f"not ready within {timeout}s")


def native_down() -> None:
    for name in (ORCHESTRATOR, REDIS_NAME):
        if name in _container_output(["container", "list"]):
            _run(["container", "stop", "--time", "30", name])
            _run(["container", "delete", name])


def native_logs() -> None:
    _run(["container", "logs", "-f", ORCHESTRATOR])


def native_status() -> None:
    _run(["container", "list"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["up", "down", "build", "logs", "status"])
    parser.add_argument("--redis", action="store_true", help="also run the redis service")
    parser.add_argument(
        "--conductor", metavar="DIR", help="mount a conductor checkpoint (eval override)"
    )
    args = parser.parse_args()
    backend = detect_backend()
    print(f"backend: {backend}")
    if backend == "docker":
        flags = []
        if args.redis:
            flags.append("redis")
        if args.conductor:
            flags.append("conductor")
        command = {"up": "up", "down": "down", "build": "build", "logs": "logs", "status": "ps"}[
            args.command
        ]
        if args.command == "up":
            flags.append("-d")
        elif args.command == "logs":
            flags.append("-f")
        _run(docker_args(command, flags))
        return
    if args.command == "up":
        native_up(args.redis, args.conductor)
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
