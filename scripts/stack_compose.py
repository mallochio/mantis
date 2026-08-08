#!/usr/bin/env python3
"""Parse docker-compose.yml into an interpolated, runtime-agnostic spec."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


class ComposeError(SystemExit):
    pass


def _interpolate(value: str, env: dict[str, str], source: Path) -> str:
    """Expand compose ${VAR}, ${VAR:-default}, ${VAR:?msg}; defaults may nest."""

    def scan_default(i: int) -> tuple[str, int]:
        out, depth = "", 1
        while depth:
            if value[i : i + 2] == "${":
                depth += 1
                out += "${"
                i += 2
            elif value[i] == "}":
                depth -= 1
                out += "}" if depth else ""
                i += 1
            else:
                out += value[i]
                i += 1
        return out, i

    def scan(i: int) -> tuple[str, str | None, str | None, int]:
        name, i = "", i
        while value[i].isalnum() or value[i] == "_":
            name += value[i]
            i += 1
        if value[i : i + 2] == ":-":
            default, i = scan_default(i + 2)
            return name, default, None, i
        if value[i : i + 2] == ":?":
            j = value.index("}", i + 2)
            return name, None, value[i + 2 : j], j + 1
        assert value[i] == "}", f"unterminated ${{...}} in {value!r}"
        return name, None, None, i + 1

    out, i = "", 0
    while i < len(value):
        if value[i : i + 2] == "${":
            name, default, required, i = scan(i + 2)
            if name in env:
                out += env[name]
            elif required is not None:
                raise ComposeError(f"{source}: {required}")
            else:
                out += _interpolate(default, env, source) if default is not None else ""
        else:
            out += value[i]
            i += 1
    return out


def load_spec(compose_path: Path, env: dict[str, str] | None = None) -> dict:
    """Load docker-compose.yml with env interpolation: the single source of truth."""
    data = yaml.safe_load(compose_path.read_text())
    env = dict(os.environ if env is None else env)
    services = {}
    for name, svc in data["services"].items():
        services[name] = {
            "image": svc.get("image"),
            "build": svc.get("build"),
            "container_name": svc.get("container_name", name),
            "ports": svc.get("ports", []),
            "environment": {
                k: _interpolate(v, env, compose_path)
                for k, v in (svc.get("environment") or {}).items()
            },
            "volumes": [_interpolate(v, env, compose_path) for v in svc.get("volumes", [])],
            "command": svc.get("command"),
        }
    volume_names = {
        key: (cfg or {}).get("name", key) for key, cfg in (data.get("volumes") or {}).items()
    }
    return {
        "services": services,
        "networks": list((data.get("networks") or {}).keys()),
        "volume_names": volume_names,
    }
