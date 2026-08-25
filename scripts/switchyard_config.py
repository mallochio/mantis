"""Render a Switchyard routes.toml from the shared, secret-free catalog.

The catalog [base] section names the efficient/capable models and the provider
those models ride on. Changing a model or provider in the catalog is enough to
retarget Base; this module never hard-codes upstream IDs.

The Switchyard route id is the public Mantis model name so the API can forward
chat requests without rewriting ``model``.
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any, TextIO

from model_catalog import catalog_path
from model_catalog_schema import (
    BaseRoute,
    BaseTarget,
    CatalogError,
    ProviderBinding,
    load_base_route,
)

SWITCHYARD_ROUTE_ID = "mantis/base"


def load_catalog_root(path: str | Path | None = None) -> tuple[Path, dict[str, Any]]:
    selected, explicit = (Path(path).expanduser(), True) if path is not None else catalog_path()
    if not selected.exists():
        raise CatalogError(f"catalog does not exist: {selected}")
    if not selected.is_file():
        raise CatalogError(f"catalog is not a file: {selected}")
    try:
        with selected.open("rb") as handle:
            root = tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise CatalogError(f"invalid TOML in catalog: {error}") from error
    if not isinstance(root, dict):
        raise CatalogError("catalog root must be a table")
    if "base" not in root:
        if explicit:
            raise CatalogError(f"catalog {selected} has no [base] route")
        raise CatalogError("catalog has no [base] route")
    return selected, root


def load_switchyard_route(path: str | Path | None = None) -> BaseRoute:
    _selected, root = load_catalog_root(path)
    return load_base_route(root)


def _toml_str(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _toml_key(name: str) -> str:
    if name.isidentifier() and "." not in name:
        return name
    return _toml_str(name)


def _client_key(provider: str) -> str:
    return provider.replace(".", "_")


def _target_extra_body(target: BaseTarget) -> str:
    lines: list[str] = []
    if target.max_tokens is not None:
        lines.append(f"max_tokens = {target.max_tokens}")
    if target.reasoning_effort is not None:
        lines.append(f"reasoning_effort = {_toml_str(target.reasoning_effort)}")
    if not lines:
        return ""
    body = "\n".join(f"  {line}" for line in lines)
    return f"\n[targets.{target.role}.extra_body]\n{body}\n"


def _render_clients(route: BaseRoute) -> str:
    blocks: list[str] = []
    seen: set[str] = set()
    for target in (route.efficient, route.capable):
        key = _client_key(target.provider)
        if key in seen:
            continue
        seen.add(key)
        provider = route.providers[target.provider]
        blocks.append(_render_client(key, provider, target.wire_format))
    return "\n".join(blocks)


def _render_client(key: str, provider: ProviderBinding, wire_format: str) -> str:
    return "\n".join(
        [
            f"[llm_clients.{_toml_key(key)}]",
            f"format = {_toml_str(wire_format)}",
            f"base_url = {_toml_str(provider.base_url)}",
            f"api_key_env = {_toml_str(provider.credential_env)}",
            "",
        ]
    )


def _render_target(target: BaseTarget) -> str:
    client = _client_key(target.provider)
    return (
        f"[targets.{target.role}]\n"
        f"id = {_toml_str(target.upstream_model)}\n"
        f"llm_client = {_toml_str(client)}\n"
        f"{_target_extra_body(target)}"
    )


def render_switchyard_toml(route: BaseRoute) -> str:
    """Render a Switchyard native TOML deployment from a parsed [base] route."""
    route_block = "\n".join(
        [
            "[routes.mantis_base]",
            f"id = {_toml_str(SWITCHYARD_ROUTE_ID)}",
            'type = "stage_router"',
            'capable_target = "capable"',
            'efficient_target = "efficient"',
            f"picker = {_toml_str(route.picker)}",
            f"confidence_threshold = {route.confidence_threshold}",
            f"recent_turn_window = {route.recent_turn_window}",
            "",
        ]
    )
    return (
        "schema_version = 1\n"
        f"# generated from catalog [base] revision {route.revision}\n"
        "\n"
        f"{_render_clients(route)}"
        f"{_render_target(route.efficient)}"
        f"{_render_target(route.capable)}"
        f"{route_block}"
    )


def write_switchyard_toml(route: BaseRoute, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    text = render_switchyard_toml(route)
    output.write_text(text)
    output.chmod(0o600)
    return output


def main(argv: list[str] | None = None, stdout: TextIO | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "render"))
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        route = load_switchyard_route(args.catalog)
    except CatalogError as error:
        print(error, file=sys.stderr)
        return 1
    if args.command == "validate":
        print(
            f"valid Switchyard base route: {SWITCHYARD_ROUTE_ID} "
            f"({route.efficient.upstream_model} -> {route.capable.upstream_model})",
            file=stdout,
        )
        return 0
    text = render_switchyard_toml(route)
    if args.output is not None:
        write_switchyard_toml(route, args.output)
        return 0
    print(text, end="" if text.endswith("\n") else "\n", file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
