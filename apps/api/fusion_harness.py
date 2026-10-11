"""Harness-neutral workspace discovery and delegation tool compatibility helpers."""

from __future__ import annotations

import copy
import logging
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

from fusion_types import allowed_workspace

LOGGER = logging.getLogger(__name__)
DELEGATION_TOOL_NAMES = ("sidekick", "read_subagent")


def workspace_from_prompt(system_prompt: str) -> str | None:
    """Extract a working directory announced in a harness system prompt."""
    for pattern in _workspace_patterns():
        match = pattern.search(system_prompt)
        if match:
            value = match.group("path").strip().strip("\"'`.,; )]")
            if value.startswith("/"):
                return value
    return None


def _workspace_patterns() -> tuple[re.Pattern[str], ...]:
    """Build patterns for common plain-text and environment-block directory forms."""
    flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
    return (
        re.compile(r"<cwd>\s*(?P<path>/[^<]+?)\s*</cwd>", flags),
        re.compile(
            r"<env[^>]*>(?:(?!</env>).)*?\b(?:PWD|CWD|WORKING_DIRECTORY)\s*[:=]\s*"
            r"(?P<path>/[^\r\n<]+)",
            flags,
        ),
        re.compile(
            r"^\s*(?:current\s+working\s+directory|working\s+directory|"
            r"current\s+directory|cwd)\s*(?::|=|\bis\b)?\s*"
            r"(?P<path>/[^\r\n<]+)",
            flags,
        ),
    )


def workspace_candidate(
    explicit: str | None,
    header: str | None,
    system_prompt: str,
    catalog_default: str | None,
    environment: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Select the highest-priority workspace and identify its source."""
    env = {} if environment is None else environment
    candidates = (
        (explicit, "explicit"),
        (header, "header"),
        (workspace_from_prompt(system_prompt), "system prompt"),
        (env.get("MANTIS_FUSION_DEFAULT_WORKSPACE") or catalog_default, "default"),
    )
    return next(((value, source) for value, source in candidates if value), (None, None))


def resolve_workspace(
    explicit: str | None,
    header: str | None,
    system_prompt: str,
    catalog_default: str | None,
) -> str | None:
    """Validate a selected workspace, rejecting explicit paths and ignoring discoveries."""
    candidate, source = workspace_candidate(
        explicit, header, system_prompt, catalog_default, os.environ
    )
    if candidate is None:
        return None
    if source == "explicit":
        return allowed_workspace(candidate)
    try:
        return allowed_workspace(candidate)
    except ValueError as error:
        LOGGER.info("Ignoring discovered Fusion workspace from %s: %s", source, error)
        return None


def effective_delegation_names(client_tools: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Choose deterministic server tool names that do not shadow client tools."""
    declared = {_tool_name(tool) for tool in client_tools}
    used = set(declared)
    names: dict[str, str] = {}
    for canonical in DELEGATION_TOOL_NAMES:
        candidate = canonical
        if candidate in declared:
            candidate = f"fusion_{canonical}"
            suffix = 2
            while candidate in used:
                candidate = f"fusion_{canonical}_{suffix}"
                suffix += 1
        names[canonical] = candidate
        used.add(candidate)
    return names


def rename_schemas(
    schemas: Sequence[Mapping[str, Any]], names: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Copy tool schemas and substitute effective names for canonical server tools."""
    renamed: list[dict[str, Any]] = [copy.deepcopy(dict(schema)) for schema in schemas]
    for schema in renamed:
        function = schema.get("function")
        if isinstance(function, dict):
            current = function.get("name")
            if isinstance(current, str) and current in names:
                function["name"] = names[current]
    return renamed


def canonicalize(name: str, names: Mapping[str, str]) -> str:
    """Map an effective per-run delegation name back to its server-side name."""
    return next((canonical for canonical, effective in names.items() if name == effective), name)


def delegation_name_policy(names: Mapping[str, str]) -> str:
    """Describe the effective delegation names for the lead prompt."""
    return (
        f"Use {names['sidekick']} to delegate work and {names['read_subagent']} to read the "
        "sidekick's status or latest report."
    )


def lead_prompt_with_names(base_prompt: str, names: Mapping[str, str]) -> str:
    """Append per-run delegation aliases only when they differ from canonical names."""
    if any(names[name] != name for name in DELEGATION_TOOL_NAMES):
        return f"{base_prompt}\n\n{delegation_name_policy(names)}"
    return base_prompt


def _tool_name(schema: Mapping[str, Any]) -> str:
    """Return the name from an OpenAI-style function schema."""
    function = schema.get("function")
    return str(function.get("name", "")) if isinstance(function, Mapping) else ""
