"""Shared request and routing types for mantis-fusion.

These types are used by both the public API (Pydantic validation) and the
internal orchestrator (converted to plain dicts when needed).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

WORKSPACE_ROOTS_ENV = "MANTIS_FUSION_WORKSPACE_ROOTS"


def allowed_workspace(value: str) -> str:
    """Resolve a client workspace path and require it under an allowlisted root."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("tool_options.workspace must be an absolute path")
    resolved = path.resolve()
    raw_roots = os.environ.get(WORKSPACE_ROOTS_ENV, "").split(os.pathsep)
    roots = [Path(root).expanduser().resolve() for root in raw_roots if root.strip()]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise ValueError(f"tool_options.workspace {resolved} is not under {WORKSPACE_ROOTS_ENV}")
    if not resolved.is_dir():
        raise ValueError(f"tool_options.workspace {resolved} is not a directory")
    return str(resolved)


class FusionRunBudget(BaseModel):
    """Run-level token/turn/wall-time budget for a Fusion run."""

    model_config = ConfigDict(extra="ignore")

    max_turns: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    timeout_ms: int | None = Field(default=None, ge=0)


class FusionToolOptions(BaseModel):
    """Capability bundles that filter the tool set available to workers."""

    model_config = ConfigDict(extra="forbid")

    enabled: list[str] = Field(default_factory=list)
    server_execution: bool = False
    workspace: str | None = None

    @field_validator("workspace")
    @classmethod
    def _check_workspace(cls, value: str | None) -> str | None:
        """Accept only workspaces under the configured allowlist roots."""
        return None if value is None else allowed_workspace(value)


class FusionRoutingConfig(BaseModel):
    """Catalog-level routing hints for main/sidekick slots."""

    model_config = ConfigDict(extra="ignore")

    main: str | list[str] = "gpt-5_6-sol"
    main_reasoning: str | None = None
    sidekick: str | list[str] = "gpt-5_6-luna"
    sidekick_reasoning: str | None = None
    fallback_on_escalate: bool = True
