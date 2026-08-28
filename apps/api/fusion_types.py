"""Shared request and plan types for mantis-fusion.

These types are used by both the public API (Pydantic validation) and the
internal orchestrator (converted to plain dicts when needed).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FusionWorkerProfile(BaseModel):
    """A specialist sidekick profile the planner may assign to a task."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    instructions: str = ""
    description: str = ""
    model: str | None = None  # overrides the default sidekick slot
    tools: list[str] | None = None  # allowed tool names, or None for all


class FusionRunBudget(BaseModel):
    """Run-level token/turn/wall-time budget for a Fusion run."""

    model_config = ConfigDict(extra="ignore")

    max_turns: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    timeout_ms: int | None = Field(default=None, ge=0)


class FusionToolOptions(BaseModel):
    """Capability bundles that filter the tool set available to workers."""

    model_config = ConfigDict(extra="ignore")

    enabled: list[str] = Field(default_factory=list)


class FusionSidekickAssignment(BaseModel):
    """One sidekick task assigned by the planner."""

    model_config = ConfigDict(extra="ignore")

    task: str = Field(min_length=1)
    profile: str | None = None


class FusionPlan(BaseModel):
    """Structured output from the main planning phase."""

    model_config = ConfigDict(extra="ignore")

    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    main_task: str = Field(min_length=1)
    sidekick_assignments: list[FusionSidekickAssignment] = Field(default_factory=list)
    verification_commands: list[str] = Field(default_factory=list)


class FusionRoutingConfig(BaseModel):
    """Catalog-level routing hints for main/sidekick slots."""

    model_config = ConfigDict(extra="ignore")

    main: str | list[str] = "gpt-5_6-sol"
    sidekick: str | list[str] = "gpt-5_6-luna"
    fallback_on_escalate: bool = True
