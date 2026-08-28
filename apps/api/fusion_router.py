"""mantis-fusion adaptive model selection.

Resolves catalog route references such as ``mantis/base`` or ``base:efficient``
into concrete provider specs and can promote through a pool of slots when a
sidekick escalates or fails.
"""

from __future__ import annotations

import tomllib
from typing import Any

from fusion_types import FusionRoutingConfig
from model_catalog import catalog_path
from model_catalog_schema import BaseRoute, load_base_route


class FusionRouter:
    """Pick a concrete model spec for one Fusion lane per turn."""

    def __init__(
        self,
        config: FusionRoutingConfig,
        role: str,
        base_route: BaseRoute | None = None,
    ) -> None:
        self.config = config
        self.role = role
        self.base_route = base_route

    @staticmethod
    def _load_base_route() -> BaseRoute | None:
        """Read the active catalog's [base] stage router, if present."""
        path, _ = catalog_path()
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                root = tomllib.load(handle)
            if root.get("version") != 1:
                return None
            return load_base_route(root)
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            return None

    @classmethod
    def from_config(cls, config: FusionRoutingConfig, role: str) -> FusionRouter:
        return cls(config, role, cls._load_base_route())

    def select(self, turn_index: int = 0, escalation_count: int = 0) -> str:
        """Return a provider-executable spec for this turn.

        ``turn_index`` counts sidekick iterations; ``escalation_count`` counts
        times the sidekick escalated or failed. When ``fallback_on_escalate``
        is set and a pool is configured, each escalation promotes to the next
        slot in the pool.
        """
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool, escalation_count)

        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")

        index = 0
        if self.config.fallback_on_escalate:
            index = min(escalation_count, len(pool) - 1)
        return self._resolve(pool[index], turn_index + escalation_count)

    def _resolve(self, value: Any, offset: int) -> str:
        spec = str(value).strip()
        if not spec:
            raise ValueError(f"fusion {self.role} slot is empty")

        # Direct references to the base stage router.
        if spec in ("mantis/base", "base"):
            return self._resolve_base("capable" if self.role == "main" else "efficient")

        if spec.startswith("base:"):
            return self._resolve_base(spec.split(":", 1)[1].strip())

        # Bare "efficient" or "capable" only make sense when the base route
        # is configured; otherwise treat them as literal slot names.
        if spec in ("efficient", "capable") and self.base_route is not None:
            return self._resolve_base(spec)

        # Plain slot or provider/model spec.
        return spec

    def _resolve_base(self, role: str) -> str:
        if self.base_route is None:
            raise ValueError(
                f"fusion {self.role} requested base route but no [base] "
                "section is configured in the catalog"
            )
        target = self.base_route.efficient if role == "efficient" else self.base_route.capable
        spec = f"{target.provider}/{target.upstream_model}"
        if target.reasoning_effort:
            spec += f"|{target.reasoning_effort}"
        return spec
