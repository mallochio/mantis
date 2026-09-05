"""mantis-fusion adaptive model selection.

Resolves catalog route references such as ``mantis/base`` or ``base:efficient``
into concrete provider specs and can promote through a pool of slots when a
sidekick escalates or fails.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
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
    def _load_base_route(catalog: str | Path | None = None) -> BaseRoute | None:
        """Read the [base] stage router from an explicit catalog if given.

        Defaults to the active catalog resolved from ``AI_ROUTING_CONFIG`` /
        ``MANTIS_CATALOG_PATH`` so production callers keep working without
        changes, while tests can inject a hermetic catalog path instead of
        depending on ``~/`` state.
        """
        if catalog is not None:
            path = Path(catalog).expanduser()
        else:
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
    def from_config(
        cls,
        config: FusionRoutingConfig,
        role: str,
        catalog: str | Path | None = None,
    ) -> FusionRouter:
        return cls(config, role, cls._load_base_route(catalog))

    def select(self, turn_index: int = 0, escalation_count: int = 0) -> str:
        """Return a provider-executable spec for this turn.

        ``turn_index`` counts sidekick iterations; ``escalation_count`` counts
        times the sidekick escalated or failed. When ``fallback_on_escalate``
        is set and a pool is configured, each escalation promotes to the next
        slot in the pool.

        Pool ordering is cheapest-first for every role, so a higher index is
        always a stronger, costlier slot. Escalation therefore always promotes
        and never degrades capability.
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

    def strongest(self) -> str:
        """Return the strongest, costliest slot in this role's pool."""
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool, 0)
        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")
        return self._resolve(pool[-1], 0)

    def select_at_compaction(
        self,
        complexity: float,
        previous: str,
        failure_count: int = 0,
        cache_warm: bool = False,
    ) -> str:
        """Reroute only where compaction already forces a cache miss.

        When ``cache_warm`` is True the previous model still has a warm
        provider prompt cache. Switching models would bust the cache for no
        quality gain, so we prefer staying on the current model unless the
        task is stuck (``failure_count > 0``) or the complexity clearly
        demands a stronger model.
        """
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool, failure_count)
        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")
        # Cache-aware heuristic: if the cache is warm and the task is not
        # stuck, stay on the current model to avoid a costly cache miss.
        if cache_warm and failure_count == 0:
            return previous
        if self.role == "main":
            if complexity < 0.85 and len(pool) > 1:
                index = 0
            else:
                resolved = [self._resolve(candidate, 0) for candidate in pool]
                index = resolved.index(previous) if previous in resolved else len(pool) - 1
        else:
            index = min(1 if failure_count > 0 else 0, len(pool) - 1)
        return self._resolve(pool[index], failure_count)

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
