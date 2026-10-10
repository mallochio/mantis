"""Explicit model and fallback-pool selection for mantis-fusion.

The dynamic ``mantis/base`` route remains virtual so Switchyard is its sole
routing authority. Explicit ``base:efficient`` and ``base:capable`` references
still resolve to concrete provider specs, as do ordinary worker slots.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from fusion_types import FusionRoutingConfig
from model_catalog import catalog_path
from model_catalog_schema import BaseRoute, load_base_route

_BASE_ROUTE_CACHE: dict[tuple[str, int, int], BaseRoute | None] = {}
_DEFAULT_COMPACTION_CUTOFF = 0.85


def _compaction_cutoff() -> float:
    """Return the complexity cutoff for post-compaction main demotion."""
    raw = os.environ.get("MANTIS_COMPACTION_CUTOFF", "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_COMPACTION_CUTOFF
    except ValueError:
        return _DEFAULT_COMPACTION_CUTOFF
    if 0.0 < value <= 1.0:
        return value
    return _DEFAULT_COMPACTION_CUTOFF


class FusionRouter:
    """Pick a virtual route or concrete model spec for one Fusion lane."""

    def __init__(
        self,
        config: FusionRoutingConfig,
        role: str,
        base_route: BaseRoute | None = None,
    ) -> None:
        self.config = config
        self.role = role
        self.base_route = base_route
        self.effort = config.main_reasoning if role == "main" else config.sidekick_reasoning

    @staticmethod
    def _load_base_route(catalog: str | Path | None = None) -> BaseRoute | None:
        """Read the [base] stage router from an explicit catalog if given.

        Defaults to the active catalog resolved from ``AI_ROUTING_CONFIG`` /
        ``MANTIS_CATALOG_PATH`` so production callers keep working without
        changes, while tests can inject a hermetic catalog path instead of
        depending on ``~/`` state. Parsed routes are cached per file identity
        so per-lane router construction stays cheap.
        """
        if catalog is not None:
            path = Path(catalog).expanduser()
        else:
            path, _ = catalog_path()
        if not path.is_file():
            return None
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key not in _BASE_ROUTE_CACHE:
            route: BaseRoute | None = None
            try:
                with path.open("rb") as handle:
                    root = tomllib.load(handle)
                if root.get("version") == 1:
                    route = load_base_route(root)
            except (OSError, tomllib.TOMLDecodeError, ValueError):
                route = None
            _BASE_ROUTE_CACHE[key] = route
        return _BASE_ROUTE_CACHE[key]

    @classmethod
    def from_config(
        cls,
        config: FusionRoutingConfig,
        role: str,
        catalog: str | Path | None = None,
    ) -> FusionRouter:
        return cls(config, role, cls._load_base_route(catalog))

    def select(self, escalation_count: int = 0) -> str:
        """Return a provider-executable spec for this escalation level.

        ``escalation_count`` counts times the sidekick escalated or failed.
        When ``fallback_on_escalate`` is set and a pool is configured, each
        escalation promotes to the next slot in the pool.

        Pool ordering is cheapest-first for every role, so a higher index is
        always a stronger, costlier slot. Escalation therefore always promotes
        and never degrades capability.
        """
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool)

        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")

        index = 0
        if self.config.fallback_on_escalate:
            index = min(escalation_count, len(pool) - 1)
        return self._resolve(pool[index])

    def strongest(self) -> str:
        """Return the strongest, costliest slot in this role's pool."""
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool)
        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")
        return self._resolve(pool[-1])

    def select_at_compaction(
        self,
        complexity: float,
        previous: str,
        failure_count: int = 0,
        cache_warm: bool = False,
    ) -> str:
        """Reroute only where compaction already forces a cache miss.

        A warm cache keeps the previous model unless the lane is stuck
        (``failure_count > 0``). On a cold cache the main role drops to the
        cheapest slot for low-complexity work, holds at least the second slot
        for high-complexity work, and never lands on a slot weaker than the
        one it already runs; any role promotes one slot per failure.
        """
        pool: str | list[str] = self.config.main if self.role == "main" else self.config.sidekick
        if isinstance(pool, str):
            return self._resolve(pool)
        if not pool:
            raise ValueError(f"fusion {self.role} pool is empty")
        resolved = [self._resolve(candidate) for candidate in pool]
        current = resolved.index(previous) if previous in resolved else 0
        if cache_warm and failure_count == 0:
            return previous
        if failure_count > 0:
            index = min(current + 1, len(pool) - 1)
        elif self.role == "main":
            if complexity < _compaction_cutoff() and len(pool) > 1:
                index = 0
            else:
                index = max(current, 1) if len(pool) > 1 else current
        else:
            index = current
        return resolved[index]

    def _resolve(self, value: Any) -> str:
        spec = str(value).strip()
        if not spec:
            raise ValueError(f"fusion {self.role} slot is empty")

        # Keep the dynamic base route virtual; Switchyard resolves it per turn.
        if spec in ("mantis/base", "base"):
            return "mantis/base"

        if spec.startswith("base:"):
            return self._apply_effort(self._resolve_base(spec.split(":", 1)[1].strip()))

        # Bare "efficient" or "capable" only make sense when the base route
        # is configured; otherwise treat them as literal slot names.
        if spec in ("efficient", "capable") and self.base_route is not None:
            return self._apply_effort(self._resolve_base(spec))

        # Plain slot or provider/model spec.
        return self._apply_effort(spec)

    def _apply_effort(self, spec: str) -> str:
        """Attach the role's default reasoning effort to a resolved spec.

        An explicit ``|effort`` suffix — including one carried by a resolved
        ``base:`` target — always wins over the role default.
        """
        if not self.effort or "|" in spec:
            return spec
        return f"{spec}|{self.effort}"

    def _resolve_base(self, role: str) -> str:
        if self.base_route is None:
            raise ValueError(
                f"fusion {self.role} requested base route but no [base] "
                "section is configured in the catalog"
            )
        if role not in ("efficient", "capable"):
            raise ValueError(f"unknown base route role {role!r} for fusion {self.role}")
        target = self.base_route.efficient if role == "efficient" else self.base_route.capable
        spec = f"{target.provider}/{target.upstream_model}"
        if target.reasoning_effort:
            spec += f"|{target.reasoning_effort}"
        return spec
