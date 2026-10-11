"""Runtime Fusion catalog configuration."""

from __future__ import annotations

import threading
import tomllib
from pathlib import Path
from typing import Any, cast

import model_catalog
import providers
from fusion_router import FusionRouter
from fusion_types import FusionRoutingConfig, FusionRunBudget, FusionToolOptions


def _validate_pool_ordering(config: FusionRoutingConfig) -> None:
    """Reject role pools whose known prices are not cheapest-first."""
    prices = providers._price_map()
    if not prices:
        return
    for role in ("main", "sidekick"):
        pool = getattr(config, role)
        if not isinstance(pool, list) or len(pool) < 2:
            continue
        raw_keys = [providers._price_key(str(slot), prices) for slot in pool]
        keys = [key for key in raw_keys if key is not None]
        if len(keys) != len(raw_keys):
            continue
        for earlier, later in zip(keys, keys[1:], strict=False):
            if prices[later][0] < prices[earlier][0]:
                raise ValueError(
                    f"fusion {role} pool is not cheapest-first: {later} is cheaper "
                    f"than {earlier} before it"
                )


class FusionConfig:
    """Runtime fusion settings loaded from the shared catalog."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or model_catalog.catalog_path()[0]
        self._raw: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "_raw": self._raw}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path = state["path"]
        self._raw = state.get("_raw")
        self._lock = threading.Lock()

    def _load(self) -> dict[str, Any]:
        if self._raw is not None:
            return self._raw
        with self._lock:
            if self._raw is not None:
                return self._raw
            if not self.path.exists():
                self._raw = {}
            else:
                with self.path.open("rb") as handle:
                    root = tomllib.load(handle)
                self._raw = root.get("fusion", {})
            self._reject_removed_controls()
            return self._raw

    def _reject_removed_controls(self) -> None:
        """Reject obsolete marker and multi-lane catalog controls."""
        removed = {
            "delegation",
            "main_tools",
            "max_follow_ups",
            "sidekick_max_follow_ups",
            "worker_profiles",
        }
        unsupported = removed.intersection(self._raw or {})
        if unsupported:
            self._raw = None
            raise ValueError(f"unsupported Fusion settings: {', '.join(sorted(unsupported))}")

    def main_slot(self) -> str:
        return self._load().get("main") or "gpt-5_6-sol"

    def sidekick_slot(self) -> str:
        return self._load().get("sidekick") or "gpt-5_6-luna"

    def sidekick_max_tool_rounds(self) -> int:
        raw = self._load().get("sidekick_max_tool_rounds")
        if raw is None or raw == "":
            return 0
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 0
        return max(0, value)

    def sidekick_max_consecutive_errors(self) -> int:
        raw = self._load().get("sidekick_max_consecutive_errors")
        if raw is None or raw == "":
            return 3
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 3

    def run_max_turns(self) -> int:
        raw = self._load().get("run_max_turns") or "64"
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return 64
        return max(1, value)

    def _explicit_limit(self, key: str) -> int | None:
        raw = self._load().get(key)
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def context_window(self) -> int | None:
        return self._explicit_limit("context_window") or self._explicit_limit(
            "context_token_limit"
        )

    def max_output_tokens(self) -> int | None:
        return self._explicit_limit("max_output_tokens")

    def main_routing(self) -> FusionRoutingConfig:
        fields = FusionRoutingConfig.model_fields
        data = {k: v for k, v in (self._load() or {}).items() if k in fields}
        config = cast(FusionRoutingConfig, FusionRoutingConfig.model_validate(data))
        _validate_pool_ordering(config)
        return config

    def main_router(self) -> FusionRouter:
        return FusionRouter.from_config(self.main_routing(), "main")

    def sidekick_router(self) -> FusionRouter:
        return FusionRouter.from_config(self.main_routing(), "sidekick")

    def default_budget(self) -> FusionRunBudget:
        raw_budget = self._load().get("budget") or {}
        payload = dict(raw_budget) if isinstance(raw_budget, dict) else {}
        return cast(FusionRunBudget, FusionRunBudget.model_validate(payload))

    def tool_options(self) -> FusionToolOptions:
        raw_tools = self._load().get("tool_options") or self._load().get("tools") or {}
        payload = dict(raw_tools) if isinstance(raw_tools, dict) else {}
        return cast(FusionToolOptions, FusionToolOptions.model_validate(payload))
