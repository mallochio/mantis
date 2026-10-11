"""Compatibility facade for routed A/B evaluation helpers."""

from __future__ import annotations

from ab_drive import drive_case
from ab_execution import run_variant
from ab_records import (
    auto_judge,
    base_target_spec,
    build_record,
    load_catalog,
    load_prices,
    request_plan,
    seed_workspace,
    selected_models,
    usage_numbers,
)

__all__ = [
    "auto_judge",
    "base_target_spec",
    "build_record",
    "drive_case",
    "load_catalog",
    "load_prices",
    "request_plan",
    "run_variant",
    "seed_workspace",
    "selected_models",
    "usage_numbers",
]
