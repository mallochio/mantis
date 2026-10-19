"""Request plans for routed A/B variants."""

from __future__ import annotations

from pathlib import Path

import pytest
import run_routed_ab as runner
from routed_ab_tools import OPENROUTER_BASE_URL


def test_request_plan_maps_every_variant():
    catalog = runner.load_catalog(runner.DEFAULT_CATALOG)
    efficient = runner.request_plan("efficient-only", catalog)
    capable = runner.request_plan("capable-only", catalog)
    assert efficient["model"] == "~openai/gpt-sol-latest"
    assert efficient["base_url"] == OPENROUTER_BASE_URL
    assert efficient["reasoning_effort"] == "medium"
    assert capable["model"] == "~openai/gpt-sol-latest"
    assert capable["reasoning_effort"] == "high"
    assert runner.request_plan("mantis-base", catalog) == {"model": "mantis/base"}


def test_request_plan_maps_tool_delegation_variant():
    catalog = runner.load_catalog(runner.DEFAULT_CATALOG)
    workspace = Path.cwd() / "fusion-arm-test"
    tool = runner.request_plan("fusion-tool", catalog, workspace)
    assert "worker_profiles" not in tool
    assert "delegation" not in tool["tool_options"]
    assert tool["tool_options"]["workspace"] == str(workspace.resolve())
    assert tool["tool_options"]["server_execution"] is True
    with pytest.raises(ValueError):
        runner.request_plan("fusion-tool", catalog)
    for variant in ("fusion-pinned", "fusion-routed", "unknown-variant"):
        with pytest.raises(ValueError):
            runner.request_plan(variant, catalog, workspace)
