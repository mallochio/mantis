"""Cover measurement helpers and lock in cache gains."""

from __future__ import annotations

import measure_cache_gains as gains


def test_azure_isolation_saves_tokens():
    result = gains.simulate_azure_interleaved(turns=4)
    assert result["new_tokens_sent"] < result["old_tokens_sent"]
    assert result["old_tokens_sent"] - result["new_tokens_sent"] > 0


def test_base_isolation_avoids_tier_pollution():
    result = gains.simulate_base_tier_pollution()
    assert result["shared_simple_tier"] == "reasoning"
    assert result["isolated_simple_tier"] == "simple"


def test_reasoning_trim_saves_tokens():
    result = gains.simulate_reasoning_trim(assistant_turns=4)
    assert result["after_tokens"] < result["before_tokens"]
