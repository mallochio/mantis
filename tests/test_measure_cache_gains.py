"""Cover measurement helpers and lock in cache gains."""

from __future__ import annotations

import measure_cache_gains as gains


def test_azure_isolation_saves_tokens():
    result = gains.simulate_azure_interleaved(turns=4)
    assert result["new_tokens_sent"] < result["old_tokens_sent"]
    assert result["old_tokens_sent"] - result["new_tokens_sent"] > 0


def test_base_isolation_scopes_sessions_per_conversation():
    result = gains.simulate_base_session_isolation()
    assert result["conversation_a"] != result["conversation_b"]


def test_reasoning_trim_saves_tokens():
    result = gains.simulate_reasoning_trim(assistant_turns=4)
    assert result["after_tokens"] < result["before_tokens"]


def test_gateway_prefix_scenarios_favor_stable_independent_lanes():
    result = gains.simulate_gateway_prefixes()
    assert result["stable_session_reused_tokens"] > 0
    assert result["main_lane_reused_tokens"] > 0
    assert result["sidekick_lane_reused_tokens"] > 0
    assert result["compaction_reused_tokens"] > 0
    assert result["stable_session_reused_tokens"] > result["mutable_prefix_reused_tokens"]
