from __future__ import annotations

import json

import fusion
import providers
import pytest
import serve_config
import utils


def test_prune_tool_result_small_content():
    content = "Hello, world!"
    assert utils.prune_tool_result(content, max_chars=100) == content


def test_prune_tool_result_large_content():
    head = "A" * 100
    middle = "M" * 5000
    tail = "Z" * 100
    content = head + middle + tail
    pruned = utils.prune_tool_result(content, max_chars=500, head_chars=100, tail_chars=100)
    assert pruned.startswith(head)
    assert pruned.endswith(tail)
    assert "[... Omitted 5000 characters of tool output for context efficiency ...]" in pruned
    assert len(pruned) < len(content)


def test_repeat_tool_guard_triggers_at_thresholds():
    guard = utils.RepeatToolGuard(thresholds=(3, 5))
    args1 = json.dumps({"command": "pytest -v", "timeout": 30})
    args2 = json.dumps({"timeout": 30, "command": "pytest -v"})  # same keys, different order

    # Turn 1
    assert guard.observe("bash", args1) is None
    assert guard.consecutive_count == 1

    # Turn 2 (canonical ordering matches)
    assert guard.observe("bash", args2) is None
    assert guard.consecutive_count == 2

    # Turn 3 -> Threshold 3 triggers advisory
    notice = guard.observe("bash", args1)
    assert notice is not None
    assert "Tool 'bash' has been called 3 times consecutively" in notice

    # Turn 4
    assert guard.observe("bash", args1) is None

    # Turn 5 -> Threshold 5 triggers advisory
    notice5 = guard.observe("bash", args1)
    assert notice5 is not None
    assert "Tool 'bash' has been called 5 times consecutively" in notice5

    # New tool resets counter
    assert guard.observe("read_file", '{"path": "foo.py"}') is None
    assert guard.consecutive_count == 1


def test_fusion_usage_counted_only_once(monkeypatch):
    calls = {"n": 0}

    def fake_provider(spec, messages, max_tokens, temperature, tools=None):
        calls["n"] += 1
        run = getattr(serve_config._history_context, "active_run", None)
        if calls["n"] == 1:
            content = "PLAN: p\nBRIEF: b"
        elif calls["n"] == 2:
            content = "Done"
        else:
            content = "ACCEPT"
        data = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        if run is not None:
            run.add_usage(data["usage"], model=spec)
        return data

    monkeypatch.setattr(providers, "_provider_response", fake_provider)
    run = fusion.create_fusion_run("brief test")
    event = fusion.advance_fusion_run(run.run_id)
    # 3 calls: planning, sidekick execution, review.
    # Total tokens should be exactly 3 * 15 = 45, not 90.
    assert event["usage"]["total_tokens"] == 45
    assert event["usage"]["prompt_tokens"] == 30
    assert event["usage"]["completion_tokens"] == 15


def test_fusion_strict_review_parsing():
    run = fusion.FusionRun("test-parse", "brief")

    # Positive acceptances
    assert run._parse_main_review("ACCEPT")[0] is True
    assert run._parse_main_review("ACCEPT: all good")[0] is True
    assert run._parse_main_review("accept")[0] is True

    # Follow-ups
    accepted, fb = run._parse_main_review("FOLLOW_UP: fix typo in README")
    assert accepted is False
    assert fb == "fix typo in README"

    # Rejection
    accepted, fb = run._parse_main_review("REJECT: tests failed")
    assert accepted is False

    # Ambiguous / malformed output raises ValueError
    with pytest.raises(ValueError, match="did not output a valid decision"):
        run._parse_main_review("I am not sure what to do next.")


def test_fusion_tool_result_validation_and_ordering():
    run = fusion.FusionRun("test-val", "brief")
    run.pending_tool_calls = [
        {"id": "call_1", "function": {"name": "read"}},
        {"id": "call_2", "function": {"name": "write"}},
    ]
    # Out of order results
    results = [
        {"tool_call_id": "call_2", "content": "wrote"},
        {"tool_call_id": "call_1", "content": "read"},
    ]
    validated = run._validate_tool_results(results)
    # Must be sorted to match pending_tool_calls order
    assert [r["tool_call_id"] for r in validated] == ["call_1", "call_2"]

    # Unknown ID raises error
    with pytest.raises(ValueError, match="mismatch"):
        run._validate_tool_results([{"tool_call_id": "call_unknown", "content": "bad"}])


def test_fusion_planning_tool_budget_cap(monkeypatch):
    calls = []
    def mock_worker(slot, messages, max_tokens=4096, temperature=0.7, worker_tools=None):
        calls.append((slot, messages, worker_tools))
        first_content = messages[0].get("content", "")
        if first_content == fusion.MAIN_PREAMBLE:
            # Main attempts to call tool again
            if worker_tools:
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": f"call_{len(calls)}",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": "{}"},
                                }
                            ],
                        }
                    }],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                }
            # When tools stripped after 2 rounds, emit plan
            return {
                "choices": [{"message": {"role": "assistant", "content": "PLAN: p\nBRIEF: b"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": "ACCEPT"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    tools = [{"type": "function", "function": {"name": "read"}}]
    monkeypatch.setattr(providers, "_provider_response", mock_worker)
    run = fusion.create_fusion_run("brief", tools=tools)
    run.main_tools_policy = frozenset({"plan"})

    # Round 1
    ev1 = fusion.advance_fusion_run(run.run_id)
    assert ev1["status"] == "awaiting_tools"

    # Round 2
    ev2 = fusion.advance_fusion_run(
        run.run_id, tool_results=[{"tool_call_id": "call_1", "content": "res1"}]
    )
    assert ev2["status"] == "awaiting_tools"

    # Round 3 -> Tools capped, main must emit plan
    fusion.advance_fusion_run(
        run.run_id, tool_results=[{"tool_call_id": "call_2", "content": "res2"}]
    )
    assert run.planning_tool_rounds == 2
    assert run.plan == "p"

