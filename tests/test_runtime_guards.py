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
    args2 = json.dumps({"timeout": 30, "command": "pytest -v"})

    assert guard.observe("bash", args1) is None
    assert guard.consecutive_count == 1

    assert guard.observe("bash", args2) is None
    assert guard.consecutive_count == 2

    notice = guard.observe("bash", args1)
    assert notice is not None
    assert "Tool 'bash' has been called 3 times consecutively" in notice

    assert guard.observe("bash", args1) is None

    notice5 = guard.observe("bash", args1)
    assert notice5 is not None
    assert "Tool 'bash' has been called 5 times consecutively" in notice5

    assert guard.observe("read_file", '{"path": "foo.py"}') is None
    assert guard.consecutive_count == 1


def test_fusion_usage_counted_only_once(monkeypatch):
    calls = {"n": 0}

    def fake_provider(spec, messages, max_tokens, temperature, tools=None):
        calls["n"] += 1
        run = getattr(serve_config._history_context, "active_run", None)
        if calls["n"] == 1:
            content = "done"
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
    assert event["usage"]["total_tokens"] == 15
    assert event["usage"]["prompt_tokens"] == 10
    assert event["usage"]["completion_tokens"] == 5


def test_fusion_tool_result_validation_and_ordering():
    run = fusion.FusionRun("test-val", "brief")
    run.pending_tool_calls = [
        {"id": "call_1", "function": {"name": "read"}},
        {"id": "call_2", "function": {"name": "write"}},
    ]
    results = [
        {"tool_call_id": "call_2", "content": "wrote"},
        {"tool_call_id": "call_1", "content": "read"},
    ]
    validated = run._validate_tool_results(results)
    assert [r["tool_call_id"] for r in validated] == ["call_1", "call_2"]

    with pytest.raises(ValueError, match="mismatch"):
        run._validate_tool_results([{"tool_call_id": "call_unknown", "content": "bad"}])
