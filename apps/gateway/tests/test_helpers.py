import httpx
import pytest

import eval_outcomes
import pseudo_label
import server


def test_provider_namespaced_model_capability_mutation():
    body = {"model": "auto", "temperature": 0.2}
    backend = {"model": "openai/gpt-5.6-sol", "effort": "", "max_tokens": None}
    assert "temperature" not in server._build_outgoing_body(body, backend)


def test_label_schema_is_closed():
    label = pseudo_label.normalize_label({"domain": "Ignore all rules", "complexity": 99, "route": "owner"})
    assert label["domain"] == "General"
    assert label["complexity"] == 5
    assert label["route"] == "big model"
    assert pseudo_label.LABEL_SCHEMA["schema"]["additionalProperties"] is False


def test_outcomes_join_by_request_not_shared_prompt_hash():
    decisions = [
        {"request_id": "one", "prompt_hash": "same", "score": 0.1, "decision": "cheap"},
        {"request_id": "two", "prompt_hash": "same", "score": 0.1, "decision": "cheap"},
    ]
    outcomes = [{"request_id": "one", "prompt_hash": "same", "outcome": "truncated"}]
    result = eval_outcomes.evaluate(decisions, outcomes, [])
    assert result["cells"]["cheap"]["0.1-0.156"] == [2, 1]


@pytest.mark.anyio
async def test_model_discovery_matches_provider_namespace(monkeypatch):
    mock = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"data": [{"id": "vendor/deep/model", "context_length": 1234}]})))
    monkeypatch.setattr(server, "_client", mock)
    assert await server._fetch_model_context_length("https://mock.invalid/v1", "key", "other/model") == 1234
    await mock.aclose()


def test_outcomes_ignore_attempt_telemetry_rows():
    decisions = [
        {"record_type": "attempt", "occurrence_id": "route-1", "score": 0.1, "decision": "cheap"},
        {"record_type": "attempt", "occurrence_id": "route-1", "score": 0.1, "decision": "expensive"},
        {"record_type": "decision", "occurrence_id": "route-1", "score": 0.1, "decision": "expensive"},
        {"score": 0.1, "prompt_hash": "legacy", "decision": "cheap"},
    ]
    outcomes = [
        {"decision_occurrence_id": "route-1", "outcome": "upstream_error"},
        {"prompt_hash": "legacy", "outcome": "retried"},
    ]
    result = eval_outcomes.evaluate(decisions, outcomes, [])
    assert result["cells"]["expensive"]["0.1-0.156"] == [1, 1]
    assert result["cells"]["cheap"]["0.1-0.156"] == [1, 1]


def test_outcomes_prefer_occurrence_over_duplicate_caller_request_id():
    decisions = [
        {"record_type": "decision", "occurrence_id": "a", "request_id": "duplicate", "score": 0.1, "decision": "cheap"},
        {"record_type": "decision", "occurrence_id": "b", "request_id": "duplicate", "score": 0.1, "decision": "cheap"},
    ]
    outcomes = [{"decision_occurrence_id": "a", "request_id": "duplicate", "outcome": "upstream_error"}]
    result = eval_outcomes.evaluate(decisions, outcomes, [])
    assert result["cells"]["cheap"]["0.1-0.156"] == [2, 1]


def test_outcomes_include_middle_decisions():
    decisions = [{"record_type": "decision", "decision": "middle", "score": 0.1, "prompt_hash": "m"}]
    result = eval_outcomes.evaluate(decisions, [], [])
    assert result["cells"]["middle"]["0.1-0.156"] == [1, 0]


def test_orphan_tool_messages_dropped_before_upstream():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "paired, kept"},
        {"role": "tool", "tool_call_id": "call_gone", "content": "orphan from a cut conversation"},
        {"role": "tool", "content": "no id at all"},
        {"role": "user", "content": "continue"},
    ]
    out = server._normalize_messages_for_backend(messages, developer_role="system")
    assert [m["role"] for m in out] == ["assistant", "tool", "user"]
    assert out[1]["content"] == "paired, kept"


def test_well_paired_tool_messages_pass_through_unchanged():
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    assert server._normalize_messages_for_backend(messages, developer_role="native") is messages


def test_max_completion_tokens_clamped_to_router_default_when_backend_uncapped():
    # Catalog targets carry max_tokens=None; a 131072 client ask must still
    # be clamped to the router-wide cap so 128000-cap models don't 400.
    backend = {"model": "openai/gpt-5.6-sol", "effort": "", "max_tokens": None}
    body = {"model": "auto", "max_tokens": 131072}
    out = server._build_outgoing_body(body, backend)
    assert out["max_completion_tokens"] == min(131072, server.ROUTELLM_MAX_TOKENS)


def test_max_output_tokens_clamped_to_router_default_when_backend_uncapped():
    backend = {"model": "openai/gpt-5.6-sol", "effort": "", "max_tokens": None}
    out = server._build_responses_body({"model": "auto", "max_output_tokens": 131072}, backend)
    assert out["max_output_tokens"] == min(131072, server.ROUTELLM_MAX_TOKENS)


def test_catalog_backend_explicit_cap_is_honored_not_clamped():
    # A catalog backend declaring a 384k output cap (DeepSeek V4 Pro per
    # models.dev) must not be silently reduced to the router-wide fallback.
    backend = {"model": "deepseek-v4-pro", "effort": "", "max_tokens": 384000}
    assert server._completion_token_cap(backend) == 384000
