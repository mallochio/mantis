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
