"""LiteLLM multi-turn: provider mix, tool loop, cache/cost via mocked litellm."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import model_catalog
import providers
import serve_config


def _setup_catalog(monkeypatch):
    cat = model_catalog.load_mantis_catalog("config/catalog.toml")
    env = model_catalog.render_mantis_environment(cat)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    keys = dict.fromkeys(cat.bindings.providers, "test-key")
    monkeypatch.setenv("MANTIS_PROVIDER_KEYS", json.dumps(keys))
    for binding in cat.bindings.providers.values():
        monkeypatch.setenv(binding.credential_env, "test-key")
    return cat


def _mock_resp(content: str | None, tool_calls=None, cached=0, cost=0.0001):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.reasoning_content = None
    msg.reasoning_details = None
    msg.annotations = None
    choice = MagicMock(message=msg, finish_reason="tool_calls" if tool_calls else "stop")
    usage = MagicMock(
        prompt_tokens=20, completion_tokens=10, total_tokens=30,
        prompt_tokens_details=MagicMock(cached_tokens=cached),
    )
    resp = MagicMock(choices=[choice], usage=usage, _hidden_params={"response_cost": cost})
    resp.choices = [choice]
    resp.usage = usage
    resp._hidden_params = {"response_cost": cost}
    return resp


def test_provider_mix_opencode_go_and_openrouter(monkeypatch):
    _setup_catalog(monkeypatch)
    deepseek = providers._resolve_model_spec("deepseek-v4-flash")
    assert deepseek.binding == "opencode-go"
    assert "opencode.ai" in deepseek.base_url
    luna = providers._resolve_model_spec("gpt-5_6-luna")
    assert luna.binding == "openrouter"
    assert "openrouter.ai" in luna.base_url


def test_litellm_kwargs_per_model_thinking(monkeypatch):
    _setup_catalog(monkeypatch)
    for slot in ("deepseek-v4-flash", "gpt-5_6-luna", "claude-opus-5"):
        resolved = providers._resolve_model_spec(slot)
        kwargs = providers._litellm_kwargs(
            resolved, [{"role": "user", "content": "hi"}], 100, 0.7, None, None, None, {},
        )
        assert "model" in kwargs and "messages" in kwargs
        assert "api_key" in kwargs


def test_multiturn_tool_loop_via_litellm(monkeypatch):
    _setup_catalog(monkeypatch)
    calls: list[str] = []

    def fake_completion(**kwargs):
        calls.append(kwargs.get("model", ""))
        if len(calls) == 1:
            func = MagicMock(name="bash", arguments='{"command": "echo hi"}')
            tc = [MagicMock(id="call_0", type="function", function=func)]
            return _mock_resp("", tool_calls=tc)
        return _mock_resp("Final answer: done")

    monkeypatch.setattr(providers, "_litellm_completion", fake_completion)
    run = SimpleNamespace(
        usage_models={}, capture_metadata=False,
        active_tool_choice=None, active_response_format=None, active_controls={},
        record_activity=lambda *a, **k: None,
        add_usage=lambda usage, model=None: run.usage_models.update(
            {model: {"prompt_tokens": 20, "completion_tokens": 10, "cached_tokens": 0}}
        ),
    )
    serve_config._history_context.active_run = run
    try:
        data = providers._provider_response(
            "deepseek-v4-flash", [{"role": "user", "content": "do it"}], 100, 0.7,
        )
        assert data["choices"][0]["message"].get("tool_calls")
        data2 = providers._provider_response(
            "gpt-5_6-luna",
            [
                {"role": "user", "content": "do it"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_0", "function": {"name": "bash", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_0", "content": "hi"},
            ],
            100, 0.7,
        )
        assert data2["choices"][0]["message"]["content"] == "Final answer: done"
        assert len(calls) == 2
    finally:
        serve_config._history_context.active_run = None


def test_azure_foundry_router_litellm_kwargs(monkeypatch):
    _setup_catalog(monkeypatch)
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    resolved = providers.ResolvedModelSpec(
        adapter="azure_ai",
        model="model-router",
        effort="medium",
        base_url="https://example.invalid/api/projects/example/openai/v1",
        credential_env="AZURE_API_KEY",
        binding=None,
        protocols=("responses",),
        slot=None,
        max_tokens=128000,
    )
    assert providers._litellm_model(resolved) == "azure_ai/model_router/model-router"
    kwargs = providers._litellm_kwargs(
        resolved, [{"role": "user", "content": "hi"}], 100, 0.7, None, None, None, {},
    )
    assert kwargs["model"] == "azure_ai/model_router/model-router"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["api_base"] == resolved.base_url
    assert kwargs["reasoning"] == {"effort": "medium"}


def test_cache_and_cost_via_litellm(monkeypatch):
    _setup_catalog(monkeypatch)
    monkeypatch.setattr(
        providers, "_litellm_completion", lambda **k: _mock_resp("hi", cached=15, cost=0.0002)
    )
    run = SimpleNamespace(
        usage_models={}, capture_metadata=False,
        active_tool_choice=None, active_response_format=None, active_controls={},
        record_activity=lambda *a, **k: None,
        add_usage=lambda usage, model=None: run.usage_models.update(
            {model: {"prompt_tokens": 20, "completion_tokens": 10, "cached_tokens": 15}}
        ),
    )
    serve_config._history_context.active_run = run
    try:
        data = providers._provider_response(
            "deepseek-v4-flash",
            [{"role": "user", "content": "hi"}],
            100,
            0.7,
        )
        assert data["usage"]["prompt_tokens"] == 20
        # cost breakdown reads litellm stash
        bd = providers._cost_breakdown(run.usage_models)
        assert bd["known"] is True
        assert bd["cache_hit_ratio"] > 0
    finally:
        serve_config._history_context.active_run = None
