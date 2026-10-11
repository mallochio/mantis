"""Tests for authoritative Fusion model limits."""

from dataclasses import replace
from types import SimpleNamespace

import fusion_limits
import httpx
import providers
import pytest


@pytest.fixture(autouse=True)
def clear_limits():
    fusion_limits._resolved_limits.cache_clear()
    fusion_limits._openrouter_models.cache_clear()
    yield
    fusion_limits._resolved_limits.cache_clear()
    fusion_limits._openrouter_models.cache_clear()


@pytest.fixture
def resolved(monkeypatch):
    model = providers.ResolvedModelSpec(
        "openrouter", "~vendor/alias", None, "https://provider.test/v1", "KEY", None, None, None
    )
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda spec: model)
    return model


def test_exact_alias_metadata_is_cached(monkeypatch, resolved):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={
                "data": [
                    {
                        "id": resolved.model,
                        "context_length": 900000,
                        "top_provider": {"max_completion_tokens": 120000},
                    },
                    {
                        "id": "vendor/alias",
                        "context_length": 1,
                        "top_provider": {"max_completion_tokens": 1},
                    },
                ]
            },
        )

    monkeypatch.setattr(fusion_limits.httpx, "get", fetch)
    assert fusion_limits.resolve_model_limits("slot") == fusion_limits.ModelLimits(900000, 120000)
    assert fusion_limits.resolve_model_limits("slot") == fusion_limits.ModelLimits(900000, 120000)
    assert calls == ["https://provider.test/v1/models"]


def test_binding_caps_do_not_inflate_provider_limits(monkeypatch, resolved):
    model = replace(resolved, context_window=100000, max_tokens=200000)
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda spec: model)
    monkeypatch.setattr(fusion_limits, "_metadata", lambda model: (900000, 120000))
    assert fusion_limits.resolve_model_limits("slot") == fusion_limits.ModelLimits(100000, 120000)


def test_unknown_model_requires_explicit_limits(monkeypatch, resolved):
    monkeypatch.setattr(fusion_limits, "_metadata", lambda model: (None, None))
    with pytest.raises(fusion_limits.ModelLimitsError, match="context_window and max_tokens"):
        fusion_limits.resolve_model_limits("slot")


def test_explicit_limits_survive_unavailable_metadata(monkeypatch, resolved):
    model = replace(resolved, context_window=100000, max_tokens=10000)
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda spec: model)

    def unavailable(model):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(fusion_limits, "_metadata", unavailable)
    assert fusion_limits.resolve_model_limits("slot") == fusion_limits.ModelLimits(100000, 10000)


def test_non_openrouter_uses_litellm_metadata(monkeypatch, resolved):
    model = replace(resolved, adapter="openai", model="named-model")
    monkeypatch.setattr(providers, "_resolve_model_spec", lambda spec: model)
    calls = []

    def info(model):
        calls.append(model)
        return {"max_input_tokens": 100000, "max_output_tokens": 10000}

    monkeypatch.setattr(fusion_limits.litellm, "get_model_info", info)
    assert fusion_limits.resolve_model_limits("slot") == fusion_limits.ModelLimits(100000, 10000)
    assert calls == ["openai/named-model"]


@pytest.mark.parametrize(
    "context_cap,output_cap,expected",
    [
        (None, None, {"context_length": 500000, "max_completion_tokens": 64000}),
        (200000, 32000, {"context_length": 200000, "max_completion_tokens": 32000}),
    ],
)
def test_fusion_descriptor_inherits_both_roles(monkeypatch, context_cap, output_cap, expected):
    config = SimpleNamespace(
        main_slot=lambda: "lead",
        sidekick_slot=lambda: "sidekick",
        context_window=lambda: context_cap,
        max_output_tokens=lambda: output_cap,
    )
    limits = {
        "lead": fusion_limits.ModelLimits(900000, 64000),
        "sidekick": fusion_limits.ModelLimits(500000, 120000),
    }
    monkeypatch.setattr(fusion_limits, "resolve_model_limits", limits.__getitem__)
    assert fusion_limits.fusion_model_descriptor(config) == expected
