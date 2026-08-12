"""mini-SWE-agent local-model configuration without provider side effects."""

from __future__ import annotations

from typing import Any

MANTIS_MODEL_REGISTRY = {
    "mantis": {"litellm_provider": "openai", "mode": "chat"},
    "mantis-trinity": {"litellm_provider": "openai", "mode": "chat"},
}


def build_local_model_config(api_base: str, model: str = "mantis") -> dict[str, Any]:
    """Return the Local models configuration consumed by mini-SWE-agent."""
    if model not in MANTIS_MODEL_REGISTRY:
        raise ValueError(f"unknown Mantis model: {model}")
    return {
        "model": model,
        "model_kwargs": {
            "custom_llm_provider": "openai",
            "api_base": api_base,
        },
        "model_registry": {model: MANTIS_MODEL_REGISTRY[model]},
    }
