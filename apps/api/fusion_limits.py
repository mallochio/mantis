"""Resolve Fusion limits from exact provider model metadata."""

from dataclasses import dataclass
from functools import cache
from typing import Any

import httpx
import litellm
import providers


class ModelLimitsError(ValueError):
    """Report missing authoritative model limits."""


@dataclass(frozen=True)
class ModelLimits:
    """Hold a model's context and output token limits."""

    context_window: int
    max_output_tokens: int


def _positive(value: Any) -> int | None:
    """Accept positive integer metadata values."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@cache
def _openrouter_models(base_url: str) -> dict[str, dict[str, Any]]:
    """Fetch an exact-id provider catalog once per process."""
    response = httpx.get(f"{base_url.rstrip('/')}/models", timeout=5.0)
    response.raise_for_status()
    return {entry["id"]: entry for entry in response.json()["data"]}


def _metadata(resolved: providers.ResolvedModelSpec) -> tuple[int | None, int | None]:
    """Read provider-owned context and output capacities."""
    if resolved.adapter == "openrouter":
        info = _openrouter_models(resolved.base_url).get(resolved.model, {})
        top = info.get("top_provider") or {}
        return _positive(info.get("context_length")), _positive(top.get("max_completion_tokens"))
    info = litellm.get_model_info(providers._litellm_model(resolved))
    return _positive(info.get("max_input_tokens")), _positive(info.get("max_output_tokens"))


def _cap(declared: int | None, discovered: int | None) -> int | None:
    """Apply a declared limit without increasing provider capacity."""
    values = [value for value in (declared, discovered) if value is not None]
    return min(values) if values else None


@cache
def _resolved_limits(resolved: providers.ResolvedModelSpec) -> ModelLimits:
    """Cache trustworthy limits for a resolved provider identity."""
    context = _positive(resolved.context_window)
    output = _positive(resolved.max_tokens)
    try:
        discovered_context, discovered_output = _metadata(resolved)
    except Exception as error:
        if context is None or output is None:
            raise ModelLimitsError(
                f"Cannot discover limits for {resolved.model}; configure worker "
                "context_window and max_tokens or restore provider metadata access"
            ) from error
        discovered_context, discovered_output = None, None
    context, output = _cap(context, discovered_context), _cap(output, discovered_output)
    if context is None or output is None:
        raise ModelLimitsError(
            f"Missing limits for {resolved.model}; configure worker context_window and max_tokens"
        )
    return ModelLimits(context, output)


def fusion_model_descriptor(config: Any) -> dict[str, int]:
    """Advertise shared Fusion capacity with optional configured caps."""
    lead = resolve_model_limits(config.main_slot())
    sidekick = resolve_model_limits(config.sidekick_slot())
    context = min(lead.context_window, sidekick.context_window)
    output = min(lead.max_output_tokens, sidekick.max_output_tokens)
    return {
        "context_length": min(context, config.context_window() or context),
        "max_completion_tokens": min(output, config.max_output_tokens() or output),
    }


def resolve_model_limits(spec: str) -> ModelLimits:
    """Resolve limits without changing the requested model identity."""
    try:
        resolved = providers._resolve_model_spec(spec)
    except ValueError:
        return _bare_limits(spec)
    return _resolved_limits(resolved)


@cache
def _bare_limits(spec: str) -> ModelLimits:
    """Resolve limits for a bare model name through LiteLLM metadata."""
    name = spec.rsplit("|", 1)[0].strip()
    try:
        info = litellm.get_model_info(name)
    except Exception as error:
        raise ModelLimitsError(
            f"Cannot discover limits for {name}; configure worker context_window and max_tokens"
        ) from error
    context, output = _positive(info.get("max_input_tokens")), _positive(
        info.get("max_output_tokens")
    )
    if context is None or output is None:
        raise ModelLimitsError(f"Missing limits for {name}; configure worker limits")
    return ModelLimits(context, output)
