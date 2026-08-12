"""Global test safety: block all real outbound provider traffic.

Tests run with the developer's real API keys present in the process
environment, so an unmocked code path can silently make real, billed calls.
These autouse guards make any missed mock fail fast and visibly instead:

* Any POST via the shared provider client raises ``httpx.NetworkError``,
  which the failover loop deliberately does not catch, so the failure is loud.
* Any direct ``httpx.get`` (e.g. the price map) raises a connection error
  that the fetch logic already treats as "no data".

Individual tests that explicitly monkeypatch ``serve._provider_client`` or
``serve.httpx.get`` replace these guards as usual.
"""

from __future__ import annotations

import os

import httpx
import pytest

# Interactive shells export endpoint/catalog overrides (~/.zshrc sets
# OPENROUTER_BASE_URL/OPENCODE_GO_ENDPOINT_URL to gateway URLs, exports
# AI_ROUTING_CONFIG and rendered catalog bindings).  Serve captures some
# of these at module import, so scrub them before any test module imports
# them; tests that need a value set it explicitly via monkeypatch.
for _name in (
    "AI_ROUTING_CONFIG",
    "MANTIS_CATALOG_PATH",
    "MANTIS_ENDPOINT_PROFILE",
    "MANTIS_PROVIDER_BINDINGS",
    "MANTIS_WORKER_BINDINGS",
    "MANTIS_IDENTITY_CONTRACT",
    "MANTIS_PROVIDER_KEYS",
    "MANTIS_WORKER_MODELS",
    "MANTIS_CONDUCTOR_MODEL",
    "OPENROUTER_BASE_URL",
    "OPENCODE_GO_ENDPOINT_URL",
    "CLOUDFLARE_GATEWAY_BASE_URL",
    "MANTIS_GATEWAY_URL",
    "MANTIS_GATEWAY_OPENCODE_URL",
    "AI_GATEWAY_API_KEY",
):
    os.environ.pop(_name, None)


@pytest.fixture(autouse=True)
def _block_external_provider_calls(monkeypatch: pytest.MonkeyPatch):
    import serve

    # Keep the dev shell's MANTIS_LEARNING=1 from polluting the production
    # learning file with test runs (pool=test-worker etc.). Tests that need
    # learning set MANTIS_LEARNING=1 themselves.
    monkeypatch.setenv("MANTIS_LEARNING", "0")

    # Scrub ambient endpoint/catalog overrides exported by an interactive
    # shell (~/.zshrc exports OPENROUTER_BASE_URL/OPENCODE_GO_ENDPOINT_URL
    # gateway URLs, AI_ROUTING_CONFIG, rendered catalog bindings, ...).
    # Serve code reads os.environ directly, so tests must not inherit
    # the host environment; tests that need these set them via monkeypatch.
    for name in (
        "AI_ROUTING_CONFIG",
        "MANTIS_CATALOG_PATH",
        "MANTIS_ENDPOINT_PROFILE",
        "MANTIS_PROVIDER_BINDINGS",
        "MANTIS_WORKER_BINDINGS",
        "MANTIS_IDENTITY_CONTRACT",
        "MANTIS_PROVIDER_KEYS",
        "MANTIS_WORKER_MODELS",
        "MANTIS_CONDUCTOR_MODEL",
        "OPENROUTER_BASE_URL",
        "OPENCODE_GO_ENDPOINT_URL",
        "CLOUDFLARE_GATEWAY_BASE_URL",
        "MANTIS_GATEWAY_URL",
        "MANTIS_GATEWAY_OPENCODE_URL",
        "AI_GATEWAY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def blocked_post(*_args, **_kwargs):
        raise httpx.NetworkError(
            "Real provider calls are blocked in tests. "
            "Monkeypatch serve._provider_client or serve._model_completion. "
            "See tests/conftest.py."
        )

    def blocked_stream(*_args, **_kwargs):
        raise httpx.NetworkError(
            "Real provider calls are blocked in tests. "
            "Monkeypatch serve._provider_client or serve._model_completion. "
            "See tests/conftest.py."
        )

    def blocked_get(url, *_args, **_kwargs):
        request = httpx.Request("GET", url)
        raise httpx.ConnectError("Network access is blocked in tests.", request=request)

    monkeypatch.setattr(serve._provider_client, "post", blocked_post)
    monkeypatch.setattr(serve._provider_client, "stream", blocked_stream)
    monkeypatch.setattr(serve.httpx, "get", blocked_get)
    yield
