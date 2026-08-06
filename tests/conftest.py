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

import httpx
import pytest


@pytest.fixture(autouse=True)
def _block_external_provider_calls(monkeypatch: pytest.MonkeyPatch):
    import serve

    def blocked_post(*_args, **_kwargs):
        raise httpx.NetworkError(
            "Real provider calls are blocked in tests. "
            "Monkeypatch serve._provider_client or serve._model_completion. "
            "See tests/conftest.py."
        )

    def blocked_get(url, *_args, **_kwargs):
        request = httpx.Request("GET", url)
        raise httpx.ConnectError("Network access is blocked in tests.", request=request)

    monkeypatch.setattr(serve._provider_client, "post", blocked_post)
    monkeypatch.setattr(serve.httpx, "get", blocked_get)
    yield
