import importlib
import os
import socket
import tempfile

import pytest


def _force_legacy_router_default() -> None:
    """Force legacy env-based routing tests to run in legacy mode.

    A catalog now normally exists at the default path, so a plain
    ``import server`` loads the catalog instead of the legacy environment.
    Point ``AI_ROUTING_CONFIG`` at an empty catalog (no ``[routellm]``
    section) so ``import server`` takes the legacy fallback.  Catalog tests
    reload ``server`` with their own fixtures and are unaffected.
    """
    os.environ.pop("ROUTELLM_SESSION_FROM_USER", None)
    os.environ["ROUTELLM_KEY"] = "sk-route-local"  # force: real key may be exported in dev shells
    try:
        import server  # noqa: F401
    except ImportError:
        return
    fd, path = tempfile.mkstemp(suffix=".toml")
    try:
        os.write(fd, b"version = 1\n")
    finally:
        os.close(fd)
    os.environ["AI_ROUTING_CONFIG"] = path
    importlib.reload(server)


_force_legacy_router_default()


@pytest.fixture(autouse=True)
def sanitize_router_env(monkeypatch):
    monkeypatch.delenv("ROUTELLM_SESSION_FROM_USER", raising=False)
    monkeypatch.setenv("ROUTELLM_KEY", os.environ.get("ROUTELLM_KEY", "sk-route-local"))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("real network access is forbidden in tests")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

@pytest.fixture
def anyio_backend():
    return "asyncio"

@pytest.fixture(autouse=True)
def reset_router_state():
    try:
        import server
    except ImportError:
        yield
        return
    server._resp_cache.clear()
    server._inflight.clear()
    server._decision_store.clear()
    server._session_state.clear()
    server._recent_prompts.clear()
    clear_cache = getattr(server._decide_cached, "cache_clear", None)
    if clear_cache:
        clear_cache()
    server._cache_bytes = 0
    for key in server._cache_metrics:
        server._cache_metrics[key] = 0
    yield
    server._resp_cache.clear()
    server._inflight.clear()
    server._decision_store.clear()
    server._session_state.clear()
    server._recent_prompts.clear()
    clear_cache = getattr(server._decide_cached, "cache_clear", None)
    if clear_cache:
        clear_cache()
    server._cache_bytes = 0
