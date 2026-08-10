import socket

import pytest


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
