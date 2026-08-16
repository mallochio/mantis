import json
import os
import socket

import pytest


def _deploy_fixture_targets() -> None:
    """Boot the router from an explicit fixture instead of the dev-shell catalog.

    A catalog normally exists at the default path, so a plain ``import server``
    would load the developer's real targets.  Point the server at a fixture
    source; catalog tests spawn their own subprocess fixtures and are
    unaffected.
    """
    os.environ.pop("MANTIS_ROUTER_SESSION_FROM_USER", None)
    os.environ.pop("AI_ROUTING_CONFIG", None)
    os.environ["MANTIS_ROUTER_KEY"] = "sk-route-local"  # force: real key may be exported in dev shells
    os.environ["MANTIS_ROUTER_TEST_CRED"] = "test-only"
    os.environ["MANTIS_ROUTER_TARGETS_JSON"] = json.dumps({
        "version": 1,
        "providers": {
            "zen": {
                "adapter": "openai-compatible",
                "base_url": "https://opencode.ai/zen/go/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions"],
            },
            "openrouter": {
                "adapter": "openai-compatible",
                "base_url": "https://openrouter.ai/api/v1",
                "credential_env": "MANTIS_ROUTER_TEST_CRED",
                "protocols": ["chat_completions", "responses"],
            },
        },
        "targets": {
            "cheap": {
                "provider": "zen", "upstream_model": "deepseek-v4-flash", "rank": 0,
                "protocols": ["chat_completions"], "fallbacks": ["middle"],
            },
            "middle": {
                "provider": "openrouter", "upstream_model": "openai/gpt-5.6-terra",
                "reasoning_effort": "max", "force_reasoning_effort": True, "rank": 1,
                "protocols": ["chat_completions", "responses"], "fallbacks": ["expensive"],
            },
            "expensive": {
                "provider": "openrouter", "upstream_model": "openai/gpt-5.6-sol",
                "reasoning_effort": "medium", "rank": 2,
                "protocols": ["chat_completions", "responses"], "fallbacks": ["middle"],
            },
        },
        "complexity_targets": ["cheap", "cheap", "middle", "middle", "expensive"],
    })
    try:
        import server  # noqa: F401
    except ImportError:
        return


_deploy_fixture_targets()


@pytest.fixture(autouse=True)
def sanitize_router_env(monkeypatch):
    monkeypatch.delenv("MANTIS_ROUTER_SESSION_FROM_USER", raising=False)
    monkeypatch.setenv("MANTIS_ROUTER_KEY", os.environ.get("MANTIS_ROUTER_KEY", "sk-route-local"))
    monkeypatch.setenv("MANTIS_ROUTER_TEST_CRED", os.environ.get("MANTIS_ROUTER_TEST_CRED", "test-only"))


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
