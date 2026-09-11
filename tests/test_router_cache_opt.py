"""Focused router-cache tests: union markers, stickiness, session fallback."""

from __future__ import annotations

from pathlib import Path

import base_proxy
import providers


def _write_catalog(
    path: Path,
    eff_provider: str,
    eff_adapter: str,
    eff_model: str,
    cap_provider: str,
    cap_adapter: str,
    cap_model: str,
) -> Path:
    catalog = path / "catalog.toml"
    catalog.write_text(
        "version = 1\n"
        f'[providers."{eff_provider}"]\n'
        f'adapter = "{eff_adapter}"\n'
        'base_url = "https://example.test/v1"\n'
        'credential_env = "TEST_KEY_A"\n'
        'protocols = ["chat_completions"]\n'
        f'[providers."{cap_provider}"]\n'
        f'adapter = "{cap_adapter}"\n'
        'base_url = "https://example.test/v1"\n'
        'credential_env = "TEST_KEY_B"\n'
        'protocols = ["chat_completions"]\n'
        "[base]\n"
        'revision = "cache-opt"\n'
        'picker = "efficient_first"\n'
        "confidence_threshold = 0.5\n"
        "recent_turn_window = 3\n"
        "[base.targets.efficient]\n"
        f'provider = "{eff_provider}"\n'
        'reasoning_effort = "high"\n'
        "max_tokens = 64000\n"
        f'upstream_model = "{eff_model}"\n'
        "[base.targets.capable]\n"
        f'provider = "{cap_provider}"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 128000\n"
        f'upstream_model = "{cap_model}"\n'
    )
    return catalog


def _use_catalog(monkeypatch, catalog: Path) -> None:
    monkeypatch.setenv("MANTIS_CATALOG_PATH", str(catalog))
    monkeypatch.delenv("AI_ROUTING_CONFIG", raising=False)
    base_proxy._load_base_route.cache_clear()


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]


def test_mixed_kimi_gpt_still_marks_openai(tmp_path, monkeypatch):
    catalog = _write_catalog(
        tmp_path,
        "modal-kimi",
        "modal",
        "kimi-k3",
        "openrouter",
        "openrouter",
        "openai/gpt-5.6-sol",
    )
    _use_catalog(monkeypatch, catalog)
    assert base_proxy._base_route_families() == frozenset({"openai"})
    body = {"messages": _messages()}
    out = base_proxy._apply_base_cache_markers(body)
    assert out["messages"][0].get("prompt_cache_breakpoint") == {"mode": "explicit"}


def test_mixed_claude_gpt_marks_both(tmp_path, monkeypatch):
    catalog = _write_catalog(
        tmp_path,
        "openrouter",
        "openrouter",
        "anthropic/claude-sonnet-5",
        "openrouter-cap",
        "openrouter",
        "openai/gpt-5.6-sol",
    )
    _use_catalog(monkeypatch, catalog)
    assert base_proxy._base_route_families() == frozenset({"anthropic", "openai"})
    body = {"messages": _messages()}
    out = base_proxy._apply_base_cache_markers(body)
    assert out["messages"][0].get("prompt_cache_breakpoint") == {"mode": "explicit"}
    assert "cache_control" in str(out["messages"][0]["content"])


def test_master_switch_disables_base_markers(tmp_path, monkeypatch):
    catalog = _write_catalog(
        tmp_path,
        "modal-kimi",
        "modal",
        "kimi-k3",
        "openrouter",
        "openrouter",
        "openai/gpt-5.6-sol",
    )
    _use_catalog(monkeypatch, catalog)
    monkeypatch.setenv("MANTIS_CACHE_BREAKPOINTS", "0")
    body = {"messages": _messages()}
    out = base_proxy._apply_base_cache_markers(body)
    assert "prompt_cache_breakpoint" not in str(out["messages"])


def test_litellm_adds_openai_breakpoints(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    resolved = providers.ResolvedModelSpec(
        adapter="openai-compatible",
        model="openai/gpt-5.6-sol",
        effort=None,
        base_url="http://127.0.0.1:8080/v1",
        credential_env="LITELLM_API_KEY",
        binding=None,
        protocols=("chat_completions",),
        slot=None,
    )
    kwargs = providers._litellm_kwargs(
        resolved,
        _messages(),
        100,
        0.7,
        None,
        None,
        None,
        {},
    )
    assert kwargs["messages"][0].get("prompt_cache_breakpoint") == {"mode": "explicit"}


def test_litellm_preserves_client_openai_breakpoint(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    resolved = providers.ResolvedModelSpec(
        adapter="openai-compatible",
        model="openai/gpt-5.6-sol",
        effort=None,
        base_url="http://127.0.0.1:8080/v1",
        credential_env="LITELLM_API_KEY",
        binding=None,
        protocols=("chat_completions",),
        slot=None,
    )
    messages = _messages()
    messages[0]["prompt_cache_breakpoint"] = {"mode": "explicit"}
    kwargs = providers._litellm_kwargs(
        resolved,
        messages,
        100,
        0.7,
        None,
        None,
        None,
        {},
    )
    assert kwargs["messages"][0].get("prompt_cache_breakpoint") == {"mode": "explicit"}


def test_shared_harness_header_scopes_sessions_per_conversation():
    """One shared header must not merge two conversations into one session."""
    key_a = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"},
        type(
            "R",
            (),
            {"messages": [{"role": "user", "content": "prove theorem"}], "tools": None},
        ),
    )
    key_b = base_proxy.session_id(
        {"x-mantis-session-id": "opencode"},
        type(
            "R",
            (),
            {"messages": [{"role": "user", "content": "what time is it"}], "tools": None},
        ),
    )
    assert key_a is not None and key_b is not None
    assert key_a != key_b


def test_synthetic_session_is_stable():
    messages = [{"role": "user", "content": "fix failing test"}]
    first = base_proxy._synthetic_session_id(messages)
    second = base_proxy._synthetic_session_id([dict(messages[0])])
    assert first and first == second
    other = base_proxy._synthetic_session_id([{"role": "user", "content": "other task"}])
    assert other != first


def test_session_id_falls_back_to_synthetic():
    body = type(
        "R",
        (),
        {
            "metadata": None,
            "user": None,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": None,
        },
    )()
    assert base_proxy.session_id({}, body).startswith("auto-")


def test_session_id_accepts_mantis_header():
    body = type("R", (), {"metadata": None, "user": None, "messages": [], "tools": None})()
    assert base_proxy.session_id({"x-mantis-session-id": "opencode"}, body) == "opencode"


def test_router_headers_never_force_a_tier(monkeypatch):
    """Tier decisions belong to Switchyard; the forwarder sends no directives."""
    body = type(
        "R",
        (),
        {
            "metadata": {"session_id": "s-9"},
            "user": None,
            "messages": [
                {"role": "user", "content": "fix"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"cmd": "pytest -q"}'}},
                    ],
                },
                {"role": "tool", "content": "Traceback: AssertionError"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"cmd": "pytest -q"}'}},
                    ],
                },
                {"role": "tool", "content": "Traceback: AssertionError"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": '{"cmd": "pytest -q"}'}},
                    ],
                },
                {"role": "tool", "content": "Traceback: AssertionError"},
            ],
        },
    )()
    headers = base_proxy.router_headers({}, body)
    assert headers[base_proxy.SWITCHYARD_SESSION_HEADER] == "s-9"
    assert "x-switchyard-force-tier" not in headers
    assert "x-switchyard-escalated" not in headers


def test_trinity_reminder_uses_user_role():
    import runs

    run = runs.TrinityRun("remind-1", [{"role": "user", "content": "hi"}], [], ["w"])
    run._pending = {
        "role": "Worker",
        "messages": [{"role": "user", "content": "hi"}],
        "asst": {
            "role": "assistant",
            "content": "t",
            "tool_calls": [
                {"id": "c0", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            ],
        },
    }
    run._expected_ids = {"c0"}
    run.repeat_guard.observe = lambda *args: "slow down"
    run._apply_tool_results([{"tool_call_id": "c0", "content": "out"}])
    last = run._pending["messages"][-1]
    assert last["role"] == "user"
    assert "<system-reminder>" in str(last["content"])
