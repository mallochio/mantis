"""Unit tests for supra-first routing, early-stop Supra scoring, and the
persistent per-prompt decision store. These tests stub the models and make no
real or billed calls."""

import json
import time

import pytest

import server


class FakeRouter:
    def __init__(self, score=0.3):
        self._score = score

    def calculate_strong_win_rate(self, prompt):
        return self._score


@pytest.fixture(autouse=True)
def isolate_store(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DECISION_STORE_PATH", tmp_path / "decision-state.jsonl")
    server._decision_store.clear()
    yield
    server._decision_store.clear()


def test_default_maps_level_3_and_4_to_middle_and_5_to_expensive(monkeypatch):
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", True)
    monkeypatch.setattr(server, "EXPENSIVE_MIN_COMPLEXITY", 5)

    monkeypatch.setattr(server, "MIDDLE_MIN_COMPLEXITY", 3)
    monkeypatch.setattr(server, "EXPENSIVE_MIN_COMPLEXITY", 5)
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "SCORE_WITH_MF", False)
    monkeypatch.setattr(server, "_supra_complexity", lambda prompt: (3, 10))
    assert server._decide_uncached("level three")[0] == "middle"
    monkeypatch.setattr(server, "_supra_complexity", lambda prompt: (4, 10))
    assert server._decide_uncached("level four")[0] == "middle"
    monkeypatch.setattr(server, "_supra_complexity", lambda prompt: (5, 10))
    assert server._decide_uncached("level five")[0] == "expensive"


def test_supra_mode_decides_on_complexity_alone(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "SCORE_WITH_MF", False)
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", False)
    monkeypatch.setattr(server, "EXPENSIVE_MIN_COMPLEXITY", 3)
    calls = []

    def fake_supra(prompt):
        calls.append(prompt)
        return (3, 120)

    monkeypatch.setattr(server, "_supra_complexity", fake_supra)
    decision, score, complexity, ms = server._decide_uncached("refactor the parser")
    assert decision == "expensive" and score is None and complexity == 3 and ms == 120

    def fake_supra2(prompt):
        return (2, 90)

    monkeypatch.setattr(server, "_supra_complexity", fake_supra2)
    decision, score, complexity, ms = server._decide_uncached("write hello world")
    assert decision == "cheap" and score is None and complexity == 2
    assert calls  # supra ran even for a short prompt (no MF short gate)


def test_supra_early_stop_waits_for_complexity_digit(monkeypatch):
    # Regression: the stop criterion must fire only after the digit is emitted;
    # stopping at the bare "Complexity:" prefix parses as 0.
    class FakeTok:
        model_max_length = 5120
        pad_token_id = 0
        eos_token_id = 0

        def __call__(self, text, return_tensors=None, truncation=False, max_length=None):
            import torch
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, ids, skip_special_tokens=False):
            ids = [int(i) for i in ids]
            if 16 in ids:  # the complexity-digit token has been generated
                return "Task: x\nAnalysis: Domain: Programming | Complexity: 3 | Math: False | Route: small model"
            if 15 in ids:  # bare "Complexity:" prefix, digit not yet emitted
                return "Task: x\nAnalysis: Domain: Programming | Complexity:"
            return "Task: x\nAnalysis: Domain:"

    class FakeModel:
        def __init__(self):
            self.stopped_at = None

        def generate(self, **kwargs):
            import torch
            criteria = kwargs["stopping_criteria"]
            input_ids = torch.tensor([[1, 2]])
            for step in range(1, 21):
                input_ids = torch.cat([input_ids, torch.tensor([[10 + step]])], dim=1)
                if any(c(input_ids, None) for c in criteria):
                    self.stopped_at = step
                    break
            return input_ids

    fake = FakeModel()
    monkeypatch.setattr(server, "_load_supra", lambda: (fake, FakeTok()))
    complexity, _ = server._supra_complexity("prompt")
    assert complexity == 3
    assert fake.stopped_at == 6  # token 16 = the digit; never stops at the bare prefix (token 15)


def test_supra_mode_short_prompt_still_uses_supra(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", True)
    monkeypatch.setattr(server, "MIDDLE_MIN_COMPLEXITY", 3)
    monkeypatch.setattr(server, "EXPENSIVE_MIN_COMPLEXITY", 4)
    ran = []

    def fake_supra(prompt):
        ran.append(prompt)
        return (4, 100)

    monkeypatch.setattr(server, "_supra_complexity", fake_supra)
    decision, _, _, _ = server._decide_uncached("Implement Paxos with a safety proof")
    assert decision == "expensive" and ran


def test_supra_mode_score_with_mf_is_observability_only(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "SCORE_WITH_MF", True)
    monkeypatch.setattr(server, "_supra_complexity", lambda p: (2, 90))
    monkeypatch.setattr(server, "_load_router", lambda: FakeRouter(0.9))
    decision, score, _, _ = server._decide_uncached("low complexity prompt")
    # MF score is recorded but never flips a supra-cheap decision to expensive.
    assert decision == "cheap" and score == 0.9


def test_supra_mode_falls_back_to_mf_when_supra_unavailable(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")

    def boom(prompt):
        raise RuntimeError("no torch")

    monkeypatch.setattr(server, "_supra_complexity", boom)
    monkeypatch.setattr(server, "_load_router", lambda: FakeRouter(0.3))
    decision, score, complexity, ms = server._decide_uncached("anything")
    assert decision == "expensive" and score == 0.3 and complexity is None
    # and low MF scores stay cheap through the legacy path
    monkeypatch.setattr(server, "_load_router", lambda: FakeRouter(0.05))
    decision, _, _, _ = server._decide_uncached("anything")
    assert decision == "cheap"


def test_mf_mode_unchanged(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "mf")
    monkeypatch.setattr(server, "_supra_complexity", lambda p: (4, 100))
    monkeypatch.setattr(server, "_load_router", lambda: FakeRouter(0.3))
    decision, score, complexity, ms = server._decide_uncached("anything")
    assert decision == "expensive" and score == 0.3 and complexity is None
    monkeypatch.setattr(server, "_load_router", lambda: FakeRouter(0.1))
    long_prompt = ("please refactor this parser module for clarity and add unit tests, then document the public API "
                   "surface and update the changelog with a summary of the behavioral changes for the next release")
    assert len(long_prompt) > 120
    decision, _, complexity, _ = server._decide_uncached(long_prompt)
    assert decision == "expensive" and complexity == 4  # supra gate below threshold


def test_cheap_successes_pin_cheap(monkeypatch):
    monkeypatch.setattr(server, "PIN_CHEAP_AFTER", 3)
    h = server._prompt_hash("Proceed")
    for _ in range(3):
        server._store_note(h, "cheap", ok=True)
    entry = server._store_pinned(h)
    assert entry is not None and entry["decision"] == "cheap"
    assert entry.get("pin_until", 0) > time.time()


def test_identical_repeats_do_not_pin_expensive(monkeypatch):
    monkeypatch.setattr(server, "RETRY_WINDOW_S", 900)
    h = server._prompt_hash("Proceed")
    req_hash = "same-request"
    # Three ordinary repeats should be logged as retried outcomes but must not
    # poison routing for common agent-loop prompts.
    for i in range(3):
        server._record_and_detect_retry(req_hash, "cheap", "deepseek-v4-flash", h,
                                        f"req_{i}", f"occ_{i}")
    assert server._store_pinned(h) is None
    assert h not in server._decision_store


def test_cheap_refusals_pin_expensive(monkeypatch):
    monkeypatch.setattr(server, "PIN_EXPENSIVE_AFTER", 2)
    h = server._prompt_hash("hardening-case")
    server._store_note(h, "cheap", ok=False)
    assert server._store_pinned(h) is None
    server._store_note(h, "cheap", ok=False)
    entry = server._store_pinned(h)
    assert entry is not None and entry["decision"] == "expensive"


def test_pin_expiry_and_streak_reset(monkeypatch):
    monkeypatch.setattr(server, "PIN_EXPENSIVE_AFTER", 2)
    h = server._prompt_hash("flaky-prompt")
    server._store_note(h, "cheap", ok=False)
    server._store_note(h, "cheap", ok=False)
    assert server._store_pinned(h) is not None
    # simulate expiry
    server._decision_store[h]["pin_until"] = time.time() - 1
    assert server._store_pinned(h) is None
    # a failure >24h after the last note starts a fresh streak (no instant re-pin)
    server._decision_store[h]["ts"] = time.time() - 90000
    server._store_note(h, "cheap", ok=False)
    assert server._store_pinned(h) is None


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DECISION_STORE_PATH", tmp_path / "decision-state.jsonl")
    h = server._prompt_hash("persist-me")
    server._store_note(h, "cheap", ok=True)
    server._store_note(h, "cheap", ok=True)
    server._decision_store.clear()
    server._store_load()
    assert h in server._decision_store
    assert server._decision_store[h]["ok"] == 2


def test_decide_uses_pinned_decision(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "SCORE_WITH_MF", False)
    called = []
    monkeypatch.setattr(server, "_supra_complexity", lambda p: called.append(p) or (1, 50))
    h = server._prompt_hash("pinned-prompt")
    server._store_note(h, "cheap", ok=True)
    server._decision_store[h]["ok"] = server.PIN_CHEAP_AFTER
    server._decision_store[h]["pin_until"] = time.time() + 3600
    decision, _, complexity, ms = server._decide("pinned-prompt")
    assert decision == "cheap" and complexity is None
    assert not called  # scoring bypassed entirely


def test_log_and_headers_accept_missing_score(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "LOG_PATH", tmp_path / "decisions.log")
    server._log("cheap", None, "deepseek-v4-flash", "hello", None)
    row = json.loads(open(tmp_path / "decisions.log").read().strip().splitlines()[-1])
    assert row["score"] is None
    headers = server._route_headers("cheap", None, server.CHEAP, "req_1", None, None, pinned=True)
    assert headers["x-route-score"] == "n/a"
    assert headers["x-route-pinned"] == "true"



def test_supra_three_tier_mapping_when_middle_configured(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", True)
    monkeypatch.setattr(server, "MIDDLE_MIN_COMPLEXITY", 3)
    monkeypatch.setattr(server, "EXPENSIVE_MIN_COMPLEXITY", 4)
    monkeypatch.setattr(server, "_supra_complexity", lambda p: (3, 10))
    assert server._decide_uncached("middle work")[0] == "middle"
    monkeypatch.setattr(server, "_supra_complexity", lambda p: (4, 10))
    assert server._decide_uncached("hard work")[0] == "expensive"


def test_session_ids_are_source_namespaced_and_hmac_opaque(monkeypatch):
    from starlette.requests import Request

    def req(headers):
        return Request({"type": "http", "method": "POST", "path": "/", "headers":
                        [(k.lower().encode(), v.encode()) for k, v in headers.items()],
                        "query_string": b"", "scheme": "http", "server": ("test", 80),
                        "client": ("test", 1), "root_path": ""})

    header_id, source = server._session_id({}, req({"X-Route-Session": "same"}))
    metadata_id, metadata_source = server._session_id({"metadata": {"session_id": "same"}}, req({}))
    assert source == "header" and metadata_source == "metadata"
    assert header_id != metadata_id and header_id != "same"
    assert server._session_id({"user": "same"}, req({})) == (None, None)
    monkeypatch.setenv("ROUTELLM_SESSION_FROM_USER", "1")
    user_id, user_source = server._session_id({"user": "same"}, req({}))
    assert user_source == "user" and user_id not in {header_id, metadata_id}


def test_continuation_only_sticks_existing_session(monkeypatch):
    server._session_state.clear()
    assert server._is_continuation("Proceed")
    assert not server._is_continuation("Continue implementing Paxos with a proof")
    assert server._session_route("missing", "Proceed", "expensive", 4)[0] == "expensive"
    server._session_state["s"] = {"tier": "middle", "last_seen": time.time(), "turns": 1}
    assert server._session_route("s", "Proceed", "cheap", 1) == ("middle", "continuation_sticky")
    assert server._session_route("s", "new task: cleanup", "cheap", 1)[0] == "cheap"



def test_failover_and_backend_matrix(monkeypatch):
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", False)
    assert server._failover_routes("cheap") == ("cheap", "expensive")
    assert server._failover_routes("expensive") == ("expensive", "cheap")
    assert server._backend_for("middle") is server.EXPENSIVE
    monkeypatch.setattr(server, "MIDDLE_CONFIGURED", True)
    assert server._failover_routes("cheap") == ("cheap", "middle")
    assert server._failover_routes("middle") == ("middle", "expensive")
    assert server._failover_routes("expensive") == ("expensive", "middle")
    assert server._backend_for("middle") is server.MIDDLE


def test_session_get_ttl_and_copy(monkeypatch):
    server._session_state.clear()
    monkeypatch.setattr(server, "SESSION_TTL_S", 10)
    server._session_state["expired"] = {"tier": "cheap", "last_seen": time.time() - 11}
    assert server._session_get("expired") is None and "expired" not in server._session_state
    server._session_state["live"] = {"tier": "cheap", "last_seen": time.time(), "turns": 1}
    got = server._session_get("live")
    got["tier"] = "expensive"
    assert server._session_state["live"]["tier"] == "cheap"


def test_session_note_lru_and_usage(monkeypatch):
    server._session_state.clear()
    monkeypatch.setattr(server, "SESSION_STATE_MAX", 2)
    server._session_note("a", "cheap", 1)
    server._session_note("b", "middle", 3, {"prompt_cache_hit_tokens": 4})
    server._session_note("a", "cheap", 1)
    server._session_note("c", "expensive", 4)
    assert "b" not in server._session_state and server._session_state["a"]["turns"] == 2


def test_user_is_opt_in(monkeypatch):
    from starlette.requests import Request
    req = Request({"type": "http", "method": "POST", "path": "/", "headers": [],
                   "query_string": b"", "scheme": "http", "server": ("test", 80),
                   "client": ("test", 1), "root_path": ""})
    monkeypatch.delenv("ROUTELLM_SESSION_FROM_USER", raising=False)
    assert server._session_id({"user": "account"}, req) == (None, None)



def test_existing_session_continuation_skips_scoring(monkeypatch):
    server._session_state.clear()
    server._session_state["s"] = {"tier": "middle", "last_seen": time.time(), "turns": 1,
                                  "last_complexity": 3}
    monkeypatch.setattr(server, "_decide_cached", lambda prompt: (_ for _ in ()).throw(AssertionError("scored")))
    assert server._decide("Proceed", "s")[:3] == ("middle", None, 3)


def test_session_ratchet_never_slides_down_but_can_climb():
    # Cache-maximizing ratchet: a session holds its warm prefix target, may
    # only climb.  Non-continuation prompt so neither sticky path intervenes.
    server._session_state.clear()
    assert not server._is_continuation("Explain the page-fault handling path in this kernel")
    prompt = "Explain the page-fault handling path in this kernel"
    server._session_state["s"] = {"tier": "expensive", "last_seen": time.time(), "turns": 1}
    assert server._session_route("s", prompt, "cheap", 1) == ("expensive", "downgrade_hysteresis")
    server._session_state["s"] = {"tier": "cheap", "last_seen": time.time(), "turns": 1}
    assert server._session_route("s", prompt, "expensive", 5) == ("expensive", "strong_upgrade")
