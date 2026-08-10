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


def test_supra_mode_decides_on_complexity_alone(monkeypatch):
    monkeypatch.setattr(server, "ROUTER_NAME", "supra")
    monkeypatch.setattr(server, "SCORE_WITH_MF", False)
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
