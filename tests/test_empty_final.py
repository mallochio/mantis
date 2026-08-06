"""Regression tests: a run whose final reply is empty (reasoning-model
workers can return content:null) must fall back to the last non-empty reply
instead of returning an empty final answer."""

from __future__ import annotations

from types import SimpleNamespace

import serve


def _run_messages(text: str = "do the task") -> list[dict[str, str]]:
    return [{"role": "system", "content": "system"}, {"role": "user", "content": text}]


def test_trinity_final_falls_back_to_last_nonempty_reply(monkeypatch):
    roles = iter([("Worker", 0), ("Worker", 0)])
    monkeypatch.setattr(
        serve,
        "get_router",
        lambda: SimpleNamespace(
            route=lambda *_a, **_k: dict(zip(("role_name", "agent_id"), next(roles), strict=True))
        ),
    )
    replies = iter(["good answer", ""])
    monkeypatch.setattr(serve, "_model_completion", lambda *_a: (next(replies), []))
    run = serve.TrinityRun("t1", _run_messages(), [], slot_models=["worker"], max_turns=2)
    assert run.advance(None)["type"] == "step_complete"
    assert run.advance(None)["type"] == "step_complete"
    final = run.advance(None)
    assert final["type"] == "final"
    assert final["text"] == "good answer"


def test_conductor_final_falls_back_to_last_nonempty_output(monkeypatch):
    monkeypatch.setattr(serve, "parse_workflow", lambda _text: ([0, 0], ["sub1", "sub2"], [[], []]))
    replies = iter([("plan", []), ("first output", []), ("", [])])
    monkeypatch.setattr(serve, "_model_completion", lambda *_a: next(replies))
    run = serve.ConductorRun("c1", _run_messages(), [], slot_models=["worker"])
    assert run.advance(None)["role"] == "Planner"
    assert run.advance(None)["type"] == "step_complete"
    assert run.advance(None)["type"] == "step_complete"
    final = run.advance(None)
    assert final["type"] == "final"
    assert final["text"] == "first output"
