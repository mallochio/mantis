from __future__ import annotations

import json

import jev_assess_routed_ab as jev


def _record() -> dict:
    return {
        "variant": "fusion-pinned",
        "case_id": "no-tool-chat",
        "provider_cost": 0.02,
        "tool_rounds": 0,
        "wall_time_s": 3.2,
        "content": "ANSWER: ROUTING_OK",
    }


def test_build_state_includes_convention_and_final_answer(tmp_path):
    transcript = tmp_path / "fusion-pinned--no-tool-chat.json"
    payload = {
        "content": "ANSWER: ROUTING_OK",
        "event": {"handoff_history": [{"kind": "answer", "follow_up": 0}]},
        "tool_rounds": 0,
    }
    transcript.write_text(json.dumps(payload))
    jev.TRANSCRIPTS = tmp_path
    case = {
        "id": "no-tool-chat",
        "prompt": "Answer with exactly ROUTING_OK and do not call tools.",
        "expected": {
            "description": "Exact response, no tools.",
            "checks": ["compare exact response"],
        },
    }
    state = jev.build_state(_record(), case)

    assert state["answer_convention"].startswith("The orchestration system")
    assert state["task"].startswith("Answer with exactly")
    assert state["agent"]["final_answer"] == "ANSWER: ROUTING_OK"
    assert state["agent"]["handoffs"] == [{"kind": "answer", "follow_up": 0}]


def test_questions_cover_three_primitives():
    case = {
        "prompt": "x",
        "expected": {"description": "d", "checks": ["a", "b"]},
    }
    questions = jev.questions_for(case)

    assert questions["success"]["type"] == "noul"
    assert "a b" in questions["success"]["instructions"]
    assert questions["quality"]["type"] == "score"
    assert len(questions["quality"]["criteria"]) == 4
    assert questions["failure_mode"]["type"] == "choice"
    assert set(questions["failure_mode"]["criteria"]) == set(jev.FAILURE_MODES)
