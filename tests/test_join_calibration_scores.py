"""Cover the Harbor-results + routing-log join for threshold calibration."""

from __future__ import annotations

import io
import json
from pathlib import Path

import join_calibration_scores as join


def _trial(
    task: str,
    reward: float,
    *,
    session: str | None = None,
    cost: float | None = None,
    model: str = "glm-5.3-flash",
    tokens: tuple[int, int] = (1000, 100),
) -> dict:
    return {
        "task_name": task,
        "trial_name": f"{task}-trial-0",
        "agent_result": {
            "cost_usd": cost,
            "n_input_tokens": tokens[0],
            "n_output_tokens": tokens[1],
            "metadata": {
                "cal_session_id": session,
                "model_requested": model,
            },
        },
        "verifier_result": {"rewards": {"reward": reward}},
    }


def _write_job(tmp: Path, name: str, trials: list[dict]) -> Path:
    job = tmp / name
    for trial in trials:
        trial_dir = job / trial["trial_name"]
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(json.dumps(trial))
    return job


def _routing_record(task=None, session=None, model="glm-5.3-flash") -> dict:
    return {
        "ts": "2026-09-10T00:00:00Z",
        "task": task,
        "trial_id": None,
        "session_id": session,
        "model": model,
        "tier": "",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }


def test_find_trial_results_reads_task_named_results(tmp_path):
    job = _write_job(tmp_path, "job", [_trial("t1", 1.0)])
    (job / "noise").mkdir()
    (job / "noise" / "result.json").write_text('{"unrelated": true}')
    (job / "noise" / "bad.json").write_text("not json")
    trials = join.find_trial_results(job)
    assert [t["task_name"] for t in trials] == ["t1"]


def test_trial_resolved_requires_full_reward(tmp_path):
    assert join.trial_resolved(_trial("t", 1.0)) is True
    assert join.trial_resolved(_trial("t", 0.5)) is False
    assert join.trial_resolved(_trial("t", 0.0)) is False
    partial = _trial("t", 0.0)
    partial["verifier_result"] = {"rewards": {"reward": 0.0, "partial": 1.0}}
    assert join.trial_resolved(partial) is True


def test_trial_cost_prefers_reported_then_prices(tmp_path):
    prices = {"glm-5.3-flash": {"input_per_token": 1e-6, "output_per_token": 2e-6}}
    assert join.trial_cost(_trial("t", 1.0, cost=0.5), prices) == 0.5
    priced = join.trial_cost(
        _trial("t", 1.0, cost=None, tokens=(1000, 100)), prices
    )
    assert priced == 1000 * 1e-6 + 100 * 2e-6
    assert join.trial_cost(_trial("t", 1.0, cost=None), {}) == 0.0


def test_extract_arm_rows_normalizes_trials(tmp_path):
    job = _write_job(
        tmp_path, "eff", [_trial("t1", 1.0, session="s1", cost=0.01)]
    )
    rows = join.extract_arm_rows(job, "efficient", {})
    assert rows == [
        {
            "instance_id": "t1",
            "tier": "efficient",
            "resolved": True,
            "cost_usd": 0.01,
            "session_id": "s1",
            "trial_name": "t1-trial-0",
        }
    ]


def test_trial_session_falls_back_to_trial_name_agent():
    trial = _trial("t", 1.0, session=None)
    assert join.trial_session(trial) == "t-trial-0__agent"


def test_routing_scores_fraction_by_task_and_session(tmp_path):
    log = tmp_path / "routing.jsonl"
    records = [
        _routing_record(task="t1", session="s1", model="glm-5.3-flash"),
        _routing_record(task="t1", session="s1", model="muse-spark-1.3-contributor"),
        _routing_record(task="t1", session="s1", model="glm-5.3-flash"),
        _routing_record(task="t1", session="s1", model="muse-spark-1.3-contributor"),
        _routing_record(session="s2", model="muse-spark-1.3-contributor"),
        _routing_record(session="s2", model="glm-5.3-flash"),
    ]
    log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    by_task, by_session = join.load_routing_scores(log, "muse-spark-1.3-contributor")
    assert by_task == {"t1": 0.5}
    assert by_session == {"s1": 0.5, "s2": 0.5}


def test_attach_scores_prefers_task_then_session(tmp_path):
    rows = [
        {"instance_id": "t1", "tier": "efficient", "resolved": True,
         "cost_usd": 0.01, "session_id": "s1", "trial_name": "tr1"},
        {"instance_id": "t2", "tier": "efficient", "resolved": False,
         "cost_usd": 0.02, "session_id": "s2", "trial_name": "tr2"},
        {"instance_id": "t3", "tier": "efficient", "resolved": True,
         "cost_usd": 0.02, "session_id": None, "trial_name": "tr3"},
        {"instance_id": "t1", "tier": "capable", "resolved": True,
         "cost_usd": 0.5, "session_id": "c1", "trial_name": "trc"},
    ]
    out = join.attach_scores(
        rows, {"t1": 0.25}, {"s2": 0.75, "tr3": 0.6}
    )
    scores = {(r["instance_id"], r["tier"]): r["score"] for r in out}
    assert scores == {
        ("t1", "efficient"): 0.25,
        ("t2", "efficient"): 0.75,
        ("t3", "efficient"): 0.6,
        ("t1", "capable"): 0.0,
    }
    assert all("session_id" not in r and "trial_name" not in r for r in out)


def test_main_end_to_end_feeds_calibrate_router(tmp_path):
    import calibrate_router

    eff = _write_job(
        tmp_path,
        "eff",
        [_trial("t-rescue", 0.0, session="se1"), _trial("t-safe", 1.0, session="se2")],
    )
    cap = _write_job(
        tmp_path,
        "cap",
        [
            _trial(
                "t-rescue", 1.0, session="sc1", cost=0.4,
                model="muse-spark-1.3-contributor",
            ),
            _trial(
                "t-safe", 1.0, session="sc2", cost=0.4,
                model="muse-spark-1.3-contributor",
            ),
        ],
    )
    log = tmp_path / "routing.jsonl"
    probe = [
        _routing_record(session="se1", model="muse-spark-1.3-contributor"),
        _routing_record(session="se1", model="glm-5.3-flash"),
        _routing_record(session="se2", model="glm-5.3-flash"),
    ]
    log.write_text("\n".join(json.dumps(r) for r in probe) + "\n")

    out = io.StringIO()
    code = join.main(
        [
            "--efficient-job", str(eff),
            "--capable-job", str(cap),
            "--routing-log", str(log),
            "--capable-model", "muse-spark-1.3-contributor",
        ],
        stdout=out,
    )
    assert code == 0
    rows = [
        json.loads(line) for line in out.getvalue().splitlines() if line.strip()
    ]
    pairs = calibrate_router.pair_rows(rows)
    quadrants = calibrate_router.classify_quadrants(pairs)
    assert quadrants["rescue"] == ["t-rescue"]
    assert quadrants["safe"] == ["t-safe"]


def test_main_fails_on_missing_log(tmp_path):
    out = io.StringIO()
    code = join.main(
        [
            "--efficient-job", str(tmp_path),
            "--capable-job", str(tmp_path),
            "--routing-log", str(tmp_path / "nope.jsonl"),
            "--capable-model", "muse-spark-1.3-contributor",
        ],
        stdout=out,
    )
    assert code == 1
