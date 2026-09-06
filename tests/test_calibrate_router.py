"""Cover router threshold calibration and calibration task sampling."""

from __future__ import annotations

import json

import calibrate_router as calibration
import sample_calibration_tasks as sampler


def _row(task: str, tier: str, resolved: bool, cost: float, score: float) -> dict:
    return {
        "instance_id": task,
        "tier": tier,
        "resolved": resolved,
        "cost_usd": cost,
        "score": score,
    }


def _paired_fixture() -> dict[str, dict[str, dict]]:
    rows = [
        _row("t-rescue", "efficient", False, 0.05, 0.8),
        _row("t-rescue", "capable", True, 0.40, 0.0),
        _row("t-loss", "efficient", True, 0.05, 0.9),
        _row("t-loss", "capable", False, 0.40, 0.0),
        _row("t-safe", "efficient", True, 0.05, 0.1),
        _row("t-safe", "capable", True, 0.40, 0.0),
        _row("t-hard", "efficient", False, 0.05, 0.2),
        _row("t-hard", "capable", False, 0.40, 0.0),
    ]
    return calibration.pair_rows(rows)


def test_quadrants_classify_each_outcome_pair():
    quadrants = calibration.classify_quadrants(_paired_fixture())
    assert quadrants["rescue"] == ["t-rescue"]
    assert quadrants["loss"] == ["t-loss"]
    assert quadrants["safe"] == ["t-safe"]
    assert quadrants["hard"] == ["t-hard"]


def test_sweep_reports_cost_per_solved_and_escalation_rate():
    sweep = calibration.sweep_thresholds(_paired_fixture(), [0.5])
    assert len(sweep) == 1
    row = sweep[0]
    assert row["tasks"] == 4
    assert row["escalation_rate"] == 0.5
    assert row["rescued"] == 1
    assert row["lost"] == 1
    assert row["cost_per_solved"] == row["total_cost_usd"] / 2


def test_recommend_prefers_solve_rate_inside_budget():
    sweep = [
        {"threshold": 0.5, "solve_rate": 0.75, "cost_per_solved": 0.5,
         "escalation_rate": 0.5},
        {"threshold": 0.7, "solve_rate": 0.5, "cost_per_solved": 0.2,
         "escalation_rate": 0.25},
        {"threshold": 0.9, "solve_rate": 0.5, "cost_per_solved": 0.1,
         "escalation_rate": 0.0},
    ]
    picked = calibration.recommend(sweep, 0.6)
    assert picked is not None
    assert picked["threshold"] == 0.5


def test_recommend_respects_escalation_budget():
    sweep = calibration.sweep_thresholds(_paired_fixture(), [0.05, 0.5, 0.95])
    picked = calibration.recommend(sweep, 0.6)
    assert picked is not None
    assert picked["threshold"] == 0.95
    assert calibration.recommend(sweep, 0.0)["threshold"] == 0.95


def test_recommend_returns_none_when_budget_covers_nothing():
    sweep = calibration.sweep_thresholds(_paired_fixture(), [0.05])
    assert calibration.recommend(sweep, 0.0) is None


def test_load_rows_skips_metadata_and_incomplete(tmp_path):
    path = tmp_path / "runs.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"record_type": "metadata"}),
                json.dumps(_row("t1", "efficient", True, 0.05, 0.1)),
                json.dumps({"instance_id": "t1", "tier": "capable"}),
                "not json",
                "",
            ]
        )
    )
    rows = calibration.load_rows(path)
    assert [row["instance_id"] for row in rows] == ["t1"]


def test_pair_rows_keeps_only_complete_pairs():
    rows = [_row("solo", "efficient", True, 0.05, 0.1)]
    assert calibration.pair_rows(rows) == {}


def test_parse_thresholds_sorts_and_dedupes():
    assert calibration.parse_thresholds("0.7, 0.3,0.7") == [0.3, 0.7]


def test_suggested_run_command_pins_both_tiers():
    command = calibration.suggested_run_command("manifest.json")
    assert "glm-only=modal.glm-5-3/zai-org/GLM-5.3" in command
    assert "grok-only=bedrock-openai/global.xai.grok-4.6" in command


def _pool(count: int, difficulty: str) -> list[dict]:
    return [{"task_id": f"{difficulty}-{i:02d}", "difficulty": difficulty} for i in range(count)]


def test_sampler_splits_quota_across_strata():
    pool = _pool(10, "easy") + _pool(10, "hard")
    picked = sampler.sample_tasks(pool, 6, seed=7)
    assert len(picked) == 6
    strata = [sampler.stratum_of(task) for task in picked]
    assert strata.count("easy") == 3
    assert strata.count("hard") == 3


def test_sampler_is_deterministic_for_fixed_seed():
    pool = _pool(10, "easy") + _pool(10, "hard")
    first = sampler.sample_tasks(pool, 6, seed=7)
    second = sampler.sample_tasks(pool, 6, seed=7)
    assert [task["task_id"] for task in first] == [task["task_id"] for task in second]


def test_sampler_backfills_when_one_stratum_is_thin():
    pool = _pool(1, "easy") + _pool(10, "hard")
    picked = sampler.sample_tasks(pool, 6, seed=7)
    assert len(picked) == 6
    assert len({task["task_id"] for task in picked}) == 6


def test_sampler_falls_back_for_unlabeled_tasks():
    assert sampler.stratum_of({"task_id": "x"}) == "unspecified"


def test_sampler_builds_manifest_with_instance_ids():
    tasks = _pool(2, "easy")
    manifest = sampler.build_manifest("dsbench-data-modeling", tasks, seed=7)
    assert manifest["instance_count"] == 2
    assert manifest["instances"][0]["instance_id"] == "easy-00"


def test_sampler_load_pool_skips_bad_lines(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text('{"task_id": "a"}\nnot json\n{"no_id": true}\n')
    assert sampler.load_pool(path) == [{"task_id": "a"}]
