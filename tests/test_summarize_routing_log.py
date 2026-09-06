"""Cover routing-log summary for threshold mining."""

from __future__ import annotations

import json

import summarize_routing_log as summary


def _record(model: str, prompt: int, cached: int, total: int) -> dict:
    return {
        "model": model,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "total_tokens": total,
    }


def test_summarize_groups_by_model_with_hit_rate():
    records = [
        _record("glm", 100, 90, 120),
        _record("glm", 100, 100, 110),
        _record("kimi", 500, 0, 560),
    ]
    report = summary.summarize(records)
    assert report["records"] == 3
    assert report["models"]["glm"]["calls"] == 2
    assert report["models"]["glm"]["cache_hit_rate"] == 190 / 200
    assert report["models"]["kimi"]["cache_hit_rate"] == 0.0
    assert report["total_tokens"] == 790


def test_summarize_handles_empty_input():
    assert summary.summarize([]) == {"records": 0, "total_tokens": 0, "models": {}}


def test_load_records_skips_bad_lines(tmp_path):
    path = tmp_path / "routing-log.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(_record("glm", 10, 5, 15)),
                "not json",
                json.dumps({"no_model": True}),
                "",
            ]
        )
    )
    assert len(summary.load_records(path)) == 1


def test_load_records_coerces_missing_token_fields(tmp_path):
    path = tmp_path / "routing-log.jsonl"
    path.write_text(json.dumps({"model": "glm"}) + "\n")
    report = summary.summarize(summary.load_records(path))
    assert report["models"]["glm"]["cache_hit_rate"] == 0.0
