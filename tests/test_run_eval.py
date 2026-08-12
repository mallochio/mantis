"""Bifrost catalog regression checks for the evaluation harness."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_eval)


def test_catalog_models_drive_eval_costs():
    models, conductor = run_eval.load_catalog_models(ROOT / "config" / "catalog.toml")
    assert models[0] == "google/gemini-3.6-flash"
    assert conductor == "google/gemini-3.5-flash-lite"
    assert run_eval.model_cost("bifrost/openai/gpt-5.6-sol", {"openai/gpt-5.6-sol": 0.04}) == 0.04


def test_default_worker_costs_path_is_tracked_config():
    path = run_eval.default_worker_costs_path()
    assert path == ROOT / "config" / "worker-costs.json"
    assert run_eval.load_worker_costs(path)


def test_prepare_output_overwrites_unless_append(tmp_path):
    output = tmp_path / "results.jsonl"
    output.write_text("old\n")
    run_eval.prepare_output(output, append=False)
    assert output.read_text() == ""
    output.write_text("old\n")
    run_eval.prepare_output(output, append=True)
    assert output.read_text() == "old\n"
