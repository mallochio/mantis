"""Tests for scripts/update_pool.py"""

import json

import pytest
import update_pool as up  # noqa: I001


def test_split_model_spec():
    assert up.split_model_spec("openai/gpt-5|high") == ("openai/gpt-5", "high")
    assert up.split_model_spec("anthropic/claude-3") == ("anthropic/claude-3", None)


def test_normalize_model_id():
    assert up.normalize_model_id("openrouter/openai/gpt-5") == "openai/gpt-5"
    assert up.normalize_model_id("openai/gpt-5.6-sol") == "openai/gpt-5.6-sol"
    assert up.normalize_model_id("claude-sonnet-5") == "anthropic/claude-sonnet-5"
    with pytest.raises(ValueError):
        up.normalize_model_id("unknown-model-name")


def test_parse_pool():
    pool = up.parse_pool(
        "anthropic/claude-sonnet-5|medium,deepseek/deepseek-v4|none"
    )
    assert pool == [
        ("anthropic/claude-sonnet-5", "medium"),
        ("deepseek/deepseek-v4", "none"),
    ]


def test_alias_for():
    assert up.alias_for("anthropic/claude-sonnet-5", "medium") == "claude-sonnet-5-medium"
    assert up.alias_for("deepseek/deepseek-v4", "none") == "deepseek-v4"
    assert up.alias_for("z-ai/glm-5.2", None) == "glm-5.2"


def test_update_litellm_config(tmp_path):
    original = """model_list:
  - model_name: old
    litellm_params:
      model: openai/old
  # 7-slot fugu worker pool
  # Reasoning effort is baked in
  - model_name: a
    litellm_params:
      model: openai/a

  # Other section
  - model_name: other
"""
    path = tmp_path / "litellm.yaml"
    path.write_text(original)
    up.update_litellm_config(path, [("anthropic/claude-3", "medium")])
    text = path.read_text()
    assert "- model_name: claude-3-medium" in text
    assert "model: openai/anthropic/claude-3" in text
    assert "reasoning_effort: medium" in text
    assert "- model_name: other" in text
    assert "- model_name: a" not in text


def test_update_yaml_env(tmp_path):
    yaml = "envs:\n  RETRAIN_WORKER_MODELS: \"old\"\n"
    path = tmp_path / "router.yaml"
    path.write_text(yaml)
    up.update_yaml_env(path, "new|medium,other")
    assert 'RETRAIN_WORKER_MODELS: "new|medium,other"' in path.read_text()


def test_update_env_file(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FOO=1\n")
    up.update_env_file(env, ["claude-sonnet-5", "glm-5.2"])
    text = env.read_text()
    assert "FUGU_WORKER_MODELS=claude-sonnet-5,glm-5.2" in text


def test_update_cost_table_with_costs(tmp_path):
    path = tmp_path / "costs.json"
    costs = {
        "anthropic/claude-sonnet-5": {"prompt": 0.001, "completion": 0.002},
    }
    table = up.update_cost_table(path, [("anthropic/claude-sonnet-5", None)], costs)
    assert table["anthropic/claude-sonnet-5"] == 4.0  # 0.001*2000 + 0.002*1000
    data = json.loads(path.read_text())
    assert "_note" in data


def test_update_cost_table_preserves_existing(tmp_path):
    path = tmp_path / "costs.json"
    path.write_text(json.dumps({"openai/gpt-old": 1.0, "_note": "x"}))
    up.update_cost_table(path, [("openai/gpt-old", None)], None)
    data = json.loads(path.read_text())
    assert data["openai/gpt-old"] == 1.0


def test_update_env_file_existing(tmp_path):
    env = tmp_path / ".env"
    env.write_text("FUGU_WORKER_MODELS=old\nFOO=1\n")
    up.update_env_file(env, ["new-a", "new-b"])
    assert env.read_text() == "FUGU_WORKER_MODELS=new-a,new-b\nFOO=1\n"


def test_update_litellm_config_no_effort(tmp_path):
    original = """model_list:
  # 7-slot fugu worker pool
  - model_name: a
    litellm_params:
      model: openai/a

  # Other
"""
    path = tmp_path / "litellm.yaml"
    path.write_text(original)
    up.update_litellm_config(path, [("deepseek/deepseek-v4", "none")])
    text = path.read_text()
    assert "model_name: deepseek-v4" in text
    assert "reasoning_effort: none" in text


def test_main(monkeypatch, tmp_path):
    litellm = tmp_path / "litellm.yaml"
    litellm.write_text("""model_list:
  # 7-slot fugu worker pool
  - model_name: a
    litellm_params:
      model: openai/a

  # Other
""")
    costs = tmp_path / "costs.json"
    router = tmp_path / "router.yaml"
    router.write_text('envs:\n  RETRAIN_WORKER_MODELS: "old"\n')
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text('envs:\n  RETRAIN_WORKER_MODELS: "old"\n')
    full = tmp_path / "full.yaml"
    full.write_text('envs:\n  RETRAIN_WORKER_MODELS: "old"\n')
    env = tmp_path / ".env"

    monkeypatch.setattr(
        up,
        "REPO_ROOT",
        tmp_path,
    )
    up.main([
        "--pool", "anthropic/claude-sonnet-5|medium,deepseek/deepseek-v4|none",
        "--litellm-config", str(litellm),
        "--worker-costs", str(costs),
        "--update-yamls",
        "--router-yaml", str(router),
        "--conductor-smoke-yaml", str(smoke),
        "--conductor-full-yaml", str(full),
        "--env-file", str(env),
    ])

    assert "claude-sonnet-5-medium" in litellm.read_text()
    expected_pool = (
        'RETRAIN_WORKER_MODELS: "'
        'anthropic/claude-sonnet-5|medium,deepseek/deepseek-v4|none"'
    )
    assert expected_pool in router.read_text()
    assert "FUGU_WORKER_MODELS=claude-sonnet-5-medium,deepseek-v4" in env.read_text()


def test_update_litellm_config_no_marker(tmp_path):
    path = tmp_path / "litellm.yaml"
    path.write_text("model_list:\n  - model_name: x\n")
    with pytest.raises(SystemExit):
        up.update_litellm_config(path, [("a/b", None)])


def test_update_litellm_config_no_entries(tmp_path):
    path = tmp_path / "litellm.yaml"
    path.write_text("model_list:\n  # 7-slot fugu worker pool\n")
    with pytest.raises(SystemExit):
        up.update_litellm_config(path, [("a/b", None)])
