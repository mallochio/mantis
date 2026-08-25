"""Catalog-driven Switchyard config generation for mantis/base."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pytest
from model_catalog_schema import CatalogError, load_base_route
from switchyard_config import (
    load_switchyard_route,
    main,
    render_switchyard_toml,
    write_switchyard_toml,
)


def _catalog(*, algorithm: str = "stage_router", extra: str = "") -> str:
    return (
        "version = 1\n\n"
        "[providers.bifrost]\n"
        'adapter = "openai-compatible"\n'
        'base_url = "http://127.0.0.1:8080/v1"\n'
        'credential_env = "BIFROST_API_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n\n'
        "[providers.\"modal.prod\"]\n"
        'adapter = "modal"\n'
        'base_url = "https://modal.example.test/v1"\n'
        'credential_env = "MODAL_KEY"\n'
        'protocols = ["chat_completions"]\n\n'
        "[base]\n"
        'revision = "test-rev"\n'
        'route_id = "mantis-base"\n'
        f'algorithm = "{algorithm}"\n'
        'picker = "efficient_first"\n'
        "confidence_threshold = 0.5\n"
        "recent_turn_window = 3\n\n"
        "[base.targets.efficient]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n"
        'protocols = ["chat_completions", "responses"]\n\n'
        "[base.targets.capable]\n"
        'provider = "bifrost"\n'
        'upstream_model = "anthropic/claude-opus-5"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 128000\n"
        'protocols = ["chat_completions", "responses"]\n'
        f"{extra}"
    )


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "catalog.toml"
    path.write_text(content)
    return path


def test_shipped_catalog_renders_stage_router():
    route = load_switchyard_route(Path("config/catalog.toml"))
    text = render_switchyard_toml(route)
    parsed = tomllib.loads(text)
    assert parsed["schema_version"] == 1
    assert parsed["routes"]["mantis_base"]["type"] == "stage_router"
    assert parsed["routes"]["mantis_base"]["id"] == "mantis-base"
    assert parsed["routes"]["mantis_base"]["picker"] == "efficient_first"
    assert parsed["targets"]["efficient"]["id"] == "google/gemini-3.7-flash"
    assert parsed["targets"]["capable"]["id"] == "anthropic/claude-opus-5"
    assert parsed["llm_clients"]["bifrost"]["api_key_env"] == "BIFROST_API_KEY"
    assert parsed["llm_clients"]["bifrost"]["base_url"] == "http://127.0.0.1:8080/v1"


def test_render_quotes_dotted_provider_names(tmp_path):
    content = _catalog().replace(
        '[base.targets.efficient]\n'
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n"
        'protocols = ["chat_completions", "responses"]\n',
        '[base.targets.efficient]\n'
        'provider = "modal.prod"\n'
        'upstream_model = "vendor/fast"\n'
        'protocols = ["chat_completions"]\n',
        1,
    )
    path = _write(tmp_path, content)
    text = render_switchyard_toml(load_switchyard_route(path))
    parsed = tomllib.loads(text)
    assert parsed["llm_clients"]["modal_prod"]["base_url"] == "https://modal.example.test/v1"
    assert parsed["targets"]["efficient"]["llm_client"] == "modal_prod"
    assert parsed["targets"]["capable"]["llm_client"] == "bifrost"


def test_escalation_uses_optional_judge_target(tmp_path):
    extra = (
        "\n[base.targets.judge]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'protocols = ["chat_completions"]\n'
    )
    path = _write(tmp_path, _catalog(algorithm="escalation", extra=extra))
    text = render_switchyard_toml(load_switchyard_route(path))
    parsed = tomllib.loads(text)
    route = parsed["routes"]["mantis_base"]
    assert route["type"] == "llm_classifier"
    assert route["mode"] == "escalation"
    assert route["classifier_target"] == "judge"
    assert route["weak_target"] == "efficient"
    assert route["strong_target"] == "capable"
    assert parsed["targets"]["judge"]["id"] == "google/gemini-3.7-flash"


def test_rejects_identical_efficient_and_capable(tmp_path):
    content = _catalog().replace(
        'upstream_model = "anthropic/claude-opus-5"',
        'upstream_model = "google/gemini-3.7-flash"',
        1,
    )
    path = _write(tmp_path, content)
    with pytest.raises(CatalogError, match="distinct"):
        load_switchyard_route(path)


def test_rejects_unknown_algorithm(tmp_path):
    path = _write(tmp_path, _catalog(algorithm="random"))
    with pytest.raises(CatalogError, match="stage_router or escalation"):
        load_base_route(tomllib.loads(path.read_text()))


def test_cli_validate_and_render(tmp_path, monkeypatch):
    catalog = _write(tmp_path, _catalog())
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    assert main(["validate", "--catalog", str(catalog)], stdout=out) == 0
    assert "mantis-base" in out.getvalue()
    rendered = tmp_path / "routes.toml"
    assert main(["render", "--catalog", str(catalog), "--output", str(rendered)]) == 0
    parsed = tomllib.loads(rendered.read_text())
    assert parsed["targets"]["efficient"]["extra_body"]["max_tokens"] == 65536
    stdout = io.StringIO()
    assert main(["render", "--catalog", str(catalog)], stdout=stdout) == 0
    assert "stage_router" in stdout.getvalue()
    write_switchyard_toml(load_switchyard_route(catalog), tmp_path / "other.toml")
    assert (tmp_path / "other.toml").exists()


def test_rejects_unknown_picker():
    content = _catalog().replace('picker = "efficient_first"', 'picker = "random"')
    with pytest.raises(CatalogError, match="efficient_first or capable_first"):
        load_base_route(tomllib.loads(content))


def test_rejects_escalation_without_efficient_first(tmp_path):
    content = _catalog(algorithm="escalation").replace(
        'picker = "efficient_first"',
        'picker = "capable_first"',
    )
    with pytest.raises(CatalogError, match="efficient_first"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_judge_on_stage_router(tmp_path):
    extra = (
        "\n[base.targets.judge]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'protocols = ["chat_completions"]\n'
    )
    with pytest.raises(CatalogError, match="does not use base.targets.judge"):
        load_switchyard_route(_write(tmp_path, _catalog(extra=extra)))


def test_rejects_unknown_target_role(tmp_path):
    extra = (
        "\n[base.targets.mid]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'protocols = ["chat_completions"]\n'
    )
    with pytest.raises(CatalogError, match="unknown roles"):
        load_switchyard_route(_write(tmp_path, _catalog(extra=extra)))


def test_rejects_unknown_base_keys(tmp_path):
    content = _catalog().replace(
        "confidence_threshold = 0.5\n",
        "confidence_threshold = 0.5\ncomplexity_threshold = 3\n",
        1,
    )
    with pytest.raises(CatalogError, match="unknown keys"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_missing_efficient_target():
    content = _catalog().replace(
        "[base.targets.efficient]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n"
        'protocols = ["chat_completions", "responses"]\n\n',
        "",
        1,
    )
    with pytest.raises(CatalogError, match="efficient and capable"):
        load_base_route(tomllib.loads(content))


def test_capable_first_stage_router(tmp_path):
    content = _catalog().replace('picker = "efficient_first"', 'picker = "capable_first"')
    route = load_switchyard_route(_write(tmp_path, content))
    parsed = tomllib.loads(render_switchyard_toml(route))
    assert parsed["routes"]["mantis_base"]["picker"] == "capable_first"


def test_omits_extra_body_when_target_has_no_caps(tmp_path):
    content = _catalog().replace(
        '[base.targets.efficient]\n'
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n"
        'protocols = ["chat_completions", "responses"]\n',
        '[base.targets.efficient]\n'
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'protocols = ["chat_completions"]\n',
        1,
    )
    route = load_switchyard_route(_write(tmp_path, content))
    parsed = tomllib.loads(render_switchyard_toml(route))
    assert "extra_body" not in parsed["targets"]["efficient"]
    assert parsed["targets"]["capable"]["extra_body"]["max_tokens"] == 128000


def test_rejects_missing_catalog(tmp_path):
    missing = tmp_path / "missing.toml"
    with pytest.raises(CatalogError, match="does not exist"):
        load_switchyard_route(missing)


def test_rejects_catalog_without_base(tmp_path):
    path = _write(tmp_path, "version = 1\n")
    with pytest.raises(CatalogError, match="no \\[base\\] route"):
        load_switchyard_route(path)


def test_rejects_invalid_toml(tmp_path):
    path = _write(tmp_path, "version = [\n")
    with pytest.raises(CatalogError, match="invalid TOML"):
        load_switchyard_route(path)


def test_cli_reports_catalog_errors(tmp_path, capsys):
    catalog = _write(tmp_path, "version = 1\n")
    assert main(["validate", "--catalog", str(catalog)]) == 1
    assert "no [base] route" in capsys.readouterr().err


def test_escalation_without_judge_uses_efficient(tmp_path):
    catalog = _write(tmp_path, _catalog(algorithm="escalation"))
    parsed = tomllib.loads(render_switchyard_toml(load_switchyard_route(catalog)))
    assert parsed["routes"]["mantis_base"]["classifier_target"] == "efficient"
    assert "judge" not in parsed["targets"]


def test_rejects_anthropic_format_on_openai_adapter(tmp_path):
    content = _catalog().replace(
        'protocols = ["chat_completions", "responses"]\n\n'
        "[base.targets.capable]",
        'protocols = ["chat_completions", "responses"]\n'
        'format = "anthropic_messages"\n\n'
        "[base.targets.capable]",
        1,
    )
    with pytest.raises(CatalogError, match="requires the anthropic adapter"):
        load_switchyard_route(_write(tmp_path, content))
