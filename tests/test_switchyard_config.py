"""Catalog-driven Switchyard config generation for mantis/base."""

from __future__ import annotations

import io
import tomllib
from pathlib import Path

import pytest
from model_catalog_schema import CatalogError, load_base_route
from switchyard_config import (
    SWITCHYARD_ROUTE_ID,
    load_switchyard_route,
    main,
    render_switchyard_toml,
    write_switchyard_toml,
)


def _catalog(*, extra: str = "") -> str:
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
        'picker = "efficient_first"\n'
        "confidence_threshold = 0.5\n"
        "recent_turn_window = 3\n\n"
        "[base.targets.efficient]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n\n"
        "[base.targets.capable]\n"
        'provider = "bifrost"\n'
        'upstream_model = "anthropic/claude-opus-5"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 128000\n"
        f"{extra}"
    )


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "catalog.toml"
    path.write_text(content)
    return path


def test_shipped_catalog_renders_stage_router():
    catalog_path = Path("config/catalog.toml")
    route = load_switchyard_route(catalog_path)
    text = render_switchyard_toml(route)
    parsed = tomllib.loads(text)

    with catalog_path.open("rb") as handle:
        catalog = tomllib.load(handle)
    base = catalog["base"]
    providers = catalog["providers"]

    assert parsed["schema_version"] == 1
    assert parsed["routes"]["mantis_base"]["type"] == "stage_router"
    assert parsed["routes"]["mantis_base"]["id"] == SWITCHYARD_ROUTE_ID
    assert parsed["routes"]["mantis_base"]["picker"] == base["picker"]
    assert parsed["routes"]["mantis_base"]["confidence_threshold"] == base["confidence_threshold"]
    assert parsed["routes"]["mantis_base"]["recent_turn_window"] == base["recent_turn_window"]

    # catalog.toml is the single source of truth — derive expected values from it
    for role in ("efficient", "capable"):
        target = base["targets"][role]
        provider = providers[target["provider"]]
        rendered_target = parsed["targets"][role]
        rendered_client = parsed["llm_clients"][rendered_target["llm_client"]]

        assert rendered_target["id"] == target["upstream_model"]
        assert rendered_client["api_key_env"] == provider["credential_env"]
        assert rendered_client["base_url"] == provider["base_url"]


def test_openrouter_free_smoke_catalog_renders_distinct_targets():
    route = load_switchyard_route(Path("config/catalog.openrouter-free.toml"))
    parsed = tomllib.loads(render_switchyard_toml(route))
    assert parsed["targets"]["efficient"]["id"] == "openrouter/free"
    assert parsed["targets"]["capable"]["id"] == "stealth/ox-alpha"
    assert parsed["llm_clients"]["openrouter"]["base_url"] == "https://openrouter.ai/api/v1"
    assert parsed["llm_clients"]["openrouter"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert parsed["routes"]["mantis_base"]["id"] == SWITCHYARD_ROUTE_ID
    assert parsed["routes"]["mantis_base"]["picker"] == "efficient_first"


def test_render_quotes_dotted_provider_names(tmp_path):
    content = _catalog().replace(
        '[base.targets.efficient]\n'
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
        'reasoning_effort = "medium"\n'
        "max_tokens = 65536\n",
        '[base.targets.efficient]\n'
        'provider = "modal.prod"\n'
        'upstream_model = "vendor/fast"\n',
        1,
    )
    path = _write(tmp_path, content)
    text = render_switchyard_toml(load_switchyard_route(path))
    parsed = tomllib.loads(text)
    assert parsed["llm_clients"]["modal_prod"]["base_url"] == "https://modal.example.test/v1"
    assert parsed["targets"]["efficient"]["llm_client"] == "modal_prod"
    assert parsed["targets"]["capable"]["llm_client"] == "bifrost"


def test_rejects_identical_efficient_and_capable(tmp_path):
    content = (
        _catalog()
        .replace(
            'upstream_model = "anthropic/claude-opus-5"\n',
            'upstream_model = "google/gemini-3.7-flash"\n',
            1,
        )
        .replace(
            "max_tokens = 128000\n",
            "max_tokens = 65536\n",
            1,
        )
    )
    path = _write(tmp_path, content)
    with pytest.raises(CatalogError, match="distinct"):
        load_switchyard_route(path)


def test_accepts_same_model_with_different_reasoning_effort(tmp_path):
    content = (
        _catalog()
        .replace(
            'upstream_model = "anthropic/claude-opus-5"\n',
            'upstream_model = "google/gemini-3.7-flash"\n',
            1,
        )
        .replace(
            'reasoning_effort = "medium"\nmax_tokens = 128000\n',
            'reasoning_effort = "high"\nmax_tokens = 65536\n',
            1,
        )
    )
    path = _write(tmp_path, content)
    route = load_switchyard_route(path)
    text = render_switchyard_toml(route)
    parsed = tomllib.loads(text)
    assert parsed["targets"]["efficient"]["id"] == "google/gemini-3.7-flash"
    assert parsed["targets"]["capable"]["id"] == "google/gemini-3.7-flash"
    assert parsed["targets"]["efficient"]["extra_body"]["reasoning_effort"] == "medium"
    assert parsed["targets"]["capable"]["extra_body"]["reasoning_effort"] == "high"


def test_rejects_inverted_reasoning_effort_for_same_model(tmp_path):
    content = (
        _catalog()
        .replace(
            'upstream_model = "anthropic/claude-opus-5"',
            'upstream_model = "google/gemini-3.7-flash"',
            1,
        )
        .replace(
            'reasoning_effort = "medium"\nmax_tokens = 65536\n',
            'reasoning_effort = "high"\nmax_tokens = 65536\n',
            1,
        )
        .replace(
            'reasoning_effort = "medium"\nmax_tokens = 128000\n',
            'reasoning_effort = "low"\nmax_tokens = 65536\n',
            1,
        )
    )
    path = _write(tmp_path, content)
    with pytest.raises(CatalogError, match="must not exceed"):
        load_switchyard_route(path)


def test_cli_validate_and_render(tmp_path, monkeypatch):
    catalog = _write(tmp_path, _catalog())
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    assert main(["validate", "--catalog", str(catalog)], stdout=out) == 0
    assert SWITCHYARD_ROUTE_ID in out.getvalue()
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


def test_rejects_unknown_target_role(tmp_path):
    extra = (
        "\n[base.targets.mid]\n"
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n'
    )
    with pytest.raises(CatalogError, match="unknown roles"):
        load_switchyard_route(_write(tmp_path, _catalog(extra=extra)))


def test_rejects_unknown_base_keys(tmp_path):
    content = _catalog().replace(
        "confidence_threshold = 0.5\n",
        "confidence_threshold = 0.5\nalgorithm = \"stage_router\"\n",
        1,
    )
    with pytest.raises(CatalogError, match="unknown keys"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_legacy_protocols_on_base_target(tmp_path):
    extra = 'protocols = ["chat_completions"]\n'
    content = _catalog().replace(
        'upstream_model = "google/gemini-3.7-flash"\n',
        'upstream_model = "google/gemini-3.7-flash"\n' + extra,
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
        "max_tokens = 65536\n\n",
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
        "max_tokens = 65536\n",
        '[base.targets.efficient]\n'
        'provider = "bifrost"\n'
        'upstream_model = "google/gemini-3.7-flash"\n',
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


def test_rejects_anthropic_format_on_openai_adapter(tmp_path):
    content = _catalog().replace(
        'upstream_model = "google/gemini-3.7-flash"\n',
        'upstream_model = "google/gemini-3.7-flash"\nformat = "anthropic_messages"\n',
        1,
    )
    with pytest.raises(CatalogError, match="requires the anthropic adapter"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_unknown_switchyard_format(tmp_path):
    content = _catalog().replace(
        'upstream_model = "google/gemini-3.7-flash"\n',
        'upstream_model = "google/gemini-3.7-flash"\nformat = "openai_compat"\n',
        1,
    )
    with pytest.raises(CatalogError, match="unsupported"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_catalog_version_other_than_1():
    content = _catalog().replace("version = 1\n", "version = 2\n", 1)
    with pytest.raises(CatalogError, match="version must be 1"):
        load_base_route(tomllib.loads(content))


def test_rejects_unknown_base_provider(tmp_path):
    content = _catalog().replace(
        '[base.targets.efficient]\nprovider = "bifrost"',
        '[base.targets.efficient]\nprovider = "missing"',
        1,
    )
    with pytest.raises(CatalogError, match="unknown provider"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_confidence_outside_unit_interval(tmp_path):
    content = _catalog().replace(
        "confidence_threshold = 0.5\n", "confidence_threshold = 1.5\n"
    )
    with pytest.raises(CatalogError, match="\\[0, 1\\]"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_non_positive_turn_window(tmp_path):
    content = _catalog().replace("recent_turn_window = 3\n", "recent_turn_window = 0\n")
    with pytest.raises(CatalogError, match="positive integer"):
        load_switchyard_route(_write(tmp_path, content))


def test_rejects_catalog_directory(tmp_path):
    with pytest.raises(CatalogError, match="is not a file"):
        load_switchyard_route(tmp_path)


def test_quotes_non_identifier_client_keys(tmp_path):
    content = (
        "version = 1\n\n"
        '[providers."edge-fast"]\n'
        'adapter = "openai-compatible"\n'
        'base_url = "https://edge.example.test/v1"\n'
        'credential_env = "EDGE_KEY"\n'
        'protocols = ["chat_completions"]\n\n'
        "[providers.bifrost]\n"
        'adapter = "openai-compatible"\n'
        'base_url = "http://127.0.0.1:8080/v1"\n'
        'credential_env = "BIFROST_API_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n\n'
        "[base]\n"
        'picker = "efficient_first"\n'
        "confidence_threshold = 0.5\n\n"
        "[base.targets.efficient]\n"
        'provider = "edge-fast"\n'
        'upstream_model = "vendor/fast"\n\n'
        "[base.targets.capable]\n"
        'provider = "bifrost"\n'
        'upstream_model = "anthropic/claude-opus-5"\n'
    )
    text = render_switchyard_toml(load_switchyard_route(_write(tmp_path, content)))
    assert '[llm_clients."edge-fast"]' in text
    parsed = tomllib.loads(text)
    assert parsed["targets"]["efficient"]["llm_client"] == "edge-fast"
    assert parsed["routes"]["mantis_base"]["id"] == SWITCHYARD_ROUTE_ID


def test_shipped_base_route_is_efficient_first():
    route = load_switchyard_route(Path("config/catalog.toml"))
    assert route.picker == "efficient_first"
    assert 'picker = "efficient_first"' in render_switchyard_toml(route)
