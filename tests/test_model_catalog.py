"""Tests for the shared Mantis catalog, anchored ABI, and stable slot resolver."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import model_catalog
import model_catalog_runtime
import pytest
import serve
import tomllib

REPO = Path(__file__).resolve().parent.parent
ABI = model_catalog.load_abi_manifest()
SLOTS = tuple(ABI.slot_order)
CONDUCTOR = ABI.conductor
CONDUCTOR_IDENTITY = ABI.workers[CONDUCTOR][0]
CONDUCTOR_EFFORT = ABI.workers[CONDUCTOR][1] or "medium"
DEEPSEEK = SLOTS[3]
DEEPSEEK_IDENTITY = ABI.workers[DEEPSEEK][0]


def _catalog(contract: str = "") -> str:
    abi = model_catalog.load_abi_manifest()
    workers = []
    for index, slot in enumerate(abi.slot_order):
        provider = "router" if index != 3 else "code"
        model = f"vendor/{slot}"
        protocols = (
            '["responses"]' if slot == abi.conductor else '["chat_completions"]'
        )
        identity, effort = abi.workers[slot]
        lines = [
            f"[mantis.workers.{slot}]",
            f'provider = "{provider}"',
            f'upstream_model = "{model}"',
            f'model_identity = "{identity}"',
            f"protocols = {protocols}",
        ]
        if effort:
            lines.append(f'reasoning_effort = "{effort}"')
        workers.append("\n".join(lines))
    expected = f'trained_slot_contract = "{contract}"\n' if contract else ""
    return (
        "version = 1\n\n"
        "[providers.router]\n"
        'adapter = "openrouter"\n'
        'base_url = "https://router.example.test/v1"\n'
        'credential_env = "ROUTER_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n\n'
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["chat_completions"]\n\n'
        "[mantis]\n"
        f"slot_order = {json.dumps(list(abi.slot_order))}\n"
        f'conductor = "{abi.conductor}"\n'
        f"{expected}\n"
        + "\n".join(workers)
    )


def _shared_catalog(contract: str) -> str:
    """Combined fixture for the Mantis and llm-router (RouteLLM) consumers."""
    abi = model_catalog.load_abi_manifest()
    provider_order = [
        "modal.prod", "generic.openai", "code", "gateway",
        "router", "generic.openai", "generic.openai",
    ]
    # The conductor slot speaks responses; route it through a responses-capable
    # provider (openrouter) regardless of its position in the slot order.
    provider_order[abi.slot_order.index(abi.conductor)] = "router"
    provider_for = dict(zip(abi.slot_order, provider_order, strict=True))
    workers = []
    for slot in abi.slot_order:
        provider = provider_for.get(slot, "router")
        protocols = '["responses"]' if slot == abi.conductor else '["chat_completions"]'
        identity, effort = abi.workers[slot]
        lines = [
            f"[mantis.workers.{slot}]",
            f'provider = "{provider}"',
            f'upstream_model = "vendor/{slot}"',
            f'model_identity = "{identity}"',
            f"protocols = {protocols}",
        ]
        if effort:
            lines.append(f'reasoning_effort = "{effort}"')
        workers.append("\n".join(lines))
    return (
        "version = 1\n\n"
        "[providers.router]\n"
        'adapter = "openrouter"\n'
        'base_url = "https://router.example.test/v1"\n'
        'credential_env = "ROUTER_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n\n'
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["chat_completions"]\n\n'
        "[providers.\"modal.prod\"]\n"
        'adapter = "modal"\n'
        'base_url = "https://modal.example.test/v1"\n'
        'credential_env = "MODAL_KEY"\n\n'
        "[providers.gateway]\n"
        'adapter = "cloudflare-gateway"\n'
        'base_url = "https://gateway.example.test/v1"\n'
        'credential_env = "GATEWAY_KEY"\n\n'
        "[providers.\"generic.openai\"]\n"
        'adapter = "openai-compatible"\n'
        'base_url = "https://generic.example.test/v1"\n'
        'credential_env = "GENERIC_KEY"\n\n'
        "[routellm]\n"
        'active_policy = "coding"\n\n'
        "[routellm.targets.low]\n"
        'provider = "generic.openai"\n'
        'upstream_model = "vendor/low"\n'
        'reasoning_effort = "none"\n'
        'protocols = ["chat_completions"]\n'
        "rank = 0\n\n"
        "[routellm.targets.mid]\n"
        'provider = "modal.prod"\n'
        'upstream_model = "vendor/mid"\n'
        'protocols = ["chat_completions"]\n'
        "rank = 1\n\n"
        "[routellm.targets.work]\n"
        'provider = "router"\n'
        'upstream_model = "vendor/work"\n'
        'protocols = ["chat_completions"]\n'
        "rank = 2\n\n"
        "[routellm.targets.responses]\n"
        'provider = "gateway"\n'
        'upstream_model = "openai/responses"\n'
        'protocols = ["chat_completions", "responses"]\n'
        "rank = 3\n\n"
        "[routellm.targets.safe]\n"
        'provider = "gateway"\n'
        'upstream_model = "openai/safe"\n'
        'protocols = ["chat_completions", "responses"]\n'
        "rank = 4\n\n"
        "[routellm.policies.coding]\n"
        'complexity_targets = ["low", "mid", "work", "responses", "safe"]\n'
        'invalid_complexity_target = "safe"\n'
        "[mantis]\n"
        f"slot_order = {json.dumps(list(abi.slot_order))}\n"
        f'conductor = "{abi.conductor}"\n'
        f'trained_slot_contract = "{contract}"\n'
        + "\n".join(workers)
    )


def _write_catalog(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "catalog.toml"
    path.write_text(content)
    return path


def _fingerprint(tmp_path: Path, content: str | None = None) -> tuple[Path, str]:
    path = _write_catalog(tmp_path, content or _catalog())
    catalog = model_catalog.load_mantis_catalog(path, require_contract=False)
    assert catalog is not None
    return path, model_catalog.abi_contract(model_catalog.load_abi_manifest())


def _full_workers(provider: str = "edge") -> dict:
    workers = {}
    for slot in SLOTS:
        identity, effort = ABI.workers[slot]
        entry = {
            "provider": provider,
            "upstream_model": f"vendor/{slot}",
            "model_identity": identity,
            "protocols": ["responses"] if slot == CONDUCTOR else ["chat_completions"],
        }
        if effort:
            entry["reasoning_effort"] = effort
        workers[slot] = entry
    return workers


def _edge_provider() -> dict:
    return {
        "edge": {
            "adapter": "openrouter",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }


def _set_runtime_env(monkeypatch, providers: dict, workers: dict, contract: str = "") -> None:
    monkeypatch.setenv("MANTIS_PROVIDER_BINDINGS", json.dumps(providers))
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", json.dumps(workers))
    monkeypatch.setenv("MANTIS_WORKER_MODELS", ",".join(SLOTS))
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", CONDUCTOR)
    if contract:
        monkeypatch.setenv("MANTIS_IDENTITY_CONTRACT", contract)


def _unchecked_catalog(content: str) -> model_catalog.MantisCatalog:
    root = tomllib.loads(content)
    section = root["mantis"]
    slots = model_catalog._slot_order(section["slot_order"])
    conductor = section["conductor"]
    bindings = model_catalog._runtime_bindings(root["providers"], section["workers"])
    return model_catalog.MantisCatalog(Path("unchecked"), bindings, slots, conductor, None)


def test_catalog_requires_version_and_contract(tmp_path):
    path = _write_catalog(tmp_path, _catalog().replace("version = 1\n", ""))
    with pytest.raises(model_catalog.CatalogError, match="version"):
        model_catalog.load_mantis_catalog(path)
    path = _write_catalog(tmp_path, _catalog())
    with pytest.raises(model_catalog.CatalogError, match="trained_slot_contract"):
        model_catalog.load_mantis_catalog(path)


def test_fingerprint_bootstraps_then_validate_and_render(tmp_path):
    path, fingerprint = _fingerprint(tmp_path)
    path.write_text(_catalog(fingerprint))
    catalog = model_catalog.load_mantis_catalog(path)
    assert catalog is not None
    rendered = model_catalog.render_mantis_environment(catalog)
    assert rendered["MANTIS_WORKER_MODELS"] == ",".join(SLOTS)
    assert rendered["MANTIS_CONDUCTOR_MODEL"] == CONDUCTOR
    assert rendered["MANTIS_IDENTITY_CONTRACT"] == model_catalog.identity_fingerprint(catalog)


def test_endpoint_provider_or_transport_model_swap_preserves_explicit_identity(tmp_path):
    _, fingerprint = _fingerprint(tmp_path)
    baseline_path = _write_catalog(tmp_path, _catalog(fingerprint))
    baseline = model_catalog.identity_fingerprint(
        model_catalog.load_mantis_catalog(baseline_path, require_contract=False)
    )
    changed = _catalog(fingerprint).replace(
        "https://router.example.test/v1", "https://new-router.example.test/v1"
    ).replace('provider = "router"', 'provider = "alternate"', 1).replace(
        f'upstream_model = "vendor/{CONDUCTOR}"', 'upstream_model = "endpoint/luna-alias"', 1
    )
    changed += (
        "\n[providers.alternate]\n"
        'adapter = "openai-compatible"\n'
        'base_url = "https://alternate.example.test/v1"\n'
        'credential_env = "ALTERNATE_KEY"\n'
        'protocols = ["chat_completions", "responses"]\n'
    )
    path = _write_catalog(tmp_path, changed)
    catalog = model_catalog.load_mantis_catalog(path)
    assert catalog is not None
    assert model_catalog.identity_fingerprint(catalog) == baseline


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (f'model_identity = "{CONDUCTOR_IDENTITY}"', 'model_identity = "logical/replaced"'),
        (f'reasoning_effort = "{CONDUCTOR_EFFORT}"', 'reasoning_effort = "low"'),
        (f'conductor = "{CONDUCTOR}"', f'conductor = "{SLOTS[0]}"'),
        (f'"{CONDUCTOR}", "{SLOTS[2]}"', f'"{SLOTS[2]}", "{CONDUCTOR}"'),
    ],
)
def test_worker_contract_mutations_fail(tmp_path, old, new):
    _, fingerprint = _fingerprint(tmp_path)
    path = _write_catalog(tmp_path, _catalog(fingerprint).replace(old, new, 1))
    with pytest.raises(model_catalog.CatalogError, match="retraining"):
        model_catalog.load_mantis_catalog(path)


def test_slot_count_mutation_fails(tmp_path):
    _, fingerprint = _fingerprint(tmp_path)
    path = _write_catalog(
        tmp_path,
        _catalog(fingerprint).replace(f', "{SLOTS[-1]}"]', "]", 1),
    )
    with pytest.raises(model_catalog.CatalogError, match="exactly seven"):
        model_catalog.load_mantis_catalog(path)


def test_worker_protocols_are_explicit(tmp_path):
    _, fingerprint = _fingerprint(tmp_path)
    path = _write_catalog(
        tmp_path,
        _catalog(fingerprint).replace(
            'protocols = ["responses"]\nreasoning_effort', "reasoning_effort"
        ),
    )
    with pytest.raises(model_catalog.CatalogError, match="explicit"):
        model_catalog.load_mantis_catalog(path)


def test_catalog_path_prefers_shared_config_name(tmp_path):
    path = tmp_path / "shared.toml"
    selected, explicit = model_catalog.catalog_path({"AI_ROUTING_CONFIG": str(path)})
    assert selected == path
    assert explicit
    legacy, explicit = model_catalog.catalog_path({"MANTIS_CATALOG_PATH": str(path)})
    assert legacy == path
    assert explicit


def test_catalog_credentials_are_required_and_not_rendered(tmp_path):
    path, fingerprint = _fingerprint(tmp_path)
    path.write_text(_catalog(fingerprint))
    catalog = model_catalog.load_mantis_catalog(path)
    assert catalog is not None
    marker = "credential-must-not-render"
    with pytest.raises(model_catalog.CatalogError, match="ROUTER_KEY") as error:
        model_catalog.resolve_provider_keys(catalog, {"CODE_KEY": marker})
    assert marker not in str(error.value)
    rendered = model_catalog.render_mantis_environment(catalog)
    assert marker not in json.dumps(rendered)


def test_omitted_model_identity_cannot_diverge_from_manifest(tmp_path):
    source = _catalog().replace(f'model_identity = "{CONDUCTOR_IDENTITY}"\n', "", 1)
    path = _write_catalog(tmp_path, source)
    with pytest.raises(model_catalog.CatalogError, match="retraining"):
        model_catalog.load_mantis_catalog(path, require_contract=False)


def test_regenerated_contract_cannot_bypass_manifest(tmp_path):
    """A self-consistent regenerated contract still cannot change slot ABI."""
    _, fingerprint = _fingerprint(tmp_path)
    mutated = _catalog(fingerprint).replace(
        f'model_identity = "{CONDUCTOR_IDENTITY}"', 'model_identity = "logical/sneaky"', 1
    )
    regenerated = model_catalog.identity_fingerprint(_unchecked_catalog(mutated))
    path = _write_catalog(tmp_path, _catalog(regenerated).replace(
        f'model_identity = "{CONDUCTOR_IDENTITY}"', 'model_identity = "logical/sneaky"', 1
    ))
    with pytest.raises(model_catalog.CatalogError, match="retraining"):
        model_catalog.load_mantis_catalog(path)


def test_manifest_anchors_repo_artifact():
    abi = model_catalog.load_abi_manifest()
    assert len(abi.slot_order) == 7
    artifact = REPO / abi.training_artifact
    assert artifact.is_file()
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == abi.training_artifact_sha256


def test_manifest_artifact_fingerprint_mismatch_rejected(tmp_path):
    artifact = tmp_path / "head.bin"
    artifact.write_bytes(b"not-the-trained-head")
    manifest = {
        "version": 1,
        "training_artifact": str(artifact),
        "training_artifact_sha256": "0" * 64,
        "slot_order": list(SLOTS),
        "conductor": CONDUCTOR,
        "workers": {
            slot: {"model_identity": f"logical/{slot}"} for slot in SLOTS
        },
    }
    path = tmp_path / "abi_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(model_catalog.CatalogError, match="fingerprint"):
        model_catalog.load_abi_manifest(path)


def test_adapter_protocol_incompatibility_rejected(tmp_path):
    # Worker asks for responses on an opencode-go adapter even though the
    # provider declares no protocols: the adapter capability check must fire.
    worker_mismatch = _catalog().replace(
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["chat_completions"]',
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"',
        1,
    ).replace(
        f'provider = "code"\nupstream_model = "vendor/{DEEPSEEK}"\n'
        f'model_identity = "{DEEPSEEK_IDENTITY}"\nprotocols = ["chat_completions"]',
        f'provider = "code"\nupstream_model = "vendor/{DEEPSEEK}"\n'
        f'model_identity = "{DEEPSEEK_IDENTITY}"\nprotocols = ["responses"]',
        1,
    )
    path = _write_catalog(tmp_path, worker_mismatch)
    with pytest.raises(model_catalog.CatalogError, match="exceeds adapter capabilities"):
        model_catalog.load_mantis_catalog(path, require_contract=False)
    provider_mismatch = _catalog().replace(
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["chat_completions"]',
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n'
        'protocols = ["responses"]',
        1,
    )
    path = _write_catalog(tmp_path, provider_mismatch)
    with pytest.raises(model_catalog.CatalogError, match="exceeds adapter capabilities"):
        model_catalog.load_mantis_catalog(path, require_contract=False)


def test_router_provider_ids_and_identifier_grammar_accepted(tmp_path):
    """Modal, openai-compatible, and cloudflare-gateway adapters are shared ids."""
    abi = model_catalog.load_abi_manifest()
    provider_order = [
        "modal.prod", "generic.openai", "code", "gateway",
        "router", "generic.openai", "generic.openai",
    ]
    # The conductor slot speaks responses; route it through a responses-capable
    # provider (openrouter) regardless of its position in the slot order.
    provider_order[abi.slot_order.index(abi.conductor)] = "router"
    provider_for = dict(zip(abi.slot_order, provider_order, strict=True))
    workers = []
    for slot in abi.slot_order:
        provider = provider_for.get(slot, "router")
        protocols = '["responses"]' if slot == abi.conductor else '["chat_completions"]'
        identity, effort = abi.workers[slot]
        lines = [
            f"[mantis.workers.{slot}]",
            f'provider = "{provider}"',
            f'upstream_model = "vendor/{slot}"',
            f'model_identity = "{identity}"',
            f"protocols = {protocols}",
        ]
        if effort:
            lines.append(f'reasoning_effort = "{effort}"')
        workers.append("\n".join(lines))
    content = (
        "version = 1\n\n"
        "[providers.router]\n"
        'adapter = "openrouter"\n'
        'base_url = "https://router.example.test/v1"\n'
        'credential_env = "ROUTER_KEY"\n\n'
        "[providers.code]\n"
        'adapter = "opencode-go"\n'
        'base_url = "https://code.example.test/v1"\n'
        'credential_env = "CODE_KEY"\n\n'
        "[providers.\"modal.prod\"]\n"
        'adapter = "modal"\n'
        'base_url = "https://modal.example.test/v1"\n'
        'credential_env = "MODAL_KEY"\n\n'
        "[providers.gateway]\n"
        'adapter = "cloudflare-gateway"\n'
        'base_url = "https://gateway.example.test/v1"\n'
        'credential_env = "GATEWAY_KEY"\n\n'
        "[providers.\"generic.openai\"]\n"
        'adapter = "openai-compatible"\n'
        'base_url = "https://generic.example.test/v1"\n'
        'credential_env = "GENERIC_KEY"\n\n'
        "[mantis]\n"
        f"slot_order = {json.dumps(list(abi.slot_order))}\n"
        f'conductor = "{abi.conductor}"\n'
        + "\n".join(workers)
    )
    path = _write_catalog(tmp_path, content)
    catalog = model_catalog.load_mantis_catalog(path, require_contract=False)
    assert catalog is not None
    adapters = {binding.adapter for binding in catalog.bindings.providers.values()}
    assert adapters == {
        "openrouter",
        "opencode-go",
        "modal",
        "openai-compatible",
        "cloudflare-gateway",
    }


def test_shared_catalog_fixture_both_consumers_parse(tmp_path, monkeypatch):
    router_home = Path.home() / ".config" / "llm-router" / "server.py"
    router_server = Path(os.environ.get("ROUTER_SERVER_PATH", str(router_home)))
    if not router_server.is_file():
        pytest.skip("llm-router consumer (server.py) is not available on this machine")
    path = _write_catalog(tmp_path, _shared_catalog(model_catalog.abi_contract(
        model_catalog.load_abi_manifest()
    )))
    catalog = model_catalog.load_mantis_catalog(path)
    assert catalog is not None
    assert {binding.adapter for binding in catalog.bindings.providers.values()} >= {
        "openrouter",
        "opencode-go",
        "modal",
        "openai-compatible",
        "cloudflare-gateway",
    }
    monkeypatch.setenv("AI_ROUTING_CONFIG", str(path))
    monkeypatch.delenv("ROUTELLM_TARGETS_JSON", raising=False)
    monkeypatch.delenv("ROUTELLM_SUPRA_TARGETS", raising=False)
    monkeypatch.delenv("ROUTELLM_SUPRA_INVALID_TARGET", raising=False)
    monkeypatch.delenv("ROUTELLM_TRAINING_LOG", raising=False)
    monkeypatch.setenv("MANTIS_DATA_DIR", str(tmp_path / "router-data"))
    for name in ("ROUTER_KEY", "CODE_KEY", "MODAL_KEY", "GATEWAY_KEY", "GENERIC_KEY"):
        monkeypatch.setenv(name, "fixture-key")
    sys.path.insert(0, str(router_server.parent))
    try:
        import importlib

        try:
            router = importlib.import_module("server")
        except Exception as error:  # noqa: BLE001 - surface the router rejection
            pytest.fail(f"router consumer rejected the shared catalog: {error}")
        assert router.TARGET_CONFIG_SOURCE == "catalog"
        assert set(router.BACKENDS) == {"low", "mid", "work", "responses", "safe"}
    finally:
        sys.path.remove(str(router_server.parent))


def test_stable_slot_resolves_to_bound_endpoint_and_model(monkeypatch):
    providers = _edge_provider()
    workers = _full_workers()
    _set_runtime_env(monkeypatch, providers, workers)
    monkeypatch.setenv("MANTIS_PROVIDER_KEYS", json.dumps({"edge": "injected-key"}))
    url, headers, body = serve._build_request(CONDUCTOR, [], 32, 0.7)
    assert url == "https://edge.example.test/v1/responses"
    assert headers["Authorization"] == "Bearer injected-key"
    assert body["model"] == f"vendor/{CONDUCTOR}"
    assert body["reasoning"] == {"effort": CONDUCTOR_EFFORT}


def test_catalog_binding_falls_back_to_named_env_for_native(monkeypatch):
    providers = {
        "edge": {
            "adapter": "opencode-go",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }
    workers = {
        slot: {
            "provider": "edge",
            "upstream_model": f"vendor/{slot}",
            "model_identity": f"logical/{slot}",
            "protocols": ["chat_completions"],
        }
        for slot in SLOTS
    }
    _set_runtime_env(monkeypatch, providers, workers)
    monkeypatch.delenv("MANTIS_PROVIDER_KEYS", raising=False)
    monkeypatch.setenv("EDGE_KEY", "native-key")
    url, headers, body = serve._build_request(DEEPSEEK, [], 32, 0.7)
    assert url == "https://edge.example.test/v1/chat/completions"
    assert headers["Authorization"] == "Bearer native-key"
    assert body["model"] == f"vendor/{DEEPSEEK}"


def test_catalog_binding_missing_credentials_is_safe(monkeypatch):
    providers = {
        "edge": {
            "adapter": "opencode-go",
            "base_url": "https://edge.example.test/v1",
            "credential_env": "EDGE_KEY",
        }
    }
    workers = {
        slot: {
            "provider": "edge",
            "upstream_model": f"vendor/{slot}",
            "model_identity": f"logical/{slot}",
            "protocols": ["chat_completions"],
        }
        for slot in SLOTS
    }
    _set_runtime_env(monkeypatch, providers, workers)
    monkeypatch.delenv("MANTIS_PROVIDER_KEYS", raising=False)
    monkeypatch.delenv("EDGE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EDGE_KEY"):
        serve._build_request(DEEPSEEK, [], 32, 0.7)


def test_runtime_bindings_enforce_seven_slots_without_contract(monkeypatch):
    providers = _edge_provider()
    workers = {
        "slot_worker": {
            "provider": "edge",
            "upstream_model": "worker",
            "model_identity": "worker",
            "protocols": ["chat_completions"],
        }
    }
    monkeypatch.setenv("MANTIS_PROVIDER_BINDINGS", json.dumps(providers))
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", json.dumps(workers))
    monkeypatch.setenv("MANTIS_WORKER_MODELS", "slot_worker")
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", "slot_worker")
    with pytest.raises(model_catalog.CatalogError, match="exactly seven"):
        model_catalog.load_runtime_bindings()


def test_runtime_bindings_reject_slot_order_mismatch(monkeypatch):
    _set_runtime_env(monkeypatch, _edge_provider(), _full_workers())
    monkeypatch.setenv(
        "MANTIS_WORKER_MODELS",
        ",".join(f"slot-{index}" for index in range(7)),
    )
    with pytest.raises(model_catalog.CatalogError, match="exactly the seven stable slots"):
        model_catalog.load_runtime_bindings()


def test_runtime_bindings_reject_identity_drift(monkeypatch):
    providers = _edge_provider()
    workers = _full_workers()
    _set_runtime_env(monkeypatch, providers, workers)
    bindings = model_catalog._runtime_bindings(
        json.loads(json.dumps(providers)), json.loads(json.dumps(workers))
    )
    contract = model_catalog_runtime._runtime_contract_hash(
        bindings, tuple(SLOTS), CONDUCTOR
    )
    _set_runtime_env(monkeypatch, providers, workers, contract=contract)
    mutated = copy.deepcopy(workers)
    mutated[CONDUCTOR]["model_identity"] = "logical/sneaky"
    monkeypatch.setenv("MANTIS_WORKER_BINDINGS", json.dumps(mutated))
    with pytest.raises(model_catalog.CatalogError, match="retraining"):
        model_catalog.load_runtime_bindings()


def test_runtime_bindings_require_conductor_slot(monkeypatch):
    _set_runtime_env(monkeypatch, _edge_provider(), _full_workers())
    monkeypatch.setenv("MANTIS_CONDUCTOR_MODEL", "slot_unknown")
    with pytest.raises(model_catalog.CatalogError, match="conductor must name"):
        model_catalog.load_runtime_bindings()


def test_runtime_binding_fingerprint_covers_mutable_bindings(monkeypatch):
    providers = _edge_provider()
    workers = _full_workers()
    _set_runtime_env(monkeypatch, providers, workers)
    baseline = model_catalog.runtime_binding_fingerprint()
    assert len(baseline) == 64
    mutated = copy.deepcopy(workers)
    mutated[CONDUCTOR]["upstream_model"] = "vendor/renamed"
    _set_runtime_env(monkeypatch, providers, mutated)
    assert model_catalog.runtime_binding_fingerprint() != baseline


def test_resolve_runtime_credentials_prefers_injected_keys(monkeypatch):
    _set_runtime_env(monkeypatch, _edge_provider(), _full_workers())
    monkeypatch.setenv("MANTIS_PROVIDER_KEYS", json.dumps({"edge": "injected"}))
    monkeypatch.setenv("EDGE_KEY", "native")
    bindings = model_catalog.load_runtime_bindings()
    assert bindings is not None
    keys = model_catalog.resolve_runtime_credentials(bindings)
    assert keys == {"edge": "injected"}
    monkeypatch.delenv("MANTIS_PROVIDER_KEYS")
    keys = model_catalog.resolve_runtime_credentials(bindings)
    assert keys == {"edge": "native"}
    monkeypatch.delenv("EDGE_KEY")
    with pytest.raises(model_catalog.CatalogError, match="EDGE_KEY"):
        model_catalog.resolve_runtime_credentials(bindings)
