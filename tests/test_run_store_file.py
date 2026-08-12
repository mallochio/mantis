"""File-backed run store (MANTIS_RUN_STORE=file): durable, daemon-free."""

import time

import pytest
import serve_config

import runs


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(runs, "RUN_STORE", "file")
    monkeypatch.setenv("MANTIS_RUN_DIR", str(tmp_path))
    return tmp_path


def _register() -> runs.NativeRun:
    run = runs.NativeRun(f"{time.time_ns():032x}"[:32])
    run.kind = "trinity"
    runs._register_run(run)
    return run


def test_put_get_roundtrip_and_permissions(store):
    run = _register()
    path = store / f"{run.run_id}.pkl"
    assert path.exists()
    assert path.stat().st_mode & 0o777 == 0o600
    got = runs.get_run(run.run_id)
    assert got.run_id == run.run_id
    assert got.kind == "trinity"


def test_survives_process_restart(store):
    """The store holds no in-memory cache: every get reads from disk, so a
    restarted API process finds the previous run exactly where it stopped."""
    run = _register()
    run.in_flight = 1
    runs._file_put(run)  # as advance_run writes at a boundary
    got = runs.get_run(run.run_id)  # fresh object, straight from disk
    assert got.run_id == run.run_id
    assert got.in_flight == 0  # restart voids stale in-flight marks


def test_unknown_run_raises(store):
    with pytest.raises(KeyError, match="unknown or expired run"):
        runs.get_run("0" * 32)


def test_duplicate_id_rejected(store):
    run = _register()
    dupe = runs.NativeRun(run.run_id)
    with pytest.raises(ValueError, match="already exists"):
        runs._register_run(dupe)


def test_delete_removes_file(store):
    run = _register()
    assert runs.delete_run(run.run_id) is True
    assert not (store / f"{run.run_id}.pkl").exists()
    assert runs.delete_run(run.run_id) is False


def test_sweep_expires_idle_runs(store, monkeypatch):
    run = _register()
    run.last_active = time.time() - serve_config.RUN_TTL - 1
    runs._file_put(run)
    monkeypatch.setenv("MANTIS_LEARNING", "0")
    runs._sweep_runs()
    assert not (store / f"{run.run_id}.pkl").exists()


def test_sweep_keeps_recent_runs(store):
    run = _register()
    runs._sweep_runs()
    assert (store / f"{run.run_id}.pkl").exists()


def test_capacity_raises_on_full_pool(store, monkeypatch):
    monkeypatch.setattr(runs, "MAX_RUNS", 1)
    _register()
    with pytest.raises(runs.providers.RunCapacityError):
        _register()


def test_memory_and_redis_modes_untouched(monkeypatch):
    monkeypatch.setattr(runs, "RUN_STORE", "memory")
    monkeypatch.setenv("MANTIS_RUN_DIR", "/nonexistent-dir")
    run = runs.NativeRun("a" * 32)
    runs._register_run(run)
    assert runs.get_run("a" * 32) is run
    serve_config._runs.pop("a" * 32)
