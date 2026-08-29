# Fusion parity runbook

Tracks execution of `fusion-kyle-parity.md`. Updated after every stage.

## Baseline

- `tests/test_fusion.py`: 62 passed.
- Full suite cannot collect `test_retrain_conductor.py` / `test_retrain_router_pool.py`
  (optional `datasets` dependency absent) — pre-existing, unrelated. API/router subset
  used as the gate: test_api, test_serve, test_base_proxy, test_model_catalog,
  test_run_store_file, test_fusion = 240 passed.

## Phase 1 — Parallel lane stepping — DONE

- `fusion.py`: structured `sidekick_pending` loop steps lanes via
  `ThreadPoolExecutor`; propagates the thread-local progress sink into lane
  threads (audit catch: `_history_context` is `threading.local`, events would
  have vanished).
- `fusion_budget.py`: `threading.Lock` around `consume_turn`/`consume_tokens`;
  `__getstate__`/`__setstate__` so budget survives run pickling.
- `runs.py`: `add_usage` wrapped in the existing `NativeRun.lock` (compound
  dict updates race otherwise).
- Tests: `test_fusion_structured_lanes_step_in_parallel` (barrier rendezvous,
  fails if sequential), `test_fusion_budget_guard_pickle_roundtrip`.
- Audit: ruff clean; 240/240 gate green.

## Phase 2 — Main-lane parity via profiles — PENDING

## Phase 3 — Server-side tool execution — PENDING

## Phase 4 — Streaming deltas — PENDING

## Final verification — prime-agent harness multi-step coding (base + fusion) — PENDING
