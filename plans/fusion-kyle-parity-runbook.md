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

## Phase 2 — Main-lane parity via profiles — DONE

- Structured runs get `STRUCTURED_PLANNING_SUFFIX` on the main preamble:
  delegate all execution (including the hardest task) to lanes, reserve main
  for planning/review, assign `frontier` to the hardest task. Profile roster
  appended when run/catalog profiles exist. Legacy preamble byte-identical.
- `_resolve_profile("frontier")` falls back to a built-in profile bound to the
  run's pinned main slot.
- Plan deviation (audit catch): do NOT add `[[fusion.worker_profiles]]` to the
  shipped catalog — the structured gate includes
  `bool(coordinator.worker_profiles)`, so a catalog profile would flip every
  existing run from legacy to structured. Built-in fallback achieves the same
  with zero behavior change.
- Test-double fix: SequenceWorker/FakeWorker now route on
  `startswith(MAIN_PREAMBLE)` since structured preambles carry a suffix.
- Tests: preamble structured-vs-legacy, frontier profile uses main slot.
  66 fusion tests green; 242 gate green.

## Phase 3 — Server-side tool execution — PENDING

## Phase 4 — Streaming deltas — PENDING

## Final verification — prime-agent harness multi-step coding (base + fusion) — PENDING
