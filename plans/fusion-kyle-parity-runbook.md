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

## Phase 3 — Server-side tool execution — DONE

- New `apps/api/tool_exec.py` (~230 lines, stdlib only): files/shell/code tools
  against a per-run workspace (`MANTIS_FUSION_WORKSPACE_ROOT`, default
  `.mantis/fusion`), path containment, scrubbed env, subprocess timeout,
  output capped at `RUN_MAX_MSG_BYTES`. Canonical schemas per bundle; alias
  names (`sh`, `python`, `exec`, ...) still execute.
- `FusionToolOptions.server_execution` (also flips the run to structured).
  `_filter_tools` offers server schemas for enabled bundles when the client
  did not declare them.
- Wiring in `_advance_structured`: when every pending call is server-known,
  execute inline (parallel via `ThreadPoolExecutor`, order-preserving) and
  feed results back into the lanes; mixed/unknown batches keep the client
  contract byte-for-byte.
- Audit catches fixed en route:
  - pending-collection now scans ALL lanes (a lane stepped inline holds its
    next batch; the old runnable-only collection would have dropped it);
  - loop ceiling raised by `sidekick_max_tool_rounds * lanes` since
    server-executed rounds consume iterations;
  - ruff S604: `/bin/sh -c` argv instead of `shell=True`.
- ponytail ceiling recorded in code: cwd jail + scrubbed env is not
  tenant-grade isolation; container backend when untrusted callers opt in.
- Tests: tool unit coverage, escape/unknown rejection, parallel+ordered
  execution, end-to-end server run with no client round-trip, mixed-batch
  suspension. 71 fusion tests green; 247 gate green.

## Phase 4 — Streaming deltas — DONE

- `providers._assemble_streamed_completion` now tees
  `provider.output.delta` / `provider.reasoning.delta` into the existing
  progress event stream (sink-gated, upstream-streamed calls only). No new
  endpoint or protocol: SSE clients see deltas through the same channel as
  other progress events.
- Audit catch fixed: `api._stream.emit` sequence increment was non-atomic;
  parallel lanes + delta volume could duplicate/lose sequence numbers. Now
  guarded by a lock.
- Test: assembly emits both delta types and assembles identical content.
  95 fusion+upstream tests green; 277 gate green (incl. parity streaming).

## Full-suite audit — DONE

- Broad run initially stalled at `test_parity_websearch`.
- `faulthandler_timeout` proved a Phase 1 self-deadlock: legacy
  `NativeRun.advance()` holds `self.lock`, then provider accounting called
  `add_usage()`, which tried to reacquire it. Fusion-only gates did not expose
  this legacy path.
- Fixed with a dedicated picklable `usage_lock`; no `RLock` or wider lock
  semantic change.
- Installed already-declared optional test dependencies (`datasets`, `boto3`)
  into the project venv. Entire suite: **521 passed**, one upstream Starlette
  deprecation warning.

## Final verification — prime-agent harness multi-step coding (base + fusion) — DONE

Environment:
- current checkout served on isolated `127.0.0.1:8090` through
  `scripts/run_mantis_native.sh` (catalog render included), reusing healthy
  Bifrost `:8080` and Switchyard `:5500`;
- real `/opt/homebrew/bin/prime-agent`, isolated
  `PRIME_AGENT_CODING_AGENT_DIR`, custom OpenAI-compatible provider;
- only the real built-in `ipython` tool enabled.

Base session:
- prompt required create `base_probe.txt`, then a separate read-back tool call,
  then exact `BASE_SESSION_DONE`;
- Prime Agent made 3 `/v1/chat/completions` turns;
- file content exactly `BASE_ROUTER_OK\n`; final exactly
  `BASE_SESSION_DONE`.

Fusion session:
- first production-equivalent run uncovered a strict-provider bug: lead emitted
  a stray planning tool call while planning tools were disabled; Fusion retried
  with a reminder but left the assistant call unpaired, so Azure rejected the
  transcript.
- fix: append synthetic error tool results for every rejected planning call
  before retry. Regression test asserts assistant-call → tool-result order.
- replay through real Prime Agent made 3 completion turns / 2 client tool calls;
  encoded Fusion run/tool IDs resumed correctly;
- file content exactly `FUSION_ROUTER_OK\n`; final exactly
  `FUSION_SESSION_DONE`.

Final gates:
- all tests, including optional training files: **521 passed**;
- ruff on every touched Python file: clean;
- one existing Starlette/httpx deprecation warning only.
