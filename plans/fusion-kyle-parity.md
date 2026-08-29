# Fusion parity with Kylejeong2/fusion — including the architectural wins

**Status:** design only. Ponytail rules apply: shortest diff that holds, stdlib
first, no abstraction with one implementation. Every phase is independently
shippable and behind the existing structured-path gate.

## What "parity" means here

Not a port. Kyle's repo is a TS in-process engine with a web app and an
OpenCode plugin; mantis is an OpenAI-compatible resumable gateway. Parity =
same orchestration capabilities, mantis shape. The client layer (web/TUI) is
out of scope forever — mantis already serves clients over the chat API.

Already ahead, keep: review/follow-up/escalation loop, durable resumable runs
with idempotency, token-budget context fitting, cache-aware compaction.

## YAGNI rulings (challenge before building)

1. **"Concurrent main lane" (Kyle's main executes `mainTask` while the
   sidekick works) dissolves into existing machinery.** Mantis's main plans
   and reviews; it never executes. But `FusionWorkerProfile.model` already
   overrides a lane's slot — the planner can assign the hard task to a lane
   bound to the frontier slot. That *is* Kyle's main lane, minus a new class.
   Cost: one sentence in the planner preamble + one catalog worker_profile.
   **Do not build a MainLane.**
2. **No Modal, no Docker, no new dependency.** Server-side execution v1 is
   stdlib `pathlib` + `subprocess` against a per-run workspace. The 8 tool
   names it must implement already exist in `utils._BUNDLE_TOOLS`
   (files/shell/code bundles) — the same surface as Kyle's
   `ExecutionEnvironment`. Container isolation: add when untrusted tenants
   exist, not before.
3. **No `ExecutionEnvironment` interface.** One implementation = functions in
   one module. Extract an interface when a second backend (container) lands.
4. **No AbortController port.** Per-call httpx timeout derived from the budget
   deadline bounds an over-budget run well enough.

## Phase 1 — Parallel lane stepping (~40 lines + tests)

Where: `FusionRun._advance_structured`, the `sidekick_pending` loop.

- Step incomplete lanes through a `ThreadPoolExecutor(max_workers=len(lanes))`
  instead of the sequential `for` loop. `_call_worker` is blocking network
  I/O, so threads give real overlap with zero async rewrite.
- Add one `threading.Lock` on `FusionRun` covering `add_usage`,
  `record_activity`, `record_tool_results`; one on `FusionBudgetGuard`
  covering `consume_*`. `# ponytail: global run lock; per-lane locks if
  contention ever shows up in profiles.`
- Lane messages and `pending_tool_calls` are already per-lane; no change.
- Tests: two fake lanes whose slots block on a `threading.Barrier(2)` —
  passes only if stepped concurrently; budget/usage totals unchanged.

Ceiling (named): lanes still serialize at `awaiting_tools` suspension, because
the client executes tools. Real overlap across the whole run needs Phase 3.

## Phase 2 — Main-lane parity via profiles (~10 lines)

- Planner preamble (structured path): add "Delegate execution work — including
  the hard integration task — to sidekick lanes. Reserve main for planning and
  review. Assign `profile: "frontier"` to the task that needs the strongest
  model."
- Ship a `frontier` worker_profile in `config/catalog.toml` whose `model` is
  the main slot. With Phase 1, that lane runs concurrently with the cheap
  lanes. This *is* the hard-win #5, at a profile's price.

## Phase 3 — Server-side tool execution (~180 lines, the real win)

New file `apps/api/tool_exec.py`, stdlib only:

- `WORKSPACE_ROOT / run_id` per-run dir, created lazily on first call.
- Implement exactly the `_BUNDLE_TOOLS["files" | "shell" | "code"]` names:
  `list_files`/`read_file`/`search_files`/`write_file`/`edit_file` via
  `pathlib` + `fnmatch`; `bash`/`execute_code` via
  `subprocess.run(cwd=workspace, timeout=…, env={})`, stdout/stderr capped at
  `utils.RUN_MAX_MSG_BYTES` (reuse the existing cap + `prune_tool_result`).
- Path containment: resolve and require `workspace in path.parents`.
  Trust boundary — no shortcuts here.
- Wire-in: one opt-in flag on `FusionToolOptions` (e.g. `server_execution:
  bool`). In `advance`, when a lane's pending calls are all server-executable
  and the flag is on, execute inline and loop `lane.step(results)` instead of
  suspending. Flag off = today's client contract, byte-for-byte.
- `# ponytail: cwd jail + scrubbed env is not a tenant-grade sandbox; move to
  a container backend when runs come from untrusted callers.`

This is what makes Phase 1's overlap real: server-runnable calls never
suspend, so lanes run end-to-end in parallel and the client gets one response.

## Phase 4 — Streaming deltas (optional, last)

`providers._stream_completion` already consumes upstream SSE and assembles the
result; `api.py` already emits progress SSE. Tee text deltas into the existing
progress event stream during `_call_worker`. No new endpoint, no new protocol.

## Explicitly skipped

- Modal/Docker/any sandbox dependency — add when untrusted tenants exist.
- MainLane class / state-machine redesign — profiles cover it (Phase 2).
- AbortController — per-call httpx timeout from the budget deadline.
- Web app, OpenCode plugin, TUI — client layer, not the gateway.
- Event-sourced durable store — mantis run persistence already beats it.

## Order and why

1 → 2 → 3 → 4. Phase 1 is useless without something to overlap (Phase 3 makes
it end-to-end), Phase 3 is safe without Phase 1 (sequential first). Each phase
lands behind the structured gate with its own tests in `tests/test_fusion.py`;
the legacy path is untouched throughout.
