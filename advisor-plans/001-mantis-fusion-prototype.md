# Plan 001: Prototype mantis-fusion as a resumable lead/sidekick orchestration endpoint

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If any STOP condition occurs, stop and report — do not improvise. When done, update the status row in `advisor-plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 7070b7f..HEAD -- apps/api/ tests/ config/`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: none
- **Category**: direction
- **Planned at**: commit `7070b7f`, 2026-08-16
- **Issue**: (none)

## Why this matters

Mantis currently has three modes: direct routing (`mantis`), multi-agent role cycling (`mantis-trinity`), and planned DAG execution (`mantis-ultra`). The lead/sidekick pattern — a frontier model delegates implementation, test, and lint work to a cheaper sidekick with persistent per-task context and `follow_up` semantics — is a distinct cost/quality contract. A minimal prototype is needed to measure whether it improves the cost/quality frontier before committing to a full production mode.

## Research basis

- `pi-fusion` (`https://github.com/JoniBR/pi-fusion`, cloned to `/var/folders/zh/_ptl4j551436df1pghnf0y5w0000gn/T/tmp.3tWZJDxMUa/pi-fusion`) defines the lead/sidekick pattern, `delegate`/`follow_up` tools, sidekick session lifecycle, cost folding, and a skill playbook (`skills/fusion/SKILL.md`, `extensions/fusion.ts`). It is a client-side plugin for `pi`/Claude Code; the Mantis prototype should replicate the *contract*, not the TypeScript implementation.
- Mantis existing modes are selected by `model` in `apps/api/serve_config.py:41-45` and implemented as coordinator/run classes in `apps/api/runs.py`. `NativeRun` (`apps/api/runs.py:361-527`) already provides resumable run state, `advance_idempotent`, usage aggregation, and storage via `MANTIS_RUN_STORE`.
- The Bifrost-driven `gpt-5.6-sol` critique (2026-08-16) concluded the idea is viable only if (a) tool resumption is an explicit, typed event protocol, (b) lead/sidekick are policy-driven roles not hard-coded models, and (c) `/v1/fusion/*` is the canonical API, with chat-completions treated as a constrained adapter. It recommended launching with client-side tools and a server-side sandbox only as a later backend option.

## Current state

- `apps/api/serve_config.py:41-45` maps `mantis-trinity` → `trinity` and `mantis-ultra` → `conductor`. No `mantis-fusion` entry exists.
- `apps/api/runs.py:361-527` defines `NativeRun` with `advance`, `advance_idempotent`, `add_usage`, and storage hooks. `get_run`/`advance_run`/`delete_run` are at `apps/api/runs.py:1261-1325`.
- `apps/api/api.py:692` defines `POST /v1/chat/completions` with `Depends(_authorize)`. No custom `/v1/fusion/*` endpoints exist.
- `apps/gateway/server.py:1-20` and `:873-918` describe the per-request Supra-Router-51M complexity gate; `apps/gateway/server.py:2977-3084` is the chat-completions handler. Fusion sits above this, routing each individual model call through the gateway.
- `config/catalog.toml:52-63` defines the `mantis` worker slot order and the `conductor` model; worker entries (`[mantis.workers.*]`) specify `provider`, `upstream_model`, `max_tokens`, `reasoning_effort`, and `protocols`.

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Install/lock | `uv sync` | exit 0, lock up to date |
| Typecheck | `uv run mypy apps/api --exclude outputs` | exit 0, no errors |
| Lint | `uv run ruff check .` | exit 0 |
| Unit tests | `uv run pytest tests/test_fusion.py -q` | all pass |
| API smoke | `uv run python -m apps.api.api` (or launch stack) | server starts on `:8088` |
| Dry-run eval | `uv run python eval/router_eval.py --manifest eval/manifests/zen-pilot-12.json --arms mantis,mantis-fusion --dry-run` | emits arm list and cap table |

## Suggested executor toolkit

- Read `apps/api/runs.py` for the `NativeRun` interface and `advance_run` lifecycle.
- Read `apps/api/trinity.py` and `apps/api/conductor.py` for existing coordinator patterns.
- Read `eval/README.md` for the harness arm convention and shadow-cost model.

## Scope

**In scope**:
- `apps/api/fusion.py` (new) — `FusionRun` and `FusionCoordinator`.
- `apps/api/api.py` — add `POST /v1/fusion/delegate`, `POST /v1/fusion/follow_up/{run_id}`, `GET /v1/fusion/status/{run_id}`, `DELETE /v1/fusion/runs/{run_id}`.
- `apps/api/serve_config.py` — add `mantis-fusion` to `MODEL_MODES` only if the chat-completions adapter is also implemented in this slice.
- `tests/test_fusion.py` (new) and any fixture files.
- `eval/fusion_manifest.json` (new) — 5-task synthetic or SWE-rebench pilot manifest.

**Out of scope** (do NOT touch):
- `apps/gateway/server.py` routing logic — Fusion uses the gateway; it does not change it.
- `apps/api/trinity.py` and `apps/api/conductor.py` internals — study for patterns, but do not refactor.
- Server-side code sandbox for tool execution; this slice uses client-side tools only.
- Full OpenAI-compatible `mantis-fusion` chat mode. If added, it is a constrained adapter and explicitly marked experimental.

## Git workflow

- Branch: `advisor/001-mantis-fusion-prototype`
- Commit per step; message style: `feat(fusion): add FusionRun state machine`, `feat(api): add /v1/fusion/delegate endpoint`, `test(fusion): cover sidekick tool resumption`.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Define the event protocol and role policy in code comments

Write the lead/sidekick state machine and event types into `apps/api/fusion.py` before implementation. The protocol must be:

- `state`: `planning | sidekick_pending | sidekick_review | awaiting_tools | completed | cancelled | error`.
- `event` types: `plan`, `sidekick_turn`, `tool_request`, `tool_result`, `review`, `follow_up`, `complete`, `error`.
- `lead_model` and `sidekick_model` resolved from env `MANTIS_FUSION_LEAD_MODEL` / `MANTIS_FUSION_SIDEKICK_MODEL`, falling back to catalog workers (`gpt-5_6-sol` for lead, `gemini-3_7-flash` for sidekick) with explicit reasoning-effort and max-tokens from the catalog.

**Verify**: `uv run ruff check apps/api/fusion.py` exits 0.

### Step 2: Implement `FusionRun` in `apps/api/fusion.py`

Implement a `FusionRun(NativeRun)` that:

- Stores `lead_messages`, `sidekick_messages`, `brief`, `pending_tool_calls`, `lead_review_pending`, and `final_report`.
- `advance(tool_results)` handles:
  - Initial planning: lead produces a plan and tool calls for `delegate_to_sidekick`.
  - Sidekick execution: sidekick receives the plan + brief and may emit tool calls.
  - Tool resumption: when `tool_results` are provided, advances the sidekick turn.
  - Review: when the sidekick reports, the lead reviews and either accepts or requests a `follow_up`.
- Uses `record_activity` for each planning/sidekick/review turn.
- Aggregates usage via `add_usage` for both lead and sidekick calls.

**Verify**: `uv run pytest tests/test_fusion.py::test_fusion_run_smoke -q` passes (create this test first if following TDD).

### Step 3: Add `/v1/fusion/*` endpoints to `apps/api/api.py`

Add four endpoints under the existing FastAPI `app`:

- `POST /v1/fusion/delegate` — accepts a JSON body `{"brief": "...", "tools": [...], "sidekick_model": "...", "lead_model": "..."}`, creates a `FusionRun`, runs until the first tool suspension or completion, and returns `{run_id, status, report, pending_tool_calls, usage}`.
- `POST /v1/fusion/follow_up/{run_id}` — accepts `{"message": "...", "tool_results": [...]}`. Resumes the run and returns the same shape.
- `GET /v1/fusion/status/{run_id}` — returns current state, activity, and usage without advancing.
- `DELETE /v1/fusion/runs/{run_id}` — cancels and removes the run from the store.

Reuse `providers._authorize` and the existing `NativeRun` storage (memory/redis/file) via `advance_run` and `delete_run`.

**Verify**: Start the API and run `curl -fsS http://127.0.0.1:8088/v1/fusion/delegate ...` with a test brief; expect a `200` with `run_id` and `status`.

### Step 4: Wire model resolution to the catalog

In `apps/api/fusion.py`, implement `_resolve_fusion_models()`:

- Read `MANTIS_FUSION_LEAD_MODEL` and `MANTIS_FUSION_SIDEKICK_MODEL`.
- Parse as `provider/upstream_model` (e.g. `bifrost/gpt-5.6-sol`).
- Look up `max_tokens` and `reasoning_effort` from `config/catalog.toml` or fallback to hard-coded safe defaults.
- Return `(lead_backend, sidekick_backend)` dicts compatible with `providers._call_worker` or the existing worker call path.

**Verify**: `uv run pytest tests/test_fusion.py::test_resolve_models -q` passes.

### Step 5: Add tests for the full lifecycle

In `tests/test_fusion.py`:

- Happy path: brief → sidekick tool call → tool result → sidekick report → lead accept.
- Retry path: brief → sidekick report → lead `follow_up` → sidekick retry → final report.
- Cancellation and status endpoints.
- Usage aggregation covers both lead and sidekick.

Use mock provider responses; do not make billed calls in tests.

**Verify**: `uv run pytest tests/test_fusion.py -q` passes.

### Step 6: Add a 5-task eval manifest

Create `eval/fusion_manifest.json` with 5 SWE-rebench or synthetic coding tasks. Add an eval command that runs `mantis-fusion` against `mantis` (direct) and `mantis-trinity` on the same tasks, recording:

- `resolved` (pass/fail),
- `cost_usd`,
- `duration_ms`,
- `turn_count`,
- `lead_calls`,
- `sidekick_calls`.

**Verify**: `uv run python eval/router_eval.py --manifest eval/fusion_manifest.json --arms mantis,mantis-fusion --dry-run` prints the arm list with caps.

### Step 7: Run the eval and decide whether to continue

Run the eval (this costs money; ensure `MANTIS_FUSION_SIDEKICK_MODEL` is a cheap model). Compare `mantis-fusion` against `mantis` on the 5 tasks. If Fusion does not improve the cost/quality frontier, stop and report. If it shows promise, move to the next plan (full productionization).

**Verify**: A `eval/report-fusion-pilot.md` is produced with cost and quality deltas.

## Test plan

- `tests/test_fusion.py`:
  - `test_fusion_run_smoke` — planning → sidekick → tool result → final report.
  - `test_follow_up_retry` — lead rejects sidekick report, sidekick retries.
  - `test_fusion_status_and_cancel` — status and delete endpoints.
  - `test_resolve_models` — catalog resolution and env overrides.
- `tests/test_api.py` should gain a `test_fusion_endpoints` that posts to `/v1/fusion/delegate` with a mocked worker.
- Verification: `uv run pytest tests/test_fusion.py tests/test_api.py -q` all pass.

## Done criteria

ALL must hold:

- [ ] `uv run ruff check .` exits 0.
- [ ] `uv run mypy apps/api --exclude outputs` exits 0.
- [ ] `uv run pytest tests/test_fusion.py -q` exits 0 with no failures.
- [ ] `curl` smoke test against a running `apps/api` returns a valid `FusionRun` state.
- [ ] `eval/fusion_manifest.json` exists and `eval/router_eval.py --dry-run` accepts the `mantis-fusion` arm.
- [ ] `advisor-plans/README.md` status row for plan 001 is updated.

## STOP conditions

Stop and report (do not improvise) if:

- `NativeRun` cannot support a two-agent turn loop without a large refactor (more than adding a subclass and a few methods).
- The client-side tool loop cannot be expressed through the existing `advance_run` / tool-result mechanism.
- `apps/api/api.py` does not accept new endpoints without changes to its FastAPI dependencies or request validation.
- The 5-task eval shows no cost/quality improvement and no clear path to one.
- Any in-scope file drifted from the "Current state" excerpts.

## Maintenance notes

- This is a prototype; expect `FusionRun` to be heavily refactored before production. Do not expose `mantis-fusion` in public docs until the eval is positive.
- If the eval is positive, the next plan will: add SSE streaming, chat-completions adapter, server-side sandbox as an optional tool backend, and a full 40-task eval.
- Reviewers should scrutinize the `FusionRun` state machine and the `/v1/fusion/*` API contract; changes after the pilot will be easier if the contract is typed and versioned from the start.
