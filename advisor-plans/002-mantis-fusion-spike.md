# Plan 002: mantis-fusion main/sidekick spike

> **Revision note (2026-08-16)**: This plan was revised during execution. The operator requested that the main (lead) and sidekick models be changeable through the shared catalog at `~/.config/ai-routing/catalog.toml`. The implementation therefore adds a top-level `[fusion]` catalog section, a two-model Fusion state machine (main plans/reviews, sidekick executes), and uses catalog worker slot IDs for both roles.

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If any STOP condition occurs, stop and report — do not improvise. When done, update the status row in `advisor-plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 7070b7f..HEAD -- apps/api/ tests/`
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

The cheapest way to test the `pi-fusion` lead/sidekick pattern in Mantis is to expose only the **sidekick** as a resumable service. The client (which may be a frontier model, an IDE, or `pi` itself) acts as the lead: it writes the brief, reviews the sidekick's report, and decides whether to `follow_up`. Mantis runs the sidekick model, persists its context, and handles tool-call suspension. If the existing `NativeRun` store cannot support this single-model resumable loop, the broader two-agent Fusion design is also infeasible.

## Research basis

- `pi-fusion` defines the sidekick contract in `extensions/fusion.ts:147-183` and `extensions/fusion.ts:186-253`: a sidekick session receives a brief, runs the implement/test/lint loop, returns a report, and can be resumed via `follow_up(task_id, message)`. The lead is the calling `pi` session, not the sidekick.
- `skills/fusion/SKILL.md` from `pi-fusion` specifies the sidekick's role: it owns execution and reporting; the lead owns plan, review, and final commit.
- `NativeRun` in `apps/api/runs.py:361-527` is the resumable run base. It supports usage aggregation (`add_usage`), idempotent advance (`advance_idempotent`), and pickle serialization (`__getstate__`/`__setstate__` at `apps/api/runs.py:450-460`). Persistence is implemented in `apps/api/runs.py:173-213` (redis/file) and `serve_config._runs` (memory) at `apps/api/serve_config.py:90-91`.
- `advance_run` and `delete_run` at `apps/api/runs.py:1273-1325` provide the public run lifecycle: lock, load, advance, save.
- `apps/api/api.py:692` shows the FastAPI endpoint pattern with `Depends(_authorize)`. This spike adds two new endpoints; it does not change the existing chat-completions handler.
- `providers._provider_response` at `apps/api/providers.py:519-525` is the low-level worker call; it returns the full provider response dict including `choices`, `usage`, and `error`. `utils._model_completion` at `apps/api/utils.py:339-367` wraps it for callers that only need text and tool calls.
- The Bifrost-driven `gpt-5.6-sol` critique (2026-08-16) recommended starting with a protocol/persistence spike, using client-side tools, and keeping `/v1/fusion/*` canonical.

## Current state

- `apps/api/runs.py:361-460` — `NativeRun` base class with serialization hooks and idempotent advance.
- `apps/api/runs.py:1261-1325` — `get_run`, `advance_run`, `delete_run` persistence lifecycle.
- `apps/api/api.py:692` — existing `POST /v1/chat/completions` FastAPI endpoint; no `/v1/fusion/*` routes.
- `apps/api/serve_config.py:82-91` — `RUN_STORE` config and the in-memory `_runs` dict.
- `apps/api/providers.py:519-525` — low-level provider response helper.
- `config/catalog.toml` — worker catalog; this spike uses an environment override for the sidekick model, not catalog parsing.

## Resolved execution semantics

These rules are authoritative and resolve any apparent conflicts in later sections.

- `POST /v1/fusion/delegate` is **not idempotent**. The client does not supply a `request_id`. The server creates a new run, saves it, runs the first sidekick turn, and returns its `run_id`.
- `POST /v1/fusion/follow_up/{run_id}` is **idempotent** by `request_id`. The client supplies a new `request_id` for each distinct `follow_up`. Repeating the same `request_id` returns the cached response.
- A `follow_up` body must contain exactly one of `message` or `tool_results`. Bodies with both, neither, or a missing `request_id` fail FastAPI/Pydantic validation and return HTTP 422 before any run is loaded or mutated.
- HTTP 422 errors do not create or mutate a run.
- Run-level errors (missing required tool results, worker exceptions, invalid state transitions) are caught inside `FusionRun.advance`, set `status = "error"`, persist the run, and return HTTP 200 with `status: "error"`.
- The assistant message that contains tool calls is appended exactly once, when the sidekick first emits those tool calls. On resumption with `tool_results`, only `tool` messages are appended.
- `_ProbeRun` in Step 0 must be declared at module scope so file-pickle round-trips work.

## Normative contract

This contract must be implemented exactly as specified. Do not expand it during execution.

### Concept

Mantis runs a cheap sidekick model. The client is the lead. The sidekick receives a brief, may emit tool calls, and returns a report. The client may call `follow_up` with feedback and/or tool results. The sidekick's conversation state is persisted by Mantis.

### Endpoints

| Endpoint | Request body | Response body |
|---|---|---|
| `POST /v1/fusion/delegate` | `{"brief": str, "tools": list[dict]?}` | `{"run_id": str, "status": "awaiting_tools" \| "sidekick_review" \| "error", "report": str\|null, "pending_tool_calls": list[dict]\|null, "usage": dict, "activity": list[dict]}` |
| `POST /v1/fusion/follow_up/{run_id}` | `{"request_id": str, "message": str? OR "tool_results": list[dict]?}` | same shape as `delegate` |

- `tool_results` items are `{"tool_call_id": str, "role": "tool", "content": str, "is_error": bool}`. When status is `awaiting_tools`, `tool_results` is required and the set of `tool_call_id` values in `tool_results` must include every `tool_call_id` in `pending_tool_calls` after first-occurrence deduplication. Missing ids raise `ValueError` and transition to `error`. Unknown ids are ignored; duplicate ids keep the first item.
- `message` is required when status is `sidekick_review` and is appended as a `user` message. It may be empty string `""`.

### State machine

- `sidekick_pending` — transient; the sidekick model is about to be called.
- `awaiting_tools` — sidekick emitted tool calls; client must execute them and post results.
- `sidekick_review` — sidekick produced a text report with no pending tool calls; client may `follow_up` with `message`.
- `error` — unhandled exception or invalid input; terminal.

### Sidekick prompt and behavior

- System prompt (preamble) is fixed in `apps/api/fusion.py` as `SIDEKICK_PREAMBLE` and instructs the sidekick to implement, test, lint, and return a final report. It is included at the start of `sidekick_messages`.
- Sidekick model is resolved from `MANTIS_FUSION_SIDEKICK_MODEL`. Default is `bifrost/gemini-3.6-flash`.
- The sidekick may use any tool from the `tools` list provided in `delegate`.
- When the sidekick response contains no tool calls, the run enters `sidekick_review` and `report` is set to the response text.
- When the sidekick response contains tool calls, the run enters `awaiting_tools` and `pending_tool_calls` is set to those calls.

### Worker-call interface

`FusionCoordinator._call_worker(model: str, messages: list[dict], tools: list[dict] | None) -> tuple[str, list[dict], dict]`:

- Calls `providers._provider_response(model, messages, 4096, 0.7, tools)` and receives the full response `data`.
- Extracts `text` from `data["choices"][0]["message"].get("content")`.
- Extracts `tool_calls` from the same message (using the same shape as `utils._model_completion` does).
- Extracts `usage` from `data.get("usage", {})`.
- Returns `(text, tool_calls, usage)`.
- Tests replace this method with a deterministic mock.

### Worker injection after restart

`FusionRun` does **not** store `FusionCoordinator`. After a process restart, `advance_fusion_run` creates a fresh `FusionCoordinator` and passes it to `run.advance(...)`. The run stores only `sidekick_model`, `sidekick_messages`, `pending_tool_calls`, `tools`, and usage.

### Idempotency

- `FusionRun.advance(tool_results=None, request_id=None, message=None, coordinator=None)` is the non-caching core. If `request_id` is `None`, it never touches `request_events`.
- `FusionRun.advance_idempotent(request_id, tool_results=None, message=None, coordinator=None)` checks `request_events` for `request_id`; on cache miss calls `self.advance(...)` with the same args and caches the result.
- `advance_fusion_run(run_id, request_id, tool_results=None, message=None)` mirrors `apps/api/runs.py:advance_run` but creates a fresh `FusionCoordinator`, loads the run, calls `FusionRun.advance_idempotent`, and saves.

### Storage guarantees

- `memory`: same-process serialization only; not restart-safe.
- `file`: restart-safe. The restart round-trip test must use `file`.
- `redis`: restart-safe if `MANTIS_RUN_STORE=redis`; otherwise optional.

### Exact message and response schemas

- Initial `sidekick_messages`: `[{"role": "system", "content": SIDEKICK_PREAMBLE}, {"role": "user", "content": brief}]`.
- Assistant text response: `{"role": "assistant", "content": text}`.
- Assistant tool-call response: `{"role": "assistant", "content": text, "tool_calls": tool_calls}`.
- Tool result message: `{"role": "tool", "tool_call_id": id, "content": content, "is_error": bool}`.
- `activity`: built by `self.record_activity(kind, ...)` as the spike uses `NativeRun.record_activity`.
- `usage`: dict returned by the worker; aggregated via `self.add_usage(usage)`.
- Error response body: `{"run_id": run.run_id, "status": "error", "report": null, "pending_tool_calls": null, "usage": run.usage, "activity": run._activity}`.

## Normative execution table

This table is the executable contract. Implement it literally. Row 5 applies only to run-level errors inside `advance`, not HTTP-level validation.

| Step | Current status | Input | Action | Persisted updates | Next status | Response | Caching |
|---|---|---|---|---|---|---|---|
| 1 | (new) | `delegate` with `brief`, `tools` | Create `FusionRun`; set `sidekick_messages = [system, user brief]`; status `sidekick_pending`. Save. Call sidekick. Append assistant response. Save again. | `sidekick_messages` including assistant response, `tools`, `sidekick_model`, `pending_tool_calls` (if any), `usage`. | `awaiting_tools` if tool calls; `sidekick_review` if text only. | `pending_tool_calls` or `report`, `usage`, `activity`. | No caching; `request_id` is `None`. |
| 2 | `awaiting_tools` | `follow_up` with new `request_id`, `tool_results` (message forbidden by schema) | Validate that `tool_results` covers every `pending_tool_call` after deduplication. Append one `tool` message per matched id. Clear `pending_tool_calls`. Call sidekick. Append assistant response. | Append tool messages and assistant response; set new `pending_tool_calls` or `report`; add usage. | `awaiting_tools` or `sidekick_review`. | `pending_tool_calls` or `report`, `usage`, `activity`. | Cache under `request_id`. |
| 3 | `sidekick_review` | `follow_up` with new `request_id`, `message` (tool_results forbidden by schema) | Append `message` as user. Call sidekick. Append assistant response. | Append user and assistant messages; set `report` or new `pending_tool_calls`; add usage. | `awaiting_tools` or `sidekick_review`. | `pending_tool_calls` or `report`, `usage`, `activity`. | Cache under `request_id`. |
| 4 | any | same `request_id` as a previous `follow_up` | Look up `request_events[request_id]`. | None. | (cached) | Return cached response. | Already cached. |
| 5 | any | missing tool results, or worker exception caught in `advance` | Inside `advance`, catch exception, set `status = "error"`, record `error` message, save. | `status`, `error` message. | `error` | Error response body. | Cache under `request_id` for `follow_up`; `delegate` has no request_id so no caching. |

Rules for Step 2:

- `tool_results` must be a list. After first-occurrence deduplication, the set of `tool_call_id` values must include every id in `pending_tool_calls`. Missing ids raise `ValueError` and trigger Step 5.
- Unknown `tool_call_id` values are ignored.
- The assistant message containing the original tool calls is already in `sidekick_messages`; do not append it again.
- `message` is forbidden in this step (schema-level).
- Each tool message is `{"role": "tool", "tool_call_id": str, "content": str, "is_error": bool}`.

Rules for Step 3:

- `tool_results` is forbidden in this step (schema-level).
- `message` is required (may be empty string).

## Commands you will need

| Purpose | Command | Expected on success |
|---------|---------|---------------------|
| Sync | `uv sync` | exit 0 |
| Typecheck | `uv run mypy apps/api --exclude outputs` | exit 0, no errors |
| Lint | `uv run ruff check .` | exit 0 |
| Unit tests | `uv run pytest tests/test_fusion_spike.py -q` | all pass |
| Integration smoke | `uv run pytest tests/test_fusion_spike.py::test_fusion_endpoints -q` | passes using in-memory API client |
| Regression check | `uv run pytest tests/test_api.py tests/test_serve.py -q` | all pass |

## Suggested executor toolkit

- Read `apps/api/runs.py` `NativeRun` and `advance_run`.
- Read `apps/api/api.py` for FastAPI route/dependency conventions.
- Read `apps/api/providers.py:519-525` and `apps/api/utils.py:339-367` for the worker-call path.

## Scope

**In scope**:
- `apps/api/fusion.py` (new) — `FusionRun` subclass, `FusionCoordinator`, `advance_fusion_run` helper, `SIDEKICK_PREAMBLE`.
- `apps/api/api.py` — add `POST /v1/fusion/delegate` and `POST /v1/fusion/follow_up/{run_id}`.
- `tests/test_fusion_spike.py` (new) — unit tests and persistence round-trip fixtures.
- `advisor-plans/README.md` — update the plan status row.

**Out of scope** (do NOT touch):
- `apps/gateway/server.py` — Fusion uses the gateway; it does not modify it.
- `apps/api/trinity.py`, `apps/api/conductor.py` — study only.
- `apps/api/serve_config.py` `MODEL_MODES` and the chat-completions adapter.
- `eval/router_eval.py` integration or paid eval.
- Status, cancel, SSE, or streaming endpoints.
- Server-side code sandbox.
- A lead model running inside Mantis.
- `advisor-plans/002-mantis-fusion-spike-results.md` — optional; create only if the go/no-go gate needs a written record.

## Git workflow

- Branch: `advisor/002-mantis-fusion-spike`
- Commit per logical unit.
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 0: Verify `NativeRun` can be subclassed and persisted

Before writing `FusionRun`, add a test file `tests/test_fusion_spike.py` and declare `class _ProbeRun(NativeRun)` at module scope. Add a test that:

1. Creates a run, advances it, and stores it with `MANTIS_RUN_STORE=memory` and `file` (`redis` if available).
2. Loads it back and confirms the extra field and usage counters survive.

**Verify**: `uv run pytest tests/test_fusion_spike.py::test_nativerun_subclass_persistence -q` passes for `memory` and `file`.

**STOP**: If `file` fails to round-trip the subclass, stop and report. Do not proceed to `FusionRun`.

### Step 1: Implement `FusionRun`, `FusionCoordinator`, and `advance_fusion_run` in `apps/api/fusion.py`

Create `apps/api/fusion.py` and define:

- `SIDEKICK_PREAMBLE` — fixed system prompt text.
- `FusionRun(NativeRun)` with fields: `brief`, `sidekick_messages`, `pending_tool_calls`, `tools`, `sidekick_model`, `report`, `error`.
- `FusionRun.advance(tool_results=None, request_id=None, message=None, coordinator=None)` — the non-caching core implementing the Normative execution table:
  - If `request_id` is `None`, do not read or write `request_events`.
  - Wrap the entire transition logic in a `try`/`except` that catches `ValueError` and any worker exception; on catch, set `status = "error"`, `error = str(exc)`, record an error activity, and return the error response body.
  - Validate status/input according to the table.
  - If `tool_results` provided, validate coverage after deduplication; append one tool message per matched id; clear `pending_tool_calls`.
  - If `message` provided, append `{"role": "user", "content": message}`.
  - Call `coordinator._call_worker(self.sidekick_model, self.sidekick_messages, self.tools)`.
  - Append the assistant response to `sidekick_messages` and set `report` or `pending_tool_calls`.
  - Set status and add usage.
  - Record an activity entry.
  - Return the response body dict.
- `FusionRun.advance_idempotent(request_id, tool_results=None, message=None, coordinator=None)` — checks `request_events` for `request_id`; on cache miss calls `self.advance(...)` and stores the result.
- `FusionCoordinator` with `_call_worker(model, messages, tools)` calling `providers._provider_response` and returning `(text, tool_calls, usage)` as described in the contract.
- `advance_fusion_run(run_id, request_id, tool_results=None, message=None)` that mirrors `apps/api/runs.py:advance_run` but creates a fresh `FusionCoordinator`, loads the run, calls `FusionRun.advance_idempotent`, and saves.
- Sidekick model resolution: read `MANTIS_FUSION_SIDEKICK_MODEL`; default to `bifrost/gemini-3.6-flash`.

**Verify**: `uv run ruff check apps/api/fusion.py` and `uv run mypy apps/api/fusion.py` exit 0.

### Step 2: Add `/v1/fusion/delegate` and `/v1/fusion/follow_up/{run_id}` to `apps/api/api.py`

Add two FastAPI endpoints under existing `app`:

- `POST /v1/fusion/delegate`
  - Body: `{"brief": str, "tools": list[dict]?}`.
  - Create `FusionRun` with `sidekick_messages = [system, user brief]`, `status = sidekick_pending`, and a fresh `run_id`.
  - Save the run using the same helper as `get_run`/`advance_run`.
  - Create `FusionCoordinator`.
  - Call `run.advance(tool_results=None, request_id=None, message=None, coordinator=coordinator)` (non-idempotent).
  - Save the run again.
  - Returns: `{"run_id", "status", "report", "pending_tool_calls", "usage", "activity"}`.
- `POST /v1/fusion/follow_up/{run_id}`
  - Body: `{"request_id": str, "message": str? OR "tool_results": list[dict]?}`. Use a Pydantic model with a validator that requires exactly one of `message` or `tool_results` and rejects `None` for the chosen field.
  - Calls `advance_fusion_run(run_id, request_id, tool_results, message)`.
  - Returns same shape as `delegate`.

Use existing `_authorize` dependency. No chat-completions framing.

**Verify**: `uv run pytest tests/test_fusion_spike.py::test_fusion_endpoints -q` passes.

### Step 3: Add unit and idempotency tests

In `tests/test_fusion_spike.py`, with a mocked `_call_worker`:

- `test_fusion_run_smoke`: brief → sidekick tool call → tool result → sidekick report → `sidekick_review`.
- `test_fusion_run_follow_up_message`: brief → sidekick report → follow_up `message` → sidekick retry with new tool call → tool result → sidekick review.
- `test_fusion_run_idempotent_follow_up`: a `follow_up` with `request_id=X` is retried with the same `X` and returns the same event; a `follow_up` with `request_id=Y` is a new turn.
- `test_fusion_run_tool_resumption`: tool results cover all pending tool calls; sidekick is called with a valid chat transcript.
- `test_fusion_run_missing_tool_result`: a `follow_up` missing a required tool result transitions to `error`.
- `test_fusion_run_message_and_tools_rejected`: a `follow_up` with both `message` and `tool_results` returns HTTP 422.
- `test_fusion_error_handling`: a worker exception sets `status = "error"` and returns a stable error response.

**Verify**: `uv run pytest tests/test_fusion_spike.py -q` passes.

### Step 4: Prove persistence and restart resumption

Add a fixture test with `MANTIS_RUN_STORE=file`:

1. Calls `POST /v1/fusion/delegate` with a brief that forces a sidekick tool call.
2. Saves `run_id` and `pending_tool_calls`.
3. Simulates a restart by creating a new `FusionCoordinator`, loading the run via `get_run(run_id)`, and calling `run.advance_idempotent(request_id_2, tool_results, message=None, coordinator=mock_coordinator)` with a **new** `request_id_2`.
4. Confirms the sidekick continues from the saved state and `pending_tool_calls` is updated or a `report` is returned.
5. If `MANTIS_RUN_STORE=redis` is available, repeats the same test with a fresh Redis load.
6. For `MANTIS_RUN_STORE=memory`, confirms `get_run(run_id)` returns the same object in the same process.

**Verify**: `uv run pytest tests/test_fusion_spike.py::test_fusion_persistence_round_trip -q` passes.

### Step 5: Define the go/no-go gate

Run a manual checklist before any further plans:

- `FusionRun` round-trips through `file` without losing messages or pending tool calls.
- `memory` store at least round-trips in the same process.
- Tool results must cover all `pending_tool_calls` after first-occurrence deduplication.
- Duplicate `follow_up` calls with the same `request_id` are idempotent.
- Different `request_id` on the same `run_id` is a new operation.
- The two endpoints do not require changes to the chat-completions handler or `serve_config.MODEL_MODES`.
- No real model calls are needed in the test suite.
- `tests/test_api.py` and `tests/test_serve.py` still pass.

**Verify**: Document the gate results in `advisor-plans/002-mantis-fusion-spike-results.md` if desired.

## Test plan

- `tests/test_fusion_spike.py`:
  - `test_nativerun_subclass_persistence` — round-trip a `NativeRun` subclass through `memory` and `file`.
  - `test_fusion_run_smoke` — brief, tool call, tool result, report.
  - `test_fusion_run_follow_up_message` — client message triggers sidekick retry.
  - `test_fusion_run_idempotent_follow_up` — same `request_id` returns cached event; different `request_id` is a new turn.
  - `test_fusion_run_tool_resumption` — tool results cover all pending calls.
  - `test_fusion_run_missing_tool_result` — missing tool result transitions to `error`.
  - `test_fusion_run_message_and_tools_rejected` — body with both fields rejected.
  - `test_fusion_error_handling` — worker exception becomes `status: "error"`.
  - `test_fusion_persistence_round_trip` — restart-safe for `file`/`redis`; same-process for `memory`.
  - `test_fusion_endpoints` — `delegate` and `follow_up` return correct JSON and usage.

Use mock worker responses to avoid billed calls.

## Done criteria

ALL must hold:

- [ ] `uv run ruff check .` exits 0.
- [ ] `uv run mypy apps/api --exclude outputs` exits 0.
- [ ] `uv run pytest tests/test_fusion_spike.py -q` exits 0 with no failures.
- [ ] `test_fusion_persistence_round_trip` passes with `MANTIS_RUN_STORE=file`.
- [ ] `uv run pytest tests/test_api.py tests/test_serve.py -q` exits 0.
- [ ] `advisor-plans/README.md` status row for plan 002 is updated.

## STOP conditions

Stop and report (do not improvise) if:

- `NativeRun` cannot be subclassed and reloaded without losing state.
- The run store cannot round-trip a `FusionRun` for `MANTIS_RUN_STORE=file`.
- Tool results cannot be correlated by `tool_call_id` after first-occurrence deduplication.
- Duplicate `follow_up` calls are not idempotent.
- Adding two FastAPI routes to `apps/api/api.py` breaks existing tests.
- Any in-scope file drifted from the "Current state" excerpts.

## Maintenance notes

- This is an experimental spike. Do not expose `/v1/fusion/*` outside trusted networks until the next plan adds authentication review and rate limits.
- The sidekick prompt and model default should be treated as unstable; the next plan will decide whether to add a `mantis-fusion` chat-completions adapter and eval harness.
- Reviewers should focus on the Resolved execution semantics, the Normative execution table, and the `FusionRun`/`advance_fusion_run` boundary.
