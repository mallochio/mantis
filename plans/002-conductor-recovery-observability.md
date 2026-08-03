# Plan 002: Make Conductor failures recoverable and visible

> **Executor**: Follow each step and gate. Update the index when done.
>
> **Drift check**: `git diff --stat 5874163..HEAD -- openfugu-patch/serve.py OpenFugu/openfugu/ultra.py tests/test_serve.py extensions/mantis.ts extensions/tests/stream-test.ts`. STOP on material drift.

## Status
- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: `plans/001-scope-consume-response-cache.md`
- **Category**: bug
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
Trinity's rejection state is incorrectly shared with Conductor. An empty Conductor node sets `force_worker`, but Conductor never calls `RejectAwareRouter`; `revision_feedback` can then alter the next unrelated DAG node. The planning call is also invisible in Pi, so users cannot distinguish planning latency/failure from dropped work.

## Current state
- `openfugu-patch/serve.py:186-232` treats every Conductor subtask as `Worker` and writes Trinity retry thread-local state.
- `openfugu-patch/serve.py:246-255` delegates `conduct()` without step events.
- `openfugu-patch/serve.py:399-428` plans then executes the DAG.
- `OpenFugu/openfugu/ultra.py:183-207` accepts empty replies and makes the last output final.
- Existing retry test: `tests/test_serve.py:169-209`.

## Scope
**In**: `openfugu-patch/serve.py`, `tests/test_serve.py`, `extensions/mantis.ts`, `extensions/tests/stream-test.ts`; touch `OpenFugu/openfugu/ultra.py` only if the root repo can publish the resulting submodule commit.
**Out**: retraining, changing workflow DSL, adding dependencies.

## Steps
1. Separate Trinity retry state from Conductor execution. Never pass an empty-node retry instruction to a different DAG subtask.
2. Implement one bounded retry of the same empty Conductor node using the same selected model and an explicit completion instruction. Preserve both attempts as visible events; avoid unbounded/configurable retry machinery.
3. Emit a native planning step with role `Planner`, selected model, prompt, and completion. Update the Pi tool schema/rendering to accept Planner without special custom UI.
4. If planning output is empty or unparseable, return a clear error; do not execute a guessed DAG.
5. Add tests for successful planning, malformed planning, empty-node retry success, retry exhaustion, no feedback leakage, and complete turn numbering.

## Verification
```bash
uv run pytest -q --no-cov tests/test_serve.py
cd extensions && npm run typecheck && bun run tests/stream-test.ts
```
Expected: all pass; new Conductor cases assert planner and retry events.

## Done criteria
- Planner and every node attempt are visible in Pi.
- Empty Conductor nodes retry once in place; exhausted retries fail clearly.
- Trinity reject behavior remains unchanged.

## STOP conditions
- The fix requires an unpublished gitlink commit in `OpenFugu/`; keep logic in the tracked overlay and report instead.
- A retry cannot preserve DAG access-list semantics.
