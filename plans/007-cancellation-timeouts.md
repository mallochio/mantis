# Plan 007: Bound and cancel backend model work

> **Drift check**: `git diff --stat 5874163..HEAD -- extensions/mantis.ts openfugu-patch/serve.py tests/test_serve.py extensions/tests/stream-test.ts README.md .env.example`. STOP on transport drift.

## Status
- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plan 004
- **Category**: perf
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
Pi cancellation aborts the client fetch, but backend LiteLLM calls are synchronous and receive no timeout. A cancelled turn can continue using provider time/cost, and the current fixed five-minute client timer is not coordinated with per-worker limits.

## Current state
- `extensions/mantis.ts:297-325` has client AbortController and a fixed 300-second timeout.
- `openfugu-patch/serve.py:67-82`, `261-286` build LiteLLM calls without timeout.
- `openfugu-patch/serve.py:641-657` reports stream errors but cannot cancel an in-flight completion.

## Scope
**In**: extension/server, tests, `.env.example`, relevant README configuration.
**Out**: async server rewrite, queues, distributed cancellation, new dependencies.

## Steps
1. Add one documented worker timeout calibration knob with a safe default below the client request timeout; pass it through LiteLLM's supported timeout parameter for planner, Trinity, and Conductor nodes.
2. Distinguish timeout, user abort, backend disconnect, and provider failure in Pi error messages without exposing credentials or raw request headers.
3. Remove the extension abort listener after each request; ensure timers are cleared on success/error/abort.
4. Stop scheduling later orchestration calls after client disconnect where the stdlib server can detect it. Do not claim cancellation of an already-dispatched provider call unless LiteLLM confirms it.
5. Add deterministic fake-worker tests for timeout before reply, cancellation between steps, disconnect, cleanup, and no subsequent calls.

## Verification
```bash
uv run pytest -q --no-cov tests/test_serve.py
cd extensions && npm run typecheck && npm test
git diff --check
```
Expected: all pass; fake call counts prove no later work starts after cancellation.

## Done criteria
- Every hosted model call has a bounded timeout.
- Pi abort cleans client listeners/timers and prevents later calls.
- Limitations of cancelling an already-running provider request are documented honestly.

## STOP conditions
- LiteLLM's installed version does not support a portable timeout parameter; report the supported provider-specific API rather than adding threads/process killing.
- The change requires replacing `ThreadingHTTPServer`.
