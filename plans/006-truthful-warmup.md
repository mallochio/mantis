# Plan 006: Make coordinator warm-up truthful

> **Drift check**: `git diff --stat 5874163..HEAD -- extensions/mantis.ts extensions/tests/stream-test.ts openfugu-patch/serve.py tests/test_serve.py`. STOP on warm-up API drift.

## Status
- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: plan 003
- **Category**: dx
- **Planned at**: `5874163`, 2026-08-03

## Why this matters
`warm()` treats an authenticated `/models` response as proof that a coordinator is loaded. The backend endpoint does not call `get_coordinator`, so Pi reports “ready” while the first real call can still download/load models.

## Current state
- `extensions/mantis.ts:256-294` returns after `/models` succeeds.
- `openfugu-patch/serve.py:678-685` serves static model metadata.
- Coordinator loading happens at `openfugu-patch/serve.py:863-882`.

## Scope
**In**: extension/server and their existing tests.
**Out**: a general health framework, background daemons, Kubernetes probes.

## Steps
1. Reuse a native endpoint if one can unambiguously warm the requested coordinator without executing paid worker calls; otherwise add a small authenticated warm endpoint that calls `get_coordinator(mode)` and returns readiness.
2. Make the extension report ready only after that operation succeeds. Keep `/models` as liveness/model metadata.
3. Add a finite timeout and cancellation for warm-up using existing AbortController conventions.
4. Test Trinity, Conductor, invalid mode, auth failure, timeout, and idempotent second warm.

## Verification
```bash
uv run pytest -q --no-cov tests/test_serve.py
cd extensions && npm run typecheck && npm test
```
Expected: all pass; tests prove warm invokes coordinator loading once.

## Done criteria
- “ready” means the selected coordinator initialized.
- Warm-up never invokes a billable worker completion.
- Failures are visible but do not corrupt active mode.

## STOP conditions
- Initializing Conductor necessarily performs a paid completion; report and change UI wording to “backend reachable” instead of simulating a ping.
