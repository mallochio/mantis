# Plan 010: Add direct execution and canary the three-way router

> **Executor instructions**: Add the smallest direct path compatible with Pi native tools. Default `/auto` behavior must remain legacy until the held-out gate passes.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- extensions/mantis.ts openfugu-patch/serve.py tests/test_serve.py extensions/tests/stream-test.ts README.md .env.example`. STOP on native-run protocol drift.

## Status
- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans 008-009
- **Category**: direction / perf
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Direct averaged 0.958 on the simple tier versus TRINITY's 0.917 while using much less aggregate latency (`eval/report-native-v2.md:37-40`). Hard tasks show the reverse (`eval/report-native-v2.md:45-50`). A real direct candidate is therefore the clearest cost win, but it must preserve Pi's native tool loop and exact-once run semantics.

## Current state
- `extensions/mantis.ts:863-891` registers only trinity, conductor, and auto models.
- `extensions/mantis.ts:707-724` always creates an orchestrated backend run.
- `openfugu-patch/serve.py:1653-1717` provides the shared resumable `NativeRun` protocol and tool observation handling.
- `openfugu-patch/serve.py:1720-1962` is the tested native-tool TRINITY implementation to copy only where necessary.

## Scope
**In scope**: extension/server runtime and their existing tests, `.env.example`, `README.md`, evaluation config support.

**Out of scope**: a second transport protocol, bypassing trust-boundary validation, new dependencies, enabling Conductor by default, online exploration.

## Steps
1. Define one direct worker model setting and one `direct` run mode. Reuse `NativeRun`, `_model_completion`, native tool continuation, request idempotency, cancellation, body limits, and cleanup. Do not duplicate the whole TRINITY state machine.
   - **Verify**: server tests cover text-only direct completion, one and multiple native tool rounds, cancellation, timeout, replay, malformed tool results, and terminal cleanup.
2. Register `mantis/direct` and update command validation/help. Ensure explicit direct mode works independently of `/auto`.
   - **Verify**: extension tests confirm model registration, explicit selection, native tool lifecycle, and exactly one final answer.
3. Add a disabled-by-default canary switch that lets the utility policy select direct or TRINITY. Conductor remains unavailable to auto unless it independently passes plan 008's gate.
   - **Verify**: default tests preserve legacy routing; canary tests route low-risk/high-confidence simple tasks direct and uncertain tasks TRINITY.
4. Add hard exclusions for security-sensitive/high-risk classifications and malformed/abstaining decisions: they must not be experimental and must use the conservative route.
5. Run the held-out direct-vs-TRINITY evaluation with at least three repetitions where stochasticity matters. Obtain operator approval before paid calls.

## Test plan
Use `tests/test_serve.py` native-run tests and `extensions/tests/stream-test.ts` as patterns. Add no framework. Include direct tool error propagation and a canary-off regression.

## Done criteria
- [ ] Explicit direct mode supports Pi native tools, cancellation, replay safety, and cleanup.
- [ ] `/auto` defaults to legacy behavior.
- [ ] Canary promotion meets plan 008's quality, cost, latency, and risk gates.
- [ ] `uv run pytest tests -q`, Ruff, mypy, extension typecheck/tests, and `git diff --check` pass.

## STOP conditions
- Direct mode cannot reuse the backend run protocol without weakening tool validation or cancellation.
- The configured direct model/provider cannot support required Pi tools.
- Held-out quality regression exceeds the agreed tolerance; keep explicit direct mode but do not promote auto selection.

## Maintenance notes
Keep one direct model knob, not a new provider abstraction. Refresh its price and capability data whenever the pool changes.
