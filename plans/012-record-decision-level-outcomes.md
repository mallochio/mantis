# Plan 012: Record decision-level outcomes for learning

> **Executor instructions**: Upgrade telemetry without changing the learned policy. Learning remains opt-in and promotion remains disabled for the new schema until plan 013.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- apps/api scripts/learn_router.py tests/test_serve.py tests/test_learn_router.py README.md .env.example`. STOP on telemetry-schema drift.

## Status
- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plan 008
- **Category**: direction / tests
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Current telemetry emits one terminal record and labels the final successful worker (`apps/api/runs.py:96-150`). It cannot teach per-turn worker/role, stopping, escalation, or budget decisions and excludes failures from training.

## Current state
- Trainability requires TRINITY + verifier acceptance + final recognized passing test at `apps/api/runs.py:96-134`.
- `label_worker` is the final Worker's ID and `label_role` is always zero at `apps/api/runs.py:131-134`.
- `scripts/learn_router.py:41-68` discards non-trainable records and only accepts schema version 1 hard labels.
- Telemetry uses private directories/files and excludes tool output at `apps/api/runs.py:138-150`; preserve those protections.

## Scope
**In scope**: telemetry schema v2, emission/load tests, migration/compatibility decision, privacy docs.

**Out of scope**: training on v2, executing counterfactual workers, logging tool output, automatic production promotion.

## Steps
1. Define schema v2 with a run envelope and one record per decision: state/task hash, route probabilities or logits, propensity, action type, worker, role, remaining turn/token/USD budget, coarse tool/test state, outcome, duration, observed tokens/cost, policy/pool versions, and terminal reward components.
2. Include failed, timed-out, cancelled, rejected, and max-turn runs as outcomes; mark censored/unknown outcomes explicitly.
3. Keep task text opt-in and redacted as today. Never store tool output, repository snapshots, credentials, commands containing secret-shaped values, or raw verifier text. Prefer hashes/coarse enums.
4. Make append and run-finalization idempotent. Preserve per-host append safety and 0600 file permissions.
5. Update the loader to report v1 counts separately. Do not silently reinterpret v1 selected-worker labels as counterfactual truth.

## Test plan
Extend `tests/test_serve.py:1951+` and `tests/test_learn_router.py`. Cover every terminal state, multiple decisions, propensities summing correctly, redaction, no tool output, permissions, duplicate finalization, malformed partial lines, and v1 separation.

## Done criteria
- [ ] Every routing decision has enough context for offline policy evaluation.
- [ ] Failures are retained without being falsely labeled.
- [ ] Privacy and file-permission guarantees remain tested.
- [ ] The v1 trainer cannot consume v2 accidentally.
- [ ] Full Python verification passes.

## STOP conditions
- A desired field requires storing raw tool output or source code.
- Propensity cannot be reconstructed exactly for the active stochastic policy; mark those records unusable rather than guessing.
- Telemetry writes become part of request correctness; logging must remain best effort.

## Maintenance notes
Schema changes require a version bump. Treat telemetry directories as sensitive even after redaction because task metadata can identify projects.
