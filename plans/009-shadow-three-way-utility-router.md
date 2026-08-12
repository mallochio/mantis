# Plan 009: Shadow-log a three-way expected-utility router

> **Executor instructions**: Implement observation only. The selected production route must remain unchanged in this plan.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- extensions/mantis.ts extensions/tests/stream-test.ts config/worker-costs.json README.md .env.example`. STOP on material routing-flow drift.

## Status
- **Priority**: P1
- **Effort**: S
- **Risk**: LOW
- **Depends on**: `plans/008-establish-routing-evaluation-gate.md`
- **Category**: direction / perf
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
This is the cheapest safe runtime experiment: calculate what a direct/TRINITY/Conductor utility policy would choose, but do not act on it. Current `/auto` is binary: `extensions/mantis.ts:372-376` maps one Supra score to TRINITY or Conductor. The README claims a direct branch that does not exist in `extensions/mantis.ts`.

## Current state
- `extensions/mantis.ts:34` defines only off, trinity, conductor, and auto.
- `extensions/mantis.ts:128` reads one fixed threshold.
- `extensions/mantis.ts:136-145` logs score, coordinator, and a task prefix.
- `extensions/mantis.ts:717-723` applies the binary decision before creating a run.
- `config/worker-costs.json` is the existing static cost source; treat missing/stale prices as uncertainty, not zero cost.

## Scope
**In scope**: `extensions/mantis.ts`, `extensions/tests/stream-test.ts`, `config/worker-costs.json` only if its schema needs a documented timestamp, `.env.example`, `README.md`.

**Out of scope**: changing the live route, online exploration, executing an extra model, storing full prompts, training a new head.

## Steps
1. Add a pure function that accepts candidate quality estimates, expected USD cost, latency, failure penalty, and policy weights and returns utilities plus the recommended action. Candidate actions are `direct`, `trinity`, and `conductor`.
   - **Verify**: deterministic extension tests cover cost-first, quality-first, missing-estimate, and tie behavior.
2. For each `/auto` decision, calculate a shadow recommendation using only already-available Supra score and configured/static priors. If evidence is missing, return `abstain` rather than fabricating precision.
   - **Verify**: mocked low/medium/high scores generate a shadow record while the POST `/runs` model remains the legacy binary choice.
3. Replace task-prefix logging with a privacy-minimized record: task hash, coarse task features, legacy decision, shadow decision, utility components, uncertainty/abstention reason, policy version, and timestamp. Do not log raw task text.
   - **Verify**: tests assert a distinctive task phrase does not occur in the log record.
4. Fail closed: malformed prices, NaN, router timeout, or missing fields must leave current production behavior unchanged.
   - **Verify**: extension tests exercise each failure and assert the same created coordinator as before.
5. Document shadow mode and how to summarize disagreement rates without exposing prompts.

## Test plan
Extend the existing single-file integration test `extensions/tests/stream-test.ts`; do not add a test framework. Test pure utility ranking and provider-level no-behavior-change.

## Done criteria
- [ ] Shadow recommendations include direct/TRINITY/Conductor or explicit abstention.
- [ ] Production routing is byte-for-byte equivalent at the create-run boundary for existing tests.
- [ ] No raw prompt or task prefix is written.
- [ ] `cd extensions && npm run typecheck && npm test` and `git diff --check` pass.

## STOP conditions
- Computing a recommendation requires a paid call or another worker execution.
- The only way to log outcomes is to persist tool output or repository content.
- Plan 008 has no stable policy-comparison report format.

## Maintenance notes
Shadow data is observational and historically biased. Do not call disagreement an accuracy metric until counterfactual outcomes exist.
