# Plan 011: Calibrate routing uncertainty and cheap-first escalation

> **Executor instructions**: Build calibration offline first. Production escalation remains gated and deterministic.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- extensions/mantis.ts apps/api eval scripts tests extensions/tests`. STOP if plans 008-010 have not landed or changed their schemas.

## Status
- **Priority**: P1
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans 008 and 010
- **Category**: perf / direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
A cheap route should be used only when its predicted success is calibrated. The fallback should add the smallest useful compute: direct, then targeted verification or TRINITY, and only then a validated workflow mode.

## Current state
- `extensions/mantis.ts:343-369` obtains only a Supra complexity header and converts failures to score 1.
- `extensions/mantis.ts:372-376` exposes no confidence, margin, entropy, or abstention.
- `apps/api/runs.py:596-644` stops TRINITY only on verifier acceptance or max turns.
- Actual test/tool observations already exist at `apps/api/runs.py:453-468`.

## Scope
**In scope**: offline calibration/report code, compact calibration artifact, utility routing and direct/TRINITY escalation path, focused tests/docs.

**Out of scope**: nonlinear router heads, global turn increases, all-model voting, security-task exploration, unbounded retries.

## Steps
1. On held-out plan-008 data, measure calibration of each candidate route using reliability bins, Brier score, expected calibration error, and selective risk/coverage. Use stdlib/NumPy or installed PyTorch; add no dependency.
2. Fit the smallest calibration that improves held-out metrics: threshold table, temperature scaling, or isotonic-like monotone bins. Prefer deletion if raw scores are already calibrated.
3. Define deterministic escalation rules using calibrated confidence, risk class, test outcome, and remaining budget. Start with direct → TRINITY. Add Conductor only after its own gate passes.
4. For coding runs, allow a successful recognized test plus route confidence to stop; failure may escalate once with the failure state. Never infer success merely from tool exit absence if no recognized test ran.
5. Shadow-evaluate first, then canary. Report quality/cost/latency against direct, TRINITY, and plan-010 router baselines.

## Test plan
Synthetic calibration tests must cover overconfidence, underconfidence, OOD/abstain, failed test escalation, successful test stop, exhausted budget, and high-risk conservative routing. Provider tests must prove at most one escalation.

## Done criteria
- [ ] Calibration metrics and artifact provenance are reproducible.
- [ ] Every production decision can abstain.
- [ ] Escalation is bounded and budget-aware.
- [ ] Held-out Pareto report passes plan 008 gate.
- [ ] Full Python and extension verification passes.

## STOP conditions
- Calibration set has fewer than 30 outcomes for a route/family being promoted.
- Confidence degrades on held-out/OOD data.
- Escalation requires replaying mutating tools without an idempotency guarantee.

## Maintenance notes
Calibration expires when models, prompts, prices, or providers change. Stamp artifacts with pool, policy, fixture, and code digests.
