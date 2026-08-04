# Plan 013: Replace hard imitation with soft and counterfactual policy learning

> **Executor instructions**: Implement offline candidate generation and evaluation first. Never promote from selected-action accuracy alone.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- scripts/learn_router.py scripts/retrain_router_pool.py tests/test_learn_router.py tests/test_retrain_router_pool.py configs/worker-costs.json`. STOP unless plan 012's schema is present.

## Status
- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: plans 008 and 012
- **Category**: direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
`scripts/learn_router.py:111-133` trains cross-entropy on a single selected worker. Successful selection is evidence that the worker can work, not that it was optimal. Offline pool retraining similarly creates hard labels at `scripts/retrain_router_pool.py:681-736`, although measured scores for all workers are available.

## Current state
- Fugu-style hidden extraction already exists at `scripts/learn_router.py:87-100`.
- The online gate compares hard-label accuracy at `scripts/learn_router.py:202-219`.
- Offline scoring caches per task/model outputs at `scripts/retrain_router_pool.py:586-658`.
- Role labels are copied from the old router, not measured outcomes, at `scripts/retrain_router_pool.py:1076-1097`.

## Scope
**In scope**: soft-target construction, failure-aware contextual-bandit estimators, offline candidate reports, tests, explicit exploration policy design.

**Out of scope**: unrestricted exploration, exploration on high-risk/security tasks, backbone fine-tuning, nonlinear head expansion, automatic rollout spending.

## Steps
1. Convert repeated measured worker rewards into temperature-controlled soft targets. Preserve ties and reward magnitude; include cost/latency penalties only as explicit utility components.
2. For logged feedback, implement a minimal inverse-propensity or doubly robust evaluator using plan-012 propensities. Clip weights and report effective sample size; abstain when support is poor.
3. Train a candidate on successful and failed outcomes. Evaluate expected held-out utility, calibration, failure rate, cost, and latency—not selected-worker accuracy alone.
4. Add controlled counterfactual collection: only low-risk, uncertain tasks; top-two actions; small global/session budget; shadow result not shown to the user; no mutating tool replay. Require operator opt-in.
5. Promote only after plan 008's held-out gate and a small canary. Keep atomic artifact promotion and pool/version checks from the existing trainer.

## Test plan
Cover soft ties, temperature extremes, negative outcomes, propensity clipping, zero/poor support, effective sample size, deterministic candidate evaluation, exploration exclusions, and stale pool/pricing rejection.

## Done criteria
- [ ] Hard selected-worker labels are no longer treated as optimal counterfactual labels.
- [ ] Candidate reports include support and uncertainty diagnostics.
- [ ] Exploration is opt-in, bounded, top-two only, and disabled for risky/mutating tasks.
- [ ] No candidate promotes on accuracy alone.
- [ ] Full Python verification passes.

## STOP conditions
- Logged records lack reliable action propensities.
- Effective sample size is below the predeclared gate.
- Counterfactual evaluation would rerun mutating tools or expose private task text to a new provider.

## Maintenance notes
Policy evaluation assumptions must be in every report. Prices and provider behavior drift; stale data should be downweighted or rejected, not silently pooled.
