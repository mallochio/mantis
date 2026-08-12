# Plan 016: Spike query-model compatibility scoring

> **Executor instructions**: This is a design/prototype plan, not a production migration. Keep the existing seven-slot head intact.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- scripts configs eval tests artifacts`. STOP unless plans 008 and 015 provide stable datasets and masks.

## Status
- **Priority**: P3
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: plans 008, 013, and 015
- **Category**: direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Fixed slot logits cannot naturally generalize to a newly released model, changed price, or smaller available pool. A query-model compatibility scorer could rank arbitrary candidates from model capability, price, latency, and availability features.

## Current state
- `scripts/learn_router.py:107-108` scores the first seven fixed head rows.
- `scripts/retrain_router_pool.py:509-520` trains one fixed 10x1024 head.
- `config/worker-costs.json` has cost estimates but no capability or latency representation.

## Scope
**In scope**: offline model-card schema, prototype scorer, leave-one-model-out evaluation, report and tests.

**Out of scope**: replacing production routing, automatically trusting provider marketing metadata, new hosted vector databases, model weight training.

## Steps
1. Specify minimal model features: stable ID, capability outcomes learned from Mantis evaluations, observed cost/latency distributions, tool support, context limit, availability, and privacy/provider class.
2. Prototype the simplest scorer that combines query embedding/hidden state with model features. Compare a linear/bilinear scorer against fixed-slot and price-only baselines; do not add a deep head unless evidence requires it.
3. Run leave-one-model-out and changed-price evaluations to test cold-start claims.
4. Test arbitrary masks and pool sizes offline.
5. Produce a go/no-go report. Production integration is a separate plan only if the scorer beats the fixed head on held-out utility and calibration.

## Test plan
Schema validation, unseen model, missing feature abstention, changed price, unavailable model, deterministic ranking, and stale feature rejection.

## Done criteria
- [ ] No production path changes.
- [ ] Leave-one-model-out results include uncertainty and baselines.
- [ ] A no-go result is acceptable and documented.
- [ ] Offline tests and report generation pass.

## STOP conditions
- Model features rely mainly on unverified vendor claims.
- There are too few evaluated models/tasks for leave-one-model-out analysis.
- The prototype cannot beat price-only or fixed-slot baselines.

## Maintenance notes
Do not market this as hot-swappable until unseen-model evaluation supports it. Observed capability data expires with model revisions.
