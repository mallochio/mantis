# Plan 019: Search workflows offline and distill a step-wise policy

> **Executor instructions**: This is the final, expensive experiment. Begin with cached/fake workers and require a written spend estimate before live rollouts.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- scripts/retrain_conductor.py scripts/retrain_router_pool.py eval tests openfugu-patch/serve.py`. STOP unless all dependencies have stable schemas and gates.

## Status
- **Priority**: P3
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: plans 008, 013-018
- **Category**: direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
One-shot natural-language workflow generation is expensive and brittle. Offline search over worker, role, prompt, dependencies, and stopping can discover workflows; a compact step-wise policy can then imitate only stable high-utility decisions at runtime.

## Current state
- `scripts/retrain_conductor.py:179-250` already parses, structurally rewards, and executes complete DAG candidates.
- `scripts/retrain_router_pool.py:586-658` already caches worker responses.
- Runtime Conductor commits to a complete plan before feedback (`openfugu-patch/serve.py:2023-2050`).

## Scope
**In scope**: offline search simulator, stability-weighted trajectory dataset, compact step-wise policy prototype, reports/tests.

**Out of scope**: immediate full GRPO retraining, unbudgeted live search, production rollout, arbitrary generated executable code.

## Steps
1. Define a finite action space from validated runtime primitives: worker, role, prompt adapter, dependency/access choice, stop, and escalate.
2. Build a deterministic simulator over cached outcomes first. Add limited live rollout only for states lacking support, with operator-approved budget.
3. Search candidate workflows with a bounded beam/tree search using quality, cost, latency, validity, and stability across repeats.
4. Keep only trajectories stable across seeds/repeats; weight supervision by stability and utility.
5. Train the smallest step-wise policy and compare it with one-shot Conductor, TRINITY, and fixed templates. Produce a go/no-go report before any runtime integration.

## Test plan
Action validation, cache reuse, search budget, duplicate-state pruning, unstable trajectory rejection, deterministic tie-breaking, policy step validity, and no-live-call default.

## Done criteria
- [ ] Default search uses cached/fake outcomes and spends $0.
- [ ] Live rollout has a hard declared budget and operator approval.
- [ ] Distilled policy beats fixed baselines on held-out utility and workflow validity before integration is proposed.
- [ ] No production runtime changes occur in this plan.
- [ ] Offline verification passes.

## STOP conditions
- Search estimates cannot be validated against real outcomes on a small sample.
- Worker response caching would reuse outputs across materially different states.
- Expected spend is not approved.
- The distilled policy cannot beat simple templates/TRINITY.

## Maintenance notes
Archive search provenance, pool/pricing versions, and stability metrics. A successful research checkpoint still needs a separate production-hardening plan.
