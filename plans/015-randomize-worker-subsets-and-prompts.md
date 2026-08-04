# Plan 015: Train worker-subset robustness and worker-specific prompt adapters

> **Executor instructions**: Run this as two independently reportable experiments sharing the same evaluation gate. Do not change worker model weights.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- scripts/retrain_router_pool.py openfugu-patch/serve.py configs tests eval`. STOP on worker-pool schema drift.

## Status
- **Priority**: P2
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans 008 and 013
- **Category**: direction
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
The current head is tied to seven positions and runtime role prompts are mostly shared. Random subset masking can make routing resilient to unavailable/expensive workers; small per-worker prompt adapters may expose complementary strengths without retraining models.

## Current state
- `scripts/learn_router.py:168-170` requires exactly seven pool entries.
- `scripts/learn_router.py:55-61` accepts records only when the entire pool exactly matches.
- `openfugu-patch/serve.py:1756-1757` maps fixed slot IDs to models.
- `openfugu-patch/serve.py:1780-1821` builds shared role prompts.

## Scope
**In scope**: masked offline training/evaluation, small checked-in prompt adapter data, runtime adapter lookup, focused tests.

**Out of scope**: dynamic-size neural architecture, model fine-tuning, prompt search in production, adding more workers by default.

## Steps
1. Add availability masks to offline examples and candidate scoring. Masked workers must receive no probability and no call.
2. Train/evaluate with randomized subsets while retaining the seven-slot artifact format. Test outages, local-only subsets, and price exclusions.
3. Define minimal worker-role prompt suffixes in one data file. Start with hand-authored empty/default adapters plus a few evidence-backed variants; no abstraction beyond a lookup table.
4. Evaluate adapters independently per worker/task family, then jointly with masked routing. Retain only variants that improve held-out utility.
5. Add runtime fallback to the unmodified shared prompt when an adapter is absent or stale.

## Test plan
Cover all workers available, one unavailable, only one available, no available workers, mask normalization, no masked dispatch, adapter present/absent, and prompt privacy.

## Done criteria
- [ ] Runtime survives worker availability changes without remapping slots.
- [ ] No masked worker can be selected.
- [ ] Prompt adapters are small, versioned, and removable.
- [ ] Each experiment has a separate held-out report.
- [ ] Full verification passes.

## STOP conditions
- Subset robustness requires changing the fixed artifact format; defer that to plan 016.
- Prompt variants improve only the tuning set.
- Adapters need repository/task text persisted as templates.

## Maintenance notes
Model aliases, prices, and adapter behavior drift. Remove adapters that cease to help; do not accumulate prompt folklore.
