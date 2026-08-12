# Plan 018: Prototype revisable and safely parallel workflows

> **Executor instructions**: Prototype behind an experimental mode. Never parallelize repository mutations.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- apps/api tests/test_serve.py eval scripts/retrain_conductor.py`. STOP on Conductor state-machine drift.

## Status
- **Priority**: P3
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: plans 008, 012, and 017
- **Category**: direction / perf
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Current Conductor commits to a complete workflow and executes nodes one at a time. It cannot cancel stale work or revise dependencies after tool feedback, and independent read-only nodes cannot reduce wall-clock time through parallel execution.

## Current state
- `apps/api/runs.py:863-865` stores one fixed workflow, outputs, and a sequential node pointer.
- `apps/api/runs.py:878-912` builds node context from fixed access lists.
- `apps/api/runs.py:1092-1125` advances one pending model action at a time.
- Conductor remains empirically weak: `eval/report-native-v2.md:65-90` recommends against deployment.

## Scope
**In scope**: experimental scheduler, read-only frontier parallelism, bounded replanning, cancellation, deterministic fake-worker tests, held-out report.

**Out of scope**: default production use, parallel writes/tests that mutate shared state, unlimited recursion, full Conductor retraining.

## Steps
1. Classify workflow nodes conservatively as read-only or mutating from declared tool permissions. Unknown means mutating.
2. Execute only independent read-only ready-frontier nodes concurrently with strict concurrency and dollar limits. Serialize mutations.
3. Permit one bounded replan after material failure/contradiction; planner may add, cancel, or rewrite only unstarted nodes. Validate the new DAG.
4. Preserve node-local tool transcripts and declared access-list communication. Use plan-017 ledger only for permitted shared artifacts.
5. Compare success, wall-clock, cost, cancellations, and conflicts against sequential Conductor and TRINITY. Do not promote unless Conductor first meets plan 008's quality gate.

## Test plan
Independent reads, dependency ordering, mutation serialization, unknown-tool serialization, cancellation, one replan, invalid revised DAG, budget exhaustion, and deterministic event ordering.

## Done criteria
- [ ] Experimental only; default behavior unchanged.
- [ ] No concurrent mutation is possible.
- [ ] Replanning and concurrency are strictly bounded.
- [ ] Report separates latency gains from added token cost.
- [ ] Full verification passes.

## STOP conditions
- Tool effects cannot be classified conservatively from the available schema.
- Parallel subagents contend on shared mutable environment state.
- Sequential Conductor still fails the quality gate; keep this as a simulator-only prototype.

## Maintenance notes
Scheduler correctness matters more than throughput. Review every new tool's effect classification before allowing parallel use.
