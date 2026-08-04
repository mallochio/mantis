# Plan 014: Learn per-turn roles, stopping, escalation, and budgets

> **Executor instructions**: Begin as an offline/shadow policy. Do not raise global turn limits.
>
> **Drift check**: `git diff --stat 3cccee1..HEAD -- openfugu-patch/serve.py scripts/learn_router.py scripts/retrain_router_pool.py tests eval`. STOP unless plans 008 and 012 are complete.

## Status
- **Priority**: P2
- **Effort**: L
- **Risk**: HIGH
- **Depends on**: plans 008, 011-013
- **Category**: direction / perf
- **Planned at**: commit `3cccee1`, 2026-08-04

## Why this matters
Mantis routes worker and role every TRINITY turn, but online learning updates only initial-task worker logits. It cannot learn when verification is worth its cost, when tests justify stopping, or when remaining budget should force a cheaper action.

## Current state
- `openfugu-patch/serve.py:1759-1778` routes role and worker each turn, with hand-written cold-role guards.
- `openfugu-patch/serve.py:1835-1848` handles Thinker suggestions and Verifier accept/reject.
- `openfugu-patch/serve.py:1936-1948` has only terminal acceptance or max-turn stopping.
- `scripts/retrain_router_pool.py:1076-1097` copies role labels from the existing router.

## Scope
**In scope**: offline action/state representation, shadow policy, bounded runtime canary, tests/evaluation.

**Out of scope**: increasing `MANTIS_MAX_TURNS`, unbounded recursion, changing the backbone, removing safety verification from high-risk tasks.

## Steps
1. Define actions: worker-role pairs plus `STOP`, `VERIFY`, `REVISE`, and `ESCALATE`. Condition on decision-level state, previous outcomes, tool/test state, uncertainty, and remaining budgets.
2. Build training rows from plan-012 trajectories with discounted terminal utility and explicit cost/latency penalties. Mark actions with unknown credit rather than assigning terminal success wholly to the final worker.
3. Shadow-log recommended action and estimated marginal value of another turn.
4. Add a canary that may stop earlier or escalate once, but may not exceed existing turn/tool limits. High-risk tasks retain conservative verification.
5. Compare against fixed-turn TRINITY on held-out quality, calls, tokens, cost, latency, and failure modes.

## Test plan
Cover cold start, no worker answer, verifier accept/reject, successful and failed tests, zero budget, stop, one escalation, max-turn preservation, and high-risk verification.

## Done criteria
- [ ] Role labels come from measured trajectories, not the prior router.
- [ ] The policy can abstain and fall back to current logic.
- [ ] Turn/tool maxima do not increase.
- [ ] Held-out utility improves under plan 008's gate.
- [ ] Full verification passes.

## STOP conditions
- Trajectory credit assignment is too sparse to beat the current rules offline.
- Earlier stopping causes a quality regression outside tolerance.
- Runtime action changes can bypass required verification or tests.

## Maintenance notes
Report action frequencies and termination reasons. A policy that always stops or always verifies is collapsed even if aggregate utility looks acceptable.
