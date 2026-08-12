# Implementation Plans

Routing research plans for the model orchestrator. Execute by dependency, not merely by number.

## Execution order & status

| Plan | Title | Priority | Effort | Depends on | Status |
|---|---|---:|---:|---|---|
| 008 | Establish a representative routing evaluation and promotion gate | P1 | S | — | TODO |
| 009 | Shadow-log a three-way expected-utility router | P1 | S | 008 | TODO |
| 011 | Calibrate routing uncertainty and cheap-first escalation | P1 | M | 008, 009 | TODO |
| 012 | Record decision-level outcomes for learning | P1 | M | 008 | TODO |
| 013 | Replace hard imitation with soft and counterfactual policy learning | P1 | L | 008, 012 | TODO |
| 014 | Learn per-turn roles, stopping, escalation, and budgets | P2 | L | 008, 011-013 | TODO |
| 015 | Train worker-subset robustness and worker-specific prompt adapters | P2 | M | 008, 013 | TODO |
| 016 | Spike query-model compatibility scoring | P3 | L | 008, 013, 015 | TODO |
| 017 | Add a compact artifact ledger for cross-agent memory | P2 | M | 008, 012 | TODO |
| 018 | Prototype revisable and safely parallel workflows | P3 | L | 008, 012, 017 | TODO |
| 019 | Search workflows offline and distill a step-wise policy | P3 | L | 008, 013-018 | TODO |
| 020 | Prune derived eval artifacts and document the eval/ artifact boundary | P2 | S | — | DONE |
| 021 | Single-source the conductor training dependency manifest | P2 | S | — | DONE |
| 022 | Decompose router/server.py into cohesive modules without behavior change | P1 | M | — | TODO |

Status: TODO | IN PROGRESS | DONE | BLOCKED (with reason) | REJECTED (with rationale).

## Cheapest-first recommendation

1. **008 evaluation gate** — cheapest prerequisite; fixes the weak 16-task/keyword evidence base before policy work.
2. **009 shadow utility router** — cheap and zero behavior risk; measures direct/TRINITY/Conductor recommendations without spending on extra calls.
3. Start **010 direct route** and **012 decision telemetry** in parallel after 008. They touch overlapping runtime files, so use separate branches/worktrees and merge 010 first, then rebase 012.
4. Start **011 uncertainty/escalation** after 009. In parallel, begin offline portions of **013 soft/counterfactual learning** after 012.
5. After 013, run **015 worker masks/prompt adapters** and **017 artifact ledger** in parallel; they target different mechanisms but both use the shared evaluation gate.
6. Run **014 learned stopping** only after calibration and decision-level learning are credible.
7. Treat **016**, **018**, and **019** as research spikes. Do not put them on the production critical path.
8. **Repo-slimming lane (020-022, planned 2026-08-12 at `168fedb`)**: 020 then 021 first (S effort, zero risk), then 022 (M effort, the flagship). 022 is independent of 008-019 but its module names (`routing.py`, `scoring.py`, `store.py` under `router/`) should be referenced by 008-019 executors once it lands.

## Parallel work graph

```text
008 evaluation gate
 ├─ 009 shadow utility ─ 011 calibrated escalation ─┐
 └─ 012 decision telemetry ─ 013 soft/bandit learning ─────────────────┤
                                  ├─ 015 masks + prompt adapters ─ 016 spike
                                  ├─ 017 artifact ledger ─ 018 workflow spike
                                  └─────────────────────────────── 014 stop policy
                                                        013-018 ─ 019 distillation
```

Recommended lanes after 008:

- **Lane A — immediate savings:** 009 → 011.
- **Lane B — learning correctness:** 012 → 013 → 014.
- **Lane C — cheap orchestration efficiency:** 015 and 017.
- **Lane D — moonshots:** 016, 018, then 019 only after earlier gates.


## Promotion gates

No routing policy becomes the default unless all apply:

- Evaluated on the same versioned held-out workload with representative task families and risk classes.
- Quality regression is within a predeclared tolerance with paired uncertainty reported.
- Cost or latency improves materially; actual usage is preferred over fixed token estimates.
- Security/high-risk failure rate does not increase; exploration is disabled for these tasks.
- Policy can abstain and fall back conservatively.
- Shadow evaluation precedes canary; canary precedes default promotion.
- Model pool, prices, prompts, calibration, policy, fixtures, and code are version-stamped.

## Verification baseline

```bash
uv run pytest tests -q
uv run ruff check .
uv run mypy openfugu-patch scripts --exclude outputs
git diff --check
```

Use focused tests during each step, then run the full baseline before marking a plan DONE. Paid evaluation commands require operator approval and a written spend bound.

## Dependency notes

- 008 is intentionally first: the current result set is too small to authorize default-policy changes.
- 009 observes only; it must not alter current `/auto` selection.
- - 012 records failures, propensities, budgets, and per-decision state before 013 attempts off-policy learning.
- 011 and 013 are independent after their prerequisites and can proceed concurrently.
- 014 consumes calibrated route confidence and decision-level training data.
- 015 deliberately retains the fixed seven-slot artifact; 016 is the separate arbitrary-pool research spike.
- 017 must preserve Conductor access-list isolation; 018 depends on that invariant.
- 019 is last because cached outcomes, stable telemetry, adapters, stopping actions, and workflow evaluation all improve its search space and safety.

## Findings considered and rejected or deferred

- **Larger/nonlinear router head now**: rejected. The strongest paper evidence favors the simple linear head; improve supervision and policy scope first.
- **Globally increase maximum turns**: rejected. Learn marginal compute allocation instead; current cost and latency are already high.
- **Default all-model voting**: rejected. It spends on every task and is difficult to justify for tool-using coding sessions.
- **Another full local Conductor retrain now**: rejected. Current local evaluation is poor and workflow validity must improve before more expensive training.
- **Production parallel Conductor now**: deferred to plan 018 and blocked on Conductor meeting the evaluation gate.
- **Unlimited recursive replanning**: rejected. Any replan remains bounded to one experimentally.
- **Backbone fine-tuning**: deferred. Head/prompt/workspace changes are cheaper and easier to verify.

### Repo-slimming audit (2026-08-12, planned at `168fedb`)

- **Merge `router/` into the root uv project (single lockfile)**: rejected. `router/README.md` states the split is deliberate so the router deploys without torch/transformers/fastapi serving deps; merging would inflate the router's dependency and deploy footprint, directly hurting its cost side of quality-to-cost.
- **Delete raw `eval/results*.jsonl` snapshots**: deferred. Plan 008's promotion gate requires checked-in results for offline report regeneration; revisit after 008 lands.
- **Merge `router/server.py` provider-calling code with `openfugu-patch/providers.py`**: rejected after audit. They are not duplicates: the router is a simple forwarder, providers.py does multi-role protocol translation for TRINITY/Conductor; a merge would couple the two deployment footprints. Plan 022 checks for byte-level shared helpers after decomposition and only then considers a tiny shared primitives package.
- **`scripts/model_catalog*.py` decomposition**: rejected. Already cleanly split (schema/abi/runtime); no action.
- **Delete `launch/sky/*.yaml` SkyPilot retrain jobs**: deferred to operator decision. Plan 021 records that their setup steps reference a missing `scripts/fetch_openfugu.sh` and the removed OpenFugu submodule (e.g. `retrain_fugu_conductor.yaml:40`); restore the submodule or retire the path.

## Evidence caveats

The immediate recommendation is partly based on a 16-task Mantis evaluation and recent 2025-2026 papers with limited independent replication. That is why plan 008 precedes every production promotion. Untracked `runs/logs/*` files may contain session data; do not commit or quote them.
