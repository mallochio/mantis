# Router quality/cost ladder reassessment

**Status:** paused. The initial baseline run was stopped after 2 mantis-direct instances because per-instance latency (~19 minutes) made the full 160-instance, 4-arm study impractical to complete in one session.

**Context:** `2ba5a96` introduced a cost-monotonic gateway ladder and a new `complexity_efforts` policy field. The catalog now maps Supra complexity 1–5 to `(target, effort)` pairs with three operating points on the cheap model. The hypothesis is that this improves the quality/cost ratio over the previous 3-target, fixed-effort mapping.

## Question

Does the new cost-monotonic, 5-point `(target, effort)` routing policy improve resolution per dollar relative to (a) the previous gateway policy and (b) fixed single-tier arms, on the `nebius/SWE-rebench-leaderboard` 40-instance manifest?

## What is already done

- `apps/gateway/server.py` supports `complexity_efforts` in policy tables; effort overrides apply to the primary route only.
- `config/catalog.toml` declares the new ladder:
  - cheap: `google/gemini-3.7-flash`
  - middle: `anthropic/claude-opus-5` ($5/$25 per 1M)
  - expensive: `gpt-5.6-sol` ($5/$30 per 1M)
  - `complexity_efforts = ["low", "medium", "high", "medium", "high"]`
- `eval/model_prices.json` and `eval/router_eval.py` were updated for the new models and endpoint auth.
- `apps/gateway/tests/test_routing.py` and `apps/gateway/tests/test_routing_quality_gate.py` were updated for the new mapping.

## Why the run was stopped

The baseline run (`eval/runs/results-quality-cost-baseline.jsonl`) was launched with:
- arms: `cheap-only,middle-only,expensive-only,mantis-direct`
- 40 instances
- 600s timeout per instance-arm

After 2 mantis-direct instances:
- ~19 minutes elapsed per instance
- $1.55 spent, $76.45 of the $78 cap remaining
- 0 resolved
- projected completion: many hours

The per-instance latency is too high for an interactive validation loop. Before running the full study, we should reassess the evaluation protocol.

## Reassessment plan

### 1. Reduce the validation surface

Run a fast pilot on a small, representative subset (5–10 instances) to get a signal in under an hour. Do not use the full 40-instance manifest until the pilot shows a meaningful spread in resolved/cost between arms.

### 2. Reduce agent latency

The dominant cost is wall-clock time, not money. Options:
- Lower `--timeout` from 600s to 120–240s for the pilot; unresolved/aborted instances are still informative for routing cost comparison.
- Use a shorter prompt / fewer tools if a minimal-mode `pi` invocation exists.
- Disable worktree recreation where possible by using `--keep-worktrees` and `--resume`.

### 3. Reconsider the control arms

The current 4-arm study is too large. A tighter comparison:
- `mantis-direct` (new policy)
- `cheap-only` (lower bound on cost)
- `expensive-only` (upper bound on quality)

Skip `middle-only` in the pilot; add it back only if the pilot shows the new policy lands between cheap and expensive.

### 4. Verify the agent is healthy on one instance end-to-end

The first two mantis-direct instances spent $0.83 each but resolved 0 tasks. Confirm that `pi` can actually resolve at least one cheap and one expensive instance with the current toolchain before running a larger comparison. If zero-resolution is systematic, routing cannot be measured.

### 5. Capture cost trace, not just resolution

Even unresolved trajectories yield per-request cost and route-decision data. The pilot should be analyzed for:
- average cost per instance by arm
- route-decision distribution (`x-route-decision` headers)
- effort-override usage (`reasoning_effort` in the proxy's outgoing bodies, if inspectable)
- failure modes (timeout, budget cap, 4xx/5xx, refusal)

### 6. Decide on full run go/no-go

Proceed to the full 40-instance, 4-arm study only if the pilot shows:
- at least one instance resolved by `mantis-direct` or `expensive-only`;
- `mantis-direct` average cost is between `cheap-only` and `expensive-only`;
- no systematic 4xx/5xx or auth issues;
- wall-clock time per instance is acceptable (e.g., < 5 minutes for the pilot).

## Proposed pilot command

```bash
cd /Users/sid/Personal/other/mantis
uv run python eval/router_eval.py \
  --manifest eval/router_manifest.json \
  --arms cheap-only,expensive-only,mantis-direct \
  --tier-models "cheap=google/gemini-3.7-flash,middle=anthropic/claude-opus-5,expensive=gpt-5.6-sol" \
  --budget-usd 20 \
  --output eval/runs/results-quality-cost-pilot.jsonl \
  --timeout 180 \
  --output-token-limit 4096
```

Modify the manifest to contain only the first 5–10 instances, or filter with `--instance-ids` if that option is added.

## Success criteria for the new policy

Only meaningful once the pilot passes:

1. `mantis-direct` resolves at least as many instances as `cheap-only` at a cost no higher than `expensive-only`.
2. The `mantis-direct` per-resolved cost is lower than `expensive-only`.
3. `mantis-direct` actually uses multiple target/effort operating points (not 100% expensive fallback).
4. No increase in 5xx, refusal, or timeout rate versus fixed arms beyond 5 percentage points.

## Files to inspect before resuming

- `eval/runs/results-quality-cost-baseline.jsonl` — partial, 2 mantis-direct instances
- `config/catalog.toml` — new policy, `[gateway]` and `[gateway.policies.coding]`
- `apps/gateway/server.py` — `SUPRA_EFFORTS`, `_effort_for_complexity`, `_apply_effort_override`
- `apps/gateway/tests/test_routing.py` — effort-override unit tests

## Open questions

- Is the current `pi` invocation with `read,bash,edit,write` the minimal viable tool set, or should a two-tool mode be used for faster iteration?
- Should the `mantis-direct` arm route through the Mantis API at `:8088` or the gateway directly at `:5500`? The current setup routes through the Mantis API (`mantis/base`), which adds a hop and model-to-target translation.
- Do we need a separate shadow-price snapshot for the new non-Zen models, or is `eval/model_prices.json` sufficient?
