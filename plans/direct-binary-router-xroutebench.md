# Direct binary router: hy3-free vs subscription DeepSeek V4 Flash

**Status:** development protocol. This changes only the Mantis direct router; Trinity and Conductor are explicitly out of scope.

## Question

Can an offline-learned policy decide when the low route (`opencode-zen/hy3-free`) is sufficient and when to escalate to the high route (`opencode-go/deepseek-v4-flash`), improving task-level resolution over random routing at the same escalation rate while using fewer high-route calls than always-high?

## Routing and data boundary

All candidate execution goes through the authenticated Bifrost OpenAI-compatible endpoint. Collection requests use explicit provider-qualified model IDs; they never use `model=mantis`, `model=auto`, the OpenRouter free router, a cloaked model, or a direct provider endpoint. Bifrost remains responsible for credentials, provider protocol normalization, usage, and served-model identity.

Allowed data is public, revision-pinned benchmark content and generated trajectories. Production prompts, private repositories, user sessions, and credentials are excluded.

## Frozen arms

| Arm | Bifrost model | Meaning |
|---|---|---|
| low | `opencode-zen/hy3-free` | free/default attempt |
| high | `opencode-go/deepseek-v4-flash` | subscription-backed escalation |

The fixed shadow-price snapshot is `eval/prices/binary-hy3-deepseek-2026-08-15.json`. Actual charge and shadow cost remain separate fields. A live preflight must verify both exact requested IDs and returned served IDs before collection.

## Prediction target

Train a calibrated probability, not an ordinal rubric:

```text
p_low = P(low route resolves | task context)
```

The decision threshold is selected on validation data to maximize a preregistered utility or satisfy a high-route-call budget. It is not fixed at 0.5.

Why probability rather than 1–5 or 1–10:

- the deployment action is binary;
- a probability preserves uncertainty without quantization;
- thresholds can move along the cost/quality curve without retraining;
- BCE/log loss and probability calibration are directly measurable;
- HPSS's result that a 1–10 scale can outperform a 1–3 scale concerns LLM-as-a-judge pointwise grading, not ex-ante binary coding-agent success prediction.

An ordinal difficulty score may be evaluated as an auxiliary feature or ablation, but it is not the primary target.

## Framework strategy

Use xRouteBench and the LLMRouter paper/library as an **offline methodology benchmark**, not as the production runtime:

- reproduce kNN, SVM, and small MLP on xRouteBench generic;
- use one pinned query encoder consistently in training and serving;
- compare calibrated binary probability prediction against ordinal 1–5 and 1–10 ablations;
- do not install `llmrouter-lib` into the Mantis runtime or replace Bifrost;
- deploy only an immutable artifact-backed decider at `apps/gateway/server.py::_decide_uncached` behind `MANTIS_ROUTER_DECIDER=xroute`;
- retain current session affinity, failover, protocol handling, telemetry, and safe fallback;
- Trinity and Conductor remain unchanged.

## Splits

### xRouteBench architecture benchmark

Pin `ulab-ai/xRouteBench` at revision `8b255161c255239ec84f2b7a740e8b7cd8ebb646`.

- Use `llmrouter_generic` for reproduction against its historical 18-model matrix.
- Derive validation once from its training queries at the unique-query level, stratified by `task_name`.
- Tune only on train/validation.
- Freeze one router design and evaluate the official test split once.
- Never hill-climb the public test split.

This benchmark selects the routing algorithm. Its historical model labels are not used to route current Bifrost models.

### Mantis binary collection

Use a fresh public SWE-rebench revision/manifest, repository-disjoint from `eval/router_manifest.json` and `eval/manifests/zen-pilot-12.json`.

| Partition | Target tasks | Purpose |
|---|---:|---|
| Development | 80–100 | verify high-route rescue rate and protocol |
| Training | 400–600 | fit router |
| Validation | 100–150 | threshold, calibration, hyperparameters |
| Final holdout | 150–250 | one frozen evaluation |

Before broad collection, proceed only if the development screen shows a meaningful routing boundary: at least 10–15 `low fail / high pass` tasks per 100, high-arm operational error below 10%, and complete accounting on at least 95% of calls.

## Labels

Every task is run against both frozen arms in fresh worktrees/sessions:

| Low | High | Interpretation |
|---|---|---|
| pass | pass | low sufficient |
| pass | fail | low sufficient |
| fail | pass | escalation rescue (critical positive boundary) |
| fail | fail | neither; train an abstention/none auxiliary target or retain for utility evaluation |

Provider/transport failures remain operational failures. Capability sensitivity may separately analyze complete opportunities, but failures are never silently dropped.

## Candidate routers

Start with small baselines:

1. constant always-low / always-high;
2. random-matched at the learned high-route rate;
3. simple heuristic;
4. kNN on pinned query embeddings;
5. RBF SVM with probability calibration;
6. small regularized MLP with sigmoid output.

Primary output is calibrated `p_low`. Tune class weighting only if needed. Apply Platt or isotonic calibration using validation only.

## Metrics

Calibration/ranking:

- log loss;
- Brier score;
- expected calibration error plus reliability plot;
- AUROC and especially AUPRC;
- precision/recall for escalation rescues.

Router utility on untouched tasks:

- paired resolved-rate difference versus random-matched;
- resolution relative to always-high;
- high-route call rate;
- shadow/actual cost per resolved task;
- latency and operational error rate;
- paired task bootstrap confidence intervals.

## Success criteria

On the untouched repository-disjoint holdout:

1. paired resolution improvement over random-matched at the same high-route frequency has 95% bootstrap CI lower bound above zero;
2. cost/latency per resolved task improves over always-high;
3. resolution is no worse than always-high beyond a preregistered non-inferiority margin;
4. operational errors are no worse by more than 2 percentage points;
5. the result is reproducible from the frozen artifact, encoder, split hashes, Bifrost IDs, and scoring code.

## Production boundary

The offline artifact contains encoder revision, prompt extraction policy, class mapping, estimator state, calibration map, threshold, candidate model identities, split/scorer hashes, and metrics. Serving validates the artifact against the current catalog; mismatch or inference failure falls back to the existing `_safe_target()` behavior.
