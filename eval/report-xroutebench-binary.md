# xRouteBench binary free-vs-paid router: offline methodology results

**Date:** 2026-08-15 · **Dataset:** `ulab-ai/xRouteBench` `llmrouter_generic` @ rev `8b255161c255239ec84f2b7a740e8b7cd8ebb646` (pinned by `plans/direct-binary-router-xroutebench.md`) · **Script:** `eval/xroutebench_binary_bench.py` · **Raw results:** `eval/results-xroutebench-binary.json`

## Question

Can the LLMRouter/xRouteBench methodology (arXiv:2608.06867) decide when a **free** model suffices and when a **paid** model is needed — the offline analog of `opencode-zen/hy3-free` vs `opencode-go/deepseek-v4-flash` — using the benchmark's precomputed 18-model performance matrix instead of collecting paired labels?

## Setup

- Queries: 4,487 unique train / 3,729 test (13 generic tasks: mbpp, human_eval, math, gsm8k, aime, mmlu, mmlu_pro, squad, boolq, hellaswag, arc_challenge, openbook_qa, commonsense_qa).
- Embeddings: `allenai/longformer-base-4096` mean-pooled, 1024 tokens — the encoder used inside `llmrouter-lib`'s KNNRouter. Computed once on Apple MPS (~30 min).
- Pools: cheap (8 models ≤ ~$0.30/M in, e.g. qwen2.5-7b, gpt-oss-20b) vs expensive (9 models, e.g. deepseek-v3.1, llama-4-maverick), Together 2026-08 list prices.
- Labels: `y_low = 1` iff any cheap model achieves performance ≥ 1 (task-correct).

## Data boundary (test split)

| Low pool | High pool | Share |
|---|---|---:|
| resolves | resolves | 82.0% |
| resolves | fails | 5.2% |
| fails | resolves (**rescue pool**) | 5.2% (195 queries) |
| fails | fails | 7.5% |

## Part 1 — Paper-style KNNRouter reproduction

Routed average performance on test (higher is better):

| Router | Avg perf |
|---|---:|
| KNNRouter (k=5, argmax-performance labels, Longformer) | 0.596 |
| Smallest-LLM (cheap pool oracle-max) | 0.876 |
| Largest-LLM (expensive pool oracle-max) | 0.908 |
| Pool oracle | 0.929 |

Note: these pool columns are per-pool oracle maxima, not single-model baselines, so they are upper bounds; the single-model numbers the paper reports are lower. Still, the argmax-label KNNRouter badly underperforms even the cheap pool on this split — nearest-neighbor transfer of "which specific model is best" does not survive the train→test task mix shift (mbpp/mmlu_pro/aime dominate test; small 7B models dominate the argmax labels).

## Part 2 — Binary calibrated routers: `p_low = P(low resolves | query)`

| Router | Log loss | Brier | ECE | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|
| kNN + Platt | 0.383 | 0.112 | 0.016 | 0.521 | 0.883 |
| SVM-RBF + Platt | 0.383 | 0.112 | 0.019 | 0.525 | 0.883 |
| SVM-RBF + isotonic | 0.383 | 0.112 | 0.014 | 0.524 | 0.880 |
| MLP + Platt | 0.384 | 0.112 | 0.029 | 0.513 | 0.872 |
| MLP + isotonic | 0.387 | 0.113 | 0.030 | 0.514 | 0.872 |

AUPRC ≈ base rate (0.872) for all routers — no ranking lift over the class prior.

**Interpretation:** AUROC ≈ 0.52 across every architecture — statistically indistinguishable from chance ranking of which queries need escalation. Calibration is excellent (ECE 0.014–0.030) but only because all routers learned to predict the base rate (~0.87). With a 95/5 class split and Longformer mean-pooled embeddings, **the escalation signal is not learnable from query text alone on this benchmark.**

## Part 3 — Policy sweep (best router, kNN+Platt)

Baselines: always-low 0.872 perf / $6.6e-6 per task · always-high 0.904 / $41e-6 · random-matched (14.7% esc) 0.875 / $10e-6. Rescue pool = 195 queries.

| Thr | Esc rate | Perf | Cost/task | Rescues caught (of 195) |
|---:|---:|---:|---:|---:|
| 0.80 | 0.9% | 0.873 | $6.9e-6 | 2 |
| 0.85 | 30.2% | 0.883 | $16.7e-6 | 69 |
| 0.90 | 100% | 0.904 | $41.4e-6 | 195 (all) |

At no threshold does the learned router dominate random-matched: e.g. thr=0.85 catches 69/195 rescues at 30% escalation versus random-matched's ~29 expected rescues at 14.7% — but rescue precision is only 6%, and the perf-per-cost curve is dominated by simply choosing always-low or always-high. Bootstrap CIs were not computed because the point estimate already fails the plan's success criterion 1.

## Conclusions

1. **xRouteBench works as a free label source** for the binary free-vs-paid question — no paired collection needed to test the methodology, exactly as the plan's "architecture benchmark" section intended.
2. **The negative result is informative:** on generic chat/code/math QA, whether a query exceeds small-model capability is not predictable from the raw query embedding (AUROC ~0.52). The paper's positive results use task-mixed training with much richer supervision; our binary split shows the ceiling for embedding-only binary routing on this distribution.
3. **Implication for the Mantis binary router plan:** do not skip the paired SWE-rebench collection on the grounds that offline labels suffice. The plan's development screen (10–15 rescue tasks per 100 on real coding-agent tasks) remains the necessary evidence of a learnable routing boundary; xRouteBench's generic distribution (85–87% low-sufficient, signal-poor text) suggests coding-agent prompts differ enough that labels must come from the actual arm pair.
4. Cost note: absolute costs here are single-turn QA costs (~$1e-5), far below agentic SWE costs (~$0.5–1.5/instance); relative escalation economics, not absolute dollars, transfer.

## Reproduction

```bash
# venv with torch, transformers, sklearn, pandas, pyarrow (Python 3.10)
python eval/xroutebench_binary_bench.py \
  --train-parquet xrb_train.parquet --test-parquet xrb_test.parquet \
  --train-embed xrb_train_embed.pt --test-embed xrb_test_embed.pt \
  --output eval/results-xroutebench-binary.json
```

Dataset files are `llmrouter_generic` train/test at the pinned revision; embeddings are Longformer mean-pooled (768-d) keyed by `embedding_id`, generated with the same encoder/seed (`SEED=20260815`) as the run above.
