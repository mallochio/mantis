"""Offline xRouteBench binary-router methodology experiment.

Two evaluations on ulab-ai/xRouteBench llmrouter_generic (rev 8b25516):
1. Paper-style KNNRouter reproduction: argmax-performance labels, kNN classify
   to best model, report routed avg performance vs Smallest/Largest baselines.
2. Binary calibrated free-vs-paid router (plans/direct-binary-router-xroutebench.md):
   p_low = P(low pool resolves | query) via kNN / RBF-SVM / MLP with Platt or
   isotonic calibration; metrics log loss, Brier, ECE, AUROC, AUPRC; policy
   sweep of escalation threshold vs always-low / always-high / random-matched,
   with cost-aware utility from dataset token counts.

Usage:
  # 1) download pinned dataset revision (files: *_train.parquet, *_test.parquet)
  # 2) embed unique queries with allenai/longformer-base-4096 (mean-pooled, 1024 tok)
  # 3) run:
  python eval/xroutebench_binary_bench.py \
      --train-parquet xrb_train.parquet --test-parquet xrb_test.parquet \
      --train-embed xrb_train_embed.pt --test-embed xrb_test_embed.pt \
      --output eval/results-xroutebench-binary.json
"""
import argparse
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 20260815
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Free/cheap pool (~<= $0.30/M input on Together, 2026-08) vs paid pool.
CHEAP = [
    "mistral-7b-instruct-v0.3", "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct-turbo", "llama-3-8b-instruct-lite",
    "gemma-2-9b-it", "gpt-oss-20b", "mistral-small-3-24b-instruct",
    "qwen3-coder-next",
]
EXPENSIVE = [
    "deepseek-v3.1", "llama-3.3-70b-instruct-turbo", "llama3-70b-instruct",
    "mixtral-8x22b-instruct-v0.1", "llama-4-maverick", "cogito-v2-1-671b",
    "qwen3-next-80b-a3b-instruct", "gpt-oss-120b", "mixtral-8x7b-instruct-v0.1",
]
# Together-style list prices USD/M tokens (input, output), 2026-08 snapshot.
PRICES = {
    "mistral-7b-instruct-v0.3": (0.05, 0.05),
    "qwen2.5-7b-instruct": (0.05, 0.05),
    "qwen2.5-7b-instruct-turbo": (0.05, 0.05),
    "llama-3-8b-instruct-lite": (0.03, 0.05),
    "gemma-2-9b-it": (0.06, 0.12),
    "gpt-oss-20b": (0.07, 0.25),
    "mistral-small-3-24b-instruct": (0.05, 0.08),
    "qwen3-coder-next": (0.28, 42),
    "deepseek-v3.1": (0.29, 0.39),
    "llama-3.3-70b-instruct-turbo": (0.88, 0.88),
    "llama3-70b-instruct": (0.88, 0.88),
    "mixtral-8x22b-instruct-v0.1": (1.20, 1.20),
    "llama-4-maverick": (0.22, 0.85),
    "cogito-v2-1-671b": (0.80, 2.00),
    "qwen3-next-80b-a3b-instruct": (0.30, 0.38),
    "gpt-oss-120b": (0.13, 0.48),
    "mixtral-8x7b-instruct-v0.1": (0.60, 0.60),
    "rnj-1-instruct": (0.10, 0.10),
}


def load_emb_by_id(embed_path):
    blob = torch.load(embed_path, weights_only=False)
    return {
        i: e.numpy()
        for i, e in zip(blob["embedding_ids"], blob["embeddings"], strict=True)
    }


def query_tables(df):
    """Per unique query: label y (low pool resolves), oracle perfs, costs."""
    piv = df.pivot_table(index="query", columns="model_name", values="performance")
    tok = df.pivot_table(index="query", columns="model_name", values="input_tokens")
    out = df.pivot_table(index="query", columns="model_name", values="output_tokens")
    cheap_cols = [m for m in CHEAP if m in piv.columns]
    exp_cols = [m for m in EXPENSIVE if m in piv.columns]
    low_perf = piv[cheap_cols].max(axis=1)
    high_perf = piv[exp_cols].max(axis=1)

    def pool_cost(row_tok, row_out, cols):
        costs = {}
        for m in cols:
            pin, pout = PRICES[m]
            costs[m] = (row_tok[m] * pin + row_out[m] * pout) / 1e6
        return costs

    rows = []
    for q in piv.index:
        cheap_c = pool_cost(tok.loc[q], out.loc[q], cheap_cols)
        exp_c = pool_cost(tok.loc[q], out.loc[q], exp_cols)
        cheapest_low = min(cheap_c, key=cheap_c.get)
        rows.append({
            "query": q,
            "y_low": int(low_perf.loc[q] >= 1),
            "y_high": int(high_perf.loc[q] >= 1),
            "low_perf": low_perf.loc[q],
            "high_perf": high_perf.loc[q],
            "low_cost": cheap_c[cheapest_low],
            "high_cost": min(exp_c.values()),
        })
    tab = pd.DataFrame(rows).set_index("query")
    first_emb = df.drop_duplicates("query").set_index("query")["embedding_id"]
    return tab, first_emb


def ece(probs, y, bins=15):
    probs, y = np.asarray(probs), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        total += mask.sum() / len(probs) * abs(probs[mask].mean() - y[mask].mean())
    return total


def make_estimators():
    return {
        "knn": KNeighborsClassifier(n_neighbors=15, metric="cosine"),
        "svm-rbf": SVC(kernel="rbf", probability=True, random_state=SEED),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 32), alpha=1e-3, max_iter=800,
            early_stopping=True, random_state=SEED,
        ),
    }


def calibrated_probs(name, est, Xtr, ytr, Xte, method):
    if name == "knn":  # kNN has no native probability; calibrate via Platt CV
        cal = CalibratedClassifierCV(est, method="sigmoid", cv=5)
        cal.fit(Xtr, ytr)
        return cal.predict_proba(Xte)[:, 1]
    if method == "isotonic":
        cal = CalibratedClassifierCV(est, method="isotonic", cv=5)
        cal.fit(Xtr, ytr)
        return cal.predict_proba(Xte)[:, 1]
    est.fit(Xtr, ytr)
    return est.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--train-embed", required=True)
    ap.add_argument("--test-embed", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    t0 = time.time()
    tr_df = pd.read_parquet(args.train_parquet)
    te_df = pd.read_parquet(args.test_parquet)
    tr_tab, tr_first = query_tables(tr_df)
    te_tab, te_first = query_tables(te_df)
    emb_by_id = load_emb_by_id(args.train_embed)
    emb_by_id.update(load_emb_by_id(args.test_embed))
    Xtr = np.stack([emb_by_id[i] for i in tr_first])
    Xte = np.stack([emb_by_id[i] for i in te_first])
    ytr = tr_tab["y_low"].to_numpy()
    yte = te_tab["y_low"].to_numpy()
    print(
        f"[data] train {len(ytr)} ({ytr.mean():.3f} pos) "
        f"test {len(yte)} ({yte.mean():.3f} pos)",
        flush=True,
    )

    results = {"data": {"train_n": int(len(ytr)), "test_n": int(len(yte)),
                        "train_pos_rate": float(ytr.mean()), "test_pos_rate": float(yte.mean())}}

    # ---------- Part 1: paper-style KNNRouter reproduction ----------
    best = tr_df.loc[tr_df.groupby("query")["performance"].idxmax()]
    best_emb = np.stack([emb_by_id[i] for i in best["embedding_id"]])
    best_lab = best["model_name"].to_numpy()
    knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
    knn.fit(best_emb, best_lab)
    perf_lookup = te_df.set_index(["query", "model_name"])["performance"]
    preds = []
    for q, eid in te_first.items():
        m = knn.predict(emb_by_id[eid].reshape(1, -1))[0]
        preds.append(perf_lookup.get((q, m), 0.0))
    smallest = te_tab["low_perf"].mean()
    largest = te_tab["high_perf"].mean()
    routed = float(np.mean(preds))
    oracle = np.maximum(te_tab["low_perf"], te_tab["high_perf"]).mean()
    results["paper_repro"] = {
        "knn_routed_avg_perf": routed,
        "smallest_llm_avg_perf": float(smallest),
        "largest_llm_avg_perf": float(largest),
        "oracle_avg_perf": float(oracle),
    }
    print(
        f"[knn-paper] routed {routed:.4f} smallest {smallest:.4f} "
        f"largest {largest:.4f} oracle {oracle:.4f}",
        flush=True,
    )

    # ---------- Part 2: binary calibrated routers ----------
    results["binary"] = {}
    for name, est in make_estimators().items():
        for method in (["sigmoid", "isotonic"] if name != "knn" else ["sigmoid"]):
            p = calibrated_probs(name, est, Xtr, ytr, Xte, method)
            key = f"{name}-{method}"
            entry = {
                "log_loss": float(log_loss(yte, p)),
                "brier": float(brier_score_loss(yte, p)),
                "ece": float(ece(p, yte)),
                "auroc": float(roc_auc_score(yte, p)),
                "auprc": float(average_precision_score(yte, p)),
            }
            results["binary"][key] = entry
            print(f"[binary {key}] {entry}", flush=True)

    # ---------- Part 3: policy sweep for the best-calibrated router ----------
    best_key, best_entry = None, None
    for key, entry in results["binary"].items():
        if best_entry is None or entry["log_loss"] < best_entry["log_loss"]:
            best_key, best_entry = key, entry
    name = best_key.split("-")[0]
    method = best_key.split("-")[1]
    est = make_estimators()[name]
    p = calibrated_probs(name, est, Xtr, ytr, Xte, method)
    y_low, y_high = te_tab["y_low"].to_numpy(), te_tab["y_high"].to_numpy()
    c_low, c_high = te_tab["low_cost"].to_numpy(), te_tab["high_cost"].to_numpy()
    rescue = (~y_low.astype(bool)) & y_high.astype(bool)

    always_low = {"perf": y_low.mean(), "cost": c_low.mean(), "esc_rate": 0.0}
    always_high = {"perf": y_high.mean(), "cost": c_high.mean(), "esc_rate": 1.0}
    esc_rate_learned = 1 - ytr.mean()
    rng = np.random.default_rng(SEED)
    rand_esc = rng.random(len(yte)) < esc_rate_learned
    perf_rand = np.where(rand_esc, y_high, y_low)
    cost_rand = np.where(rand_esc, c_high, c_low)
    random_matched = {
        "perf": perf_rand.mean(),
        "cost": cost_rand.mean(),
        "esc_rate": rand_esc.mean(),
    }

    sweep = []
    for thr in np.linspace(0.05, 0.95, 19):
        esc = p < thr
        perf = np.where(esc, y_high, y_low)
        cost = np.where(esc, c_high, c_low)
        sweep.append({
            "threshold": round(float(thr), 2),
            "esc_rate": float(esc.mean()),
            "perf": float(perf.mean()),
            "cost_per_task_usd": float(cost.mean()),
            "cost_per_resolved_usd": float(cost.sum() / max(perf.sum(), 1e-9)),
            "rescues_caught": int((esc & rescue).sum()),
            "rescue_recall": float((esc & rescue).sum() / max(rescue.sum(), 1)),
            "rescue_precision": float((esc & rescue).sum() / max(esc.sum(), 1)),
        })
    results["policy"] = {
        "router": best_key,
        "always_low": always_low,
        "always_high": always_high,
        "random_matched": random_matched,
        "rescue_pool_test": int(rescue.sum()),
        "sweep": sweep,
    }
    print(f"[policy] always_low {always_low} always_high {always_high}", flush=True)
    print(f"[policy] random_matched {random_matched} rescues {int(rescue.sum())}", flush=True)

    with open(args.output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"saved {args.output} ({time.time()-t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
