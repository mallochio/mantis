#!/usr/bin/env python3
"""Deduplicate router observations and label them offline with Supra."""
import argparse
import json
from pathlib import Path


def label_rows(rows, scorer, threshold=3):
    latest = {}
    counts = {}
    for row in rows:
        prompt = row.get("prompt", "").strip()
        if not prompt:
            continue
        latest[prompt] = row
        counts[prompt] = counts.get(prompt, 0) + 1
    for prompt, row in latest.items():
        complexity, elapsed_ms = scorer(prompt)
        label = "expensive" if complexity >= threshold else "cheap"
        yield {
            "prompt": prompt,
            "pseudo_label": label,
            "supra_complexity": complexity,
            "supra_ms": elapsed_ms,
            "mf_score": row.get("score"),
            "routed_decision": row.get("decision"),
            "disagreement": label != row.get("decision"),
            "observations": counts[prompt],
            "last_seen": row.get("ts"),
        }


def demo():
    rows = [
        {"prompt": "Reply OK", "score": 0.17, "decision": "expensive", "ts": 1},
        {"prompt": "Reply OK", "score": 0.17, "decision": "expensive", "ts": 2},
        {"prompt": "Prove this", "score": 0.1, "decision": "cheap", "ts": 3},
    ]
    labels = list(label_rows(rows, lambda p: (1 if p == "Reply OK" else 4, 5)))
    assert labels[0]["pseudo_label"] == "cheap" and labels[0]["disagreement"]
    assert labels[0]["observations"] == 2 and labels[0]["last_seen"] == 2
    assert labels[1]["pseudo_label"] == "expensive" and labels[1]["disagreement"]


def main():
    parser = argparse.ArgumentParser()
    data_dir = Path.home() / ".local/share/mantis/router"
    parser.add_argument("input", nargs="?", type=Path, default=data_dir / "training.jsonl")
    parser.add_argument("-o", "--output", type=Path, default=data_dir / "pseudo-labels.jsonl")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        demo()
        return

    from server import SUPRA_THRESHOLD, _supra_complexity
    with args.input.open() as source:
        rows = (json.loads(line) for line in source if line.strip())
        with args.output.open("w") as output:
            for row in label_rows(rows, _supra_complexity, SUPRA_THRESHOLD):
                output.write(json.dumps(row) + "\n")
        args.output.chmod(0o600)


if __name__ == "__main__":
    main()
