#!/usr/bin/env python3
"""Measure where a boundary method improves over prototype confidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def load_scores(root: Path, dataset: str, method: str) -> dict:
    path = root / "artifacts" / "results" / dataset / method / "raw_scores.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return {key: value for key, value in np.load(path, allow_pickle=False).items()}


def auroc(known: np.ndarray, unknown: np.ndarray) -> float:
    labels = np.r_[np.ones(len(known)), np.zeros(len(unknown))]
    return float(roc_auc_score(labels, np.r_[known, unknown]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", default="uncertainty_boundary")
    parser.add_argument("--baseline", default="prototype")
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()
    if args.bins < 2:
        raise ValueError("at least two difficulty bins are required")

    baseline = load_scores(ROOT, args.dataset, args.baseline)
    method = load_scores(ROOT, args.dataset, args.method)
    for key in ("known_scores", "unknown_scores", "positive_unknown"):
        if key not in baseline:
            raise ValueError(f"baseline ledger lacks {key}")
    if len(method["unknown_scores"]) != len(baseline["unknown_scores"]):
        raise ValueError("method and baseline ledgers use different unknown query counts")

    difficulty = baseline["positive_unknown"]
    edges = np.quantile(difficulty, np.linspace(0.0, 1.0, args.bins + 1))
    rows = []
    for index in range(args.bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        selected = (difficulty >= lower) & (
            difficulty <= upper if index + 1 == args.bins else difficulty < upper
        )
        baseline_auc = auroc(baseline["known_scores"], baseline["unknown_scores"][selected])
        method_auc = auroc(method["known_scores"], method["unknown_scores"][selected])
        rows.append({
            "difficulty_bin": index + 1,
            "prototype_similarity_min": lower,
            "prototype_similarity_max": upper,
            "unknown_queries": int(selected.sum()),
            "baseline_auroc": baseline_auc,
            "method_auroc": method_auc,
            "gain_percentage_points": 100.0 * (method_auc - baseline_auc),
        })

    result = {
        "dataset": args.dataset,
        "method": args.method,
        "baseline": args.baseline,
        "difficulty_definition": "maximum positive-prototype similarity of unknown query",
        "rows": rows,
    }
    output = ROOT / "artifacts" / "results" / args.dataset / args.method / f"difficulty_vs_{args.baseline}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
