#!/usr/bin/env python3
"""Paired episode-level AUROC comparison for a frozen FSOR ledger."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def episode_aurocs(payload: dict[str, np.ndarray], episodes: int) -> np.ndarray:
    known = payload["known_scores"]; unknown = payload["unknown_scores"]
    known_per_episode = len(known) // episodes
    unknown_per_episode = len(unknown) // episodes
    if known_per_episode * episodes != len(known):
        raise ValueError("known score count is not divisible by episode count")
    if unknown_per_episode * episodes != len(unknown):
        raise ValueError("unknown score count is not divisible by episode count")
    labels = np.r_[np.ones(known_per_episode), np.zeros(unknown_per_episode)]
    values = []
    for episode in range(episodes):
        ks = known[episode * known_per_episode:(episode + 1) * known_per_episode]
        us = unknown[episode * unknown_per_episode:(episode + 1) * unknown_per_episode]
        values.append(roc_auc_score(labels, np.r_[ks, us]))
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", default="prototype")
    parser.add_argument("--method", default="uncertainty_boundary")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3420)
    args = parser.parse_args()
    result_root = ROOT / "artifacts" / "results" / args.dataset
    baseline = dict(np.load(result_root / args.baseline / "raw_scores.npz"))
    method = dict(np.load(result_root / args.method / "raw_scores.npz"))
    baseline_auc = episode_aurocs(baseline, args.episodes)
    method_auc = episode_aurocs(method, args.episodes)
    difference = method_auc - baseline_auc
    rng = np.random.default_rng(args.seed)
    means = np.empty(args.bootstrap, dtype=np.float64)
    for draw in range(args.bootstrap):
        means[draw] = rng.choice(difference, len(difference), replace=True).mean()
    statistic = wilcoxon(difference, alternative="greater", zero_method="pratt")
    output = {
        "dataset": args.dataset,
        "episodes": args.episodes,
        "baseline": args.baseline,
        "method": args.method,
        "mean_episode_auroc_gain_pp": float(100.0 * difference.mean()),
        "bootstrap_95_ci_pp": [float(value) for value in
                                100.0 * np.quantile(means, [0.025, 0.975])],
        "episode_win_rate": float(np.mean(difference > 0.0)),
        "paired_wilcoxon_p": float(statistic.pvalue),
    }
    destination = result_root / args.method / f"paired_vs_{args.baseline}.json"
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
