#!/usr/bin/env python3
"""Friedman and Nemenyi analysis for paired method scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Wide CSV: one paired task per row and one method per column.")
    parser.add_argument("--metric", choices=["acc", "auroc"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    table = pd.read_csv(args.input)
    table = table.loc[:, ~table.columns.str.startswith("Unnamed:")]
    nonnumeric = [column for column in table if not pd.api.types.is_numeric_dtype(table[column])]
    if nonnumeric:
        raise ValueError(f"non-numeric method columns: {nonnumeric}")
    if table.isna().any().any():
        raise ValueError("paired score table contains missing values")
    scores = table.to_numpy(dtype=np.float64)
    blocks, methods = scores.shape
    if blocks < 2 or methods < 3:
        raise ValueError("Friedman analysis requires at least two tasks and three methods")

    ranks = np.stack([rankdata(-row, method="average") for row in scores])
    if not np.allclose(ranks, np.rint(ranks)):
        raise ValueError("rank-frequency tables require scores without within-task ties")
    average_ranks = ranks.mean(0)
    frequency = np.stack([
        np.bincount(ranks[:, index].astype(int), minlength=methods + 1)[1:]
        for index in range(methods)
    ])
    friedman = friedmanchisquare(*[scores[:, index] for index in range(methods)])
    q_alpha = float(studentized_range.ppf(1.0 - args.alpha, methods, np.inf) / np.sqrt(2.0))
    critical_difference = q_alpha * np.sqrt(methods * (methods + 1) / (6.0 * blocks))
    differences = np.abs(average_ranks[:, None] - average_ranks[None, :])
    significant = differences > critical_difference

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frequency_table = pd.DataFrame(
        frequency,
        index=[f"{name} [{rank:.2f}]" for name, rank in zip(table.columns, average_ranks)],
        columns=[str(rank) for rank in range(1, methods + 1)],
    ).sort_index(key=lambda index: np.asarray([
        average_ranks[list(table.columns).index(label.rsplit(" [", 1)[0])]
        for label in index
    ]), ascending=False)
    frequency_table.to_csv(output / f"{args.metric}_rank_frequency.csv")
    pd.DataFrame(
        significant, index=table.columns, columns=table.columns,
    ).to_csv(output / f"{args.metric}_nemenyi_significant.csv")
    summary = {
        "metric": args.metric,
        "blocks": blocks,
        "methods": methods,
        "alpha": args.alpha,
        "friedman_statistic": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "nemenyi_critical_difference": float(critical_difference),
        "average_ranks": {
            method: float(rank) for method, rank in zip(table.columns, average_ranks)
        },
    }
    (output / f"{args.metric}_significance.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
