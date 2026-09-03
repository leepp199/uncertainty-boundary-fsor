#!/usr/bin/env python3
"""Create CSV and Markdown tables directly from frozen result ledgers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("known_acc", "auroc", "aupr_known", "fpr95", "oscr")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--methods", nargs="+", default=[
        "prototype", "energy", "global_boundary", "class_boundary",
        "uncertainty_class", "uncertainty_pair", "uncertainty_boundary",
    ])
    args = parser.parse_args()
    root = ROOT / "artifacts" / "results" / args.dataset
    rows = []
    for method in args.methods:
        path = root / method / "summary.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {"method": method}
        row.update({metric: 100.0 * float(payload[metric]) for metric in METRICS})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"no summaries found below {root}")
    table_root = ROOT / "artifacts" / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    csv_path = table_root / f"{args.dataset}_fsor.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method",) + METRICS)
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "| Method | Known Acc. | AUROC | AUPR | FPR95 ↓ | OSCR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        "| {method} | {known_acc:.2f} | {auroc:.2f} | {aupr_known:.2f} | {fpr95:.2f} | {oscr:.2f} |".format(**row)
        for row in rows
    )
    md_path = table_root / f"{args.dataset}_fsor.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path); print(md_path)


if __name__ == "__main__":
    main()
