#!/usr/bin/env python3
"""Collect the varying-openness evaluation used by Fig. 4."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ns100")
    parser.add_argument("--known-ways", type=int, default=5)
    parser.add_argument("--open-ways", nargs="+", type=int, default=[1, 3, 5, 7, 9, 12])
    args = parser.parse_args()

    result_root = ROOT / "artifacts" / "results" / args.dataset / "shot_5"
    rows = []
    for open_ways in args.open_ways:
        path = result_root / f"ces_tbg_open_{open_ways}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        openness = (1.0 - math.sqrt(
            2.0 * args.known_ways / (2.0 * args.known_ways + open_ways)
        )) * 100.0
        rows.append({
            "known_ways": args.known_ways,
            "unknown_ways": open_ways,
            "openness": openness,
            "auroc": 100.0 * float(payload["auroc"]),
        })

    output = ROOT / "artifacts" / "tables" / f"{args.dataset}_openness.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
