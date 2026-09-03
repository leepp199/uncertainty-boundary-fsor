#!/usr/bin/env python3
"""Create a machine-readable manifest before opening final-test features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_fsor.audit import (
    git_revision, offline_environment, rows_manifest, validate_class_disjointness,
    write_json,
)
from boundary_fsor.config import load_config
from boundary_fsor.data import load_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify-audio", action="store_true")
    parser.add_argument("--include-test", action="store_true",
                        help="Open final-test metadata only for the frozen audit.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    partitions = validate_class_disjointness(cfg)
    splits = ["train", "val"] + (["test"] if args.include_test else [])
    manifests = {
        split: rows_manifest(load_rows(cfg, split), args.verify_audio)
        for split in splits
    }
    payload = {
        "dataset": cfg["name"],
        "config_sha256": cfg["config_sha256"],
        "class_partitions": partitions,
        "data": manifests,
        "test_metadata_opened": bool(args.include_test),
        "episode_protocol": {
            "ways": int(cfg["ways"]), "shots": int(cfg["shots"]),
            "known_queries_per_class": int(cfg["queries"]),
            "unknown_ways": int(cfg["open_ways"]),
            "validation_episodes": int(cfg["validation_episodes"]),
            "test_episodes": int(cfg["test_episodes"]),
        },
        "repository": git_revision(ROOT),
        "offline_environment": offline_environment(),
    }
    suffix = "frozen" if args.include_test else "development"
    output = ROOT / "artifacts" / "manifests" / f"{cfg['name']}_{suffix}.json"
    print(write_json(output, payload))


if __name__ == "__main__":
    main()
