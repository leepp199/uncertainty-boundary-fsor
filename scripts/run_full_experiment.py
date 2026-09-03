#!/usr/bin/env python3
"""Run the complete class-disjoint FSOR protocol without network access.

Development runs stop at validation by default.  Final-test metadata and audio
are opened only when ``--frozen-test`` is supplied after the configuration and
method choices have been frozen.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(arguments: Iterable[str], skip_output: Path | None = None,
        force: bool = False) -> None:
    command = [str(value) for value in arguments]
    if skip_output is not None and skip_output.exists() and not force:
        print(json.dumps({"status": "reuse", "artifact": str(skip_output)}))
        return
    print(json.dumps({"status": "run", "command": command}))
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild artifacts that already exist.")
    parser.add_argument("--skip-base-training", action="store_true",
                        help="Use the local checkpoint named by the config environment variable.")
    parser.add_argument("--frozen-test", action="store_true",
                        help="Open the sealed test classes and produce final tables.")
    args = parser.parse_args()

    os.environ.update({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
    })
    config = Path(args.config).resolve()
    import yaml
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    name = str(cfg["name"])
    checkpoint_root = ROOT / "artifacts" / "checkpoints" / name
    feature_root = ROOT / "artifacts" / "features" / name
    result_root = ROOT / "artifacts" / "results" / name

    run([PYTHON, "scripts/audit_protocol.py", "--config", config],
        ROOT / "artifacts" / "manifests" / f"{name}_development.json", args.force)
    if not args.skip_base_training:
        run([
            PYTHON, "scripts/train_base.py", "--config", config,
            "--device", args.device, "--workers", args.workers,
            "--batch-size", args.batch_size,
        ], checkpoint_root / "base.pth", args.force)

    for split in ("train", "val"):
        run([
            PYTHON, "scripts/extract_features.py", "--config", config,
            "--split", split, "--device", args.device,
            "--workers", args.workers, "--batch-size", args.batch_size,
        ], feature_root / f"{split}.npz", args.force)

    training_variants = [
        ("uniform", ["--sampling", "uniform", "--tag", "uniform"]),
        ("global", ["--sampling", "uniform", "--global-boundary", "--tag", "global"]),
        ("uncertainty_class", [
            "--sampling", "uncertainty", "--curriculum-components", "class",
            "--tag", "uncertainty_class",
        ]),
        ("uncertainty_pair", [
            "--sampling", "uncertainty", "--curriculum-components", "pair",
            "--tag", "uncertainty_pair",
        ]),
        ("uncertainty", [
            "--sampling", "uncertainty", "--curriculum-components", "joint",
            "--tag", "uncertainty",
        ]),
    ]
    for tag, extra in training_variants:
        run([
            PYTHON, "scripts/train_boundary.py", "--config", config,
            "--device", args.device, *extra,
        ], checkpoint_root / f"boundary_{tag}.pth", args.force)

    if args.frozen_test:
        run([
            PYTHON, "scripts/audit_protocol.py", "--config", config, "--include-test",
        ], ROOT / "artifacts" / "manifests" / f"{name}_frozen.json", args.force)
        run([
            PYTHON, "scripts/extract_features.py", "--config", config,
            "--split", "test", "--device", args.device,
            "--workers", args.workers, "--batch-size", args.batch_size,
        ], feature_root / "test.npz", args.force)

    evaluations = [
        ("prototype", "prototype", None),
        ("energy", "energy", None),
        ("global_boundary", "global_boundary", checkpoint_root / "boundary_global.pth"),
        ("class_boundary", "class_boundary", checkpoint_root / "boundary_uniform.pth"),
        ("uncertainty_class", "uncertainty_boundary",
         checkpoint_root / "boundary_uncertainty_class.pth"),
        ("uncertainty_pair", "uncertainty_boundary",
         checkpoint_root / "boundary_uncertainty_pair.pth"),
        ("uncertainty_boundary", "uncertainty_boundary",
         checkpoint_root / "boundary_uncertainty.pth"),
    ]
    for result_name, method, checkpoint in evaluations:
        command = [
            PYTHON, "scripts/evaluate_fsor.py", "--config", config,
            "--method", method, "--result-name", result_name,
            "--device", args.device,
        ]
        if checkpoint is not None:
            command.extend(["--checkpoint", checkpoint])
        if not args.frozen_test:
            command.append("--validation-only")
        evaluation_output = result_root / result_name / (
            "summary.json" if args.frozen_test else "validation.json"
        )
        run(command, evaluation_output, args.force)

    if not args.frozen_test:
        print("Development protocol complete. Test classes were not opened.")
        return

    # The first evaluation loop may have reused existing results.  These
    # analyses consume only complete frozen ledgers.
    run([
        PYTHON, "scripts/paired_significance.py", "--dataset", name,
        "--baseline", "prototype", "--method", "uncertainty_boundary",
        "--episodes", int(cfg["test_episodes"]),
    ], result_root / "uncertainty_boundary" / "paired_vs_prototype.json", args.force)
    run([
        PYTHON, "scripts/analyze_boundary_difficulty.py", "--dataset", name,
        "--baseline", "prototype", "--method", "uncertainty_boundary",
    ], result_root / "uncertainty_boundary" / "difficulty_vs_prototype.json", args.force)
    run([PYTHON, "scripts/build_results_table.py", "--dataset", name],
        ROOT / "artifacts" / "tables" / f"{name}_fsor.csv", args.force)
    print("Frozen test protocol complete.")


if __name__ == "__main__":
    main()
