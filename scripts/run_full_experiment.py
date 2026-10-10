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
    parser.add_argument("--shots", type=int, nargs="+", choices=[1, 5], default=None)
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

    shots_to_run = args.shots or [int(value) for value in cfg["shot_settings"]]
    training_variants = [
        ("uniform", ["--sampling", "uniform", "--tag", "uniform"]),
        ("global", ["--sampling", "uniform", "--shared-negative", "--tag", "global"]),
        ("ces_class", [
            "--sampling", "ces", "--ces-components", "class",
            "--tag", "ces_class",
        ]),
        ("ces_pair", [
            "--sampling", "ces", "--ces-components", "pair",
            "--tag", "ces_pair",
        ]),
        ("ces", [
            "--sampling", "ces", "--ces-components", "joint",
            "--tag", "ces",
        ]),
    ]
    for shots in shots_to_run:
        for tag, extra in training_variants:
            run([
                PYTHON, "scripts/train_boundary.py", "--config", config,
                "--device", args.device, "--shots", shots, *extra,
            ], checkpoint_root / f"tbg_{tag}_shot{shots}.pth", args.force)

    if args.frozen_test:
        run([
            PYTHON, "scripts/audit_protocol.py", "--config", config, "--include-test",
        ], ROOT / "artifacts" / "manifests" / f"{name}_frozen.json", args.force)
        run([
            PYTHON, "scripts/extract_features.py", "--config", config,
            "--split", "test", "--device", args.device,
            "--workers", args.workers, "--batch-size", args.batch_size,
        ], feature_root / "test.npz", args.force)

    for shots in shots_to_run:
        evaluations = [
            ("prototype", "prototype", None),
            ("energy", "energy", None),
            ("global_tbg", "global_tbg", checkpoint_root / f"tbg_global_shot{shots}.pth"),
            ("uniform_tbg", "uniform_tbg", checkpoint_root / f"tbg_uniform_shot{shots}.pth"),
            ("ces_class", "ces_tbg", checkpoint_root / f"tbg_ces_class_shot{shots}.pth"),
            ("ces_pair", "ces_tbg", checkpoint_root / f"tbg_ces_pair_shot{shots}.pth"),
            ("ces_tbg", "ces_tbg", checkpoint_root / f"tbg_ces_shot{shots}.pth"),
        ]
        for result_name, method, checkpoint in evaluations:
            command = [
                PYTHON, "scripts/evaluate_fsor.py", "--config", config,
                "--method", method, "--result-name", result_name,
                "--device", args.device, "--shots", shots,
            ]
            if checkpoint is not None:
                command.extend(["--checkpoint", checkpoint])
            if not args.frozen_test:
                command.append("--validation-only")
            evaluation_output = result_root / f"shot_{shots}" / result_name / (
                "summary.json" if args.frozen_test else "validation.json"
            )
            run(command, evaluation_output, args.force)

    if not args.frozen_test:
        print("Development protocol complete. Test classes were not opened.")
        return

    for shots in shots_to_run:
        run([
            PYTHON, "scripts/paired_significance.py", "--dataset", name,
            "--shots", shots, "--baseline", "prototype", "--method", "ces_tbg",
            "--episodes", int(cfg["test_episodes"]),
        ], result_root / f"shot_{shots}" / "ces_tbg" / "paired_vs_prototype.json", args.force)
        run([
            PYTHON, "scripts/build_results_table.py", "--dataset", name,
            "--shots", shots,
        ], ROOT / "artifacts" / "tables" / f"{name}_shot{shots}.csv", args.force)

    if 5 in shots_to_run:
        for open_ways in (1, 3, 5, 7, 9, 12):
            result_name = f"ces_tbg_open_{open_ways}"
            run([
                PYTHON, "scripts/evaluate_fsor.py", "--config", config,
                "--method", "ces_tbg", "--result-name", result_name,
                "--checkpoint", checkpoint_root / "tbg_ces_shot5.pth",
                "--device", args.device, "--shots", 5,
                "--open-ways", open_ways,
            ], result_root / "shot_5" / result_name / "summary.json", args.force)
        run([
            PYTHON, "scripts/build_openness_table.py", "--dataset", name,
        ], ROOT / "artifacts" / "tables" / f"{name}_openness.csv", args.force)
    print("Frozen test protocol complete.")


if __name__ == "__main__":
    main()
