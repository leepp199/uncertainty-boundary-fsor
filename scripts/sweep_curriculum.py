#!/usr/bin/env python3
"""Validation-only curriculum sweep; this script never reads test features."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_capture(command: list[str]) -> dict:
    completed = subprocess.run(
        command, cwd=ROOT, check=True, text=True, capture_output=True,
        env=os.environ.copy(),
    )
    start = completed.stdout.rfind("{")
    if start < 0:
        raise RuntimeError(f"command did not emit JSON: {' '.join(command)}")
    return json.loads(completed.stdout[start:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmups", nargs="+", type=float,
                        default=[0.1, 0.25, 0.5])
    parser.add_argument("--components", nargs="+",
                        choices=["class", "pair", "joint"],
                        default=["class", "pair", "joint"])
    args = parser.parse_args()
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "WANDB_DISABLED": "true",
    })
    import yaml
    config = Path(args.config).resolve()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    output_root = ROOT / "artifacts" / "checkpoints" / cfg["name"]
    records = []
    for component in args.components:
        for warmup in args.warmups:
            tag = f"sweep_{component}_w{warmup:g}".replace(".", "p")
            checkpoint = output_root / f"boundary_{tag}.pth"
            subprocess.run([
                sys.executable, "scripts/train_boundary.py", "--config", str(config),
                "--sampling", "uncertainty", "--curriculum-components", component,
                "--warmup", str(warmup), "--tag", tag, "--device", args.device,
            ], cwd=ROOT, check=True, env=os.environ.copy())
            validation = run_capture([
                sys.executable, "scripts/evaluate_fsor.py", "--config", str(config),
                "--method", "uncertainty_boundary", "--checkpoint", str(checkpoint),
                "--result-name", tag, "--validation-only", "--device", args.device,
            ])
            records.append({
                "component": component, "warmup": warmup,
                "checkpoint": str(checkpoint), **validation,
            })
    records.sort(key=lambda row: row["validation_auroc"], reverse=True)
    destination = ROOT / "artifacts" / "tables" / f"{cfg['name']}_curriculum_sweep.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": records[0], "ledger": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
