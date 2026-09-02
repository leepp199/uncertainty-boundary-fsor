#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_fsor.config import load_config, require_assets
from boundary_fsor.data import WaveDataset, load_rows, pad_collate
from boundary_fsor.encoder import AudioEncoder, load_local_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    local_base = ROOT / "artifacts" / "checkpoints" / cfg["name"] / "base.pth"
    if local_base.exists():
        cfg["checkpoint"] = str(local_base)
    require_assets(cfg)
    rows = load_rows(cfg, args.split)
    if not rows:
        raise RuntimeError(f"no rows found for {args.split}")
    device = torch.device(args.device)
    model = AudioEncoder(cfg).to(device)
    load_local_checkpoint(model, cfg["checkpoint"])
    model.eval()
    loader = DataLoader(WaveDataset(rows), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=pad_collate,
                        pin_memory=device.type == "cuda")
    embeddings, probabilities, labels, paths = [], [], [], []
    with torch.inference_mode():
        for wave, target, batch_paths in tqdm(loader, desc=f"extract {cfg['name']} {args.split}"):
            wave = wave.to(device, non_blocking=True)
            z, p = model.stochastic_predictions(wave, views=int(cfg["views"]))
            embeddings.append(z.mean(1).cpu().numpy())
            probabilities.append(p.cpu().numpy())
            labels.append(target.numpy())
            paths.extend(batch_paths)
    output = ROOT / "artifacts" / "features" / cfg["name"] / f"{args.split}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, embeddings=np.concatenate(embeddings),
                        view_probabilities=np.concatenate(probabilities),
                        labels=np.concatenate(labels), paths=np.asarray(paths))
    manifest = {
        "dataset": cfg["name"], "split": args.split, "rows": len(rows),
        "config_sha256": cfg["config_sha256"],
        "checkpoint_sha256": hashlib.sha256(Path(cfg["checkpoint"]).read_bytes()).hexdigest(),
        "class_range": cfg[{"train": "meta_train_classes", "val": "validation_classes",
                            "test": "test_classes"}[args.split]],
        "offline": True,
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
