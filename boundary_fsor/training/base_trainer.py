#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from boundary_fsor.config import load_config
from boundary_fsor.data.audio import WaveDataset, load_rows, pad_collate
from boundary_fsor.models.encoder import AudioEncoder


def accuracy(model, loader, device, class_start, class_end):
    model.eval(); correct = total = 0
    with torch.inference_mode():
        for wave, labels, _ in loader:
            wave, labels = wave.to(device), labels.to(device)
            logits = model.classifier(model(wave, normalize=False))[:, class_start:class_end + 1]
            predictions = logits.argmax(1)
            correct += int((predictions == labels - class_start).sum()); total += labels.numel()
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(); cfg = load_config(args.config)
    if not cfg["data_root"] or not Path(cfg["data_root"]).exists():
        raise FileNotFoundError(f"dataset is unavailable: {cfg['data_root']!r}")
    seed = int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    lo, hi = (int(value) for value in cfg["meta_train_classes"])
    classes = range(lo, hi + 1)
    train_rows = load_rows(cfg, "train", classes); val_rows = load_rows(cfg, "val", classes)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(WaveDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=pad_collate, generator=generator,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(WaveDataset(val_rows), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, collate_fn=pad_collate,
                            pin_memory=torch.cuda.is_available())
    device = torch.device(args.device); model = AudioEncoder(cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg["base_lr"]), momentum=0.9,
                                nesterov=True, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["base_epochs"]))
    history = []; best = -1.0; output = ROOT / "artifacts" / "checkpoints" / cfg["name"] / "base.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(int(cfg["base_epochs"])):
        model.train(); loss_sum = correct = count = 0
        for wave, labels, _ in train_loader:
            wave, labels = wave.to(device), labels.to(device)
            logits = model.classifier(model(wave, augment=True, normalize=False))[:, lo:hi + 1]
            local_labels = labels - lo
            loss = F.cross_entropy(logits, local_labels)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == local_labels).sum()); count += labels.numel()
        scheduler.step(); val_acc = accuracy(model, val_loader, device, lo, hi)
        record = {"epoch": epoch + 1, "loss": loss_sum / count, "train_acc": correct / count,
                  "val_acc": val_acc}; history.append(record); print(json.dumps(record))
        if val_acc > best:
            best = val_acc
            state = {**{f"encoder.{k}": v.cpu() for k, v in model.encoder.state_dict().items()},
                     **{f"bn0.{k}": v.cpu() for k, v in model.bn0.state_dict().items()},
                     "fc.weight": model.classifier.weight.cpu(), "fc.bias": model.classifier.bias.cpu()}
            torch.save({"params": state, "history": history, "meta_train_classes": [lo, hi],
                        "config_sha256": cfg["config_sha256"], "best_val_acc": best}, output)
    print(output)


if __name__ == "__main__": main()
