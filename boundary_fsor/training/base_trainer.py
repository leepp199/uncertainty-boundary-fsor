"""Supervised pre-training of the audio encoder."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config
from ..data.audio import WaveDataset, load_rows, pad_collate
from ..models.encoder import AudioEncoder


ROOT = Path(__file__).resolve().parents[2]


def classification_accuracy(
    model: AudioEncoder,
    loader: DataLoader,
    device: torch.device,
    class_start: int,
    class_end: int,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for wave, labels, _ in loader:
            wave = wave.to(device)
            labels = labels.to(device)
            features = model(wave, normalize=False)
            logits = model.classifier(features)[:, class_start:class_end + 1]
            predictions = logits.argmax(1)
            correct += int((predictions == labels - class_start).sum())
            total += labels.numel()
    return correct / max(total, 1)


def checkpoint_state(model: AudioEncoder) -> dict[str, torch.Tensor]:
    state = {
        f"encoder.{key}": value.cpu()
        for key, value in model.encoder.state_dict().items()
    }
    state.update({
        f"bn0.{key}": value.cpu()
        for key, value in model.bn0.state_dict().items()
    })
    state["fc.weight"] = model.classifier.weight.cpu()
    state["fc.bias"] = model.classifier.bias.cpu()
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg["data_root"] or not Path(cfg["data_root"]).exists():
        raise FileNotFoundError(f"dataset is unavailable: {cfg['data_root']!r}")
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    class_start, class_end = (int(value) for value in cfg["meta_train_classes"])
    classes = range(class_start, class_end + 1)
    train_rows = load_rows(cfg, "train", classes)
    validation_rows = load_rows(cfg, "val", classes)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        WaveDataset(train_rows), batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=pad_collate, generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    validation_loader = DataLoader(
        WaveDataset(validation_rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = AudioEncoder(cfg).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(cfg["base_lr"]), momentum=0.9,
        nesterov=True, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(cfg["base_epochs"]),
    )
    output = ROOT / "artifacts" / "checkpoints" / cfg["name"] / "base.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_accuracy = -1.0

    for epoch in range(int(cfg["base_epochs"])):
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for wave, labels, _ in train_loader:
            wave = wave.to(device)
            labels = labels.to(device)
            features = model(wave, augment=True, normalize=False)
            logits = model.classifier(features)[:, class_start:class_end + 1]
            local_labels = labels - class_start
            loss = F.cross_entropy(logits, local_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * labels.numel()
            correct += int((logits.argmax(1) == local_labels).sum())
            count += labels.numel()

        scheduler.step()
        validation_accuracy = classification_accuracy(
            model, validation_loader, device, class_start, class_end,
        )
        record = {
            "epoch": epoch + 1,
            "loss": loss_sum / count,
            "train_acc": correct / count,
            "val_acc": validation_accuracy,
        }
        history.append(record)
        print(json.dumps(record))
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save({
                "params": checkpoint_state(model),
                "history": history,
                "meta_train_classes": [class_start, class_end],
                "config_sha256": cfg["config_sha256"],
                "best_val_acc": best_accuracy,
            }, output)
    print(output)


if __name__ == "__main__":
    main()
