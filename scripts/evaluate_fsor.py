#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_fsor.models.boundary import BoundaryCompanion, boundary_margin, prototypes_from_support
from boundary_fsor.config import load_config
from boundary_fsor.data.audio import load_cache
from boundary_fsor.data.episodes import EpisodeSampler
from boundary_fsor.evaluation.metrics import summarize


BOUNDARY_METHODS = {"global_boundary", "class_boundary", "uncertainty_boundary"}


def score_episode(episode, method: str, boundary_model, device: torch.device):
    support = torch.from_numpy(episode.support).to(device)
    support_labels = torch.from_numpy(episode.support_labels).to(device)
    known = torch.from_numpy(episode.known_query).to(device)
    unknown = torch.from_numpy(episode.unknown_query).to(device)
    positive = F.normalize(prototypes_from_support(support, support_labels), dim=-1)
    known_logits = F.normalize(known, dim=-1) @ positive.t()
    unknown_logits = F.normalize(unknown, dim=-1) @ positive.t()
    predictions = known_logits.argmax(-1).cpu().numpy()
    correct = predictions == episode.known_labels
    if method == "prototype":
        return (
            known_logits.max(-1).values.cpu().numpy(),
            unknown_logits.max(-1).values.cpu().numpy(),
            known_logits.max(-1).values.cpu().numpy(),
            unknown_logits.max(-1).values.cpu().numpy(),
            correct,
        )
    if method == "energy":
        temperature = 0.1
        return (
            torch.logsumexp(known_logits / temperature, -1).cpu().numpy(),
            torch.logsumexp(unknown_logits / temperature, -1).cpu().numpy(),
            known_logits.max(-1).values.cpu().numpy(),
            unknown_logits.max(-1).values.cpu().numpy(),
            correct,
        )
    if boundary_model is None:
        raise ValueError(f"{method} requires a boundary checkpoint")
    boundary = boundary_model(positive)
    known_margin, _, _ = boundary_margin(known, positive, boundary)
    unknown_margin, _, _ = boundary_margin(unknown, positive, boundary)
    return (
        known_margin.cpu().numpy(),
        unknown_margin.cpu().numpy(),
        known_logits.max(-1).values.cpu().numpy(),
        unknown_logits.max(-1).values.cpu().numpy(),
        correct,
    )


def evaluate_split(cfg: dict, split: str, method: str, checkpoint: Path,
                   device: torch.device, episodes: int):
    cache = load_cache(ROOT / "artifacts" / "features" / cfg["name"] / f"{split}.npz")
    offset = 0 if split == "val" else 10_000_019
    sampler = EpisodeSampler(
        cache["embeddings"], cache["labels"], cfg["ways"], cfg["shots"],
        cfg["queries"], cfg["open_ways"], cfg["seed"] + offset,
    )
    model = None
    if method in BOUNDARY_METHODS:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model = BoundaryCompanion(
            cache["embeddings"].shape[1], cfg["hidden_dim"],
            global_only=bool(payload.get("global_boundary", False)),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
    known_scores, unknown_scores = [], []
    known_positive, unknown_positive, known_correct, raw = [], [], [], []
    with torch.inference_mode():
        for episode_id in range(int(episodes)):
            episode = sampler.sample(episode_id)
            known, unknown, known_proto, unknown_proto, correct = score_episode(
                episode, method, model, device,
            )
            known_scores.append(known)
            unknown_scores.append(unknown)
            known_positive.append(known_proto)
            unknown_positive.append(unknown_proto)
            known_correct.append(correct)
            raw.append({
                "episode": episode_id,
                "known_score_mean": float(known.mean()),
                "unknown_score_mean": float(unknown.mean()),
                "known_acc": float(correct.mean()),
                "known_classes": episode.known_classes.tolist(),
                "unknown_classes": episode.unknown_classes.tolist(),
            })
    return (
        np.concatenate(known_scores),
        np.concatenate(unknown_scores),
        np.concatenate(known_positive),
        np.concatenate(unknown_positive),
        np.concatenate(known_correct),
        raw,
    )


def select_boundary_weight(validation) -> tuple[float, float]:
    labels = np.r_[np.ones(len(validation[0])), np.zeros(len(validation[1]))]
    best_auroc, best_weight = -float("inf"), 0.0
    for weight in np.linspace(0.0, 5.0, 101):
        known = validation[2] + float(weight) * validation[0]
        unknown = validation[3] + float(weight) * validation[1]
        auroc = float(roc_auc_score(labels, np.r_[known, unknown]))
        if auroc > best_auroc:
            best_auroc, best_weight = auroc, float(weight)
    return best_weight, best_auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method",
        choices=["prototype", "energy", "global_boundary", "class_boundary",
                 "uncertainty_boundary"],
        default="uncertainty_boundary",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--result-name", default=None,
                        help="Output directory name for component ablations.")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint_tag = {
        "global_boundary": "global",
        "class_boundary": "uniform",
        "uncertainty_boundary": "uncertainty",
    }.get(args.method, "")
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        ROOT / "artifacts" / "checkpoints" / cfg["name"] /
        f"boundary_{checkpoint_tag}.pth"
    )
    if args.method in BOUNDARY_METHODS and not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    validation = evaluate_split(
        cfg, "val", args.method, checkpoint, torch.device(args.device),
        int(cfg["validation_episodes"]),
    )
    boundary_weight, validation_auroc = 0.0, None
    if args.method in BOUNDARY_METHODS:
        boundary_weight, validation_auroc = select_boundary_weight(validation)
        validation_known = validation[2] + boundary_weight * validation[0]
        validation_unknown = validation[3] + boundary_weight * validation[1]
    else:
        validation_known, validation_unknown = validation[0], validation[1]
        labels = np.r_[np.ones(len(validation_known)), np.zeros(len(validation_unknown))]
        validation_auroc = float(
            roc_auc_score(labels, np.r_[validation_known, validation_unknown])
        )
    result_name = args.result_name or args.method
    validation_threshold = float(np.quantile(validation_known, 0.05))
    if args.validation_only:
        validation_summary = summarize(
            validation_known, validation_unknown, validation[4],
        )
        validation_summary.update({
            "method": args.method,
            "result_name": result_name,
            "checkpoint": str(checkpoint),
            "boundary_weight": boundary_weight,
            "validation_auroc": validation_auroc,
            "validation_threshold": validation_threshold,
            "split": "validation",
        })
        output = ROOT / "artifacts" / "results" / cfg["name"] / result_name
        output.mkdir(parents=True, exist_ok=True)
        (output / "validation.json").write_text(
            json.dumps(validation_summary, indent=2) + "\n", encoding="utf-8",
        )
        print(json.dumps(validation_summary, indent=2))
        return
    test = evaluate_split(
        cfg, "test", args.method, checkpoint, torch.device(args.device),
        int(cfg["test_episodes"]),
    )
    if args.method in BOUNDARY_METHODS:
        test_known = test[2] + boundary_weight * test[0]
        test_unknown = test[3] + boundary_weight * test[1]
    else:
        test_known, test_unknown = test[0], test[1]
    summary = summarize(test_known, test_unknown, test[4])
    summary.update({
        "method": args.method,
        "result_name": result_name,
        "checkpoint": str(checkpoint) if args.method in BOUNDARY_METHODS else None,
        "boundary_weight": boundary_weight,
        "validation_auroc_at_selected_weight": validation_auroc,
        "validation_threshold": validation_threshold,
        "known_acceptance_at_threshold": float(np.mean(test_known >= validation_threshold)),
        "unknown_rejection_at_threshold": float(np.mean(test_unknown < validation_threshold)),
    })
    output = ROOT / "artifacts" / "results" / cfg["name"] / result_name
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_test.jsonl").write_text(
        "\n".join(json.dumps(row) for row in test[5]) + "\n", encoding="utf-8",
    )
    np.savez_compressed(
        output / "raw_scores.npz", known_scores=test_known, unknown_scores=test_unknown,
        known_correct=test[4], positive_known=test[2], positive_unknown=test[3],
        boundary_known=test[0], boundary_unknown=test[1],
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
