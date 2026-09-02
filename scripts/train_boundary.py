#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boundary_fsor.boundary import BoundaryCompanion, boundary_loss
from boundary_fsor.config import load_config
from boundary_fsor.data import load_cache
from boundary_fsor.episodes import EpisodeSampler, class_pair_similarity, curriculum_weights
from boundary_fsor.uncertainty import class_uncertainty, predictive_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sampling", choices=["uniform", "uncertainty"],
                        default="uncertainty")
    parser.add_argument("--global-boundary", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cache = load_cache(ROOT / "artifacts" / "features" / cfg["name"] / "train.npz")
    if "view_probabilities" not in cache:
        raise ValueError("training cache lacks stochastic-view probabilities; rerun extract_features")
    view_probabilities = torch.from_numpy(cache["view_probabilities"])
    class_start, class_end = (int(value) for value in cfg["meta_train_classes"])
    view_probabilities = view_probabilities[..., class_start:class_end + 1]
    view_probabilities = view_probabilities / view_probabilities.sum(
        dim=-1, keepdim=True,
    ).clamp_min(1e-8)
    uncertainty = predictive_uncertainty(
        view_probabilities, cfg["uncertainty_mi_weight"],
    )
    class_scores = class_uncertainty(uncertainty, torch.from_numpy(cache["labels"]))
    sampler = EpisodeSampler(
        cache["embeddings"], cache["labels"], cfg["ways"], cfg["shots"],
        cfg["queries"], cfg["open_ways"], seed,
    )
    pair_scores = class_pair_similarity(cache["embeddings"], cache["labels"])
    device = torch.device(args.device)
    model = BoundaryCompanion(
        cache["embeddings"].shape[1], cfg["hidden_dim"],
        global_only=args.global_boundary,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["boundary_lr"]), weight_decay=1e-4,
    )
    total = int(cfg["train_episodes"])
    history = []
    model.train()
    for episode_id in range(total):
        progress = episode_id / max(total - 1, 1)
        weights = None
        if args.sampling == "uncertainty":
            weights = curriculum_weights(class_scores, progress, cfg["curriculum_warmup"])
        episode = sampler.sample(
            episode_id, weights, pair_scores if args.sampling == "uncertainty" else None,
            progress=progress, warmup=cfg["curriculum_warmup"],
        )
        tensors = [
            torch.from_numpy(value).to(device)
            for value in (
                episode.support, episode.support_labels, episode.known_query,
                episode.known_labels, episode.unknown_query,
            )
        ]
        loss, terms = boundary_loss(model, *tensors, margin=cfg["margin"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if episode_id % 100 == 0 or episode_id == total - 1:
            record = {
                "episode": episode_id,
                "loss": float(loss.detach()),
                **{key: float(value) for key, value in terms.items()},
            }
            history.append(record)
            print(json.dumps(record))

    tag = "global" if args.global_boundary else args.sampling
    output = ROOT / "artifacts" / "checkpoints" / cfg["name"] / f"boundary_{tag}.pth"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config_sha256": cfg["config_sha256"],
        "class_uncertainty": class_scores,
        "history": history,
        "sampling": args.sampling,
        "global_boundary": args.global_boundary,
    }, output)
    print(output)


if __name__ == "__main__":
    main()
