#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[2]

from ..config import load_config
from ..data.audio import load_cache
from ..data.episodes import EpisodeSampler
from ..models.boundary import TaskConditionedBoundaryGenerator, boundary_loss
from .confusability import ConfusabilityBank, linear_beta
from .uncertainty import class_uncertainty, predictive_uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sampling", choices=["uniform", "ces"], default="ces")
    parser.add_argument("--ces-components", choices=["class", "pair", "joint"],
                        default="joint")
    parser.add_argument("--shots", type=int, choices=[1, 5], default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--shared-negative", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cfg = load_config(args.config)
    shots = int(cfg["shots"] if args.shots is None else args.shots)
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
    confusability = ConfusabilityBank.from_embeddings(
        cache["embeddings"], cache["labels"], class_scores,
    )
    sampler = EpisodeSampler(
        cache["embeddings"], cache["labels"], cfg["ways"], shots,
        cfg["queries"], cfg["open_ways"], seed,
    )
    device = torch.device(args.device)
    model = TaskConditionedBoundaryGenerator(
        cache["embeddings"].shape[1], cfg["hidden_dim"],
        attention_heads=cfg["attention_heads"],
        attention_dropout=cfg["attention_dropout"],
        global_only=args.shared_negative,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["boundary_lr"]),
        weight_decay=float(cfg["boundary_weight_decay"]),
    )
    total = int(cfg["train_episodes"])
    history = []
    model.train()
    for episode_id in range(total):
        beta = linear_beta(
            episode_id,
            cfg["ces_transition_episodes"],
            cfg["ces_beta_start"],
            cfg["ces_beta_end"],
        )
        known_probabilities = None
        unknown_distribution = None
        if args.sampling == "ces" and args.ces_components in {"class", "joint"}:
            known_probabilities = confusability.known_distribution(
                beta, cfg["ces_epsilon"], cfg["normalization_stabilizer"],
            )
        if args.sampling == "ces" and args.ces_components in {"pair", "joint"}:
            unknown_distribution = lambda known, candidates: confusability.unknown_distribution(
                known, candidates, beta, cfg["ces_epsilon"],
                cfg["normalization_stabilizer"],
            )
        episode = sampler.sample(
            episode_id, known_probabilities, unknown_distribution,
        )
        tensors = [
            torch.from_numpy(value).to(device)
            for value in (
                episode.support, episode.support_labels, episode.known_query,
                episode.known_labels, episode.unknown_query,
            )
        ]
        loss, terms = boundary_loss(
            model, *tensors, margin=cfg["margin"],
            unknown_weight=cfg["unknown_loss_weight"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if episode_id % 100 == 0 or episode_id == total - 1:
            record = {
                "episode": episode_id,
                "beta": beta,
                "loss": float(loss.detach()),
                **{key: float(value) for key, value in terms.items()},
            }
            history.append(record)
            print(json.dumps(record))

    tag = args.tag or ("global" if args.shared_negative else args.sampling)
    output = (
        ROOT / "artifacts" / "checkpoints" / cfg["name"] /
        f"tbg_{tag}_shot{shots}.pth"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config_sha256": cfg["config_sha256"],
        "class_uncertainty": class_scores,
        "confusability_bank": {
            key: torch.from_numpy(value)
            for key, value in confusability.state_dict().items()
        },
        "history": history,
        "shots": shots,
        "attention_heads": int(cfg["attention_heads"]),
        "attention_dropout": float(cfg["attention_dropout"]),
        "sampling": args.sampling,
        "ces_components": args.ces_components,
        "ces_epsilon": float(cfg["ces_epsilon"]),
        "ces_beta_start": float(cfg["ces_beta_start"]),
        "ces_beta_end": float(cfg["ces_beta_end"]),
        "ces_transition_episodes": int(cfg["ces_transition_episodes"]),
        "shared_negative": args.shared_negative,
    }, output)
    print(output)


if __name__ == "__main__":
    main()
