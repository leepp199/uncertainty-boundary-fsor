"""Episode-level evaluation for the paper's FOAC protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")

import numpy as np
import torch
import torch.nn.functional as F

from ..config import load_config
from ..data.audio import load_cache
from ..data.episodes import EpisodeSampler
from ..models.boundary import (
    TaskConditionedBoundaryGenerator,
    boundary_margin,
    prototypes_from_support,
)
from .metrics import summarize


ROOT = Path(__file__).resolve().parents[2]
TBG_METHODS = {"global_tbg", "uniform_tbg", "ces_tbg"}


def score_episode(episode, method: str, generator, device: torch.device):
    support = torch.from_numpy(episode.support).to(device)
    support_labels = torch.from_numpy(episode.support_labels).to(device)
    known = torch.from_numpy(episode.known_query).to(device)
    unknown = torch.from_numpy(episode.unknown_query).to(device)
    positive = prototypes_from_support(support, support_labels)
    known_logits = F.normalize(known, dim=-1) @ positive.t()
    unknown_logits = F.normalize(unknown, dim=-1) @ positive.t()
    correct = known_logits.argmax(-1).cpu().numpy() == episode.known_labels

    if method == "prototype":
        return (
            known_logits.max(-1).values.cpu().numpy(),
            unknown_logits.max(-1).values.cpu().numpy(),
            correct,
        )
    if method == "energy":
        temperature = 0.1
        return (
            torch.logsumexp(known_logits / temperature, -1).cpu().numpy(),
            torch.logsumexp(unknown_logits / temperature, -1).cpu().numpy(),
            correct,
        )
    if generator is None:
        raise ValueError(f"{method} requires a TBG checkpoint")
    negative = generator(positive)
    known_margin, _, _ = boundary_margin(known, positive, negative)
    unknown_margin, _, _ = boundary_margin(unknown, positive, negative)
    return known_margin.cpu().numpy(), unknown_margin.cpu().numpy(), correct


def evaluate_split(
    cfg: dict,
    split: str,
    method: str,
    checkpoint: Path | None,
    device: torch.device,
    episodes: int,
    shots: int,
    open_ways: int,
):
    cache = load_cache(ROOT / "artifacts" / "features" / cfg["name"] / f"{split}.npz")
    seed_offset = 0 if split == "val" else 10_000_019
    sampler = EpisodeSampler(
        cache["embeddings"], cache["labels"], cfg["ways"], shots,
        cfg["queries"], open_ways, cfg["seed"] + seed_offset,
    )
    generator = None
    if method in TBG_METHODS:
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if int(payload.get("shots", shots)) != shots:
            raise ValueError("checkpoint and evaluation shot counts differ")
        generator = TaskConditionedBoundaryGenerator(
            cache["embeddings"].shape[1], cfg["hidden_dim"],
            attention_heads=int(payload.get("attention_heads", cfg["attention_heads"])),
            attention_dropout=float(payload.get("attention_dropout", cfg["attention_dropout"])),
            global_only=bool(payload.get("shared_negative", False)),
        ).to(device)
        generator.load_state_dict(payload["state_dict"])
        generator.eval()

    known_scores, unknown_scores, known_correct, records = [], [], [], []
    with torch.inference_mode():
        for episode_id in range(int(episodes)):
            episode = sampler.sample(episode_id)
            known, unknown, correct = score_episode(
                episode, method, generator, device,
            )
            known_scores.append(known)
            unknown_scores.append(unknown)
            known_correct.append(correct)
            records.append({
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
        np.concatenate(known_correct),
        records,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--method", choices=["prototype", "energy", *sorted(TBG_METHODS)],
        default="ces_tbg",
    )
    parser.add_argument("--shots", type=int, choices=[1, 5], default=None)
    parser.add_argument("--open-ways", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--result-name", default=None)
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    shots = int(cfg["shots"] if args.shots is None else args.shots)
    open_ways = int(cfg["open_ways"] if args.open_ways is None else args.open_ways)
    checkpoint_tag = {
        "global_tbg": "global",
        "uniform_tbg": "uniform",
        "ces_tbg": "ces",
    }.get(args.method)
    checkpoint = None
    if checkpoint_tag is not None:
        checkpoint = Path(args.checkpoint) if args.checkpoint else (
            ROOT / "artifacts" / "checkpoints" / cfg["name"] /
            f"tbg_{checkpoint_tag}_shot{shots}.pth"
        )

    split = "val" if args.validation_only else "test"
    episodes = int(
        cfg["validation_episodes"] if args.validation_only else cfg["test_episodes"]
    )
    known, unknown, correct, records = evaluate_split(
        cfg, split, args.method, checkpoint, torch.device(args.device),
        episodes, shots, open_ways,
    )
    summary = summarize(known, unknown, correct)
    summary.update({
        "dataset": cfg["name"],
        "split": split,
        "method": args.method,
        "shots": shots,
        "ways": int(cfg["ways"]),
        "open_ways": open_ways,
        "queries_per_class": int(cfg["queries"]),
        "episodes": episodes,
        "decision_threshold": 0.0 if args.method in TBG_METHODS else None,
        "known_acceptance": (
            float(np.mean(known > 0.0)) if args.method in TBG_METHODS else None
        ),
        "unknown_rejection": (
            float(np.mean(unknown <= 0.0)) if args.method in TBG_METHODS else None
        ),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
    })
    result_name = args.result_name or args.method
    output = (
        ROOT / "artifacts" / "results" / cfg["name"] /
        f"shot_{shots}" / result_name
    )
    output.mkdir(parents=True, exist_ok=True)
    filename = "validation.json" if args.validation_only else "summary.json"
    (output / filename).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    if not args.validation_only:
        (output / "raw_test.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            output / "raw_scores.npz",
            known_scores=known,
            unknown_scores=unknown,
            known_correct=correct,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
