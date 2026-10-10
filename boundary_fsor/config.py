from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "name", "dataset", "data_env", "checkpoint_env", "sample_rate",
    "window_size", "hop_size", "mel_bins", "num_classes",
    "meta_train_classes", "validation_classes", "test_classes", "ways",
    "shots", "shot_settings", "queries", "open_ways", "train_episodes",
    "validation_episodes", "test_episodes", "seed", "base_epochs", "base_lr",
    "views", "feature_dropout", "uncertainty_mi_weight",
    "normalization_stabilizer", "ces_epsilon", "ces_beta_start",
    "ces_beta_end", "ces_transition_episodes", "hidden_dim", "attention_heads",
    "attention_dropout", "boundary_lr", "boundary_weight_decay", "margin",
    "unknown_loss_weight",
}


def validate_config(cfg: dict) -> None:
    missing = sorted(REQUIRED_KEYS - set(cfg))
    if missing:
        raise ValueError(f"configuration is missing: {missing}")
    if sorted(int(value) for value in cfg["shot_settings"]) != [1, 5]:
        raise ValueError("shot_settings must contain the paper's 1-shot and 5-shot protocols")
    if not 0.0 <= float(cfg["ces_epsilon"]) <= 1.0:
        raise ValueError("ces_epsilon must lie in [0, 1]")
    if float(cfg["ces_beta_start"]) >= float(cfg["ces_beta_end"]):
        raise ValueError("CES beta must increase from the easy to the hard regime")


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    cfg["config_path"] = str(path)
    cfg["project_root"] = str(path.parents[1])
    data = os.environ.get(cfg["data_env"], "")
    checkpoint = os.environ.get(cfg["checkpoint_env"], "")
    metadata = os.environ.get(cfg.get("metadata_env", ""), "") if cfg.get("metadata_env") else ""
    cfg["data_root"] = data
    cfg["checkpoint"] = checkpoint
    cfg["metadata_root"] = metadata
    cfg["config_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return cfg


def require_assets(cfg: dict) -> None:
    required = {"dataset": cfg["data_root"], "checkpoint": cfg["checkpoint"]}
    if cfg["dataset"] in {"ns100", "fsc89"}:
        required["metadata"] = cfg["metadata_root"]
    missing = [f"{name}={value!r}" for name, value in required.items() if not value or not Path(value).exists()]
    if missing:
        raise FileNotFoundError("missing local assets: " + ", ".join(missing))


def stable_json_sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()
