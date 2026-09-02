from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
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
