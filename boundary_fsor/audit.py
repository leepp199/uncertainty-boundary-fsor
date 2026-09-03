"""Reproducibility manifests for class-disjoint FSOR experiments."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .data import AudioRow


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def class_range(values: Sequence[int]) -> set:
    if len(values) != 2:
        raise ValueError("class range must contain inclusive lower and upper bounds")
    low, high = (int(value) for value in values)
    if low < 0 or high < low:
        raise ValueError(f"invalid class range: {values}")
    return set(range(low, high + 1))


def validate_class_disjointness(cfg: Mapping) -> dict:
    groups = {
        "meta_train": class_range(cfg["meta_train_classes"]),
        "validation": class_range(cfg["validation_classes"]),
        "final_test": class_range(cfg["test_classes"]),
    }
    overlaps = {}
    names = list(groups)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            shared = sorted(groups[left] & groups[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared
    if overlaps:
        raise ValueError(f"FSOR class partitions overlap: {overlaps}")
    expected = set(range(int(cfg["num_classes"])))
    covered = set().union(*groups.values())
    if covered != expected:
        raise ValueError({
            "missing_classes": sorted(expected - covered),
            "unexpected_classes": sorted(covered - expected),
        })
    return {name: [min(values), max(values)] for name, values in groups.items()}


def rows_manifest(rows: Iterable[AudioRow], verify_files: bool = False) -> dict:
    rows = list(rows)
    records = []
    missing = []
    counts = Counter()
    for row in rows:
        path = Path(row.path)
        counts[int(row.label)] += 1
        record = {"path": str(path), "label": int(row.label)}
        if verify_files:
            if not path.is_file():
                missing.append(str(path))
                record["size"] = None
            else:
                record["size"] = path.stat().st_size
        records.append(record)
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"{len(missing)} audio files are missing; first entries: {preview}")
    return {
        "samples": len(records),
        "classes": len(counts),
        "per_class": {str(key): counts[key] for key in sorted(counts)},
        "ordered_rows_sha256": stable_hash(records),
        "files_verified": bool(verify_files),
    }


def cache_manifest(path: str | Path) -> dict:
    import numpy as np

    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        shapes = {key: list(payload[key].shape) for key in sorted(payload.files)}
        labels = payload["labels"] if "labels" in payload.files else None
        label_counts = None
        if labels is not None:
            counts = Counter(int(value) for value in labels)
            label_counts = {str(key): counts[key] for key in sorted(counts)}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "arrays": shapes,
        "per_class": label_counts,
    }


def git_revision(root: str | Path) -> dict:
    root = Path(root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        ).strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def offline_environment() -> dict:
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_DISABLED")
    return {key: os.environ.get(key) for key in keys}


def write_json(path: str | Path, payload: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
