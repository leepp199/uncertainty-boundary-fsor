from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioRow:
    path: str
    label: int


def _class_range(cfg: dict, split: str) -> range:
    key = {"train": "meta_train_classes", "val": "validation_classes", "test": "test_classes"}[split]
    lo, hi = cfg[key]
    return range(int(lo), int(hi) + 1)


def load_rows(cfg: dict, split: str, classes=None) -> list[AudioRow]:
    classes = set(_class_range(cfg, split) if classes is None else classes)
    root = Path(cfg["data_root"])
    if cfg["dataset"] == "ls100":
        table = pd.read_csv(root / f"librispeech_fscil_{split}.csv")
        return [AudioRow(str(root / "100spks_segments" / str(r.filename)), int(r.label))
                for r in table.itertuples() if int(r.label) in classes]
    if cfg["dataset"] == "ns100":
        meta = Path(cfg["metadata_root"])
        # The published NS-100 validation CSV contains only base labels 0--54.
        # For class-disjoint FSOR model selection, labels 60--79 are therefore
        # taken from the training CSV; the final labels 80--99 remain confined
        # to the test CSV.  Class filters below keep both pools disjoint.
        source_split = "train" if split == "val" else split
        table = pd.read_csv(meta / f"nsynth-100-fs_{source_split}.csv")
        vocab = json.loads((meta / "nsynth-100-fs_vocab.json").read_text(encoding="utf-8"))
        rows = []
        for r in table.itertuples():
            label = int(vocab[str(r.instrument)])
            if label in classes:
                rows.append(AudioRow(str(root / str(r.audio_source) / "audio" / f"{r.filename}.wav"), label))
        return rows
    if cfg["dataset"] == "fsc89":
        name = {"train": "Fsc89-mini-fsci_train.csv", "val": "Fsc89-mini-fsci_val.csv",
                "test": "Fsc89-mini-fsci_test.csv"}[split]
        table = pd.read_csv(Path(cfg["metadata_root"]) / name)
        rows = []
        for r in table.itertuples():
            label = int(r.label)
            if label in classes:
                filename = str(r.FSD_MIX_SED_filename).replace(
                    ".wav", f"_{int(float(r.start_time) * 44100)}.wav")
                rows.append(AudioRow(str(root / "audio" / str(r.data_folder) / filename), label))
        return rows
    raise ValueError(f"unknown dataset {cfg['dataset']}")


class WaveDataset(Dataset):
    def __init__(self, rows: list[AudioRow]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        wave, _ = torchaudio.load(row.path)
        return wave.mean(0), row.label, row.path


def pad_collate(batch):
    waves, labels, paths = zip(*batch)
    return pad_sequence(waves, batch_first=True), torch.tensor(labels), list(paths)


def load_cache(path: str | Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    required = {"embeddings", "labels", "paths"}
    if not required.issubset(payload.files):
        raise ValueError(f"feature cache is missing {sorted(required - set(payload.files))}")
    return {key: payload[key] for key in payload.files}
