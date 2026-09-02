from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchvision.models import resnet18


class AudioEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.spectrogram = Spectrogram(
            n_fft=cfg["window_size"], hop_length=cfg["hop_size"],
            win_length=cfg["window_size"], window="hann", center=True,
            pad_mode="reflect", freeze_parameters=True)
        self.logmel = LogmelFilterBank(
            sr=cfg["sample_rate"], n_fft=cfg["window_size"],
            n_mels=cfg["mel_bins"], fmin=cfg["fmin"], fmax=cfg["fmax"],
            ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.augment = SpecAugmentation(
            time_drop_width=64, time_stripes_num=2,
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(cfg["mel_bins"])
        self.encoder = resnet18(weights=None)
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, cfg["num_classes"])
        self.feature_dropout = float(cfg.get("feature_dropout", 0.3))

    def forward_features(self, wave: torch.Tensor, augment: bool = False) -> torch.Tensor:
        x = self.logmel(self.spectrogram(wave))
        if augment:
            x = self.augment(x)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3).repeat(1, 3, 1, 1)
        return self.encoder(x)

    def forward(self, wave: torch.Tensor, augment: bool = False, normalize: bool = True):
        features = self.forward_features(wave, augment=augment)
        return F.normalize(features, dim=-1) if normalize else features

    def stochastic_predictions(self, wave: torch.Tensor, views: int):
        probabilities, embeddings = [], []
        for _ in range(int(views)):
            embedding = self.forward(wave, augment=True)
            dropped = F.dropout(embedding, p=self.feature_dropout, training=True)
            embeddings.append(embedding)
            probabilities.append(F.softmax(self.classifier(dropped), dim=-1))
        return torch.stack(embeddings, 1), torch.stack(probabilities, 1)

def load_local_checkpoint(model: AudioEncoder, path: str | Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("params", payload.get("cls_params", payload))
    encoder = {key[len("encoder."):]: value for key, value in state.items()
               if key.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(encoder, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"encoder checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    frontend = {key: value for key, value in state.items() if key.startswith("bn0.")}
    model.load_state_dict(frontend, strict=False)
    if "fc.weight" in state:
        rows = min(model.classifier.weight.size(0), state["fc.weight"].size(0))
        model.classifier.weight.data[:rows].copy_(state["fc.weight"][:rows])
        if "fc.bias" in state:
            model.classifier.bias.data[:rows].copy_(state["fc.bias"][:rows])
    return payload
