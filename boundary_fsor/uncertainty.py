from __future__ import annotations

import torch


def entropy(probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = probabilities.clamp_min(1e-8)
    return -(probabilities * probabilities.log()).sum(dim=-1)


def predictive_uncertainty(view_probabilities: torch.Tensor,
                           mi_weight: float = 1.0) -> torch.Tensor:
    """Entropy plus mutual information for probabilities shaped [N, R, C]."""
    if view_probabilities.ndim != 3:
        raise ValueError("view probabilities must have shape [samples, views, classes]")
    mean_probability = view_probabilities.mean(1)
    predictive_entropy = entropy(mean_probability)
    mutual_information = (
        predictive_entropy - entropy(view_probabilities).mean(1)
    ).clamp_min(0)
    return predictive_entropy + float(mi_weight) * mutual_information


def class_uncertainty(sample_uncertainty: torch.Tensor,
                      labels: torch.Tensor) -> dict[int, float]:
    return {
        int(label): float(sample_uncertainty[labels == label].mean())
        for label in labels.unique(sorted=True)
    }
