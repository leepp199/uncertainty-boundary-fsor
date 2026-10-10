"""Confusability statistics and the CES sampling distribution.

The quantities in this module correspond to Sec. 2.3 of the paper.  They are
computed once from the frozen encoder and remain fixed during meta-training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def minmax_normalize(values: np.ndarray, stabilizer: float = 1e-8) -> np.ndarray:
    """Map a one-dimensional score vector to [0, 1]."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("confusability scores must be a non-empty vector")
    span = float(values.max() - values.min())
    return (values - values.min()) / (span + float(stabilizer))


def ces_sampling_probabilities(
    scores: np.ndarray,
    beta: float,
    epsilon: float,
    stabilizer: float = 1e-8,
) -> np.ndarray:
    """Evaluate Eq. (4): softmax sampling mixed with a uniform component."""
    epsilon = float(epsilon)
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0, 1]")
    normalized = minmax_normalize(scores, stabilizer)
    logits = float(beta) * normalized
    logits -= logits.max()
    softmax = np.exp(logits)
    softmax /= softmax.sum()
    return (1.0 - epsilon) * softmax + epsilon / len(softmax)


def linear_beta(
    episode: int,
    transition_episodes: int,
    beta_start: float,
    beta_end: float,
) -> float:
    """Linearly move from easy-class preference to hard-class preference."""
    transition_episodes = int(transition_episodes)
    if transition_episodes <= 0:
        return float(beta_end)
    fraction = min(max(int(episode), 0) / transition_episodes, 1.0)
    return float(beta_start) + fraction * (float(beta_end) - float(beta_start))


@dataclass(frozen=True)
class ConfusabilityBank:
    """Fixed class uncertainty and class-mean cosine similarity statistics."""

    classes: np.ndarray
    centers: np.ndarray
    uncertainty: np.ndarray
    similarity: np.ndarray

    def __post_init__(self) -> None:
        classes = np.asarray(self.classes, dtype=np.int64)
        centers = np.asarray(self.centers, dtype=np.float64)
        uncertainty = np.asarray(self.uncertainty, dtype=np.float64)
        similarity = np.asarray(self.similarity, dtype=np.float64)
        count = len(classes)
        if centers.ndim != 2 or centers.shape[0] != count:
            raise ValueError("centers must have shape [classes, feature_dim]")
        if uncertainty.shape != (count,):
            raise ValueError("uncertainty must have one value per class")
        if similarity.shape != (count, count):
            raise ValueError("similarity must have shape [classes, classes]")
        if len(np.unique(classes)) != count:
            raise ValueError("class identifiers must be unique")
        if not np.all(np.isfinite(uncertainty)) or not np.all(np.isfinite(similarity)):
            raise ValueError("confusability statistics must be finite")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "similarity", similarity)

    @classmethod
    def from_embeddings(
        cls,
        embeddings: np.ndarray,
        labels: np.ndarray,
        uncertainty_by_class: Mapping[int, float],
    ) -> "ConfusabilityBank":
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        classes = np.unique(labels)
        missing = [int(c) for c in classes if int(c) not in uncertainty_by_class]
        if missing:
            raise ValueError(f"classes without uncertainty values: {missing}")
        centers = _normalize_rows(
            np.stack([embeddings[labels == c].mean(0) for c in classes])
        )
        similarity = centers @ centers.T
        np.fill_diagonal(similarity, -2.0)
        uncertainty = np.asarray([uncertainty_by_class[int(c)] for c in classes])
        return cls(classes, centers, uncertainty, similarity)

    @property
    def index(self) -> dict[int, int]:
        return {int(label): position for position, label in enumerate(self.classes)}

    def known_distribution(
        self, beta: float, epsilon: float, stabilizer: float = 1e-8
    ) -> dict[int, float]:
        probabilities = ces_sampling_probabilities(
            self.uncertainty, beta, epsilon, stabilizer
        )
        return {
            int(label): float(probability)
            for label, probability in zip(self.classes, probabilities)
        }

    def unknown_distribution(
        self,
        known_classes: Iterable[int],
        candidates: Iterable[int],
        beta: float,
        epsilon: float,
        stabilizer: float = 1e-8,
    ) -> dict[int, float]:
        known = [int(c) for c in known_classes]
        candidates = [int(c) for c in candidates]
        lookup = self.index
        missing = [c for c in known + candidates if c not in lookup]
        if missing:
            raise ValueError(f"classes absent from the confusability bank: {missing}")
        scores = self.similarity[
            np.ix_([lookup[c] for c in known], [lookup[c] for c in candidates])
        ].max(0)
        probabilities = ces_sampling_probabilities(
            scores, beta, epsilon, stabilizer
        )
        return dict(zip(candidates, probabilities.astype(float)))

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "classes": self.classes.copy(),
            "centers": self.centers.copy(),
            "uncertainty": self.uncertainty.copy(),
            "similarity": self.similarity.copy(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> "ConfusabilityBank":
        return cls(
            state["classes"], state["centers"],
            state["uncertainty"], state["similarity"],
        )
