"""Boundary-risk estimation and curriculum scheduling.

The paper separates two sources of difficult pseudo-open episodes:

* class instability: stochastic views disagree around a known region;
* intrusion risk: a held-out class lies close to a known class prototype.

This module keeps those quantities explicit.  It is deliberately independent
of the neural boundary generator so that every curriculum control can reuse
the same risk table and episode seeds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def _rank01(values: np.ndarray) -> np.ndarray:
    """Stable ranks in [0,1], with tied values receiving their mean rank."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be one-dimensional")
    if len(values) <= 1:
        return np.zeros_like(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result / (len(values) - 1)


def class_centers(embeddings: np.ndarray, labels: np.ndarray,
                  classes: Optional[Iterable[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must have shapes [N,D] and [N]")
    selected = np.asarray(sorted(np.unique(labels) if classes is None else classes), dtype=np.int64)
    missing = [int(label) for label in selected if not np.any(labels == label)]
    if missing:
        raise ValueError(f"classes without embeddings: {missing}")
    centers = np.stack([embeddings[labels == label].mean(0) for label in selected])
    return selected, _normalize_rows(centers)


@dataclass(frozen=True)
class BoundaryRiskTable:
    """Class-level statistics used to sample pseudo-open episodes."""

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
            raise ValueError("risk statistics must be finite")
        object.__setattr__(self, "classes", classes)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "similarity", similarity)

    @classmethod
    def from_statistics(cls, embeddings: np.ndarray, labels: np.ndarray,
                        class_uncertainty: Mapping[int, float]) -> "BoundaryRiskTable":
        classes, centers = class_centers(embeddings, labels)
        missing = [int(label) for label in classes if int(label) not in class_uncertainty]
        if missing:
            raise ValueError(f"classes without uncertainty values: {missing}")
        uncertainty = np.asarray([class_uncertainty[int(label)] for label in classes])
        similarity = centers @ centers.T
        # Keep the matrix serializable and finite.  The value is lower than
        # the cosine-similarity range and self-pairs are excluded everywhere.
        np.fill_diagonal(similarity, -2.0)
        return cls(classes, centers, uncertainty, similarity)

    @property
    def index(self) -> Dict[int, int]:
        return {int(label): position for position, label in enumerate(self.classes)}

    def class_probability(self, progress: float, warmup: float = 0.25,
                          floor: float = 0.1) -> Dict[int, float]:
        weights = curriculum_rank_weights(self.uncertainty, progress, warmup, floor)
        weights /= weights.sum()
        return {int(label): float(value) for label, value in zip(self.classes, weights)}

    def intrusion_probability(self, known_classes: Iterable[int], progress: float,
                              warmup: float = 0.25, floor: float = 0.1) -> Dict[int, float]:
        known = np.asarray(list(known_classes), dtype=np.int64)
        lookup = self.index
        missing = [int(label) for label in known if int(label) not in lookup]
        if missing:
            raise ValueError(f"known classes absent from risk table: {missing}")
        candidates = np.asarray([label for label in self.classes if label not in set(known)])
        known_index = [lookup[int(label)] for label in known]
        candidate_index = [lookup[int(label)] for label in candidates]
        intrusion = self.similarity[np.ix_(known_index, candidate_index)].max(0)
        weights = curriculum_rank_weights(intrusion, progress, warmup, floor)
        weights /= weights.sum()
        return {int(label): float(value) for label, value in zip(candidates, weights)}

    def pair_records(self) -> list:
        records = []
        for i, known in enumerate(self.classes):
            for j, pseudo_unknown in enumerate(self.classes):
                if i == j:
                    continue
                records.append({
                    "known_class": int(known),
                    "pseudo_unknown_class": int(pseudo_unknown),
                    "known_uncertainty": float(self.uncertainty[i]),
                    "prototype_similarity": float(self.similarity[i, j]),
                    "joint_risk": float(
                        0.5 * _rank01(self.uncertainty)[i]
                        + 0.5 * _rank01(self.similarity[i].copy())[j]
                    ),
                })
        return records

    def pair_score_dict(self) -> Dict[Tuple[int, int], float]:
        return {
            (int(first), int(second)): float(self.similarity[i, j])
            for i, first in enumerate(self.classes)
            for j, second in enumerate(self.classes)
            if i != j
        }

    def state_dict(self) -> dict:
        return {
            "classes": self.classes.copy(),
            "centers": self.centers.copy(),
            "uncertainty": self.uncertainty.copy(),
            "similarity": self.similarity.copy(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, np.ndarray]) -> "BoundaryRiskTable":
        return cls(state["classes"], state["centers"], state["uncertainty"], state["similarity"])


def curriculum_rank_weights(values: np.ndarray, progress: float,
                            warmup: float = 0.25, floor: float = 0.1) -> np.ndarray:
    """Easy-to-uniform-to-hard weighting used by the paper.

    ``warmup`` marks the uniform midpoint.  Before it, low-risk items receive
    more mass; afterwards, the distribution moves toward high-risk items.
    ``floor`` keeps every class or pair active throughout training.
    """
    progress = float(np.clip(progress, 0.0, 1.0))
    warmup = float(np.clip(warmup, 1e-6, 1.0 - 1e-6))
    floor = float(floor)
    if floor <= 0:
        raise ValueError("curriculum floor must be positive")
    ranks = _rank01(np.asarray(values, dtype=np.float64))
    if progress < warmup:
        blend = progress / warmup
        emphasis = (1.0 - blend) * (1.0 - ranks) + blend * 0.5
    else:
        blend = (progress - warmup) / (1.0 - warmup)
        emphasis = (1.0 - blend) * 0.5 + blend * ranks
    return floor + emphasis
