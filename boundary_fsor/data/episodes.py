from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Episode:
    support: np.ndarray
    support_labels: np.ndarray
    known_query: np.ndarray
    known_labels: np.ndarray
    unknown_query: np.ndarray
    known_classes: np.ndarray
    unknown_classes: np.ndarray


class EpisodeSampler:
    def __init__(self, embeddings, labels, ways=5, shots=5, queries=15,
                 open_ways=5, seed=3420):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.ways = int(ways)
        self.shots = int(shots)
        self.queries = int(queries)
        self.open_ways = int(open_ways)
        self.seed = int(seed)
        self.by_class = {int(c): np.flatnonzero(self.labels == c) for c in np.unique(self.labels)}
        minimum = self.shots + self.queries
        bad = [c for c, idx in self.by_class.items() if len(idx) < minimum]
        if bad:
            raise ValueError(f"classes with fewer than {minimum} examples: {bad}")
        if len(self.by_class) < self.ways + self.open_ways:
            raise ValueError("known and unknown episode classes cannot be disjoint")

    def sample(self, episode_id: int, class_weights: dict[int, float] | None = None,
               pair_scores: dict[tuple[int, int], float] | None = None,
               progress: float = 0.0, warmup: float = 0.25) -> Episode:
        rng = np.random.default_rng(self.seed + 1009 * int(episode_id))
        classes = np.array(sorted(self.by_class))
        probabilities = None
        if class_weights:
            probabilities = np.array([max(float(class_weights.get(int(c), 0.0)), 1e-8) for c in classes])
            probabilities /= probabilities.sum()
        known = rng.choice(classes, self.ways, replace=False, p=probabilities)
        remaining = np.setdiff1d(classes, known)
        unknown_probabilities = None
        if pair_scores:
            difficulty = np.array([
                max(float(pair_scores[int(known_class), int(candidate)])
                    for known_class in known)
                for candidate in remaining
            ], dtype=np.float64)
            unknown_probabilities = scheduled_rank_weights(
                difficulty, progress, warmup,
            )
            unknown_probabilities /= unknown_probabilities.sum()
        unknown = rng.choice(
            remaining, self.open_ways, replace=False, p=unknown_probabilities,
        )
        support, support_y, query, query_y, open_query = [], [], [], [], []
        for local, cls in enumerate(known):
            chosen = rng.choice(self.by_class[int(cls)], self.shots + self.queries, replace=False)
            support.append(self.embeddings[chosen[:self.shots]])
            support_y.extend([local] * self.shots)
            query.append(self.embeddings[chosen[self.shots:]])
            query_y.extend([local] * self.queries)
        for cls in unknown:
            chosen = rng.choice(self.by_class[int(cls)], self.queries, replace=False)
            open_query.append(self.embeddings[chosen])
        return Episode(np.concatenate(support), np.asarray(support_y), np.concatenate(query),
                       np.asarray(query_y), np.concatenate(open_query), known, unknown)


def scheduled_rank_weights(values: np.ndarray, progress: float,
                           warmup: float = 0.25) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranks = values.argsort().argsort() / max(len(values) - 1, 1)
    progress = float(np.clip(progress, 0.0, 1.0))
    warmup = float(np.clip(warmup, 1e-6, 1.0 - 1e-6))
    if progress < warmup:
        alpha = progress / warmup
        emphasis = (1.0 - alpha) * (1.0 - ranks) + alpha * 0.5
    else:
        alpha = (progress - warmup) / (1.0 - warmup)
        emphasis = (1.0 - alpha) * 0.5 + alpha * ranks
    return 0.1 + emphasis


def curriculum_weights(class_scores: dict[int, float], progress: float,
                       warmup: float = 0.25):
    keys = np.array(sorted(class_scores))
    values = np.array([class_scores[int(k)] for k in keys], dtype=np.float64)
    emphasis = scheduled_rank_weights(values, progress, warmup)
    return {int(key): float(weight) for key, weight in zip(keys, emphasis)}


def class_pair_similarity(embeddings: np.ndarray, labels: np.ndarray) -> dict[tuple[int, int], float]:
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    classes = np.unique(labels)
    centers = []
    for label in classes:
        center = embeddings[labels == label].mean(0)
        center /= max(float(np.linalg.norm(center)), 1e-8)
        centers.append(center)
    similarity = np.asarray(centers) @ np.asarray(centers).T
    return {
        (int(first), int(second)): float(similarity[i, j])
        for i, first in enumerate(classes)
        for j, second in enumerate(classes)
        if i != j
    }
