from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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

    def sample(
        self,
        episode_id: int,
        known_probabilities: dict[int, float] | None = None,
        unknown_distribution: Callable[[np.ndarray, np.ndarray], dict[int, float]] | None = None,
    ) -> Episode:
        rng = np.random.default_rng(self.seed + 1009 * int(episode_id))
        classes = np.array(sorted(self.by_class))
        probabilities = None
        if known_probabilities:
            probabilities = np.array([
                max(float(known_probabilities.get(int(c), 0.0)), 1e-12)
                for c in classes
            ])
            probabilities /= probabilities.sum()
        known = rng.choice(classes, self.ways, replace=False, p=probabilities)
        remaining = np.setdiff1d(classes, known)
        unknown_probabilities = None
        if unknown_distribution is not None:
            probability_map = unknown_distribution(known, remaining)
            unknown_probabilities = np.asarray([
                max(float(probability_map[int(c)]), 1e-12) for c in remaining
            ])
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
