from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryCompanion(nn.Module):
    """Generate one task-conditioned boundary companion per positive prototype."""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 256,
                 global_only: bool = False):
        super().__init__()
        self.global_only = bool(global_only)
        self.generator = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        task_context = prototypes.mean(0, keepdim=True).expand_as(prototypes)
        if self.global_only:
            shared = task_context[:1]
            delta = self.generator(torch.cat([shared, shared], dim=-1))
            return F.normalize(shared + delta, dim=-1).expand_as(prototypes)
        delta = self.generator(torch.cat([prototypes, task_context], dim=-1))
        return F.normalize(prototypes + delta, dim=-1)


def prototypes_from_support(support: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = labels.unique(sorted=True)
    return torch.stack([support[labels == label].mean(0) for label in classes])


def boundary_margin(query: torch.Tensor, positive: torch.Tensor,
                    boundary: torch.Tensor):
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    boundary = F.normalize(boundary, dim=-1)
    positive_scores = query @ positive.t()
    winning = positive_scores.argmax(-1)
    boundary_scores = query @ boundary.t()
    paired_positive = positive_scores.gather(1, winning[:, None]).squeeze(1)
    paired_boundary = boundary_scores.gather(1, winning[:, None]).squeeze(1)
    return paired_positive - paired_boundary, positive_scores, winning


def boundary_loss(model: BoundaryCompanion, support: torch.Tensor,
                  support_labels: torch.Tensor, known_query: torch.Tensor,
                  known_labels: torch.Tensor, unknown_query: torch.Tensor,
                  margin: float):
    positive = prototypes_from_support(support, support_labels)
    boundary = model(positive)
    known_margin, known_scores, _ = boundary_margin(known_query, positive, boundary)
    unknown_margin, _, _ = boundary_margin(unknown_query, positive, boundary)
    classification = F.cross_entropy(known_scores, known_labels)
    known_boundary = F.relu(float(margin) - known_margin).mean()
    unknown_boundary = F.relu(float(margin) + unknown_margin).mean()
    total = classification + known_boundary + unknown_boundary
    return total, {
        "classification": classification.detach(),
        "known_boundary": known_boundary.detach(),
        "unknown_boundary": unknown_boundary.detach(),
    }
