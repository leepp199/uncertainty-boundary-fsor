from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedBoundaryGenerator(nn.Module):
    """Generate one negative prototype for each episode-specific positive prototype."""

    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        attention_heads: int = 1,
        attention_dropout: float = 0.1,
        global_only: bool = False,
    ):
        super().__init__()
        self.global_only = bool(global_only)
        self.conditioner = nn.MultiheadAttention(
            feature_dim, int(attention_heads), dropout=float(attention_dropout),
            batch_first=True,
        )
        self.generator = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        prototypes = F.normalize(prototypes, dim=-1)
        task_context = prototypes.mean(0, keepdim=True).expand_as(prototypes)
        tokens = torch.stack([prototypes, task_context], dim=1)
        conditioned, _ = self.conditioner(tokens, tokens, tokens, need_weights=False)
        if self.global_only:
            shared = task_context[:1]
            global_tokens = conditioned.mean(0, keepdim=True)
            delta = self.generator(torch.cat([global_tokens[:, 0], global_tokens[:, 1]], dim=-1))
            return F.normalize(shared + delta, dim=-1).expand_as(prototypes)
        delta = self.generator(torch.cat([conditioned[:, 0], conditioned[:, 1]], dim=-1))
        return F.normalize(prototypes + delta, dim=-1)


def prototypes_from_support(support: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    classes = labels.unique(sorted=True)
    prototypes = torch.stack([support[labels == label].mean(0) for label in classes])
    return F.normalize(prototypes, dim=-1)


def paired_margin_matrix(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    """Return the positive-to-negative margin for every query-class pair."""
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)
    return query @ (positive - negative).t()


def boundary_margin(
    query: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
):
    """Return the margin of the nearest positive-prototype class."""
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    positive_scores = query @ positive.t()
    winning = positive_scores.argmax(-1)
    margins = paired_margin_matrix(query, positive, negative)
    candidate_margin = margins.gather(1, winning[:, None]).squeeze(1)
    return candidate_margin, positive_scores, winning


def boundary_loss(
    model: TaskConditionedBoundaryGenerator,
    support: torch.Tensor,
    support_labels: torch.Tensor,
    known_query: torch.Tensor,
    known_labels: torch.Tensor,
    unknown_query: torch.Tensor,
    margin: float,
    unknown_weight: float = 1.0,
):
    """Positive-to-negative hinge objective from Eq. (9)."""
    positive = prototypes_from_support(support, support_labels)
    negative = model(positive)
    known_margins = paired_margin_matrix(known_query, positive, negative)
    known_margin = known_margins.gather(1, known_labels[:, None]).squeeze(1)
    unknown_margin, _, _ = boundary_margin(unknown_query, positive, negative)
    known_boundary = F.relu(float(margin) - known_margin).mean()
    unknown_boundary = F.relu(float(margin) + unknown_margin).mean()
    total = known_boundary + float(unknown_weight) * unknown_boundary
    return total, {
        "known_boundary": known_boundary.detach(),
        "unknown_boundary": unknown_boundary.detach(),
    }


BoundaryCompanion = TaskConditionedBoundaryGenerator
