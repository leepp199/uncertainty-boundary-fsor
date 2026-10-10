"""Evaluation utilities for class-disjoint FOAC episodes."""

from .evaluator import evaluate_split, score_episode
from .metrics import fpr95, oscr, summarize

__all__ = ["evaluate_split", "fpr95", "oscr", "score_episode", "summarize"]
