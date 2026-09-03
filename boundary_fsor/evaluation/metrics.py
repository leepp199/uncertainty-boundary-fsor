from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def fpr95(known_scores: np.ndarray, unknown_scores: np.ndarray) -> float:
    labels = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    scores = np.r_[known_scores, unknown_scores]
    fpr, tpr, _ = roc_curve(labels, scores)
    eligible = np.flatnonzero(tpr >= 0.95)
    return float(fpr[eligible[0]]) if len(eligible) else 1.0


def oscr(known_scores, known_correct, unknown_scores) -> float:
    thresholds = np.sort(np.unique(np.r_[known_scores, unknown_scores]))[::-1]
    ccr = np.array([(known_correct & (known_scores >= t)).mean() for t in thresholds])
    fpr = np.array([(unknown_scores >= t).mean() for t in thresholds])
    order = np.argsort(fpr)
    return float(np.trapz(ccr[order], fpr[order]))


def summarize(known_scores, unknown_scores, known_correct) -> dict[str, float]:
    y = np.r_[np.ones(len(known_scores)), np.zeros(len(unknown_scores))]
    s = np.r_[known_scores, unknown_scores]
    return {
        "known_acc": float(np.mean(known_correct)),
        "auroc": float(roc_auc_score(y, s)),
        "aupr_known": float(average_precision_score(y, s)),
        "fpr95": fpr95(known_scores, unknown_scores),
        "oscr": oscr(known_scores, known_correct, unknown_scores),
    }
