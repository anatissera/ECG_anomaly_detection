"""Decision-threshold selection.

The classifier outputs a probability of the anomalous class; a threshold turns
it into a Normal/Anomalous label. Two criteria are provided:

* Youden's J -- the point that maximizes sensitivity + specificity - 1. It
  balances both errors equally.
* recall target -- the highest threshold that still reaches a minimum recall on
  the anomalous class. In a clinical setting a missed anomaly (false negative)
  is worse than a false alarm, so we prefer to fix a floor on recall and let
  precision absorb the cost.
"""
import numpy as np
from sklearn.metrics import roc_curve


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximizing TPR - FPR."""
    fpr, tpr, th = roc_curve(y_true, y_score)
    return float(th[int(np.argmax(tpr - fpr))])


def threshold_for_recall(y_true, y_score, target_recall: float = 0.80) -> float:
    """Highest threshold whose recall on the positive class is >= target_recall.

    Positives are the anomalies (label 1). Raising the threshold lowers recall,
    so we take the largest threshold that still clears the floor -- that keeps
    precision as high as the recall constraint allows.
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))          # high score first
    ys = y_true[order]
    total_pos = ys.sum()
    if total_pos == 0:
        raise ValueError("no positive samples to compute recall")
    tp = np.cumsum(ys)
    recall = tp / total_pos
    scores_sorted = np.asarray(y_score)[order]
    ok = np.where(recall >= target_recall)[0]
    if len(ok) == 0:
        return float(scores_sorted[-1])               # cannot reach it; loosest
    # first index (highest score) where recall first reaches the target
    return float(scores_sorted[ok[0]])
