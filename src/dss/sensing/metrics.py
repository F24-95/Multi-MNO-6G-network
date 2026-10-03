"""
Sensing evaluation metrics: Pd, Pf, ROC-AUC, Brier score, calibration.
Only uses p_busy (model output) vs pu_state (ground truth) — no duplicate sklearn code.
"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    roc_curve,
    confusion_matrix,
)
from typing import Tuple


def detection_metrics(
    p_busy: np.ndarray,
    pu_true: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Computes all sensing quality metrics.

    Args:
        p_busy  : (N,) array of predicted busy probability ∈ [0, 1]
        pu_true : (N,) binary ground-truth  (1=busy, 0=idle)
        threshold: decision threshold for Pd/Pf

    Returns dict with keys:
        Pd, Pf, AUC, Brier, TP, FP, TN, FN, support_busy, support_idle
    """
    p_busy  = np.asarray(p_busy).ravel()
    pu_true = np.asarray(pu_true).ravel().astype(int)

    y_pred = (p_busy >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(pu_true, y_pred, labels=[0, 1]).ravel()

    pd  = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # Detection probability
    pf  = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False alarm probability
    auc = roc_auc_score(pu_true, p_busy) if len(np.unique(pu_true)) > 1 else float("nan")
    brier = brier_score_loss(pu_true, p_busy)

    return {
        "Pd":           float(pd),
        "Pf":           float(pf),
        "AUC":          float(auc),
        "Brier":        float(brier),
        "TP":           int(tp),
        "FP":           int(fp),
        "TN":           int(tn),
        "FN":           int(fn),
        "support_busy": int(tp + fn),
        "support_idle": int(tn + fp),
        "threshold":    float(threshold),
    }


def roc_data(p_busy: np.ndarray, pu_true: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (fpr, tpr, thresholds) for ROC curve plotting."""
    return roc_curve(pu_true.ravel(), p_busy.ravel())


def calibration_table(
    p_busy: np.ndarray,
    pu_true: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """
    Returns a list of dicts per probability bin:
    [{'bin_low', 'bin_high', 'mean_pred', 'frac_busy', 'count'}]
    """
    bins   = np.linspace(0, 1, n_bins + 1)
    table  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_busy >= lo) & (p_busy < hi)
        if not mask.any():
            continue
        table.append({
            "bin_low":   round(float(lo), 3),
            "bin_high":  round(float(hi), 3),
            "mean_pred": round(float(p_busy[mask].mean()), 4),
            "frac_busy": round(float(pu_true[mask].mean()), 4),
            "count":     int(mask.sum()),
        })
    return table
