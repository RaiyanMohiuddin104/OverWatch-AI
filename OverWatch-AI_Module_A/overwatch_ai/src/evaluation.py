"""
evaluation.py
-------------
Metrics and group-aware validation utilities for Module A.

Because false negatives (missed anomalies) are the dangerous failure
mode, every threshold-selection routine here optimizes F-beta with
beta > 1 (recall-weighted) rather than plain accuracy or F1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """
    Compute the full metric suite required by Step 7.

    y_true, y_pred: binary arrays (1 = anomaly / positive class)
    y_score: continuous anomaly score, used for ROC-AUC / PR-AUC if given
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = fbeta_score(y_true, y_pred, beta=1, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    f3 = fbeta_score(y_true, y_pred, beta=3, zero_division=0)

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def group_train_val_split(df: pd.DataFrame, group_col: str = "Batch_ID", test_size: float = 0.25, random_state: int = 42):
    """
    Split so that entire batches land in either train or validation
    (never both), preventing batch-composition leakage across the
    split (Step 6).
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def group_kfold_indices(df: pd.DataFrame, group_col: str = "Batch_ID", n_splits: int = 5):
    """Yield (train_idx, val_idx) pairs using GroupKFold on Batch_ID."""
    gkf = GroupKFold(n_splits=n_splits)
    yield from gkf.split(df, groups=df[group_col])


def select_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    candidate_thresholds: list[float],
    beta: float = 2.0,
    higher_is_more_anomalous: bool = True,
) -> tuple[float, dict]:
    """
    Try each candidate threshold, score with F-beta (recall-weighted),
    and return the threshold + its full metric dict that maximizes
    F-beta. Ties are broken by higher recall, then lower FN.
    """
    best_thr, best_metrics, best_key = None, None, (-1, -1, 1e18)
    for thr in candidate_thresholds:
        pred = (scores >= thr).astype(int) if higher_is_more_anomalous else (scores <= thr).astype(int)
        m = compute_metrics(y_true, pred, y_score=scores if higher_is_more_anomalous else -scores)
        fbeta = fbeta_score(y_true, pred, beta=beta, zero_division=0)
        key = (round(fbeta, 6), round(m["recall"], 6), -m["FN"])
        if key > best_key:
            best_key = key
            best_thr = thr
            best_metrics = m
            best_metrics["fbeta"] = fbeta
            best_metrics["beta"] = beta
    return best_thr, best_metrics


def traditional_screening_baseline(value_0h: np.ndarray, y_true: np.ndarray, limit: float = 50.0) -> dict:
    """
    Baseline rule from Step 8: FAIL if Value_0h_uA > limit else PASS.
    'FAIL' is treated as the positive (anomaly) prediction.
    """
    y_pred = (value_0h > limit).astype(int)
    return compute_metrics(y_true, y_pred)
