"""
feature_engineering.py
-----------------------
Batch-aware statistical features for Module A.

All statistics are recomputed from `Value_0h_uA` grouped by `Batch_ID`
in the dataframe passed in -- the caller controls whether that
dataframe is "fit" data (e.g. a training fold, ideally dominated by
Normal components) or "new" data at inference time. Ground_Truth is
never read by this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9  # generic safe-division fallback


def compute_batch_statistics(df: pd.DataFrame, value_col: str = "Value_0h_uA") -> pd.DataFrame:
    """
    Compute per-batch summary statistics.

    Returns a dataframe indexed by Batch_ID with columns:
    batch_mean, batch_median, batch_std, q1, q3, iqr, mad,
    batch_min, batch_max, batch_n
    """
    grouped = df.groupby("Batch_ID")[value_col]

    stats = grouped.agg(
        batch_mean="mean",
        batch_median="median",
        batch_std="std",
        batch_min="min",
        batch_max="max",
        batch_n="count",
    )
    stats["q1"] = grouped.quantile(0.25)
    stats["q3"] = grouped.quantile(0.75)
    stats["iqr"] = stats["q3"] - stats["q1"]

    # Median Absolute Deviation: median(|x - median(x)|)
    def _mad(s: pd.Series) -> float:
        med = s.median()
        return float((s - med).abs().median())

    stats["mad"] = grouped.apply(_mad)

    # std is NaN for batches of size 1 -- fall back to 0
    stats["batch_std"] = stats["batch_std"].fillna(0.0)

    return stats


def add_component_features(
    df: pd.DataFrame,
    batch_stats: pd.DataFrame,
    value_col: str = "Value_0h_uA",
) -> pd.DataFrame:
    """
    Join batch statistics onto each component row and compute the
    component-level batch-relative features described in Step 3.

    `batch_stats` should come from `compute_batch_statistics`, typically
    computed on the TRAINING fold only, then applied (joined) onto both
    train and validation/inference rows to avoid leaking validation-fold
    batch composition into the statistics -- for genuinely UNSEEN
    batches at inference time, batch_stats must instead be computed on
    the new batch itself (handled explicitly in anomaly_detector.py).
    """
    out = df.merge(batch_stats, left_on="Batch_ID", right_index=True, how="left")
    v = out[value_col]

    # Standard z-score, safe against std == 0
    safe_std = out["batch_std"].where(out["batch_std"] > EPS, other=np.nan)
    out["z_score"] = (v - out["batch_mean"]) / safe_std
    out["z_score"] = out["z_score"].fillna(0.0)

    # Robust z-score using MAD, safe against MAD == 0
    safe_mad = out["mad"].where(out["mad"] > EPS, other=np.nan)
    out["robust_z"] = 0.6745 * (v - out["batch_median"]) / safe_mad
    out["robust_z"] = out["robust_z"].fillna(0.0)

    # Relative-to-median ratio and differences
    safe_median = out["batch_median"].where(out["batch_median"].abs() > EPS, other=np.nan)
    out["relative_to_median"] = v / safe_median
    out["relative_to_median"] = out["relative_to_median"].fillna(1.0)

    out["diff_from_median"] = v - out["batch_median"]
    out["pct_diff_from_median"] = ((v - out["batch_median"]) / safe_median) * 100
    out["pct_diff_from_median"] = out["pct_diff_from_median"].fillna(0.0)

    # IQR outlier flag + continuous distance outside IQR bounds
    lower_bound = out["q1"] - 1.5 * out["iqr"]
    upper_bound = out["q3"] + 1.5 * out["iqr"]
    out["iqr_lower_bound"] = lower_bound
    out["iqr_upper_bound"] = upper_bound
    out["is_iqr_outlier"] = (v < lower_bound) | (v > upper_bound)

    dist_below = (lower_bound - v).clip(lower=0)
    dist_above = (v - upper_bound).clip(lower=0)
    out["iqr_distance"] = dist_below + dist_above  # 0 if inside bounds

    # Percentile rank of the component within its batch (0-100)
    out["percentile_rank"] = out.groupby("Batch_ID")[value_col].rank(pct=True) * 100

    return out


FEATURE_COLUMNS = [
    "robust_z",
    "z_score",
    "relative_to_median",
    "pct_diff_from_median",
    "percentile_rank",
    "iqr_distance",
]
