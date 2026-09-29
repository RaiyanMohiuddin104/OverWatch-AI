"""
preprocessing.py
-----------------
Loading, validation and basic cleaning for OverWatch-AI Module A.

Design notes
============
- Module A only uses `Value_0h_uA` (incoming inspection reading) plus
  `Component_ID` / `Batch_ID`. All other time-series columns
  (`Value_24h_uA`, `Value_96h_uA`, `Value_168h_uA`, `Drift_*_pct`,
  `Absolute_Limit_50uA_Pass`) belong conceptually to Module B (the
  168h degradation predictor) and are NOT used as model inputs here.
  They are only used, outside of the model, to build the "traditional
  screening" baseline in evaluation.py for comparison purposes.
- `Ground_Truth` is NEVER used as a model input. It is only used to
  compute evaluation metrics.
- The CSV ships with precomputed `Batch_Mean_0h_uA` / `Batch_Std_0h_uA`
  / `Relative_to_Batch_Mean` columns. EDA showed these are computed
  over ALL components in a batch (anomalies included), so they are
  contaminated by the very anomalies we are trying to detect. We
  therefore RECOMPUTE all batch statistics from scratch in
  feature_engineering.py and never feed the precomputed columns into
  the model.
"""

from __future__ import annotations

import pandas as pd

# Columns Module A is allowed to use as raw inputs.
MODULE_A_INPUT_COLUMNS = ["Component_ID", "Batch_ID", "Value_0h_uA"]

# Columns that exist in the training CSV but must never be used as
# model *inputs* for Module A (label, future/Module-B information, or
# statistics precomputed over a contaminated population).
LEAKAGE_COLUMNS = [
    "Ground_Truth",
    "Value_24h_uA",
    "Value_96h_uA",
    "Value_168h_uA",
    "Drift_0h_to_24h_pct",
    "Drift_24h_to_168h_pct",
    "Absolute_Limit_50uA_Pass",
    "Batch_Mean_0h_uA",
    "Batch_Std_0h_uA",
    "Relative_to_Batch_Mean",
]

REQUIRED_TRAINING_COLUMNS = MODULE_A_INPUT_COLUMNS + ["Ground_Truth"]
REQUIRED_INFERENCE_COLUMNS = MODULE_A_INPUT_COLUMNS


def load_training_data(path: str) -> pd.DataFrame:
    """Load the OverWatch-AI training CSV and run basic validation."""
    df = pd.read_csv(path)
    validate_columns(df, REQUIRED_TRAINING_COLUMNS)
    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input dataframe is missing required column(s): {missing}. "
            f"Module A requires at minimum: {REQUIRED_INFERENCE_COLUMNS}"
        )


def clean_inference_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and lightly clean a new-batch dataframe submitted for
    inference (see anomaly_detector.DynamicOutlierDetector.predict).

    - Validates required columns are present.
    - Coerces Value_0h_uA to numeric, dropping rows that cannot be
      parsed (with a warning count returned to the caller via
      attribute on the dataframe is avoided -- callers should check
      row counts before/after if they need auditing).
    - Drops rows with missing Component_ID / Batch_ID / Value_0h_uA.
    """
    validate_columns(df, REQUIRED_INFERENCE_COLUMNS)
    out = df.copy()
    out["Value_0h_uA"] = pd.to_numeric(out["Value_0h_uA"], errors="coerce")
    out = out.dropna(subset=["Component_ID", "Batch_ID", "Value_0h_uA"]).reset_index(
        drop=True
    )
    return out


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return a dict of EDA summary statistics (used by notebook/report)."""
    summary = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "n_batches": df["Batch_ID"].nunique(),
        "components_per_batch": df["Batch_ID"].value_counts().describe().to_dict(),
    }
    if "Ground_Truth" in df.columns:
        summary["ground_truth_counts"] = df["Ground_Truth"].value_counts().to_dict()
        summary["ground_truth_pct"] = (
            df["Ground_Truth"].value_counts(normalize=True) * 100
        ).round(2).to_dict()
    return summary
