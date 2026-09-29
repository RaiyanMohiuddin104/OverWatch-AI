"""
explainability.py
-------------------
Deterministic, template-based explanations for QA/reliability
inspectors (Step 9). No LLM / randomness involved -- every sentence is
generated directly from the calculated numeric evidence so the same
component always produces the same explanation.
"""

from __future__ import annotations

import pandas as pd


def explain_row(row: pd.Series, absolute_limit: float = 50.0) -> str:
    """Build a human-readable explanation string for a single scored component row.

    `row` is expected to come from DynamicOutlierDetector.predict() output,
    optionally joined with Absolute_Limit_50uA_Pass if available.
    """
    value = row["Value_0h_uA"]
    abs_pass = value <= absolute_limit
    pct_above_median = (
        100 * (row["Relative_to_Batch_Median"] - 1) if pd.notna(row["Relative_to_Batch_Median"]) else 0.0
    )

    lines = []
    lines.append(f"Component ID: {row['Component_ID']}")
    lines.append(f"Batch: {row['Batch_ID']}")
    lines.append("")
    lines.append(f"Measured Value: {value:.4f} µA")
    lines.append(f"Absolute Limit: {absolute_limit:.0f} µA")
    lines.append(f"Absolute Screening: {'PASS' if abs_pass else 'FAIL'}")
    lines.append("")
    lines.append(f"Batch Median: {row['Batch_Median']:.4f} µA")
    lines.append(f"Batch MAD: {row['Batch_MAD']:.4f} µA")
    lines.append(f"Robust Z-score: {row['Robust_Z']:.2f}")
    lines.append(f"Percent Above Batch Median: {pct_above_median:.1f}%")
    lines.append(f"Isolation Forest Score: {row['Isolation_Forest_Score']:.2f}")
    lines.append(f"Dynamic Risk Score: {row['Dynamic_Anomaly_Score']:.1f}/100")
    lines.append("")
    lines.append(f"Classification: {row['Risk_Level'].upper()}")
    lines.append("")
    lines.append("Reason:")

    if row["Risk_Level"] == "Anomaly":
        if abs_pass:
            reason = (
                f"Although the component is below the absolute {absolute_limit:.0f} µA datasheet limit, "
                f"its leakage current is significantly higher than the normal population of its "
                f"manufacturing batch (robust Z-score {row['Robust_Z']:.1f}, "
                f"{pct_above_median:.0f}% above the batch median). The component therefore represents "
                f"a strong batch-relative statistical anomaly."
            )
        else:
            reason = (
                f"The component fails the absolute {absolute_limit:.0f} µA datasheet limit AND is "
                f"statistically abnormal relative to its own manufacturing batch "
                f"(robust Z-score {row['Robust_Z']:.1f}). This is a high-confidence defect."
            )
    elif row["Risk_Level"] == "Warning":
        reason = (
            f"The component shows a moderate deviation from its batch's normal population "
            f"(robust Z-score {row['Robust_Z']:.1f}, {pct_above_median:.0f}% from the batch median). "
            f"It does not yet meet the threshold for a confirmed anomaly but warrants closer inspection."
        )
    else:
        reason = (
            f"The component's leakage current is consistent with the normal population of its "
            f"manufacturing batch (robust Z-score {row['Robust_Z']:.1f})."
        )

    lines.append(reason)
    return "\n".join(lines)


def explain_component(component_id: str, predictions_df: pd.DataFrame, absolute_limit: float = 50.0) -> str:
    """Look up a Component_ID in a predictions dataframe and print its explanation."""
    matches = predictions_df[predictions_df["Component_ID"] == component_id]
    if matches.empty:
        raise ValueError(f"Component_ID '{component_id}' not found in predictions.")
    return explain_row(matches.iloc[0], absolute_limit=absolute_limit)


def add_explanations(predictions_df: pd.DataFrame, absolute_limit: float = 50.0) -> pd.DataFrame:
    """Vectorized wrapper: add an 'Explanation' column to a predictions dataframe."""
    out = predictions_df.copy()
    out["Explanation"] = out.apply(lambda r: explain_row(r, absolute_limit=absolute_limit), axis=1)
    return out
