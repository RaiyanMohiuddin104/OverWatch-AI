"""
anomaly_detector.py
--------------------
Module A -- Dynamic, batch-aware outlier detection.

Contains:
- Model 1: Robust Z-score detector
- Model 2: Batch-specific IQR detector
- Model 3: Isolation Forest (trained on batch-relative features)
- HybridScorer: combines the three into an explainable 0-100 score
- DynamicOutlierDetector: end-to-end fit/predict pipeline suitable for
  deployment behind a Flask/FastAPI/Streamlit frontend.

Ground_Truth is never used as a *feature*. It is only read (when
present) during `fit()` to select thresholds/weights on a held-out,
batch-disjoint validation split.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import feature_engineering as fe
from . import evaluation as ev

RANDOM_STATE = 42

ROBUST_Z_CANDIDATE_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
IQR_MULTIPLIER_CANDIDATES = [1.5, 1.2, 1.0]
SCORE_BAND_NORMAL_MAX = 30
SCORE_BAND_WARNING_MAX = 60


# ---------------------------------------------------------------------------
# Model 1 -- Robust Z-score detector
# ---------------------------------------------------------------------------

def robust_z_predict(robust_z: pd.Series, threshold: float) -> np.ndarray:
    return (robust_z.abs() >= threshold).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Model 2 -- IQR detector
# ---------------------------------------------------------------------------

def iqr_predict(value: pd.Series, q1: pd.Series, q3: pd.Series, iqr: pd.Series, multiplier: float) -> np.ndarray:
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return ((value < lower) | (value > upper)).astype(int).to_numpy()


# ---------------------------------------------------------------------------
# Model 3 -- Isolation Forest
# ---------------------------------------------------------------------------

def fit_isolation_forest(features: pd.DataFrame, contamination: float = "auto", random_state: int = RANDOM_STATE) -> IsolationForest:
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features.values)
    return model


def isolation_forest_anomaly_score(model: IsolationForest, features: pd.DataFrame) -> np.ndarray:
    """
    Map IsolationForest's raw decision_function (higher = more normal)
    to a 0-1 anomaly score where higher = more anomalous, via a min-max
    normalization of -decision_function.
    """
    raw = -model.decision_function(features.values)  # higher = more anomalous
    lo, hi = raw.min(), raw.max()
    if hi - lo < 1e-9:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Hybrid score
# ---------------------------------------------------------------------------

@dataclass
class HybridWeights:
    robust_z: float = 0.5
    isolation_forest: float = 0.3
    iqr: float = 0.2

    def as_dict(self) -> dict:
        return {"robust_z": self.robust_z, "isolation_forest": self.isolation_forest, "iqr": self.iqr}


def normalize_robust_z_evidence(robust_z: pd.Series, cap: float = 10.0) -> np.ndarray:
    """Map |robust_z| to [0, 1] evidence, saturating at `cap`."""
    return np.clip(robust_z.abs() / cap, 0, 1).to_numpy()


def normalize_iqr_evidence(iqr_distance: pd.Series, iqr: pd.Series, cap_multiples_of_iqr: float = 3.0) -> np.ndarray:
    """Map continuous IQR-overshoot distance to [0, 1], scaled by the batch's own IQR."""
    safe_iqr = iqr.replace(0, np.nan)
    ratio = (iqr_distance / safe_iqr).fillna(0.0)
    return np.clip(ratio / cap_multiples_of_iqr, 0, 1).to_numpy()


def compute_hybrid_score(
    robust_z: pd.Series,
    iqr_distance: pd.Series,
    iqr: pd.Series,
    if_score_01: np.ndarray,
    weights: HybridWeights,
) -> np.ndarray:
    rz_evidence = normalize_robust_z_evidence(robust_z)
    iqr_evidence = normalize_iqr_evidence(iqr_distance, iqr)
    combined = (
        weights.robust_z * rz_evidence
        + weights.isolation_forest * if_score_01
        + weights.iqr * iqr_evidence
    )
    return np.clip(combined * 100, 0, 100)


def score_to_risk_level(score: np.ndarray) -> np.ndarray:
    levels = np.where(
        score >= SCORE_BAND_WARNING_MAX,
        "Anomaly",
        np.where(score >= SCORE_BAND_NORMAL_MAX, "Warning", "Normal"),
    )
    return levels


# ---------------------------------------------------------------------------
# End-to-end detector
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    robust_z_threshold: float = 3.5
    iqr_multiplier: float = 1.5
    hybrid_weights: dict = field(default_factory=HybridWeights().as_dict)
    score_band_normal_max: float = SCORE_BAND_NORMAL_MAX
    score_band_warning_max: float = SCORE_BAND_WARNING_MAX
    feature_columns: list = field(default_factory=lambda: list(fe.FEATURE_COLUMNS))
    if_feature_columns: list = field(default_factory=lambda: list(fe.FEATURE_COLUMNS))
    random_state: int = RANDOM_STATE

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DetectorConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class DynamicOutlierDetector:
    """
    End-to-end batch-aware anomaly detector.

    Usage
    -----
        detector = DynamicOutlierDetector()
        detector.fit(training_dataframe)          # training_dataframe has Ground_Truth
        results = detector.predict(new_batch_df)  # new_batch_df: Component_ID, Batch_ID, Value_0h_uA

    `predict` recomputes batch statistics from whatever batch(es) are
    present in the input (works for previously-unseen batches, since
    all comparisons are relative to the batch the component arrives
    with -- there is nothing "global" to have not-seen-before).
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.isolation_forest: IsolationForest | None = None
        self._is_fitted = False

    # -- fitting -----------------------------------------------------------

    def fit(self, training_df: pd.DataFrame, value_col: str = "Value_0h_uA", verbose: bool = True) -> "DynamicOutlierDetector":
        df = training_df.copy()
        has_labels = "Ground_Truth" in df.columns
        if has_labels:
            df["_y"] = (df["Ground_Truth"] == "Anomaly").astype(int)

        # Batch-disjoint train/validation split for threshold/weight selection
        if has_labels:
            train_part, val_part = ev.group_train_val_split(df, test_size=0.3, random_state=self.config.random_state)
        else:
            train_part, val_part = df, df

        # 1) Recompute batch statistics + component features on TRAIN fold,
        #    apply the same statistics onto the VAL fold (val batches are
        #    disjoint from train batches, so this simulates "new batches").
        # Because batches don't overlap between train/val here, each side
        # needs its own batch stats computed from itself (that's the point
        # of batch-relative detection -- we can't borrow another batch's
        # statistics). So we compute stats per-fold, from each fold's own data.
        train_stats = fe.compute_batch_statistics(train_part, value_col)
        train_feat = fe.add_component_features(train_part, train_stats, value_col)

        val_stats = fe.compute_batch_statistics(val_part, value_col)
        val_feat = fe.add_component_features(val_part, val_stats, value_col)

        # 2) Fit Isolation Forest on TRAIN features only
        self.isolation_forest = fit_isolation_forest(
            train_feat[self.config.if_feature_columns], random_state=self.config.random_state
        )

        # 3) Score validation fold with the fitted IF + compute hybrid score
        #    using default weights first, then thresholds/weights are tuned
        #    against validation labels if available.
        val_if_score = isolation_forest_anomaly_score(self.isolation_forest, val_feat[self.config.if_feature_columns])

        if has_labels:
            y_val = val_feat["_y"].to_numpy()

            # --- Tune robust-z threshold (Step 4, Model 1) ---
            best_thr, best_m = ev.select_best_threshold(
                y_val, val_feat["robust_z"].abs().to_numpy(), ROBUST_Z_CANDIDATE_THRESHOLDS, beta=2.0
            )
            self.config.robust_z_threshold = best_thr
            if verbose:
                print(f"[fit] Selected robust_z threshold = {best_thr} "
                      f"(F2={best_m['f2']:.3f}, recall={best_m['recall']:.3f}, FN={best_m['FN']})")

            # --- Tune IQR multiplier (Step 4, Model 2) ---
            best_iqr_thr, best_iqr_m = None, None
            best_key = (-1, -1)
            for mult in IQR_MULTIPLIER_CANDIDATES:
                pred = iqr_predict(val_feat[value_col], val_feat["q1"], val_feat["q3"], val_feat["iqr"], mult)
                from sklearn.metrics import fbeta_score
                f2 = fbeta_score(y_val, pred, beta=2, zero_division=0)
                rec = (pred & (y_val == 1)).sum() / max(y_val.sum(), 1)
                key = (round(f2, 6), round(rec, 6))
                if key > best_key:
                    best_key = key
                    best_iqr_thr = mult
            self.config.iqr_multiplier = best_iqr_thr
            if verbose:
                print(f"[fit] Selected IQR multiplier = {best_iqr_thr}")

            # --- Tune hybrid weights: small grid search, optimize F2 ---
            weight_grid = [
                HybridWeights(0.5, 0.3, 0.2),
                HybridWeights(0.4, 0.4, 0.2),
                HybridWeights(0.6, 0.2, 0.2),
                HybridWeights(0.3, 0.5, 0.2),
                HybridWeights(0.5, 0.2, 0.3),
                HybridWeights(0.4, 0.3, 0.3),
                HybridWeights(0.7, 0.2, 0.1),
            ]
            best_w, best_w_key, best_w_score, best_w_thr = None, (-1, -1), None, None
            for w in weight_grid:
                scores = compute_hybrid_score(val_feat["robust_z"], val_feat["iqr_distance"], val_feat["iqr"], val_if_score, w)
                thr, m = ev.select_best_threshold(y_val, scores, list(range(10, 91, 5)), beta=2.0)
                key = (round(m["f2"], 6), round(m["recall"], 6))
                if key > best_w_key:
                    best_w_key = key
                    best_w = w
                    best_w_score = m
                    best_w_thr = thr
            self.config.hybrid_weights = best_w.as_dict()
            self.config.score_band_warning_max = best_w_thr
            self.config.score_band_normal_max = max(10.0, best_w_thr - 25)
            if verbose:
                print(f"[fit] Selected hybrid weights = {self.config.hybrid_weights}, "
                      f"anomaly-band threshold = {best_w_thr} "
                      f"(F2={best_w_score['f2']:.3f}, recall={best_w_score['recall']:.3f}, FN={best_w_score['FN']})")

        self._is_fitted = True
        return self

    # -- inference -----------------------------------------------------------

    def predict(self, new_df: pd.DataFrame, value_col: str = "Value_0h_uA") -> pd.DataFrame:
        if not self._is_fitted or self.isolation_forest is None:
            raise RuntimeError("DynamicOutlierDetector must be fit() before predict().")

        from . import preprocessing as pp
        df = pp.clean_inference_frame(new_df)

        stats = fe.compute_batch_statistics(df, value_col)
        feat = fe.add_component_features(df, stats, value_col)

        if_score = isolation_forest_anomaly_score(self.isolation_forest, feat[self.config.if_feature_columns])
        weights = HybridWeights(**self.config.hybrid_weights)
        hybrid = compute_hybrid_score(feat["robust_z"], feat["iqr_distance"], feat["iqr"], if_score, weights)

        risk_level = np.where(
            hybrid >= self.config.score_band_warning_max, "Anomaly",
            np.where(hybrid >= self.config.score_band_normal_max, "Warning", "Normal"),
        )

        result = pd.DataFrame({
            "Component_ID": feat["Component_ID"],
            "Batch_ID": feat["Batch_ID"],
            "Value_0h_uA": feat[value_col],
            "Batch_Median": feat["batch_median"],
            "Batch_MAD": feat["mad"],
            "Robust_Z": feat["robust_z"],
            "Relative_to_Batch_Median": feat["relative_to_median"],
            "IQR_Outlier": feat["is_iqr_outlier"],
            "Isolation_Forest_Score": if_score,
            "Dynamic_Anomaly_Score": hybrid,
            "Risk_Level": risk_level,
            "Predicted_Class": np.where(risk_level == "Anomaly", "Anomaly", "Normal"),
        })
        return result

    # -- persistence -----------------------------------------------------------

    def save(self, model_path: str, config_path: str) -> None:
        joblib.dump(self.isolation_forest, model_path)
        self.config.save(config_path)

    @classmethod
    def load(cls, model_path: str, config_path: str) -> "DynamicOutlierDetector":
        config = DetectorConfig.load(config_path)
        obj = cls(config=config)
        obj.isolation_forest = joblib.load(model_path)
        obj._is_fitted = True
        return obj
