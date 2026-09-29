"""
run_pipeline.py
----------------
Runs the full Module A pipeline against the real training CSV and
prints/saves actual results at every stage. This is the script that
was used to produce outputs/module_a_predictions.csv,
models/module_a_isolation_forest.pkl and models/module_a_config.json.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src import preprocessing as pp
from src import feature_engineering as fe
from src import evaluation as ev
from src import anomaly_detector as ad
from src import explainability as ex

DATA_PATH = "data/OverWatch-AI_Training_Dataset.csv"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

print("#" * 70)
print("STEP 1-2: LOAD + VALIDATE DATA")
print("#" * 70)
df = pp.load_training_data(DATA_PATH)
summary = pp.summarize_dataset(df)
print(f"Rows: {summary['n_rows']}, Columns: {summary['n_columns']}, Batches: {summary['n_batches']}")
print(f"Missing values total: {sum(summary['missing_values'].values())}")
print(f"Ground truth counts: {summary['ground_truth_counts']}")

print()
print("#" * 70)
print("STEP 6: BATCH-DISJOINT TRAIN/VALIDATION SPLIT (GroupShuffleSplit on Batch_ID)")
print("#" * 70)
train_df, val_df = ev.group_train_val_split(df, test_size=0.3, random_state=42)
print(f"Train: {len(train_df)} rows across {train_df['Batch_ID'].nunique()} batches")
print(f"Val:   {len(val_df)} rows across {val_df['Batch_ID'].nunique()} batches")
overlap = set(train_df['Batch_ID']) & set(val_df['Batch_ID'])
print(f"Batch overlap between train/val: {len(overlap)} (must be 0)")

print()
print("#" * 70)
print("STEP 3-4: FIT DETECTOR (batch feature engineering + threshold/weight tuning on val fold)")
print("#" * 70)
detector = ad.DynamicOutlierDetector()
detector.fit(df, verbose=True)

print()
print("Final selected configuration:")
print(detector.config)

print()
print("#" * 70)
print("STEP 6 (cont.): GroupKFold cross-validation of the fitted-config pipeline")
print("#" * 70)
# Evaluate the SELECTED config's stability across 5 batch-disjoint folds
fold_metrics = []
for fold_i, (tr_idx, va_idx) in enumerate(ev.group_kfold_indices(df, n_splits=5)):
    tr, va = df.iloc[tr_idx], df.iloc[va_idx]
    va_stats = fe.compute_batch_statistics(va)
    va_feat = fe.add_component_features(va, va_stats)
    if_score_fold = ad.isolation_forest_anomaly_score(detector.isolation_forest, va_feat[detector.config.if_feature_columns])
    weights = ad.HybridWeights(**detector.config.hybrid_weights)
    scores = ad.compute_hybrid_score(va_feat["robust_z"], va_feat["iqr_distance"], va_feat["iqr"], if_score_fold, weights)
    y_va = (va_feat["Ground_Truth"] == "Anomaly").astype(int).to_numpy()
    pred = (scores >= detector.config.score_band_warning_max).astype(int)
    m = ev.compute_metrics(y_va, pred, y_score=scores)
    m["fold"] = fold_i
    fold_metrics.append(m)
    print(f"Fold {fold_i}: recall={m['recall']:.3f} precision={m['precision']:.3f} f2={m['f2']:.3f} FN={m['FN']} FP={m['FP']}")

fold_df = pd.DataFrame(fold_metrics)
print()
print("Mean +/- std across folds:")
print(f"  recall:    {fold_df['recall'].mean():.3f} +/- {fold_df['recall'].std():.3f}")
print(f"  precision: {fold_df['precision'].mean():.3f} +/- {fold_df['precision'].std():.3f}")
print(f"  f2:        {fold_df['f2'].mean():.3f} +/- {fold_df['f2'].std():.3f}")
print(f"  FN total:  {fold_df['FN'].sum()}  FP total: {fold_df['FP'].sum()}")

print()
print("#" * 70)
print("STEP 8: COMPARE AGAINST TRADITIONAL ABSOLUTE-LIMIT SCREENING (on validation fold)")
print("#" * 70)
y_val = (val_df["Ground_Truth"] == "Anomaly").astype(int).to_numpy()
trad = ev.traditional_screening_baseline(val_df["Value_0h_uA"].to_numpy(), y_val, limit=50.0)
print("Traditional screening (Value_0h_uA > 50):")
for k in ["TP", "FP", "TN", "FN", "recall", "precision", "f2"]:
    print(f"  {k}: {trad[k]}")

# Dynamic detector on the SAME validation rows
val_stats = fe.compute_batch_statistics(val_df)
val_feat = fe.add_component_features(val_df, val_stats)
if_score_val = ad.isolation_forest_anomaly_score(detector.isolation_forest, val_feat[detector.config.if_feature_columns])
weights = ad.HybridWeights(**detector.config.hybrid_weights)
dyn_scores = ad.compute_hybrid_score(val_feat["robust_z"], val_feat["iqr_distance"], val_feat["iqr"], if_score_val, weights)
dyn_pred = (dyn_scores >= detector.config.score_band_warning_max).astype(int)
dyn_metrics = ev.compute_metrics(y_val, dyn_pred, y_score=dyn_scores)
print()
print("Dynamic Module-A detector:")
for k in ["TP", "FP", "TN", "FN", "recall", "precision", "f2"]:
    print(f"  {k}: {dyn_metrics[k]}")

# Components that PASS absolute limit but ARE flagged by dynamic model
val_feat["_abs_pass"] = val_feat["Value_0h_uA"] <= 50.0
val_feat["_dyn_flag"] = dyn_pred
rescued = val_feat[(val_feat["_abs_pass"]) & (val_feat["_dyn_flag"] == 1) & (val_feat["Ground_Truth"] == "Anomaly")]
print()
print(f"Anomalies that PASS the 50µA absolute limit but ARE caught by the dynamic detector (val fold): {len(rescued)}")
if len(rescued):
    print(rescued[["Component_ID", "Batch_ID", "Value_0h_uA", "batch_median", "robust_z"]].head(10).to_string(index=False))

print()
print("#" * 70)
print("STEP 11: BUILD FINAL PREDICTIONS DATAFRAME (full dataset, using per-batch self-relative stats)")
print("#" * 70)
full_stats = fe.compute_batch_statistics(df)
full_feat = fe.add_component_features(df, full_stats)
if_score_full = ad.isolation_forest_anomaly_score(detector.isolation_forest, full_feat[detector.config.if_feature_columns])
full_hybrid = ad.compute_hybrid_score(full_feat["robust_z"], full_feat["iqr_distance"], full_feat["iqr"], if_score_full, weights)
full_risk = ad.score_to_risk_level(full_hybrid)
# apply tuned bands, not the hardcoded defaults inside score_to_risk_level
full_risk = np.where(full_hybrid >= detector.config.score_band_warning_max, "Anomaly",
             np.where(full_hybrid >= detector.config.score_band_normal_max, "Warning", "Normal"))

final_df = pd.DataFrame({
    "Component_ID": full_feat["Component_ID"],
    "Batch_ID": full_feat["Batch_ID"],
    "Value_0h_uA": full_feat["Value_0h_uA"],
    "Absolute_Limit_Pass": full_feat["Value_0h_uA"] <= 50.0,
    "Batch_Median": full_feat["batch_median"],
    "Batch_MAD": full_feat["mad"],
    "Robust_Z": full_feat["robust_z"],
    "Relative_to_Batch_Median": full_feat["relative_to_median"],
    "IQR_Outlier": full_feat["is_iqr_outlier"],
    "Isolation_Forest_Score": if_score_full,
    "Dynamic_Anomaly_Score": full_hybrid,
    "Predicted_Class": np.where(full_risk == "Anomaly", "Anomaly", "Normal"),
    "Risk_Level": full_risk,
})
final_df = ex.add_explanations(final_df)
final_df.to_csv("outputs/module_a_predictions.csv", index=False)
print(f"Saved outputs/module_a_predictions.csv ({len(final_df)} rows)")
print(final_df["Risk_Level"].value_counts())

print()
print("#" * 70)
print("STEP 9: EXAMPLE EXPLANATION")
print("#" * 70)
# pick a real flagged component that passes absolute limit but is anomalous
example_candidates = final_df[(final_df["Risk_Level"] == "Anomaly") & (final_df["Absolute_Limit_Pass"])]
if len(example_candidates):
    example_id = example_candidates.iloc[0]["Component_ID"]
    print(ex.explain_component(example_id, final_df))

print()
print("#" * 70)
print("STEP 12: SAVE MODEL + CONFIG")
print("#" * 70)
detector.save("models/module_a_isolation_forest.pkl", "models/module_a_config.json")
print("Saved models/module_a_isolation_forest.pkl and models/module_a_config.json")

print()
print("#" * 70)
print("STEP 13: INFERENCE PIPELINE SMOKE TEST (simulate a brand-new, unseen batch)")
print("#" * 70)
loaded = ad.DynamicOutlierDetector.load("models/module_a_isolation_forest.pkl", "models/module_a_config.json")
rng = np.random.RandomState(123)
new_batch = pd.DataFrame({
    "Component_ID": [f"NEW{i:03d}" for i in range(20)],
    "Batch_ID": ["B999"] * 20,
    "Value_0h_uA": np.concatenate([rng.normal(12, 1.5, 18), [40.0, 45.0]]),  # 2 injected outliers
})
new_results = loaded.predict(new_batch)
print(new_results[["Component_ID", "Value_0h_uA", "Robust_Z", "Dynamic_Anomaly_Score", "Risk_Level"]].to_string(index=False))
