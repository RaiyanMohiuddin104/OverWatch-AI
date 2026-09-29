"""
run_pipeline.py
----------------
OverWatch-AI -- Module B: Time-Series Drift Predictor.

Runs the full Module B pipeline against the real training CSV and
prints/saves actual results at every stage: load, clean, EDA, train
and compare regressors, select + tune the best one, apply the
safety-drift logic, and save everything app.py needs at inference time
(models/module_b_regressor.pkl, models/module_b_config.json).

Scope note: this file implements Module B only (168h time-series drift
prediction from Value_0h_uA / Value_24h_uA). Module A (batch-relative
anomaly detection at 0h) is out of scope and not used here.

Run with:
    pip install -r requirements.txt
    python run_pipeline.py
    streamlit run app.py
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
sns.set_style("whitegrid")
pd.set_option("display.width", 200)

DATA_PATH = "data/OverWatch-AI_Training_Dataset.csv"
COL_0H, COL_24H, COL_168H = "Value_0h_uA", "Value_24h_uA", "Value_168h_uA"

print("#" * 70)
print("STEP 1-2: LOAD + INSPECT DATA")
print("#" * 70)
df = pd.read_csv(DATA_PATH)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
for c in (COL_0H, COL_24H, COL_168H):
    assert c in df.columns, f"Expected column '{c}' not found."
print(f"Confirmed columns -> input: {COL_0H}, {COL_24H} | target: {COL_168H}")

print()
print("#" * 70)
print("STEP 3: CLEAN DATA")
print("#" * 70)
n_before = len(df)
df = df.drop_duplicates().copy()
for c in (COL_0H, COL_24H, COL_168H):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[COL_0H, COL_24H, COL_168H])
df = df[(df[COL_0H] > 0) & (df[COL_24H] > 0) & (df[COL_168H] > 0)]
print(f"Rows before: {n_before}  ->  after cleaning: {len(df)}")

print()
print("#" * 70)
print("STEP 4: EXPLORATORY DATA ANALYSIS")
print("#" * 70)
corr = df[[COL_0H, COL_24H, COL_168H]].corr()
print("Correlation with target (Value_168h_uA):")
print(corr[COL_168H])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, [COL_0H, COL_24H, COL_168H]):
    sns.histplot(df[col], kde=True, ax=ax, color="#3B7DD8")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig("outputs/figures/01_distributions.png", dpi=140)
plt.close()

plt.figure(figsize=(5, 4))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("outputs/figures/02_correlation_heatmap.png", dpi=140)
plt.close()

plt.figure(figsize=(6, 5))
sns.scatterplot(data=df, x=COL_24H, y=COL_168H, alpha=0.4, color="#2E8B57")
plt.title(f"{COL_24H} vs {COL_168H}")
plt.tight_layout()
plt.savefig("outputs/figures/03_feature_relationship.png", dpi=140)
plt.close()
print("Saved EDA figures to outputs/figures/")

print()
print("#" * 70)
print("STEP 5: TRAIN / TEST SPLIT")
print("#" * 70)
X = df[[COL_0H, COL_24H]].values
y = df[COL_168H].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"Train: {len(X_train)}  Test: {len(X_test)}")

print()
print("#" * 70)
print("STEP 6: TRAIN + COMPARE REGRESSION MODELS")
print("#" * 70)
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
}
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        random_state=RANDOM_STATE, verbosity=0
    )
else:
    print("[NOTE] xgboost not installed -- skipping (spec says 'if available').")

results, fitted = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds),
    }
    fitted[name] = model

results_df = pd.DataFrame(results).T.sort_values("RMSE")
print(results_df.to_string())
results_df.to_csv("outputs/model_comparison.csv")

best_name = results_df.index[0]
best_model = fitted[best_name]
print(f"\nBest model selected: {best_name}")

print()
print("#" * 70)
print("STEP 7: HYPERPARAMETER TUNING (if applicable to the winning model)")
print("#" * 70)
tuned = False
if best_name == "Random Forest":
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3, 5]},
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    candidate, tuned = grid.best_estimator_, True
elif best_name == "Gradient Boosting":
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {"n_estimators": [150, 300], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 4]},
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    candidate, tuned = grid.best_estimator_, True
elif XGBOOST_AVAILABLE and best_name == "XGBoost":
    grid = GridSearchCV(
        XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        {"n_estimators": [200, 400], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4, 6]},
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    candidate, tuned = grid.best_estimator_, True

if tuned:
    tuned_rmse = np.sqrt(mean_squared_error(y_test, candidate.predict(X_test)))
    baseline_rmse = results_df.loc[best_name, "RMSE"]
    print(f"Tuned RMSE={tuned_rmse:.4f} vs baseline RMSE={baseline_rmse:.4f}")
    if tuned_rmse < baseline_rmse:
        best_model = candidate
        print("Tuning improved performance -- using tuned model.")
    else:
        print("Tuning did not improve performance -- keeping baseline model.")
else:
    print(f"{best_name} has no hyperparameters worth grid-searching -- skipped.")

final_preds = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_preds)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)
print(f"\nFinal model: {best_name}  MAE={final_mae:.4f}  RMSE={final_rmse:.4f}  R2={final_r2:.4f}")

print()
print("#" * 70)
print("STEP 8: RESULT VISUALIZATIONS")
print("#" * 70)
plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_preds, alpha=0.4, color="#3B7DD8")
lims = [min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())]
plt.plot(lims, lims, "r--", label="Perfect Prediction")
plt.xlabel("Actual Value_168h_uA"); plt.ylabel("Predicted Value_168h_uA")
plt.title(f"Predicted vs Actual -- {best_name}"); plt.legend()
plt.tight_layout()
plt.savefig("outputs/figures/04_predicted_vs_actual.png", dpi=140)
plt.close()

residuals = y_test - final_preds
plt.figure(figsize=(6, 5))
sns.scatterplot(x=final_preds, y=residuals, alpha=0.4, color="#C0392B")
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted Value_168h_uA"); plt.ylabel("Residual")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("outputs/figures/05_residuals.png", dpi=140)
plt.close()

if hasattr(best_model, "feature_importances_"):
    vals = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    vals = best_model.coef_
else:
    vals = None
if vals is not None:
    plt.figure(figsize=(5, 4))
    sns.barplot(x=[COL_0H, COL_24H], y=vals, color="#2E8B57")
    plt.title("Feature Importance / Coefficients")
    plt.tight_layout()
    plt.savefig("outputs/figures/06_feature_importance.png", dpi=140)
    plt.close()
print("Saved result figures to outputs/figures/")

print()
print("#" * 70)
print("STEP 9: SAFETY DRIFT LOGIC (computed on training data)")
print("#" * 70)
historical_slope = (df[COL_168H] - df[COL_24H]) / (168 - 24)
safety_threshold = float(historical_slope.mean() + 2 * historical_slope.std())
print(f"Historical slope mean: {historical_slope.mean():.5f}")
print(f"Historical slope std:  {historical_slope.std():.5f}")
print(f"Safety threshold (mean + 2*std): {safety_threshold:.5f}")

print()
print("#" * 70)
print("STEP 10: BUILD FULL-DATASET PREDICTIONS + DECISIONS")
print("#" * 70)
all_preds = best_model.predict(df[[COL_0H, COL_24H]].values)
drift_rate = (all_preds - df[COL_24H].values) / (168 - 24)
decision = np.where(drift_rate > safety_threshold, "EARLY REJECTION", "ACCEPTABLE")

final_df = pd.DataFrame({
    "Component_ID": df.get("Component_ID", pd.Series(range(len(df)))),
    "Batch_ID": df.get("Batch_ID", ""),
    COL_0H: df[COL_0H],
    COL_24H: df[COL_24H],
    "Actual_168h": df[COL_168H],
    "Predicted_168h": all_preds,
    "Drift_Rate": drift_rate,
    "Safety_Threshold": safety_threshold,
    "Decision": decision,
})
final_df.to_csv("outputs/module_b_predictions.csv", index=False)
print(f"Saved outputs/module_b_predictions.csv ({len(final_df)} rows)")
print(pd.Series(decision).value_counts())

print()
print("#" * 70)
print("STEP 11: SAVE MODEL + CONFIG (used by app.py at inference time)")
print("#" * 70)
joblib.dump(best_model, "models/module_b_regressor.pkl")
config = {
    "model_name": best_name,
    "feature_order": [COL_0H, COL_24H],
    "safety_threshold": safety_threshold,
    "historical_slope_mean": float(historical_slope.mean()),
    "historical_slope_std": float(historical_slope.std()),
    "test_MAE": float(final_mae),
    "test_RMSE": float(final_rmse),
    "test_R2": float(final_r2),
}
with open("models/module_b_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("Saved models/module_b_regressor.pkl and models/module_b_config.json")


def predict_component_health(value_0h, value_24h, model=best_model, threshold=safety_threshold):
    """Deployment function: predict 168h value + safety-drift decision for one component."""
    pred = float(model.predict(np.array([[value_0h, value_24h]]))[0])
    rate = (pred - value_24h) / (168 - 24)
    return {
        "Predicted_168h": round(pred, 4),
        "Drift_Rate": round(rate, 5),
        "Safety_Threshold": round(threshold, 5),
        "Decision": "EARLY REJECTION" if rate > threshold else "ACCEPTABLE",
    }


print()
print("#" * 70)
print("STEP 12: DEPLOYMENT FUNCTION SMOKE TEST")
print("#" * 70)
for _, row in df.sample(3, random_state=RANDOM_STATE).iterrows():
    out = predict_component_health(row[COL_0H], row[COL_24H])
    print(f"Component {row.get('Component_ID', '?')}: actual_168h={row[COL_168H]:.3f}  ->  {out}")
