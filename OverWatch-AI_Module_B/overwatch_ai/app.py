"""
app.py
------
OverWatch-AI Module B -- Streamlit drift-prediction dashboard.

Run with:
    streamlit run app.py

Upload a CSV with at least Value_0h_uA and Value_24h_uA (Component_ID,
Batch_ID, Value_168h_uA are optional and only used for display/comparison
if present). Requires models/module_b_regressor.pkl and
models/module_b_config.json -- produced by `python run_pipeline.py`.
"""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="OverWatch-AI · Module B", layout="wide")

MODEL_PATH = "models/module_b_regressor.pkl"
CONFIG_PATH = "models/module_b_config.json"
COL_0H, COL_24H, COL_168H = "Value_0h_uA", "Value_24h_uA", "Value_168h_uA"


@st.cache_resource
def load_model_and_config():
    model = joblib.load(MODEL_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    return model, config


def predict_component_health(value_0h, value_24h, model, threshold):
    pred = float(model.predict(np.array([[value_0h, value_24h]]))[0])
    rate = (pred - value_24h) / (168 - 24)
    return {
        "Predicted_168h": round(pred, 4),
        "Drift_Rate": round(rate, 5),
        "Safety_Threshold": round(threshold, 5),
        "Decision": "EARLY REJECTION" if rate > threshold else "ACCEPTABLE",
    }


st.title("OverWatch-AI — Module B: Time-Series Drift Predictor")
st.caption(
    "Predicts 168h leakage current from 0h/24h readings and flags components "
    "whose predicted drift rate exceeds the historical safety threshold -- "
    "enabling early rejection before the full 168h burn-in completes."
)

try:
    model, config = load_model_and_config()
except FileNotFoundError:
    st.error("Model not found. Run `python run_pipeline.py` first to train and save it.")
    st.stop()

threshold = config["safety_threshold"]

c1, c2, c3 = st.columns(3)
c1.metric("Model", config["model_name"])
c2.metric("Test R²", f"{config['test_R2']:.3f}")
c3.metric("Safety Threshold", f"{threshold:.5f} µA/hr")

st.divider()

tab_batch, tab_single = st.tabs(["Batch scoring (CSV upload)", "Single-component lookup"])

with tab_batch:
    uploaded = st.file_uploader(
        f"Upload a CSV with {COL_0H} and {COL_24H} columns", type="csv"
    )
    if uploaded is None:
        st.info("Upload a CSV to score components, or use the bundled training set as a demo.")
        use_demo = st.button("Use bundled training dataset as demo")
        if not use_demo:
            st.stop()
        raw_df = pd.read_csv("data/OverWatch-AI_Training_Dataset.csv")
    else:
        raw_df = pd.read_csv(uploaded)

    if COL_0H not in raw_df.columns or COL_24H not in raw_df.columns:
        st.error(f"CSV must contain '{COL_0H}' and '{COL_24H}' columns.")
        st.stop()

    with st.spinner("Scoring components..."):
        preds = model.predict(raw_df[[COL_0H, COL_24H]].values)
        drift_rate = (preds - raw_df[COL_24H].values) / (168 - 24)
        decision = np.where(drift_rate > threshold, "EARLY REJECTION", "ACCEPTABLE")

        results = raw_df.copy()
        results["Predicted_168h"] = preds
        results["Drift_Rate"] = drift_rate
        results["Decision"] = decision

    total = len(results)
    n_reject = (results["Decision"] == "EARLY REJECTION").sum()
    n_accept = total - n_reject

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Components", total)
    m2.metric("Acceptable", n_accept)
    m3.metric("Early Rejection", n_reject)

    st.subheader(f"Scored components ({total} shown)")
    display_cols = [c for c in ["Component_ID", "Batch_ID", COL_0H, COL_24H, COL_168H,
                                 "Predicted_168h", "Drift_Rate", "Decision"] if c in results.columns]
    st.dataframe(
        results[display_cols].sort_values("Drift_Rate", ascending=False),
        use_container_width=True, height=420,
    )

    st.divider()
    st.subheader("Predicted vs Actual (if Value_168h_uA present)")
    if COL_168H in results.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = results["Decision"].map({"EARLY REJECTION": "#C44E52", "ACCEPTABLE": "#4C72B0"})
        ax.scatter(results[COL_168H], results["Predicted_168h"], c=colors, alpha=0.5, edgecolor="k", linewidth=0.2)
        lims = [results[COL_168H].min(), results[COL_168H].max()]
        ax.plot(lims, lims, "k--", label="Perfect prediction")
        ax.set_xlabel("Actual Value_168h_uA"); ax.set_ylabel("Predicted Value_168h_uA")
        ax.legend()
        st.pyplot(fig)
    else:
        st.caption(f"No '{COL_168H}' column in the uploaded CSV -- skipping actual-vs-predicted comparison.")

with tab_single:
    st.subheader("Predict a single component")
    v0 = st.number_input("Value_0h_uA", min_value=0.0, value=10.0, step=0.1)
    v24 = st.number_input("Value_24h_uA", min_value=0.0, value=10.5, step=0.1)
    if st.button("Predict"):
        out = predict_component_health(v0, v24, model, threshold)
        d1, d2, d3 = st.columns(3)
        d1.metric("Predicted 168h", f"{out['Predicted_168h']:.3f} µA")
        d2.metric("Drift Rate", f"{out['Drift_Rate']:.5f} µA/hr")
        d3.metric("Decision", out["Decision"])
        if out["Decision"] == "EARLY REJECTION":
            st.error(f"⚠ EARLY REJECTION — drift rate exceeds safety threshold ({threshold:.5f} µA/hr).")
        else:
            st.success("✓ ACCEPTABLE — drift rate within the historical safety envelope.")
