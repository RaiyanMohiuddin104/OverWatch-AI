"""
app.py
------
OverWatch-AI Module A -- Streamlit QA dashboard.

Run with:
    streamlit run app.py

Upload a CSV with at least Component_ID, Batch_ID, Value_0h_uA
(Ground_Truth / other columns are optional and ignored by the model,
but Absolute_Limit_50uA_Pass or Value_168h_uA will be used purely for
side-by-side display if present).
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src import anomaly_detector as ad
from src import explainability as ex

st.set_page_config(page_title="OverWatch-AI · Module A", layout="wide")
sns.set_style("whitegrid")

MODEL_PATH = "models/module_a_isolation_forest.pkl"
CONFIG_PATH = "models/module_a_config.json"


@st.cache_resource
def load_detector():
    return ad.DynamicOutlierDetector.load(MODEL_PATH, CONFIG_PATH)


st.title("OverWatch-AI — Module A: Dynamic Outlier Detection")
st.caption(
    "Batch-aware anomaly screening for electronic component reliability testing. "
    "Flags components that pass the absolute datasheet limit but are statistically "
    "abnormal relative to their own manufacturing batch."
)

uploaded = st.file_uploader("Upload a component CSV (Component_ID, Batch_ID, Value_0h_uA required)", type="csv")

if uploaded is None:
    st.info("Upload a CSV to run the detector, or use the bundled training set as a demo.")
    use_demo = st.button("Use bundled training dataset as demo")
    if not use_demo:
        st.stop()
    raw_df = pd.read_csv("data/OverWatch-AI_Training_Dataset.csv")
else:
    raw_df = pd.read_csv(uploaded)

detector = load_detector()

with st.spinner("Scoring components..."):
    results = detector.predict(raw_df)
    results = ex.add_explanations(results)
    # attach absolute limit pass info if the source had it, else compute from 0h value
    if "Absolute_Limit_50uA_Pass" in raw_df.columns:
        results = results.merge(
            raw_df[["Component_ID", "Absolute_Limit_50uA_Pass"]], on="Component_ID", how="left"
        )
    else:
        results["Absolute_Limit_50uA_Pass"] = results["Value_0h_uA"] <= 50.0

total = len(results)
n_normal = (results["Risk_Level"] == "Normal").sum()
n_warning = (results["Risk_Level"] == "Warning").sum()
n_anomaly = (results["Risk_Level"] == "Anomaly").sum()
n_rescued = ((results["Risk_Level"] == "Anomaly") & (results["Absolute_Limit_50uA_Pass"])).sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Components", total)
c2.metric("Normal", n_normal)
c3.metric("Warning", n_warning)
c4.metric("Anomalies", n_anomaly)
c5.metric("Anomalies that PASSED absolute limit", n_rescued)

st.divider()

col_filters = st.columns(3)
batch_options = ["All"] + sorted(results["Batch_ID"].unique().tolist())
sel_batch = col_filters[0].selectbox("Filter by Batch", batch_options)
sel_risk = col_filters[1].multiselect(
    "Filter by Risk Level", ["Normal", "Warning", "Anomaly"], default=["Warning", "Anomaly"]
)
sel_class = col_filters[2].multiselect(
    "Filter by Predicted Class", ["Normal", "Anomaly"], default=["Anomaly"]
)

filtered = results.copy()
if sel_batch != "All":
    filtered = filtered[filtered["Batch_ID"] == sel_batch]
if sel_risk:
    filtered = filtered[filtered["Risk_Level"].isin(sel_risk)]
if sel_class:
    filtered = filtered[filtered["Predicted_Class"].isin(sel_class)]

st.subheader(f"Flagged components ({len(filtered)} shown)")
display_cols = [
    "Component_ID", "Batch_ID", "Value_0h_uA", "Dynamic_Anomaly_Score",
    "Predicted_Class", "Risk_Level", "Explanation",
]
st.dataframe(
    filtered[display_cols].sort_values("Dynamic_Anomaly_Score", ascending=False),
    use_container_width=True,
    height=420,
)

st.divider()
st.subheader("Batch visualization")
if sel_batch != "All":
    batch_plot_df = results[results["Batch_ID"] == sel_batch]
else:
    top_batch = results.groupby("Batch_ID")["Dynamic_Anomaly_Score"].max().idxmax()
    batch_plot_df = results[results["Batch_ID"] == top_batch]
    st.caption(f"No batch selected — showing batch with highest risk score: {top_batch}")

fig, ax = plt.subplots(figsize=(11, 4.5))
colors = batch_plot_df["Risk_Level"].map(
    {"Anomaly": "#C44E52", "Warning": "#DD8452", "Normal": "#4C72B0"}
)
ax.scatter(range(len(batch_plot_df)), batch_plot_df["Value_0h_uA"], c=colors, edgecolor="k", linewidth=0.3)
ax.axhline(batch_plot_df["Batch_Median"].iloc[0], color="green", label="Batch median")
ax.axhline(50, color="red", linestyle=":", label="50 µA absolute limit")
ax.set_xlabel("Component index within batch")
ax.set_ylabel("Value_0h_uA (µA)")
ax.legend()
st.pyplot(fig)

st.divider()
with st.expander("Explain a specific component"):
    comp_id = st.selectbox("Component ID", results["Component_ID"].tolist())
    st.text(ex.explain_component(comp_id, results))
