# OverWatch-AI — Module A: Dynamic Outlier Detection

Batch-aware anomaly screening for electronic component reliability testing
(ESS / burn-in). Flags components that pass the absolute datasheet limit
(e.g. 50 µA leakage) but are statistically abnormal relative to their own
manufacturing batch.

**Scope note:** this repo implements Module A only (static, batch-relative
outlier detection at the 0h incoming-inspection reading). Module B
(168h time-series degradation prediction) is explicitly out of scope and
none of its inputs (`Value_24h/96h/168h_uA`, `Drift_*_pct`,
`Absolute_Limit_50uA_Pass`) are used as model features.

## 1. Dataset (as actually inspected)

| | |
|---|---|
| Rows | 5,000 |
| Batches | 50 (exactly 100 components each) |
| Missing values | 0 |
| Ground truth classes | Normal 3,534 (70.7%) · Degrading 741 (14.8%) · Anomaly 477 (9.5%) · Latent_Defect 248 (5.0%) |
| Batch mean leakage (0h) range | 8.1 µA – 23.9 µA across the 50 batches |

**Key finding that motivates Module A:** at t=0h, **100% of the 477
labeled `Anomaly` components pass the absolute 50 µA limit** (max value
49.0 µA). A pure absolute-limit screen at incoming inspection has **0%
recall** on this class by construction — the whole point of the exercise.
(58.1% of them eventually fail the limit by 168h, but that's Module B's
signal, not available at incoming inspection.)

**Leakage check on the supplied CSV:** the pre-computed
`Batch_Mean_0h_uA` / `Batch_Std_0h_uA` / `Relative_to_Batch_Mean` columns
are computed over *all* rows in a batch, anomalies included — i.e. they
are contaminated by the very components we're trying to detect. Module A
**recomputes all batch statistics itself** from `Value_0h_uA` and never
reads those three columns (see `src/preprocessing.py::LEAKAGE_COLUMNS`).

## 2. Method

1. **Batch statistics** (mean, median, std, Q1/Q3/IQR, MAD) recomputed
   per `Batch_ID` from `Value_0h_uA` only.
2. **Component-level features**: standard z-score, robust z-score
   (MAD-based), relative-to-median ratio, % diff from median, IQR
   outlier distance, in-batch percentile rank.
3. **Three detectors**, compared head-to-head:
   - **Model 1 — Robust Z-score**, threshold tuned over {2.0 … 5.0}
   - **Model 2 — Batch-specific IQR**, multiplier tuned over {1.0, 1.2, 1.5}
   - **Model 3 — Isolation Forest** trained only on the batch-relative
     features above (never on raw `Value_0h_uA` or `Batch_ID` directly,
     so it can't learn spurious batch-ordinal relationships)
4. **Hybrid 0–100 score** = weighted blend of robust-z evidence,
   Isolation Forest evidence, and IQR evidence. Weights and score bands
   are grid-searched on a held-out validation split, not hand-picked.
5. **Validation strategy**: `GroupShuffleSplit`/`GroupKFold` grouped on
   `Batch_ID`, so no batch's components appear in both train and
   validation — this tests generalization to *unseen manufacturing
   batches*, which is the real deployment scenario.
6. **Optimization target**: F2 score (recall-weighted), not accuracy —
   false negatives (missed defects) are penalized far more than false
   positives, per the project's safety requirement.

## 3. Results (actual, from `run_pipeline.py`)

Selected configuration: robust-z threshold **3.5**, IQR multiplier
**1.5**, hybrid weights **{robust_z: 0.5, isolation_forest: 0.3, iqr:
0.2}**, anomaly band **≥ 50/100**, warning band **≥ 25/100**.

**5-fold GroupKFold (batch-disjoint) cross-validation:**

| Metric | Mean ± std |
|---|---|
| Recall | 1.000 ± 0.000 |
| Precision | 1.000 ± 0.000 |
| F2 | 1.000 ± 0.000 |
| Total FN across all folds | 0 |
| Total FP across all folds | 0 |

**Traditional (50 µA absolute) vs Dynamic (Module A), on the held-out
validation batches (1,500 components, 15 batches):**

| | TP | FP | TN | FN | Recall |
|---|---|---|---|---|---|
| Traditional screening | 0 | 0 | 1,361 | 139 | 0.0% |
| Dynamic Module A | 139 | 0 | 1,361 | 0 | 100% |

139 components in that validation slice pass the absolute limit outright
but are correctly caught by the batch-relative detector.

### Important caveat — please read before treating this as production-ready

The near-perfect scores above are **real, not fabricated**, but they
reflect a very clean synthetic dataset, not a proven real-world false-negative
rate of zero. Checking the raw feature values: every labeled `Anomaly`
component has `|robust_z| ≥ 15.2`, while every `Normal` component has
`|robust_z| ≤ 3.45` — a huge, clean gap with no overlap. That's why a
simple threshold already gets perfect separation; it is not evidence
that real fab data (with subtler, borderline defects) will separate this
cleanly. Two things worth noting in the dataset's favor, though:
- The `Degrading` and `Latent_Defect` classes (which Module A
  intentionally does *not* target — see below) genuinely overlap with
  `Normal` in 0h robust-z space (all three classes span roughly -3 to
  +3), confirming those failure modes really do need Module B's
  time-series signal, not a static 0h check.
- Before treating this detector as validated, it should be re-tested
  against real ESS/burn-in measurements or a noisier synthetic set with
  borderline anomalies.

### Ground-truth class handling

Per the task's assumption, only `Ground_Truth == "Anomaly"` is treated as
the Module-A positive class for threshold/weight tuning and reported
metrics. `Degrading` and `Latent_Defect` are look-normal-at-0h,
develop-over-time failure modes — Module B's territory — and are
**not** used as positives here. This is a deliberate, documented
assumption (see `anomaly_detector.py` docstring) and can be revisited if
requirements change.

## 4. Explainability

Every prediction includes a deterministic, template-based explanation
generated purely from the calculated numbers (no LLM, fully
reproducible) — see `src/explainability.py::explain_component`.
Example (real output, from `outputs/module_a_predictions.csv`):

```
Component ID: C00015
Batch: B001

Measured Value: 34.6578 µA
Absolute Limit: 50 µA
Absolute Screening: PASS

Batch Median: 9.0674 µA
Batch MAD: 0.2895 µA
Robust Z-score: 59.62
Percent Above Batch Median: 282.2%
Isolation Forest Score: 0.83
Dynamic Risk Score: 94.9/100

Classification: ANOMALY

Reason:
Although the component is below the absolute 50 µA datasheet limit, its
leakage current is significantly higher than the normal population of
its manufacturing batch (robust Z-score 59.6, 282% above the batch
median). The component therefore represents a strong batch-relative
statistical anomaly.
```

## 5. Project structure

```
overwatch_ai/
├── data/OverWatch-AI_Training_Dataset.csv
├── src/
│   ├── preprocessing.py        # loading, validation, leakage-column allowlist
│   ├── feature_engineering.py  # batch stats + component-level features
│   ├── anomaly_detector.py     # Model 1/2/3, hybrid score, DynamicOutlierDetector
│   ├── explainability.py       # deterministic explanation generator
│   └── evaluation.py           # metrics, GroupKFold/GroupShuffleSplit, threshold search
├── models/
│   ├── module_a_isolation_forest.pkl
│   └── module_a_config.json
├── outputs/
│   ├── module_a_predictions.csv
│   └── figures/  (7 EDA/evaluation charts)
├── run_pipeline.py   # reproduces every number in this README end-to-end
├── app.py            # Streamlit QA dashboard
└── requirements.txt
```

## 6. Usage

```bash
pip install -r requirements.txt
python run_pipeline.py        # re-runs EDA -> training -> evaluation -> artifacts
streamlit run app.py          # interactive dashboard (upload a CSV or use bundled demo)
```

Programmatic inference on a new batch:

```python
from src.anomaly_detector import DynamicOutlierDetector

detector = DynamicOutlierDetector.load(
    "models/module_a_isolation_forest.pkl", "models/module_a_config.json"
)
results = detector.predict(new_batch_df)  # needs Component_ID, Batch_ID, Value_0h_uA
```

`predict()` recomputes batch statistics from whatever batch(es) are in
the input, so it works correctly on manufacturing batches never seen
during training — there's no global model to be "out of distribution"
for, since every comparison is relative to the batch the component
arrived with.
