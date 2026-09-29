# OverWatch AI

**AI-Assisted Anomaly Detection and Predictive Burn-In Screening**

OverWatch AI is an intelligent decision-support system for electronic component Burn-In screening. It is designed to identify abnormal component behaviour, predict early degradation, and provide engineers with an interpretable reliability assessment beyond conventional fixed pass/fail limits.

The system combines batch-aware anomaly detection with degradation prediction and presents the results through an interactive web dashboard.

---

## Problem

Environmental Stress Screening (ESS) and Burn-In testing are widely used in high-reliability electronics to identify components that may fail prematurely.

Traditional screening often relies on fixed acceptance limits. However, a component can remain within an absolute limit while still behaving abnormally compared with other components in the same manufacturing batch.

Similarly, early changes in electrical parameters may indicate degradation even when the component has not yet crossed a failure threshold.

OverWatch AI adds a data-driven analysis layer to the existing Burn-In process to help identify these cases earlier.

---

## Proposed Solution

OverWatch AI analyses Burn-In measurements using two independent AI modules.

### Module A — Batch-Aware Anomaly Detection

Module A compares each component with the behaviour of other components in its batch.

It uses a hybrid anomaly detection approach based on:

- Isolation Forest
- Robust Z-score
- Interquartile Range (IQR)
- Batch-relative statistical features

The outputs are combined into a dynamic anomaly score and component classification.

This allows the system to identify components that may appear acceptable under an absolute limit but behave unusually relative to their batch.

### Module B — Degradation Prediction

Module B analyses early Burn-In measurements to estimate how a component may behave later in the test.

The current model uses:

- 0-hour measurement
- 24-hour measurement

to predict the expected value at **168 hours**.

The predicted degradation rate is then compared with a safety threshold derived from historical behaviour. Components showing excessive predicted degradation can be flagged for early rejection or further inspection.

### Risk Engine

The outputs of Module A and Module B are evaluated by a rule-based Risk Engine.

Each component receives a final classification:

- **Low Risk**
- **Medium Risk**
- **High Risk**

The Risk Engine provides a single reliability assessment while preserving the individual results from both AI modules.

---

## System Architecture

```text
                         Burn-In Dataset
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
        Module A                         Module B
   Anomaly Detection              Degradation Prediction
                 |                           |
                 +-------------+-------------+
                               |
                               v
                          Risk Engine
                               |
                               v
                    Reliability Assessment
                               |
                               v
                       React Dashboard
```

The React frontend communicates with the Spring Boot backend. Spring Boot coordinates requests to the two Python FastAPI services and combines their outputs through the Risk Engine.

---

## Key Features

### Batch-Aware Screening

Components are evaluated relative to their own batch rather than relying only on a global fixed threshold.

### Early Degradation Prediction

Early measurements from 0h and 24h are used to estimate component behaviour at 168h.

### Unified Risk Assessment

Anomaly detection and degradation prediction results are combined to determine the final component risk level.

### OverWatch Explain

Component-level results include information that helps engineers understand why a component was flagged, including anomaly behaviour, degradation trend, thresholds and model outputs.

### Batch Health Intelligence

The dashboard aggregates component results at batch level to highlight batches containing higher concentrations of suspicious or high-risk components.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React, JavaScript, CSS, Vite |
| Backend | Java, Spring Boot |
| AI Services | Python, FastAPI |
| Module A | Isolation Forest, Robust Z-score, IQR |
| Module B | Regression-based degradation prediction |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Recharts |
| API Communication | REST / JSON |

---

## Repository Structure

```text
OverWatch-AI/
|
├── OverWatch-AI_Module_A/
│   └── overwatch_ai/
│       ├── models/
│       ├── src/
│       ├── api.py
│       ├── app.py
│       ├── run_pipeline.py
│       └── requirements.txt
│
├── OverWatch-AI_Module_B/
│   └── overwatch_ai/
│       ├── models/
│       ├── api.py
│       ├── app.py
│       ├── run_pipeline.py
│       └── requirements.txt
│
├── overwatch-ai-backend/
│   ├── src/
│   └── pom.xml
│
├── overwatch-ai-frontend/
│   └── ...
│
├── .gitignore
└── README.md
```

---

## Processing Workflow

1. The user uploads Burn-In component data through the dashboard.
2. The Spring Boot backend receives the dataset.
3. The dataset is sent to Module A and Module B.
4. Module A performs batch-aware anomaly detection.
5. Module B predicts long-term degradation from early measurements.
6. The Risk Engine combines the outputs of both modules.
7. Components are classified as Low, Medium or High Risk.
8. Component and batch-level results are displayed on the dashboard.

---

## Input Data

The prototype works with Burn-In measurements containing fields such as:

```text
Component_ID
Batch_ID
Value_0h_uA
Value_24h_uA
Value_96h_uA
Value_168h_uA
```

Module A primarily analyses component behaviour relative to its batch, while Module B uses the early 0h and 24h readings for prediction.

Later measurements can be retained for evaluation and visualization.

---

## Running the Project

The application consists of four services.

### 1. Module A

Navigate to the Module A project directory and install its dependencies:

```bash
pip install -r requirements.txt
```

Start the FastAPI service:

```bash
uvicorn api:app --reload --port 8001
```

Module A runs on:

```text
http://localhost:8001
```

### 2. Module B

Navigate to the Module B project directory:

```bash
pip install -r requirements.txt
```

Start the service:

```bash
uvicorn api:app --reload --port 8000
```

Module B runs on:

```text
http://localhost:8000
```

### 3. Spring Boot Backend

Navigate to:

```text
overwatch-ai-backend
```

Run the Spring Boot application using Maven:

```bash
mvnw.cmd spring-boot:run
```

The backend runs on:

```text
http://localhost:8080
```

### 4. React Frontend

Navigate to:

```text
overwatch-ai-frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

Open the local URL displayed by Vite in the terminal.

---

## Example Analysis

For every component, OverWatch AI can provide information including:

```text
Component ID
Batch ID
Module A Classification
Dynamic Anomaly Score
Module B Decision
Predicted 168h Value
Predicted Drift Rate
Safety Threshold
Final Risk
```

This allows the dashboard to present both the final reliability assessment and the factors contributing to it.

---

## Current Prototype

The current prototype supports:

- CSV-based Burn-In dataset upload
- Batch-aware anomaly detection
- Early degradation prediction
- Low / Medium / High risk classification
- Component-level analysis
- Batch-level analysis
- Burn-In trend visualization
- Explainable component results
- Light and dark dashboard themes

The system is intended as a **decision-support layer for Burn-In screening**, not as a replacement for established ESS/Burn-In procedures or engineering qualification standards.

---

## Future Scope

Future development can include:

- Integration with real Burn-In test equipment
- Live telemetry ingestion
- Support for additional electrical parameters
- Model retraining using larger industry datasets
- Historical component tracking
- Automated reliability reports
- Deployment within production quality-control environments

---

## Project Information

**Project:** OverWatch AI  
**Problem Statement ID:** SIH26170  
**Problem Statement:** AI-Driven Anomaly Detection in Component Burn-In & Screening  
**Theme:** Smart Automation  
**Category:** Software  

Developed for **Smart India Hackathon 2026**.
