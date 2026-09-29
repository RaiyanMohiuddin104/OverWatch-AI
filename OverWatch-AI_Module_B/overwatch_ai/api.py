import json
import io
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# CREATE FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="OverWatch AI - Module B API",
    description="Time-series drift and degradation prediction API",
    version="1.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = "models/module_b_regressor.pkl"
CONFIG_PATH = "models/module_b_config.json"

COL_0H = "Value_0h_uA"
COL_24H = "Value_24h_uA"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = joblib.load(MODEL_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

threshold = config["safety_threshold"]


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "OverWatch AI - Module B",
        "status": "running",
        "model": config["model_name"]
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "UP",
        "module": "Module B",
        "model": config["model_name"]
    }


# --------------------------------------------------
# PREDICT FROM CSV
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:

        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        required_columns = [
            COL_0H,
            COL_24H
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Missing required columns",
                    "missing_columns": missing_columns
                }
            )

        # Model prediction
        predictions = model.predict(
            df[[COL_0H, COL_24H]].values
        )

        # Calculate degradation/drift rate
        drift_rate = (
            predictions - df[COL_24H].values
        ) / (168 - 24)

        # Decision
        decision = np.where(
            drift_rate > threshold,
            "EARLY REJECTION",
            "ACCEPTABLE"
        )

        # Create results
        results = df.copy()

        results["Predicted_168h"] = predictions
        results["Drift_Rate"] = drift_rate
        results["Safety_Threshold"] = threshold
        results["Decision"] = decision

        # Summary
        total = len(results)

        n_reject = int(
            (results["Decision"] == "EARLY REJECTION").sum()
        )

        n_accept = total - n_reject

        return {
            "status": "success",

            "model": {
                "name": config["model_name"],
                "test_R2": config["test_R2"],
                "safety_threshold": threshold
            },

            "summary": {
                "total_components": total,
                "acceptable": n_accept,
                "early_rejection": n_reject
            },

            "results": results.to_dict(
                orient="records"
            )
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# PREDICT SINGLE COMPONENT
# --------------------------------------------------

@app.post("/predict-single")
def predict_single(
    value_0h_uA: float,
    value_24h_uA: float
):

    try:

        predicted_168h = float(
            model.predict(
                np.array([
                    [value_0h_uA, value_24h_uA]
                ])
            )[0]
        )

        drift_rate = (
            predicted_168h - value_24h_uA
        ) / (168 - 24)

        decision = (
            "EARLY REJECTION"
            if drift_rate > threshold
            else "ACCEPTABLE"
        )

        return {
            "Value_0h_uA": value_0h_uA,
            "Value_24h_uA": value_24h_uA,
            "Predicted_168h": round(predicted_168h, 4),
            "Drift_Rate": round(drift_rate, 5),
            "Safety_Threshold": round(threshold, 5),
            "Decision": decision
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )