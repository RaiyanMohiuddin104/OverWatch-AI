from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from src import anomaly_detector as ad
from src import explainability as ex


app = FastAPI(
    title="OverWatch AI - Module A API",
    description="Batch-aware dynamic anomaly detection API",
    version="1.0.0"
)


# Allow requests from the Spring Boot backend / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = "models/module_a_isolation_forest.pkl"
CONFIG_PATH = "models/module_a_config.json"


# Load the trained detector once when the API starts
detector = ad.DynamicOutlierDetector.load(
    MODEL_PATH,
    CONFIG_PATH
)


@app.get("/")
def root():
    return {
        "service": "OverWatch AI - Module A",
        "status": "running",
        "description": "Dynamic batch-aware anomaly detection API"
    }


@app.get("/health")
def health():
    return {
        "status": "UP",
        "module": "Module A"
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file."
        )

    try:
        contents = await file.read()

        df = pd.read_csv(io.BytesIO(contents))

        required_columns = [
            "Component_ID",
            "Batch_ID",
            "Value_0h_uA"
        ]

        missing_columns = [
            column for column in required_columns
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

        # Run Module A
        results = detector.predict(df)

        # Add explanations
        results = ex.add_explanations(results)

        # Preserve absolute-limit information if supplied
        if "Absolute_Limit_50uA_Pass" in df.columns:

            results = results.merge(
                df[
                    [
                        "Component_ID",
                        "Absolute_Limit_50uA_Pass"
                    ]
                ],
                on="Component_ID",
                how="left"
            )

        else:

            results["Absolute_Limit_50uA_Pass"] = (
                results["Value_0h_uA"] <= 50.0
            )

        # Summary
        total = len(results)

        normal = int(
            (results["Risk_Level"] == "Normal").sum()
        )

        warning = int(
            (results["Risk_Level"] == "Warning").sum()
        )

        anomaly = int(
            (results["Risk_Level"] == "Anomaly").sum()
        )

        rescued = int(
            (
                (results["Risk_Level"] == "Anomaly")
                & (results["Absolute_Limit_50uA_Pass"])
            ).sum()
        )

        return {
            "status": "success",

            "summary": {
                "total_components": total,
                "normal": normal,
                "warning": warning,
                "anomaly": anomaly,
                "anomalies_passing_absolute_limit": rescued
            },

            "results": results.to_dict(orient="records")
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )