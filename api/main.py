
import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .database import (
    get_elliptic_predictions,
    get_investigation_cases,
    get_investigation_statistics,
    initialize_database,
    save_elliptic_prediction,
    update_investigation_case,
)

# --------------------------------------------------
# Paths and model loading
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgboost" / "elliptic_xgboost.joblib"
FEATURE_PATH = ROOT / "models" / "xgboost" / "feature_columns.joblib"
METRICS_PATH = ROOT / "reports" / "elliptic_xgboost_metrics.json"
COMPARISON_PATH = ROOT / "reports" / "elliptic_hybrid_metrics.json"
TEST_DATA_PATH = ROOT / "data" / "processed" / "elliptic" / "test.parquet"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

with open(METRICS_PATH, "r", encoding="utf-8") as file:
    metrics = json.load(file)

with open(COMPARISON_PATH, "r", encoding="utf-8") as file:
    comparison_metrics = json.load(file)

THRESHOLD = float(metrics["threshold"])


# Load held-out test transactions for the demonstration interface.
test_data = pd.read_parquet(TEST_DATA_PATH)

# Verify the dataset contains every required model feature.
missing_columns = set(feature_columns) - set(test_data.columns)

if missing_columns:
    raise RuntimeError(
        f"Test dataset is missing model features: {sorted(missing_columns)}"
    )


# --------------------------------------------------
# FastAPI application lifecycle
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Elliptic Fraud Detection API",
    description="Illicit transaction classification using Elliptic XGBoost",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Request schemas
# --------------------------------------------------

class TransactionFeatures(BaseModel):
    tx_id: str = Field(min_length=1)
    time_step: int = Field(ge=1, le=49)
    features: dict[str, float]


class InvestigationUpdate(BaseModel):
    status: str
    analyst_verdict: str
    notes: str = ""


# --------------------------------------------------
# Prediction helper
# --------------------------------------------------

def classify_transaction(tx_id, time_step, feature_values):
    if set(feature_values.keys()) != set(feature_columns):
        missing = sorted(set(feature_columns) - set(feature_values))
        extra = sorted(set(feature_values) - set(feature_columns))

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Incorrect feature schema",
                "missing_features": missing,
                "extra_features": extra,
            },
        )

    for name, value in feature_values.items():
        if not math.isfinite(value):
            raise HTTPException(
                status_code=400,
                detail=f"Feature {name} must be a finite number",
            )

    features = pd.DataFrame(
        [[feature_values[name] for name in feature_columns]],
        columns=feature_columns,
    )

    probability = float(model.predict_proba(features)[0, 1])
    predicted_illicit = probability >= THRESHOLD

    if probability >= 0.90:
        risk = "HIGH"
    elif probability >= 0.50:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    prediction_record = {
        "tx_id": str(tx_id),
        "time_step": int(time_step),
        "illicit_probability": probability,
        "predicted_illicit": predicted_illicit,
        "risk_level": risk,
    }

    prediction_id = save_elliptic_prediction(prediction_record)

    return {
        "prediction_id": prediction_id,
        "tx_id": str(tx_id),
        "time_step": int(time_step),
        "illicit_probability": round(probability, 6),
        "predicted_illicit": bool(predicted_illicit),
        "risk_level": risk,
        "decision_threshold": THRESHOLD,
        "model": "Elliptic XGBoost",
    }


# --------------------------------------------------
# General API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "application": "Elliptic Fraud Detection",
        "status": "running",
        "model": "Elliptic XGBoost",
        "features": len(feature_columns),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_count": len(feature_columns),
        "threshold": THRESHOLD,
    }


# --------------------------------------------------
# Model information endpoints
# --------------------------------------------------

@app.get("/model/metrics")
def model_metrics():
    return metrics


@app.get("/model/comparison")
def model_comparison():
    return {
        "selection_method": comparison_metrics["selection_method"],
        "selected_blend": comparison_metrics["selected_blend"],
        "individual_test_results": comparison_metrics[
            "individual_test_results"
        ],
        "test_results": comparison_metrics["test_results"],
        "interpretation": {
            "selected_model": "XGBoost",
            "xgboost_weight": comparison_metrics[
                "selected_blend"
            ]["xgb_weight"],
            "graphsage_weight": comparison_metrics[
                "selected_blend"
            ]["gnn_weight"],
            "conclusion": (
                "GraphSAGE did not improve validation average precision. "
                "XGBoost was selected as the standalone model."
            ),
            "note": (
                "This comparison uses the drift-ablation XGBoost artifact, "
                "not the production API's elliptic_xgboost.joblib artifact."
            ),
        },
    }


# --------------------------------------------------
# Transaction endpoints
# --------------------------------------------------

@app.get("/samples")
def get_samples(
    limit: int = Query(default=100, ge=1, le=1000),
):
    """Return transaction IDs and time steps, not their labels."""

    samples = test_data[["txId", "time_step"]].head(limit)

    return {
        "count": len(samples),
        "samples": [
            {
                "tx_id": str(row.txId),
                "time_step": int(row.time_step),
            }
            for row in samples.itertuples(index=False)
        ],
    }


@app.post("/predict")
def predict(transaction: TransactionFeatures):
    return classify_transaction(
        transaction.tx_id,
        transaction.time_step,
        transaction.features,
    )


@app.post("/predict/sample/{tx_id}")
def predict_sample(tx_id: str):
    """Predict an existing held-out transaction without exposing its label."""

    matching = test_data[test_data["txId"].astype(str) == tx_id]

    if matching.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found in the held-out test dataset",
        )

    row = matching.iloc[0]

    feature_values = {
        name: float(row[name])
        for name in feature_columns
    }

    return classify_transaction(
        tx_id,
        int(row["time_step"]),
        feature_values,
    )


@app.get("/transactions")
def transaction_history(
    limit: int = Query(default=100, ge=1, le=1000),
):
    return {
        "transactions": get_elliptic_predictions(limit)
    }


# --------------------------------------------------
# Analyst investigation endpoints
# --------------------------------------------------

@app.get("/investigations")
def list_investigations(
    limit: int = Query(default=100, ge=1, le=1000),
):
    return {
        "cases": get_investigation_cases(limit)
    }


@app.get("/investigations/statistics")
def investigation_statistics():
    return get_investigation_statistics()


@app.put("/investigations/{case_id}")
def update_investigation(
    case_id: int,
    update: InvestigationUpdate,
):
    try:
        updated = update_investigation_case(
            case_id=case_id,
            status=update.status,
            analyst_verdict=update.analyst_verdict,
            notes=update.notes,
        )

        if not updated:
            raise HTTPException(
                status_code=404,
                detail="Investigation case not found",
            )

        return {
            "message": "Investigation updated successfully",
            "case_id": case_id,
        }

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )
