
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/xgboost/elliptic_xgboost.joblib"
FEATURE_PATH = ROOT / "models/xgboost/feature_columns.joblib"
METRICS_PATH = ROOT / "reports/elliptic_xgboost_metrics.json"

# Load existing artifacts
model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

with open(METRICS_PATH, "r") as f:
    metrics = json.load(f)

# Configure MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FraudGuard-Elliptic-XGBoost")

with mlflow.start_run(run_name="Elliptic-XGBoost-Complete-Tracking"):

    # Log model hyperparameters
    mlflow.log_params({
        "model": "XGBoost",
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": 8.107989,
        "tree_method": "hist",
        "random_state": 42,
        "feature_count": len(feature_columns)
    })

    # Log evaluation metrics
    for key in [
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "threshold"
    ]:
        if key in metrics:
            mlflow.log_metric(key, float(metrics[key]))

    # Log confusion matrix as an artifact
    mlflow.log_dict(
        {"confusion_matrix": metrics["confusion_matrix"]},
        "confusion_matrix.json"
    )

    # Log model and supporting artifacts
    mlflow.xgboost.log_model(
        model,
        name="xgboost_model"
    )

    mlflow.log_artifact(str(FEATURE_PATH))
    mlflow.log_artifact(str(METRICS_PATH))

print("MLflow experiment tracking completed successfully.")
