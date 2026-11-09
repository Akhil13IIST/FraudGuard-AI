from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FraudGuard-Elliptic-XGBoost")

model = joblib.load(
    ROOT / "models/xgboost/elliptic_xgboost.joblib"
)

with mlflow.start_run(run_name="Elliptic-XGBoost-Model-Artifact"):
    mlflow.xgboost.log_model(
        model,
        name="xgboost_model"
    )

    mlflow.log_artifact(
        str(ROOT / "models/xgboost/feature_columns.joblib")
    )

    mlflow.log_artifact(
        str(ROOT / "reports/elliptic_xgboost_metrics.json")
    )

print("Existing XGBoost model successfully logged to MLflow.")
