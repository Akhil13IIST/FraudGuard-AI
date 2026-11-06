
import mlflow.xgboost
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

mlflow.set_tracking_uri("sqlite:///mlflow.db")

run_id = "43e0694e7157410395d77fec4a045111"
model_uri = f"runs:/{run_id}/xgboost_model"

print("Loading model from MLflow...")
model = mlflow.xgboost.load_model(model_uri)

test = pd.read_parquet(
    ROOT / "data/processed/elliptic/test.parquet"
)

feature_columns = pd.read_pickle(
    ROOT / "models/xgboost/feature_columns.joblib"
)

sample = test[feature_columns].head(5)
probabilities = model.predict_proba(sample)[:, 1]

print("\nMLflow model loaded successfully!")
print("Predictions:", probabilities)
print("Number of predictions:", len(probabilities))
