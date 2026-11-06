import mlflow
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_DIR = Path("data/processed/elliptic")
MODEL_DIR = Path("models/xgboost")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading temporal splits...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

FEATURE_COLUMNS = [
    col for col in train.columns
    if col.startswith("feature_")
]

X_train = train[FEATURE_COLUMNS]
y_train = train["label"].astype(int)

X_val = validation[FEATURE_COLUMNS]
y_val = validation["label"].astype(int)

X_test = test[FEATURE_COLUMNS]
y_test = test["label"].astype(int)

scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Features:", len(FEATURE_COLUMNS))
print("Scale positive weight:", scale_pos_weight)

model = xgb.XGBClassifier(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    tree_method="hist",
    random_state=42,
    n_jobs=4
)
# --------------------------------------------------
# MLFLOW EXPERIMENT TRACKING
# --------------------------------------------------


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("FraudGuard-Elliptic-XGBoost")

mlflow.start_run(run_name="Elliptic-XGBoost-Temporal")

mlflow.log_params({
    "model": "XGBoost",
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": float(scale_pos_weight),
    "tree_method": "hist",
    "random_state": 42,
    "features": len(FEATURE_COLUMNS),
    "training_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test)
})
print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50
)

# --------------------------------------------------
# Validation threshold selection
# --------------------------------------------------

val_prob = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)

best_threshold = 0.5
best_f1 = -1

for threshold in thresholds:
    predictions = (val_prob >= threshold).astype(int)

    score = f1_score(
        y_val,
        predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)
val_pred = (val_prob >= best_threshold).astype(int)

mlflow.log_metrics({
    "validation_average_precision": float(
        average_precision_score(y_val, val_prob)
    ),
    "validation_roc_auc": float(
        roc_auc_score(y_val, val_prob)
    ),
    "validation_precision": float(
        precision_score(y_val, val_pred, zero_division=0)
    ),
    "validation_recall": float(
        recall_score(y_val, val_pred, zero_division=0)
    ),
    "validation_f1": float(best_f1),
    "selected_threshold": float(best_threshold)
})

# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

metrics = {
    "model": "Elliptic XGBoost",
    "threshold": best_threshold,
    "average_precision": float(
        average_precision_score(y_test, test_prob)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_prob)
    ),
    "precision": float(
        precision_score(y_test, test_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_pred, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_pred, zero_division=0)
    ),
    "confusion_matrix": confusion_matrix(
        y_test, test_pred
    ).tolist()
}

joblib.dump(
    model,
    MODEL_DIR / "elliptic_xgboost.joblib"
)

joblib.dump(
    FEATURE_COLUMNS,
    MODEL_DIR / "feature_columns.joblib"
)

with open(REPORT_DIR / "elliptic_xgboost_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
# --------------------------------------------------
# LOG FINAL RESULTS AND ARTIFACTS
# --------------------------------------------------

mlflow.log_metrics({
    "test_average_precision": metrics["average_precision"],
    "test_roc_auc": metrics["roc_auc"],
    "test_precision": metrics["precision"],
    "test_recall": metrics["recall"],
    "test_f1": metrics["f1"]
})

mlflow.log_artifact(
    str(MODEL_DIR / "feature_columns.joblib")
)

mlflow.log_artifact(
    str(REPORT_DIR / "elliptic_xgboost_metrics.json")
)

mlflow.xgboost.log_model(
    model,
    name="xgboost_model"
)

mlflow.end_run()
print("\n========== TEST RESULTS ==========")
print(json.dumps(metrics, indent=4))

print("\nClassification report:")
print(classification_report(y_test, test_pred, zero_division=0))

print("\nModel saved successfully.")
