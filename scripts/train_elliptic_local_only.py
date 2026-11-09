
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "elliptic"
MODEL_DIR = ROOT / "models" / "xgboost" / "local_only"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLDS = np.arange(0.10, 0.91, 0.01)

print("Loading temporal splits...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

# Local transaction features: feature_1 through feature_93.
FEATURE_COLUMNS = [
    f"feature_{i}" for i in range(1, 94)
]

for name, frame in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"{name} is missing features: {sorted(missing)}"
        )

X_train = train[FEATURE_COLUMNS]
y_train = train["label"].astype(int)

X_val = validation[FEATURE_COLUMNS]
y_val = validation["label"].astype(int)

X_test = test[FEATURE_COLUMNS]
y_test = test["label"].astype(int)

scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
print("Local features:", len(FEATURE_COLUMNS))
print("Positive class weight:", scale_pos_weight)

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
    n_jobs=4,
)

print("\nTraining local-only XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# Select threshold using validation F1.
val_prob = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_f1 = -1.0

for threshold in THRESHOLDS:
    val_pred = (val_prob >= threshold).astype(int)

    score = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

# Evaluate the held-out test set once.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

metrics = {
    "model": "Elliptic Local-Only XGBoost",
    "feature_group": "Local transaction features",
    "feature_count": len(FEATURE_COLUMNS),
    "features": FEATURE_COLUMNS,
    "threshold": best_threshold,
    "validation_f1": float(best_f1),
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
    ).tolist(),
}

joblib.dump(
    model,
    MODEL_DIR / "elliptic_local_only.joblib",
)

joblib.dump(
    FEATURE_COLUMNS,
    MODEL_DIR / "elliptic_local_only_features.joblib",
)

with open(
    REPORT_DIR / "elliptic_local_only_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

print("\n========== LOCAL-ONLY RESULTS ==========")
print(json.dumps(metrics, indent=4))
print("\nModel saved:", MODEL_DIR)
print("Report saved:", REPORT_DIR / "elliptic_local_only_metrics.json")
