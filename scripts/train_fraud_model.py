
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NEGATIVE_RATIO = 20

print("Loading processed datasets...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
valid = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

# Sample only legitimate training transactions.
fraud = train[train["isFraud"] == 1]
legitimate = train[train["isFraud"] == 0]

legitimate_sample = legitimate.sample(
    n=min(len(legitimate), len(fraud) * NEGATIVE_RATIO),
    random_state=RANDOM_STATE,
)

train_sample = (
    pd.concat([fraud, legitimate_sample])
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print(f"Training rows: {len(train_sample):,}")
print(f"Training fraud cases: {int(train_sample['isFraud'].sum()):,}")

# Prepare features.
TARGET = "isFraud"

def prepare_features(df):
    X = df.drop(columns=[TARGET]).copy()

    X = pd.get_dummies(
        X,
        columns=["type"],
        dtype="int8",
    )

    return X

X_train = prepare_features(train_sample)
X_valid = prepare_features(valid)
X_test = prepare_features(test)

# Ensure identical feature columns across splits.
X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_train = train_sample[TARGET]
y_valid = valid[TARGET]
y_test = test[TARGET]

negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())

scale_weight = negative_count / positive_count

print(f"Features: {len(X_train.columns)}")
print(f"Scale positive weight: {scale_weight:.2f}")

# Train model.
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_lambda=2,
    scale_pos_weight=scale_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    n_jobs=6,
    random_state=RANDOM_STATE,
)

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=50,
)

# Select threshold using validation data only.
valid_prob = model.predict_proba(X_valid)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_valid,
    valid_prob,
)

f1_values = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = int(np.argmax(f1_values))
best_threshold = float(thresholds[best_index])

print(f"\nSelected validation threshold: {best_threshold:.6f}")

# Final test evaluation.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

metrics = {
    "average_precision": float(
        average_precision_score(y_test, test_prob)
    ),
    "roc_auc": float(roc_auc_score(y_test, test_prob)),
    "precision": float(precision_score(y_test, test_pred, zero_division=0)),
    "recall": float(recall_score(y_test, test_pred, zero_division=0)),
    "f1": float(f1_score(y_test, test_pred, zero_division=0)),
    "threshold": best_threshold,
    "confusion_matrix": confusion_matrix(
        y_test, test_pred
    ).tolist(),
}

print("\nTEST RESULTS")
print("=" * 45)

for key, value in metrics.items():
    print(f"{key}: {value}")

joblib.dump(model, MODEL_DIR / "xgboost_fraudguard.joblib")

with open(REPORT_DIR / "xgboost_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

joblib.dump(
    list(X_train.columns),
    MODEL_DIR / "feature_columns.joblib",
)

print("\nModel and evaluation results saved successfully.")
