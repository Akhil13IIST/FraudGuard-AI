
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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/processed/elliptic")
MODEL_DIR = Path("models/xgboost/ablation")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_FEATURES = [
    "feature_100",
    "feature_101",
    "feature_103",
    "feature_136",
    "feature_137",
    "feature_139",
]

THRESHOLDS = np.arange(0.10, 0.91, 0.01)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("Loading temporal splits...")

train = pd.read_parquet(DATA_DIR / "train.parquet")
validation = pd.read_parquet(DATA_DIR / "validation.parquet")
test = pd.read_parquet(DATA_DIR / "test.parquet")

ALL_FEATURES = [
    col for col in train.columns
    if col.startswith("feature_")
]

y_train = train["label"].astype(int)
y_val = validation["label"].astype(int)
y_test = test["label"].astype(int)

scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))
print("Original features:", len(ALL_FEATURES))
print("Positive weight:", scale_pos_weight)

# --------------------------------------------------
# Experiment configurations
# --------------------------------------------------

experiments = {
    "baseline": {
        "features": ALL_FEATURES,
        "params": {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 4,
        },
    },

    "drift_ablation": {
        "features": [
            f for f in ALL_FEATURES
            if f not in DRIFT_FEATURES
        ],
        "params": {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 4,
        },
    },

    "regularized": {
        "features": ALL_FEATURES,
        "params": {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "reg_alpha": 0.1,
            "gamma": 0.1,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 4,
        },
    },
}

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

all_results = []

for experiment_name, config in experiments.items():

    print("\n" + "=" * 55)
    print("EXPERIMENT:", experiment_name.upper())
    print("=" * 55)

    features = config["features"]

    X_train = train[features]
    X_val = validation[features]
    X_test = test[features]

    print("Feature count:", len(features))

    model = xgb.XGBClassifier(**config["params"])

    print("Training...")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Validation threshold selection
    val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold = 0.5
    best_f1 = -1

    for threshold in THRESHOLDS:

        val_pred = (
            val_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0,
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    # Final test evaluation
    test_prob = model.predict_proba(X_test)[:, 1]

    test_pred = (
        test_prob >= best_threshold
    ).astype(int)

    metrics = {
        "experiment": experiment_name,
        "feature_count": len(features),
        "threshold": best_threshold,
        "validation_f1": float(best_f1),
        "average_precision": float(
            average_precision_score(y_test, test_prob)
        ),
        "roc_auc": float(
            roc_auc_score(y_test, test_prob)
        ),
        "precision": float(
            precision_score(
                y_test, test_pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, test_pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, test_pred, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test, test_pred
        ).tolist(),
    }

    all_results.append(metrics)

    # Save each experiment separately
    joblib.dump(
        model,
        MODEL_DIR / f"{experiment_name}.joblib",
    )

    joblib.dump(
        features,
        MODEL_DIR / f"{experiment_name}_features.joblib",
    )

    print(json.dumps(metrics, indent=4))

# --------------------------------------------------
# Save comparison
# --------------------------------------------------

results_df = pd.DataFrame(all_results)

results_df.to_csv(
    REPORT_DIR / "elliptic_ablation_comparison.csv",
    index=False,
)

with open(
    REPORT_DIR / "elliptic_ablation_comparison.json",
    "w",
) as f:
    json.dump(all_results, f, indent=4)

print("\n========== FINAL COMPARISON ==========")

print(
    results_df[
        [
            "experiment",
            "feature_count",
            "average_precision",
            "roc_auc",
            "precision",
            "recall",
            "f1",
            "threshold",
        ]
    ].to_string(index=False)
)

print("\nAll experiments completed.")
print("Results: reports/elliptic_ablation_comparison.csv")
