
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

REPORTS.mkdir(exist_ok=True)

print("Loading model and test data...")

model = joblib.load(MODELS / "xgboost_fraudguard.joblib")
feature_columns = joblib.load(MODELS / "feature_columns.joblib")

test = pd.read_parquet(DATA / "test.parquet")

def prepare_features(df):
    X = df.drop(columns=["isFraud"]).copy()

    X = pd.get_dummies(
        X,
        columns=["type"],
        dtype="int8",
    )

    return X.reindex(
        columns=feature_columns,
        fill_value=0,
    )

# Sample to control memory consumption.
sample = test.sample(
    n=min(1000, len(test)),
    random_state=42,
)

X = prepare_features(sample)

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Global feature importance.
shap.plots.bar(
    shap_values,
    max_display=15,
    show=False,
)

plt.tight_layout()
plt.savefig(
    REPORTS / "shap_global_importance.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# Beeswarm plot.
shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False,
)

plt.tight_layout()
plt.savefig(
    REPORTS / "shap_summary.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close()

# Explain one transaction.
fraud_positions = [
    i for i, value in enumerate(sample["isFraud"].values)
    if value == 1
]

if fraud_positions:
    position = fraud_positions[0]

    shap.plots.waterfall(
        shap_values[position],
        max_display=12,
        show=False,
    )

    plt.tight_layout()
    plt.savefig(
        REPORTS / "shap_transaction_explanation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

print("\nSHAP EXPLANATIONS GENERATED")

for path in REPORTS.glob("shap_*.png"):
    print(path.name)
