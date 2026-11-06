
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

model = joblib.load(ROOT / "models" / "xgboost_fraudguard.joblib")
features = joblib.load(ROOT / "models" / "feature_columns.joblib")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTOP FEATURE IMPORTANCE")
print("=" * 45)
print(importance.to_string(index=False))

print("\nTEMPORAL SPLIT AUDIT")
print("=" * 45)

for name in ["train", "validation", "test"]:
    df = pd.read_parquet(
        ROOT / "data" / "processed" / f"{name}.parquet"
    )

    print(
        f"{name.upper():12} "
        f"steps={df.step.min()}–{df.step.max()} | "
        f"fraud={df.isFraud.mean():.5%} | "
        f"rows={len(df):,}"
    )

print("\nFRAUD VS LEGITIMATE FEATURE MEANS")
print("=" * 45)

test = pd.read_parquet(ROOT / "data" / "processed" / "test.parquet")

cols = [
    "amount",
    "log_amount",
    "orig_balance_error",
    "dest_balance_error",
    "orig_zero_balance",
    "dest_zero_balance",
]

print(test.groupby("isFraud")[cols].mean().T)
