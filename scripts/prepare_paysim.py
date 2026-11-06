
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No PaySim CSV found in data/raw")

csv_path = csv_files[0]

print(f"Loading: {csv_path.name}")

df = pd.read_csv(csv_path)

# Basic validation
required = {
    "step", "type", "amount", "oldbalanceOrg",
    "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isFraud", "nameOrig", "nameDest"
}

missing_columns = required - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

assert df["isFraud"].isin([0, 1]).all()
assert df["amount"].ge(0).all()

# Feature engineering
df["log_amount"] = np.log1p(df["amount"])

df["orig_balance_error"] = (
    df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
)

df["dest_balance_error"] = (
    df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
)

df["orig_zero_balance"] = (
    df["oldbalanceOrg"] == 0
).astype("int8")

df["dest_zero_balance"] = (
    df["oldbalanceDest"] == 0
).astype("int8")

# Exclude identifiers and the rule-generated flag from model features.
# Keep isFraud as the target.
df = df.drop(columns=["nameOrig", "nameDest", "isFlaggedFraud"])

# Chronological split using transaction steps.
steps = np.sort(df["step"].unique())

train_end = steps[int(len(steps) * 0.70)]
valid_end = steps[int(len(steps) * 0.85)]

train = df[df["step"] <= train_end]
valid = df[
    (df["step"] > train_end) &
    (df["step"] <= valid_end)
]
test = df[df["step"] > valid_end]

# Save datasets
train.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
valid.to_parquet(PROCESSED_DIR / "validation.parquet", index=False)
test.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

print("\nPREPROCESSING COMPLETE")
print("-" * 45)

for name, split in [
    ("Train", train),
    ("Validation", valid),
    ("Test", test),
]:
    fraud_count = int(split["isFraud"].sum())
    fraud_rate = split["isFraud"].mean() * 100

    print(
        f"{name}: {len(split):,} rows | "
        f"Frauds: {fraud_count:,} | "
        f"Fraud rate: {fraud_rate:.5f}%"
    )

print(f"\nSaved to: {PROCESSED_DIR}")
