
from pathlib import Path

import pandas as pd

BASE = Path("data/processed/elliptic")

df = pd.read_parquet(BASE / "transactions.parquet")

features = [
    c for c in df.columns
    if c.startswith("feature_")
]

partitions = {
    "Train": (1, 30),
    "Validation": (31, 38),
    "Test": (39, 49),
}

print("\n========== TEMPORAL DISTRIBUTION ANALYSIS ==========")

for name, (start, end) in partitions.items():

    subset = df[df["time_step"].between(start, end)]
    labelled = subset[subset["label"].notna()]

    positive = (labelled["label"] == 1).sum()
    negative = (labelled["label"] == 0).sum()

    print(f"\n{name.upper()}")
    print("Time steps:", start, "-", end)
    print("Total nodes:", len(subset))
    print("Labelled nodes:", len(labelled))
    print("Illicit:", positive)
    print("Licit:", negative)
    print("Illicit prevalence:", round(
        positive / len(labelled), 5
    ))

    print("Feature mean:", round(
        subset[features].mean().mean(), 5
    ))

    print("Feature std:", round(
        subset[features].std().mean(), 5
    ))

print("\n========== PER-TIME-STEP CLASS DISTRIBUTION ==========")

summary = (
    df[df["label"].notna()]
    .groupby("time_step")["label"]
    .agg(
        transactions="count",
        illicit=lambda x: (x == 1).sum(),
        licit=lambda x: (x == 0).sum(),
        prevalence=lambda x: (x == 1).mean()
    )
)

print(summary.to_string())

print("\n========== ANALYSIS COMPLETE ==========")
