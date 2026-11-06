
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/processed/elliptic")

df = pd.read_parquet(BASE / "transactions.parquet")

features = [
    "feature_100",
    "feature_101",
    "feature_103",
    "feature_136",
    "feature_137",
    "feature_139",
]

partitions = {
    "TRAIN": (1, 30),
    "VALIDATION": (31, 38),
    "TEST": (39, 49),
}

print("\n========== EXTREME FEATURE INVESTIGATION ==========")

for feature in features:

    print(f"\n\nFEATURE: {feature}")

    for name, (start, end) in partitions.items():

        subset = df[
            df["time_step"].between(start, end)
        ][feature]

        print(f"\n{name}")

        print("Mean:", round(subset.mean(), 5))
        print("Std:", round(subset.std(), 5))
        print("Min:", round(subset.min(), 5))
        print("Median:", round(subset.median(), 5))
        print("Max:", round(subset.max(), 5))
        print(
            "Zero percentage:",
            round((subset == 0).mean() * 100, 2),
            "%"
        )

        print("Unique values:", subset.nunique())

print("\n========== INVESTIGATION COMPLETE ==========")
