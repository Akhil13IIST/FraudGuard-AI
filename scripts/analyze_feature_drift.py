
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

BASE = Path("data/processed/elliptic")

df = pd.read_parquet(BASE / "transactions.parquet")

features = [
    c for c in df.columns
    if c.startswith("feature_")
]

partitions = {
    "train": df[df["time_step"].between(1, 30)],
    "validation": df[df["time_step"].between(31, 38)],
    "test": df[df["time_step"].between(39, 49)],
}

comparisons = [
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
]

results = []

for feature in features:

    row = {"feature": feature}

    for left_name, right_name in comparisons:

        left = partitions[left_name][feature].dropna()
        right = partitions[right_name][feature].dropna()

        statistic, p_value = ks_2samp(
            left,
            right
        )

        row[f"{left_name}_vs_{right_name}"] = statistic

    results.append(row)

drift = pd.DataFrame(results)

print("\n========== FEATURE DRIFT ANALYSIS ==========")

for comparison in [
    "train_vs_validation",
    "train_vs_test",
    "validation_vs_test",
]:

    print(f"\nTOP 15 FEATURES: {comparison}")

    print(
        drift.sort_values(
            comparison,
            ascending=False
        )[["feature", comparison]]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\nFeatures with KS statistic > 0.20:",
        (drift[comparison] > 0.20).sum()
    )

# Save full results
output = Path("reports")
output.mkdir(exist_ok=True)

drift.to_csv(
    output / "elliptic_feature_drift.csv",
    index=False
)

print("\nSaved: reports/elliptic_feature_drift.csv")
