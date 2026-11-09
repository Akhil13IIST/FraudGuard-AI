
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

ablation_path = REPORTS / "elliptic_ablation_comparison.json"
local_path = REPORTS / "elliptic_local_only_metrics.json"

with ablation_path.open("r", encoding="utf-8") as file:
    ablation_results = json.load(file)

with local_path.open("r", encoding="utf-8") as file:
    local_result = json.load(file)

feature_groups = {
    "baseline": "All 165 features",
    "drift_ablation": "159 features after drift ablation",
    "regularized": "All 165 features with regularization",
    "local_only": "93 local transaction features",
}

experiments = []

for result in ablation_results:
    name = result["experiment"]

    experiments.append({
        "experiment": name,
        "feature_group": feature_groups[name],
        **result,
    })

experiments.append({
    "experiment": "local_only",
    "feature_group": local_result["feature_group"],
    **{
        key: value
        for key, value in local_result.items()
        if key not in ("model", "feature_group", "features")
    },
})

baseline = next(
    item for item in experiments
    if item["experiment"] == "baseline"
)

for item in experiments:
    item["ap_delta_vs_baseline"] = (
        item["average_precision"] - baseline["average_precision"]
    )
    item["roc_auc_delta_vs_baseline"] = (
        item["roc_auc"] - baseline["roc_auc"]
    )
    item["f1_delta_vs_baseline"] = (
        item["f1"] - baseline["f1"]
    )

comparison = {
    "title": "Elliptic Feature Group Comparison",
    "description": (
        "Comparison of XGBoost models trained using different "
        "feature configurations on the same temporal splits."
    ),
    "evaluation": {
        "training_time_steps": "1-30",
        "validation_time_steps": "31-38",
        "test_time_steps": "39-49",
        "test_metrics_used_for_comparison": True,
        "threshold_selection": (
            "Threshold selected using validation F1; "
            "test set evaluated afterward."
        ),
    },
    "experiments": experiments,
    "best_by_average_precision": max(
        experiments, key=lambda x: x["average_precision"]
    )["experiment"],
    "best_by_roc_auc": max(
        experiments, key=lambda x: x["roc_auc"]
    )["experiment"],
    "best_by_f1": max(
        experiments, key=lambda x: x["f1"]
    )["experiment"],
}

json_path = REPORTS / "elliptic_feature_group_comparison.json"

with json_path.open("w", encoding="utf-8") as file:
    json.dump(comparison, file, indent=4)

csv_path = REPORTS / "elliptic_feature_group_comparison.csv"

csv_columns = [
    "experiment",
    "feature_group",
    "feature_count",
    "threshold",
    "validation_f1",
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "ap_delta_vs_baseline",
    "roc_auc_delta_vs_baseline",
    "f1_delta_vs_baseline",
]

with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=csv_columns)
    writer.writeheader()

    for item in experiments:
        writer.writerow({
            column: item.get(column)
            for column in csv_columns
        })

print("\n========== FEATURE GROUP COMPARISON ==========")

for item in experiments:
    print(
        f"{item['experiment']:15} | "
        f"Features: {item['feature_count']:3} | "
        f"AP: {item['average_precision']:.5f} | "
        f"ROC-AUC: {item['roc_auc']:.5f} | "
        f"F1: {item['f1']:.5f}"
    )

print("\nBest by Average Precision:", comparison["best_by_average_precision"])
print("Best by ROC-AUC:", comparison["best_by_roc_auc"])
print("Best by F1:", comparison["best_by_f1"])

print("\nSaved:")
print(json_path)
print(csv_path)
