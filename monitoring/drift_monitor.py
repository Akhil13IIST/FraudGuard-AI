import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "elliptic"
REPORTS = ROOT / "reports"

TRAIN_FILE = DATA / "train.parquet"
VALIDATION_FILE = DATA / "validation.parquet"

REPORTS.mkdir(parents=True, exist_ok=True)

print("Loading Elliptic datasets...")
train = pd.read_parquet(TRAIN_FILE)
validation = pd.read_parquet(VALIDATION_FILE)

# Only use the 165 numeric transaction features.
feature_columns = sorted(
    [c for c in train.columns if c.startswith("feature_")],
    key=lambda c: int(c.split("_")[1])
)

if not feature_columns:
    raise ValueError("No feature_* columns found in training data.")

missing = set(feature_columns) - set(validation.columns)
if missing:
    raise ValueError(f"Validation data is missing features: {sorted(missing)}")

reference = train[feature_columns].copy()
current = validation[feature_columns].copy()

# Ensure numeric values and replace infinities with missing values.
reference = reference.apply(pd.to_numeric, errors="coerce")
current = current.apply(pd.to_numeric, errors="coerce")

reference = reference.replace([np.inf, -np.inf], np.nan)
current = current.replace([np.inf, -np.inf], np.nan)

# Remove columns that contain no usable values in either dataset.
usable = [
    col for col in feature_columns
    if reference[col].notna().any() and current[col].notna().any()
]

reference = reference[usable]
current = current[usable]

if reference.empty or current.empty:
    raise ValueError("No usable features remain for drift analysis.")

print(f"Reference rows: {len(reference):,}")
print(f"Current rows:   {len(current):,}")
print(f"Features:       {len(usable)}")
print("Running Evidently data drift analysis...")

report = Report([DataDriftPreset()])
snapshot = report.run(
    reference_data=reference,
    current_data=current
)

html_path = REPORTS / "elliptic_drift_report.html"
json_path = REPORTS / "elliptic_drift_report.json"
summary_path = REPORTS / "elliptic_drift_summary.json"

snapshot.save_html(str(html_path))
snapshot.save_json(str(json_path))

summary = {
    "monitor": "FraudGuard AI - Elliptic Data Drift",
    "generated_at_utc": datetime.now(UTC).isoformat(),
    "reference_dataset": str(TRAIN_FILE.relative_to(ROOT)),
    "current_dataset": str(VALIDATION_FILE.relative_to(ROOT)),
    "reference_rows": len(reference),
    "current_rows": len(current),
    "features_analyzed": len(usable),
    "excluded_columns": [
        "txId", "node_id", "time_step", "class", "label"
    ],
    "report_html": str(html_path.relative_to(ROOT)),
    "report_json": str(json_path.relative_to(ROOT)),
    "interpretation": (
        "This report compares feature distributions. "
        "Detected drift is a monitoring signal, not proof of model "
        "performance degradation or a reason to retrain automatically."
    )
}

summary_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

print("\nDrift monitoring completed.")
print(f"HTML report: {html_path}")
print(f"JSON report: {json_path}")
print(f"Summary:     {summary_path}")
