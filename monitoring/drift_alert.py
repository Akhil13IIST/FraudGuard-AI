import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "elliptic_drift_report.json"
OUTPUT = ROOT / "reports" / "elliptic_drift_alert.json"

def find_drift_metric(obj):
    if isinstance(obj, dict):
        name = str(obj.get("metric_name", ""))
        value = obj.get("value")

        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            return value

        for child in obj.values():
            result = find_drift_metric(child)
            if result is not None:
                return result

    elif isinstance(obj, list):
        for child in obj:
            result = find_drift_metric(child)
            if result is not None:
                return result

    return None

if not REPORT.exists():
    raise FileNotFoundError(f"Run drift_monitor.py first: {REPORT}")

report = json.loads(REPORT.read_text(encoding="utf-8"))
metric = find_drift_metric(report)

if metric is None:
    raise ValueError("Could not find Evidently's DriftedColumnsCount metric.")

drifted = int(metric["count"])
share = float(metric["share"])

if share > 0.50:
    status = "CRITICAL"
    action = "Recommend model evaluation and retraining review."
elif share >= 0.30:
    status = "WARNING"
    action = "Investigate feature distribution changes."
else:
    status = "NORMAL"
    action = "Continue routine monitoring."

alert = {
    "monitor": "FraudGuard AI",
    "status": status,
    "drifted_features": drifted,
    "total_features": 165,
    "drift_share": round(share, 6),
    "drift_percentage": round(share * 100, 2),
    "thresholds": {
        "warning": 0.30,
        "critical": 0.50
    },
    "recommended_action": action,
    "automatic_retraining_executed": False
}

OUTPUT.write_text(json.dumps(alert, indent=4), encoding="utf-8")

print(json.dumps(alert, indent=4))
print(f"\nAlert saved to: {OUTPUT}")
