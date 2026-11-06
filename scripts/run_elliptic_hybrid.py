
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROCESSED = ROOT / "data" / "processed" / "elliptic"
REPORTS = ROOT / "reports"
XGB_DIR = ROOT / "models" / "xgboost" / "ablation"
GNN_PATH = ROOT / "models" / "gnn" / "elliptic_graphsage.pt"

REPORTS.mkdir(parents=True, exist_ok=True)


class GraphSAGE(nn.Module):
    def __init__(self, input_channels, hidden_channels, dropout):
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = nn.functional.dropout(
            x, p=self.dropout, training=self.training
        )
        x = self.conv2(x, edge_index).relu()
        return self.classifier(x).squeeze(-1)


def load_partition(name):
    df = pd.read_parquet(PROCESSED / f"{name}.parquet")
    return df.reset_index(drop=True)


def make_graph(df, edges, all_nodes=None):
    import numpy as np
    import torch

    # Keep a consistent local node ordering
    df = df.sort_values("node_id").reset_index(drop=True)

    node_ids = df["node_id"].to_numpy()
    local_lookup = {
        int(node_id): idx
        for idx, node_id in enumerate(node_ids)
    }

    # Select edges whose source AND destination exist
    # in this partition
    partition_edges = edges[
        edges["txId1"].isin(local_lookup)
        & edges["txId2"].isin(local_lookup)
    ]

    # Convert global transaction IDs to local indices
    src = partition_edges["txId1"].map(local_lookup).to_numpy(
        dtype=np.int64
    )
    dst = partition_edges["txId2"].map(local_lookup).to_numpy(
        dtype=np.int64
    )

    # Make graph undirected, matching training
    edge_array = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ])

    edge_index = torch.from_numpy(edge_array).long()

    # Feature matrix
    feature_cols = [
        f"feature_{i}" for i in range(1, 166)
    ]

    x = torch.tensor(
        df[feature_cols].to_numpy(dtype=np.float32),
        dtype=torch.float32
    )

    # Validate graph indices
    if edge_index.numel() > 0:
        assert edge_index.min().item() >= 0
        assert edge_index.max().item() < x.size(0)

    return x, edge_index, df


def get_gnn_probabilities(df, edges, checkpoint):

    # Build graph
    x, edge_index, graph_df = make_graph(
        df, edges, all_nodes=203769
    )

    # Load model on the selected device
    model = GraphSAGE(
        checkpoint["input_channels"],
        checkpoint["hidden_channels"],
        checkpoint["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Move BOTH graph tensors to GPU (or CPU)
    x = x.to(DEVICE)
    edge_index = edge_index.to(DEVICE)

    # Generate probabilities
    with torch.no_grad():
        logits = model(x, edge_index)
        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()

def evaluate(name, y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "model": name,
        "threshold": float(threshold),
        "average_precision": float(
            average_precision_score(y, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(precision_score(
            y, predictions, zero_division=0
        )),
        "recall": float(recall_score(
            y, predictions, zero_division=0
        )),
        "f1": float(f1_score(
            y, predictions, zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y, predictions
        ).tolist(),
    }


def best_f1_threshold(y, probabilities):
    precision, recall, thresholds = precision_recall_curve(
        y, probabilities
    )

    f1 = (
        2 * precision[:-1] * recall[:-1] /
        (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])


def main():
    print(f"Device: {DEVICE}")

    # Load saved XGBoost model and its exact feature list.
    xgb_model = joblib.load(XGB_DIR / "drift_ablation.joblib")
    xgb_features = joblib.load(
        XGB_DIR / "drift_ablation_features.joblib"
    )

    # Load GraphSAGE checkpoint.
    checkpoint = torch.load(
        GNN_PATH, map_location=DEVICE, weights_only=False
    )

    # Load temporal partitions and full edge list.
    train = load_partition("train")
    val = load_partition("validation")
    test = load_partition("test")
    edges = pd.read_parquet(PROCESSED / "edges.parquet")

    # Labels must be known for evaluation.
    val = val.dropna(subset=["label"]).reset_index(drop=True)
    test = test.dropna(subset=["label"]).reset_index(drop=True)

    y_val = val["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    print("\nGenerating XGBoost probabilities...")
    xgb_val = xgb_model.predict_proba(
        val[xgb_features]
    )[:, 1]
    xgb_test = xgb_model.predict_proba(
        test[xgb_features]
    )[:, 1]

    print("Generating GraphSAGE probabilities...")
    gnn_val = get_gnn_probabilities(val, edges, checkpoint)
    gnn_test = get_gnn_probabilities(test, edges, checkpoint)

    # Align probabilities with the original labelled rows.
    # The graph builder uses the rows in each partition in their
    # existing order; labels and probabilities therefore align.
    print("\nIndividual validation performance:")
    print("XGBoost AP:", average_precision_score(y_val, xgb_val))
    print("GraphSAGE AP:", average_precision_score(y_val, gnn_val))

    # Blend using validation data only.
    # Weight=1 means XGBoost alone; weight=0 means GraphSAGE alone.
    candidates = []

    for xgb_weight in np.linspace(0.0, 1.0, 21):
        blended = (
            xgb_weight * xgb_val +
            (1.0 - xgb_weight) * gnn_val
        )

        ap = average_precision_score(y_val, blended)
        threshold, val_f1 = best_f1_threshold(y_val, blended)

        candidates.append({
            "xgb_weight": float(xgb_weight),
            "gnn_weight": float(1.0 - xgb_weight),
            "validation_ap": float(ap),
            "validation_f1": float(val_f1),
            "threshold": threshold,
        })

    # Select by validation AP; use validation F1 only to report
    # the associated threshold performance.
    best = max(candidates, key=lambda row: row["validation_ap"])

    print("\nSelected blend (validation only):")
    print(json.dumps(best, indent=2))

    # Evaluate test once, after weight and threshold are fixed.
    test_blend = (
        best["xgb_weight"] * xgb_test +
        best["gnn_weight"] * gnn_test
    )

    results = {
        "selection_method": "Validation average precision",
        "selected_blend": best,
        "validation_candidates": candidates,
        "test_results": evaluate(
            "XGBoost + GraphSAGE",
            y_test,
            test_blend,
            best["threshold"],
        ),
        "individual_test_results": {
            "xgboost": {
                "average_precision": float(
                    average_precision_score(y_test, xgb_test)
                ),
                "roc_auc": float(roc_auc_score(y_test, xgb_test)),
            },
            "graphsage": {
                "average_precision": float(
                    average_precision_score(y_test, gnn_test)
                ),
                "roc_auc": float(roc_auc_score(y_test, gnn_test)),
            },
        },
    }

    output = REPORTS / "elliptic_hybrid_metrics.json"
    output.write_text(json.dumps(results, indent=2))

    print("\nFinal test results:")
    print(json.dumps(results["test_results"], indent=2))
    print(f"\nSaved report: {output}")


if __name__ == "__main__":
    main()
