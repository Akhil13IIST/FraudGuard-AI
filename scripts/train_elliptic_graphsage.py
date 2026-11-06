
import json
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
HIDDEN_CHANNELS = 128
DROPOUT = 0.30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 12

BASE = Path("data/processed/elliptic")
MODEL_DIR = Path("models/gnn")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# --------------------------------------------------
# Load data
# --------------------------------------------------

print("\nLoading transactions and graph...")

transactions = pd.read_parquet(
    BASE / "transactions.parquet"
)

edges = pd.read_parquet(
    BASE / "edges.parquet"
)

train_df = pd.read_parquet(BASE / "train.parquet")
val_df = pd.read_parquet(BASE / "validation.parquet")
test_df = pd.read_parquet(BASE / "test.parquet")

feature_cols = [
    c for c in transactions.columns
    if c.startswith("feature_")
]

# Ensure consistent feature order
transactions = transactions.sort_values("node_id").reset_index(drop=True)

assert transactions["node_id"].tolist() == list(
    range(len(transactions))
)

X = transactions[feature_cols].to_numpy(
    dtype=np.float32
)

# Labels: 1 = illicit, 0 = licit, -1 = unknown
y = transactions["label"].fillna(-1).to_numpy(
    dtype=np.int64
)

time_steps = transactions["time_step"].to_numpy()

# Build global node mapping
tx_to_node = dict(
    zip(transactions["txId"], transactions["node_id"])
)

src = edges["txId1"].map(tx_to_node).to_numpy(
    dtype=np.int64
)

dst = edges["txId2"].map(tx_to_node).to_numpy(
    dtype=np.int64
)

print("Nodes:", len(X))
print("Features:", X.shape[1])
print("Edges:", len(src))

# --------------------------------------------------
# Temporal masks
# --------------------------------------------------

train_mask = np.isin(time_steps, np.arange(1, 31)) & (y >= 0)
val_mask = np.isin(time_steps, np.arange(31, 39)) & (y >= 0)
test_mask = np.isin(time_steps, np.arange(39, 50)) & (y >= 0)

train_nodes = np.isin(time_steps, np.arange(1, 31))
val_nodes = np.isin(time_steps, np.arange(31, 39))
test_nodes = np.isin(time_steps, np.arange(39, 50))

# Verify the masks match prepared splits
assert train_mask.sum() == len(train_df)
assert val_mask.sum() == len(val_df)
assert test_mask.sum() == len(test_df)

print("\nLabelled nodes:")
print("Train:", train_mask.sum())
print("Validation:", val_mask.sum())
print("Test:", test_mask.sum())

# --------------------------------------------------
# Build separate temporal graphs
# --------------------------------------------------

def build_graph(node_mask, label_mask):
    selected_nodes = np.flatnonzero(node_mask)

    global_to_local = np.full(
        len(node_mask), -1, dtype=np.int64
    )
    global_to_local[selected_nodes] = np.arange(
        len(selected_nodes)
    )

    keep_edges = (
        node_mask[src] &
        node_mask[dst]
    )

    local_src = global_to_local[src[keep_edges]]
    local_dst = global_to_local[dst[keep_edges]]

    edge_index = torch.tensor(
        np.vstack([local_src, local_dst]),
        dtype=torch.long
    )

    # Treat the graph as undirected for message passing
    edge_index = torch.cat(
        [edge_index, edge_index.flip(0)],
        dim=1
    )

    x = torch.tensor(
        X[selected_nodes],
        dtype=torch.float32
    )

    labels = torch.tensor(
        y[selected_nodes],
        dtype=torch.long
    )

    local_label_mask = labels >= 0

    data = Data(
        x=x,
        edge_index=edge_index,
        y=labels,
        label_mask=local_label_mask,
    )

    return data


train_graph = build_graph(train_nodes, train_mask)
val_graph = build_graph(val_nodes, val_mask)
test_graph = build_graph(test_nodes, test_mask)

print("\nGraph sizes:")
print("Train:", train_graph)
print("Validation:", val_graph)
print("Test:", test_graph)

# --------------------------------------------------
# GraphSAGE model
# --------------------------------------------------

class FraudGraphSAGE(nn.Module):

    def __init__(
        self,
        input_channels,
        hidden_channels,
        dropout
    ):
        super().__init__()

        self.conv1 = SAGEConv(
            input_channels,
            hidden_channels
        )

        self.conv2 = SAGEConv(
            hidden_channels,
            hidden_channels
        )

        self.classifier = nn.Linear(
            hidden_channels,
            1
        )

        self.dropout = dropout

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training
        )

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.classifier(x).squeeze(-1)

        return x


model = FraudGraphSAGE(
    input_channels=len(feature_cols),
    hidden_channels=HIDDEN_CHANNELS,
    dropout=DROPOUT
).to(device)

train_graph = train_graph.to(device)
val_graph = val_graph.to(device)
test_graph = test_graph.to(device)

# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

train_labels = train_graph.y[
    train_graph.label_mask
]

positive = (train_labels == 1).sum().item()
negative = (train_labels == 0).sum().item()

pos_weight = torch.tensor(
    [negative / positive],
    dtype=torch.float32,
    device=device
)

criterion = nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nPositive training labels:", positive)
print("Negative training labels:", negative)
print("Positive class weight:", pos_weight.item())

# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

@torch.no_grad()
def predict(graph):

    model.eval()

    logits = model(
        graph.x,
        graph.edge_index
    )

    probabilities = torch.sigmoid(logits)

    mask = graph.label_mask

    return (
        graph.y[mask].cpu().numpy(),
        probabilities[mask].cpu().numpy()
    )


# --------------------------------------------------
# Training with early stopping
# --------------------------------------------------

best_ap = -1
best_state = None
best_epoch = 0
epochs_without_improvement = 0

print("\n========== TRAINING GRAPH SAGE ==========")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    logits = model(
        train_graph.x,
        train_graph.edge_index
    )

    mask = train_graph.label_mask

    loss = criterion(
        logits[mask],
        train_graph.y[mask].float()
    )

    loss.backward()
    optimizer.step()

    val_y, val_probs = predict(val_graph)

    val_ap = average_precision_score(
        val_y,
        val_probs
    )

    if val_ap > best_ap:
        best_ap = val_ap
        best_epoch = epoch
        best_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss.item():.5f} | "
            f"Validation AP {val_ap:.5f} | "
            f"Best AP {best_ap:.5f}"
        )

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# Restore best model
model.load_state_dict(best_state)

print("\nBest epoch:", best_epoch)
print("Best validation AP:", best_ap)

# --------------------------------------------------
# Select threshold using validation only
# --------------------------------------------------

val_y, val_probs = predict(val_graph)

precision, recall, thresholds = precision_recall_curve(
    val_y,
    val_probs
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)

best_threshold = float(
    thresholds[np.argmax(f1_values)]
)

print("\nSelected validation threshold:", best_threshold)

# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------

test_y, test_probs = predict(test_graph)

test_predictions = (
    test_probs >= best_threshold
).astype(int)

results = {
    "model": "Elliptic GraphSAGE",
    "best_epoch": int(best_epoch),
    "validation_average_precision": float(best_ap),
    "threshold": best_threshold,
    "average_precision": float(
        average_precision_score(test_y, test_probs)
    ),
    "roc_auc": float(
        roc_auc_score(test_y, test_probs)
    ),
    "precision": float(
        precision_score(
            test_y,
            test_predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            test_y,
            test_predictions,
            zero_division=0
        )
    ),
    "f1": float(
        f1_score(
            test_y,
            test_predictions,
            zero_division=0
        )
    ),
    "confusion_matrix": confusion_matrix(
        test_y,
        test_predictions
    ).tolist()
}

print("\n========== TEST RESULTS ==========")
print(json.dumps(results, indent=4))

print("\nClassification report:")
print(
    classification_report(
        test_y,
        test_predictions,
        zero_division=0
    )
)

# --------------------------------------------------
# Save model and results
# --------------------------------------------------

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "input_channels": len(feature_cols),
        "hidden_channels": HIDDEN_CHANNELS,
        "dropout": DROPOUT,
        "feature_columns": feature_cols,
        "threshold": best_threshold,
        "best_epoch": best_epoch,
    },
    MODEL_DIR / "elliptic_graphsage.pt"
)

with open(
    REPORT_DIR / "elliptic_graphsage_metrics.json",
    "w"
) as f:
    json.dump(results, f, indent=4)

print("\nModel and metrics saved successfully.")

if torch.cuda.is_available():
    print(
        "\nPeak GPU memory:",
        round(
            torch.cuda.max_memory_allocated() / 1024**3,
            3
        ),
        "GB"
    )
