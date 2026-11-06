
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/elliptic/elliptic_bitcoin_dataset")
OUT_DIR = Path("data/processed/elliptic")

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Elliptic data...")

# --------------------------------------------------
# Load raw files
# --------------------------------------------------

features = pd.read_csv(
    RAW_DIR / "elliptic_txs_features.csv",
    header=None
)

classes = pd.read_csv(
    RAW_DIR / "elliptic_txs_classes.csv"
)

edges = pd.read_csv(
    RAW_DIR / "elliptic_txs_edgelist.csv"
)

# --------------------------------------------------
# Name feature columns
# --------------------------------------------------

feature_columns = (
    ["txId", "time_step"]
    + [f"feature_{i}" for i in range(1, 166)]
)

features.columns = feature_columns

# --------------------------------------------------
# Attach labels
# --------------------------------------------------

data = features.merge(
    classes,
    on="txId",
    how="left",
    validate="one_to_one"
)

data["class"] = data["class"].fillna("unknown")

data["label"] = data["class"].map({
    "1": 1,
    "2": 0
})

# --------------------------------------------------
# Build transaction ID mapping
# --------------------------------------------------

data = data.reset_index(drop=True)

data["node_id"] = np.arange(len(data), dtype=np.int64)

tx_to_node = dict(
    zip(data["txId"], data["node_id"])
)

# --------------------------------------------------
# Map graph edges
# --------------------------------------------------

edges["source"] = edges["txId1"].map(tx_to_node)
edges["target"] = edges["txId2"].map(tx_to_node)

edges = edges.dropna(
    subset=["source", "target"]
).copy()

edges["source"] = edges["source"].astype("int64")
edges["target"] = edges["target"].astype("int64")

# --------------------------------------------------
# Temporal split
# --------------------------------------------------

# Chronological split:
# Train: time steps 1-30
# Validation: time steps 31-38
# Test: time steps 39-49

train = data[
    (data["time_step"] <= 30)
    & data["label"].notna()
].copy()

validation = data[
    (data["time_step"].between(31, 38))
    & data["label"].notna()
].copy()

test = data[
    (data["time_step"] >= 39)
    & data["label"].notna()
].copy()

# --------------------------------------------------
# Save outputs
# --------------------------------------------------

data.to_parquet(
    OUT_DIR / "transactions.parquet",
    index=False
)

edges.to_parquet(
    OUT_DIR / "edges.parquet",
    index=False
)

train.to_parquet(
    OUT_DIR / "train.parquet",
    index=False
)

validation.to_parquet(
    OUT_DIR / "validation.parquet",
    index=False
)

test.to_parquet(
    OUT_DIR / "test.parquet",
    index=False
)

# Store mapping for graph model
mapping = data[["txId", "node_id"]]

mapping.to_parquet(
    OUT_DIR / "node_mapping.parquet",
    index=False
)

# --------------------------------------------------
# Report
# --------------------------------------------------

print("\n========== PREPROCESSING SUMMARY ==========")

print("All transactions:", len(data))
print("Graph edges:", len(edges))
print("Unique time steps:", data["time_step"].nunique())

for name, split in [
    ("Train", train),
    ("Validation", validation),
    ("Test", test)
]:
    print(f"\n{name}:")
    print("Rows:", len(split))
    print("Illicit:", int(split["label"].sum()))
    print("Licit:", int((split["label"] == 0).sum()))
    print("Illicit rate:", round(split["label"].mean(), 5))

print("\nSaved outputs to:", OUT_DIR)
