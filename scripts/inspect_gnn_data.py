import pandas as pd
from pathlib import Path

BASE = Path("data/processed/elliptic")

transactions = pd.read_parquet(
    BASE / "transactions.parquet"
)

edges = pd.read_parquet(
    BASE / "edges.parquet"
)

mapping = pd.read_parquet(
    BASE / "node_mapping.parquet"
)

train = pd.read_parquet(BASE / "train.parquet")
validation = pd.read_parquet(BASE / "validation.parquet")
test = pd.read_parquet(BASE / "test.parquet")

print("\n========== GNN DATA AUDIT ==========")

print("Transactions:", transactions.shape)
print("Edges:", edges.shape)
print("Node mapping:", mapping.shape)

print("\nNode mapping columns:")
print(mapping.columns.tolist())

print("\nTransaction columns:")
print(transactions.columns.tolist())

print("\nTemporal split sizes:")
print("Train:", len(train))
print("Validation:", len(validation))
print("Test:", len(test))

print("\nNode ID checks:")
print("Unique node IDs:", transactions["node_id"].nunique())
print("Missing node IDs:", transactions["node_id"].isna().sum())

print("\nLabel distribution:")
print(transactions["class"].value_counts(dropna=False))

print("\nTime-step distribution:")
print(transactions["time_step"].value_counts().sort_index())

print("\n========== AUDIT COMPLETE ==========")
