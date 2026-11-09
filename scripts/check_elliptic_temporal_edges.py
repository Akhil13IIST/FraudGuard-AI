from pathlib import Path

import pandas as pd

BASE = Path("data/processed/elliptic")

transactions = pd.read_parquet(
    BASE / "transactions.parquet",
    columns=["txId", "time_step"]
)

edges = pd.read_parquet(BASE / "edges.parquet")

time_map = transactions.set_index("txId")["time_step"]

edges["source_time"] = edges["txId1"].map(time_map)
edges["target_time"] = edges["txId2"].map(time_map)

print("\n========== TEMPORAL GRAPH AUDIT ==========")

print("Total edges:", len(edges))

print("\nMissing timestamps:")
print(edges[["source_time", "target_time"]].isna().sum())

print("\nTime difference (target - source):")
print(
    (edges["target_time"] - edges["source_time"])
    .describe()
)

print("\nEdge time relationship:")
print(
    (edges["target_time"] - edges["source_time"])
    .value_counts()
    .sort_index()
)

print("\nFuture-directed edges:")
print(
    (edges["target_time"] > edges["source_time"]).sum()
)

print("\nSame-time-step edges:")
print(
    (edges["target_time"] == edges["source_time"]).sum()
)

print("\nPast-directed edges:")
print(
    (edges["target_time"] < edges["source_time"]).sum()
)
