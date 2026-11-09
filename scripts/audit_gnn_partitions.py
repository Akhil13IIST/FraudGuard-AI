from pathlib import Path

import pandas as pd

BASE = Path("data/processed/elliptic")

transactions = pd.read_parquet(
    BASE / "transactions.parquet",
    columns=["txId", "node_id", "time_step", "label"]
)

edges = pd.read_parquet(
    BASE / "edges.parquet",
    columns=["txId1", "txId2"]
)

# Create transaction ID -> node ID mapping
tx_to_node = dict(
    zip(transactions["txId"], transactions["node_id"])
)

edges["source"] = edges["txId1"].map(tx_to_node)
edges["target"] = edges["txId2"].map(tx_to_node)

assert edges[["source", "target"]].notna().all().all()

edges["source"] = edges["source"].astype("int64")
edges["target"] = edges["target"].astype("int64")

print("\n========== TEMPORAL GRAPH PARTITIONS ==========")

partitions = {
    "train": (1, 30),
    "validation": (31, 38),
    "test": (39, 49),
}

for name, (start, end) in partitions.items():

    nodes = transactions[
        transactions["time_step"].between(start, end)
    ]

    node_ids = set(nodes["node_id"])

    partition_edges = edges[
        edges["source"].isin(node_ids)
        & edges["target"].isin(node_ids)
    ]

    connected = set(partition_edges["source"]) | set(
        partition_edges["target"]
    )

    print(f"\n{name.upper()}")
    print("Time steps:", start, "-", end)
    print("Total nodes:", len(nodes))
    print("Known labels:", nodes["label"].notna().sum())
    print("Edges:", len(partition_edges))
    print("Nodes with connections:", len(connected))
    print("Isolated nodes:", len(nodes) - len(connected))

    assert len(connected) <= len(nodes)

print("\n========== AUDIT COMPLETE ==========")
