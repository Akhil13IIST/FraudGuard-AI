
from pathlib import Path

import pandas as pd

DATA_DIR = Path(
    "data/raw/elliptic/elliptic_bitcoin_dataset"
)

features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

print("\n========== FEATURES ==========")

features = pd.read_csv(features_path, header=None)

print("Shape:", features.shape)
print(features.head())
print("Missing values:", features.isna().sum().sum())

print("\n========== CLASSES ==========")

classes = pd.read_csv(classes_path)

print("Shape:", classes.shape)
print(classes.head())
print("\nClass distribution:")
print(classes["class"].value_counts(dropna=False))

print("\n========== EDGES ==========")

edges = pd.read_csv(edges_path)

print("Shape:", edges.shape)
print(edges.head())

print("\n========== TIME STEPS ==========")

print("Number of time steps:", features.iloc[:, 1].nunique())
print(features.iloc[:, 1].value_counts().sort_index())

print("\nInspection completed.")
