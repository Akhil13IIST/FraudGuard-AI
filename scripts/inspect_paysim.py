
from pathlib import Path

import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

csv_files = list(RAW_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV file found inside {RAW_DIR}"
    )

DATA_PATH = csv_files[0]

CHUNK_SIZE = 250_000

print("=" * 55)
print("FRAUDGUARD AI - PAYSIM DATA INSPECTION")
print("=" * 55)

print("Dataset:", DATA_PATH.name)
print("File size (MB):", round(DATA_PATH.stat().st_size / 1e6, 2))


# --------------------------------------------------
# CHUNKED DATA INSPECTION
# --------------------------------------------------

total_rows = 0
fraud_counts = {0: 0, 1: 0}
type_counts = {}
missing_counts = {}
amount_sum = 0.0
amount_min = float("inf")
amount_max = float("-inf")
duplicate_count = 0
column_names = None

for chunk_number, chunk in enumerate(
    pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
):

    if column_names is None:
        column_names = chunk.columns.tolist()
        print("\nColumns:")
        print(column_names)

    total_rows += len(chunk)

    # Fraud distribution
    counts = chunk["isFraud"].value_counts()

    for label, count in counts.items():
        fraud_counts[int(label)] += int(count)

    # Transaction types
    for transaction_type, count in (
        chunk["type"].value_counts().items()
    ):
        type_counts[transaction_type] = (
            type_counts.get(transaction_type, 0) + int(count)
        )

    # Missing values
    for column, count in chunk.isnull().sum().items():
        missing_counts[column] = (
            missing_counts.get(column, 0) + int(count)
        )

    # Amount statistics
    amount_sum += chunk["amount"].sum()
    amount_min = min(amount_min, chunk["amount"].min())
    amount_max = max(amount_max, chunk["amount"].max())

    duplicate_count += int(chunk.duplicated().sum())

    print(
        f"Processed chunk {chunk_number + 1} | "
        f"Rows processed: {total_rows:,}"
    )


# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 55)
print("DATASET OVERVIEW")
print("=" * 55)

print("Total rows:", f"{total_rows:,}")
print("Total columns:", len(column_names))

print("\nFraud distribution:")

for label, count in fraud_counts.items():
    percentage = count / total_rows * 100

    name = "Fraud" if label == 1 else "Legitimate"

    print(
        f"{name}: {count:,} ({percentage:.5f}%)"
    )

print("\nFraud ratio:")

print(
    f"Fraud rate: "
    f"{fraud_counts[1] / total_rows * 100:.5f}%"
)

print("\nTransaction types:")

for name, count in sorted(
    type_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{name}: {count:,}")

print("\nMissing values:")

for column, count in missing_counts.items():
    print(f"{column}: {count:,}")

print("\nAmount statistics:")

print("Minimum:", amount_min)
print("Maximum:", amount_max)
print("Mean:", amount_sum / total_rows)

print("\nDuplicate rows within individual chunks:", duplicate_count)

print("\nInspection completed successfully.")
