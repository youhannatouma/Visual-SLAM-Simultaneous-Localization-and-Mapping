"""
Aggregates all raw CSV session data into a single processed dataset file.
"""

import pandas as pd
import glob

def main():
    """
    Finds all CSV files in data/raw/, concatenates them, and saves to data/processed/.
    """
    files = glob.glob("data/raw/*.csv")

    dfs = [pd.read_csv(f) for f in files]

    if not dfs:
        print("No raw data found to merge.")
        return

    merged = pd.concat(dfs, ignore_index=True)

    merged.to_csv("data/processed/full_dataset.csv", index=False)

print("Merged:", len(merged))