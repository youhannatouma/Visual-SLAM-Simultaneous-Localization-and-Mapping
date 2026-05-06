import pandas as pd
import glob

files = glob.glob("data/raw/*.csv")

dfs = [pd.read_csv(f) for f in files]

merged = pd.concat(dfs, ignore_index=True)

merged.to_csv("data/processed/full_dataset.csv", index=False)

print("Merged:", len(merged))