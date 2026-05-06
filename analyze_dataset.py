# aim for 20-30 % per class
import pandas as pd

df = pd.read_csv("data/processed/full_dataset.csv")
df = df.drop_duplicates()

print(df["label"].value_counts())

print()

print(
    df["label"].value_counts(normalize=True) * 100
)