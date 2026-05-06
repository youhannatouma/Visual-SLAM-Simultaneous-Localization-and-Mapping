import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "data/processed/full_dataset.csv"

print("Loading dataset...")
df = pd.read_csv(INPUT)

print("Total rows:", len(df))
print(df["label"].value_counts())

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))