"""
Analyzes the class distribution of the processed dataset to ensure 
balanced representation for training.
"""

import pandas as pd

def main():
    """
    Loads the full dataset, removes duplicates, and prints raw and normalized class counts.
    """
    df = pd.read_csv("data/processed/full_dataset.csv")
    df = df.drop_duplicates()

    print("Class Counts:")
    print(df["label"].value_counts())

    print("\nClass Percentages:")
    print(
        df["label"].value_counts(normalize=True) * 100
    )