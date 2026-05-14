import argparse
from pathlib import Path

import pandas as pd


TARGET_FEATURE_COUNT = 43
TARGET_FEATURE_COLS = [f"f{i}" for i in range(TARGET_FEATURE_COUNT)]
OPTIONAL_METADATA_COLS = [
    "label",
    "source_type",
    "source_file",
    "frame_index",
    "auto_label",
    "needs_review",
    "batch_id",
    "scenario",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Upgrade reasoning CSVs to the 43-feature schema")
    parser.add_argument("paths", nargs="+", help="CSV file paths to rewrite in place")
    return parser.parse_args()


def feature_cols(df):
    cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    return sorted(cols, key=lambda col: int(col[1:]))


def main():
    args = parse_args()

    for raw_path in args.paths:
        path = Path(raw_path)
        df = pd.read_csv(path)
        existing = feature_cols(df)

        if len(existing) > TARGET_FEATURE_COUNT:
            raise ValueError(f"{path} has {len(existing)} feature columns; expected <= {TARGET_FEATURE_COUNT}")

        for idx in range(len(existing), TARGET_FEATURE_COUNT):
            df[f"f{idx}"] = 0.0

        ordered_cols = [col for col in TARGET_FEATURE_COLS if col in df.columns]
        ordered_cols.extend(col for col in OPTIONAL_METADATA_COLS if col in df.columns)
        remaining_cols = [col for col in df.columns if col not in ordered_cols]
        ordered_cols.extend(remaining_cols)

        df = df[ordered_cols]
        df.to_csv(path, index=False)
        print(f"Upgraded {path} to {TARGET_FEATURE_COUNT} features")


if __name__ == "__main__":
    main()
