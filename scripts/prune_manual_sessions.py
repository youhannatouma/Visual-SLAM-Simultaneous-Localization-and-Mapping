#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


DEFAULT_THRESHOLD = 0.65
SLOT_ONEHOT = [
    (0, 1, 2, 3, 4, 5),
    (10, 11, 12, 13, 14, 15),
    (20, 21, 22, 23, 24, 25),
]
SLOT_CONF = [6, 16, 26]


def col(n):
    return f"f{n}"


def prune_and_relabel(df, threshold):
    slot_present = []
    for cols in SLOT_ONEHOT:
        present = df[[col(c) for c in cols]].sum(axis=1) > 0.5
        slot_present.append(present)

    present_mask = slot_present[0] | slot_present[1] | slot_present[2]

    conf_sum = pd.Series(0.0, index=df.index)
    conf_count = pd.Series(0, index=df.index)
    for present, conf_col in zip(slot_present, SLOT_CONF):
        conf_sum = conf_sum + df[col(conf_col)].where(present, other=0.0)
        conf_count = conf_count + present.astype(int)

    avg_conf = conf_sum / conf_count.replace(0, pd.NA)
    avg_conf = avg_conf.fillna(0.0)

    low_conf_mask = (avg_conf < threshold) | (~present_mask)
    pruned = df.loc[~low_conf_mask].copy()

    chair_present = pruned[[col(1), col(11), col(21)]].sum(axis=1) > 0.5
    table_present = pruned[[col(2), col(12), col(22)]].sum(axis=1) > 0.5

    relabel_to_chair = (pruned["label"] == "CHECK_TABLE") & chair_present & (~table_present)
    relabel_to_table = (pruned["label"] == "MOVE_TO_CHAIR") & table_present & (~chair_present)

    relabel_count = int(relabel_to_chair.sum() + relabel_to_table.sum())
    pruned.loc[relabel_to_chair, "label"] = "MOVE_TO_CHAIR"
    pruned.loc[relabel_to_table, "label"] = "CHECK_TABLE"

    return pruned, int(low_conf_mask.sum()), relabel_count


def parse_args():
    parser = argparse.ArgumentParser(description="Prune low-confidence manual session rows and relabel chair/table mismatches")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Avg confidence threshold")
    parser.add_argument("--files", nargs="+", required=True, help="Session CSV paths")
    return parser.parse_args()


def main():
    args = parse_args()
    threshold = float(args.threshold)

    summary_rows = []
    out_paths = []

    for file_path in args.files:
        path = Path(file_path)
        df = pd.read_csv(path)
        before_counts = df["label"].value_counts().to_dict()
        before_rows = len(df)

        pruned, dropped, relabels = prune_and_relabel(df, threshold)
        pruned.to_csv(path, index=False)

        after_counts = pruned["label"].value_counts().to_dict()
        after_rows = len(pruned)

        summary_rows.append(
            {
                "file": path.name,
                "rows_before": before_rows,
                "rows_after": after_rows,
                "dropped_low_conf": dropped,
                "relabels": relabels,
                "before_counts": before_counts,
                "after_counts": after_counts,
            }
        )
        out_paths.append(path)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    all_df = pd.concat([pd.read_csv(p) for p in out_paths], ignore_index=True)
    print("\nCombined class counts after pruning:")
    print(all_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
