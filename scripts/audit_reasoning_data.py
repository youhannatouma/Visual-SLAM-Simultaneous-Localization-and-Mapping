import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw reasoning dataset quality")
    parser.add_argument("--input-glob", default="data/raw/*.csv", help="Glob for raw CSV files")
    parser.add_argument("--min-per-class", type=int, default=50, help="Minimum target rows per class")
    parser.add_argument(
        "--max-class-imbalance-ratio",
        type=float,
        default=2.0,
        help="Max allowed ratio max_class_count / min_nonzero_class_count",
    )
    parser.add_argument("--report", default="reports/dataset_audit.json", help="Path to JSON report")
    return parser.parse_args()


def load_dataset(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["__source_file"] = os.path.basename(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), files


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return float("inf")
    return float(numerator) / float(denominator)


def main():
    args = parse_args()

    df, files = load_dataset(args.input_glob)

    has_label = "label" in df.columns
    feature_cols = sorted([c for c in df.columns if c.startswith("f")], key=lambda c: int(c[1:]) if c[1:].isdigit() else c)

    if not has_label or not feature_cols:
        raise ValueError("Dataset must have label column and feature columns f0..fN")

    working = df.copy()
    for col in feature_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    total_rows = len(working)
    missing_required = int(total_rows - len(working.dropna(subset=feature_cols + ["label"])))

    working = working.dropna(subset=feature_cols + ["label"]).copy()

    finite_mask = np.isfinite(working[feature_cols]).all(axis=1)
    non_finite = int((~finite_mask).sum())
    working = working.loc[finite_mask].copy()

    invalid_label_mask = ~working["label"].isin(ACTION_CLASSES)
    invalid_labels = int(invalid_label_mask.sum())
    working = working.loc[~invalid_label_mask].copy()

    dedup_subset = feature_cols + ["label"]
    duplicates = int(working.duplicated(subset=dedup_subset, keep="first").sum())
    working = working.drop_duplicates(subset=dedup_subset, keep="first").copy()

    counts = working["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    nonzero_counts = [int(x) for x in counts.values if x > 0]

    min_nonzero = min(nonzero_counts) if nonzero_counts else 0
    max_count = int(counts.max()) if len(counts) else 0
    imbalance_ratio = safe_ratio(max_count, min_nonzero)

    low_classes = {label: int(count) for label, count in counts.items() if int(count) < args.min_per_class}

    checks = {
        "enough_total_rows": len(working) >= args.min_per_class * len(ACTION_CLASSES),
        "all_classes_present": all(int(counts[label]) > 0 for label in ACTION_CLASSES),
        "min_per_class_met": len(low_classes) == 0,
        "imbalance_ok": imbalance_ratio <= args.max_class_imbalance_ratio,
    }
    ready_for_training = all(checks.values())

    summary = {
        "input_glob": args.input_glob,
        "files": files,
        "rows": {
            "raw": int(total_rows),
            "clean": int(len(working)),
            "dropped_missing_required": int(missing_required),
            "dropped_non_finite": int(non_finite),
            "dropped_invalid_label": int(invalid_labels),
            "dropped_duplicates": int(duplicates),
        },
        "features": {
            "count": len(feature_cols),
            "columns": feature_cols,
        },
        "class_distribution": {label: int(counts[label]) for label in ACTION_CLASSES},
        "imbalance": {
            "max_over_min_nonzero": imbalance_ratio,
            "threshold": args.max_class_imbalance_ratio,
        },
        "minimum_per_class": {
            "target": args.min_per_class,
            "low_classes": low_classes,
        },
        "checks": checks,
        "ready_for_training": ready_for_training,
    }

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Dataset audit summary")
    print(f"  Files: {len(files)}")
    print(f"  Raw rows: {total_rows}")
    print(f"  Clean rows: {len(working)}")
    print("  Class counts:")
    for label in ACTION_CLASSES:
        print(f"    {label:<15} {int(counts[label])}")
    print(f"  Imbalance ratio (max/min_nonzero): {imbalance_ratio:.2f}")
    print(f"  Ready for training: {ready_for_training}")
    print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
