import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]
REAL_SOURCE_TYPES = {"real_media", "manual_live"}
SYNTHETIC_SOURCE_TYPES = {"synthetic", "simulated", "rebalance"}


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw reasoning dataset quality")
    parser.add_argument("--input-glob", default="data/raw/*.csv", help="Glob for raw CSV files")
    parser.add_argument("--min-per-class", type=int, default=50, help="Minimum target rows per class")
    parser.add_argument(
        "--max-class-imbalance-ratio",
        type=float,
        default=1.3,
        help="Max allowed ratio max_class_count / min_nonzero_class_count",
    )
    parser.add_argument(
        "--min-real-share",
        type=float,
        default=0.6,
        help="Minimum required share of real rows (manual_live + real_media) after cleaning",
    )
    parser.add_argument(
        "--max-synthetic-share",
        type=float,
        default=0.4,
        help="Maximum allowed share of synthetic/simulated/rebalance rows after cleaning",
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


def infer_source_type(source_file, explicit_value=None):
    if isinstance(explicit_value, str) and explicit_value.strip():
        return explicit_value.strip().lower()

    name = os.path.basename(str(source_file)).lower()
    if name.startswith("session_") or "reasoning_data" in name:
        return "manual_live"
    if "media_labeled" in name:
        return "real_media"
    if "simulated" in name:
        return "simulated"
    if "balanced_synthetic" in name or "synthetic" in name:
        return "synthetic"
    if "rebalance_patch" in name:
        return "rebalance"
    return "unknown"


def main():
    args = parse_args()

    df, files = load_dataset(args.input_glob)

    has_label = "label" in df.columns
    feature_cols = sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()], key=lambda c: int(c[1:]))

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

    working["source_type"] = [
        infer_source_type(source_file, explicit)
        for source_file, explicit in zip(
            working["__source_file"],
            working["source_type"] if "source_type" in working.columns else [None] * len(working),
        )
    ]

    dedup_subset = feature_cols + ["label"]
    duplicates = int(working.duplicated(subset=dedup_subset, keep="first").sum())
    working = working.drop_duplicates(subset=dedup_subset, keep="first").copy()

    counts = working["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    nonzero_counts = [int(x) for x in counts.values if x > 0]

    min_nonzero = min(nonzero_counts) if nonzero_counts else 0
    max_count = int(counts.max()) if len(counts) else 0
    imbalance_ratio = safe_ratio(max_count, min_nonzero)
    total_clean = int(len(working))

    low_classes = {label: int(count) for label, count in counts.items() if int(count) < args.min_per_class}
    source_counts = working["source_type"].value_counts().sort_index()
    source_type_distribution = {str(k): int(v) for k, v in source_counts.items()}
    source_class_table = (
        working.groupby(["source_type", "label"]).size().unstack(fill_value=0).reindex(columns=ACTION_CLASSES, fill_value=0)
    )
    source_class_distribution = {
        str(src): {label: int(source_class_table.loc[src, label]) for label in ACTION_CLASSES}
        for src in source_class_table.index
    }

    real_rows = int(working["source_type"].isin(REAL_SOURCE_TYPES).sum())
    synthetic_rows = int(working["source_type"].isin(SYNTHETIC_SOURCE_TYPES).sum())
    real_share = safe_ratio(real_rows, total_clean)
    synthetic_share = safe_ratio(synthetic_rows, total_clean)

    checks = {
        "enough_total_rows": len(working) >= args.min_per_class * len(ACTION_CLASSES),
        "all_classes_present": all(int(counts[label]) > 0 for label in ACTION_CLASSES),
        "min_per_class_met": len(low_classes) == 0,
        "imbalance_ok": imbalance_ratio <= args.max_class_imbalance_ratio,
        "real_share_ok": real_share >= args.min_real_share,
        "synthetic_share_ok": synthetic_share <= args.max_synthetic_share,
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
        "source_type_distribution": source_type_distribution,
        "source_class_distribution": source_class_distribution,
        "imbalance": {
            "max_over_min_nonzero": imbalance_ratio,
            "threshold": args.max_class_imbalance_ratio,
        },
        "source_quality": {
            "real_rows": real_rows,
            "synthetic_rows": synthetic_rows,
            "total_clean": total_clean,
            "real_share": real_share,
            "synthetic_share": synthetic_share,
            "min_real_share": args.min_real_share,
            "max_synthetic_share": args.max_synthetic_share,
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
    print(f"  Real share: {real_share:.3f} (min {args.min_real_share:.3f})")
    print(f"  Synthetic share: {synthetic_share:.3f} (max {args.max_synthetic_share:.3f})")
    print(f"  Ready for training: {ready_for_training}")
    print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
