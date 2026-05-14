import argparse
import fnmatch
import glob
import json
import os

import numpy as np
import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]
REAL_SOURCE_TYPES = {"real_media", "manual_live"}
EXCLUDED_RAW_FILE_PATTERNS = (
    "zz_fresh_real_holdout_*.csv",
    "media_labeled_stage2_train_refix_*.csv",
    "media_labeled_stage2_move_hardneg_*.csv",
    "move_recovery_pool_*.csv",
    "vid*.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Audit raw reasoning dataset quality")
    parser.add_argument("--input-glob", default="data/raw/*.csv", help="Glob for raw CSV files")
    parser.add_argument("--min-per-class", type=int, default=50, help="Minimum target rows per class")
    parser.add_argument(
        "--max-class-imbalance-ratio",
        type=float,
        default=1.35,
        help="Max allowed ratio max_class_count / min_nonzero_class_count",
    )
    parser.add_argument(
        "--min-real-share",
        type=float,
        default=0.6,
        help="Minimum required share of real rows (manual_live + real_media) after cleaning",
    )
    parser.add_argument(
        "--require-two-real-batches",
        action="store_true",
        help="Require at least 2 independent real batches/sources in cleaned data",
    )
    parser.add_argument(
        "--scenario-quota-json",
        default="",
        help="Optional JSON map like {\"low_light\":{\"CHECK_TABLE\":50}} to enforce scenario/class quotas",
    )
    parser.add_argument("--report", default="reports/dataset_audit.json", help="Path to JSON report")
    return parser.parse_args()


def load_dataset(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    filtered_files = []
    excluded_files = []
    for path in files:
        name = os.path.basename(path)
        if any(fnmatch.fnmatch(name, pat) for pat in EXCLUDED_RAW_FILE_PATTERNS):
            excluded_files.append(path)
            continue
        filtered_files.append(path)

    if not filtered_files:
        raise FileNotFoundError(
            "All matched CSV files were excluded by raw-file policy. "
            f"Pattern={pattern} excluded={excluded_files}"
        )

    frames = []
    for path in filtered_files:
        frame = pd.read_csv(path)
        frame["__source_file"] = os.path.basename(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), filtered_files


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
    real_share = safe_ratio(real_rows, total_clean)
    scenario_class_distribution = {}
    if "scenario" in working.columns:
        scenario_table = (
            working.groupby(["scenario", "label"]).size().unstack(fill_value=0).reindex(columns=ACTION_CLASSES, fill_value=0)
        )
        scenario_class_distribution = {
            str(scenario): {label: int(scenario_table.loc[scenario, label]) for label in ACTION_CLASSES}
            for scenario in scenario_table.index
        }

    independent_real_batches = set()
    real_df = working[working["source_type"].isin(REAL_SOURCE_TYPES)]
    if "batch_id" in real_df.columns:
        independent_real_batches = {
            str(x).strip() for x in real_df["batch_id"].dropna().astype(str).tolist() if str(x).strip()
        }
    if len(independent_real_batches) < 2:
        independent_real_batches = set(real_df["__source_file"].astype(str).tolist())

    scenario_quota_ok = True
    scenario_quota_failures = []
    if args.scenario_quota_json:
        quota = json.loads(args.scenario_quota_json)
        for scenario, label_map in quota.items():
            for label, target in label_map.items():
                actual = int(scenario_class_distribution.get(scenario, {}).get(label, 0))
                if actual < int(target):
                    scenario_quota_ok = False
                    scenario_quota_failures.append(
                        {
                            "scenario": scenario,
                            "label": label,
                            "target": int(target),
                            "actual": int(actual),
                        }
                    )

    checks = {
        "enough_total_rows": len(working) >= args.min_per_class * len(ACTION_CLASSES),
        "all_classes_present": all(int(counts[label]) > 0 for label in ACTION_CLASSES),
        "min_per_class_met": len(low_classes) == 0,
        "imbalance_ok": imbalance_ratio <= args.max_class_imbalance_ratio,
        "real_share_ok": real_share >= args.min_real_share,
        "two_real_batches_ok": (len(independent_real_batches) >= 2) if args.require_two_real_batches else True,
        "scenario_quota_ok": scenario_quota_ok,
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
        "scenario_class_distribution": scenario_class_distribution,
        "imbalance": {
            "max_over_min_nonzero": imbalance_ratio,
            "threshold": args.max_class_imbalance_ratio,
        },
        "source_quality": {
            "real_rows": real_rows,
            "total_clean": total_clean,
            "real_share": real_share,
            "min_real_share": args.min_real_share,
            "independent_real_batches_count": int(len(independent_real_batches)),
            "independent_real_batches": sorted(list(independent_real_batches)),
        },
        "scenario_quota": {
            "requested": bool(args.scenario_quota_json),
            "failures": scenario_quota_failures,
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
    if args.require_two_real_batches:
        print(f"  Independent real batches/sources: {len(independent_real_batches)} (min 2)")
    print(f"  Ready for training: {ready_for_training}")
    print(f"  Report: {args.report}")


if __name__ == "__main__":
    main()
