import argparse
import glob
import json
import os
import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]
REAL_SOURCE_TYPES = {"real_media", "manual_live"}
SYNTHETIC_SOURCE_TYPES = {"synthetic", "simulated", "rebalance"}
EXCLUDED_RAW_FILE_PATTERNS = (
    "zz_fresh_real_holdout_*.csv",
    "media_labeled_stage2_train_refix_*.csv",
    "media_labeled_stage2_move_hardneg_*.csv",
    "move_recovery_pool_*.csv",
    "vid*.csv",
)
FRESH_HOLDOUT_EXCLUDED_PATTERNS = (
    "curated_real_balanced.csv",
    "media_labeled_stage2_check_table_train_*.csv",
    "media_labeled_stage2_curated_train_*.csv",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare reasoning dataset with pandas/numpy")
    parser.add_argument("--input-glob", default="data/raw/*.csv", help="Glob for raw CSV files")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory")
    parser.add_argument("--balance", choices=["none", "cap", "oversample"], default="cap", help="Class balancing strategy")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
        help="Minimum samples required per action class after cleaning (fails if unmet)",
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
    parser.add_argument(
        "--holdout-latest-real-source",
        action="store_true",
        help="Deprecated compatibility flag. Enables deterministic multi-source holdout.",
    )
    parser.add_argument(
        "--holdout-per-class",
        type=int,
        default=12,
        help="Target rows per class for fresh real holdout assembly.",
    )
    parser.add_argument(
        "--holdout-min-total",
        type=int,
        default=48,
        help="Minimum required total rows for fresh real holdout.",
    )
    parser.add_argument(
        "--holdout-min-sources",
        type=int,
        default=4,
        help="Minimum required independent real sources represented in holdout.",
    )
    parser.add_argument(
        "--enforce-review-applied",
        action="store_true",
        help="Fail if pending review status exists for input CSVs",
    )
    parser.add_argument(
        "--require-two-real-batches-for-holdout",
        action="store_true",
        help="When holdout is enabled, require at least 2 independent real batches/sources",
    )
    return parser.parse_args()


def load_raw_frames(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

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


def validate_and_clean(df):
    if "label" not in df.columns:
        raise ValueError("Input CSV must include a 'label' column")

    feature_cols = [c for c in df.columns if c.startswith("f") and c[1:].isdigit()]
    if not feature_cols:
        raise ValueError("Input CSV must include feature columns named f0..fN")

    # Stable numeric order for feature columns: f0, f1, ...
    feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]))

    working = df.copy()
    for col in feature_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    required_cols = feature_cols + ["label"]
    rows_initial = len(working)
    working = working.dropna(subset=required_cols)
    rows_after_dropna = len(working)

    working = working[np.isfinite(working[feature_cols]).all(axis=1)]
    rows_after_finite = len(working)

    before_label_filter = len(working)
    working = working[working["label"].isin(ACTION_CLASSES)].copy()
    rows_after_label_filter = len(working)

    working["source_type"] = [
        infer_source_type(source_file, explicit)
        for source_file, explicit in zip(
            working["__source_file"],
            working["source_type"] if "source_type" in working.columns else [None] * len(working),
        )
    ]

    provenance_cols = [
        c
        for c in ["__source_file", "source_file", "frame_index", "batch_id", "scenario"]
        if c in working.columns
    ]
    dedup_cols = required_cols + provenance_cols
    before_dedup = len(working)
    working = working.drop_duplicates(subset=dedup_cols, keep="first")
    rows_after_dedup = len(working)

    if working.empty:
        raise ValueError("No valid rows after cleaning")

    summary = {
        "rows_initial": int(rows_initial),
        "rows_after_dropna": int(rows_after_dropna),
        "rows_after_finite_check": int(rows_after_finite),
        "rows_after_label_filter": int(rows_after_label_filter),
        "rows_after_dedup": int(rows_after_dedup),
        "dropped_missing_required": int(rows_initial - rows_after_dropna),
        "dropped_non_finite": int(rows_after_dropna - rows_after_finite),
        "dropped_unknown_label": int(before_label_filter - rows_after_label_filter),
        "dropped_duplicates": int(before_dedup - rows_after_dedup),
        "feature_count": len(feature_cols),
    }

    return working, feature_cols, summary


def enforce_minimum_per_class(df, minimum):
    counts = df["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    low_classes = {label: int(count) for label, count in counts.items() if count < minimum}
    if low_classes:
        details = ", ".join([f"{label}={count}" for label, count in low_classes.items()])
        raise ValueError(
            "Dataset quality gate failed: not enough samples per class. "
            f"Need at least {minimum} each, but got: {details}"
        )
    return counts


def enforce_source_quality(df, min_real_share, max_synthetic_share):
    total = len(df)
    real_rows = int(df["source_type"].isin(REAL_SOURCE_TYPES).sum())
    synthetic_rows = int(df["source_type"].isin(SYNTHETIC_SOURCE_TYPES).sum())
    real_share = (real_rows / total) if total else 0.0
    synthetic_share = (synthetic_rows / total) if total else 0.0

    if real_share < min_real_share:
        raise ValueError(
            "Dataset quality gate failed: real data share too low. "
            f"Need >= {min_real_share:.3f}, got {real_share:.3f}"
        )
    if synthetic_share > max_synthetic_share:
        raise ValueError(
            "Dataset quality gate failed: synthetic share too high. "
            f"Need <= {max_synthetic_share:.3f}, got {synthetic_share:.3f}"
        )

    return {
        "real_rows": int(real_rows),
        "synthetic_rows": int(synthetic_rows),
        "total_clean": int(total),
        "real_share": float(real_share),
        "synthetic_share": float(synthetic_share),
        "min_real_share": float(min_real_share),
        "max_synthetic_share": float(max_synthetic_share),
    }


def print_distribution(df, title):
    print(f"\n{title}")
    counts = df["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    total = counts.sum()
    for label, count in counts.items():
        pct = (count / total * 100.0) if total else 0.0
        print(f"  {label:<15} {count:>6} ({pct:5.1f}%)")


def balance_classes(df, strategy, seed):
    if strategy == "none":
        return df

    rng = np.random.default_rng(seed)
    grouped = {label: grp for label, grp in df.groupby("label")}
    for label in ACTION_CLASSES:
        grouped.setdefault(label, df.iloc[0:0])

    non_empty_counts = [len(grouped[label]) for label in ACTION_CLASSES if len(grouped[label]) > 0]
    if not non_empty_counts:
        raise ValueError("No non-empty classes to balance")

    if strategy == "cap":
        target = int(np.median(non_empty_counts))
        target = max(1, target)
        balanced_parts = []
        for label in ACTION_CLASSES:
            grp = grouped[label]
            if grp.empty:
                continue
            sample_n = min(len(grp), target)
            idx = rng.choice(grp.index.to_numpy(), size=sample_n, replace=False)
            balanced_parts.append(grp.loc[idx])
        balanced = pd.concat(balanced_parts, ignore_index=True)
        return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if strategy == "oversample":
        target = max(non_empty_counts)
        balanced_parts = []
        for label in ACTION_CLASSES:
            grp = grouped[label]
            if grp.empty:
                continue
            replace = len(grp) < target
            idx = rng.choice(grp.index.to_numpy(), size=target, replace=replace)
            balanced_parts.append(grp.loc[idx])
        balanced = pd.concat(balanced_parts, ignore_index=True)
        return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    raise ValueError(f"Unsupported balance strategy: {strategy}")


def stratified_split(df, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    train_parts, val_parts, test_parts = [], [], []

    for label, grp in df.groupby("label"):
        grp = grp.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(grp)

        n_train = int(np.floor(n * train_ratio))
        n_val = int(np.floor(n * val_ratio))
        n_test = n - n_train - n_val

        # Ensure non-empty slices where possible.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
        elif n == 2:
            n_train, n_val, n_test = 1, 0, 1
        elif n == 1:
            n_train, n_val, n_test = 1, 0, 0

        train_parts.append(grp.iloc[:n_train])
        val_parts.append(grp.iloc[n_train:n_train + n_val])
        test_parts.append(grp.iloc[n_train + n_val:n_train + n_val + n_test])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def save_outputs(train_df, val_df, test_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_path, val_path, test_path


def _real_unit_columns(df):
    batch_col = "batch_id" if "batch_id" in df.columns else None
    source_col = "__source_file" if "__source_file" in df.columns else None
    return batch_col, source_col


def holdout_multisource_balanced(df, out_dir, holdout_per_class, holdout_min_total, holdout_min_sources):
    real_mask = df["source_type"].isin(REAL_SOURCE_TYPES)
    real_df = df.loc[real_mask].copy()
    if real_df.empty:
        raise ValueError("Cannot build fresh real holdout: no real rows found")

    batch_col, source_col = _real_unit_columns(real_df)
    if source_col is None:
        raise ValueError("Cannot build fresh real holdout: source metadata missing")

    working = real_df.copy()
    if "__source_file" in working.columns:
        holdout_mask = ~working["__source_file"].astype(str).apply(
            lambda name: any(fnmatch.fnmatch(name, pat) for pat in FRESH_HOLDOUT_EXCLUDED_PATTERNS)
        )
        holdout_candidates = working.loc[holdout_mask].copy()
        if not holdout_candidates.empty:
            working = holdout_candidates
    if batch_col is not None:
        working["__holdout_unit"] = working[batch_col].fillna("").astype(str).str.strip()
        working.loc[working["__holdout_unit"].eq(""), "__holdout_unit"] = working[source_col].astype(str)
    else:
        working["__holdout_unit"] = working[source_col].astype(str)

    unit_counts = working["__holdout_unit"].value_counts()
    if len(unit_counts) < holdout_min_sources:
        raise ValueError(
            "Holdout gate failed: requires at least "
            f"{holdout_min_sources} independent real batches/sources; found {len(unit_counts)}"
        )

    # Deterministic order: larger units first, then lexical key.
    ordered_units = sorted(unit_counts.index.tolist(), key=lambda u: (-int(unit_counts[u]), str(u)))

    hold_parts = []
    used_idx = set()
    for label_idx, label in enumerate(ACTION_CLASSES):
        target = int(max(1, holdout_per_class))
        class_rows = working[working["label"] == label].copy()
        if class_rows.empty:
            continue
        picked = []
        rotated_units = ordered_units[label_idx % len(ordered_units):] + ordered_units[:label_idx % len(ordered_units)]
        unit_cursor = 0
        while len(picked) < target and unit_cursor < (len(rotated_units) * 4):
            unit = rotated_units[unit_cursor % len(rotated_units)]
            unit_rows = class_rows[class_rows["__holdout_unit"] == unit]
            added = False
            for idx, row in unit_rows.iterrows():
                if idx in used_idx:
                    continue
                picked.append(idx)
                used_idx.add(idx)
                added = True
                break
            if not added and len(picked) >= len(class_rows):
                break
            unit_cursor += 1
            if not added and unit_cursor >= len(rotated_units):
                # fallback: grab any remaining row for this class
                for idx, row in class_rows.iterrows():
                    if idx in used_idx:
                        continue
                    picked.append(idx)
                    used_idx.add(idx)
                    break
        if picked:
            hold_parts.append(working.loc[picked].copy())

    if not hold_parts:
        raise ValueError("Holdout gate failed: no rows selected for holdout")

    holdout_df = pd.concat(hold_parts, ignore_index=True)
    holdout_df = holdout_df.drop(columns=["__holdout_unit"], errors="ignore")
    remaining_df = df.drop(index=list(used_idx), errors="ignore").copy()
    if holdout_df.empty or remaining_df.empty:
        raise ValueError("Invalid holdout split: holdout or remaining set is empty")
    if len(holdout_df) < int(holdout_min_total):
        raise ValueError(
            "Holdout gate failed: holdout too small. "
            f"Need >= {holdout_min_total} rows, got {len(holdout_df)}"
        )

    holdout_label_counts = holdout_df["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    zero_labels = [label for label, n in holdout_label_counts.items() if int(n) == 0]
    if zero_labels:
        raise ValueError(
            "Holdout gate failed: missing class coverage in holdout. "
            f"Missing: {', '.join(zero_labels)}"
        )

    holdout_sources = (
        holdout_df[batch_col].fillna("").astype(str).str.strip()
        if batch_col is not None
        else pd.Series([], dtype=str)
    )
    if batch_col is not None:
        holdout_sources = holdout_sources[holdout_sources.ne("")]
    if batch_col is None or holdout_sources.empty:
        holdout_sources = holdout_df[source_col].astype(str)
    source_set = set(holdout_sources.tolist())
    if len(source_set) < holdout_min_sources:
        missing_units = [u for u in ordered_units if u not in source_set]
        for unit in missing_units:
            unit_candidates = working[working["__holdout_unit"] == unit]
            if unit_candidates.empty:
                continue
            for idx, row in unit_candidates.iterrows():
                if idx in used_idx:
                    continue
                holdout_df = pd.concat([holdout_df, row.to_frame().T], ignore_index=True)
                used_idx.add(idx)
                source_set.add(unit)
                break
            if len(source_set) >= holdout_min_sources:
                break

    if len(source_set) < holdout_min_sources:
        raise ValueError(
            "Holdout gate failed: holdout lacks source diversity. "
            f"Need >= {holdout_min_sources} sources in holdout."
        )

    holdout_path = os.path.join(out_dir, "fresh_real_eval.csv")
    holdout_df.to_csv(holdout_path, index=False)
    qa = {
        "rows": int(len(holdout_df)),
        "per_class": {label: int(holdout_label_counts[label]) for label in ACTION_CLASSES},
        "per_source": {str(k): int(v) for k, v in pd.Series(holdout_sources).value_counts().to_dict().items()},
        "dropped_from_train_due_to_holdout": int(len(used_idx)),
        "min_required_total": int(holdout_min_total),
        "min_required_sources": int(holdout_min_sources),
    }
    return remaining_df, holdout_path, qa


def enforce_review_status(input_files):
    status_dir = Path("reports/review_status")
    if not status_dir.exists():
        return
    input_set = {os.path.basename(x) for x in input_files}
    pending = []
    for status_file in sorted(status_dir.glob("*.json")):
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        out_csv = payload.get("output_csv", "")
        status = str(payload.get("status", "")).lower()
        if not out_csv:
            continue
        if os.path.basename(out_csv) in input_set and status != "applied":
            pending.append((status_file.name, os.path.basename(out_csv), status))
    if pending:
        details = ", ".join([f"{sfile}:{csv_name}:{status}" for sfile, csv_name, status in pending])
        raise ValueError(
            "Review gate failed: unapplied review statuses detected for current inputs. "
            f"Details: {details}"
        )


def main():
    args = parse_args()

    merged_df, files = load_raw_frames(args.input_glob)
    if args.enforce_review_applied:
        enforce_review_status(files)
    clean_df, feature_cols, clean_summary = validate_and_clean(merged_df)
    counts_after_clean = enforce_minimum_per_class(clean_df, args.min_per_class)
    source_quality = enforce_source_quality(clean_df, args.min_real_share, args.max_synthetic_share)

    print(f"Loaded {len(files)} file(s)")
    print_distribution(clean_df, "Class distribution before balancing")

    holdout_path = ""
    holdout_qa = {}
    holdout_rows = 0
    split_input_df = clean_df
    if args.holdout_latest_real_source:
        min_sources = args.holdout_min_sources
        if args.require_two_real_batches_for_holdout:
            min_sources = max(min_sources, 2)
        split_input_df, holdout_path, holdout_qa = holdout_multisource_balanced(
            clean_df,
            args.out_dir,
            args.holdout_per_class,
            args.holdout_min_total,
            min_sources,
        )
        holdout_rows = int(holdout_qa.get("rows", 0))
        print(
            "Held out deterministic multi-source fresh eval "
            f"({holdout_rows} rows; sources={len(holdout_qa.get('per_source', {}))})"
        )
        print(f"Saved fresh real eval set: {holdout_path}")

    balanced_df = balance_classes(split_input_df, args.balance, args.seed)
    print_distribution(balanced_df, f"Class distribution after balancing ({args.balance})")

    train_df, val_df, test_df = stratified_split(
        balanced_df,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    train_path, val_path, test_path = save_outputs(train_df, val_df, test_df, args.out_dir)

    metadata = {
        "input_glob": args.input_glob,
        "input_files": files,
        "balance": args.balance,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "cleaning": clean_summary,
        "minimum_per_class": {
            "required": int(args.min_per_class),
            "counts_after_clean": {label: int(counts_after_clean[label]) for label in ACTION_CLASSES},
        },
        "source_quality": source_quality,
        "holdout_latest_real_source": {
            "enabled": bool(args.holdout_latest_real_source),
            "rows": int(holdout_rows),
            "path": holdout_path,
            "require_two_real_batches_for_holdout": bool(args.require_two_real_batches_for_holdout),
            "holdout_per_class": int(args.holdout_per_class),
            "holdout_min_total": int(args.holdout_min_total),
            "holdout_min_sources": int(min_sources if args.holdout_latest_real_source else args.holdout_min_sources),
            "qa": holdout_qa,
        },
        "feature_columns": feature_cols,
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "total": int(len(train_df) + len(val_df) + len(test_df)),
        },
    }

    os.makedirs(args.out_dir, exist_ok=True)
    metadata_path = os.path.join(args.out_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved processed datasets:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val:   {val_path} ({len(val_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")
    print(f"  Meta:  {metadata_path}")
    if holdout_path:
        print(f"  Fresh real eval: {holdout_path} ({holdout_rows} rows)")


if __name__ == "__main__":
    main()
