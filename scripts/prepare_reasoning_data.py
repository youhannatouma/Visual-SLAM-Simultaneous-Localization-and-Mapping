import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


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
    return parser.parse_args()


def load_raw_frames(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files matched pattern: {pattern}")

    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["__source_file"] = os.path.basename(path)
        frames.append(frame)

    return pd.concat(frames, ignore_index=True), files


def validate_and_clean(df):
    if "label" not in df.columns:
        raise ValueError("Input CSV must include a 'label' column")

    feature_cols = [c for c in df.columns if c.startswith("f")]
    if not feature_cols:
        raise ValueError("Input CSV must include feature columns named f0..fN")

    # Stable numeric order for feature columns: f0, f1, ...
    feature_cols = sorted(feature_cols, key=lambda c: int(c[1:]) if c[1:].isdigit() else c)

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

    before_dedup = len(working)
    working = working.drop_duplicates(subset=required_cols, keep="first")
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


def main():
    args = parse_args()

    merged_df, files = load_raw_frames(args.input_glob)
    clean_df, feature_cols, clean_summary = validate_and_clean(merged_df)
    counts_after_clean = enforce_minimum_per_class(clean_df, args.min_per_class)

    print(f"Loaded {len(files)} file(s)")
    print_distribution(clean_df, "Class distribution before balancing")

    balanced_df = balance_classes(clean_df, args.balance, args.seed)
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


if __name__ == "__main__":
    main()
