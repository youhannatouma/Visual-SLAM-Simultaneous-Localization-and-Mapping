import argparse
import json
from pathlib import Path

import pandas as pd

FEATURE_COLS = [f"f{i}" for i in range(43)]
META_COLS = [
    "label",
    "source_type",
    "source_file",
    "frame_index",
    "auto_label",
    "needs_review",
    "batch_id",
    "scenario",
]
CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build stage-2 real holdout/training CSVs while excluding feature+label "
            "duplicates already present in active raw data."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--holdout-per-class",
        type=int,
        default=12,
        help="Rows per class for holdout (default: 12).",
    )
    parser.add_argument(
        "--train-move-to-chair",
        type=int,
        default=60,
        help="Rows for MOVE_TO_CHAIR in training augmentation.",
    )
    parser.add_argument(
        "--train-check-table",
        type=int,
        default=40,
        help="Rows for CHECK_TABLE in training augmentation.",
    )
    parser.add_argument(
        "--train-avoid-person",
        type=int,
        default=20,
        help="Rows for AVOID_PERSON in training augmentation.",
    )
    parser.add_argument(
        "--train-explore",
        type=int,
        default=20,
        help="Rows for EXPLORE in training augmentation.",
    )
    parser.add_argument(
        "--train-hard-negative-target",
        default="",
        help=(
            "Optional target class; when set, training rows are drawn from hard negatives "
            "where auto_label equals this target but label does not."
        ),
    )
    parser.add_argument(
        "--train-hard-negative-only",
        action="store_true",
        help="Only use hard-negative pool for training augmentation (no fallback to all rows).",
    )
    parser.add_argument(
        "--train-hard-negative-require-review",
        action="store_true",
        help="Require needs_review=1 when building the hard-negative pool.",
    )
    parser.add_argument(
        "--metrics-path",
        default="reports/metrics.json",
        help="Metrics JSON used to infer the worst fresh-real class when no hard-negative target is set.",
    )
    return parser.parse_args()


def infer_worst_fresh_real_class(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return ""
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    per_class = payload.get("fresh_real_eval", {}).get("per_class", {})
    if not isinstance(per_class, dict):
        return ""
    ranked = []
    for label in CLASSES:
        row = per_class.get(label, {})
        try:
            score = float(row.get("f1"))
        except Exception:
            continue
        ranked.append((score, label))
    if not ranked:
        return ""
    ranked.sort(key=lambda item: (item[0], item[1]))
    return ranked[0][1]


def load_and_upgrade(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    feature_cols = sorted(
        [c for c in df.columns if c.startswith("f") and c[1:].isdigit()],
        key=lambda c: int(c[1:]),
    )
    for idx in range(len(feature_cols), len(FEATURE_COLS)):
        df[f"f{idx}"] = 0.0
    for col in META_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    ordered = FEATURE_COLS + META_COLS
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def signature(row: pd.Series) -> tuple:
    values = tuple(round(float(row[c]), 6) for c in FEATURE_COLS)
    return values + (str(row["label"]),)


def unique_take(df: pd.DataFrame, label: str, count: int, used: set) -> tuple[pd.DataFrame, set]:
    picked = []
    for _, row in df[df["label"] == label].iterrows():
        sig = signature(row)
        if sig in used:
            continue
        used.add(sig)
        picked.append(row)
        if len(picked) >= count:
            break
    return pd.DataFrame(picked), used


def normalize_label_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def build_hard_negative_pool(df: pd.DataFrame, target: str, require_review: bool) -> pd.DataFrame:
    labels = normalize_label_series(df["label"])
    auto_labels = normalize_label_series(df.get("auto_label", pd.Series([""] * len(df))))
    mask = (labels != target) & (auto_labels == target)
    if require_review:
        needs_review = df.get("needs_review", pd.Series([0] * len(df))).fillna(0).astype(int)
        mask &= needs_review.eq(1)
    return df[mask].copy()


def main():
    args = parse_args()
    root = Path(args.repo_root).resolve()

    raw_dir = root / "data" / "raw"
    archive_dir = root / "data" / "raw_archive"
    staging_dir = root / "data" / "raw_staging_20260506"

    holdout_path = raw_dir / "zz_fresh_real_holdout_20260506.csv"
    train_path = raw_dir / "media_labeled_stage2_train_refix_20260506.csv"

    active_raw_paths = sorted(raw_dir.glob("*.csv"))
    active_raw_paths = [p for p in active_raw_paths if p.name not in {holdout_path.name, train_path.name}]

    used_signatures = set()
    for path in active_raw_paths:
        df = load_and_upgrade(path)
        for _, row in df.iterrows():
            used_signatures.add(signature(row))

    candidate_paths = [
        archive_dir / "media_labeled_batch_C_real.csv",
        archive_dir / "media_table_override_20260503_185656.csv",
        staging_dir / "media_labeled_batch_E_real.csv",
        staging_dir / "media_labeled_batch_F_real.csv",
        staging_dir / "media_labeled_batch_G_real.csv",
        staging_dir / "media_labeled_batch_H_real.csv",
    ]
    candidates = [load_and_upgrade(p) for p in candidate_paths if p.exists()]
    candidate_df = pd.concat(candidates, ignore_index=True)
    candidate_df["source_type"] = "real_media"

    hard_negative_pool = None
    hard_negative_target = args.train_hard_negative_target.strip().upper()
    if not hard_negative_target:
        inferred_target = infer_worst_fresh_real_class(root / args.metrics_path)
        if inferred_target:
            hard_negative_target = inferred_target
            print(f"Inferred hard-negative target from fresh-real metrics: {hard_negative_target}")
    if hard_negative_target:
        if hard_negative_target not in CLASSES:
            raise ValueError(
                "Invalid --train-hard-negative-target. "
                f"Expected one of {CLASSES}, got: {hard_negative_target}"
            )
        hard_negative_pool = build_hard_negative_pool(
            candidate_df,
            hard_negative_target,
            args.train_hard_negative_require_review,
        )
        print(
            "Hard-negative pool:",
            len(hard_negative_pool),
            hard_negative_pool["label"].value_counts().to_dict(),
        )

    hold_parts = []
    for label in CLASSES:
        part, used_signatures = unique_take(
            candidate_df,
            label,
            args.holdout_per_class,
            used_signatures,
        )
        hold_parts.append(part)

    holdout_df = pd.concat(hold_parts, ignore_index=True)
    holdout_df["batch_id"] = "batch_stage2_holdout_refix_20260506"
    holdout_df.to_csv(holdout_path, index=False)

    train_targets = {
        "MOVE_TO_CHAIR": args.train_move_to_chair,
        "CHECK_TABLE": args.train_check_table,
        "AVOID_PERSON": args.train_avoid_person,
        "EXPLORE": args.train_explore,
    }
    train_parts = []
    for label, count in train_targets.items():
        if hard_negative_pool is None:
            part, used_signatures = unique_take(candidate_df, label, count, used_signatures)
            train_parts.append(part)
            continue

        part, used_signatures = unique_take(hard_negative_pool, label, count, used_signatures)
        if len(part) < count:
            if args.train_hard_negative_only:
                raise ValueError(
                    "Hard-negative pool too small for training augmentation. "
                    f"Label={label} requested={count} got={len(part)}"
                )
            remaining = count - len(part)
            extra, used_signatures = unique_take(candidate_df, label, remaining, used_signatures)
            if not extra.empty:
                part = pd.concat([part, extra], ignore_index=True)
        train_parts.append(part)

    train_df = pd.concat(train_parts, ignore_index=True)
    train_df["batch_id"] = "batch_stage2_train_refix_20260506"
    train_df.to_csv(train_path, index=False)

    print("Wrote holdout:", holdout_path, len(holdout_df), holdout_df["label"].value_counts().to_dict())
    print("Wrote train augmentation:", train_path, len(train_df), train_df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
