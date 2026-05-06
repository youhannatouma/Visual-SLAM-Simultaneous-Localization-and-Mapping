import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from build_stage2_real_sets import FEATURE_COLS, META_COLS, load_and_upgrade


ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a reviewed-only MOVE_TO_CHAIR recovery pool from archive sources"
    )
    parser.add_argument(
        "--source-glob",
        default="",
        help="Glob for source CSVs (e.g., data/raw_archive/*.csv).",
    )
    parser.add_argument(
        "--source-files",
        nargs="*",
        default=[],
        help="Explicit source CSV paths (space-separated).",
    )
    parser.add_argument(
        "--source-list",
        default="",
        help="Optional text file listing source CSVs (one per line).",
    )
    parser.add_argument(
        "--review-status-dir",
        default="reports/review_status",
        help="Directory containing review status JSON files.",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow sources without applied review status.",
    )
    parser.add_argument(
        "--mark-unreviewed-applied",
        action="store_true",
        help="Write applied review status entries for unreviewed sources that are used.",
    )
    parser.add_argument(
        "--target-label",
        default="MOVE_TO_CHAIR",
        help="Target label to keep in the recovery pool.",
    )
    parser.add_argument(
        "--needs-review-only",
        action="store_true",
        help="Keep only rows where needs_review == 1 after label filtering.",
    )
    parser.add_argument(
        "--curation-csv",
        default="",
        help="Optional curation CSV with source_file, frame_index, and keep/drop columns.",
    )
    parser.add_argument(
        "--exclude-existing-glob",
        default="data/raw/*.csv",
        help="Glob for existing raw CSVs to exclude from the recovery pool.",
    )
    parser.add_argument(
        "--export-candidates",
        default="",
        help="Optional CSV path to export candidates for manual curation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum rows to keep in the recovery pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling down to the limit.",
    )
    parser.add_argument(
        "--batch-id",
        default="",
        help="Batch id to stamp on output rows.",
    )
    parser.add_argument(
        "--scenario",
        default="",
        help="Optional scenario override for output rows.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Output CSV path (default: data/raw/move_recovery_pool_<date>.csv).",
    )
    parser.add_argument(
        "--write-review-status",
        action="store_true",
        help="Write an applied review status file for the output CSV.",
    )
    return parser.parse_args()


def load_source_list(path):
    sources = []
    if not path:
        return sources
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            sources.append(item)
    return sources


def load_reviewed_sources(status_dir):
    reviewed = set()
    status_path = Path(status_dir)
    if not status_path.exists():
        return reviewed
    for path in status_path.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(payload.get("status", "")).lower()
        output_csv = str(payload.get("output_csv", ""))
        if status == "applied" and output_csv:
            reviewed.add(os.path.basename(output_csv))
    return reviewed


def normalize_label(series):
    return series.fillna("").astype(str).str.strip().str.upper()


def ensure_feature_validity(df):
    working = df.copy()
    for col in FEATURE_COLS:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    valid_mask = working[FEATURE_COLS].notna().all(axis=1)
    valid_mask &= np.isfinite(working[FEATURE_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    return working.loc[valid_mask].copy()


def apply_curation_filter(df, curation_csv):
    if not curation_csv:
        return df
    curation = pd.read_csv(curation_csv)
    if "source_file" not in curation.columns or "frame_index" not in curation.columns:
        raise ValueError("Curation CSV must include source_file and frame_index columns")

    curation["frame_index"] = pd.to_numeric(curation["frame_index"], errors="coerce")
    curation = curation.dropna(subset=["frame_index"]).copy()
    curation["frame_index"] = curation["frame_index"].astype(int)

    keep_mask = None
    if "keep" in curation.columns:
        keep_mask = curation["keep"].fillna(0).astype(int) == 1
    elif "drop_row" in curation.columns:
        keep_mask = curation["drop_row"].fillna(0).astype(int) == 0
    else:
        keep_mask = pd.Series([True] * len(curation), index=curation.index)

    curation = curation.loc[keep_mask].copy()
    if curation.empty:
        return df.iloc[0:0].copy()

    if "source_csv" in curation.columns:
        keys = {
            (
                str(row["source_csv"]).strip(),
                str(row["source_file"]).strip(),
                int(row["frame_index"]),
            )
            for _, row in curation.iterrows()
        }
        return df[
            df.apply(
                lambda r: (
                    str(r.get("__source_csv", "")).strip(),
                    str(r.get("source_file", "")).strip(),
                    int(r.get("frame_index", -1)),
                )
                in keys,
                axis=1,
            )
        ].copy()

    keys = {
        (str(row["source_file"]).strip(), int(row["frame_index"]))
        for _, row in curation.iterrows()
        if pd.notna(row["frame_index"])
    }
    return df[
        df.apply(
            lambda r: (
                str(r.get("source_file", "")).strip(),
                int(r.get("frame_index", -1)) if pd.notna(r.get("frame_index", -1)) else -1,
            )
            in keys,
            axis=1,
        )
    ].copy()


def export_candidates(df, out_path):
    cols = [
        "__source_csv",
        "source_file",
        "frame_index",
        "label",
        "auto_label",
        "needs_review",
        "batch_id",
        "scenario",
    ]
    out_df = df[[c for c in cols if c in df.columns]].copy()
    out_df["keep"] = 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)


def write_review_status(status_dir, out_csv, total_rows):
    status_path = Path(status_dir)
    status_path.mkdir(parents=True, exist_ok=True)
    status_file = status_path / f"{Path(out_csv).stem}.json"
    payload = {
        "timestamp": int(time.time()),
        "output_csv": str(Path(out_csv).resolve()),
        "review_file": "",
        "corrections_file": "",
        "status": "applied",
        "rows_total": int(total_rows),
        "rows_relabeled": 0,
        "rows_dropped": 0,
        "issues": [],
    }
    status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def signature_row(row):
    values = tuple(round(float(row[c]), 6) for c in FEATURE_COLS)
    return values + (str(row["label"]),)


def collect_existing_signatures(existing_glob, ignore_paths=None):
    if not existing_glob:
        return set()
    ignore = {str(Path(p).resolve()) for p in (ignore_paths or [])}
    signatures = set()
    paths = sorted(glob.glob(existing_glob))
    for path in paths:
        if str(Path(path).resolve()) in ignore:
            continue
        try:
            df = load_and_upgrade(Path(path))
        except Exception:
            continue
        if "label" not in df.columns:
            continue
        df["label"] = normalize_label(df["label"])
        df = ensure_feature_validity(df)
        for _, row in df.iterrows():
            try:
                signatures.add(signature_row(row))
            except Exception:
                continue
    return signatures


def drop_existing_rows(df, signatures):
    if not signatures:
        return df
    keep_mask = []
    for _, row in df.iterrows():
        try:
            sig = signature_row(row)
        except Exception:
            sig = None
        keep_mask.append(sig not in signatures)
    return df.loc[keep_mask].copy()


def main():
    args = parse_args()

    sources = []
    if args.source_glob:
        sources.extend(glob.glob(args.source_glob))
    sources.extend(args.source_files)
    sources.extend(load_source_list(args.source_list))
    sources = sorted({str(Path(s).resolve()) for s in sources})
    if not sources:
        raise ValueError("No sources provided. Use --source-glob or --source-files.")

    reviewed = load_reviewed_sources(args.review_status_dir)
    if reviewed and not args.allow_unreviewed:
        sources = [s for s in sources if os.path.basename(s) in reviewed]

    if not sources:
        raise ValueError("No reviewed sources matched the provided inputs.")

    frames = []
    source_rows = {}
    for path in sources:
        df = load_and_upgrade(Path(path))
        df["__source_csv"] = os.path.basename(path)
        frames.append(df)
        source_rows[os.path.basename(path)] = len(df)

    if args.allow_unreviewed and args.mark_unreviewed_applied:
        for path in sources:
            base = os.path.basename(path)
            if base in reviewed:
                continue
            write_review_status(args.review_status_dir, path, source_rows.get(base, 0))

    merged = pd.concat(frames, ignore_index=True)

    before_validity = len(merged)
    merged = ensure_feature_validity(merged)
    after_validity = len(merged)

    if "label" not in merged.columns:
        raise ValueError("Input CSVs must include a label column")

    target_label = str(args.target_label).strip().upper()
    if target_label not in ACTION_CLASSES:
        raise ValueError(f"Unsupported target label: {target_label}")

    merged["label"] = normalize_label(merged["label"])
    merged = merged[merged["label"] == target_label].copy()
    if args.needs_review_only:
        if "needs_review" not in merged.columns:
            merged["needs_review"] = 0
        needs_review = pd.to_numeric(merged["needs_review"], errors="coerce").fillna(0).astype(int)
        merged = merged.loc[needs_review.eq(1)].copy()

    out_csv = args.out_csv or f"data/raw/move_recovery_pool_{time.strftime('%Y%m%d')}.csv"
    existing_signatures = collect_existing_signatures(
        args.exclude_existing_glob,
        ignore_paths=[out_csv],
    )
    if existing_signatures:
        before_existing = len(merged)
        merged = drop_existing_rows(merged, existing_signatures)
        print("Dropped existing raw duplicates:", before_existing - len(merged))

    merged = apply_curation_filter(merged, args.curation_csv)

    if args.export_candidates:
        export_candidates(merged, args.export_candidates)

    if merged.empty:
        raise ValueError("No rows remain after filtering. Check review coverage and curation.")

    rng = np.random.default_rng(args.seed)
    if args.limit > 0 and len(merged) > args.limit:
        idx = rng.choice(np.arange(len(merged)), size=args.limit, replace=False)
        merged = merged.iloc[idx].copy()

    if "source_type" not in merged.columns:
        merged["source_type"] = "real_media"
    else:
        merged["source_type"] = merged["source_type"].fillna("real_media")

    if "auto_label" not in merged.columns:
        merged["auto_label"] = merged["label"]
    else:
        merged["auto_label"] = merged["auto_label"].fillna(merged["label"])

    if "needs_review" not in merged.columns:
        merged["needs_review"] = 0

    today = time.strftime("%Y%m%d")
    batch_id = args.batch_id or f"batch_move_recovery_{today}"
    merged["batch_id"] = batch_id
    if args.scenario:
        merged["scenario"] = args.scenario

    out_csv = args.out_csv or f"data/raw/move_recovery_pool_{today}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ordered = FEATURE_COLS + META_COLS
    extras = [c for c in merged.columns if c not in ordered]
    merged[ordered + extras].to_csv(out_csv, index=False)

    if args.write_review_status:
        write_review_status(args.review_status_dir, out_csv, len(merged))

    print("Reviewed sources:", len(sources))
    print("Rows before validity:", before_validity)
    print("Rows after validity:", after_validity)
    print("Rows after label filter:", len(merged))
    print("Output:", out_csv, len(merged))


if __name__ == "__main__":
    main()
