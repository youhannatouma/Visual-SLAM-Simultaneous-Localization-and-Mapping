#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a row-level correction shortlist and capture plan from misclassification CSVs"
    )
    parser.add_argument(
        "--misclass-csv",
        default="reports/fresh_real_misclassifications_cycle_20260514_v2.csv",
        help="Misclassification CSV generated from fresh-real eval",
    )
    parser.add_argument(
        "--raw-glob",
        default="data/raw/*.csv",
        help="Raw dataset glob used to resolve source rows",
    )
    parser.add_argument(
        "--out-shortlist",
        default="reports/correction_shortlist_cycle_20260514.csv",
        help="Output CSV for concrete row-level review shortlist",
    )
    parser.add_argument(
        "--out-plan",
        default="reports/capture/TARGETED_CAPTURE_PLAN_CYCLE_20260514.md",
        help="Output markdown plan for next targeted capture/review batch",
    )
    return parser.parse_args()


def load_raw_lookup(raw_glob):
    rows = []
    for path in sorted(glob.glob(raw_glob)):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty or "label" not in df.columns:
            continue
        df = df.copy()
        df["__raw_csv"] = path
        df["__row_index"] = df.index.astype(int)
        for col in ("batch_id", "source_file", "frame_index", "scenario", "auto_label", "needs_review"):
            if col not in df.columns:
                df[col] = ""
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    merged = pd.concat(rows, ignore_index=True)
    merged["batch_id"] = merged["batch_id"].fillna("").astype(str)
    merged["source_file"] = merged["source_file"].fillna("").astype(str)
    merged["scenario"] = merged["scenario"].fillna("").astype(str)
    merged["frame_index"] = pd.to_numeric(merged["frame_index"], errors="coerce")
    return merged


def recommend_action(true_label, pred_label):
    pair = (str(true_label), str(pred_label))
    if pair in {
        ("MOVE_TO_CHAIR", "EXPLORE"),
        ("CHECK_TABLE", "EXPLORE"),
    }:
        return "review_or_capture_harder_positive"
    if pair in {
        ("EXPLORE", "MOVE_TO_CHAIR"),
        ("EXPLORE", "CHECK_TABLE"),
    }:
        return "review_or_capture_cleaner_negative"
    return "review_row"


def priority_score(true_label, pred_label):
    pair = (str(true_label), str(pred_label))
    if pair == ("CHECK_TABLE", "EXPLORE"):
        return 100
    if pair == ("MOVE_TO_CHAIR", "EXPLORE"):
        return 95
    if pair[0] == "EXPLORE":
        return 80
    return 60


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def build_shortlist(misclass_df, raw_df):
    records = []
    for _, row in misclass_df.iterrows():
        batch_id = clean_text(row.get("batch_id", ""))
        source_file = clean_text(row.get("source_file", ""))
        scenario = clean_text(row.get("scenario", ""))
        frame_index = pd.to_numeric(pd.Series([row.get("frame_index")]), errors="coerce").iloc[0]
        true_label = str(row.get("true_label", "") or "")
        pred_label = str(row.get("pred_label", "") or "")
        has_locator = bool(batch_id or source_file or scenario or pd.notna(frame_index))

        if not has_locator:
            records.append(
                {
                    "priority": priority_score(true_label, pred_label),
                    "recommended_action": recommend_action(true_label, pred_label),
                    "match_status": "unresolved",
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "batch_id": batch_id,
                    "scenario": scenario,
                    "source_file": source_file,
                    "frame_index": row.get("frame_index", ""),
                    "raw_csv": "",
                    "raw_row_index": "",
                    "auto_label": "",
                    "needs_review": "",
                }
            )
            continue

        matches = raw_df.copy()
        if batch_id:
            matches = matches[matches["batch_id"].eq(batch_id)]
        if source_file:
            matches = matches[matches["source_file"].eq(source_file)]
        if pd.notna(frame_index):
            matches = matches[matches["frame_index"].eq(frame_index)]
        if true_label:
            matches = matches[matches["label"].astype(str).eq(true_label)]

        if matches.empty:
            records.append(
                {
                    "priority": priority_score(true_label, pred_label),
                    "recommended_action": recommend_action(true_label, pred_label),
                    "match_status": "unresolved",
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "batch_id": batch_id,
                    "scenario": scenario,
                    "source_file": source_file,
                    "frame_index": row.get("frame_index", ""),
                    "raw_csv": "",
                    "raw_row_index": "",
                    "auto_label": "",
                    "needs_review": "",
                }
            )
            continue

        match = matches.iloc[0]
        needs_review = match.get("needs_review", "")
        if pd.isna(needs_review) or str(needs_review).strip() == "":
            needs_review_value = ""
        else:
            needs_review_value = int(float(needs_review))
        records.append(
            {
                "priority": priority_score(true_label, pred_label),
                "recommended_action": recommend_action(true_label, pred_label),
                "match_status": "matched",
                "true_label": true_label,
                "pred_label": pred_label,
                "batch_id": batch_id or str(match.get("batch_id", "")),
                "scenario": scenario or str(match.get("scenario", "")),
                "source_file": source_file or str(match.get("source_file", "")),
                "frame_index": int(match["frame_index"]) if pd.notna(match["frame_index"]) else "",
                "raw_csv": str(match["__raw_csv"]),
                "raw_row_index": int(match["__row_index"]),
                "auto_label": str(match.get("auto_label", "")),
                "needs_review": needs_review_value,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        return out
    out = out.sort_values(
        by=["priority", "batch_id", "source_file", "frame_index"],
        ascending=[False, True, True, True],
        kind="stable",
    ).reset_index(drop=True)
    return out


def write_plan(shortlist_df, out_path):
    lines = []
    lines.append("# Targeted Capture And Review Plan")
    lines.append("")
    lines.append("## Priority Failure Modes")
    lines.append("")
    if shortlist_df.empty:
        lines.append("- No misclassifications were available to summarize.")
    else:
        pair_counts = shortlist_df.groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
        for (true_label, pred_label), count in pair_counts.items():
            lines.append(f"- `{true_label} -> {pred_label}`: `{int(count)}` rows")
        lines.append("")
        lines.append("## Batch Targets")
        lines.append("")
        batch_counts = shortlist_df.groupby(["batch_id", "true_label", "pred_label"]).size().sort_values(ascending=False)
        for (batch_id, true_label, pred_label), count in batch_counts.head(20).items():
            batch_label = batch_id if batch_id else "unresolved_source"
            lines.append(f"- `{batch_label}`: `{true_label} -> {pred_label}` x `{int(count)}`")
        lines.append("")
        lines.append("## Review Actions")
        lines.append("")
        lines.append("- Re-review all `CHECK_TABLE -> EXPLORE` rows first; these are the highest-value regressions.")
        lines.append("- Re-review all `MOVE_TO_CHAIR -> EXPLORE` rows next, especially clutter and transition scenes.")
        lines.append("- For `EXPLORE -> MOVE_TO_CHAIR/CHECK_TABLE`, add negative examples with visible furniture but no navigation intent.")
        lines.append("- For unresolved rows, capture or relabel replacement examples from the same scenario bucket.")
        lines.append("")
        lines.append("## Capture Focus")
        lines.append("")
        lines.append("- `low_light`: chair/table visible but partially occluded, maintain clear intent labels.")
        lines.append("- `clutter`: multiple objects present, include positive chair-approach and table-interaction frames.")
        lines.append("- `transitions`: approach sequences where intent should stay `MOVE_TO_CHAIR` before close-up.")
        lines.append("- `mixed`: explicit negative `EXPLORE` frames with tables/chairs present but no target behavior.")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    misclass_df = pd.read_csv(args.misclass_csv)
    raw_df = load_raw_lookup(args.raw_glob)
    shortlist_df = build_shortlist(misclass_df, raw_df)

    out_csv = Path(args.out_shortlist)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    shortlist_df.to_csv(out_csv, index=False)
    write_plan(shortlist_df, args.out_plan)

    print(f"Shortlist rows: {len(shortlist_df)}")
    print(f"Shortlist CSV: {out_csv}")
    print(f"Plan MD: {args.out_plan}")


if __name__ == "__main__":
    main()
