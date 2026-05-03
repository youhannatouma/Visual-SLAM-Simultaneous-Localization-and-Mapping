import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning import ReasoningEngine

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
ALLOWED_LABELS = {"AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build reasoning dataset CSV from image/video media")
    parser.add_argument("--media-dir", required=True, help="Directory containing images/videos")
    parser.add_argument("--out-csv", default="", help="Output CSV path (default: data/raw/media_labeled_<ts>.csv)")
    parser.add_argument("--video-stride", type=int, default=10, help="Sample one frame every N frames from videos")
    parser.add_argument(
        "--name-contains",
        default="",
        help="Optional case-insensitive filename substring filter (e.g., 'mixkit_table_')",
    )
    parser.add_argument(
        "--review-sample-ratio",
        type=float,
        default=0.12,
        help="Sample ratio for manual label review export (typically 0.10-0.15)",
    )
    parser.add_argument(
        "--review-corrections",
        default="",
        help="Optional CSV with corrections: row_id,final_label,drop_row(0/1)",
    )
    parser.add_argument(
        "--review-out",
        default="",
        help="Review export path (default: reports/media_review_<timestamp>.csv)",
    )
    parser.add_argument(
        "--correction-audit-out",
        default="",
        help="Correction audit output JSON path (default: reports/correction_audit_<review_stem>.json)",
    )
    parser.add_argument("--batch-id", default="", help="Batch identifier (e.g., batch_A_real, batch_B_real)")
    parser.add_argument(
        "--scenario",
        default="mixed",
        help="Scenario tag (clutter|low_light|occlusion|mixed|long_range)",
    )
    return parser.parse_args()


def list_media_files(media_dir):
    files = []
    for root, _, names in os.walk(media_dir):
        for name in names:
            p = Path(root) / name
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS or ext in VIDEO_EXTS:
                files.append(p)
    return sorted(files)


def infer_motion(prev_gray, gray):
    if prev_gray is None:
        return "No movement"
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0)
    dx = float(np.mean(flow[..., 0]))
    dy = float(np.mean(flow[..., 1]))
    if abs(dx) > abs(dy):
        if dx > 0.15:
            return "Moving Right"
        if dx < -0.15:
            return "Moving Left"
    else:
        if dy > 0.15:
            return "Moving Down"
        if dy < -0.15:
            return "Moving Up"
    return "No movement"


def action_from_detections(detections, frame_center, frame_area):
    people = [d for d in detections if d["label"] == "person"]
    chairs = [d for d in detections if d["label"] == "chair"]
    tables = [d for d in detections if d["label"] == "table"]

    if people:
        p = max(people, key=lambda d: d["confidence"] * d["area"])
        area_ratio = p["area"] / frame_area
        center_dx = abs(p["center"][0] - frame_center[0]) / max(1, frame_center[0])
        if area_ratio > 0.045 or center_dx < 0.35:
            return "AVOID_PERSON"

    if chairs:
        return "MOVE_TO_CHAIR"
    if tables:
        return "CHECK_TABLE"
    return "EXPLORE"


def should_review_row(detections):
    chairs = [d for d in detections if d["label"] == "chair"]
    tables = [d for d in detections if d["label"] == "table"]
    people = [d for d in detections if d["label"] == "person"]
    mixed_chair_table = bool(chairs and tables)
    weak_conf = any(d["confidence"] < 0.45 for d in detections)
    crowded = len(people) >= 2
    return mixed_chair_table or weak_conf or crowded


def detections_summary(detections):
    out = {}
    for d in detections:
        out[d["label"]] = out.get(d["label"], 0) + 1
    return out


def yolo_to_detections(result, model):
    detections = []
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        area = max(1, (x2 - x1) * (y2 - y1))
        detections.append(
            {
                "label": label,
                "confidence": conf,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "area": area,
            }
        )
    return detections


def write_rows(out_csv, rows):
    if not rows:
        raise ValueError("No rows generated from media")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    feature_count = rows[0]["feature_count"]
    header = [f"f{i}" for i in range(feature_count)] + [
        "label",
        "source_type",
        "source_file",
        "frame_index",
        "auto_label",
        "needs_review",
        "batch_id",
        "scenario",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                row["features"]
                + [
                    row["label"],
                    "real_media",
                    row["source_file"],
                    row["frame_index"],
                    row["auto_label"],
                    int(row["needs_review"]),
                    row.get("batch_id", ""),
                    row.get("scenario", "mixed"),
                ]
            )


def write_review_file(review_out, rows, sample_ratio):
    os.makedirs(os.path.dirname(review_out), exist_ok=True)
    hard_rows = [r for r in rows if r["needs_review"]]
    remaining = [r for r in rows if not r["needs_review"]]
    sample_n = max(1, int(len(rows) * sample_ratio)) if rows else 0

    rng = np.random.default_rng(42)
    sampled_rows = []
    if remaining and sample_n > 0:
        idx = rng.choice(np.arange(len(remaining)), size=min(sample_n, len(remaining)), replace=False)
        sampled_rows = [remaining[i] for i in idx]
    selected = hard_rows + sampled_rows

    with open(review_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "row_id",
                "source_file",
                "frame_index",
                "auto_label",
                "final_label",
                "drop_row",
                "needs_review",
                "detections_json",
            ]
        )
        for r in selected:
            writer.writerow(
                [
                    r["row_id"],
                    r["source_file"],
                    r["frame_index"],
                    r["auto_label"],
                    "",
                    0,
                    int(r["needs_review"]),
                    json.dumps(r["detections_summary"], ensure_ascii=True),
                ]
            )
    return len(selected)


def apply_review_corrections(rows, corrections_csv):
    if not corrections_csv:
        return rows, 0, 0, []
    corr_df = pd.read_csv(corrections_csv)
    required_cols = {"row_id", "final_label", "drop_row"}
    if not required_cols.issubset(set(corr_df.columns)):
        raise ValueError(f"Correction CSV missing required columns: {required_cols}")

    corrections = {}
    issues = []
    for _, row in corr_df.iterrows():
        row_id = int(row["row_id"])
        final_label = str(row.get("final_label", "")).strip()
        drop_row = int(row.get("drop_row", 0))
        if final_label and final_label not in ALLOWED_LABELS:
            issues.append({"row_id": row_id, "issue": f"invalid final_label={final_label}"})
            continue
        if drop_row not in (0, 1):
            issues.append({"row_id": row_id, "issue": f"invalid drop_row={drop_row}"})
            continue
        corrections[row_id] = {"final_label": final_label, "drop_row": drop_row}

    updated = []
    dropped = 0
    relabeled = 0
    for r in rows:
        info = corrections.get(r["row_id"])
        if info and info["drop_row"] == 1:
            dropped += 1
            continue
        if info and info["final_label"] in ALLOWED_LABELS:
            if r["label"] != info["final_label"]:
                relabeled += 1
            r["label"] = info["final_label"]
        updated.append(r)
    return updated, relabeled, dropped, issues


def write_correction_audit(
    *,
    path,
    rows_before,
    rows_after,
    review_file,
    corrections_file,
    sample_ratio,
    relabeled,
    dropped,
    issues,
):
    review_rows = []
    if review_file and os.path.exists(review_file):
        try:
            review_rows = pd.read_csv(review_file).to_dict(orient="records")
        except Exception:
            review_rows = []

    before_by_id = {int(r["row_id"]): r for r in rows_before}
    after_ids = {int(r["row_id"]) for r in rows_after}
    correction_rows = []
    if corrections_file and os.path.exists(corrections_file):
        try:
            correction_rows = pd.read_csv(corrections_file).to_dict(orient="records")
        except Exception:
            correction_rows = []

    relabel_by_class = {label: 0 for label in sorted(ALLOWED_LABELS)}
    override_without_detection = 0
    unresolved = []
    for row in correction_rows:
        try:
            row_id = int(row.get("row_id"))
        except Exception:
            continue
        final_label = str(row.get("final_label", "")).strip()
        try:
            drop_row = int(row.get("drop_row", 0))
        except Exception:
            drop_row = -1
        original = before_by_id.get(row_id)
        if original is None:
            unresolved.append({"row_id": row_id, "issue": "row_id_not_found_in_generated_rows"})
            continue
        if drop_row == 1:
            continue
        if final_label in ALLOWED_LABELS and final_label != original["label"]:
            relabel_by_class[final_label] += 1
            det = original.get("detections_summary", {})
            has_support = False
            if isinstance(det, dict):
                if final_label == "AVOID_PERSON":
                    has_support = int(det.get("person", 0)) > 0
                elif final_label == "MOVE_TO_CHAIR":
                    has_support = int(det.get("chair", 0)) > 0
                elif final_label == "CHECK_TABLE":
                    has_support = int(det.get("table", 0)) > 0 or int(det.get("dining table", 0)) > 0
                elif final_label == "EXPLORE":
                    has_support = False
            if not has_support:
                override_without_detection += 1

    hard_case_count = sum(1 for r in rows_before if bool(r.get("needs_review")))
    review_sample_target = int(max(1, len(rows_before) * float(sample_ratio))) if rows_before else 0
    payload = {
        "timestamp": int(time.time()),
        "review_file": review_file,
        "corrections_file": corrections_file,
        "rows_before": int(len(rows_before)),
        "rows_after": int(len(rows_after)),
        "qa_sample": {
            "review_sample_ratio_target": float(sample_ratio),
            "review_sample_target_rows": int(review_sample_target),
            "review_rows_exported": int(len(review_rows)),
            "hard_case_rows": int(hard_case_count),
            "hard_cases_plus_sampled": bool(len(review_rows) >= hard_case_count),
        },
        "correction_actions": {
            "relabeled_rows": int(relabeled),
            "dropped_rows": int(dropped),
            "unchanged_rows": int(len(rows_before) - relabeled - dropped),
            "relabeled_by_target_class": relabel_by_class,
        },
        "reviewer_override_summary": {
            "overrides_without_detection_support": int(override_without_detection),
            "overrides_allowed": True,
        },
        "unresolved_issues": unresolved + list(issues),
        "gates": {
            "schema_valid": len(issues) == 0,
            "corrections_applied": bool(corrections_file),
            "qa_sample_present": len(review_rows) > 0,
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_review_status(status_path, *, out_csv, review_file, corrections_file, status, relabeled, dropped, total_rows, issues):
    os.makedirs(os.path.dirname(status_path), exist_ok=True)
    payload = {
        "timestamp": int(time.time()),
        "output_csv": out_csv,
        "review_file": review_file,
        "corrections_file": corrections_file,
        "status": status,
        "rows_total": int(total_rows),
        "rows_relabeled": int(relabeled),
        "rows_dropped": int(dropped),
        "issues": issues,
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_coverage_report(path, rows, batch_id, scenario):
    counts = {}
    for r in rows:
        key = (r.get("batch_id", batch_id) or "unassigned", r.get("scenario", scenario) or "mixed", r["label"])
        counts[key] = counts.get(key, 0) + 1

    report = {
        "timestamp": int(time.time()),
        "batch_id": batch_id or "unassigned",
        "scenario": scenario,
        "rows_total": len(rows),
        "scenario_class_counts": [
            {"batch_id": b, "scenario": s, "label": l, "count": c}
            for (b, s, l), c in sorted(counts.items())
        ],
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def process_image(path, model, engine, batch_id, scenario):
    frame = cv2.imread(str(path))
    if frame is None:
        return []
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_center = (320, 240)
    frame_area = 640 * 480
    results = model(frame, verbose=False)
    detections = yolo_to_detections(results[0], model)
    motion = infer_motion(None, gray)
    label = action_from_detections(detections, frame_center, frame_area)
    features = engine.extract_features(detections, frame_center, frame_area, motion)
    return [
        {
            "features": features,
            "feature_count": len(features),
            "label": label,
            "auto_label": label,
            "source_file": path.name,
            "frame_index": 0,
            "needs_review": should_review_row(detections),
            "detections_summary": detections_summary(detections),
            "batch_id": batch_id,
            "scenario": scenario,
        }
    ]


def process_video(path, model, engine, stride, batch_id, scenario):
    rows = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return rows
    frame_idx = 0
    prev_gray = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_center = (320, 240)
        frame_area = 640 * 480
        results = model(frame, verbose=False)
        detections = yolo_to_detections(results[0], model)
        motion = infer_motion(prev_gray, gray)
        label = action_from_detections(detections, frame_center, frame_area)
        features = engine.extract_features(detections, frame_center, frame_area, motion)
        rows.append(
            {
                "features": features,
                "feature_count": len(features),
                "label": label,
                "auto_label": label,
                "source_file": path.name,
                "frame_index": frame_idx,
                "needs_review": should_review_row(detections),
                "detections_summary": detections_summary(detections),
                "batch_id": batch_id,
                "scenario": scenario,
            }
        )
        prev_gray = gray
    cap.release()
    return rows


def main():
    args = parse_args()
    media_dir = Path(args.media_dir).resolve()
    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    out_csv = args.out_csv or f"data/raw/media_labeled_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    review_out = args.review_out or f"reports/media_review_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    correction_audit_out = args.correction_audit_out or f"reports/correction_audit_{Path(review_out).stem}.json"
    status_out = f"reports/review_status/{Path(review_out).stem}.json"
    coverage_out = f"reports/batch_coverage_{time.strftime('%Y%m%d_%H%M%S')}.json"

    files = list_media_files(str(media_dir))
    if args.name_contains:
        needle = args.name_contains.lower()
        files = [p for p in files if needle in p.name.lower()]
    if not files:
        raise FileNotFoundError(f"No image/video files found under: {media_dir}")

    model = YOLO(str(PROJECT_ROOT / "yolov8n.pt"))
    engine = ReasoningEngine()

    all_rows = []
    for path in files:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            rows = process_image(path, model, engine, args.batch_id, args.scenario)
        else:
            rows = process_video(path, model, engine, args.video_stride, args.batch_id, args.scenario)
        all_rows.extend(rows)

    for i, row in enumerate(all_rows):
        row["row_id"] = i

    reviewed_rows, relabeled, dropped, issues = apply_review_corrections(all_rows, args.review_corrections)
    if args.review_corrections and issues:
        raise ValueError(
            "Correction validation failed. Fix correction CSV issues before preprocessing/training. "
            f"Issues found: {len(issues)}"
        )
    review_count = write_review_file(review_out, reviewed_rows, args.review_sample_ratio)
    write_coverage_report(coverage_out, reviewed_rows, args.batch_id, args.scenario)
    write_rows(out_csv, reviewed_rows)

    if args.review_corrections:
        status = "applied"
    else:
        status = "pending"

    write_review_status(
        status_out,
        out_csv=out_csv,
        review_file=review_out,
        corrections_file=args.review_corrections,
        status=status,
        relabeled=relabeled,
        dropped=dropped,
        total_rows=len(reviewed_rows),
        issues=issues,
    )
    write_correction_audit(
        path=correction_audit_out,
        rows_before=all_rows,
        rows_after=reviewed_rows,
        review_file=review_out,
        corrections_file=args.review_corrections,
        sample_ratio=args.review_sample_ratio,
        relabeled=relabeled,
        dropped=dropped,
        issues=issues,
    )

    print(f"Processed files: {len(files)}")
    print(f"Wrote rows: {len(reviewed_rows)}")
    print(f"Output CSV: {out_csv}")
    print(f"Review rows exported: {review_count}")
    print(f"Review file: {review_out}")
    print(f"Review status: {status_out}")
    print(f"Correction audit: {correction_audit_out}")
    print(f"Coverage report: {coverage_out}")
    if args.review_corrections:
        print(f"Corrections applied: relabeled={relabeled} dropped={dropped}")
    if issues:
        print(f"Correction issues: {len(issues)}")


if __name__ == "__main__":
    main()
