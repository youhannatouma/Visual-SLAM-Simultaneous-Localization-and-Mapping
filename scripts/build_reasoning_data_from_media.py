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
        return rows, 0, 0
    corr_df = pd.read_csv(corrections_csv)
    corrections = {}
    for _, row in corr_df.iterrows():
        row_id = int(row["row_id"])
        final_label = str(row.get("final_label", "")).strip()
        drop_row = int(row.get("drop_row", 0))
        corrections[row_id] = {"final_label": final_label, "drop_row": drop_row}

    updated = []
    dropped = 0
    relabeled = 0
    for r in rows:
        info = corrections.get(r["row_id"])
        if info and info["drop_row"] == 1:
            dropped += 1
            continue
        if info and info["final_label"] in {"AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"}:
            if r["label"] != info["final_label"]:
                relabeled += 1
            r["label"] = info["final_label"]
        updated.append(r)
    return updated, relabeled, dropped


def process_image(path, model, engine):
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
        }
    ]


def process_video(path, model, engine, stride):
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
            rows = process_image(path, model, engine)
        else:
            rows = process_video(path, model, engine, args.video_stride)
        all_rows.extend(rows)

    for i, row in enumerate(all_rows):
        row["row_id"] = i

    reviewed_rows, relabeled, dropped = apply_review_corrections(all_rows, args.review_corrections)
    review_count = write_review_file(review_out, reviewed_rows, args.review_sample_ratio)

    write_rows(out_csv, reviewed_rows)
    print(f"Processed files: {len(files)}")
    print(f"Wrote rows: {len(reviewed_rows)}")
    print(f"Output CSV: {out_csv}")
    print(f"Review rows exported: {review_count}")
    print(f"Review file: {review_out}")
    if args.review_corrections:
        print(f"Corrections applied: relabeled={relabeled} dropped={dropped}")


if __name__ == "__main__":
    main()
