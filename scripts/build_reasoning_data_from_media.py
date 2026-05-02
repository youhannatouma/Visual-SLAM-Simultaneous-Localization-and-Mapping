import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
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
    header = [f"f{i}" for i in range(len(rows[0]) - 1)] + ["label"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


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
    return [features + [label]]


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
        rows.append(features + [label])
        prev_gray = gray
    cap.release()
    return rows


def main():
    args = parse_args()
    media_dir = Path(args.media_dir).resolve()
    if not media_dir.exists():
        raise FileNotFoundError(f"Media directory not found: {media_dir}")

    out_csv = args.out_csv or f"data/raw/media_labeled_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    files = list_media_files(str(media_dir))
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

    write_rows(out_csv, all_rows)
    print(f"Processed files: {len(files)}")
    print(f"Wrote rows: {len(all_rows)}")
    print(f"Output CSV: {out_csv}")


if __name__ == "__main__":
    main()
