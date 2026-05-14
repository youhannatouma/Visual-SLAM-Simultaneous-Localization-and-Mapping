import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reasoning import ACTION_CLASSES, ReasoningEngine

FRAME_W = 640
FRAME_H = 480
FRAME_CENTER = (FRAME_W // 2, FRAME_H // 2)
FRAME_AREA = FRAME_W * FRAME_H


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate large simulated reasoning dataset with people/chairs/tables scenes"
    )
    parser.add_argument("--out-dir", default="data/raw", help="Output directory for generated CSV")
    parser.add_argument("--rows-per-class", type=int, default=1000, help="Rows per action class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def rand_bbox_around(rng, cx, cy, base_w, base_h, jitter=0.45):
    # Aggressive aspect ratio jitter to simulate different angles (front, side, back)
    aspect_jitter = rng.uniform(0.6, 1.4)
    w = int(max(12, base_w * (1 + rng.uniform(-jitter, jitter)) * aspect_jitter))
    h = int(max(12, base_h * (1 + rng.uniform(-jitter, jitter)) / aspect_jitter))
    
    # Center jitter to simulate imperfect YOLO detections
    cx += rng.integers(-int(w * 0.1), int(w * 0.1) + 1)
    cy += rng.integers(-int(h * 0.1), int(h * 0.1) + 1)

    x1 = int(np.clip(cx - w // 2, 0, FRAME_W - 2))
    y1 = int(np.clip(cy - h // 2, 0, FRAME_H - 2))
    x2 = int(np.clip(x1 + w, x1 + 1, FRAME_W - 1))
    y2 = int(np.clip(y1 + h, y1 + 1, FRAME_H - 1))
    area = max(1, (x2 - x1) * (y2 - y1))
    return (x1, y1, x2, y2), area


def make_det(label, conf, center, bbox, area):
    return {
        "label": label,
        "confidence": float(conf),
        "center": (int(center[0]), int(center[1])),
        "bbox": bbox,
        "area": int(area),
    }


def scene_avoid_person(rng):
    detections = []
    px = int(np.clip(rng.normal(FRAME_CENTER[0], 70), 60, FRAME_W - 60))
    py = int(np.clip(rng.normal(FRAME_CENTER[1], 60), 60, FRAME_H - 60))
    person_bbox, person_area = rand_bbox_around(rng, px, py, base_w=120, base_h=230, jitter=0.5)
    detections.append(make_det("person", rng.uniform(0.75, 0.99), (px, py), person_bbox, person_area))

    if rng.random() < 0.7:
        cx = int(np.clip(rng.normal(FRAME_CENTER[0] - 140, 120), 40, FRAME_W - 40))
        cy = int(np.clip(rng.normal(FRAME_CENTER[1] + 40, 100), 40, FRAME_H - 40))
        b, a = rand_bbox_around(rng, cx, cy, 110, 120, 0.55)
        detections.append(make_det("chair", rng.uniform(0.45, 0.9), (cx, cy), b, a))

    if rng.random() < 0.6:
        tx = int(np.clip(rng.normal(FRAME_CENTER[0] + 110, 110), 50, FRAME_W - 50))
        ty = int(np.clip(rng.normal(FRAME_CENTER[1] + 40, 80), 50, FRAME_H - 50))
        b, a = rand_bbox_around(rng, tx, ty, 180, 95, 0.3)
        detections.append(make_det("table", rng.uniform(0.45, 0.92), (tx, ty), b, a))

    return detections, rng.choice(["Moving Left", "Moving Right", "No movement"], p=[0.38, 0.38, 0.24])


def scene_move_to_chair(rng):
    detections = []
    cx = int(np.clip(rng.normal(FRAME_CENTER[0], 90), 50, FRAME_W - 50))
    cy = int(np.clip(rng.normal(FRAME_CENTER[1] + 20, 80), 50, FRAME_H - 50))
    chair_bbox, chair_area = rand_bbox_around(rng, cx, cy, base_w=130, base_h=140, jitter=0.55)
    detections.append(make_det("chair", rng.uniform(0.72, 0.99), (cx, cy), chair_bbox, chair_area))

    if rng.random() < 0.4:
        px = int(np.clip(rng.normal(FRAME_CENTER[0] + 180, 80), 50, FRAME_W - 50))
        py = int(np.clip(rng.normal(FRAME_CENTER[1] - 20, 90), 50, FRAME_H - 50))
        b, a = rand_bbox_around(rng, px, py, 90, 180, 0.35)
        detections.append(make_det("person", rng.uniform(0.35, 0.8), (px, py), b, a))

    if rng.random() < 0.65:
        tx = int(np.clip(rng.normal(FRAME_CENTER[0] - 150, 110), 50, FRAME_W - 50))
        ty = int(np.clip(rng.normal(FRAME_CENTER[1] + 60, 70), 50, FRAME_H - 50))
        b, a = rand_bbox_around(rng, tx, ty, 175, 90, 0.35)
        detections.append(make_det("table", rng.uniform(0.4, 0.85), (tx, ty), b, a))

    return detections, rng.choice(["Moving Up", "Moving Down", "No movement"], p=[0.36, 0.36, 0.28])


def scene_check_table(rng):
    detections = []
    tx = int(np.clip(rng.normal(FRAME_CENTER[0], 110), 50, FRAME_W - 50))
    ty = int(np.clip(rng.normal(FRAME_CENTER[1] + 40, 70), 50, FRAME_H - 50))
    table_bbox, table_area = rand_bbox_around(rng, tx, ty, base_w=210, base_h=110, jitter=0.35)
    detections.append(make_det("table", rng.uniform(0.72, 0.99), (tx, ty), table_bbox, table_area))

    if rng.random() < 0.65:
        cx = int(np.clip(rng.normal(FRAME_CENTER[0] - 130, 95), 50, FRAME_W - 50))
        cy = int(np.clip(rng.normal(FRAME_CENTER[1] + 10, 80), 50, FRAME_H - 50))
        b, a = rand_bbox_around(rng, cx, cy, 120, 130, 0.35)
        detections.append(make_det("chair", rng.uniform(0.45, 0.9), (cx, cy), b, a))

    if rng.random() < 0.35:
        px = int(np.clip(rng.normal(FRAME_CENTER[0] + 160, 90), 50, FRAME_W - 50))
        py = int(np.clip(rng.normal(FRAME_CENTER[1], 90), 50, FRAME_H - 50))
        b, a = rand_bbox_around(rng, px, py, 95, 175, 0.35)
        detections.append(make_det("person", rng.uniform(0.35, 0.8), (px, py), b, a))

    return detections, rng.choice(["Moving Left", "Moving Right", "No movement"], p=[0.34, 0.34, 0.32])


def scene_explore(rng):
    detections = []
    k = rng.integers(0, 3)
    labels = ["chair", "table", "sofa", "tv", "other"]
    for _ in range(k):
        label = rng.choice(labels)
        x = rng.integers(40, FRAME_W - 40)
        y = rng.integers(40, FRAME_H - 40)
        b, a = rand_bbox_around(rng, x, y, base_w=rng.integers(60, 170), base_h=rng.integers(50, 160), jitter=0.45)
        detections.append(make_det(label, rng.uniform(0.25, 0.75), (x, y), b, a))
    motion = rng.choice(["No movement", "Moving Left", "Moving Right", "Moving Up", "Moving Down"],
                        p=[0.55, 0.12, 0.12, 0.11, 0.10])
    return detections, motion


SCENE_BUILDERS = {
    "AVOID_PERSON": scene_avoid_person,
    "MOVE_TO_CHAIR": scene_move_to_chair,
    "CHECK_TABLE": scene_check_table,
    "EXPLORE": scene_explore,
}


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    engine = ReasoningEngine()

    rows = []
    for action in ACTION_CLASSES:
        builder = SCENE_BUILDERS[action]
        for _ in range(args.rows_per_class):
            detections, motion_text = builder(rng)
            features = engine.extract_features(detections, FRAME_CENTER, FRAME_AREA, motion_text)
            rows.append(features + [action, "simulated"])

    feature_count = len(rows[0]) - 2
    columns = [f"f{i}" for i in range(feature_count)] + ["label", "source_type"]
    df = pd.DataFrame(rows, columns=columns).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(
        args.out_dir,
        f"simulated_room_people_chairs_tables_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    df.to_csv(out_path, index=False)

    counts = df["label"].value_counts().reindex(ACTION_CLASSES, fill_value=0)
    print(f"Saved simulated dataset: {out_path}")
    print(f"Total rows: {len(df)}")
    for label in ACTION_CLASSES:
        print(f"  {label:<15} {int(counts[label])}")


if __name__ == "__main__":
    main()
