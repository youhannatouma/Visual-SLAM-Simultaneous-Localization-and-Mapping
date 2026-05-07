import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


ACTION_CLASSES = ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"]


@dataclass
class PoseSample:
    x: float
    y: float
    theta: float
    timestamp: float


@dataclass
class ActionSample:
    action: str
    confidence: float
    source_mode: str
    timestamp: float


@dataclass
class MapEvent:
    event_type: str
    grid_xy: Tuple[int, int]
    world_xy: Tuple[float, float]
    label: str
    confidence: float
    track_id: int
    timestamp: float


def normalize_angle(theta: float) -> float:
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def parse_action_label(decision_text: str) -> str:
    txt = (decision_text or "").upper()
    if "AVOID PERSON" in txt:
        return "AVOID_PERSON"
    if "CHAIR" in txt:
        return "MOVE_TO_CHAIR"
    if "CHECK TABLE" in txt:
        return "CHECK_TABLE"
    return "EXPLORE"


def action_confidence_from_tracked(action_label: str, tracked: Dict[int, dict]) -> float:
    if not tracked:
        return 0.5 if action_label == "EXPLORE" else 0.0
    target_label = {
        "AVOID_PERSON": "person",
        "MOVE_TO_CHAIR": "chair",
        "CHECK_TABLE": "table",
    }.get(action_label)
    if not target_label:
        return min(1.0, max(0.2, float(np.mean([o.get("confidence", 0.0) for o in tracked.values()]))))
    candidates = [float(obj.get("confidence", 0.0)) for obj in tracked.values() if obj.get("label") == target_label]
    if not candidates:
        return 0.0
    return float(max(candidates))


class LiveMapper:
    def __init__(
        self,
        grid_size: int = 120,
        meters_per_cell: float = 0.10,
        decay: float = 0.985,
        obstacle_increment: float = 0.20,
        free_decrement: float = 0.05,
        camera_fov_deg: float = 70.0,
        ray_step_cells: int = 1,
    ):
        self.grid_size = int(grid_size)
        self.meters_per_cell = float(meters_per_cell)
        self.decay = float(decay)
        self.obstacle_increment = float(obstacle_increment)
        self.free_decrement = float(free_decrement)
        self.camera_fov_deg = float(camera_fov_deg)
        self.ray_step_cells = max(1, int(ray_step_cells))

        self.grid = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float32)
        self.pose = PoseSample(
            x=(self.grid_size * self.meters_per_cell) / 2.0,
            y=(self.grid_size * self.meters_per_cell) / 2.0,
            theta=0.0,
            timestamp=time.time(),
        )
        self.pose_history: List[PoseSample] = [self.pose]
        self.pose_grid_history: List[Tuple[int, int]] = [self.world_to_grid(self.pose.x, self.pose.y)]
        self.map_events: List[MapEvent] = []
        self.frame_obstacles: Dict[int, Set[Tuple[int, int]]] = {}
        self.cell_event_counts: Dict[Tuple[int, int], Dict[str, int]] = {}

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round(x / self.meters_per_cell))
        gy = int(round(y / self.meters_per_cell))
        gx = int(np.clip(gx, 0, self.grid_size - 1))
        gy = int(np.clip(gy, 0, self.grid_size - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.meters_per_cell, gy * self.meters_per_cell

    def update_pose_from_orb(self, dx_px: float, dy_px: float, timestamp: float, motion_to_meter_scale: float) -> PoseSample:
        # Convert image-plane flow to rough planar translation.
        tx = float(dx_px) * float(motion_to_meter_scale)
        ty = float(dy_px) * float(motion_to_meter_scale)
        theta = self.pose.theta
        if abs(tx) > 1e-6 or abs(ty) > 1e-6:
            theta = math.atan2(ty, tx)

        self.pose = PoseSample(
            x=self.pose.x + tx,
            y=self.pose.y + ty,
            theta=normalize_angle(theta),
            timestamp=float(timestamp),
        )
        self.pose_history.append(self.pose)
        self.pose_grid_history.append(self.world_to_grid(self.pose.x, self.pose.y))
        return self.pose

    def _range_from_area_ratio(self, area_ratio: float) -> float:
        area_ratio = max(1e-6, float(area_ratio))
        # Heuristic inverse relationship: larger object -> closer obstacle.
        return float(np.clip(0.30 / math.sqrt(area_ratio), 0.4, 4.0))

    def project_detection_to_world(
        self,
        det: dict,
        frame_shape: Sequence[int],
    ) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy = det["center"]
        area = max(1.0, float(det.get("area", 1.0)))
        area_ratio = area / float(max(1, w * h))
        bearing = ((float(cx) / max(1.0, float(w))) - 0.5) * math.radians(self.camera_fov_deg)
        rng = self._range_from_area_ratio(area_ratio)
        ang = self.pose.theta + bearing
        wx = self.pose.x + rng * math.cos(ang)
        wy = self.pose.y + rng * math.sin(ang)
        gx, gy = self.world_to_grid(wx, wy)
        return (wx, wy), (gx, gy)

    def _ray_cells(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        cells = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells

    def _record_cell_event(self, cell: Tuple[int, int], key: str):
        row = self.cell_event_counts.setdefault(cell, {"hit": 0, "free": 0})
        row[key] = row.get(key, 0) + 1

    def update_from_tracked(self, tracked: Dict[int, dict], frame_shape: Sequence[int], frame_index: int, timestamp: float) -> List[MapEvent]:
        self.grid = np.clip(self.grid * self.decay, 0.0, 1.0)
        pose_cell = self.world_to_grid(self.pose.x, self.pose.y)
        events: List[MapEvent] = []
        frame_cells: Set[Tuple[int, int]] = set()

        for obj_id, obj in tracked.items():
            label = obj.get("label", "other")
            if label not in ("person", "chair", "table", "sofa", "tv"):
                continue
            (wx, wy), (gx, gy) = self.project_detection_to_world(obj, frame_shape)
            event = MapEvent(
                event_type="obstacle_mark",
                grid_xy=(gx, gy),
                world_xy=(wx, wy),
                label=label,
                confidence=float(obj.get("confidence", 0.0)),
                track_id=int(obj_id),
                timestamp=float(timestamp),
            )
            self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + self.obstacle_increment)
            self._record_cell_event((gx, gy), "hit")
            frame_cells.add((gx, gy))
            events.append(event)

            ray = self._ray_cells(pose_cell, (gx, gy))
            for cell_idx, (rx, ry) in enumerate(ray[:-1]):
                if cell_idx % self.ray_step_cells != 0:
                    continue
                self.grid[ry, rx] = max(0.0, self.grid[ry, rx] - self.free_decrement)
                self._record_cell_event((rx, ry), "free")
                events.append(
                    MapEvent(
                        event_type="free_space_ray",
                        grid_xy=(rx, ry),
                        world_xy=self.grid_to_world(rx, ry),
                        label=label,
                        confidence=float(obj.get("confidence", 0.0)),
                        track_id=int(obj_id),
                        timestamp=float(timestamp),
                    )
                )

        self.frame_obstacles[int(frame_index)] = frame_cells
        self.map_events.extend(events)
        return events

    def render_map(self, out_size: int = 320) -> np.ndarray:
        occ = (self.grid * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

        if len(self.pose_grid_history) > 1:
            pts = np.array([[gx, gy] for gx, gy in self.pose_grid_history], dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, self.grid_size - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.grid_size - 1)
            cv2.polylines(bgr, [pts.reshape((-1, 1, 2))], False, (0, 255, 255), 1, cv2.LINE_AA)

        gx, gy = self.world_to_grid(self.pose.x, self.pose.y)
        cv2.circle(bgr, (gx, gy), 2, (0, 0, 255), -1)

        return cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    def pose_stats(self) -> dict:
        if len(self.pose_history) < 2:
            return {"path_length_m": 0.0, "pose_samples": len(self.pose_history)}
        dist = 0.0
        for i in range(1, len(self.pose_history)):
            a, b = self.pose_history[i - 1], self.pose_history[i]
            dist += math.hypot(b.x - a.x, b.y - a.y)
        return {"path_length_m": float(dist), "pose_samples": len(self.pose_history)}


def _micro_prf(tp: int, fp: int, fn: int) -> dict:
    precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def compute_label_metrics(gt_labels_by_frame: Dict[int, str], pred_actions_by_frame: Dict[int, str]) -> dict:
    frames = sorted(set(gt_labels_by_frame.keys()) & set(pred_actions_by_frame.keys()))
    if not frames:
        return {"available": False, "reason": "No overlapping frame labels"}
    y_true = [gt_labels_by_frame[f] for f in frames]
    y_pred = [pred_actions_by_frame[f] for f in frames]
    accuracy = float(sum(1 for t, p in zip(y_true, y_pred) if t == p)) / float(len(frames))
    per_class = {}
    macro_f1 = 0.0
    for cls in ACTION_CLASSES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
        stats = _micro_prf(tp, fp, fn)
        per_class[cls] = stats
        macro_f1 += stats["f1"]
    macro_f1 /= float(len(ACTION_CLASSES))
    return {
        "available": True,
        "frames_evaluated": len(frames),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def compute_loop_closure_drift(
    poses: Sequence[PoseSample],
    closure_radius_m: float = 0.35,
    min_frame_gap: int = 30,
) -> dict:
    if len(poses) < 2:
        return {"available": False, "reason": "Insufficient pose samples"}
    trans_errors = []
    heading_errors = []
    for i in range(len(poses)):
        pi = poses[i]
        for j in range(i + min_frame_gap, len(poses)):
            pj = poses[j]
            d = math.hypot(pj.x - pi.x, pj.y - pi.y)
            if d <= closure_radius_m:
                trans_errors.append(d)
                heading_errors.append(abs(normalize_angle(pj.theta - pi.theta)))
    if not trans_errors:
        return {"available": False, "reason": "No loop-closure candidates detected"}
    return {
        "available": True,
        "closure_pairs": len(trans_errors),
        "translation_error_mean_m": float(np.mean(trans_errors)),
        "translation_error_max_m": float(np.max(trans_errors)),
        "heading_error_mean_rad": float(np.mean(heading_errors)),
        "heading_error_max_rad": float(np.max(heading_errors)),
    }


def compute_map_consistency_score(cell_event_counts: Dict[Tuple[int, int], Dict[str, int]], min_events: int = 2) -> dict:
    scores = []
    for cell_counts in cell_event_counts.values():
        hit = int(cell_counts.get("hit", 0))
        free = int(cell_counts.get("free", 0))
        total = hit + free
        if total < min_events:
            continue
        scores.append(float(max(hit, free)) / float(total))
    if not scores:
        return {"available": False, "reason": "No revisited cells with enough events"}
    return {
        "available": True,
        "cell_count": len(scores),
        "score_mean": float(np.mean(scores)),
        "score_min": float(np.min(scores)),
    }


def compute_obstacle_precision_recall(
    gt_obstacles_by_frame: Dict[int, Set[Tuple[int, int]]],
    pred_obstacles_by_frame: Dict[int, Set[Tuple[int, int]]],
) -> dict:
    frames = sorted(set(gt_obstacles_by_frame.keys()) & set(pred_obstacles_by_frame.keys()))
    if not frames:
        return {"available": False, "reason": "No overlapping obstacle frames"}
    tp = fp = fn = 0
    for f in frames:
        gt = gt_obstacles_by_frame.get(f, set())
        pred = pred_obstacles_by_frame.get(f, set())
        tp += len(gt & pred)
        fp += len(pred - gt)
        fn += len(gt - pred)
    out = _micro_prf(tp, fp, fn)
    out.update({"available": True, "frames_evaluated": len(frames)})
    return out


def load_run_annotations(path: str) -> dict:
    if not path:
        return {"available": False, "reason": "No annotation file provided"}
    if not os.path.exists(path):
        return {"available": False, "reason": f"Annotation file not found: {path}"}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    labels = {}
    for row in raw.get("frame_labels", []):
        if "frame" in row and "label" in row:
            labels[int(row["frame"])] = str(row["label"]).strip().upper()
    obstacles = {}
    for row in raw.get("obstacles", []):
        if "frame" not in row or "grid_xy" not in row:
            continue
        frame = int(row["frame"])
        gx, gy = int(row["grid_xy"][0]), int(row["grid_xy"][1])
        obstacles.setdefault(frame, set()).add((gx, gy))
    return {
        "available": True,
        "frame_labels": labels,
        "obstacles_by_frame": obstacles,
        "raw": raw,
    }


def write_joint_report(report_path: str, payload: dict):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dataclass_to_dict(obj):
    return asdict(obj)
