import json
import math
import os
import time
from collections import deque
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


@dataclass
class CameraCalibration:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    dist_coeffs: Tuple[float, ...] = ()


def load_camera_calibration(path: str) -> Optional[CameraCalibration]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Camera calibration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    camera = raw.get("camera_matrix", raw)
    if isinstance(camera, list):
        matrix = np.asarray(camera, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("camera_matrix must be a 3x3 matrix")
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    else:
        fx = camera.get("fx")
        fy = camera.get("fy")
        cx = camera.get("cx")
        cy = camera.get("cy")
    values = [fx, fy, cx, cy]
    if any(v is None for v in values):
        raise ValueError("Camera calibration must provide fx, fy, cx, cy")

    dist = raw.get("dist_coeffs", raw.get("distortion_coefficients", ()))
    return CameraCalibration(
        fx=float(fx),
        fy=float(fy),
        cx=float(cx),
        cy=float(cy),
        width=int(raw.get("width", 640)),
        height=int(raw.get("height", 480)),
        dist_coeffs=tuple(float(x) for x in dist),
    )


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
        pose_smoothing_window: int = 5,
        max_translation_m_per_frame: float = 0.08,
        max_rotation_rad_per_frame: float = 0.35,
        confidence_weighting: bool = True,
        confidence_strength: float = 1.0,
        obstacle_persistence_frames: int = 2,
        loop_closure_enabled: bool = True,
        loop_closure_radius_m: float = 0.25,
        loop_closure_min_frame_gap: int = 80,
        loop_closure_max_heading_delta_rad: float = 0.55,
        loop_closure_correction_alpha: float = 0.25,
        loop_closure_cooldown_frames: int = 45,
        render_raw_trajectory: bool = False,
        camera_calibration: Optional[CameraCalibration] = None,
        mapping_backend: str = "heuristic",
        depth_unit_scale: float = 1.0,
        inverse_depth: bool = False,
    ):
        if mapping_backend not in ("heuristic", "depth", "orb_slam_like"):
            raise ValueError("mapping_backend must be one of: heuristic, depth, orb_slam_like")
        self.grid_size = int(grid_size)
        self.meters_per_cell = float(meters_per_cell)
        self.decay = float(decay)
        self.obstacle_increment = float(obstacle_increment)
        self.free_decrement = float(free_decrement)
        self.camera_fov_deg = float(camera_fov_deg)
        self.ray_step_cells = max(1, int(ray_step_cells))
        self.pose_smoothing_window = max(1, int(pose_smoothing_window))
        self.max_translation_m_per_frame = max(1e-3, float(max_translation_m_per_frame))
        self.max_rotation_rad_per_frame = max(1e-3, float(max_rotation_rad_per_frame))
        self.confidence_weighting = bool(confidence_weighting)
        self.confidence_strength = max(0.0, float(confidence_strength))
        self.obstacle_persistence_frames = max(1, int(obstacle_persistence_frames))
        self.loop_closure_enabled = bool(loop_closure_enabled)
        self.loop_closure_radius_m = max(0.05, float(loop_closure_radius_m))
        self.loop_closure_min_frame_gap = max(10, int(loop_closure_min_frame_gap))
        self.loop_closure_max_heading_delta_rad = max(0.05, float(loop_closure_max_heading_delta_rad))
        self.loop_closure_correction_alpha = float(np.clip(loop_closure_correction_alpha, 0.05, 1.0))
        self.loop_closure_cooldown_frames = max(1, int(loop_closure_cooldown_frames))
        self.render_raw_trajectory = bool(render_raw_trajectory)
        self.camera_calibration = camera_calibration
        self.mapping_backend = str(mapping_backend)
        self.depth_unit_scale = max(1e-6, float(depth_unit_scale))
        self.inverse_depth = bool(inverse_depth)
        self.backend_status = (
            "external_backend_required"
            if self.mapping_backend == "orb_slam_like"
            else "active"
        )

        self.grid = np.full((self.grid_size, self.grid_size), 0.5, dtype=np.float32)
        self.pose = PoseSample(
            x=(self.grid_size * self.meters_per_cell) / 2.0,
            y=(self.grid_size * self.meters_per_cell) / 2.0,
            theta=0.0,
            timestamp=time.time(),
        )
        self.pose_history: List[PoseSample] = [self.pose]
        self.pose_grid_history: List[Tuple[int, int]] = [self.world_to_grid(self.pose.x, self.pose.y)]
        self.corrected_pose = PoseSample(self.pose.x, self.pose.y, self.pose.theta, self.pose.timestamp)
        self.corrected_pose_history: List[PoseSample] = [self.corrected_pose]
        self.corrected_pose_grid_history: List[Tuple[int, int]] = [self.world_to_grid(self.corrected_pose.x, self.corrected_pose.y)]
        self.raw_pose_grid_history: List[Tuple[int, int]] = list(self.pose_grid_history)
        self.map_events: List[MapEvent] = []
        self.frame_obstacles: Dict[int, Set[Tuple[int, int]]] = {}
        self.cell_event_counts: Dict[Tuple[int, int], Dict[str, int]] = {}
        self.pose_delta_history: deque = deque(maxlen=self.pose_smoothing_window)
        self.detection_cell_history: Dict[Tuple[str, int], deque] = {}
        self.last_frame_obstacle_counts = {"strong": 0, "weak": 0}
        self.frame_counter = 0
        self.loop_closure_state = "idle"
        self.loop_closure_cooldown_remaining = 0
        self.pending_correction = {"dx": 0.0, "dy": 0.0, "dtheta": 0.0}
        self.loop_closure_corrections_applied = 0
        self.loop_closure_candidate_count = 0
        self.loop_closure_rejections = 0
        self.loop_closure_correction_records: List[Dict[str, float]] = []
        self.post_closure_alignment_dists: List[float] = []

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        gx = int(round(x / self.meters_per_cell))
        gy = int(round(y / self.meters_per_cell))
        gx = int(np.clip(gx, 0, self.grid_size - 1))
        gy = int(np.clip(gy, 0, self.grid_size - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        return gx * self.meters_per_cell, gy * self.meters_per_cell

    def _clip_pose_delta(self, tx: float, ty: float) -> Tuple[float, float]:
        mag = math.hypot(tx, ty)
        if mag <= self.max_translation_m_per_frame:
            return tx, ty
        scale = self.max_translation_m_per_frame / max(1e-9, mag)
        return tx * scale, ty * scale

    def _smooth_pose_delta(self, tx: float, ty: float) -> Tuple[float, float]:
        self.pose_delta_history.append((float(tx), float(ty)))
        sx = float(np.mean([p[0] for p in self.pose_delta_history]))
        sy = float(np.mean([p[1] for p in self.pose_delta_history]))
        return sx, sy

    def update_pose_from_orb(self, dx_px: float, dy_px: float, timestamp: float, motion_to_meter_scale: float) -> PoseSample:
        return self.update_pose_from_flow(dx_px, dy_px, 0.0, timestamp, motion_to_meter_scale, flow_quality=1.0)

    def update_pose_from_flow(
        self,
        dx_px: float,
        dy_px: float,
        dtheta_rad: float,
        timestamp: float,
        motion_to_meter_scale: float,
        flow_quality: float = 1.0,
    ) -> PoseSample:
        tx = float(dx_px) * float(motion_to_meter_scale)
        # Image Y grows downward; invert for map/world coordinates.
        ty = -float(dy_px) * float(motion_to_meter_scale)
        q = float(np.clip(flow_quality, 0.0, 1.0))
        tx *= q
        ty *= q
        tx, ty = self._clip_pose_delta(tx, ty)
        tx, ty = self._smooth_pose_delta(tx, ty)

        target_theta = self.pose.theta
        if abs(dtheta_rad) > 1e-8:
            target_theta = normalize_angle(self.pose.theta + float(dtheta_rad) * q)
        elif abs(tx) > 1e-6 or abs(ty) > 1e-6:
            target_theta = math.atan2(ty, tx)
        heading_delta = normalize_angle(target_theta - self.pose.theta)
        heading_delta = float(np.clip(heading_delta, -self.max_rotation_rad_per_frame, self.max_rotation_rad_per_frame))
        theta = normalize_angle(self.pose.theta + heading_delta)

        self.pose = PoseSample(
            x=self.pose.x + tx,
            y=self.pose.y + ty,
            theta=normalize_angle(theta),
            timestamp=float(timestamp),
        )
        self.frame_counter += 1
        self.pose_history.append(self.pose)
        self.raw_pose_grid_history.append(self.world_to_grid(self.pose.x, self.pose.y))
        self._update_corrected_pose()
        self.pose_grid_history = self.corrected_pose_grid_history
        return self.pose

    def update_pose_from_calibrated_flow(
        self,
        dx_px: float,
        dy_px: float,
        dtheta_rad: float,
        timestamp: float,
        nominal_depth_m: float,
        flow_quality: float = 1.0,
    ) -> PoseSample:
        if self.camera_calibration is None:
            raise ValueError("Camera calibration is required for calibrated pose updates")
        depth_m = max(1e-6, float(nominal_depth_m))
        fx = max(1e-6, float(self.camera_calibration.fx))
        fy = max(1e-6, float(self.camera_calibration.fy))
        tx = float(dx_px) * depth_m / fx
        ty = float(dy_px) * depth_m / fy
        return self.update_pose_from_flow(
            dx_px=tx,
            dy_px=ty,
            dtheta_rad=dtheta_rad,
            timestamp=timestamp,
            motion_to_meter_scale=1.0,
            flow_quality=flow_quality,
        )

    def _find_loop_closure_candidate(self) -> Optional[Dict[str, float]]:
        if not self.loop_closure_enabled:
            return None
        if len(self.pose_history) < self.loop_closure_min_frame_gap + 2:
            return None
        current_idx = len(self.pose_history) - 1
        current = self.pose_history[current_idx]
        best = None
        best_dist = float("inf")
        cutoff = current_idx - self.loop_closure_min_frame_gap
        for i in range(max(0, cutoff)):
            prev = self.pose_history[i]
            d = math.hypot(current.x - prev.x, current.y - prev.y)
            if d > self.loop_closure_radius_m:
                continue
            heading_delta = abs(normalize_angle(current.theta - prev.theta))
            if heading_delta > self.loop_closure_max_heading_delta_rad:
                continue
            if d < best_dist:
                best_dist = d
                best = {
                    "index": i,
                    "distance": d,
                    "heading_delta": heading_delta,
                    "target_x": prev.x,
                    "target_y": prev.y,
                    "target_theta": prev.theta,
                }
        return best

    def _apply_loop_closure_logic(self):
        if self.loop_closure_cooldown_remaining > 0:
            self.loop_closure_state = "cooldown"
            self.loop_closure_cooldown_remaining -= 1
            return
        candidate = self._find_loop_closure_candidate()
        if candidate is None:
            self.loop_closure_state = "idle"
            return
        self.loop_closure_state = "candidate"
        self.loop_closure_candidate_count += 1
        dx = candidate["target_x"] - self.corrected_pose.x
        dy = candidate["target_y"] - self.corrected_pose.y
        dtheta = normalize_angle(candidate["target_theta"] - self.corrected_pose.theta)
        correction_norm = math.hypot(dx, dy)
        if correction_norm > (self.loop_closure_radius_m * 2.0):
            self.loop_closure_rejections += 1
            self.loop_closure_state = "idle"
            return
        self.pending_correction["dx"] += float(dx * self.loop_closure_correction_alpha)
        self.pending_correction["dy"] += float(dy * self.loop_closure_correction_alpha)
        self.pending_correction["dtheta"] += float(dtheta * self.loop_closure_correction_alpha)
        self.loop_closure_state = "correcting"
        self.loop_closure_corrections_applied += 1
        self.loop_closure_correction_records.append(
            {
                "translation_m": float(math.hypot(dx * self.loop_closure_correction_alpha, dy * self.loop_closure_correction_alpha)),
                "heading_rad": float(abs(dtheta * self.loop_closure_correction_alpha)),
                "raw_distance_to_anchor_m": float(candidate["distance"]),
            }
        )
        self.post_closure_alignment_dists.append(float(math.hypot(dx, dy)))
        self.loop_closure_cooldown_remaining = self.loop_closure_cooldown_frames

    def _update_corrected_pose(self):
        prev_corr = self.corrected_pose_history[-1]
        dx_raw = self.pose.x - self.pose_history[-2].x if len(self.pose_history) > 1 else 0.0
        dy_raw = self.pose.y - self.pose_history[-2].y if len(self.pose_history) > 1 else 0.0
        dtheta_raw = normalize_angle(self.pose.theta - self.pose_history[-2].theta) if len(self.pose_history) > 1 else 0.0
        self._apply_loop_closure_logic()

        corr_dx = self.pending_correction["dx"] * self.loop_closure_correction_alpha
        corr_dy = self.pending_correction["dy"] * self.loop_closure_correction_alpha
        corr_dtheta = self.pending_correction["dtheta"] * self.loop_closure_correction_alpha
        self.pending_correction["dx"] -= corr_dx
        self.pending_correction["dy"] -= corr_dy
        self.pending_correction["dtheta"] -= corr_dtheta

        new_pose = PoseSample(
            x=prev_corr.x + dx_raw + corr_dx,
            y=prev_corr.y + dy_raw + corr_dy,
            theta=normalize_angle(prev_corr.theta + dtheta_raw + corr_dtheta),
            timestamp=self.pose.timestamp,
        )
        self.corrected_pose = new_pose
        self.corrected_pose_history.append(new_pose)
        self.corrected_pose_grid_history.append(self.world_to_grid(new_pose.x, new_pose.y))

    def _range_from_area_ratio(self, area_ratio: float) -> float:
        area_ratio = max(1e-6, float(area_ratio))
        # Heuristic inverse relationship: larger object -> closer obstacle.
        return float(np.clip(0.30 / math.sqrt(area_ratio), 0.4, 4.0))

    def _class_range_adjustment(self, label: str, rng: float) -> float:
        if label == "person":
            return float(np.clip(rng * 0.85, 0.3, 4.0))
        if label == "chair":
            return float(np.clip(rng * 0.95, 0.3, 4.0))
        if label == "table":
            return float(np.clip(rng * 1.1, 0.3, 5.0))
        return rng

    def _depth_range_for_detection(self, det: dict, depth_map: Optional[np.ndarray]) -> Optional[float]:
        if depth_map is None:
            return None
        if depth_map.size == 0:
            return None
        h, w = depth_map.shape[:2]
        if "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            x1 = int(np.clip(x1, 0, w - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            y2 = int(np.clip(y2, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                return None
            crop = depth_map[y1:y2 + 1, x1:x2 + 1]
        else:
            cx, cy = det["center"]
            cx = int(np.clip(cx, 0, w - 1))
            cy = int(np.clip(cy, 0, h - 1))
            r = 2
            crop = depth_map[max(0, cy - r): min(h, cy + r + 1), max(0, cx - r): min(w, cx + r + 1)]
        vals = np.asarray(crop, dtype=np.float32)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            return None
        depth_value = float(np.median(vals))
        if self.inverse_depth:
            depth_value = 1.0 / max(depth_value, 1e-6)
        return float(np.clip(depth_value * self.depth_unit_scale, 0.2, 8.0))

    def _bearing_for_detection(self, cx: float, frame_width: int) -> float:
        if self.camera_calibration is not None:
            return math.atan2(float(cx) - self.camera_calibration.cx, max(1e-6, self.camera_calibration.fx))
        return ((float(cx) / max(1.0, float(frame_width))) - 0.5) * math.radians(self.camera_fov_deg)

    def _weighted_step(self, base_value: float, confidence: float, strong: bool) -> float:
        if not self.confidence_weighting:
            return base_value * (1.0 if strong else 0.5)
        conf = float(np.clip(confidence, 0.0, 1.0))
        weight = 0.35 + (conf ** max(0.1, self.confidence_strength))
        if not strong:
            weight *= 0.55
        return base_value * weight

    def _is_persistent_detection(self, track_id: int, label: str, cell: Tuple[int, int]) -> bool:
        key = (str(label), int(track_id))
        hist = self.detection_cell_history.setdefault(key, deque(maxlen=self.obstacle_persistence_frames))
        hist.append(cell)
        if len(hist) < self.obstacle_persistence_frames:
            return False
        return len(set(hist)) <= 2

    def project_detection_to_world(
        self,
        det: dict,
        frame_shape: Sequence[int],
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float], Tuple[int, int]]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy = det["center"]
        area = max(1.0, float(det.get("area", 1.0)))
        area_ratio = area / float(max(1, w * h))
        bearing = self._bearing_for_detection(float(cx), w)
        depth_range = self._depth_range_for_detection(det, depth_map)
        if self.mapping_backend in ("depth", "orb_slam_like") and depth_range is not None:
            rng = depth_range
        else:
            rng = self._range_from_area_ratio(area_ratio)
            rng = self._class_range_adjustment(str(det.get("label", "")), rng)
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

    def update_from_tracked(
        self,
        tracked: Dict[int, dict],
        frame_shape: Sequence[int],
        frame_index: int,
        timestamp: float,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[MapEvent]:
        self.grid = np.clip(self.grid * self.decay, 0.0, 1.0)
        pose_cell = self.world_to_grid(self.pose.x, self.pose.y)
        events: List[MapEvent] = []
        frame_cells: Set[Tuple[int, int]] = set()
        strong_count = 0
        weak_count = 0

        for obj_id, obj in tracked.items():
            label = obj.get("label", "other")
            if label not in ("person", "chair", "table", "sofa", "tv"):
                continue
            (wx, wy), (gx, gy) = self.project_detection_to_world(obj, frame_shape, depth_map=depth_map)
            conf = float(obj.get("confidence", 0.0))
            is_persistent = self._is_persistent_detection(int(obj_id), str(label), (gx, gy))
            strong_count += 1 if is_persistent else 0
            weak_count += 0 if is_persistent else 1
            event = MapEvent(
                event_type="obstacle_mark",
                grid_xy=(gx, gy),
                world_xy=(wx, wy),
                label=label,
                confidence=conf,
                track_id=int(obj_id),
                timestamp=float(timestamp),
            )
            inc = self._weighted_step(self.obstacle_increment, conf, strong=is_persistent)
            self.grid[gy, gx] = min(1.0, self.grid[gy, gx] + inc)
            self._record_cell_event((gx, gy), "hit")
            frame_cells.add((gx, gy))
            events.append(event)

            ray = self._ray_cells(pose_cell, (gx, gy))
            for cell_idx, (rx, ry) in enumerate(ray[:-1]):
                if cell_idx % self.ray_step_cells != 0:
                    continue
                dec = self._weighted_step(self.free_decrement, conf, strong=is_persistent)
                self.grid[ry, rx] = max(0.0, self.grid[ry, rx] - dec)
                self._record_cell_event((rx, ry), "free")
                events.append(
                    MapEvent(
                        event_type="free_space_ray",
                        grid_xy=(rx, ry),
                        world_xy=self.grid_to_world(rx, ry),
                        label=label,
                        confidence=conf,
                        track_id=int(obj_id),
                        timestamp=float(timestamp),
                    )
                )

        self.frame_obstacles[int(frame_index)] = frame_cells
        self.map_events.extend(events)
        self.last_frame_obstacle_counts = {"strong": int(strong_count), "weak": int(weak_count)}
        return events

    def render_map(self, out_size: int = 320) -> np.ndarray:
        occ = (self.grid * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

        if len(self.corrected_pose_grid_history) > 1:
            pts = np.array([[gx, gy] for gx, gy in self.corrected_pose_grid_history], dtype=np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, self.grid_size - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, self.grid_size - 1)
            cv2.polylines(bgr, [pts.reshape((-1, 1, 2))], False, (0, 255, 255), 1, cv2.LINE_AA)
        if self.render_raw_trajectory and len(self.raw_pose_grid_history) > 1:
            raw_pts = np.array([[gx, gy] for gx, gy in self.raw_pose_grid_history], dtype=np.int32)
            raw_pts[:, 0] = np.clip(raw_pts[:, 0], 0, self.grid_size - 1)
            raw_pts[:, 1] = np.clip(raw_pts[:, 1], 0, self.grid_size - 1)
            cv2.polylines(bgr, [raw_pts.reshape((-1, 1, 2))], False, (255, 120, 0), 1, cv2.LINE_AA)

        gx, gy = self.world_to_grid(self.corrected_pose.x, self.corrected_pose.y)
        cv2.circle(bgr, (gx, gy), 2, (0, 0, 255), -1)

        return cv2.resize(bgr, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    def pose_stats(self) -> dict:
        if len(self.pose_history) < 2:
            return {"path_length_m": 0.0, "pose_samples": len(self.pose_history)}
        dist = 0.0
        for i in range(1, len(self.pose_history)):
            a, b = self.pose_history[i - 1], self.pose_history[i]
            dist += math.hypot(b.x - a.x, b.y - a.y)
        corrected_dist = 0.0
        for i in range(1, len(self.corrected_pose_history)):
            a, b = self.corrected_pose_history[i - 1], self.corrected_pose_history[i]
            corrected_dist += math.hypot(b.x - a.x, b.y - a.y)
        return {
            "path_length_m": float(dist),
            "corrected_path_length_m": float(corrected_dist),
            "pose_samples": len(self.pose_history),
        }

    def backend_summary(self) -> dict:
        return {
            "backend": self.mapping_backend,
            "status": self.backend_status,
            "has_camera_calibration": self.camera_calibration is not None,
            "depth_unit_scale": float(self.depth_unit_scale),
            "inverse_depth": bool(self.inverse_depth),
        }

    def loop_closure_summary(self) -> dict:
        if self.loop_closure_correction_records:
            mean_t = float(np.mean([r["translation_m"] for r in self.loop_closure_correction_records]))
            mean_h = float(np.mean([r["heading_rad"] for r in self.loop_closure_correction_records]))
        else:
            mean_t = 0.0
            mean_h = 0.0
        if self.post_closure_alignment_dists:
            post_align = float(np.mean(self.post_closure_alignment_dists))
        else:
            post_align = 0.0
        return {
            "state": self.loop_closure_state,
            "corrections_applied": int(self.loop_closure_corrections_applied),
            "candidates": int(self.loop_closure_candidate_count),
            "rejections": int(self.loop_closure_rejections),
            "mean_correction_translation_m": mean_t,
            "mean_correction_heading_rad": mean_h,
            "post_closure_path_alignment_score": float(1.0 / (1.0 + post_align)),
            "post_closure_path_alignment_mean_dist_m": post_align,
        }


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


def compute_pose_jitter_score(poses: Sequence[PoseSample], min_motion_m: float = 0.002) -> dict:
    if len(poses) < 4:
        return {"available": False, "reason": "Insufficient pose samples"}
    deltas_all = []
    headings_all = []
    for i in range(1, len(poses)):
        a, b = poses[i - 1], poses[i]
        deltas_all.append(math.hypot(b.x - a.x, b.y - a.y))
        headings_all.append(normalize_angle(b.theta - a.theta))
    delta_all_arr = np.array(deltas_all, dtype=np.float32)
    moving_mask = delta_all_arr >= float(min_motion_m)
    if int(np.sum(moving_mask)) < 3:
        return {"available": False, "reason": "Insufficient moving pose deltas"}
    delta_arr = delta_all_arr[moving_mask]
    head_arr = np.array(headings_all, dtype=np.float32)[moving_mask]
    motion_cv = float(np.std(delta_arr) / (float(np.mean(delta_arr)) + 1e-6))
    heading_std = float(np.std(head_arr))
    jitter_score = float(1.0 / (1.0 + motion_cv + heading_std))
    return {
        "available": True,
        "moving_samples": int(delta_arr.size),
        "total_samples": int(delta_all_arr.size),
        "min_motion_m": float(min_motion_m),
        "motion_cv": motion_cv,
        "heading_std_rad": heading_std,
        "jitter_score": jitter_score,
    }


def compute_obstacle_persistence_stability(frame_obstacles: Dict[int, Set[Tuple[int, int]]]) -> dict:
    frames = sorted(frame_obstacles.keys())
    if len(frames) < 3:
        return {"available": False, "reason": "Insufficient obstacle frames"}
    ious = []
    for i in range(1, len(frames)):
        prev_set = frame_obstacles.get(frames[i - 1], set())
        cur_set = frame_obstacles.get(frames[i], set())
        union = prev_set | cur_set
        if not union:
            continue
        iou = float(len(prev_set & cur_set)) / float(len(union))
        ious.append(iou)
    if not ious:
        return {"available": False, "reason": "No obstacle overlap samples"}
    return {
        "available": True,
        "samples": len(ious),
        "iou_mean": float(np.mean(ious)),
        "iou_min": float(np.min(ious)),
    }


def compute_occupancy_confidence_concentration(grid: np.ndarray) -> dict:
    if grid is None or grid.size == 0:
        return {"available": False, "reason": "Empty occupancy grid"}
    occ = np.clip(grid.astype(np.float32), 1e-6, 1.0 - 1e-6)
    entropy = -(occ * np.log(occ) + (1.0 - occ) * np.log(1.0 - occ))
    norm_entropy = float(np.mean(entropy) / math.log(2.0))
    concentration = float(1.0 - norm_entropy)
    return {
        "available": True,
        "entropy_mean_bits": float(np.mean(entropy) / math.log(2.0)),
        "concentration_score": concentration,
    }


def compute_mapping_quality_summary(
    loop_closure_drift: dict,
    map_consistency_score: dict,
    pose_jitter: dict,
    obstacle_persistence: dict,
    occupancy_concentration: dict,
    obstacle_precision_recall: dict,
    require_benchmark: bool = True,
    threshold_overrides: Optional[Dict[str, float]] = None,
) -> dict:
    thresholds = {
        "map_consistency_min": 0.70,
        "pose_jitter_min": 0.40,
        "obstacle_persistence_iou_min": 0.20,
        "occupancy_concentration_min": 0.08,
        "benchmark_obstacle_f1_min": 0.45,
    }
    if threshold_overrides:
        for key, value in threshold_overrides.items():
            if key in thresholds:
                thresholds[key] = float(value)
    checks = {}
    checks["map_consistency"] = bool(map_consistency_score.get("available")) and float(map_consistency_score.get("score_mean", 0.0)) >= thresholds["map_consistency_min"]
    checks["pose_jitter"] = bool(pose_jitter.get("available")) and float(pose_jitter.get("jitter_score", 0.0)) >= thresholds["pose_jitter_min"]
    checks["obstacle_persistence"] = bool(obstacle_persistence.get("available")) and float(obstacle_persistence.get("iou_mean", 0.0)) >= thresholds["obstacle_persistence_iou_min"]
    checks["occupancy_concentration"] = bool(occupancy_concentration.get("available")) and float(occupancy_concentration.get("concentration_score", 0.0)) >= thresholds["occupancy_concentration_min"]
    if require_benchmark:
        checks["benchmark_obstacle_f1"] = bool(obstacle_precision_recall.get("available")) and float(obstacle_precision_recall.get("f1", 0.0)) >= thresholds["benchmark_obstacle_f1_min"]
    else:
        checks["benchmark_obstacle_f1"] = True
    missing_benchmark = require_benchmark and not bool(obstacle_precision_recall.get("available"))
    promotable = all(checks.values()) and not missing_benchmark
    lane = "benchmark_supervised" if bool(obstacle_precision_recall.get("available")) else "live_unsupervised"
    status = "promotable" if promotable else ("insufficient_evidence" if missing_benchmark else "not_promotable")
    return {
        "lane": lane,
        "status": status,
        "promotable": bool(promotable),
        "require_benchmark": bool(require_benchmark),
        "thresholds": thresholds,
        "checks": checks,
        "missing_benchmark": bool(missing_benchmark),
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
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dataclass_to_dict(obj):
    return asdict(obj)
