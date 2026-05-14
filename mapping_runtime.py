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
    """
    Represents a single pose measurement at a specific timestamp.
    """
    x: float
    y: float
    theta: float
    timestamp: float


@dataclass
class ActionSample:
    """
    Represents a navigation action decided by the system.
    """
    action: str
    confidence: float
    source_mode: str
    timestamp: float


@dataclass
class MapEvent:
    """
    Represents an event recorded in the occupancy map, such as obstacle detection.
    """
    event_type: str
    grid_xy: Tuple[int, int]
    world_xy: Tuple[float, float]
    label: str
    confidence: float
    track_id: int
    timestamp: float
    anchor_grid_xy: Optional[Tuple[int, int]] = None
    anchor_world_xy: Optional[Tuple[float, float]] = None
    footprint_role: str = ""


@dataclass
class CameraCalibration:
    """
    Holds intrinsic camera parameters for metric 3D projection.
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 640
    height: int = 480
    dist_coeffs: Tuple[float, ...] = ()


def load_camera_calibration(path: str) -> Optional[CameraCalibration]:
    """
    Loads camera calibration parameters from a JSON file.
    """
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
    """
    Normalizes an angle in radians to the range [-pi, pi].
    """
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    return theta


def parse_action_label(decision_text: str) -> str:
    """
    Maps high-level decision text to a canonical action class.
    """
    txt = (decision_text or "").upper()
    if "AVOID PERSON" in txt:
        return "AVOID_PERSON"
    if "CHAIR" in txt:
        return "MOVE_TO_CHAIR"
    if "CHECK TABLE" in txt:
        return "CHECK_TABLE"
    return "EXPLORE"


def action_confidence_from_tracked(action_label: str, tracked: Dict[int, dict]) -> float:
    """
    Derives a confidence score for an action based on the confidence of relevant tracked objects.
    """
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
    """
    Implements a real-time occupancy grid mapper with pose estimation, 
    loop closure, and object-based mapping.
    """
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
        obstacle_footprint_radius_cells: int = 0,
        obstacle_footprint_shape: str = "square",
        obstacle_temporal_persistence_frames: int = 0,
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
        """
        Initializes the mapper with configurable parameters for grid size, 
        motion constraints, and mapping heuristics.
        """
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
        self.obstacle_footprint_radius_cells = max(0, int(obstacle_footprint_radius_cells))
        self.obstacle_footprint_shape = str(obstacle_footprint_shape)
        if self.obstacle_footprint_shape not in ("square", "horizontal", "vertical", "cross", "class_aware"):
            raise ValueError("obstacle_footprint_shape must be one of: square, horizontal, vertical, cross, class_aware")
        self.obstacle_temporal_persistence_frames = max(0, int(obstacle_temporal_persistence_frames))
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
        self.projection_anchor_mode = "bottom_center"
        self.default_footprint_shape = "class_aware"
        self.default_footprint_radius_by_label = {
            "person": 1,
            "chair": 1,
            "table": 2,
            "sofa": 2,
            "tv": 1,
        }
        self.range_hint_by_label = {
            "person": {"multiplier": 0.75, "min": 0.25, "max": 3.5},
            "chair": {"multiplier": 0.90, "min": 0.35, "max": 4.0},
            "table": {"multiplier": 1.05, "min": 0.45, "max": 4.8},
            "sofa": {"multiplier": 1.00, "min": 0.45, "max": 4.8},
            "tv": {"multiplier": 1.05, "min": 0.40, "max": 4.5},
        }
        self.object_projection_history: Dict[Tuple[str, int], Tuple[int, int]] = {}

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
        self.temporal_obstacle_history: deque = deque(maxlen=self.obstacle_temporal_persistence_frames)
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

    def mapping_pose(self) -> PoseSample:
        """
        Returns the current pose used for mapping, corrected for loop closures if enabled.
        """
        return self.corrected_pose if self.loop_closure_enabled else self.pose

    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """
        Converts metric world coordinates to discrete grid cell coordinates.
        """
        gx = int(round(x / self.meters_per_cell))
        gy = int(round(y / self.meters_per_cell))
        gx = int(np.clip(gx, 0, self.grid_size - 1))
        gy = int(np.clip(gy, 0, self.grid_size - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """
        Converts discrete grid cell coordinates to metric world coordinates.
        """
        return gx * self.meters_per_cell, gy * self.meters_per_cell

    def _clip_pose_delta(self, tx: float, ty: float) -> Tuple[float, float]:
        """
        Clips the translation delta to stay within the configured maximum per-frame motion.
        """
        mag = math.hypot(tx, ty)
        if mag <= self.max_translation_m_per_frame:
            return tx, ty
        scale = self.max_translation_m_per_frame / max(1e-9, mag)
        return tx * scale, ty * scale

    def _smooth_pose_delta(self, tx: float, ty: float) -> Tuple[float, float]:
        """
        Applies a moving average filter to the translation deltas for smoother trajectory.
        """
        self.pose_delta_history.append((float(tx), float(ty)))
        sx = float(np.mean([p[0] for p in self.pose_delta_history]))
        sy = float(np.mean([p[1] for p in self.pose_delta_history]))
        return sx, sy

    def update_pose_from_orb(self, dx_px: float, dy_px: float, timestamp: float, motion_to_meter_scale: float) -> PoseSample:
        """
        Legacy wrapper for updating pose from pixel-based motion.
        """
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
        """
        Updates the internal pose estimate based on camera optical flow and rotation.
        """
        tx = float(dx_px) * float(motion_to_meter_scale)
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
        """
        Updates the pose using camera intrinsics and depth to estimate metric translation.
        """
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
        """
        Searches the pose history for recent revisits to previously mapped areas.
        """
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
        """
        Implements the state machine for detecting and smoothing loop-closure corrections.
        """
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
        """
        Generates the corrected pose estimate by merging raw odometry with pending loop-closure deltas.
        """
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
        """
        Estimates the distance to an object based on its relative bounding box area.
        """
        area_ratio = max(1e-6, float(area_ratio))
        return float(np.clip(0.30 / math.sqrt(area_ratio), 0.4, 4.0))

    def _class_range_adjustment(self, label: str, rng: float) -> float:
        """
        Applies class-specific multipliers to the distance estimate for better accuracy.
        """
        cfg = self.range_hint_by_label.get(label)
        if not cfg:
            return rng
        return float(np.clip(rng * float(cfg["multiplier"]), float(cfg["min"]), float(cfg["max"])))

    def _detection_anchor(self, det: dict, frame_shape: Sequence[int]) -> Tuple[float, float]:
        """
        Finds the base anchoring point (usually bottom-center) of a detected object.
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if "bbox" in det:
            x1, y1, x2, y2 = det["bbox"]
            x1 = float(np.clip(x1, 0, w - 1))
            x2 = float(np.clip(x2, 0, w - 1))
            y2 = float(np.clip(y2, 0, h - 1))
            return (0.5 * (x1 + x2), y2)
        cx, cy = det.get("center", (w / 2.0, h / 2.0))
        return float(np.clip(cx, 0, w - 1)), float(np.clip(cy, 0, h - 1))

    def _depth_range_for_detection(self, det: dict, depth_map: Optional[np.ndarray]) -> Optional[float]:
        """
        Extracts a metric depth value for a detection from an external depth map.
        """
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
            box_h = max(1, y2 - y1 + 1)
            lower_start = y1 + int(round(box_h * 0.65))
            lower_start = int(np.clip(lower_start, y1, y2))
            crop = depth_map[lower_start:y2 + 1, x1:x2 + 1]
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
        """
        Calculates the angular bearing of a detection relative to the camera center.
        """
        if self.camera_calibration is not None:
            return math.atan2(float(cx) - self.camera_calibration.cx, max(1e-6, self.camera_calibration.fx))
        return ((float(cx) / max(1.0, float(frame_width))) - 0.5) * math.radians(self.camera_fov_deg)

    def _weighted_step(self, base_value: float, confidence: float, strong: bool) -> float:
        """
        Scales a mapping update increment based on object detection confidence.
        """
        if not self.confidence_weighting:
            return base_value * (1.0 if strong else 0.5)
        conf = float(np.clip(confidence, 0.0, 1.0))
        weight = 0.35 + (conf ** max(0.1, self.confidence_strength))
        
        if conf > 0.90:
            weight *= 1.5
            
        if not strong:
            weight *= 0.55
        return base_value * weight

    def _is_persistent_detection(self, track_id: int, label: str, cell: Tuple[int, int]) -> bool:
        """
        Checks if a detection is stable across multiple frames before applying strong map updates.
        """
        key = (str(label), int(track_id))
        hist = self.detection_cell_history.setdefault(key, deque(maxlen=self.obstacle_persistence_frames))
        hist.append(cell)
        if len(hist) < self.obstacle_persistence_frames:
            return False
        return len(set(hist)) <= 2

    def _smoothed_projection_cell(self, track_id: int, label: str, cell: Tuple[int, int]) -> Tuple[int, int]:
        """
        Reduces jitter in projected grid coordinates for tracked objects.
        """
        key = (str(label), int(track_id))
        prev = self.object_projection_history.get(key)
        if prev is None:
            self.object_projection_history[key] = cell
            return cell
        if max(abs(cell[0] - prev[0]), abs(cell[1] - prev[1])) <= 1:
            self.object_projection_history[key] = prev
            return prev
        self.object_projection_history[key] = cell
        return cell

    def _footprint_shape_for_label(self, label: str) -> str:
        """
        Determines the geometric shape used to mark an object in the grid.
        """
        shape = self.obstacle_footprint_shape
        if self.obstacle_footprint_radius_cells == 0 and shape == "square":
            shape = self.default_footprint_shape
        if shape == "class_aware":
            if label == "person":
                return "vertical"
            if label in ("table", "sofa", "tv"):
                return "horizontal"
            if label == "chair":
                return "cross"
            return "square"
        return shape

    def _footprint_radius_for_detection(self, center: Tuple[int, int], label: str, det: Optional[dict] = None) -> int:
        """
        Calculates the grid radius of an object's footprint based on its class and image size.
        """
        radius = self.obstacle_footprint_radius_cells
        if radius <= 0:
            radius = int(self.default_footprint_radius_by_label.get(label, 0))
        
        if label == "person":
            radius += 2
            
        if not det or "bbox" not in det:
            return max(0, int(radius))
        x1, y1, x2, y2 = det["bbox"]
        box_w = max(1.0, float(x2) - float(x1))
        box_h = max(1.0, float(y2) - float(y1))
        box_span = max(box_w, box_h)
        frame_h = max(1.0, float(det.get("frame_h", 480)))
        frame_w = max(1.0, float(det.get("frame_w", 640)))
        span_ratio = box_span / max(1.0, min(frame_h, frame_w))
        if span_ratio >= 0.28:
            radius += 2
        elif span_ratio >= 0.16:
            radius += 1
        return max(0, int(radius))

    def project_detection_to_world(
        self,
        det: dict,
        frame_shape: Sequence[int],
        depth_map: Optional[np.ndarray] = None,
    ) -> Tuple[Tuple[float, float], Tuple[int, int], Tuple[float, float]]:
        """
        Projects an image-space detection into metric 3D world coordinates.
        """
        pose_ref = self.mapping_pose()
        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy = self._detection_anchor(det, frame_shape)
        area = max(1.0, float(det.get("area", 1.0)))
        area_ratio = area / float(max(1, w * h))
        bearing = self._bearing_for_detection(float(cx), w)
        depth_range = self._depth_range_for_detection(det, depth_map)
        if self.mapping_backend in ("depth", "orb_slam_like") and depth_range is not None:
            rng = depth_range
        else:
            rng = self._range_from_area_ratio(area_ratio)
            rng = self._class_range_adjustment(str(det.get("label", "")), rng)
        ang = pose_ref.theta + bearing
        wx = pose_ref.x + rng * math.cos(ang)
        wy = pose_ref.y + rng * math.sin(ang)
        gx, gy = self.world_to_grid(wx, wy)
        smooth_gx, smooth_gy = self._smoothed_projection_cell(int(det.get("track_id", -1)), str(det.get("label", "")), (gx, gy))
        smooth_wx, smooth_wy = self.grid_to_world(smooth_gx, smooth_gy)
        return (smooth_wx, smooth_wy), (smooth_gx, smooth_gy), (wx, wy)

    def _ray_cells(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Traces a ray between two grid points using Bresenham's algorithm.
        """
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
        """
        Logs a grid cell update event for quality metrics.
        """
        row = self.cell_event_counts.setdefault(cell, {"hit": 0, "free": 0})
        row[key] = row.get(key, 0) + 1

    def _obstacle_footprint_cells(self, center: Tuple[int, int], label: str = "", det: Optional[dict] = None) -> List[Tuple[int, int]]:
        """
        Generates the set of grid cells covered by an object's footprint.
        """
        cx, cy = center
        radius = self._footprint_radius_for_detection(center, label, det=det)
        shape = self._footprint_shape_for_label(label)

        cells = []
        for gy in range(cy - radius, cy + radius + 1):
            if gy < 0 or gy >= self.grid_size:
                continue
            for gx in range(cx - radius, cx + radius + 1):
                if gx < 0 or gx >= self.grid_size:
                    continue
                if shape == "horizontal" and gy != cy:
                    continue
                if shape == "vertical" and gx != cx:
                    continue
                if shape == "cross" and gx != cx and gy != cy:
                    continue
                cells.append((gx, gy))
        return cells

    def update_from_tracked(
        self,
        tracked: Dict[int, dict],
        frame_shape: Sequence[int],
        frame_index: int,
        timestamp: float,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[MapEvent]:
        """
        Core mapping function: updates the grid with obstacles and free-space rays.
        """
        self.grid = np.clip(self.grid * self.decay, 0.0, 1.0)
        pose_ref = self.mapping_pose()
        pose_cell = self.world_to_grid(pose_ref.x, pose_ref.y)
        events: List[MapEvent] = []
        frame_cells: Set[Tuple[int, int]] = set()
        strong_count = 0
        weak_count = 0

        for obj_id, obj in tracked.items():
            label = obj.get("label", "other")
            if label not in ("person", "chair", "table", "sofa", "tv"):
                continue
            obj_with_meta = dict(obj)
            obj_with_meta["track_id"] = int(obj_id)
            obj_with_meta["frame_h"] = int(frame_shape[0])
            obj_with_meta["frame_w"] = int(frame_shape[1])
            (wx, wy), (gx, gy), (anchor_wx, anchor_wy) = self.project_detection_to_world(obj_with_meta, frame_shape, depth_map=depth_map)
            conf = float(obj.get("confidence", 0.0))
            is_persistent = self._is_persistent_detection(int(obj_id), str(label), (gx, gy))
            strong_count += 1 if is_persistent else 0
            weak_count += 0 if is_persistent else 1
            inc = self._weighted_step(self.obstacle_increment, conf, strong=is_persistent)
            footprint_cells = self._obstacle_footprint_cells((gx, gy), label=str(label), det=obj_with_meta)
            events.append(
                MapEvent(
                    event_type="obstacle_anchor",
                    grid_xy=(gx, gy),
                    world_xy=(wx, wy),
                    label=label,
                    confidence=conf,
                    track_id=int(obj_id),
                    timestamp=float(timestamp),
                    anchor_grid_xy=(gx, gy),
                    anchor_world_xy=(anchor_wx, anchor_wy),
                    footprint_role="anchor",
                )
            )
            for cell in footprint_cells:
                cell_x, cell_y = cell
                self.grid[cell_y, cell_x] = min(1.0, self.grid[cell_y, cell_x] + inc)
                self._record_cell_event(cell, "hit")
                frame_cells.add(cell)
                events.append(
                    MapEvent(
                        event_type="obstacle_mark",
                        grid_xy=cell,
                        world_xy=self.grid_to_world(cell_x, cell_y),
                        label=label,
                        confidence=conf,
                        track_id=int(obj_id),
                        timestamp=float(timestamp),
                        anchor_grid_xy=(gx, gy),
                        anchor_world_xy=(anchor_wx, anchor_wy),
                        footprint_role=("anchor" if cell == (gx, gy) else "footprint"),
                    )
                )

            obstacle_cells = set(footprint_cells)
            ray = self._ray_cells(pose_cell, (gx, gy))
            carve_cells = list(ray[:-1])
            while carve_cells and carve_cells[-1] in obstacle_cells:
                carve_cells.pop()
            if carve_cells:
                carve_cells = carve_cells[:-1]
            for cell_idx, (rx, ry) in enumerate(carve_cells):
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
                        anchor_grid_xy=(gx, gy),
                        anchor_world_xy=(anchor_wx, anchor_wy),
                        footprint_role="ray",
                    )
                )

        current_frame_cells = set(frame_cells)
        if self.obstacle_temporal_persistence_frames > 0:
            for past_cells in self.temporal_obstacle_history:
                frame_cells.update(past_cells)
            self.temporal_obstacle_history.append(current_frame_cells)

        self.frame_obstacles[int(frame_index)] = frame_cells
        self.map_events.extend(events)
        self.last_frame_obstacle_counts = {"strong": int(strong_count), "weak": int(weak_count)}
        return events

    def render_map(self, out_size: int = 320) -> np.ndarray:
        """
        Renders the occupancy grid and robot trajectory into a visual BGR image.
        """
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
        """
        Calculates path length and sample count statistics for the run.
        """
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
        """
        Provides a summary of the active mapping backend and configuration.
        """
        return {
            "backend": self.mapping_backend,
            "status": self.backend_status,
            "has_camera_calibration": self.camera_calibration is not None,
            "depth_unit_scale": float(self.depth_unit_scale),
            "inverse_depth": bool(self.inverse_depth),
            "projection_anchor_mode": self.projection_anchor_mode,
        }

    def event_summary(self) -> dict:
        """
        Summarizes map update events and unique cell activations.
        """
        anchor_events = [e for e in self.map_events if e.event_type == "obstacle_anchor"]
        footprint_events = [e for e in self.map_events if e.event_type == "obstacle_mark"]
        return {
            "projection_anchor_mode": self.projection_anchor_mode,
            "anchor_events": len(anchor_events),
            "footprint_events": len(footprint_events),
            "anchor_cells_unique": len({e.grid_xy for e in anchor_events}),
            "footprint_cells_unique": len({e.grid_xy for e in footprint_events}),
        }

    def loop_closure_summary(self) -> dict:
        """
        Summarizes loop closure detection and correction metrics.
        """
        if self.loop_closure_correction_records:
            mean_t = float(np.mean([r["translation_m"] for r in self.loop_closure_correction_records]))
            mean_h = float(np.mean([r["heading_rad"] for r in self.loop_closure_correction_records]))
        else:
            mean_t = 0.0
            mean_h = 0.0
        
        alignment_score = 0.0
        if self.post_closure_alignment_dists:
            alignment_score = float(np.mean(self.post_closure_alignment_dists))

        return {
            "state": self.loop_closure_state,
            "candidates_detected": self.loop_closure_candidate_count,
            "corrections_applied": self.loop_closure_corrections_applied,
            "rejections": self.loop_closure_rejections,
            "mean_correction_translation_m": mean_t,
            "mean_correction_heading_rad": mean_h,
            "post_closure_path_alignment_score": alignment_score,
        }
