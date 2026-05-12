import argparse
import math
import os
import threading
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from mapping_runtime import (
    ActionSample,
    LiveMapper,
    compute_label_metrics,
    compute_loop_closure_drift,
    compute_map_consistency_score,
    compute_mapping_quality_summary,
    compute_obstacle_persistence_stability,
    compute_obstacle_precision_recall,
    compute_occupancy_confidence_concentration,
    compute_pose_jitter_score,
    dataclass_to_dict,
    load_run_annotations,
    parse_action_label,
    action_confidence_from_tracked,
    write_joint_report,
)
from reasoning import ReasoningEngine


class SharedState:
    def __init__(self):
        self.frame = None
        self.detections = []
        self.tracked = {}
        self.decision = "Initializing..."
        self.engine_state = "EXPLORE"
        self.motion_text = "No movement"
        self.use_model = True
        self.running = True
        self.fps = 0
        self.lock = threading.Lock()


state = SharedState()


def detection_worker(model_path="yolov8n.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Detection] Running on: {device.upper()}")
    model = YOLO(model_path)

    while state.running:
        with state.lock:
            if state.frame is None:
                continue
            frame = state.frame.copy()

        results = model(frame, imgsz=320, verbose=False, device=device)
        dets = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                lbl = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append(
                    {
                        "label": lbl,
                        "confidence": conf,
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "bbox": (x1, y1, x2, y2),
                        "area": (x2 - x1) * (y2 - y1),
                    }
                )

        with state.lock:
            state.detections = dets

        time.sleep(0.005)


def reasoning_worker():
    engine = ReasoningEngine()
    while state.running:
        with state.lock:
            dets = state.detections.copy()
            motion = state.motion_text
            use_model = state.use_model

        decision, tracked = engine.decide(dets, (320, 240), 640 * 480, motion, use_model)

        with state.lock:
            state.tracked = {k: dict(v) for k, v in tracked.items()}
            state.decision = decision
            state.engine_state = engine.state

        time.sleep(0.04)


C_PERSON = (60, 60, 220)
C_CHAIR = (30, 160, 255)
C_TABLE = (50, 200, 100)
C_OTHER = (180, 180, 180)
C_CYAN = (255, 220, 40)
C_WHITE = (255, 255, 255)
C_GREEN = (60, 220, 60)
C_ORANGE = (40, 140, 255)

LABEL_COLOR = {
    "person": C_PERSON,
    "chair": C_CHAIR,
    "table": C_TABLE,
}

STATE_COLOR = {
    "AVOID": (60, 60, 220),
    "TARGET": (30, 160, 255),
    "INVESTIGATE": (50, 200, 100),
    "EXPLORE": (200, 200, 200),
}


def draw_hud_panel(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 1)


def draw_text(frame, text, pos, color=C_WHITE, scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_bbox(frame, obj, obj_id):
    x1, y1, x2, y2 = obj["bbox"]
    label = obj["label"]
    conf = obj["confidence"]
    color = LABEL_COLOR.get(label, C_OTHER)

    corner = 14
    t = 2
    for (px, py, dx, dy) in [
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ]:
        cv2.line(frame, (px, py), (px + dx * corner, py), color, t + 1)
        cv2.line(frame, (px, py), (px, py + dy * corner), color, t + 1)

    chip_text = f"#{obj_id} {label.upper()} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    chip_y = max(y1 - 24, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, chip_y), (x1 + tw + 8, chip_y + th + 8), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, chip_text, (x1 + 4, chip_y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1, cv2.LINE_AA)


def draw_motion_arrow(frame, arrow):
    cx, cy = 320, 240
    ex, ey = cx + arrow[0], cy + arrow[1]
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_GREEN, 2, tipLength=0.3)
    cv2.circle(frame, (cx, cy), 5, C_GREEN, -1)


def draw_confidence_bars(frame, detections, panel_x, start_y):
    for i, det in enumerate(detections[:4]):
        y = start_y + i * 22
        lbl = det["label"]
        conf = det["confidence"]
        color = LABEL_COLOR.get(lbl, C_OTHER)
        bar_w = int(conf * 90)
        draw_text(frame, f"{lbl[:6]}", (panel_x + 8, y + 11), color, 0.38)
        cv2.rectangle(frame, (panel_x + 52, y), (panel_x + 52 + bar_w, y + 12), color, -1)
        cv2.rectangle(frame, (panel_x + 52, y), (panel_x + 142, y + 12), (70, 70, 70), 1)
        draw_text(frame, f"{conf:.0%}", (panel_x + 148, y + 11), C_WHITE, 0.36)


def robust_orb_flow_delta(prev_kp, prev_des, kp, des, matcher, top_k: int = 80):
    if prev_des is None or des is None or len(prev_des) < 8 or len(des) < 8:
        return 0.0, 0.0, 0.0, 0.0
    matches = matcher.match(prev_des, des)
    if len(matches) < 12:
        return 0.0, 0.0, 0.0, 0.0
    matches = sorted(matches, key=lambda m: m.distance)[: max(12, int(top_k))]
    src = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    aff, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if aff is None or inliers is None:
        return 0.0, 0.0, 0.0, 0.0
    inlier_mask_aff = inliers.ravel().astype(bool)
    inlier_ratio = float(np.mean(inlier_mask_aff)) if inlier_mask_aff.size else 0.0
    if np.sum(inlier_mask_aff) < 8:
        return 0.0, 0.0, 0.0, inlier_ratio

    dxs = np.array([kp[m.trainIdx].pt[0] - prev_kp[m.queryIdx].pt[0] for m in matches], dtype=np.float32)
    dys = np.array([kp[m.trainIdx].pt[1] - prev_kp[m.queryIdx].pt[1] for m in matches], dtype=np.float32)
    med_dx = float(np.median(dxs))
    med_dy = float(np.median(dys))
    residual = np.sqrt((dxs - med_dx) ** 2 + (dys - med_dy) ** 2)
    mad = float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
    thr = max(1.5, 3.5 * mad)
    inlier_mask = (residual <= thr) & inlier_mask_aff
    if int(np.sum(inlier_mask)) < 8:
        return 0.0, 0.0, 0.0, inlier_ratio
    in_dx = float(np.mean(dxs[inlier_mask]))
    in_dy = float(np.mean(dys[inlier_mask]))
    dtheta = float(np.arctan2(aff[1, 0], aff[0, 0]))
    return in_dx, in_dy, dtheta, inlier_ratio


def parse_args():
    p = argparse.ArgumentParser(description="AI Visual Navigation System")
    p.add_argument("--video", default="", help="Path to a video file. Empty uses webcam.")
    p.add_argument("--camera", type=int, default=0, help="Webcam device index.")
    p.add_argument("--loop", action="store_true", help="Loop video file.")
    p.add_argument("--no-depth", action="store_true", help="Disable MiDaS depth estimation.")

    p.add_argument("--no-mapping", action="store_true", help="Disable live occupancy-grid mapping.")
    p.add_argument("--map-grid-size", type=int, default=120, help="Occupancy grid size in cells.")
    p.add_argument("--map-meters-per-cell", type=float, default=0.10, help="Map scale in meters per cell.")
    p.add_argument("--map-decay", type=float, default=0.985, help="Per-frame occupancy decay.")
    p.add_argument("--map-obstacle-increment", type=float, default=0.20, help="Obstacle occupancy increment.")
    p.add_argument("--map-free-decrement", type=float, default=0.05, help="Free-space decrement along rays.")
    p.add_argument("--map-camera-fov-deg", type=float, default=70.0, help="Camera horizontal FOV in degrees.")
    p.add_argument("--map-ray-step-cells", type=int, default=1, help="Step size for free-space rays.")
    p.add_argument("--pose-motion-to-meter-scale", type=float, default=0.0025, help="ORB px-motion to meter scale.")
    p.add_argument("--pose-motion-deadband-px", type=float, default=0.8, help="Ignore tiny ORB motion under this pixel magnitude.")
    p.add_argument("--pose-min-flow-quality", type=float, default=0.30, help="Minimum ORB inlier ratio required to apply pose update.")
    p.add_argument("--pose-jitter-min-motion-m", type=float, default=0.002, help="Min per-frame motion used in jitter evaluation.")
    p.add_argument("--pose-smoothing-window", type=int, default=5, help="Smoothing window for pose deltas.")
    p.add_argument("--pose-max-translation-m", type=float, default=0.08, help="Max translation per frame in meters.")
    p.add_argument("--pose-max-rotation-rad", type=float, default=0.35, help="Max heading change per frame in radians.")
    p.add_argument("--map-confidence-weighting", action="store_true", default=True, help="Enable confidence-weighted occupancy updates.")
    p.add_argument("--no-map-confidence-weighting", action="store_false", dest="map_confidence_weighting", help="Disable confidence-weighted occupancy updates.")
    p.add_argument("--map-confidence-strength", type=float, default=1.0, help="Exponent controlling confidence-weighting strength.")
    p.add_argument("--map-obstacle-persistence-frames", type=int, default=2, help="Frames required for strong obstacle reinforcement.")
    p.add_argument("--map-loop-closure-enabled", action="store_true", default=True, help="Enable lightweight loop-closure snap-back.")
    p.add_argument("--no-map-loop-closure-enabled", action="store_false", dest="map_loop_closure_enabled", help="Disable loop-closure snap-back.")
    p.add_argument("--map-loop-closure-radius-m", type=float, default=0.25, help="Revisit radius for loop-closure candidates.")
    p.add_argument("--map-loop-closure-min-frame-gap", type=int, default=80, help="Minimum frame gap between closure candidates.")
    p.add_argument("--map-loop-closure-max-heading-delta-rad", type=float, default=0.55, help="Maximum heading delta for closure candidates.")
    p.add_argument("--map-loop-closure-correction-alpha", type=float, default=0.25, help="Smoothing factor for applying loop-closure corrections.")
    p.add_argument("--map-loop-closure-cooldown-frames", type=int, default=45, help="Cooldown frames after applying a loop closure.")
    p.add_argument("--map-render-raw-trajectory", action="store_true", default=False, help="Overlay raw (uncorrected) trajectory in map view.")
    p.add_argument("--map-require-benchmark-for-promotion", action="store_true", default=True, help="Require GT benchmark lane to mark mapping as promotable.")
    p.add_argument("--no-map-require-benchmark-for-promotion", action="store_false", dest="map_require_benchmark_for_promotion", help="Allow unsupervised lane to be promotable.")
    p.add_argument("--map-gate-pose-jitter-min", type=float, default=0.40, help="Minimum pose jitter score gate.")
    p.add_argument("--map-gate-benchmark-f1-min", type=float, default=0.45, help="Minimum obstacle F1 gate in benchmark lane.")

    p.add_argument(
        "--run-annotations",
        default="",
        help="Optional JSON annotations with frame_labels and obstacles for run metrics.",
    )
    p.add_argument(
        "--run-report-out",
        default="",
        help="Output path for combined label+map report (default: reports/runtime/run_report_<ts>.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_label = f"VIDEO: {os.path.basename(args.video)}"
        is_video_file = True
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = max(1, int(1000 / source_fps))
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_label = f"LIVE CAM #{args.camera}"
        is_video_file = False
        frame_delay = 1

    use_depth = not args.no_depth
    if use_depth:
        print("[Depth] Loading MiDaS...")
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas.to(device)
        midas.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.small_transform
        print("[Depth] MiDaS ready.")
    else:
        print("[Depth] Disabled.")

    use_mapping = not args.no_mapping
    mapper = None
    if use_mapping:
        mapper = LiveMapper(
            grid_size=args.map_grid_size,
            meters_per_cell=args.map_meters_per_cell,
            decay=args.map_decay,
            obstacle_increment=args.map_obstacle_increment,
            free_decrement=args.map_free_decrement,
            camera_fov_deg=args.map_camera_fov_deg,
            ray_step_cells=args.map_ray_step_cells,
            pose_smoothing_window=args.pose_smoothing_window,
            max_translation_m_per_frame=args.pose_max_translation_m,
            max_rotation_rad_per_frame=args.pose_max_rotation_rad,
            confidence_weighting=args.map_confidence_weighting,
            confidence_strength=args.map_confidence_strength,
            obstacle_persistence_frames=args.map_obstacle_persistence_frames,
            loop_closure_enabled=args.map_loop_closure_enabled,
            loop_closure_radius_m=args.map_loop_closure_radius_m,
            loop_closure_min_frame_gap=args.map_loop_closure_min_frame_gap,
            loop_closure_max_heading_delta_rad=args.map_loop_closure_max_heading_delta_rad,
            loop_closure_correction_alpha=args.map_loop_closure_correction_alpha,
            loop_closure_cooldown_frames=args.map_loop_closure_cooldown_frames,
            render_raw_trajectory=args.map_render_raw_trajectory,
        )

    annotations_path = args.run_annotations
    default_benchmark_gt = "data/annotations/mapping_benchmark_gt.json"
    if not annotations_path and args.video and os.path.exists(default_benchmark_gt):
        annotations_path = default_benchmark_gt
        print(f"[Eval] Auto-using benchmark GT: {annotations_path}")

    run_annotations = load_run_annotations(annotations_path)
    if run_annotations.get("available"):
        print(f"[Eval] Loaded annotations: {annotations_path}")
    elif annotations_path:
        print(f"[Eval] {run_annotations.get('reason', 'Annotation unavailable')}")

    orb = cv2.ORB_create(nfeatures=400)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp, prev_des = None, None
    session_csv = f"data/raw/session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("data/raw", exist_ok=True)

    action_samples = []
    pred_actions_by_frame = {}
    map_events = []
    frame_idx = 0
    run_start_ts = time.time()

    label_msg = ""
    label_color = C_GREEN
    label_timer = 0.0

    fps_t0 = time.time()
    fps_count = 0

    threading.Thread(target=detection_worker, daemon=True).start()
    threading.Thread(target=reasoning_worker, daemon=True).start()

    print(f"[Main] Source: {source_label}")
    print(f"[Main] Session CSV: {session_csv}")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video_file and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if is_video_file:
                print("[Main] End of video.")
            else:
                print("[Main] Camera read failed.")
            break

        frame_idx += 1
        now_ts = time.time()

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if use_depth:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = transform(rgb).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = F.interpolate(prediction.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False).squeeze()
            depth_map = prediction.cpu().numpy()
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)

        with state.lock:
            state.frame = frame.copy()
            snap_tracked = {k: dict(v) for k, v in state.tracked.items()}
            snap_dets = list(state.detections)
            snap_decision = state.decision
            snap_state = state.engine_state
            snap_mode = state.use_model

        kp, des = orb.detectAndCompute(gray, None)
        motion_text = "No movement"
        arrow = None
        dx = 0.0
        dy = 0.0
        dtheta = 0.0
        flow_quality = 0.0
        if prev_kp is not None and prev_des is not None and kp is not None and des is not None:
            dx, dy, dtheta, flow_quality = robust_orb_flow_delta(prev_kp, prev_des, kp, des, bf, top_k=80)
            if math.hypot(dx, dy) < float(args.pose_motion_deadband_px):
                dx, dy, dtheta = 0.0, 0.0, 0.0
            if flow_quality < float(args.pose_min_flow_quality):
                dx, dy, dtheta = 0.0, 0.0, 0.0
            if abs(dx) > abs(dy):
                if dx > 2:
                    motion_text, arrow = "Moving Right", (80, 0)
                if dx < -2:
                    motion_text, arrow = "Moving Left", (-80, 0)
            else:
                if dy > 2:
                    motion_text, arrow = "Moving Down", (0, 80)
                if dy < -2:
                    motion_text, arrow = "Moving Up", (0, -80)
        prev_kp, prev_des = kp, des

        if use_mapping and mapper is not None:
            mapper.update_pose_from_flow(
                dx_px=dx,
                dy_px=dy,
                dtheta_rad=dtheta,
                timestamp=now_ts,
                motion_to_meter_scale=args.pose_motion_to_meter_scale,
                flow_quality=flow_quality,
            )

        with state.lock:
            state.motion_text = motion_text

        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            state.fps = int(fps_count / elapsed)
            fps_count = 0
            fps_t0 = time.time()

        out = frame.copy()
        for obj_id, obj in snap_tracked.items():
            draw_bbox(out, obj, obj_id)

        if use_mapping and mapper is not None:
            frame_events = mapper.update_from_tracked(tracked=snap_tracked, frame_shape=frame.shape, frame_index=frame_idx, timestamp=now_ts)
            map_events.extend(frame_events)

        if arrow:
            draw_motion_arrow(out, arrow)

        action_label = parse_action_label(snap_decision)
        action_conf = action_confidence_from_tracked(action_label, snap_tracked)
        action_samples.append(
            ActionSample(
                action=action_label,
                confidence=action_conf,
                source_mode=("model" if snap_mode else "rule"),
                timestamp=now_ts,
            )
        )
        pred_actions_by_frame[frame_idx] = action_label

        draw_hud_panel(out, 0, 0, 640, 38, alpha=0.60)
        state_col = STATE_COLOR.get(snap_state, C_WHITE)
        draw_text(out, f"  {snap_decision}", (4, 26), state_col, 0.65, 2)
        draw_text(out, f"{state.fps} FPS", (565, 26), C_GREEN, 0.60, 1)

        draw_hud_panel(out, 0, 38, 175, 200, alpha=0.55)
        draw_text(out, "STATE", (8, 58), (130, 130, 130), 0.38)
        draw_text(out, snap_state, (8, 76), state_col, 0.52, 1)
        draw_text(out, "MOTION", (8, 100), (130, 130, 130), 0.38)
        draw_text(out, motion_text, (8, 118), C_WHITE, 0.46, 1)

        mode_lbl = "AI MODEL" if snap_mode else "RULE-BASED"
        mode_col = C_CYAN if snap_mode else C_ORANGE
        draw_text(out, "MODE", (8, 142), (130, 130, 130), 0.38)
        draw_text(out, mode_lbl, (8, 160), mode_col, 0.50, 1)
        draw_text(out, source_label, (8, 182), (100, 100, 100), 0.34, 1)

        if snap_dets:
            draw_hud_panel(out, 0, 380, 175, 100, alpha=0.55)
            draw_text(out, "DETECTIONS", (8, 398), (130, 130, 130), 0.38)
            draw_confidence_bars(out, snap_dets, 0, 403)

        if label_msg and time.time() - label_timer < 2.0:
            draw_hud_panel(out, 0, 450, 640, 30, alpha=0.70)
            draw_text(out, label_msg, (10, 470), label_color, 0.55, 1)
        elif time.time() - label_timer >= 2.0:
            label_msg = ""

        draw_text(out, "ESC:exit  M:toggle  A/C/T/E:label", (178, 472), (100, 100, 100), 0.38)

        cv2.imshow("AI Visual Navigation System", out)
        if use_mapping and mapper is not None:
            cv2.imshow("Occupancy Map", mapper.render_map(out_size=300))
        if use_depth:
            cv2.imshow("Depth Map", depth_colored)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 27:
            state.running = False
            break
        elif key == ord("m"):
            with state.lock:
                state.use_model = not state.use_model
            label_msg = f"Switched to: {'AI MODEL' if state.use_model else 'RULE-BASED'}"
            label_color = C_CYAN
            label_timer = time.time()
        elif key in [ord("a"), ord("c"), ord("t"), ord("e")]:
            lmap = {ord("a"): "AVOID_PERSON", ord("c"): "MOVE_TO_CHAIR", ord("t"): "CHECK_TABLE", ord("e"): "EXPLORE"}
            action = lmap[key]
            logger = ReasoningEngine()
            ok = logger.log_example(snap_dets, (320, 240), 640 * 480, motion_text, action, session_csv)
            if ok:
                label_msg = f"[OK]  Saved: {action}"
                label_color = C_GREEN
            else:
                label_msg = "[X]  Rejected — low confidence"
                label_color = C_ORANGE
            label_timer = time.time()

    cap.release()
    cv2.destroyAllWindows()

    run_end_ts = time.time()
    if run_annotations.get("available", False):
        label_metrics = compute_label_metrics(run_annotations.get("frame_labels", {}), pred_actions_by_frame)
        obstacle_pr = compute_obstacle_precision_recall(
            run_annotations.get("obstacles_by_frame", {}),
            mapper.frame_obstacles if mapper is not None else {},
        )
    else:
        reason = run_annotations.get("reason", "No annotation file provided")
        label_metrics = {"available": False, "reason": reason}
        obstacle_pr = {"available": False, "reason": reason}

    if mapper is not None:
        loop_closure = compute_loop_closure_drift(mapper.pose_history)
        corrected_loop_closure = compute_loop_closure_drift(mapper.corrected_pose_history)
        consistency = compute_map_consistency_score(mapper.cell_event_counts)
        pose_jitter = compute_pose_jitter_score(mapper.pose_history, min_motion_m=args.pose_jitter_min_motion_m)
        obstacle_persistence = compute_obstacle_persistence_stability(mapper.frame_obstacles)
        occupancy_confidence = compute_occupancy_confidence_concentration(mapper.grid)
        lc_summary = mapper.loop_closure_summary()
        mapping_quality_summary = compute_mapping_quality_summary(
            loop_closure_drift=loop_closure,
            map_consistency_score=consistency,
            pose_jitter=pose_jitter,
            obstacle_persistence=obstacle_persistence,
            occupancy_concentration=occupancy_confidence,
            obstacle_precision_recall=obstacle_pr,
            require_benchmark=bool(args.map_require_benchmark_for_promotion),
            threshold_overrides={
                "pose_jitter_min": float(args.map_gate_pose_jitter_min),
                "benchmark_obstacle_f1_min": float(args.map_gate_benchmark_f1_min),
            },
        )
        map_metrics = {
            "evaluation_lane": "benchmark_supervised" if obstacle_pr.get("available") else "live_unsupervised",
            "loop_closure_drift": loop_closure,
            "loop_closure_drift_corrected": corrected_loop_closure,
            "loop_closure_drift_delta": {
                "available": bool(loop_closure.get("available") and corrected_loop_closure.get("available")),
                "translation_error_mean_m_delta": float(loop_closure.get("translation_error_mean_m", 0.0) - corrected_loop_closure.get("translation_error_mean_m", 0.0))
                if loop_closure.get("available") and corrected_loop_closure.get("available")
                else 0.0,
                "heading_error_mean_rad_delta": float(loop_closure.get("heading_error_mean_rad", 0.0) - corrected_loop_closure.get("heading_error_mean_rad", 0.0))
                if loop_closure.get("available") and corrected_loop_closure.get("available")
                else 0.0,
            },
            "map_consistency_score": consistency,
            "pose_jitter": pose_jitter,
            "obstacle_persistence_stability": obstacle_persistence,
            "occupancy_confidence_concentration": occupancy_confidence,
            "obstacle_precision_recall": obstacle_pr,
            "loop_closure_corrections_applied": lc_summary.get("corrections_applied", 0),
            "mean_correction_translation_m": lc_summary.get("mean_correction_translation_m", 0.0),
            "mean_correction_heading_rad": lc_summary.get("mean_correction_heading_rad", 0.0),
            "post_closure_path_alignment_score": lc_summary.get("post_closure_path_alignment_score", 0.0),
            "loop_closure_state": lc_summary.get("state", "idle"),
            "loop_closure_summary": lc_summary,
            "mapping_quality_summary": mapping_quality_summary,
        }
        pose_stats = mapper.pose_stats()
    else:
        map_metrics = {
            "evaluation_lane": "mapping_disabled",
            "loop_closure_drift": {"available": False, "reason": "Mapping disabled"},
            "map_consistency_score": {"available": False, "reason": "Mapping disabled"},
            "pose_jitter": {"available": False, "reason": "Mapping disabled"},
            "obstacle_persistence_stability": {"available": False, "reason": "Mapping disabled"},
            "occupancy_confidence_concentration": {"available": False, "reason": "Mapping disabled"},
            "obstacle_precision_recall": obstacle_pr,
            "mapping_quality_summary": {
                "lane": "mapping_disabled",
                "status": "not_available",
                "promotable": False,
                "require_benchmark": bool(args.map_require_benchmark_for_promotion),
            },
        }
        pose_stats = {"available": False, "reason": "Mapping disabled"}

    report_out = args.run_report_out or f"reports/runtime/run_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "label_metrics": label_metrics,
        "map_metrics": map_metrics,
        "pose_stats": pose_stats,
        "config": {
            "source_label": source_label,
            "video": args.video,
            "camera": args.camera,
            "use_depth": bool(use_depth),
            "use_mapping": bool(use_mapping),
            "map_grid_size": int(args.map_grid_size),
            "map_meters_per_cell": float(args.map_meters_per_cell),
            "map_decay": float(args.map_decay),
            "map_obstacle_increment": float(args.map_obstacle_increment),
            "map_free_decrement": float(args.map_free_decrement),
            "map_camera_fov_deg": float(args.map_camera_fov_deg),
            "map_ray_step_cells": int(args.map_ray_step_cells),
            "pose_motion_to_meter_scale": float(args.pose_motion_to_meter_scale),
            "pose_motion_deadband_px": float(args.pose_motion_deadband_px),
            "pose_min_flow_quality": float(args.pose_min_flow_quality),
            "pose_jitter_min_motion_m": float(args.pose_jitter_min_motion_m),
            "pose_smoothing_window": int(args.pose_smoothing_window),
            "pose_max_translation_m": float(args.pose_max_translation_m),
            "pose_max_rotation_rad": float(args.pose_max_rotation_rad),
            "map_confidence_weighting": bool(args.map_confidence_weighting),
            "map_confidence_strength": float(args.map_confidence_strength),
            "map_obstacle_persistence_frames": int(args.map_obstacle_persistence_frames),
            "map_loop_closure_enabled": bool(args.map_loop_closure_enabled),
            "map_loop_closure_radius_m": float(args.map_loop_closure_radius_m),
            "map_loop_closure_min_frame_gap": int(args.map_loop_closure_min_frame_gap),
            "map_loop_closure_max_heading_delta_rad": float(args.map_loop_closure_max_heading_delta_rad),
            "map_loop_closure_correction_alpha": float(args.map_loop_closure_correction_alpha),
            "map_loop_closure_cooldown_frames": int(args.map_loop_closure_cooldown_frames),
            "map_render_raw_trajectory": bool(args.map_render_raw_trajectory),
            "map_require_benchmark_for_promotion": bool(args.map_require_benchmark_for_promotion),
            "map_gate_pose_jitter_min": float(args.map_gate_pose_jitter_min),
            "map_gate_benchmark_f1_min": float(args.map_gate_benchmark_f1_min),
            "annotations_path": annotations_path,
        },
        "timing": {
            "start_ts": float(run_start_ts),
            "end_ts": float(run_end_ts),
            "duration_s": float(run_end_ts - run_start_ts),
            "frames_processed": int(frame_idx),
        },
        "action_samples": [dataclass_to_dict(a) for a in action_samples],
        "map_events": [dataclass_to_dict(e) for e in map_events],
        "warnings": [] if run_annotations.get("available", False) or not annotations_path else [run_annotations.get("reason")],
        "failures": [],
    }
    write_joint_report(report_out, payload)
    print(f"[Report] Joint run report written: {report_out}")


if __name__ == "__main__":
    main()
