import argparse
import os
import time
import threading
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from reasoning import ReasoningEngine

# -----------------------------------------------------------------------
# SHARED STATE
# -----------------------------------------------------------------------
class SharedState:
    def __init__(self):
        self.frame        = None
        self.detections   = []
        self.tracked      = {}
        self.decision     = "Initializing..."
        self.engine_state = "EXPLORE"
        self.motion_text  = "No movement"
        self.use_model    = True
        self.running      = True
        self.fps          = 0
        self.lock         = threading.Lock()

state = SharedState()

# -----------------------------------------------------------------------
# WORKER: YOLO DETECTION
# -----------------------------------------------------------------------
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
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                lbl  = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                dets.append({
                    "label": lbl, "confidence": conf,
                    "center": ((x1+x2)//2, (y1+y2)//2),
                    "bbox": (x1, y1, x2, y2),
                    "area": (x2-x1)*(y2-y1)
                })

        with state.lock:
            state.detections = dets

        time.sleep(0.005)

# -----------------------------------------------------------------------
# WORKER: REASONING ENGINE
# -----------------------------------------------------------------------
def reasoning_worker():
    engine = ReasoningEngine()
    while state.running:
        with state.lock:
            dets      = state.detections.copy()
            motion    = state.motion_text
            use_model = state.use_model

        decision, tracked = engine.decide(dets, (320, 240), 640*480, motion, use_model)

        with state.lock:
            state.tracked      = {k: dict(v) for k, v in tracked.items()}
            state.decision     = decision
            state.engine_state = engine.state

        time.sleep(0.04)

# -----------------------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------------------
C_PERSON   = (60,  60,  220)
C_CHAIR    = (30,  160, 255)
C_TABLE    = (50,  200, 100)
C_OTHER    = (180, 180, 180)
C_CYAN     = (255, 220,  40)
C_WHITE    = (255, 255, 255)
C_YELLOW   = (0,   220, 220)
C_GREEN    = (60,  220,  60)
C_ORANGE   = (40,  140, 255)

LABEL_COLOR = {
    "person": C_PERSON,
    "chair":  C_CHAIR,
    "table":  C_TABLE,
}

STATE_COLOR = {
    "AVOID":       (60,  60,  220),
    "TARGET":      (30,  160, 255),
    "INVESTIGATE": (50,  200, 100),
    "EXPLORE":     (200, 200, 200),
}

def draw_hud_panel(frame, x, y, w, h, alpha=0.55):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (60, 60, 60), 1)

def draw_text(frame, text, pos, color=C_WHITE, scale=0.55, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_bbox(frame, obj, obj_id):
    x1, y1, x2, y2 = obj["bbox"]
    label = obj["label"]
    conf  = obj["confidence"]
    color = LABEL_COLOR.get(label, C_OTHER)

    corner = 14
    t = 2
    for (px, py, dx, dy) in [
        (x1, y1,  1,  1), (x2, y1, -1,  1),
        (x1, y2,  1, -1), (x2, y2, -1, -1)
    ]:
        cv2.line(frame, (px, py), (px + dx*corner, py), color, t+1)
        cv2.line(frame, (px, py), (px, py + dy*corner), color, t+1)

    chip_text = f"#{obj_id} {label.upper()} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(chip_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    chip_y = max(y1 - 24, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, chip_y), (x1 + tw + 8, chip_y + th + 8), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, chip_text, (x1 + 4, chip_y + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_WHITE, 1, cv2.LINE_AA)

def draw_motion_arrow(frame, arrow):
    cx, cy = 320, 240
    ex, ey = cx + arrow[0], cy + arrow[1]
    cv2.arrowedLine(frame, (cx, cy), (ex, ey), C_GREEN, 2, tipLength=0.3)
    cv2.circle(frame, (cx, cy), 5, C_GREEN, -1)

def draw_confidence_bars(frame, detections, panel_x, start_y):
    for i, det in enumerate(detections[:4]):
        y = start_y + i * 22
        lbl   = det["label"]
        conf  = det["confidence"]
        color = LABEL_COLOR.get(lbl, C_OTHER)
        bar_w = int(conf * 90)
        draw_text(frame, f"{lbl[:6]}", (panel_x + 8, y + 11), color, 0.38)
        cv2.rectangle(frame, (panel_x + 52, y), (panel_x + 52 + bar_w, y + 12), color, -1)
        cv2.rectangle(frame, (panel_x + 52, y), (panel_x + 142, y + 12), (70, 70, 70), 1)
        draw_text(frame, f"{conf:.0%}", (panel_x + 148, y + 11), C_WHITE, 0.36)

# -----------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="AI Visual Navigation System")
    p.add_argument(
        "--video", default="",
        help=(
            "Path to a video file to process instead of the live webcam. "
            "Supports any format OpenCV can read (mp4, avi, mkv, mov …). "
            "Leave empty (default) to use webcam."
        )
    )
    p.add_argument(
        "--camera", type=int, default=0,
        help="Webcam device index (default: 0). Ignored when --video is set."
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Loop the video indefinitely (only relevant with --video)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ── Open video source ──────────────────────────────────────────────
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_label = f"VIDEO: {os.path.basename(args.video)}"
        is_video_file = True
        # For video files, slow the loop to ~30 fps so it's watchable
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = max(1, int(1000 / source_fps))
    else:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        source_label  = f"LIVE CAM #{args.camera}"
        is_video_file = False
        frame_delay   = 1

    # ---------------------------------------------------
    # DEPTH ESTIMATION (MiDaS)
    # ---------------------------------------------------
    print("[Depth] Loading MiDaS...")

    midas = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small"
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms"
    )

    transform = midas_transforms.small_transform

    print("[Depth] MiDaS ready.")

    orb = cv2.ORB_create(nfeatures=400)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    prev_kp, prev_des = None, None
    session_csv = f"data/raw/session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    # ---------------------------------------------------
    # OCCUPANCY GRID
    # ---------------------------------------------------
    grid_size = 100

    occupancy = np.zeros(
        (grid_size, grid_size),
        dtype=np.uint8
    )
    os.makedirs("data/raw", exist_ok=True)

    label_msg   = ""
    label_color = C_GREEN
    label_timer = 0.0

    fps_t0    = time.time()
    fps_count = 0

    threading.Thread(target=detection_worker, daemon=True).start()
    threading.Thread(target=reasoning_worker,  daemon=True).start()

    print(f"[Main] Source: {source_label}")
    print(f"[Main] Session CSV: {session_csv}")

    while True:
        ret, frame = cap.read()

        # ── Handle end-of-video ──────────────────────────────────────
        if not ret:
            if is_video_file and args.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            elif is_video_file:
                print("[Main] End of video.")
                break
            else:
                print("[Main] Camera read failed.")
                break

        frame = cv2.resize(frame, (640, 480))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ---------------------------------------------------
        # DEPTH ESTIMATION
        # ---------------------------------------------------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_batch = transform(rgb).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        depth_normalized = cv2.normalize(
            depth_map,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        depth_colored = cv2.applyColorMap(
            depth_normalized,
            cv2.COLORMAP_MAGMA
        )

        with state.lock:
            state.frame = frame.copy()
            snap_tracked  = {k: dict(v) for k, v in state.tracked.items()}
            snap_dets     = list(state.detections)
            snap_decision = state.decision
            snap_state    = state.engine_state
            snap_mode     = state.use_model

        # --- ORB Motion ---
        kp, des = orb.detectAndCompute(gray, None)
        motion_text = "No movement"
        arrow = None
        if prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) > 10:
                pts = [(kp[m.trainIdx].pt[0] - prev_kp[m.queryIdx].pt[0],
                        kp[m.trainIdx].pt[1] - prev_kp[m.queryIdx].pt[1])
                       for m in matches[:20]]
                dx = sum(p[0] for p in pts) / len(pts)
                dy = sum(p[1] for p in pts) / len(pts)
                if abs(dx) > abs(dy):
                    if dx >  2: motion_text, arrow = "Moving Right", ( 80,  0)
                    if dx < -2: motion_text, arrow = "Moving Left",  (-80,  0)
                else:
                    if dy >  2: motion_text, arrow = "Moving Down",  (  0, 80)
                    if dy < -2: motion_text, arrow = "Moving Up",    (  0,-80)
        prev_kp, prev_des = kp, des
        with state.lock:
            state.motion_text = motion_text

        # --- FPS ---
        fps_count += 1
        elapsed = time.time() - fps_t0
        if elapsed >= 1.0:
            state.fps = int(fps_count / elapsed)
            fps_count = 0
            fps_t0 = time.time()

        # ================================================================
        # RENDERING
        # ================================================================
        out = frame.copy()
        occupancy[:] = occupancy * 0.95
        
        for obj_id, obj in snap_tracked.items():
            cx, cy = obj["center"]

            gx = int(cx / 640 * grid_size)
            gy = int(cy / 480 * grid_size)

            gx = np.clip(gx, 0, grid_size - 1)
            gy = np.clip(gy, 0, grid_size - 1)

            occupancy[gy, gx] = 255
            
            draw_bbox(out, obj, obj_id)

        if arrow:
            draw_motion_arrow(out, arrow)

        # Top HUD
        draw_hud_panel(out, 0, 0, 640, 38, alpha=0.60)
        state_col = STATE_COLOR.get(snap_state, C_WHITE)
        draw_text(out, f"  {snap_decision}", (4, 26), state_col, 0.65, 2)
        draw_text(out, f"{state.fps} FPS", (565, 26), C_GREEN, 0.60, 1)

        # Left panel
        draw_hud_panel(out, 0, 38, 175, 200, alpha=0.55)
        draw_text(out, "STATE",  (8, 58),  (130,130,130), 0.38)
        draw_text(out, snap_state, (8, 76), state_col, 0.52, 1)

        draw_text(out, "MOTION", (8, 100), (130,130,130), 0.38)
        draw_text(out, motion_text, (8, 118), C_WHITE, 0.46, 1)

        mode_lbl = "AI MODEL" if snap_mode else "RULE-BASED"
        mode_col = C_CYAN if snap_mode else C_ORANGE
        draw_text(out, "MODE",   (8, 142), (130,130,130), 0.38)
        draw_text(out, mode_lbl, (8, 160), mode_col, 0.50, 1)

        # Source indicator (shows VIDEO vs LIVE)
        draw_text(out, source_label, (8, 182), (100, 100, 100), 0.34, 1)

        # Confidence bars
        if snap_dets:
            draw_hud_panel(out, 0, 380, 175, 100, alpha=0.55)
            draw_text(out, "DETECTIONS", (8, 398), (130,130,130), 0.38)
            draw_confidence_bars(out, snap_dets, 0, 403)

        # Label feedback
        if label_msg and time.time() - label_timer < 2.0:
            draw_hud_panel(out, 0, 450, 640, 30, alpha=0.70)
            draw_text(out, label_msg, (10, 470), label_color, 0.55, 1)
        elif time.time() - label_timer >= 2.0:
            label_msg = ""

        # Key hints
        hints = "ESC:exit  M:toggle  A/C/T/E:label"
        draw_text(out, hints, (178, 472), (100, 100, 100), 0.38)

        cv2.imshow("AI Visual Navigation System", out)
        occ_vis = cv2.resize(
            occupancy,
            (300, 300),
            interpolation=cv2.INTER_NEAREST
        )

        cv2.imshow("Occupancy Map", occ_vis)
        cv2.imshow("Depth Map", depth_colored)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == 27:
            state.running = False
            break
        elif key == ord('m'):
            with state.lock:
                state.use_model = not state.use_model
            label_msg   = f"Switched to: {'AI MODEL' if state.use_model else 'RULE-BASED'}"
            label_color = C_CYAN
            label_timer = time.time()
        elif key in [ord('a'), ord('c'), ord('t'), ord('e')]:
            lmap = {ord('a'): "AVOID_PERSON", ord('c'): "MOVE_TO_CHAIR",
                    ord('t'): "CHECK_TABLE",  ord('e'): "EXPLORE"}
            action = lmap[key]
            logger = ReasoningEngine()
            ok = logger.log_example(snap_dets, (320, 240), 640*480, motion_text, action, session_csv)
            if ok:
                label_msg   = f"[OK]  Saved: {action}"
                label_color = C_GREEN
            else:
                label_msg   = "[X]  Rejected — low confidence"
                label_color = C_ORANGE
            label_timer = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()