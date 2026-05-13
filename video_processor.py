"""
video_processor.py  –  Offline video analysis and labeling for the AI Visual Navigation System.

Usage examples
--------------
# Interactively label a video (keyboard: A/C/T/E to label, SPACE to pause, ESC to quit)
python video_processor.py label --video path/to/video.mp4 --session my_session

# Auto-label an entire video without keyboard interaction (rule-based heuristics)
python video_processor.py autolabel --video path/to/video.mp4

# Just run inference on a video and save an annotated output video
python video_processor.py infer --video path/to/video.mp4 --output out/annotated.mp4
"""

import argparse
import os
import time
import sys

import cv2
import torch
from ultralytics import YOLO

from mapping_runtime import LiveMapper, load_camera_calibration
from reasoning import ReasoningEngine


# ──────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {
    ord('a'): "AVOID_PERSON",
    ord('c'): "MOVE_TO_CHAIR",
    ord('t'): "CHECK_TABLE",
    ord('e'): "EXPLORE",
}

C_WHITE  = (255, 255, 255)
C_GREEN  = (60,  220,  60)
C_ORANGE = (40,  140, 255)
C_CYAN   = (255, 220,  40)
C_RED    = (60,   60, 220)

STATE_COLOR = {
    "AVOID":       C_RED,
    "TARGET":      C_ORANGE,
    "INVESTIGATE": C_GREEN,
    "EXPLORE":     (200, 200, 200),
}


def load_yolo(model_path="yolov8n.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[YOLO] Loading model on {device.upper()} …")
    model = YOLO(model_path)
    return model, device


def detect(model, frame, device):
    results = model(frame, imgsz=320, verbose=False, device=device)
    dets = []
    for r in results:
        for box in r.boxes:
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            lbl   = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            dets.append({
                "label": lbl, "confidence": conf,
                "center": ((x1+x2)//2, (y1+y2)//2),
                "bbox": (x1, y1, x2, y2),
                "area": (x2-x1)*(y2-y1),
            })
    return dets


def draw_overlay(frame, detections, decision, engine_state, motion_text, fps=0):
    """Minimal HUD drawn directly on a frame (no threading needed for offline use)."""
    out = frame.copy()
    h, w = out.shape[:2]

    # ── bounding boxes ──────────────────────────────────────────────────
    COLORS = {"person": C_RED, "chair": C_ORANGE, "table": C_GREEN}
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        color = COLORS.get(det["label"], (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_txt = f"#{i} {det['label'].upper()} {det['confidence']:.0%}"
        cv2.putText(out, label_txt, (x1 + 4, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    # ── top bar ─────────────────────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 36), (15, 15, 15), -1)
    state_col = STATE_COLOR.get(engine_state, C_WHITE)
    cv2.putText(out, f"  {decision}", (4, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_col, 2, cv2.LINE_AA)
    cv2.putText(out, f"{fps} FPS", (w - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_GREEN, 1, cv2.LINE_AA)

    # ── bottom bar ──────────────────────────────────────────────────────
    cv2.rectangle(out, (0, h - 28), (w, h), (15, 15, 15), -1)
    cv2.putText(out, f"Motion: {motion_text}  |  State: {engine_state}", (6, h - 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1: INTERACTIVE LABELING
# ──────────────────────────────────────────────────────────────────────────────

def mode_label(video_path: str, session_name: str, model_path: str, output_csv: str):
    """
    Play a video frame-by-frame and let you assign action labels with keys.

    Keys
    ----
    A  → AVOID_PERSON
    C  → MOVE_TO_CHAIR
    T  → CHECK_TABLE
    E  → EXPLORE
    SPACE → pause / resume
    ESC   → quit
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[Label] Video: {video_path}  frames={total_frames}  fps={fps_src:.1f}")

    yolo, device = load_yolo(model_path)
    engine = ReasoningEngine()

    if not output_csv:
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_csv = f"data/raw/session_{session_name or ts}.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    saved      = 0
    rejected   = 0
    paused     = False
    frame_no   = 0
    motion_text = "No movement"
    label_msg  = ""
    label_color = C_GREEN
    label_t    = 0.0

    print(f"\n[Label] Saving to: {output_csv}")
    print("[Label] Keys: A=AVOID_PERSON  C=MOVE_TO_CHAIR  T=CHECK_TABLE  E=EXPLORE  SPACE=pause  ESC=quit\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Label] End of video.")
                break

            frame     = cv2.resize(frame, (640, 480))
            frame_no += 1
            dets      = detect(yolo, frame, device)

            decision, _ = engine.decide(dets, (320, 240), 640 * 480, motion_text, use_model=False)
            engine_state = engine.state

            progress = f"{frame_no}/{total_frames}"
            frame_display = draw_overlay(frame, dets, decision, engine_state, motion_text)

            # Progress bar
            bar_w = int((frame_no / max(total_frames, 1)) * 640)
            cv2.rectangle(frame_display, (0, 476), (bar_w, 480), (60, 180, 60), -1)

            # Feedback message
            if label_msg and time.time() - label_t < 2.0:
                cv2.rectangle(frame_display, (0, 44), (640, 72), (15, 15, 15), -1)
                cv2.putText(frame_display, label_msg, (8, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, label_color, 1, cv2.LINE_AA)

            # Stats
            cv2.putText(frame_display, f"Frame {progress}  saved={saved}  rejected={rejected}",
                        (8, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

            cv2.imshow("Label Video – AI Nav System", frame_display)

        key = cv2.waitKey(1 if not paused else 50) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):
            paused = not paused
            print("[Label] " + ("Paused" if paused else "Resumed"))
        elif key in LABEL_MAP:
            action = LABEL_MAP[key]
            ok = engine.log_example(dets, (320, 240), 640 * 480, motion_text, action, output_csv)
            if ok:
                saved += 1
                label_msg   = f"[OK]  Saved: {action}  (total={saved})"
                label_color = C_GREEN
                print(f"[Label] Frame {frame_no} → {action}")
            else:
                rejected += 1
                label_msg   = f"[X]  Rejected — low confidence  (total_rejected={rejected})"
                label_color = C_ORANGE
                print(f"[Label] Frame {frame_no} → REJECTED (low confidence)")
            label_t = time.time()

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[Label] Done. saved={saved}  rejected={rejected}  csv={output_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2: AUTOMATIC LABELING (rule-based heuristics, no keyboard needed)
# ──────────────────────────────────────────────────────────────────────────────

def mode_autolabel(video_path: str, model_path: str, output_csv: str,
                   every_n: int = 5, min_conf: float = 0.55):
    """
    Automatically label every N-th frame using the rule-based engine.
    Great for building a large initial dataset quickly.

    Parameters
    ----------
    every_n   : sample every N frames (default 5) to avoid near-duplicate rows
    min_conf  : minimum average detection confidence to accept a sample
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_src      = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"[AutoLabel] Video: {video_path}  frames={total_frames}  fps={fps_src:.1f}")
    print(f"[AutoLabel] Sampling every {every_n} frames  min_conf={min_conf}")

    if not output_csv:
        ts = time.strftime('%Y%m%d_%H%M%S')
        output_csv = f"data/raw/autolabel_{ts}.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    yolo, device = load_yolo(model_path)
    engine  = ReasoningEngine()
    saved   = 0
    skipped = 0

    for frame_no in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % every_n != 0:
            continue

        frame = cv2.resize(frame, (640, 480))
        dets  = detect(yolo, frame, device)

        if not dets:
            skipped += 1
            continue

        # Use rule-based decide to derive the label automatically
        decision, _ = engine.decide(dets, (320, 240), 640 * 480, "No movement", use_model=False)

        # Map decision string → ACTION_CLASS
        if "AVOID" in decision:
            action = "AVOID_PERSON"
        elif "CHAIR" in decision:
            action = "MOVE_TO_CHAIR"
        elif "TABLE" in decision:
            action = "CHECK_TABLE"
        else:
            action = "EXPLORE"

        ok = engine.log_example(
            dets, (320, 240), 640 * 480, "No movement",
            action, output_csv,
            source_type="autolabel_video",
            min_confidence=min_conf,
        )
        if ok:
            saved += 1
        else:
            skipped += 1

        if frame_no % 100 == 0:
            pct = frame_no / max(total_frames, 1) * 100
            print(f"  {pct:5.1f}%  frame={frame_no}  saved={saved}  skipped={skipped}")

    cap.release()
    print(f"\n[AutoLabel] Done. saved={saved}  skipped={skipped}  csv={output_csv}")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 3: INFERENCE / ANNOTATED VIDEO OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def mode_infer(
    video_path: str,
    output_path: str,
    model_path: str,
    use_model: bool,
    use_mapping: bool = False,
    camera_calibration_path: str = "",
    mapping_backend: str = "heuristic",
):
    """
    Run the full pipeline on a video and save an annotated MP4.
    No CSV output – pure visualization.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not output_path:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"out/{base}_annotated.mp4"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))

    print(f"[Infer] {video_path}  →  {output_path}  (use_model={use_model})")

    yolo, device = load_yolo(model_path)
    engine       = ReasoningEngine()
    motion_text  = "No movement"
    calibration = load_camera_calibration(camera_calibration_path) if camera_calibration_path else None
    mapper = LiveMapper(camera_calibration=calibration, mapping_backend=mapping_backend) if use_mapping else None

    t0 = time.time()
    for frame_no in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        dets  = detect(yolo, frame, device)

        decision, tracked = engine.decide(dets, (320, 240), 640 * 480, motion_text, use_model)
        fps_current  = (frame_no + 1) / max(time.time() - t0, 1e-6)

        annotated = draw_overlay(frame, dets, decision, engine.state, motion_text, int(fps_current))
        if mapper is not None:
            mapper.update_pose_from_orb(0.0, 0.0, time.time(), 0.0025)
            mapper.update_from_tracked(tracked, frame.shape, frame_no + 1, time.time())
            map_img = mapper.render_map(out_size=130)
            annotated[40:170, 500:630] = map_img
            cv2.rectangle(annotated, (500, 40), (630, 170), (255, 220, 40), 1)
        writer.write(annotated)

        if frame_no % 30 == 0:
            pct = frame_no / max(total_frames, 1) * 100
            print(f"  {pct:5.1f}%  frame={frame_no}/{total_frames}  fps={fps_current:.1f}")

    cap.release()
    writer.release()
    print(f"[Infer] Done. Output saved to: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline video tools for AI Visual Navigation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # ── label ──────────────────────────────────────────────────────────
    sl = sub.add_parser("label", help="Interactively label a video with keyboard shortcuts")
    sl.add_argument("--video",   required=True, help="Path to input video file")
    sl.add_argument("--session", default="",    help="Session name (used in CSV filename)")
    sl.add_argument("--output",  default="",    help="Override output CSV path")
    sl.add_argument("--model",   default="yolov8n.pt", help="YOLO model path")

    # ── autolabel ──────────────────────────────────────────────────────
    sa = sub.add_parser("autolabel", help="Auto-label a video using rule-based heuristics")
    sa.add_argument("--video",    required=True, help="Path to input video file")
    sa.add_argument("--output",   default="",    help="Override output CSV path")
    sa.add_argument("--model",    default="yolov8n.pt", help="YOLO model path")
    sa.add_argument("--every-n",  type=int, default=5,  help="Sample every N frames (default: 5)")
    sa.add_argument("--min-conf", type=float, default=0.55, help="Min avg confidence to accept (default: 0.55)")

    # ── infer ──────────────────────────────────────────────────────────
    si = sub.add_parser("infer", help="Run inference on a video and save annotated output")
    si.add_argument("--video",      required=True,  help="Path to input video file")
    si.add_argument("--output",     default="",     help="Override output MP4 path")
    si.add_argument("--model",      default="yolov8n.pt", help="YOLO model path")
    si.add_argument("--no-ai",      action="store_true",  help="Use rule-based mode instead of AI model")
    si.add_argument("--mapping",    action="store_true",  help="Render offline heuristic/depth map inset during inference")
    si.add_argument("--camera-calibration", default="", help="Optional JSON camera calibration for mapping")
    si.add_argument("--mapping-backend", choices=["heuristic", "depth", "orb_slam_like"], default="heuristic")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "label":
        mode_label(args.video, args.session, args.model, args.output)

    elif args.mode == "autolabel":
        mode_autolabel(args.video, args.model, args.output, args.every_n, args.min_conf)

    elif args.mode == "infer":
        mode_infer(
            args.video,
            args.output,
            args.model,
            use_model=not args.no_ai,
            use_mapping=args.mapping,
            camera_calibration_path=args.camera_calibration,
            mapping_backend=args.mapping_backend,
        )


if __name__ == "__main__":
    main()
