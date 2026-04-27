import cv2
from ultralytics import YOLO
from reasoning import ReasoningEngine

# -------------------------------
# INIT
# -------------------------------
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

orb = cv2.ORB_create()
engine = ReasoningEngine()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_kp = None
prev_des = None

decision = "Initializing..."

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # ORB FEATURE DETECTION
    # -------------------------------
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    annotated_frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # -------------------------------
    # MOTION ESTIMATION (BASIC)
    # -------------------------------
    motion_text = "No movement"
    arrow = None

    if prev_des is not None and descriptors is not None:
        matches = bf.match(prev_des, descriptors)

        if len(matches) > 10:
            dx, dy = 0, 0

            for m in matches[:20]:
                x1, y1 = prev_kp[m.queryIdx].pt
                x2, y2 = keypoints[m.trainIdx].pt

                dx += (x2 - x1)
                dy += (y2 - y1)

            dx /= 20
            dy /= 20

            if abs(dx) > abs(dy):
                if dx > 2:
                    motion_text = "Moving Right"
                    arrow = (100, 0)
                elif dx < -2:
                    motion_text = "Moving Left"
                    arrow = (-100, 0)
            else:
                if dy > 2:
                    motion_text = "Moving Down"
                    arrow = (0, 100)
                elif dy < -2:
                    motion_text = "Moving Up"
                    arrow = (0, -100)

    prev_kp = keypoints
    prev_des = descriptors

    # -------------------------------
    # YOLO DETECTION
    # -------------------------------
    results = model(frame)

    detections_list = []

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections_list.append({
                "label": label,
                "confidence": conf,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2)
            })

    # -------------------------------
    # AI DECISION + TRACKING
    # -------------------------------
    decision, tracked_objects = engine.decide(detections_list)

    # -------------------------------
    # DRAW YOLO BOXES
    # -------------------------------
    annotated_frame = results[0].plot()

    # -------------------------------
    # DRAW TRACKING IDS
    # -------------------------------
    for obj_id, obj in tracked_objects.items():
        cx, cy = obj["center"]

        cv2.putText(annotated_frame, f"ID {obj_id}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 2)

    # -------------------------------
    # CONFIDENCE BARS
    # -------------------------------
    y_offset = 120

    for obj in detections_list[:5]:
        label = obj["label"]
        conf = obj["confidence"]

        bar_len = int(conf * 150)

        cv2.putText(annotated_frame, f"{label} ({conf:.2f})",
                    (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)

        cv2.rectangle(annotated_frame,
                      (160, y_offset - 10),
                      (160 + bar_len, y_offset),
                      (0, 255, 0), -1)

        y_offset += 25

    # -------------------------------
    # MOTION ARROW
    # -------------------------------
    if arrow is not None:
        start = (320, 240)
        end = (320 + arrow[0], 240 + arrow[1])

        cv2.arrowedLine(annotated_frame, start, end,
                        (0, 255, 0), 3)

    # -------------------------------
    # UI TEXT
    # -------------------------------
    cv2.putText(annotated_frame, decision, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.putText(annotated_frame, motion_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)

    # -------------------------------
    # DISPLAY
    # -------------------------------
    cv2.imshow("AI Visual System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()