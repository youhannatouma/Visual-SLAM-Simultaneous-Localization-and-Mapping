import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open camera
cap = cv2.VideoCapture(0)

# ORB feature detector
orb = cv2.ORB_create()

# Previous frame (for motion tracking)
prev_gray = None

# Default decision
decision = "Initializing..."

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize (optional for performance)
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale for ORB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------------
    # 🔍 ORB FEATURE DETECTION
    # -------------------------------
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0))

    # -------------------------------
    # 📐 SIMPLE MOTION ESTIMATION
    # -------------------------------
    if prev_gray is not None:
        # (Basic placeholder for now)
        motion_text = "Tracking movement..."
    else:
        motion_text = "Initializing motion..."

    prev_gray = gray

    # -------------------------------
    # 🧠 OBJECT DETECTION (YOLO)
    # -------------------------------
    results = model(frame)

    # Default decision each frame
    decision = "EXPLORE"

    for r in results:
        boxes = r.boxes

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # -------------------------------
            # 🧠 AI REASONING RULES
            # -------------------------------
            if label == "chair" and conf > 0.6:
                decision = "TARGET CHAIR"
            elif label == "person":
                decision = "AVOID PERSON"

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # -------------------------------
    # 🎨 OVERLAY TEXT
    # -------------------------------
    cv2.putText(annotated_frame, decision, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    cv2.putText(annotated_frame, motion_text, (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 0, 0), 2)

    # -------------------------------
    # 🖥️ DISPLAY
    # -------------------------------
    cv2.imshow("AI Visual System", annotated_frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()