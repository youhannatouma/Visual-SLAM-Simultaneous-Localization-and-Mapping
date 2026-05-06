import csv
import math
import os
import time
from collections import deque, Counter

import numpy as np
import torch
import torch.nn as nn

LABEL_CLASSES = ["person", "chair", "table", "sofa", "tv", "other"]
ACTION_CLASSES = [
    "MOVE_FORWARD",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "AVOID_PERSON",
    "MOVE_TO_CHAIR",
    "CHECK_TABLE",
    "EXPLORE"
]
MOTION_CLASSES = ["NONE", "LEFT", "RIGHT", "UP", "DOWN"]
NUM_OBJECT_SLOTS = 3


# =============================================================================
# IMPROVEMENT 1: DEEPER NETWORK WITH DROPOUT
# =============================================================================
# Why deepen the network?
#   A 2-layer network can only learn simple straight-line boundaries.
#   A 4-layer network can learn complex, curved decision boundaries.
#   Example: "person close AND chair far AND moving right" = very complex rule.
#
# Why Dropout(0.3)?
#   During training, we randomly DISABLE 30% of neurons each step.
#   This forces the network to NOT rely on any single neuron,
#   so it learns more robust, general patterns instead of memorizing data.
#
# Why BatchNorm1d?
#   Normalizes the values between layers. Prevents "exploding" or "vanishing"
#   gradients that slow down or break training.
# =============================================================================
class ReasoningGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(ACTION_CLASSES))
        )

    def forward(self, x):
        out, _ = self.gru(x)

        final = out[:, -1]

        return self.classifier(final)


class ReasoningEngine:
    def __init__(self, model_path="models/reasoning_model.pt"):
        self.goal = "find_chair"

        self.memory = []
        self.memory_size = 30
        self.memory_ttl = 2.0

        self.tracked_objects = {}
        self.next_id = 0

        self.trajectory_history = {}
        
        # BBox Smoothing
        self.decision_history = deque(maxlen=10)
        self.action_history = deque(maxlen=8)
        self.bbox_history = {}  # ID -> deque(maxlen=5)

        # =================================================================
        # IMPROVEMENT 3: TEMPORAL MEMORY
        # =================================================================
        # This stores which object LABELS were seen in the PREVIOUS frame.
        # Example: If a person was seen last frame but disappeared now,
        # the model still "knows" about it via this memory.
        # This makes the model time-aware, not just snapshot-aware.
        # =================================================================
        self.prev_labels_seen = set()

        self.state = "EXPLORE"
        self.last_decision = "Initializing..."

        self.priority = {
            "person": 6,
            "chair": 5,
            "table": 3,
            "sofa": 2,
            "tv": 1
        }

        self.model_path = model_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            # NOTE: input_size now includes +6 temporal features (one per LABEL_CLASS).
            # If an OLD model (without temporal features) is loaded, it will fail
            # silently and fall back to rule-based reasoning. This is intentional.
            input_size = NUM_OBJECT_SLOTS * (len(LABEL_CLASSES) + 4) + len(MOTION_CLASSES) + 2 + len(LABEL_CLASSES)
            try:
                self.model = ReasoningGRU(input_size).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.model.eval()
                print(f"[ReasoningEngine] Loaded model from '{self.model_path}'")
            except Exception as e:
                print(f"[ReasoningEngine] WARNING: Model incompatible ({e}). Using rule-based fallback.")
                self.model = None
        else:
            self.model = None

    @staticmethod
    def label_to_onehot(label):
        onehot = [0.0] * len(LABEL_CLASSES)
        if label in LABEL_CLASSES:
            onehot[LABEL_CLASSES.index(label)] = 1.0
        else:
            onehot[-1] = 1.0
        return onehot

    @staticmethod
    def action_to_index(action_label):
        return ACTION_CLASSES.index(action_label)

    @staticmethod
    def index_to_action(index):
        return ACTION_CLASSES[index]

    @staticmethod
    def motion_to_onehot(motion_text):
        motion = "NONE"
        if motion_text and motion_text != "No movement":
            text = motion_text.upper()
            if "LEFT" in text:
                motion = "LEFT"
            elif "RIGHT" in text:
                motion = "RIGHT"
            elif "UP" in text:
                motion = "UP"
            elif "DOWN" in text:
                motion = "DOWN"
        onehot = [0.0] * len(MOTION_CLASSES)
        onehot[MOTION_CLASSES.index(motion)] = 1.0
        return onehot

    def compute_velocity(points):
        if len(points) < 2:
            return 0, 0

        x1, y1 = points[-2]
        x2, y2 = points[-1]

        return x2 - x1, y2 - y1

    def track_objects(self, detections):
        updated = {}
        used_ids = set()

        for det in detections:
            cx, cy = det["center"]
            matched_id = None
            min_dist = 999999

            for obj_id, obj in self.tracked_objects.items():
                if obj_id in used_ids:
                    continue

                ox, oy = obj["center"]
                dist = math.hypot(cx - ox, cy - oy)

                if dist < 60 and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                obj_id = matched_id
            else:
                obj_id = self.next_id
                self.next_id += 1

            # BBox Smoothing: Average last 5 positions
            if obj_id not in self.bbox_history:
                self.bbox_history[obj_id] = deque(maxlen=5)

            self.bbox_history[obj_id].append(det["bbox"])

            history = self.bbox_history[obj_id]
            avg_x1 = sum(b[0] for b in history) // len(history)
            avg_y1 = sum(b[1] for b in history) // len(history)
            avg_x2 = sum(b[2] for b in history) // len(history)
            avg_y2 = sum(b[3] for b in history) // len(history)

            det["bbox"] = (avg_x1, avg_y1, avg_x2, avg_y2)
            det["center"] = ((avg_x1 + avg_x2) // 2, (avg_y1 + avg_y2) // 2)
            det["area"] = (avg_x2 - avg_x1) * (avg_y2 - avg_y1)

            if obj_id not in self.trajectory_history:
                self.trajectory_history[obj_id] = deque(maxlen=15)

            self.trajectory_history[obj_id].append(det["center"])
            
            updated[obj_id] = det
            used_ids.add(obj_id)

        # Cleanup old bbox histories for objects no longer seen
        current_ids = set(updated.keys())
        self.bbox_history = {k: v for k, v in self.bbox_history.items() if k in current_ids}

        self.tracked_objects = updated
        return updated

    def update_memory(self, tracked):
        now = time.time()
        for obj in tracked.values():
            self.memory.append({
                "label": obj["label"],
                "confidence": obj["confidence"],
                "center": obj["center"],
                "bbox": obj["bbox"],
                "area": obj["area"],
                "timestamp": now
            })

        self.memory = [obj for obj in self.memory if now - obj["timestamp"] < self.memory_ttl]
        self.memory = self.memory[-self.memory_size:]

    def frame_direction(self, obj_center, frame_center):
        dx = obj_center[0] - frame_center[0]
        dy = obj_center[1] - frame_center[1]
        if abs(dx) > abs(dy):
            return "LEFT" if dx < 0 else "RIGHT"
        return "UP" if dy < 0 else "DOWN"

    def compute_score(self, obj, frame_center, frame_area):
        label = obj["label"]
        conf = obj["confidence"]
        area_ratio = obj["area"] / frame_area

        score = conf * 2.5
        score += self.priority.get(label, 0) * 3
        score += min(area_ratio, 0.4) * 12

        if label == "chair":
            dx = abs(obj["center"][0] - frame_center[0])
            score += max(0, (frame_center[0] * 0.6 - dx) / frame_center[0]) * 4

        if label == "person":
            if area_ratio > 0.05:
                score += 6
            if abs(obj["center"][0] - frame_center[0]) < frame_center[0] * 0.3:
                score += 3

        return score

    def choose_target(self, tracked, frame_center, frame_area):
        person_threat = 0
        nearest_person = None
        best_target = None
        best_score = -1

        for obj in tracked.values():
            label = obj["label"]
            score = self.compute_score(obj, frame_center, frame_area)

            if label == "person" and score > person_threat:
                person_threat = score
                nearest_person = obj

            if label != "person" and score > best_score:
                best_score = score
                best_target = obj

        return nearest_person, person_threat, best_target

    def object_score(self, det, frame_center, frame_area):
        label = det["label"]
        score = det["confidence"] + self.priority.get(label, 0) * 0.15 + (det["area"] / frame_area)
        return score

    def extract_features(self, detections, frame_center, frame_area, motion_text):
        sorted_detections = sorted(
            detections,
            key=lambda det: self.object_score(det, frame_center, frame_area),
            reverse=True
        )
        slot_features = []

        for det in sorted_detections[:NUM_OBJECT_SLOTS]:
            label_onehot = self.label_to_onehot(det["label"])
            conf = det["confidence"]
            area = det["area"] / frame_area
            cx, cy = frame_center
            dx = 0.0 if cx == 0 else (det["center"][0] - cx) / cx
            dy = 0.0 if cy == 0 else (det["center"][1] - cy) / cy
            
            slot_features.extend(label_onehot + [conf, area, dx, dy])

        while len(slot_features) < NUM_OBJECT_SLOTS * (len(LABEL_CLASSES) + 4):
            slot_features.extend([0.0] * (len(LABEL_CLASSES) + 4))

        motion_onehot = self.motion_to_onehot(motion_text)
        motion_flag = 0.0 if motion_text == "No movement" or motion_text is None else 1.0
        object_count = min(len(detections), NUM_OBJECT_SLOTS) / NUM_OBJECT_SLOTS

        # =====================================================================
        # IMPROVEMENT 3: TEMPORAL FEATURES
        # =====================================================================
        # For each label class, we add 1.0 if that label was seen LAST frame.
        # This gives the model "short-term memory":
        #   - "person was there last frame but disappeared" → might be hiding
        #   - "chair appeared this frame for the first time" → new target
        # The model can now make decisions based on CHANGE, not just the current snapshot.
        # =====================================================================
        temporal_features = [
            1.0 if label in self.prev_labels_seen else 0.0
            for label in LABEL_CLASSES
        ]

        features = np.array(slot_features + motion_onehot + [motion_flag, object_count] + temporal_features, dtype=np.float32)

        # HARD SAFETY CLEANUP (critical)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features.tolist()


    def predict_action(self, features):
        if self.model is None:
            return None
        # eval() mode disables Dropout so predictions are deterministic
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(tensor)
            action_index = int(logits.argmax(dim=-1).item())
            return self.index_to_action(action_index)

    def format_action(self, action, tracked, frame_center, frame_area):
        person = next((obj for obj in tracked.values() if obj["label"] == "person"), None)
        chair = next((obj for obj in tracked.values() if obj["label"] == "chair"), None)
        table = next((obj for obj in tracked.values() if obj["label"] == "table"), None)
        if action == "MOVE_FORWARD":
            self.state = "NAVIGATE"
            return "MOVE FORWARD"

        if action == "TURN_LEFT":
            self.state = "NAVIGATE"
            return "TURN LEFT"

        if action == "TURN_RIGHT":
            self.state = "NAVIGATE"
            return "TURN RIGHT"

        if action == "STOP":
            self.state = "STOP"
            return "STOP"
        
        if action == "AVOID_PERSON":
            self.state = "AVOID"
            if person:
                direction = self.frame_direction(person["center"], frame_center)
                return f"AVOID PERSON ({direction})"
            return "AVOID PERSON"

        if action == "MOVE_TO_CHAIR":
            self.state = "TARGET"
            if chair:
                direction = self.frame_direction(chair["center"], frame_center)
                if chair["area"] / frame_area > 0.08:
                    return f"MOVE TO CHAIR ({direction})"
                return f"TURN TOWARD CHAIR ({direction})"
            return "MOVE TO CHAIR"

        if action == "CHECK_TABLE":
            self.state = "INVESTIGATE"
            if table:
                direction = self.frame_direction(table["center"], frame_center)
                return f"CHECK TABLE ({direction})"
            return "CHECK TABLE"

        self.state = "EXPLORE"
        return "EXPLORE AND SEARCH"

    def log_example(
        self,
        detections,
        frame_center,
        frame_area,
        motion_text,
        action_label,
        output_path="data/raw/reasoning_data.csv",
        source_type="manual_live",
        min_confidence=0.65,
    ):
        # =====================================================================
        # IMPROVEMENT 2: QUALITY GATE
        # =====================================================================
        # Before saving training data, we check the average detection confidence.
        # If it's below min_confidence (default 60%), we REJECT the sample.
        #
        # Why? Blurry or partially visible objects get low confidence scores.
        # Training on low-confidence data is like teaching with blurry textbooks -
        # the model learns wrong patterns and becomes less accurate.
        #
        # This ensures ONLY high-quality, clear observations are used for training.
        # =====================================================================
        if len(detections) > 8:
            print("[QualityGate] REJECTED: too many objects")
            return False
        if detections:
            large_objects = [
                d for d in detections
                if d["area"] > frame_area * 0.01
            ]

            if not large_objects:
                print("[QualityGate] REJECTED: objects too small")
                return False
            avg_conf = sum(d["confidence"] for d in detections) / len(detections)
            if avg_conf < min_confidence:
                print(f"[QualityGate] REJECTED: avg_conf={avg_conf:.2f} < threshold={min_confidence:.2f}")
                return False  # Return False to signal the caller the sample was rejected

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features = self.extract_features(detections, frame_center, frame_area, motion_text)
        row = features + [action_label, source_type]
        header = [f"f{i}" for i in range(len(features))] + ["label", "source_type"]

        write_header = not os.path.exists(output_path)
        with open(output_path, mode="a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            if write_header:
                writer.writerow(header)
            writer.writerow(row)
        return True  # Return True to signal success

    def decide(self, detections, frame_center=(320, 240), frame_area=640 * 480, motion_text=None, use_model=True):
        tracked = self.track_objects(detections)
        self.update_memory(tracked)

        raw_decision = ""
        
        if use_model and self.model is not None:
            features = self.extract_features(
                detections,
                frame_center,
                frame_area,
                motion_text
            )

            action = self.predict_action(features)

            if action is not None:

                # ---------------------------------------------------
                # TEMPORAL ACTION SMOOTHING
                # ---------------------------------------------------
                self.action_history.append(action)

                final_action = Counter(
                    self.action_history
                ).most_common(1)[0][0]

                raw_decision = self.format_action(
                    final_action,
                    tracked,
                    frame_center,
                    frame_area
                )

        if not raw_decision:
            nearest_person, person_threat, best_target = self.choose_target(tracked, frame_center, frame_area)

            if nearest_person and person_threat > 8:
                self.state = "AVOID"
                direction = self.frame_direction(nearest_person["center"], frame_center)
                raw_decision = f"AVOID PERSON ({direction})"
            elif best_target and best_target["label"] == "chair":
                self.state = "TARGET"
                direction = self.frame_direction(best_target["center"], frame_center)
                if best_target["area"] / frame_area > 0.08:
                    raw_decision = f"MOVE TO CHAIR ({direction})"
                else:
                    raw_decision = f"TURN TOWARD CHAIR ({direction})"
            elif best_target and best_target["label"] == "table":
                self.state = "INVESTIGATE"
                direction = self.frame_direction(best_target["center"], frame_center)
                raw_decision = f"CHECK TABLE ({direction})"
            else:

                if motion_text == "No movement":
                    self.state = "NAVIGATE"
                    raw_decision = "MOVE FORWARD"

                else:
                    self.state = "EXPLORE"
                    raw_decision = "EXPLORE AND SEARCH"

        # Hysteresis: Return the most common decision in recent history
        self.decision_history.append(raw_decision)
        counts = Counter(self.decision_history)
        self.last_decision = counts.most_common(1)[0][0]

        # Update temporal memory for the NEXT frame's feature extraction
        self.prev_labels_seen = {obj["label"] for obj in tracked.values()}

        return self.last_decision, tracked
