import math

class ReasoningEngine:
    def __init__(self):
        self.goal = "find_chair"

        # memory + tracking
        self.memory = []
        self.memory_size = 15

        self.tracked_objects = {}
        self.next_id = 0

        self.state = "EXPLORE"

        self.priority = {
            "person": 3,
            "chair": 2,
            "table": 1
        }

    # -------------------------------
    # TRACKING (simple centroid matching)
    # -------------------------------
    def track_objects(self, detections):
        updated = {}

        for det in detections:
            cx, cy = det["center"]

            matched_id = None
            min_dist = 999999

            for obj_id, obj in self.tracked_objects.items():
                ox, oy = obj["center"]
                dist = math.hypot(cx - ox, cy - oy)

                if dist < 50 and dist < min_dist:
                    min_dist = dist
                    matched_id = obj_id

            if matched_id is not None:
                updated[matched_id] = det
            else:
                updated[self.next_id] = det
                self.next_id += 1

        self.tracked_objects = updated
        return updated

    # -------------------------------
    # MEMORY
    # -------------------------------
    def update_memory(self, detections):
        self.memory.extend(detections)

        if len(self.memory) > self.memory_size:
            self.memory = self.memory[-self.memory_size:]

    # -------------------------------
    # MAIN AI FUNCTION
    # -------------------------------
    def decide(self, detections):
        tracked = self.track_objects(detections)
        self.update_memory(list(tracked.values()))

        best_label = None
        best_score = -1

        for obj in self.memory:
            label = obj["label"]
            conf = obj["confidence"]
            area_score = conf * 2

            priority = self.priority.get(label, 0)

            score = area_score + priority

            if self.goal == "find_chair" and label == "chair":
                score += 2

            if score > best_score:
                best_score = score
                best_label = label

        # -------------------------------
        # STATE MACHINE
        # -------------------------------
        if best_label == "person":
            self.state = "AVOID"
            decision = "AVOID PERSON"

        elif best_label == "chair":
            self.state = "TARGET"
            decision = "MOVE TO CHAIR"

        else:
            self.state = "EXPLORE"
            decision = "EXPLORE"


        return decision, self.tracked_objects