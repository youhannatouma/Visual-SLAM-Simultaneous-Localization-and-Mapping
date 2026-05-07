import unittest

from mapping_runtime import LiveMapper, PoseSample, compute_loop_closure_drift


class MappingRuntimeTests(unittest.TestCase):
    def test_pose_timestamp_monotonic(self):
        mapper = LiveMapper(grid_size=40, meters_per_cell=0.2)
        p1 = mapper.update_pose_from_orb(dx_px=5, dy_px=0, timestamp=10.0, motion_to_meter_scale=0.01)
        p2 = mapper.update_pose_from_orb(dx_px=3, dy_px=1, timestamp=11.0, motion_to_meter_scale=0.01)
        self.assertGreaterEqual(p2.timestamp, p1.timestamp)

    def test_projection_bounds(self):
        mapper = LiveMapper(grid_size=30, meters_per_cell=0.2)
        det = {"center": (639, 479), "area": 8000, "confidence": 0.9}
        _, (gx, gy) = mapper.project_detection_to_world(det, (480, 640, 3))
        self.assertGreaterEqual(gx, 0)
        self.assertGreaterEqual(gy, 0)
        self.assertLess(gx, 30)
        self.assertLess(gy, 30)

    def test_ray_marks_free_and_obstacle(self):
        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, free_decrement=0.2, obstacle_increment=0.4)
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (500, 200), "area": 5000}}
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        self.assertTrue(len(mapper.frame_obstacles.get(1, set())) >= 1)
        self.assertTrue(any(v.get("free", 0) > 0 for v in mapper.cell_event_counts.values()))

    def test_loop_closure_metric_detects_pairs(self):
        poses = [
            PoseSample(0.0, 0.0, 0.0, 0.0),
            PoseSample(1.0, 0.0, 0.0, 1.0),
            PoseSample(0.05, 0.02, 0.05, 2.0),
        ]
        stats = compute_loop_closure_drift(poses, closure_radius_m=0.2, min_frame_gap=2)
        self.assertTrue(stats["available"])
        self.assertGreaterEqual(stats["closure_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
