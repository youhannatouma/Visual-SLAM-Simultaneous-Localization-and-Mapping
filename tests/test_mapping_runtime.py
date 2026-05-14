import unittest
import json
import tempfile
from pathlib import Path

from mapping_runtime import (
    CameraCalibration,
    LiveMapper,
    PoseSample,
    compute_loop_closure_drift,
    compute_map_consistency_score,
    compute_mapping_quality_summary,
    compute_obstacle_object_precision_recall,
    compute_obstacle_object_precision_recall_from_components,
    compute_obstacle_precision_recall,
    compute_obstacle_persistence_stability,
    compute_occupancy_confidence_concentration,
    compute_pose_jitter_score,
    load_camera_calibration,
    load_run_annotations,
    select_benchmark_obstacle_metric,
)


class MappingRuntimeTests(unittest.TestCase):
    def test_pose_timestamp_monotonic(self):
        mapper = LiveMapper(grid_size=40, meters_per_cell=0.2)
        p1 = mapper.update_pose_from_orb(dx_px=5, dy_px=0, timestamp=10.0, motion_to_meter_scale=0.01)
        p2 = mapper.update_pose_from_orb(dx_px=3, dy_px=1, timestamp=11.0, motion_to_meter_scale=0.01)
        self.assertGreaterEqual(p2.timestamp, p1.timestamp)

    def test_projection_bounds(self):
        mapper = LiveMapper(grid_size=30, meters_per_cell=0.2)
        det = {"center": (639, 479), "area": 8000, "confidence": 0.9}
        _, (gx, gy), _ = mapper.project_detection_to_world(det, (480, 640, 3))
        self.assertGreaterEqual(gx, 0)
        self.assertGreaterEqual(gy, 0)
        self.assertLess(gx, 30)
        self.assertLess(gy, 30)

    def test_projection_uses_corrected_pose_when_loop_closure_enabled(self):
        mapper = LiveMapper(grid_size=80, meters_per_cell=0.1, loop_closure_enabled=True)
        mapper.pose = PoseSample(5.0, 5.0, 0.0, 1.0)
        mapper.corrected_pose = PoseSample(2.0, 3.0, 0.0, 1.0)
        det = {"center": (320, 240), "area": 12000, "confidence": 0.9, "label": "chair"}
        (wx, wy), _, _ = mapper.project_detection_to_world(det, (480, 640, 3))
        self.assertLess(abs(wx - mapper.corrected_pose.x), abs(wx - mapper.pose.x))
        self.assertLess(abs(wy - mapper.corrected_pose.y), abs(wy - mapper.pose.y))

    def test_projection_uses_bottom_center_anchor_for_tall_boxes(self):
        mapper = LiveMapper(grid_size=80, meters_per_cell=0.1)
        det = {"center": (320, 140), "bbox": (300, 40, 340, 300), "area": 10400, "confidence": 0.9, "label": "person", "track_id": 1}
        _, (gx, gy), (raw_wx, raw_wy) = mapper.project_detection_to_world(det, (480, 640, 3))
        self.assertEqual((gx, gy), mapper.world_to_grid(raw_wx, raw_wy))
        self.assertGreaterEqual(raw_wy, mapper.pose.y)

    def test_gitignore_has_no_conflict_markers(self):
        gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
        text = gitignore.read_text(encoding="utf-8")
        self.assertNotIn("<<<<<<<", text)
        self.assertNotIn("=======", text)
        self.assertNotIn(">>>>>>>", text)

    def test_camera_calibration_loads_matrix_and_fields(self):
        payload = {
            "camera_matrix": [[500.0, 0.0, 320.0], [0.0, 510.0, 240.0], [0.0, 0.0, 1.0]],
            "dist_coeffs": [0.1, -0.01, 0.0, 0.0],
            "width": 640,
            "height": 480,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        cal = load_camera_calibration(path)
        self.assertEqual(cal.fx, 500.0)
        self.assertEqual(cal.fy, 510.0)
        self.assertEqual(cal.cx, 320.0)
        self.assertEqual(cal.dist_coeffs[0], 0.1)

    def test_depth_projection_uses_depth_map_range(self):
        cal = CameraCalibration(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        mapper = LiveMapper(
            grid_size=80,
            meters_per_cell=0.1,
            camera_calibration=cal,
            mapping_backend="depth",
        )
        det = {"center": (320, 240), "bbox": (310, 230, 330, 250), "area": 400, "confidence": 0.9, "label": "chair"}
        depth_map = __import__("numpy").full((480, 640), 2.0, dtype="float32")
        (wx, wy), _, _ = mapper.project_detection_to_world(det, (480, 640, 3), depth_map=depth_map)
        self.assertAlmostEqual(wx - mapper.pose.x, 2.0, places=3)
        self.assertAlmostEqual(wy - mapper.pose.y, 0.0, places=3)

    def test_depth_projection_prefers_lower_bbox_slice_and_filters_invalids(self):
        cal = CameraCalibration(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        mapper = LiveMapper(grid_size=80, meters_per_cell=0.1, camera_calibration=cal, mapping_backend="depth")
        det = {"center": (320, 220), "bbox": (300, 200, 340, 320), "area": 4800, "confidence": 0.9, "label": "chair", "track_id": 1}
        depth_map = __import__("numpy").full((480, 640), 6.0, dtype="float32")
        depth_map[278:321, 300:341] = 1.5
        depth_map[290:295, 315:320] = 0.0
        (wx, _), _, _ = mapper.project_detection_to_world(det, (480, 640, 3), depth_map=depth_map)
        self.assertAlmostEqual(wx - mapper.pose.x, 1.5, places=2)

    def test_calibrated_pose_update_uses_intrinsics_and_depth(self):
        cal = CameraCalibration(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
        mapper = LiveMapper(
            camera_calibration=cal,
            pose_smoothing_window=1,
            max_translation_m_per_frame=10.0,
        )
        start = mapper.pose
        pose = mapper.update_pose_from_calibrated_flow(
            dx_px=50.0,
            dy_px=0.0,
            dtheta_rad=0.0,
            timestamp=1.0,
            nominal_depth_m=2.0,
            flow_quality=1.0,
        )
        self.assertAlmostEqual(pose.x - start.x, 0.2, places=4)
        self.assertAlmostEqual(pose.y - start.y, 0.0, places=4)

    def test_ray_marks_free_and_obstacle(self):
        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, free_decrement=0.2, obstacle_increment=0.4)
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (500, 200), "area": 5000}}
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        self.assertTrue(len(mapper.frame_obstacles.get(1, set())) >= 1)
        self.assertTrue(any(v.get("free", 0) > 0 for v in mapper.cell_event_counts.values()))

    def test_obstacle_footprint_marks_square_neighbor_cells(self):
        mapper = LiveMapper(
            grid_size=50,
            meters_per_cell=0.1,
            obstacle_footprint_radius_cells=1,
            obstacle_footprint_shape="square",
            confidence_weighting=False,
        )
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (320, 240), "area": 12000}}
        events = mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        obstacle_events = [e for e in events if e.event_type == "obstacle_mark"]

        self.assertEqual(len(mapper.frame_obstacles[1]), 9)
        self.assertEqual(len(obstacle_events), 9)
        self.assertTrue(all(0 <= gx < mapper.grid_size and 0 <= gy < mapper.grid_size for gx, gy in mapper.frame_obstacles[1]))

    def test_obstacle_footprint_shapes(self):
        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, obstacle_footprint_radius_cells=1, obstacle_footprint_shape="horizontal")
        self.assertEqual(len(mapper._obstacle_footprint_cells((25, 25), label="table")), 3)

        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, obstacle_footprint_radius_cells=1, obstacle_footprint_shape="cross")
        self.assertEqual(len(mapper._obstacle_footprint_cells((25, 25), label="chair")), 5)

        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, obstacle_footprint_radius_cells=1, obstacle_footprint_shape="class_aware")
        self.assertEqual(len(mapper._obstacle_footprint_cells((25, 25), label="chair")), 5)
        self.assertEqual(len(mapper._obstacle_footprint_cells((25, 25), label="table")), 3)
        self.assertEqual(len(mapper._obstacle_footprint_cells((25, 25), label="person")), 3)

    def test_temporal_obstacle_persistence_keeps_recent_cells_active(self):
        mapper = LiveMapper(
            grid_size=50,
            meters_per_cell=0.1,
            obstacle_temporal_persistence_frames=2,
            confidence_weighting=False,
        )
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (320, 240), "area": 12000}}
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        first_cells = set(mapper.frame_obstacles[1])
        self.assertGreater(len(first_cells), 0)

        mapper.update_from_tracked({}, frame_shape=(480, 640, 3), frame_index=2, timestamp=2.0)
        self.assertEqual(mapper.frame_obstacles[2], first_cells)

    def test_loop_closure_metric_detects_pairs(self):
        poses = [
            PoseSample(0.0, 0.0, 0.0, 0.0),
            PoseSample(1.0, 0.0, 0.0, 1.0),
            PoseSample(0.05, 0.02, 0.05, 2.0),
        ]
        stats = compute_loop_closure_drift(poses, closure_radius_m=0.2, min_frame_gap=2)
        self.assertTrue(stats["available"])
        self.assertGreaterEqual(stats["closure_pairs"], 1)

    def test_pose_clamp_and_smoothing_reduce_spike(self):
        mapper = LiveMapper(
            max_translation_m_per_frame=0.05,
            pose_smoothing_window=3,
            meters_per_cell=0.1,
        )
        start = mapper.pose
        mapper.update_pose_from_orb(dx_px=500.0, dy_px=0.0, timestamp=1.0, motion_to_meter_scale=0.01)
        p1 = mapper.pose
        moved = ((p1.x - start.x) ** 2 + (p1.y - start.y) ** 2) ** 0.5
        self.assertLessEqual(moved, 0.051)

        mapper.update_pose_from_orb(dx_px=0.0, dy_px=0.0, timestamp=2.0, motion_to_meter_scale=0.01)
        mapper.update_pose_from_orb(dx_px=0.0, dy_px=0.0, timestamp=3.0, motion_to_meter_scale=0.01)
        p_end = mapper.pose
        moved_after = ((p_end.x - p1.x) ** 2 + (p_end.y - p1.y) ** 2) ** 0.5
        self.assertGreater(moved_after, 0.0)

    def test_confidence_weighted_obstacle_update(self):
        mapper = LiveMapper(
            confidence_weighting=True,
            confidence_strength=1.0,
            obstacle_persistence_frames=1,
            obstacle_increment=0.2,
        )
        base = float(mapper.grid[10, 10])
        mapper.frame_obstacles = {}
        tracked = {1: {"label": "chair", "confidence": 1.0, "center": (320, 240), "area": 20000}}
        _, (gx, gy), _ = mapper.project_detection_to_world(tracked[1], (480, 640, 3))
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        high_conf = float(mapper.grid[gy, gx] - base)

        mapper2 = LiveMapper(
            confidence_weighting=True,
            confidence_strength=1.0,
            obstacle_persistence_frames=1,
            obstacle_increment=0.2,
        )
        base2 = float(mapper2.grid[10, 10])
        tracked2 = {1: {"label": "chair", "confidence": 0.1, "center": (320, 240), "area": 20000}}
        _, (gx2, gy2), _ = mapper2.project_detection_to_world(tracked2[1], (480, 640, 3))
        mapper2.update_from_tracked(tracked2, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        low_conf = float(mapper2.grid[gy2, gx2] - base2)
        self.assertGreater(high_conf, low_conf)

    def test_projection_smoothing_reuses_previous_cell_for_small_jitter(self):
        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1)
        det1 = {"label": "chair", "confidence": 0.9, "center": (320, 240), "bbox": (300, 200, 340, 260), "area": 2400, "track_id": 7}
        det2 = {"label": "chair", "confidence": 0.9, "center": (324, 242), "bbox": (304, 202, 344, 262), "area": 2400, "track_id": 7}
        _, cell1, _ = mapper.project_detection_to_world(det1, (480, 640, 3))
        _, cell2, _ = mapper.project_detection_to_world(det2, (480, 640, 3))
        self.assertEqual(cell1, cell2)

    def test_free_space_ray_stops_before_footprint(self):
        mapper = LiveMapper(
            grid_size=60,
            meters_per_cell=0.1,
            obstacle_footprint_radius_cells=1,
            obstacle_footprint_shape="square",
            confidence_weighting=False,
        )
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (320, 240), "bbox": (260, 180, 380, 340), "area": 19200}}
        events = mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        footprint = set(mapper.frame_obstacles[1])
        ray_cells = {e.grid_xy for e in events if e.event_type == "free_space_ray"}
        self.assertTrue(footprint)
        self.assertTrue(ray_cells)
        self.assertTrue(ray_cells.isdisjoint(footprint))

    def test_persistence_weak_then_strong(self):
        mapper = LiveMapper(obstacle_persistence_frames=2, obstacle_increment=0.2, confidence_weighting=False)
        tracked = {1: {"label": "chair", "confidence": 0.9, "center": (320, 240), "area": 20000}}
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        c1 = mapper.last_frame_obstacle_counts
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=2, timestamp=2.0)
        c2 = mapper.last_frame_obstacle_counts
        self.assertGreaterEqual(c1["weak"], 1)
        self.assertGreaterEqual(c2["strong"], 1)

    def test_mapping_quality_summary_deterministic(self):
        mapper = LiveMapper(obstacle_persistence_frames=1)
        for i in range(6):
            mapper.update_pose_from_orb(dx_px=3.0, dy_px=1.0, timestamp=float(i + 1), motion_to_meter_scale=0.001)
            tracked = {1: {"label": "chair", "confidence": 0.95, "center": (320, 240), "area": 18000}}
            mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=i + 1, timestamp=float(i + 1))

        loop = compute_loop_closure_drift(mapper.pose_history, closure_radius_m=5.0, min_frame_gap=1)
        consistency = compute_map_consistency_score(mapper.cell_event_counts)
        jitter = compute_pose_jitter_score(mapper.pose_history)
        persistence = compute_obstacle_persistence_stability(mapper.frame_obstacles)
        concentration = compute_occupancy_confidence_concentration(mapper.grid)
        obstacle_pr = {"available": True, "f1": 0.8}
        summary = compute_mapping_quality_summary(
            loop_closure_drift=loop,
            map_consistency_score=consistency,
            pose_jitter=jitter,
            obstacle_persistence=persistence,
            occupancy_concentration=concentration,
            obstacle_precision_recall=obstacle_pr,
            require_benchmark=True,
        )
        self.assertIn(summary["status"], ("promotable", "not_promotable", "insufficient_evidence"))
        self.assertTrue(isinstance(summary["promotable"], bool))
        self.assertIn("checks", summary)
        self.assertTrue(all(isinstance(v, bool) for v in summary["checks"].values()))
        self.assertIn("moving_samples", jitter)

    def test_backend_summary_records_mapping_contract(self):
        mapper = LiveMapper(mapping_backend="orb_slam_like")
        summary = mapper.backend_summary()
        self.assertEqual(summary["backend"], "orb_slam_like")
        self.assertEqual(summary["status"], "external_backend_required")

    def test_benchmark_fixture_loads(self):
        fixture = Path(__file__).parent / "fixtures" / "mapping_benchmark_sample.json"
        ann = load_run_annotations(str(fixture))
        self.assertTrue(ann["available"])
        self.assertEqual(len(ann["frame_labels"]), 2)
        self.assertGreaterEqual(len(ann["obstacles_by_frame"]), 2)
        self.assertGreaterEqual(len(ann["obstacle_components_by_frame"]), 2)

    def test_annotation_loader_supports_grouped_components(self):
        payload = {
            "frame_labels": [{"frame": 1, "label": "EXPLORE"}],
            "obstacles": [
                {"frame": 1, "component_cells": [[10, 10], [10, 11]]},
                {"frame": 1, "grid_cells": [[20, 20], [21, 20]]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(payload, f)
            path = f.name
        ann = load_run_annotations(path)
        self.assertTrue(ann["available"])
        self.assertEqual(len(ann["obstacle_components_by_frame"][1]), 2)
        self.assertIn((10, 10), ann["obstacles_by_frame"][1])
        self.assertIn((21, 20), ann["obstacles_by_frame"][1])

    def test_obstacle_precision_recall_exact_and_radius_matching(self):
        gt = {1: {(10, 10), (20, 20)}}
        pred = {1: {(11, 10), (30, 30)}}

        exact = compute_obstacle_precision_recall(gt, pred, match_radius_cells=0)
        self.assertTrue(exact["available"])
        self.assertEqual(exact["tp"], 0)
        self.assertEqual(exact["fp"], 2)
        self.assertEqual(exact["fn"], 2)
        self.assertEqual(exact["match_radius_cells"], 0)

        tolerant = compute_obstacle_precision_recall(gt, pred, match_radius_cells=1)
        self.assertTrue(tolerant["available"])
        self.assertEqual(tolerant["tp"], 1)
        self.assertEqual(tolerant["fp"], 1)
        self.assertEqual(tolerant["fn"], 1)
        self.assertGreater(tolerant["f1"], exact["f1"])
        self.assertEqual(tolerant["match_radius_cells"], 1)

    def test_obstacle_object_precision_recall_scores_components(self):
        gt = {1: {(10, 10), (10, 11), (11, 10), (30, 30), (30, 31)}}
        pred = {1: {(10, 10), (30, 31), (45, 45)}}

        out = compute_obstacle_object_precision_recall(gt, pred)
        self.assertTrue(out["available"])
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fp"], 1)
        self.assertEqual(out["fn"], 0)
        self.assertEqual(out["metric"], "object_components")

    def test_obstacle_object_precision_recall_uses_grouped_components(self):
        gt_components = {
            1: [
                {(10, 10), (12, 10)},
                {(30, 30), (31, 30)},
            ]
        }
        pred = {1: {(10, 10), (11, 10), (30, 30)}}
        out = compute_obstacle_object_precision_recall_from_components(gt_components, pred, match_radius_cells=1)
        self.assertTrue(out["available"])
        self.assertEqual(out["tp"], 2)
        self.assertEqual(out["fn"], 0)

    def test_metric_selection_reports_primary_and_alternate(self):
        selection = select_benchmark_obstacle_metric(
            cell_metric={"available": True, "f1": 0.12},
            object_metric={"available": True, "f1": 0.41},
            configured_metric="cell",
        )
        self.assertEqual(selection["selected_metric"], "object")
        self.assertEqual(selection["alternate_metric"], "cell")
        self.assertEqual(selection["selection_reason"], "object_metric_better_matches_blob_annotations")

    def test_default_class_aware_footprint_expands_large_tables(self):
        mapper = LiveMapper(grid_size=80, meters_per_cell=0.1)
        small = mapper._obstacle_footprint_cells((20, 20), label="table", det={"bbox": (300, 200, 340, 240), "frame_h": 480, "frame_w": 640})
        large = mapper._obstacle_footprint_cells((20, 20), label="table", det={"bbox": (200, 120, 520, 400), "frame_h": 480, "frame_w": 640})
        self.assertGreater(len(large), len(small))

    def test_event_summary_tracks_anchor_and_footprint_counts(self):
        mapper = LiveMapper(grid_size=50, meters_per_cell=0.1, confidence_weighting=False)
        tracked = {1: {"label": "chair", "confidence": 0.95, "center": (320, 240), "bbox": (290, 190, 350, 280), "area": 5400}}
        mapper.update_from_tracked(tracked, frame_shape=(480, 640, 3), frame_index=1, timestamp=1.0)
        summary = mapper.event_summary()
        self.assertEqual(summary["projection_anchor_mode"], "bottom_center")
        self.assertGreaterEqual(summary["anchor_events"], 1)
        self.assertGreaterEqual(summary["footprint_events"], 1)

    def test_promotion_status_variants(self):
        base = {
            "loop_closure_drift": {"available": True},
            "map_consistency_score": {"available": True, "score_mean": 0.9},
            "pose_jitter": {"available": True, "jitter_score": 0.8},
            "obstacle_persistence": {"available": True, "iou_mean": 0.5},
            "occupancy_concentration": {"available": True, "concentration_score": 0.3},
        }
        s_pass = compute_mapping_quality_summary(
            loop_closure_drift=base["loop_closure_drift"],
            map_consistency_score=base["map_consistency_score"],
            pose_jitter=base["pose_jitter"],
            obstacle_persistence=base["obstacle_persistence"],
            occupancy_concentration=base["occupancy_concentration"],
            obstacle_precision_recall={"available": True, "f1": 0.9},
            require_benchmark=True,
        )
        self.assertEqual(s_pass["status"], "promotable")

        s_fail = compute_mapping_quality_summary(
            loop_closure_drift=base["loop_closure_drift"],
            map_consistency_score={"available": True, "score_mean": 0.2},
            pose_jitter=base["pose_jitter"],
            obstacle_persistence=base["obstacle_persistence"],
            occupancy_concentration=base["occupancy_concentration"],
            obstacle_precision_recall={"available": True, "f1": 0.9},
            require_benchmark=True,
        )
        self.assertEqual(s_fail["status"], "not_promotable")

        s_insufficient = compute_mapping_quality_summary(
            loop_closure_drift=base["loop_closure_drift"],
            map_consistency_score=base["map_consistency_score"],
            pose_jitter=base["pose_jitter"],
            obstacle_persistence=base["obstacle_persistence"],
            occupancy_concentration=base["occupancy_concentration"],
            obstacle_precision_recall={"available": False, "reason": "missing"},
            require_benchmark=True,
        )
        self.assertEqual(s_insufficient["status"], "insufficient_evidence")

    def test_pose_jitter_filters_idle_frames(self):
        poses = [
            PoseSample(0.0, 0.0, 0.0, 0.0),
            PoseSample(0.0, 0.0, 0.0, 1.0),
            PoseSample(0.0, 0.0, 0.0, 2.0),
            PoseSample(0.01, 0.0, 0.01, 3.0),
            PoseSample(0.02, 0.0, 0.02, 4.0),
            PoseSample(0.03, 0.0, 0.03, 5.0),
        ]
        out = compute_pose_jitter_score(poses, min_motion_m=0.005)
        self.assertTrue(out["available"])
        self.assertEqual(out["moving_samples"], 3)

    def test_loop_closure_revisit_reduces_drift(self):
        mapper = LiveMapper(
            loop_closure_enabled=True,
            loop_closure_radius_m=0.30,
            loop_closure_min_frame_gap=15,
            loop_closure_max_heading_delta_rad=3.2,
            loop_closure_correction_alpha=0.5,
            loop_closure_cooldown_frames=2,
            pose_smoothing_window=1,
        )
        for i in range(20):
            mapper.update_pose_from_flow(dx_px=10, dy_px=0, dtheta_rad=0.0, timestamp=float(i + 1), motion_to_meter_scale=0.01, flow_quality=1.0)
        for i in range(20):
            mapper.update_pose_from_flow(dx_px=-7.5, dy_px=0, dtheta_rad=0.0, timestamp=float(i + 21), motion_to_meter_scale=0.01, flow_quality=1.0)
        raw_end = mapper.pose_history[-1]
        corrected_end = mapper.corrected_pose_history[-1]
        origin = mapper.pose_history[0]
        raw_dist = ((raw_end.x - origin.x) ** 2 + (raw_end.y - origin.y) ** 2) ** 0.5
        corr_dist = ((corrected_end.x - origin.x) ** 2 + (corrected_end.y - origin.y) ** 2) ** 0.5
        self.assertGreaterEqual(mapper.loop_closure_corrections_applied, 1)
        self.assertLess(corr_dist, raw_dist)

    def test_loop_closure_rejects_heading_mismatch(self):
        mapper = LiveMapper(
            loop_closure_enabled=True,
            loop_closure_radius_m=0.35,
            loop_closure_min_frame_gap=10,
            loop_closure_max_heading_delta_rad=0.10,
            loop_closure_correction_alpha=0.5,
            pose_smoothing_window=1,
        )
        for i in range(15):
            mapper.update_pose_from_flow(dx_px=8, dy_px=0, dtheta_rad=0.0, timestamp=float(i + 1), motion_to_meter_scale=0.01, flow_quality=1.0)
        for i in range(15):
            mapper.update_pose_from_flow(dx_px=-8, dy_px=0, dtheta_rad=1.2, timestamp=float(i + 16), motion_to_meter_scale=0.01, flow_quality=1.0)
        self.assertEqual(mapper.loop_closure_corrections_applied, 0)

    def test_loop_closure_cooldown_prevents_oscillation(self):
        mapper = LiveMapper(
            loop_closure_enabled=True,
            loop_closure_radius_m=0.30,
            loop_closure_min_frame_gap=8,
            loop_closure_max_heading_delta_rad=3.2,
            loop_closure_correction_alpha=0.4,
            loop_closure_cooldown_frames=10,
            pose_smoothing_window=1,
        )
        for i in range(30):
            dx = 9 if i < 15 else -9
            mapper.update_pose_from_flow(dx_px=dx, dy_px=0, dtheta_rad=0.0, timestamp=float(i + 1), motion_to_meter_scale=0.01, flow_quality=1.0)
        self.assertGreaterEqual(mapper.loop_closure_corrections_applied, 1)
        self.assertLessEqual(mapper.loop_closure_corrections_applied, 4)
        summary = mapper.loop_closure_summary()
        self.assertIn(summary["state"], ("idle", "candidate", "correcting", "cooldown"))
        self.assertGreaterEqual(summary["post_closure_path_alignment_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
