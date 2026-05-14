import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_mapping_benchmark_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_mapping_benchmark_sweep", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MappingBenchmarkSweepTests(unittest.TestCase):
    def test_guardrails_require_core_map_health(self):
        summary = {
            "map_consistency": 0.9,
            "pose_jitter": 0.6,
            "obstacle_persistence": 0.25,
        }
        self.assertTrue(MODULE.passes_guardrails(summary, MODULE.DEFAULT_THRESHOLDS))
        summary["pose_jitter"] = 0.2
        self.assertFalse(MODULE.passes_guardrails(summary, MODULE.DEFAULT_THRESHOLDS))

    def test_candidate_sort_key_prefers_object_then_cell_f1(self):
        a = {
            "map_consistency": 0.9,
            "pose_jitter": 0.7,
            "obstacle_persistence": 0.3,
            "object_f1": 0.5,
            "cell_f1": 0.2,
            "occupancy_concentration": 0.8,
        }
        b = {
            "map_consistency": 0.9,
            "pose_jitter": 0.7,
            "obstacle_persistence": 0.3,
            "object_f1": 0.4,
            "cell_f1": 0.9,
            "occupancy_concentration": 0.8,
        }
        self.assertGreater(MODULE.candidate_sort_key(a), MODULE.candidate_sort_key(b))


if __name__ == "__main__":
    unittest.main()
