import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_release.py"
SPEC = importlib.util.spec_from_file_location("validate_release", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseValidationTests(unittest.TestCase):
    def test_validate_report_sections_accepts_complete_report(self):
        report = {
            "label_metrics": {},
            "map_metrics": {},
            "pose_stats": {},
            "config": {},
            "timing": {},
        }
        MODULE.validate_report_sections(report, ["label_metrics", "map_metrics", "pose_stats", "config", "timing"])

    def test_validate_report_sections_rejects_missing_sections(self):
        report = {"label_metrics": {}, "timing": {}}
        with self.assertRaises(ValueError):
            MODULE.validate_report_sections(report, ["label_metrics", "map_metrics", "pose_stats", "config", "timing"])

    def test_release_contract_has_required_fields(self):
        contract_path = Path(__file__).resolve().parents[1] / "deployment" / "release_contract.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        self.assertEqual(contract["deployment_target"], "single_machine_cli")
        self.assertIn("artifacts", contract)
        self.assertIn("runtime_profile", contract)
        self.assertIn("required_report_sections", contract)
        self.assertIn("promotion_policy", contract)


if __name__ == "__main__":
    unittest.main()
