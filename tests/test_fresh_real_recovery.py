import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    script_path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PREPARE = load_module("prepare_reasoning_data", "scripts/prepare_reasoning_data.py")
SEED_SWEEP = load_module("run_reasoning_seed_sweep", "scripts/run_reasoning_seed_sweep.py")
TRACK1 = load_module("run_track1_reasoning_loop", "scripts/run_track1_reasoning_loop.py")


class FreshRealRecoveryTests(unittest.TestCase):
    def test_holdout_builder_rejects_insufficient_per_class_source_diversity(self):
        rows = []
        for label in PREPARE.ACTION_CLASSES:
            for idx in range(8):
                rows.append(
                    {
                        "label": label,
                        "source_type": "real_media",
                        "__source_file": f"{label.lower()}_shared.csv",
                        "batch_id": f"{label.lower()}_shared",
                        "needs_review": 0,
                        "scenario": "clutter_low_light",
                    }
                )
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "per-class source diversity"):
                PREPARE.holdout_multisource_balanced(
                    df=df,
                    out_dir=tmpdir,
                    holdout_per_class=4,
                    holdout_min_total=4,
                    holdout_min_sources=1,
                    holdout_min_reviewed_per_class=4,
                    holdout_min_class_sources=2,
                    holdout_require_scenario_tags=True,
                    holdout_sequence_length=2,
                    holdout_summary_path=str(Path(tmpdir) / "holdout_summary.json"),
                )

    def test_seed_sweep_ranking_prefers_fresh_real_and_worst_class(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            candidate_a = tmp / "a.json"
            candidate_b = tmp / "b.json"
            baseline = tmp / "baseline.json"

            baseline.write_text(
                json.dumps(
                    {
                        "accuracy": {"test": 0.80},
                        "per_class": {
                            label: {"f1": 0.70} for label in SEED_SWEEP.ACTION_CLASSES
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_a.write_text(
                json.dumps(
                    {
                        "accuracy": {"test": 0.91},
                        "macro_f1": 0.89,
                        "per_class": {
                            "AVOID_PERSON": {"f1": 0.90},
                            "MOVE_TO_CHAIR": {"f1": 0.90},
                            "CHECK_TABLE": {"f1": 0.90},
                            "EXPLORE": {"f1": 0.90},
                        },
                        "fresh_real_eval": {
                            "accuracy": 0.66,
                            "macro_f1": 0.64,
                            "per_class": {
                                "AVOID_PERSON": {"f1": 0.90},
                                "MOVE_TO_CHAIR": {"f1": 0.60},
                                "CHECK_TABLE": {"f1": 0.52},
                                "EXPLORE": {"f1": 0.66},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidate_b.write_text(
                json.dumps(
                    {
                        "accuracy": {"test": 0.93},
                        "macro_f1": 0.91,
                        "per_class": {
                            "AVOID_PERSON": {"f1": 0.92},
                            "MOVE_TO_CHAIR": {"f1": 0.92},
                            "CHECK_TABLE": {"f1": 0.92},
                            "EXPLORE": {"f1": 0.92},
                        },
                        "fresh_real_eval": {
                            "accuracy": 0.63,
                            "macro_f1": 0.64,
                            "per_class": {
                                "AVOID_PERSON": {"f1": 0.90},
                                "MOVE_TO_CHAIR": {"f1": 0.55},
                                "CHECK_TABLE": {"f1": 0.41},
                                "EXPLORE": {"f1": 0.70},
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            best, ranked = SEED_SWEEP.select_and_rank_runs(
                SEED_SWEEP.load_metrics(str(baseline)),
                {1: str(candidate_a), 2: str(candidate_b)},
                {1: {"promotable": False}, 2: {"promotable": False}},
            )

            self.assertEqual(best["seed"], 1)
            self.assertGreaterEqual(ranked[0]["worst_fresh_real_f1"], ranked[1]["worst_fresh_real_f1"])

    def test_recovery_policy_fails_when_check_table_is_still_too_low(self):
        evaluation = TRACK1.evaluate_recovery_policy(
            {
                "fresh_real_eval": {
                    "accuracy": 0.72,
                    "macro_f1": 0.67,
                    "per_class": {
                        "AVOID_PERSON": {"f1": 0.80},
                        "MOVE_TO_CHAIR": {"f1": 0.60},
                        "CHECK_TABLE": {"f1": 0.45},
                        "EXPLORE": {"f1": 0.66},
                    },
                }
            }
        )
        self.assertFalse(evaluation["passed"])
        self.assertEqual(evaluation["worst_class"], "CHECK_TABLE")
        self.assertIn("CHECK_TABLE", evaluation["failed_labels"])


if __name__ == "__main__":
    unittest.main()
