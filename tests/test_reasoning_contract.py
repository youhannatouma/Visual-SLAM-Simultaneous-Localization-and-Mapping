import importlib.util
import unittest


class ReasoningContractTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for reasoning contract tests")
    def test_reasoning_feature_contract_matches_model_constants(self):
        import reasoning

        engine = reasoning.ReasoningEngine(model_path="models/__missing_test_model__.pt")
        features = engine.extract_features(
            detections=[
                {
                    "label": "chair",
                    "confidence": 0.9,
                    "center": (320, 240),
                    "bbox": (300, 220, 340, 260),
                    "area": 1600,
                }
            ],
            frame_center=(320, 240),
            frame_area=640 * 480,
            motion_text="Moving Right",
        )
        self.assertEqual(len(features), reasoning.FEATURE_SIZE)
        self.assertEqual(reasoning.SEQUENCE_LENGTH, 10)
        self.assertEqual(reasoning.ACTION_CLASSES, ["AVOID_PERSON", "MOVE_TO_CHAIR", "CHECK_TABLE", "EXPLORE"])


if __name__ == "__main__":
    unittest.main()
