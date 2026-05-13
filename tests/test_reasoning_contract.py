import importlib.util
import tempfile
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

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch is required for reasoning contract tests")
    def test_reasoning_engine_loads_checkpoint_with_metadata(self):
        import torch
        import reasoning

        model = reasoning.ReasoningMLP(reasoning.FEATURE_SIZE, sequence_length=reasoning.SEQUENCE_LENGTH)
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "sequence_length": reasoning.SEQUENCE_LENGTH,
            "feature_size": reasoning.FEATURE_SIZE,
            "action_classes": list(reasoning.ACTION_CLASSES),
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            torch.save(checkpoint, f.name)
            engine = reasoning.ReasoningEngine(model_path=f.name)

        self.assertIsNotNone(engine.model)
        self.assertEqual(engine.model_metadata["feature_size"], reasoning.FEATURE_SIZE)


if __name__ == "__main__":
    unittest.main()
