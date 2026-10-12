import json
import unittest
from pathlib import Path

import numpy as np

from live_pipeline import count_detected_hands, preprocess_sequence, resample_window

ROOT = Path(__file__).resolve().parents[1]


class ResampleWindowTests(unittest.TestCase):
    def test_keeps_first_and_last_frame(self):
        window = np.arange(32, dtype=np.float32)[:, None].repeat(126, axis=1)
        sampled = resample_window(window, 16)
        self.assertEqual(sampled.shape, (16, 126))
        self.assertEqual(sampled[0, 0], 0)
        self.assertEqual(sampled[-1, 0], 31)
        self.assertTrue(np.all(np.diff(sampled[:, 0]) > 0))

    def test_same_length_is_unchanged(self):
        window = np.random.default_rng(1).normal(size=(16, 126)).astype(np.float32)
        np.testing.assert_array_equal(resample_window(window, 16), window)

    def test_empty_window_is_rejected(self):
        with self.assertRaises(ValueError):
            resample_window([], 16)


class PreprocessingTests(unittest.TestCase):
    def test_capture_window_becomes_model_input(self):
        window = np.random.default_rng(42).normal(size=(32, 126)).astype(np.float32)
        features = preprocess_sequence(resample_window(window, 16), np.zeros(156), np.ones(156))
        self.assertEqual(features.shape, (1, 16, 156))
        self.assertTrue(np.isfinite(features).all())

    def test_detected_hand_count(self):
        landmarks = np.zeros(126, dtype=np.float32)
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 0)
        landmarks[0] = 1.0
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 1)
        landmarks[63] = 1.0
        self.assertEqual(count_detected_hands(landmarks, minimum_norm=0.1), 2)
        self.assertEqual(count_detected_hands(np.zeros(10), minimum_norm=0.1), 0)

    def test_model_config_contract(self):
        config = json.loads((ROOT / "demo_assets_179" / "demo_config.json").read_text(encoding="utf-8"))
        stats = json.loads((ROOT / "demo_assets_179" / "norm_stats.json").read_text(encoding="utf-8"))
        labels = json.loads((ROOT / "demo_assets_179" / "label_map.json").read_text(encoding="utf-8"))
        self.assertEqual(config["seq_len"], 16)
        self.assertEqual(config["capture_window"], 32)
        self.assertEqual(len(stats["mean"]), config["feat_dim"])
        self.assertEqual(len(stats["std"]), config["feat_dim"])
        self.assertEqual(sorted(int(k) for k in labels), list(range(config["num_classes"])))


if __name__ == "__main__":
    unittest.main()
