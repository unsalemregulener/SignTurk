import json
import unittest
from pathlib import Path

import numpy as np

from signturk_runtime.config import load_config, load_vocabulary
from signturk_runtime.features import (
    LEFT_HAND_NODES,
    RIGHT_HAND_NODES,
    WHOLEBODY_LEFT_HAND_START,
    WHOLEBODY_RIGHT_HAND_START,
    body_center_normalize,
    build_feature_bundle,
    hand_motion,
    sample_indices,
    wholebody_to_landmarks,
)
from signturk_runtime.pose_quality import evaluate_pose_quality
from text_processing.lexicon import SIGN_GLOSS_TURKISH

ROOT = Path(__file__).resolve().parents[1]


def person(width=640, height=480):
    """133 COCO-WholeBody keypoints in pixels for a centred, upright signer."""
    keypoints = np.zeros((133, 2), dtype=np.float32)
    keypoints[0] = (320, 120)                     # nose
    keypoints[5], keypoints[6] = (260, 220), (380, 220)   # shoulders
    keypoints[9], keypoints[10] = (250, 330), (390, 330)  # wrists
    keypoints[11], keypoints[12] = (280, 420), (360, 420)  # hips
    for i in range(21):
        keypoints[WHOLEBODY_LEFT_HAND_START + i] = (240 + i, 300 + i)
        keypoints[WHOLEBODY_RIGHT_HAND_START + i] = (400 - i, 300 + i)
    scores = np.full(133, 0.9, dtype=np.float32)
    return keypoints, scores, (height, width, 3)


class SamplingTests(unittest.TestCase):
    def test_long_clip_is_spread_evenly(self):
        indices = sample_indices(64, 32)
        self.assertEqual(len(indices), 32)
        self.assertEqual((indices[0], indices[-1]), (0, 63))
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_short_clip_repeats_last_frame(self):
        self.assertEqual(sample_indices(3, 5).tolist(), [0, 1, 2, 2, 2])
        self.assertEqual(sample_indices(0, 4).tolist(), [0, 0, 0, 0])


class LandmarkTests(unittest.TestCase):
    def test_wholebody_points_land_in_their_slots(self):
        keypoints, scores, shape = person()
        landmarks = wholebody_to_landmarks(keypoints, scores, shape)
        self.assertEqual(landmarks.shape, (75, 4))
        np.testing.assert_allclose(landmarks[11, :2], (260 / 640, 220 / 480))   # left shoulder
        np.testing.assert_allclose(landmarks[LEFT_HAND_NODES[0], :2], (240 / 640, 300 / 480))
        np.testing.assert_allclose(landmarks[RIGHT_HAND_NODES[0], :2], (400 / 640, 300 / 480))
        self.assertTrue(np.all(landmarks[:, 2] == 0))

    def test_low_scores_stay_empty_and_best_person_wins(self):
        keypoints, scores, shape = person()
        weak = scores.copy()
        weak[WHOLEBODY_LEFT_HAND_START:WHOLEBODY_LEFT_HAND_START + 21] = 0.01
        landmarks = wholebody_to_landmarks(keypoints, weak, shape)
        self.assertFalse(landmarks[LEFT_HAND_NODES].any())
        two = wholebody_to_landmarks(np.stack([keypoints * 0.5, keypoints]), np.stack([scores * 0.5, scores]), shape)
        np.testing.assert_allclose(two, wholebody_to_landmarks(keypoints, scores, shape))
        self.assertFalse(wholebody_to_landmarks(np.zeros((0, 133, 2)), np.zeros((0, 133)), shape).any())


class FeatureTests(unittest.TestCase):
    def setUp(self):
        keypoints, scores, shape = person()
        frame = wholebody_to_landmarks(keypoints, scores, shape)
        moves = np.zeros((32, 75, 4), dtype=np.float32)
        moves[:, LEFT_HAND_NODES, 0] = np.linspace(0, 0.05, 32)[:, None]
        self.clip = (frame[None] + moves).astype(np.float32)

    def test_streams_have_the_trained_shapes(self):
        bundle = build_feature_bundle(self.clip)
        self.assertEqual([x.shape for x in bundle.skeleton_inputs], [(1, 32, d) for d in (300, 144, 300, 144, 39)])
        self.assertEqual([x.shape for x in bundle.hand_inputs], [(1, 32, d) for d in (168, 168, 39)])
        self.assertTrue(all(np.isfinite(x).all() for x in bundle.skeleton_inputs + bundle.hand_inputs))

    def test_body_is_centred_on_the_shoulders(self):
        normalized = body_center_normalize(self.clip)
        np.testing.assert_allclose((normalized[0, 11, :2] + normalized[0, 12, :2]) / 2, (0, 0), atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(normalized[0, 11, :3] - normalized[0, 12, :3])), 1.0, places=5)

    def test_hand_motion(self):
        self.assertGreater(hand_motion(self.clip), 0.0)
        self.assertEqual(hand_motion(np.repeat(self.clip[:1], 5, axis=0)), 0.0)

    def test_framing_feedback(self):
        self.assertTrue(evaluate_pose_quality(self.clip[16]).ok)
        self.assertEqual(evaluate_pose_quality(np.zeros((75, 4))).feedback, "Kameraya geçin")


class VocabularyTests(unittest.TestCase):
    def test_every_class_has_a_gloss_with_avatar_and_spelling(self):
        vocabulary = load_vocabulary()
        glosses = [entry["gloss"] for entry in vocabulary]
        self.assertEqual(len(vocabulary), 226)
        self.assertEqual(sorted(entry["autsl_id"] for entry in vocabulary), list(range(226)))
        self.assertEqual(set(glosses), {p.stem for p in (ROOT / "landmarks").glob("*.json")})
        self.assertTrue(set(glosses) <= set(SIGN_GLOSS_TURKISH))

    def test_config(self):
        config = load_config()
        self.assertEqual((config["num_classes"], config["frame_count"]), (226, 32))
        self.assertAlmostEqual(sum(stream["weight"] for stream in config["streams"].values()), 1.0)
        self.assertEqual({stream["inputs"] for stream in config["streams"].values()}, {"skeleton", "hand"})

    def test_manifest_lists_every_model_file(self):
        config = load_config()
        manifest = json.loads((ROOT / "model-assets.json").read_text(encoding="utf-8"))["assets"]
        wanted = {f"research/model_226/{stream['file']}" for stream in config["streams"].values()}
        wanted |= {f"research/model_226/{config['extractor'][key]}" for key in ("detector", "pose")}
        self.assertEqual({a["path"] for a in manifest}, wanted)
        for asset in manifest:
            self.assertTrue(asset["url"].startswith("https://"), asset["name"])
            self.assertEqual(len(asset["sha256"]), 64, asset["name"])
            self.assertGreater(asset["bytes"], 1_000_000, asset["name"])


if __name__ == "__main__":
    unittest.main()
