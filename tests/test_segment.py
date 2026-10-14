import unittest

import numpy as np

from signturk_runtime.features import LANDMARK_DIM, TOTAL_LANDMARKS
from signturk_runtime.segment import Segmenter, active_span, resample, sign_clip, wrist_height

FPS = 12
SHOULDER_Y, SHOULDER_WIDTH = 0.5, 0.2


def frame(height):
    """Landmarks with both wrists `height` shoulder widths below the shoulders (None: wrists unseen)."""
    landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    landmarks[11] = (0.4, SHOULDER_Y, 0, 0.9)
    landmarks[12] = (0.6, SHOULDER_Y, 0, 0.9)
    if height is not None:
        for wrist, x in ((15, 0.35), (16, 0.65)):
            landmarks[wrist] = (x, SHOULDER_Y + height * SHOULDER_WIDTH, 0, 0.9)
    return landmarks


def recording(heights):
    return np.stack([frame(h) for h in heights])


REST = 1.4
SIGN = [REST] * 12 + [1.1, 0.8, 0.5, 0.2] + [0.0, 0.1] * 8 + [0.3, 0.7, 1.0, 1.3] + [REST] * 12


class WristHeightTests(unittest.TestCase):
    def test_height_in_shoulder_widths(self):
        np.testing.assert_allclose(wrist_height(recording([1.4, 0.0, -0.5])), [1.4, 0.0, -0.5], atol=1e-5)

    def test_unseen_wrists_have_no_height(self):
        self.assertTrue(np.isnan(wrist_height(recording([None]))[0]))


class ActiveSpanTests(unittest.TestCase):
    def test_finds_the_sign_from_rest_to_rest(self):
        first, last = active_span(recording(SIGN), FPS)
        self.assertTrue(9 <= first <= 12, first)          # the hands start rising at frame 12
        self.assertTrue(35 <= last <= 38, last)           # and are back at rest at frame 36

    def test_rest_wobbling_after_the_sign_is_not_kept(self):
        wobbly = SIGN[:36] + [1.25, 1.4, 1.2, 1.35, 1.3, 1.15, 1.4, 1.3, 1.25, 1.35, 1.2, 1.3]
        self.assertLessEqual(active_span(recording(wobbly), FPS)[1], 39)

    def test_no_sign_without_raised_hands(self):
        self.assertIsNone(active_span(recording([REST, 1.35, 1.3, REST] * 12), FPS))
        self.assertIsNone(active_span(recording([None] * 20), FPS))

    def test_hands_that_never_rest_keep_the_whole_recording(self):
        self.assertEqual(active_span(recording([0.2, 0.0, 0.3] * 16), FPS), (0, 47))


class ResampleTests(unittest.TestCase):
    def test_count_and_ends(self):
        clip = recording(SIGN)
        out = resample(clip, 32)
        self.assertEqual(out.shape, (32, TOTAL_LANDMARKS, LANDMARK_DIM))
        np.testing.assert_allclose(out[0], clip[0])
        np.testing.assert_allclose(out[-1], clip[-1])

    def test_unseen_points_take_the_nearer_frame(self):
        out = resample(recording([0.0, None]), 4)
        self.assertAlmostEqual(float(out[1, 15, 1]), SHOULDER_Y)   # not pulled towards 0
        self.assertEqual(float(out[2, 15, 3]), 0.0)                # still unseen

    def test_sign_clip_reads_the_span(self):
        window, span = sign_clip(recording(SIGN), FPS, 32)
        self.assertEqual(window.shape[0], 32)
        np.testing.assert_allclose(window[0], recording(SIGN)[span[0]])


class SegmenterTests(unittest.TestCase):
    @staticmethod
    def cuts(heights, **options):
        """(stream position when cut, (first, last), frames) for every sign the Segmenter returns."""
        segmenter, found = Segmenter(FPS, **options), []
        for position, landmarks in enumerate(recording(heights)):
            cut = segmenter.push(landmarks)
            if cut is not None:
                found.append((position, cut[1], len(cut[0])))
        return found

    def test_each_sign_is_cut_once_after_the_hands_rest(self):
        found = self.cuts(SIGN + SIGN[12:])       # rest, sign, rest, sign, rest
        self.assertEqual(len(found), 2, found)
        (position, (first, last), frames), (_, (second_first, _), _) = found
        self.assertTrue(9 <= first <= 12 and 35 <= last <= 38, found)
        self.assertLessEqual(position - 36, 6)    # within half a second of the hands resting
        self.assertEqual(frames, last - first + 1)
        self.assertTrue(45 <= second_first <= 48, found)

    def test_a_twitch_is_not_a_sign(self):
        self.assertEqual(self.cuts([REST] * 12 + [1.2, 0.9, 0.9, 1.2] + [REST] * 12), [])

    def test_rest_alone_gives_nothing(self):
        self.assertEqual(self.cuts([REST, 1.35] * 60), [])
        self.assertEqual(self.cuts([None] * 60), [])

    def test_hands_kept_up_are_cut_once(self):
        found = self.cuts([REST] * 12 + [0.0] * 120 + [REST] * 24, max_seconds=4.0)
        self.assertEqual(len(found), 1, found)

    def test_a_new_rest_height_still_ends_the_sign(self):
        lower_rest = [h if h < 1.1 else 1.1 for h in SIGN[12:]]   # back down to 1.1 instead of 1.4
        found = self.cuts(SIGN[:36] + lower_rest[24:] + lower_rest)
        self.assertEqual(len(found), 2, found)

    def test_the_stream_stays_short_while_the_hands_never_settle(self):
        segmenter = Segmenter(FPS, max_seconds=4.0)
        for landmarks in recording([REST] * 12 + [0.0] * 60 + [REST, REST, 0.0, 0.0] * 150):
            segmenter.push(landmarks)
        self.assertLessEqual(len(segmenter.frames), segmenter.max_frames)


if __name__ == "__main__":
    unittest.main()
