"""Framing check for one frame of 226-class landmarks: is the signer in view?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import HAND_NODES, LANDMARK_DIM, LEFT_HAND_NODES, RIGHT_HAND_NODES, TOTAL_LANDMARKS


@dataclass
class PoseQuality:
    ok: bool
    feedback: str   # shown to the user, Turkish
    score: float


def evaluate_pose_quality(
    landmarks: np.ndarray,
    min_shoulder_width: float = 0.16,
    max_shoulder_width: float = 0.55,
    center_x_range: tuple = (0.28, 0.72),
    upper_body_y_range: tuple = (0.15, 0.68),
    min_pose_confidence: float = 0.25,
    min_hand_visibility: float = 0.25,
) -> PoseQuality:
    """Judge the last frame of `landmarks` ((75, 4) or (T, 75, 4))."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return PoseQuality(False, "Kameraya geçin", 0.0)
    frame = arr.reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)[-1]

    nose, left_shoulder, right_shoulder = frame[0, 3], frame[11, 3], frame[12, 3]
    left_wrist, right_wrist = frame[15, 3], frame[16, 3]
    hand_visibility = max(float(frame[LEFT_HAND_NODES, 3].mean()), float(frame[RIGHT_HAND_NODES, 3].mean()))
    hand_signal = max(hand_visibility, float(left_wrist), float(right_wrist))

    visible_parts = sum(value >= min_pose_confidence for value in (nose, left_shoulder, right_shoulder, left_wrist, right_wrist))
    if visible_parts < 2:
        return PoseQuality(False, "Kameraya geçin", 0.0)
    if left_shoulder < min_pose_confidence or right_shoulder < min_pose_confidence:
        return PoseQuality(False, "Üst gövdeniz görünmeli", 0.25)

    shoulder_width = float(abs(frame[11, 0] - frame[12, 0]))
    torso_x = float((frame[11, 0] + frame[12, 0]) / 2.0)
    upper_body_y = float(frame[0, 1] if nose >= min_pose_confidence else (frame[11, 1] + frame[12, 1]) / 2.0)

    points = frame[HAND_NODES]
    points = points[points[:, 3] > 0]
    wrists = frame[[15, 16]]
    wrists = wrists[wrists[:, 3] >= min_pose_confidence]
    if len(wrists):
        points = np.concatenate([points, wrists], axis=0)
    in_frame = 0.0
    if len(points):
        xy = points[:, :2]
        in_frame = float(((xy[:, 0] >= 0.02) & (xy[:, 0] <= 0.98) & (xy[:, 1] >= 0.02) & (xy[:, 1] <= 0.98)).mean())

    if shoulder_width < min_shoulder_width:
        return PoseQuality(False, "Biraz yaklaşın", 0.45)
    if shoulder_width > max_shoulder_width:
        return PoseQuality(False, "Biraz uzaklaşın", 0.45)
    if not center_x_range[0] <= torso_x <= center_x_range[1]:
        return PoseQuality(False, "Kadrajın ortasına geçin", 0.55)
    if not upper_body_y_range[0] <= upper_body_y <= upper_body_y_range[1]:
        return PoseQuality(False, "Üst gövdeniz görünmeli", 0.55)
    if hand_signal < min_hand_visibility or in_frame < 0.15:
        return PoseQuality(False, "Ellerinizi kadraja alın", 0.65)

    score = float(np.clip(np.mean([
        min(1.0, shoulder_width / max(min_shoulder_width, 1e-6)),
        1.0 - min(abs(torso_x - 0.5) / 0.5, 1.0),
        min(hand_signal / max(min_hand_visibility, 1e-6), 1.0),
        in_frame,
    ]), 0.0, 1.0))
    return PoseQuality(True, "Kadraj uygun", score)
