"""Landmarks and model inputs for the 226-class ensemble (NumPy only).

Each frame becomes 75 landmarks with (x, y, z, confidence): 33 body slots in
MediaPipe's pose layout (only the 17 COCO body points are filled, z stays 0),
then 21 points for the left hand and 21 for the right hand, as named by
COCO-WholeBody (the signer's own left and right). Coordinates are normalised
to the image size.

A clip of 32 frames turns into five skeleton streams (joint, bone, their
frame-to-frame motion, extra geometry) and three hand streams.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

POSE_COUNT = 33
HAND_COUNT = 21
LANDMARK_DIM = 4
TOTAL_LANDMARKS = POSE_COUNT + 2 * HAND_COUNT

LEFT_HAND_NODES = list(range(POSE_COUNT, POSE_COUNT + HAND_COUNT))
RIGHT_HAND_NODES = list(range(POSE_COUNT + HAND_COUNT, TOTAL_LANDMARKS))
HAND_NODES = LEFT_HAND_NODES + RIGHT_HAND_NODES

# COCO-WholeBody keypoints: 17 body, 6 feet, 68 face, 21 left hand, 21 right hand.
WHOLEBODY_LEFT_HAND_START = 17 + 6 + 68
WHOLEBODY_RIGHT_HAND_START = WHOLEBODY_LEFT_HAND_START + HAND_COUNT

# COCO body keypoint -> MediaPipe pose slot
COCO17_TO_MEDIAPIPE33 = {
    0: 0,    # nose
    1: 2,    # left eye
    2: 5,    # right eye
    3: 7,    # left ear
    4: 8,    # right ear
    5: 11,   # left shoulder
    6: 12,   # right shoulder
    7: 13,   # left elbow
    8: 14,   # right elbow
    9: 15,   # left wrist
    10: 16,  # right wrist
    11: 23,  # left hip
    12: 24,  # right hip
    13: 25,  # left knee
    14: 26,  # right knee
    15: 27,  # left ankle
    16: 28,  # right ankle
}

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
POSE_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
FINGER_ANGLES = [(0, 2, 4), (0, 6, 8), (0, 10, 12), (0, 14, 16), (0, 18, 20)]


def sample_indices(total_frames: int, target_frames: int) -> np.ndarray:
    """Evenly spaced indices; a short clip is padded with its last frame."""
    if total_frames <= 0:
        return np.zeros(target_frames, dtype=np.int64)
    if total_frames >= target_frames:
        return np.linspace(0, total_frames - 1, target_frames).round().astype(np.int64)
    indices = list(range(total_frames)) + [total_frames - 1] * (target_frames - total_frames)
    return np.asarray(indices, dtype=np.int64)


def wholebody_to_landmarks(keypoints, scores, image_shape, keypoint_threshold: float = 0.05) -> np.ndarray:
    """(75, 4) landmarks from RTMW whole-body output for one frame.

    `keypoints` is (people, 133, 2) in pixels or (133, 2); with several people
    the one with the highest mean score is kept. Points below the threshold
    stay zero.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    landmarks = np.zeros((TOTAL_LANDMARKS, LANDMARK_DIM), dtype=np.float32)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim == 3:
        if keypoints.shape[0] == 0:
            return landmarks
        person = int(np.argmax(scores.mean(axis=1))) if scores.ndim == 2 else 0
        keypoints = keypoints[person]
        scores = scores[person] if scores.ndim == 2 else np.ones(len(keypoints), dtype=np.float32)
    if keypoints.ndim != 2 or keypoints.shape[0] < 17:
        return landmarks
    if scores.ndim != 1 or scores.shape[0] != keypoints.shape[0]:
        scores = np.ones(keypoints.shape[0], dtype=np.float32)

    xy = keypoints[:, :2].copy()
    xy[:, 0] = np.clip(xy[:, 0] / max(width, 1), 0.0, 1.0)
    xy[:, 1] = np.clip(xy[:, 1] / max(height, 1), 0.0, 1.0)

    def copy(source: int, target: int) -> None:
        if source < len(scores) and scores[source] >= keypoint_threshold:
            landmarks[target, :2] = xy[source]
            landmarks[target, 3] = scores[source]

    for source, target in COCO17_TO_MEDIAPIPE33.items():
        copy(source, target)
    for i in range(HAND_COUNT):
        copy(WHOLEBODY_LEFT_HAND_START + i, POSE_COUNT + i)
        copy(WHOLEBODY_RIGHT_HAND_START + i, POSE_COUNT + HAND_COUNT + i)
    return landmarks


def body_center_normalize(landmarks: np.ndarray) -> np.ndarray:
    """Origin at the middle of the shoulders, unit = shoulder width."""
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    xyz = arr[:, :, :3]
    left_shoulder, right_shoulder = xyz[:, 11:12], xyz[:, 12:13]
    center = (left_shoulder + right_shoulder) / 2.0
    scale = np.clip(np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True), 1e-3, None)
    return np.concatenate([(xyz - center) / scale, arr[:, :, 3:4]], axis=-1).astype(np.float32)


def bone_stream(normalized: np.ndarray) -> np.ndarray:
    bones = [normalized[:, b, :3] - normalized[:, a, :3] for a, b in POSE_BONES]
    for offset in (POSE_COUNT, POSE_COUNT + HAND_COUNT):
        bones += [normalized[:, offset + b, :3] - normalized[:, offset + a, :3] for a, b in HAND_BONES]
    return np.concatenate(bones, axis=-1).astype(np.float32)


def motion_stream(stream: np.ndarray) -> np.ndarray:
    motion = np.zeros_like(stream)
    motion[1:] = stream[1:] - stream[:-1]
    return motion.astype(np.float32)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    valid = denominator > 1e-6
    angle = np.zeros_like(denominator, dtype=np.float32)
    cosine = np.sum(ba[valid] * bc[valid], axis=-1) / denominator[valid]
    angle[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return angle[..., None]


def extra_stream(normalized: np.ndarray) -> np.ndarray:
    """39 values per frame: wrist offsets, hand speeds and distances, masks, finger angles."""
    xyz, visibility = normalized[:, :, :3], normalized[:, :, 3]
    nose, left_shoulder, right_shoulder = xyz[:, 0], xyz[:, 11], xyz[:, 12]
    torso = (left_shoulder + right_shoulder) / 2.0
    left_wrist, right_wrist = xyz[:, 15], xyz[:, 16]
    left_hand_wrist, right_hand_wrist = xyz[:, POSE_COUNT], xyz[:, POSE_COUNT + HAND_COUNT]
    hands_distance = np.linalg.norm(left_hand_wrist - right_hand_wrist, axis=-1, keepdims=True)
    pose_mask = (visibility[:, :POSE_COUNT].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    left_mask = (visibility[:, LEFT_HAND_NODES].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    right_mask = (visibility[:, RIGHT_HAND_NODES].sum(axis=1, keepdims=True) > 0).astype(np.float32)
    parts = [
        left_wrist - right_wrist,
        left_wrist - nose,
        right_wrist - nose,
        left_wrist - left_shoulder,
        right_wrist - right_shoulder,
        left_wrist - torso,
        right_wrist - torso,
        np.linalg.norm(motion_stream(left_hand_wrist), axis=-1, keepdims=True),
        np.linalg.norm(motion_stream(right_hand_wrist), axis=-1, keepdims=True),
        hands_distance,
        motion_stream(hands_distance),
        pose_mask,
        left_mask,
        right_mask,
        left_mask * right_mask,
    ]
    for offset in (POSE_COUNT, POSE_COUNT + HAND_COUNT):
        hand = xyz[:, offset:offset + HAND_COUNT]
        parts += [_angle(hand[:, a], hand[:, b], hand[:, c]) for a, b, c in FINGER_ANGLES]
    return np.concatenate(parts, axis=-1).astype(np.float32)


@dataclass
class FeatureBundle:
    skeleton_inputs: list   # joint (T, 300), bone (T, 144), their motion, extra (T, 39); each with a batch axis
    hand_inputs: list       # hand joint (T, 168), hand joint motion (T, 168), extra (T, 39)


def _hand_columns(stream: np.ndarray) -> np.ndarray:
    frames = stream.reshape(stream.shape[0], TOTAL_LANDMARKS, LANDMARK_DIM)
    return frames[:, HAND_NODES, :].reshape(stream.shape[0], -1)


def build_feature_bundle(landmarks: np.ndarray) -> FeatureBundle:
    """Model inputs from (T, 75, 4) landmarks."""
    normalized = body_center_normalize(landmarks)
    joint = normalized.reshape(normalized.shape[0], -1)
    bone = bone_stream(normalized)
    joint_motion = motion_stream(joint)
    extra = extra_stream(normalized)
    batch = [joint, bone, joint_motion, motion_stream(bone), extra]
    hands = [_hand_columns(joint), _hand_columns(joint_motion), extra]
    return FeatureBundle(
        skeleton_inputs=[x[None].astype(np.float32) for x in batch],
        hand_inputs=[x[None].astype(np.float32) for x in hands],
    )


def hand_motion(landmarks: np.ndarray) -> float:
    """Mean frame-to-frame movement of visible hand points, in image units."""
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    points, visible = arr[:, HAND_NODES, :2], arr[:, HAND_NODES, 3] > 0
    if len(arr) < 2 or not visible.any():
        return 0.0
    distance = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    both = visible[1:] & visible[:-1]
    return float(distance[both].mean()) if both.any() else 0.0
