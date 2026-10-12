"""Preprocessing for the live 179-class model (NumPy only, no TensorFlow).

The browser streams frames into a capture window (32 frames by default).
When the window is full it is resampled to the model's fixed input length
(16 frames) and turned into 156 features per frame:

    126 hand coordinates (MediaPipe "Left" hand 63 + "Right" hand 63)
      → wrist-relative, scale-normalized coordinates
      → + 30 finger-joint angles
      → z-score normalization
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# The live path runs MediaPipe's Hands solution, which names hands as if the
# image were mirrored: on an unmirrored camera frame its "Left" hand is the
# signer's right hand. The training data was extracted the same way, with
# "Left" first. (MediaPipe's Tasks HandLandmarker, used for the avatar files,
# names the signer's own hands, so its labels are the other way round.)
TRAINING_HAND_ORDER = ("Left", "Right")


def hands_to_landmarks(hands: Iterable[Tuple[str, np.ndarray]]) -> np.ndarray:
    """126 values in the training layout from (Hands solution label, 21x3 points) pairs.

    Expects an unmirrored frame. A missing hand stays zero.
    """
    slots = {label: np.zeros(63, dtype=np.float32) for label in TRAINING_HAND_ORDER}
    for label, points in hands:
        if label in slots and points is not None:
            slots[label] = np.asarray(points, dtype=np.float32).reshape(63)
    return np.concatenate([slots[label] for label in TRAINING_HAND_ORDER])


def hand_is_present(hand_vector: np.ndarray, minimum_norm: float) -> bool:
    return bool(np.linalg.norm(hand_vector) > minimum_norm)


def count_detected_hands(landmarks: Optional[np.ndarray], minimum_norm: float) -> int:
    if landmarks is None or landmarks.shape != (126,):
        return 0
    return int(hand_is_present(landmarks[:63], minimum_norm)) + int(
        hand_is_present(landmarks[63:], minimum_norm)
    )


def resample_window(frames: Sequence[np.ndarray], target_len: int) -> np.ndarray:
    """Pick `target_len` evenly spaced frames, keeping the first and the last one."""
    window = np.asarray(frames, dtype=np.float32)
    if len(window) == 0:
        raise ValueError("empty window")
    if len(window) == target_len:
        return window
    indices = np.linspace(0, len(window) - 1, target_len).round().astype(int)
    return window[indices]


def normalize_hands_relative(sequence: np.ndarray) -> np.ndarray:
    result = sequence.copy()
    frame_count = result.shape[0]
    for start in (0, 63):
        hand = result[:, start:start + 63].reshape(frame_count, 21, 3)
        hand_relative = hand - hand[:, 0:1, :]
        scale = np.linalg.norm(hand_relative[:, 9, :], axis=-1, keepdims=True)[:, :, np.newaxis]
        scale = np.where(scale < 1e-6, 1.0, scale)
        result[:, start:start + 63] = (hand_relative / scale).reshape(frame_count, 63)
    return result


def compute_finger_angles(sequence: np.ndarray) -> np.ndarray:
    frame_count = sequence.shape[0]
    chains = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    angle_sets = []
    for hand_start in (0, 63):
        hand = sequence[:, hand_start:hand_start + 63].reshape(frame_count, 21, 3)
        hand_angles = np.zeros((frame_count, 15), dtype=np.float32)
        angle_index = 0
        for chain in chains:
            for index in range(len(chain) - 1):
                first = hand[:, 0 if index == 0 else chain[index - 1], :]
                centre = hand[:, chain[index], :]
                last = hand[:, chain[index + 1], :]
                first_vector, last_vector = first - centre, last - centre
                cosine = np.sum(first_vector * last_vector, axis=-1) / (
                    np.linalg.norm(first_vector, axis=-1) * np.linalg.norm(last_vector, axis=-1) + 1e-8
                )
                hand_angles[:, angle_index] = np.arccos(np.clip(cosine, -1, 1))
                angle_index += 1
        angle_sets.append(hand_angles)
    return np.concatenate([sequence, np.concatenate(angle_sets, axis=-1)], axis=-1)


def preprocess_sequence(
    raw_landmarks: np.ndarray,
    normalization_mean: Optional[np.ndarray],
    normalization_std: Optional[np.ndarray],
) -> np.ndarray:
    """(T, 126) raw coordinates -> (1, T, 156) model input."""
    features = compute_finger_angles(normalize_hands_relative(raw_landmarks))
    if normalization_mean is not None and normalization_std is not None:
        features = (features - normalization_mean) / normalization_std
    return features[np.newaxis, :, :].astype(np.float32)
