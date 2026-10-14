"""Find the sign in a recording.

The AUTSL clips the model learnt from are cut tightly: the hands rest for a
moment, rise, sign and come back down, and the clip ends there. A recording
from the live page can hold seconds of rest around the sign, so the server
cuts it the same way before the model reads it.

The cut follows the height of the higher wrist, in shoulder widths below the
shoulder line. Its rest height is the lowest position in the recording. The
sign is where the wrist is clearly above it (`RAISED`). From there the cut
follows the hands back to rest on both sides: through every frame that is
still well above rest (`AWAY`) or lower than the frame before it, so a rest
height that wobbles on a webcam does not stretch the cut. A short margin of
rest is kept, as in AUTSL.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .features import LANDMARK_DIM, TOTAL_LANDMARKS

LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

RAISED = 0.40     # shoulder widths above rest: the hands are signing
AWAY = 0.25       # shoulder widths above rest: the hands are not back yet
MARGIN_SECONDS = 0.15
MIN_REST = 0.6    # if even the lowest wrist position is higher, the hands never rested


def wrist_height(landmarks: np.ndarray) -> np.ndarray:
    """Per frame, the higher visible wrist in shoulder widths below the shoulders; NaN if none is seen."""
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    shoulders_seen = (arr[:, LEFT_SHOULDER, 3] > 0) & (arr[:, RIGHT_SHOULDER, 3] > 0)
    centre_y = (arr[:, LEFT_SHOULDER, 1] + arr[:, RIGHT_SHOULDER, 1]) / 2.0
    width = np.linalg.norm(arr[:, LEFT_SHOULDER, :2] - arr[:, RIGHT_SHOULDER, :2], axis=-1)
    heights = np.full((len(arr), 2), np.nan, dtype=np.float32)
    for column, wrist in enumerate((LEFT_WRIST, RIGHT_WRIST)):
        seen = shoulders_seen & (arr[:, wrist, 3] > 0) & (width > 1e-3)
        heights[seen, column] = (arr[seen, wrist, 1] - centre_y[seen]) / width[seen]
    return np.fmin(heights[:, 0], heights[:, 1])


def _median3(values: np.ndarray) -> np.ndarray:
    """Three-frame median over the frames where a wrist is seen; NaN stays NaN."""
    if len(values) < 3:
        return values.copy()
    padded = np.concatenate([values[:1], values, values[-1:]])
    window = np.sort(np.stack([padded[:-2], padded[1:-1], padded[2:]], axis=1), axis=1)  # NaN sorts last
    seen = (~np.isnan(window)).sum(axis=1)
    smoothed = np.where(seen == 3, window[:, 1],
                        np.where(seen == 2, (window[:, 0] + window[:, 1]) / 2.0, window[:, 0]))
    smoothed[np.isnan(values)] = np.nan
    return smoothed.astype(np.float32)


def active_span(landmarks: np.ndarray, fps: float, raised: float = RAISED, away: float = AWAY,
                margin_seconds: float = MARGIN_SECONDS) -> Optional[Tuple[int, int]]:
    """First and last frame of the sign, margins included; None when the hands never rise."""
    height = _median3(wrist_height(landmarks))
    seen = ~np.isnan(height)
    if not seen.any():
        return None
    rest = float(np.nanmax(height))
    if rest < MIN_REST:
        return 0, len(height) - 1
    up = seen & (np.nan_to_num(height, nan=rest) < rest - raised)
    if not up.any():
        return None

    def returning(current: int, step: int) -> bool:
        following = current + step
        return (0 <= following < len(height) and bool(seen[following])
                and (height[following] < rest - away or height[following] > height[current]))

    first, last = int(np.argmax(up)), len(up) - 1 - int(np.argmax(up[::-1]))
    while returning(first, -1):
        first -= 1
    while returning(last, 1):
        last += 1
    margin = int(round(margin_seconds * fps))
    return max(0, first - margin), min(len(height) - 1, last + margin)


def resample(landmarks: np.ndarray, count: int) -> np.ndarray:
    """`count` frames evenly spaced in time, linearly interpolated.

    A point missing from one of the two neighbouring frames takes the nearer
    frame's value, so an unseen point never drifts towards the origin.
    """
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    if len(arr) == 1:
        return np.repeat(arr, count, axis=0)
    times = np.linspace(0.0, len(arr) - 1, count)
    before = np.floor(times).astype(np.int64)
    after = np.minimum(before + 1, len(arr) - 1)
    weight = (times - before)[:, None, None].astype(np.float32)
    mixed = arr[before] * (1.0 - weight) + arr[after] * weight
    nearest = np.where(weight >= 0.5, arr[after], arr[before])
    both = (arr[before, :, 3:4] > 0) & (arr[after, :, 3:4] > 0)
    return np.where(both, mixed, nearest).astype(np.float32)


def sign_clip(landmarks: np.ndarray, fps: float, count: int, **span_options) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    """The model's `count` frames, taken from the sign's span when one is found."""
    arr = np.asarray(landmarks, dtype=np.float32).reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)
    span = active_span(arr, fps, **span_options)
    first, last = span if span else (0, len(arr) - 1)
    return resample(arr[first:last + 1], count), span

