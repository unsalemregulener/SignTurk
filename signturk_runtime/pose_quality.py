"""Framing check for one frame of 226-class landmarks: can the camera see the signing space?

The model learnt from AUTSL, where the signer stands back from the camera:
the frame reaches about 2.4 shoulder widths below the shoulder line, so the
hands are in view at rest and while signing in front of the chest. Sitting
close to a laptop camera cuts that space off. On the AUTSL test split, with
the frame ending 1.5 shoulder widths below the shoulders accuracy drops from
95.0% to 93.5%, at 1.0 to 91.5% and at 0.5 to 81.8%.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import LANDMARK_DIM, TOTAL_LANDMARKS, frame_bounds

NOSE, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 11, 12
SEEN = 0.25              # keypoint score above which a point counts as seen
MIN_SPACE_BELOW = 1.2    # shoulder widths of frame needed below the shoulder line
CLOSE_WIDTH = 0.22       # shoulders wider than this share of the frame: the signer is close
MIN_WIDTH = 0.08         # narrower than this: too far for the hands to be read
CENTRE = (0.25, 0.75)    # where the middle of the shoulders may lie, as a share of the width


@dataclass
class Framing:
    ok: bool
    hint: Optional[str]                # shown to the user, Turkish
    space_below: Optional[float] = None
    width: Optional[float] = None      # shoulder width as a share of the frame width

    def as_dict(self) -> dict:
        return {"ok": self.ok, "hint": self.hint,
                "space_below": None if self.space_below is None else round(self.space_below, 2),
                "width": None if self.width is None else round(self.width, 3)}


def evaluate_framing(landmarks: np.ndarray, image_shape=None) -> Framing:
    """Judge the last frame of `landmarks` ((75, 4) or (T, 75, 4)) taken from an image of `image_shape`."""
    arr = np.asarray(landmarks, dtype=np.float32)
    if arr.size == 0:
        return Framing(False, "Kameraya geçin")
    frame = arr.reshape(-1, TOTAL_LANDMARKS, LANDMARK_DIM)[-1]
    left, top, right, bottom = frame_bounds(image_shape) if image_shape is not None else (0.0, 0.0, 1.0, 1.0)

    nose, left_shoulder, right_shoulder = frame[NOSE], frame[LEFT_SHOULDER], frame[RIGHT_SHOULDER]
    if left_shoulder[3] < SEEN or right_shoulder[3] < SEEN:
        return Framing(False, "Kameraya geçin" if nose[3] < SEEN else "Omuzlarınız görünmeli")

    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_width < 1e-3:
        return Framing(False, "Kameraya geçin")
    width = abs(float(left_shoulder[0] - right_shoulder[0])) / max(right - left, 1e-6)
    space_below = (bottom - float(left_shoulder[1] + right_shoulder[1]) / 2.0) / shoulder_width
    centre = ((float(left_shoulder[0] + right_shoulder[0]) / 2.0) - left) / max(right - left, 1e-6)

    if space_below < MIN_SPACE_BELOW:
        hint = "Biraz geri çekilin" if width > CLOSE_WIDTH else "Kamerayı biraz aşağı eğin"
        return Framing(False, hint, space_below, width)
    if width < MIN_WIDTH:
        return Framing(False, "Biraz yaklaşın", space_below, width)
    if not CENTRE[0] <= centre <= CENTRE[1]:
        return Framing(False, "Kadrajın ortasına geçin", space_below, width)
    if nose[3] >= SEEN and float(nose[1]) < top + 0.01:
        return Framing(False, "Başınız görünmeli", space_below, width)
    return Framing(True, None, space_below, width)
