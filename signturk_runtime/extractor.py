"""RTMW whole-body keypoints (YOLOX person detector + RTMW pose, ONNX Runtime)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import MODEL_ROOT
from .features import wholebody_to_landmarks


class Extractor:
    def __init__(self, config: dict, root: Path = MODEL_ROOT, device: str = None):
        import onnxruntime
        from rtmlib import RTMPose, YOLOX

        settings = config["extractor"]
        if device is None:
            device = "cuda" if "CUDAExecutionProvider" in onnxruntime.get_available_providers() else "cpu"
        self.device = device
        self.keypoint_threshold = float(settings["keypoint_threshold"])
        self.detector = YOLOX(str(root / settings["detector"]), model_input_size=tuple(settings["detector_input_size"]),
                              backend="onnxruntime", device=device)
        self.pose = RTMPose(str(root / settings["pose"]), model_input_size=tuple(settings["pose_input_size"]),
                            to_openpose=False, backend="onnxruntime", device=device)
        self.last_detect_ms = 0.0   # 0 when the previous boxes were reused
        self.last_pose_ms = 0.0

    def __call__(self, frame_rgb: np.ndarray, boxes=None):
        """(75, 4) landmarks for one RGB frame, and the person boxes that were used.

        Pass the returned boxes back in to skip detection on the next frame.
        """
        image_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        self.last_detect_ms = 0.0
        if boxes is None:
            started = time.perf_counter()
            boxes = self.detector(image_bgr)
            self.last_detect_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        keypoints, scores = self.pose(image_bgr, bboxes=boxes)
        self.last_pose_ms = (time.perf_counter() - started) * 1000.0
        return wholebody_to_landmarks(keypoints, scores, frame_rgb.shape, self.keypoint_threshold), boxes
