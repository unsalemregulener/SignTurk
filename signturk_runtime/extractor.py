"""RTMW whole-body keypoints (YOLOX person detector + RTMW pose, ONNX Runtime)."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from .config import MODEL_ROOT
from .features import wholebody_to_landmarks

logger = logging.getLogger("signturk")


def choose_device(setting: str, available_providers) -> str:
    """Pick "cuda" or "cpu" for a SIGNTURK_226_DEVICE value: "auto", "cuda" or "cpu"."""
    if (setting or "auto").strip().lower() == "cpu":
        return "cpu"
    return "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"


class Extractor:
    def __init__(self, config: dict, root: Path = MODEL_ROOT, device: str = None):
        import onnxruntime

        # Only errors: the models trigger harmless warnings (unused initializers, shape ops kept on the CPU).
        onnxruntime.set_default_logger_severity(3)
        settings = config["extractor"]
        if device is None:
            setting = os.environ.get("SIGNTURK_226_DEVICE", "auto")
            device = choose_device(setting, onnxruntime.get_available_providers())
            if setting.strip().lower() == "cuda" and device != "cuda":
                logger.warning("[226] SIGNTURK_226_DEVICE=cuda, but this ONNX Runtime has no CUDA provider")
        self.keypoint_threshold = float(settings["keypoint_threshold"])
        if device == "cuda":
            # onnxruntime-gpu[cuda,cudnn] installs CUDA and cuDNN through pip; load them before the sessions.
            try:
                onnxruntime.preload_dlls()
            except Exception as exc:  # noqa: BLE001 - older releases or a partial install; the check below decides
                logger.warning("[226] could not preload the CUDA libraries: %s", exc)
            try:
                self._create_sessions(root, settings, "cuda")
            except Exception as exc:  # noqa: BLE001 - fall back to the CPU below
                logger.warning("[226] CUDA sessions failed: %s", exc)
                device = "cpu"
            else:
                if not self._sessions_on_cuda():
                    logger.warning("[226] ONNX Runtime fell back to the CPU; check the CUDA and cuDNN install")
                    device = "cpu"
        if device != "cuda":
            device = "cpu"
            self._create_sessions(root, settings, device)
        self.device = device
        self.last_detect_ms = 0.0   # 0 when the previous boxes were reused
        self.last_pose_ms = 0.0

    def _create_sessions(self, root: Path, settings: dict, device: str) -> None:
        from rtmlib import RTMPose, YOLOX

        self.detector = YOLOX(str(root / settings["detector"]), model_input_size=tuple(settings["detector_input_size"]),
                              backend="onnxruntime", device=device)
        self.pose = RTMPose(str(root / settings["pose"]), model_input_size=tuple(settings["pose_input_size"]),
                            to_openpose=False, backend="onnxruntime", device=device)

    def _sessions_on_cuda(self) -> bool:
        return all(tool.session.get_providers()[0] == "CUDAExecutionProvider" for tool in (self.detector, self.pose))

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
