"""226-class live mode: loads the ensemble and RTMW, turns one recorded sign into a prediction.

The browser records a sign as a short clip and streams its frames. The server
extracts landmarks from each frame as it arrives (`landmarks`) and asks the
ensemble once the clip is complete (`predict`).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from signturk_runtime.config import load_config, load_vocabulary, missing_files
from signturk_runtime.features import build_feature_bundle, hand_motion, sample_indices
from signturk_runtime.pose_quality import evaluate_pose_quality
from text_processing.lexicon import restore_turkish

logger = logging.getLogger("signturk")

# Mean frame-to-frame hand movement below which the clip holds no sign.
MIN_HAND_MOTION = 0.004
MAX_FRAME_WIDTH = 640


def prepare_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """RGB frame at most 640 px wide, as the runtime was used before."""
    height, width = frame_bgr.shape[:2]
    if width > MAX_FRAME_WIDTH:
        frame_bgr = cv2.resize(frame_bgr, (MAX_FRAME_WIDTH, int(height * MAX_FRAME_WIDTH / width)),
                               interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class Live226:
    def __init__(self):
        self.config = load_config()
        self.vocabulary = load_vocabulary()
        self.frame_count = int(self.config["frame_count"])
        self.enabled = os.environ.get("SIGNTURK_ENABLE_226", "1").strip().lower() not in {"0", "false", "no"}
        # Run the person detector on every Nth frame and reuse its box in between.
        self.detect_every = max(1, int(os.environ.get("SIGNTURK_226_DETECT_EVERY", "4")))
        self.state = "idle" if self.enabled else "disabled"   # idle, loading, ready, missing, error, disabled
        self.message = "" if self.enabled else "SIGNTURK_ENABLE_226=0 ile kapatıldı"
        self.device = None
        self.ensemble = None
        self.extractor = None
        self.frame_ms = deque(maxlen=200)
        self.detect_ms = deque(maxlen=200)
        self.pose_ms = deque(maxlen=200)

    @staticmethod
    def _average(values):
        return round(float(np.mean(values)), 1) if values else None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def status(self) -> dict:
        return {
            "enabled": self.enabled,
            "state": self.state,
            "ready": self.ready,
            "message": self.message,
            "device": self.device,
            "num_classes": len(self.vocabulary),
            "frame_count": self.frame_count,
            "detect_every": self.detect_every,
            "avg_frame_ms": self._average(self.frame_ms),
            "avg_detect_ms": self._average(self.detect_ms),
            "avg_pose_ms": self._average(self.pose_ms),
        }

    def reset_timings(self) -> None:
        self.frame_ms.clear()
        self.detect_ms.clear()
        self.pose_ms.clear()

    def start_loading(self) -> None:
        """Load the models in a background thread so the server starts at once."""
        if self.state != "idle":
            return
        missing = missing_files(self.config)
        if missing:
            self.state = "missing"
            self.message = "Model dosyaları eksik: python tools/download_models.py --bundle 226"
            logger.warning("[226] missing files: %s", ", ".join(missing))
            return
        self.state = "loading"
        self.message = "Model yükleniyor"
        threading.Thread(target=self._load, name="live226-load", daemon=True).start()

    def _load(self) -> None:
        started = time.time()
        try:
            from signturk_runtime.ensemble import Ensemble
            from signturk_runtime.extractor import Extractor

            ensemble = Ensemble(self.config)
            ensemble.load()
            ensemble.warm_up()
            extractor = Extractor(self.config)
            self.ensemble, self.extractor, self.device = ensemble, extractor, extractor.device
            self.state, self.message = "ready", ""
            logger.info("[226] ready on %s in %.1fs", self.device, time.time() - started)
        except Exception as exc:
            self.state, self.message = "error", f"226 modeli yüklenemedi: {exc}"
            logger.exception("[226] load failed")

    def landmarks(self, frame_rgb: np.ndarray, boxes=None):
        """(75, 4) landmarks for one frame and the person boxes that were used."""
        started = time.perf_counter()
        result = self.extractor(frame_rgb, boxes)
        self.frame_ms.append((time.perf_counter() - started) * 1000.0)
        detect_ms = getattr(self.extractor, "last_detect_ms", 0.0)
        if detect_ms:
            self.detect_ms.append(detect_ms)
        self.pose_ms.append(getattr(self.extractor, "last_pose_ms", 0.0))
        return result

    def predict(self, clip: list) -> dict:
        """Top-3 for a clip of per-frame landmarks."""
        window = np.stack([clip[int(i)] for i in sample_indices(len(clip), self.frame_count)])
        started = time.perf_counter()
        probabilities = self.ensemble.predict(build_feature_bundle(window))
        latency_ms = (time.perf_counter() - started) * 1000.0

        top = []
        for index in np.argsort(-probabilities)[:3]:
            entry = self.vocabulary[int(index)]
            top.append({
                "class_id": entry["autsl_id"],
                "gloss": entry["gloss"],
                "text_tr": restore_turkish(entry["gloss"]),
                "confidence": round(float(probabilities[index]), 4),
            })
        margin = top[0]["confidence"] - top[1]["confidence"]
        thresholds = self.config["thresholds"]
        motion = hand_motion(window)
        quality = evaluate_pose_quality(clip[len(clip) // 2])
        feedback = None
        if motion < MIN_HAND_MOTION:
            feedback = "Hareket algılanmadı"
        elif not quality.ok:
            feedback = quality.feedback
        return {
            "top": top,
            "confident": top[0]["confidence"] >= thresholds["top1"] and margin >= thresholds["margin"],
            "margin": round(margin, 4),
            "hand_motion": round(motion, 5),
            "feedback": feedback,
            "model_ms": round(latency_ms, 1),
        }
