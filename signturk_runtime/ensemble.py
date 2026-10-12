"""The four Keras streams of the 226-class model and their weighted average."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import MODEL_ROOT
from .features import FeatureBundle


@lru_cache(maxsize=1)
def _custom_objects() -> dict:
    import keras

    # The streams were saved with this layer under the "signaturk" package.
    @keras.saving.register_keras_serializable(package="signaturk")
    class TemporalAttentionPool(keras.layers.Layer):
        def call(self, inputs):
            values, weights = inputs
            return keras.ops.sum(values * weights, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0][0], input_shape[0][-1])

    return {"TemporalAttentionPool": TemporalAttentionPool}


class Ensemble:
    def __init__(self, config: dict, root: Path = MODEL_ROOT):
        self.config = config
        self.root = root
        self.models = {}

    def load(self) -> None:
        import keras

        for name, spec in self.config["streams"].items():
            self.models[name] = keras.models.load_model(
                self.root / spec["file"], custom_objects=_custom_objects(), compile=False)

    def warm_up(self) -> None:
        frames = int(self.config["frame_count"])
        self.predict(FeatureBundle(
            skeleton_inputs=[np.zeros((1, frames, d), np.float32) for d in (300, 144, 300, 144, 39)],
            hand_inputs=[np.zeros((1, frames, d), np.float32) for d in (168, 168, 39)],
        ))

    def predict(self, bundle: FeatureBundle) -> np.ndarray:
        """Class probabilities, shape (num_classes,)."""
        total = np.zeros(int(self.config["num_classes"]), dtype=np.float64)
        weight_sum = 0.0
        for name, spec in self.config["streams"].items():
            inputs = bundle.hand_inputs if spec["inputs"] == "hand" else bundle.skeleton_inputs
            probabilities = np.asarray(self.models[name].predict_on_batch(inputs), dtype=np.float64)[0]
            total += spec["weight"] * probabilities
            weight_sum += spec["weight"]
        total /= max(weight_sum, 1e-8)
        return (total / max(total.sum(), 1e-8)).astype(np.float32)
