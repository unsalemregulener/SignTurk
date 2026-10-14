"""YOLOX person boxes and RTMPose/RTMW keypoints with ONNX Runtime.

Adapted from rtmlib 0.0.16 (https://github.com/Tau-J/rtmlib), Apache-2.0; the
license is in LICENSES/rtmlib-Apache-2.0.txt. Only the parts SignTurk uses are
kept: YOLOX for models exported with NMS, the RTMPose top-down crop and the
SimCC decoding. Changes: sessions take a provider list with a CPU fallback,
the detector returns boxes only, and the OpenCV and OpenVINO backends, model
downloads and the other model classes are left out. The numbers are the same
as rtmlib's.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


def create_session(path: str, device: str = "cpu"):
    import onnxruntime

    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers.insert(0, "CUDAExecutionProvider")
    return onnxruntime.InferenceSession(path, providers=providers)


def run_session(session, image: np.ndarray) -> list:
    """Outputs for one HWC image, passed to the model as a (1, 3, H, W) float32 batch."""
    batch = np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None, :, :, :]
    names = [output.name for output in session.get_outputs()]
    return session.run(names, {session.get_inputs()[0].name: batch})


class YOLOX:
    """Person boxes (x1, y1, x2, y2) from a YOLOX model exported with NMS, as in rtmlib's human mode."""

    def __init__(self, path: str, input_size: Tuple[int, int] = (640, 640), device: str = "cpu",
                 score_threshold: float = 0.3):
        self.session = create_session(path, device)
        self.input_size = tuple(input_size)
        self.score_threshold = score_threshold

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        padded, ratio = self.preprocess(image_bgr)
        return self.postprocess(run_session(self.session, padded)[0], ratio)

    def preprocess(self, img: np.ndarray):
        """The image resized to fit, top-left aligned on a gray (114) canvas, and the scale factor."""
        if img.shape[:2] == tuple(self.input_size[:2]):
            return img.copy(), 1.
        padded_img = np.ones((self.input_size[0], self.input_size[1], 3), dtype=np.uint8) * 114
        ratio = min(self.input_size[0] / img.shape[0], self.input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[:int(img.shape[0] * ratio), :int(img.shape[1] * ratio)] = resized_img
        return padded_img, ratio

    def postprocess(self, outputs: np.ndarray, ratio: float) -> np.ndarray:
        if outputs.shape[-1] != 5:
            raise ValueError("expected a YOLOX model exported with NMS, with outputs of shape (1, N, 5)")
        final_boxes, final_scores = outputs[0, :, :4], outputs[0, :, 4]
        final_boxes /= ratio
        return final_boxes[final_scores > self.score_threshold]


class RTMPose:
    """Keypoints and scores for each box, from an RTMPose/RTMW model with SimCC outputs."""

    def __init__(self, path: str, input_size: Tuple[int, int] = (192, 256), device: str = "cpu",
                 mean: Sequence[float] = (123.675, 116.28, 103.53), std: Sequence[float] = (58.395, 57.12, 57.375)):
        self.session = create_session(path, device)
        self.input_size = tuple(input_size)   # (width, height)
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image_bgr: np.ndarray, boxes=()) -> Tuple[np.ndarray, np.ndarray]:
        """Keypoints (people, K, 2) in pixels and scores (people, K); the whole image without boxes."""
        if len(boxes) == 0:
            boxes = [[0, 0, image_bgr.shape[1], image_bgr.shape[0]]]
        keypoints, scores = [], []
        for box in boxes:
            crop, center, scale = self.preprocess(image_bgr, box)
            simcc_x, simcc_y = run_session(self.session, crop)
            points, score = self.postprocess(simcc_x, simcc_y, center, scale)
            keypoints.append(points)
            scores.append(score)
        return np.concatenate(keypoints, axis=0), np.concatenate(scores, axis=0)

    def preprocess(self, img: np.ndarray, box):
        center, scale = bbox_xyxy2cs(np.array(box), padding=1.25)
        crop, scale = top_down_affine(self.input_size, scale, center, img)
        return (crop - self.mean) / self.std, center, scale

    def postprocess(self, simcc_x: np.ndarray, simcc_y: np.ndarray, center: np.ndarray, scale: np.ndarray,
                    simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio
        keypoints = keypoints / self.input_size * scale
        keypoints = keypoints + center - scale / 2
        return keypoints, scores


def bbox_xyxy2cs(bbox: np.ndarray, padding: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    """Center (x, y) and scale (w, h) of an (x1, y1, x2, y2) box, the scale multiplied by `padding`."""
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _rotate_point(pt: np.ndarray, angle_rad: float) -> np.ndarray:
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The point that `a - b` rotated by 90 degrees anticlockwise around `b` reaches."""
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(center: np.ndarray, scale: np.ndarray, rot: float, output_size: Tuple[int, int],
                    shift: Tuple[float, float] = (0., 0.), inv: bool = False) -> np.ndarray:
    """The 2x3 affine matrix that maps the box area of the image onto `output_size`."""
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def top_down_affine(input_size: Tuple[int, int], bbox_scale: np.ndarray, bbox_center: np.ndarray,
                    img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The box area as a model input of `input_size` (w, h), and the box scale widened to that aspect ratio."""
    w, h = input_size
    warp_size = (int(w), int(h))

    aspect_ratio = w / h
    b_w, b_h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(b_w > b_h * aspect_ratio,
                          np.hstack([b_w, b_w / aspect_ratio]),
                          np.hstack([b_h * aspect_ratio, b_h]))

    warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale


def get_simcc_maximum(simcc_x: np.ndarray, simcc_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locations (N, K, 2) and scores (N, K) of the strongest SimCC responses; -1 where the score is not positive."""
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    vals = 0.5 * (max_val_x + max_val_y)
    locs[vals <= 0.] = -1

    return locs.reshape(N, K, 2), vals.reshape(N, K)
