"""Extract the avatar landmark files from the AUTSL training videos.

For every sign it tries the first few training videos, keeps the one where
MediaPipe found hands in the most frames, resamples it to 30 frames and writes
`<output>/<gloss>.json` with 9 upper-body pose points and 21 points per hand.

    python tools/extract_landmarks.py --dataset /path/to/AUTSL --output landmarks

The dataset folder needs `SignList_ClassId_TR_EN.csv`, `train_labels.csv` and
`train/<sample>_color.mp4`. The output is not smoothed; the server smooths a
file when it serves it unless the file is marked `"smoothed": true`.

`left_hand` and `right_hand` keep MediaPipe's handedness labels, which assume a
mirrored image. On these unmirrored videos `left_hand` is the signer's right
hand; `live_pipeline.hands_to_landmarks` converts to the model's layout.
"""

import argparse
import csv
import json
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

TARGET_FRAMES = 30   # frames per output file
MIN_FRAMES = 8       # shorter videos are skipped
CANDIDATES = 3       # training videos tried per sign

# Upper-body pose points: nose, shoulders, elbows, wrists, hips
POSE_UPPER_BODY = [0, 11, 12, 13, 14, 15, 16, 23, 24]

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
HAND_MODEL_PATH = Path("hand_landmarker.task")
POSE_MODEL_PATH = Path("pose_landmarker.task")


def ensure_models():
    for url, path in ((HAND_MODEL_URL, HAND_MODEL_PATH), (POSE_MODEL_URL, POSE_MODEL_PATH)):
        if not path.exists():
            print(f"downloading {path}")
            urllib.request.urlretrieve(url, path)


def resample_frames(frames, target=TARGET_FRAMES):
    """Linear interpolation to `target` frames; a missing hand takes its neighbour."""
    count = len(frames)
    if count in (0, target):
        return frames
    resampled = []
    for position in np.linspace(0, count - 1, target):
        lo = int(position)
        hi = min(lo + 1, count - 1)
        t = position - lo
        frame = {}
        for hand in ("left_hand", "right_hand"):
            a, b = frames[lo].get(hand), frames[hi].get(hand)
            if a is None or b is None:
                frame[hand] = a if b is None else b
            else:
                frame[hand] = (np.array(a) * (1 - t) + np.array(b) * t).tolist()
        pose_a, pose_b = frames[lo].get("pose", {}), frames[hi].get("pose", {})
        pose = {}
        for key in pose_a.keys() | pose_b.keys():
            a, b = pose_a.get(key), pose_b.get(key)
            pose[key] = a if b is None else b if a is None else (np.array(a) * (1 - t) + np.array(b) * t).tolist()
        frame["pose"] = pose
        resampled.append(frame)
    return resampled


def process_video(video_path):
    """Pose and hand landmarks for every frame, plus the number of frames with a hand."""
    pose_detector = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    hand_detector = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    capture = cv2.VideoCapture(str(video_path))
    frames, with_hands, index = [], 0, 0
    try:
        if not capture.isOpened():
            return None, 0
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(index * 1000 / fps)

            pose = {}
            pose_result = pose_detector.detect_for_video(image, timestamp_ms)
            if pose_result.pose_landmarks:
                points = pose_result.pose_landmarks[0]
                pose = {str(i): [points[i].x, points[i].y, points[i].z] for i in POSE_UPPER_BODY}

            hands = {"left_hand": None, "right_hand": None}
            hand_result = hand_detector.detect_for_video(image, timestamp_ms)
            if hand_result.hand_landmarks and hand_result.handedness:
                with_hands += 1
                for points, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                    key = "left_hand" if handedness[0].category_name == "Left" else "right_hand"
                    hands[key] = [[p.x, p.y, p.z] for p in points]

            frames.append({"pose": pose, **hands})
            index += 1
    finally:
        capture.release()
        pose_detector.close()
        hand_detector.close()
    return frames, with_hands


def best_video(samples, dataset):
    """The candidate video with the most frames that contain a hand."""
    best = (None, -1, None)
    for sample in samples[:CANDIDATES]:
        path = dataset / "train" / f"{sample}_color.mp4"
        if not path.exists():
            continue
        frames, score = process_video(path)
        if frames is not None and len(frames) >= MIN_FRAMES and score > best[1]:
            best = (frames, score, path.name)
    return best


def extract(task):
    class_id, gloss, samples, dataset, output = task
    try:
        frames, score, source = best_video(samples, Path(dataset))
        if frames is None:
            return gloss, False, "no usable video"
        frames = resample_frames(frames)
        document = {
            "class_id": class_id,
            "word_tr": gloss,
            "frame_count": len(frames),
            "hand_detected": score,
            "pose_indices": POSE_UPPER_BODY,
            "source_video": source,
            "frames": frames,
        }
        (Path(output) / f"{gloss}.json").write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
        return gloss, True, f"{source}, hands in {score} frames"
    except Exception as exc:
        return gloss, False, str(exc)


def read_signs(dataset):
    with open(dataset / "SignList_ClassId_TR_EN.csv", encoding="utf-8") as handle:
        signs = {int(row["ClassId"]): row["TR"] for row in csv.DictReader(handle)}
    samples = {}
    with open(dataset / "train_labels.csv", encoding="utf-8") as handle:
        for sample, class_id in csv.reader(handle):
            samples.setdefault(int(class_id), []).append(sample)
    return signs, samples


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="AUTSL folder")
    parser.add_argument("--output", type=Path, default=Path("landmarks"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    signs, samples = read_signs(args.dataset)
    tasks = [(class_id, gloss, samples[class_id], str(args.dataset), str(args.output))
             for class_id, gloss in signs.items() if samples.get(class_id)]
    ensure_models()

    started = time.time()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, task) for task in tasks]
        for number, future in enumerate(as_completed(futures), 1):
            gloss, ok, message = future.result()
            print(f"[{number:3}/{len(tasks)}] {'ok  ' if ok else 'FAIL'} {gloss:<22} {message}")
            done += ok
            failed += not ok
    print(f"\n{done} written, {failed} failed in {time.time() - started:.0f}s -> {args.output.resolve()}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
