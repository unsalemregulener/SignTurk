"""Measure the 226-class model on an AUTSL split with the code the server uses.

Each video is read with OpenCV and 32 evenly spaced frames are taken, as in the
model card. Every frame goes through YOLOX and RTMW (`Live226.landmarks`),
then the ensemble scores the clip.

    python tools/evaluate_226.py --videos /path/to/AUTSL/test --labels /path/to/AUTSL/test/ground_truth.csv

The folder needs `<sample>_color.mp4` files; the labels file has one
`<sample>,<AUTSL class id>` row per sample. Rows are appended to a CSV, so an
interrupted run continues where it stopped. On an NVIDIA GPU install
requirements-gpu.txt first; on the CPU a full split takes many hours.
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from live226 import Live226, prepare_frame  # noqa: E402
from signturk_runtime.features import build_feature_bundle, sample_indices  # noqa: E402

REPORTED = ROOT / "research" / "model_226" / "evaluation" / "four_stream_metrics.json"
FIELDS = ["sample", "label", "prediction", "correct", "top5", "confidence", "frames", "detect_ms", "pose_ms", "seconds"]


def read_labels(path: Path) -> list:
    with path.open(newline="", encoding="utf-8") as handle:
        return [(row[0].strip(), int(row[1])) for row in csv.reader(handle) if len(row) >= 2]


def read_video(path: Path) -> list:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def evaluate_clip(live: Live226, frames: list, detect_every: int):
    """Class probabilities for one video and the time spent on detection and pose."""
    clip, boxes = [], None
    detect_ms = pose_ms = 0.0
    for index in sample_indices(len(frames), live.frame_count):
        # The same box reuse as the live socket in backend.py
        found = boxes is not None and len(boxes) > 0
        reuse = boxes if found and len(clip) % detect_every else None
        landmarks, boxes = live.landmarks(prepare_frame(frames[int(index)]), reuse)
        clip.append(landmarks)
        detect_ms += live.extractor.last_detect_ms
        pose_ms += live.extractor.last_pose_ms
    probabilities = live.ensemble.predict(build_feature_bundle(np.stack(clip)))
    return probabilities, detect_ms / len(clip), pose_ms / len(clip)


def macro_f1(rows: list) -> float:
    true_positive, predicted, actual = Counter(), Counter(), Counter()
    for row in rows:
        label, prediction = int(row["label"]), int(row["prediction"])
        actual[label] += 1
        predicted[prediction] += 1
        if label == prediction:
            true_positive[label] += 1
    scores = []
    for label in actual:
        precision = true_positive[label] / predicted[label] if predicted[label] else 0.0
        recall = true_positive[label] / actual[label]
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def summarize(rows: list, names: dict, detect_every: int) -> dict:
    count = len(rows)
    top = {k: sum(int(row["label"]) in [int(x) for x in row["top5"].split()[:k]] for row in rows) / count
           for k in (1, 3, 5)}
    per_class = Counter(int(row["label"]) for row in rows)
    correct = Counter(int(row["label"]) for row in rows if row["correct"] == "1")
    weakest = sorted(per_class, key=lambda c: correct[c] / per_class[c])[:10]
    confusions = Counter((int(row["label"]), int(row["prediction"])) for row in rows if row["correct"] != "1")
    return {
        "samples": count,
        "detect_every": detect_every,
        "top_1_accuracy": top[1],
        "top_3_accuracy": top[3],
        "top_5_accuracy": top[5],
        "macro_f1": macro_f1(rows),
        "correct_top_1": sum(correct.values()),
        "avg_detect_ms": float(np.mean([float(row["detect_ms"]) for row in rows])),
        "avg_pose_ms": float(np.mean([float(row["pose_ms"]) for row in rows])),
        "avg_seconds_per_video": float(np.mean([float(row["seconds"]) for row in rows])),
        "weakest_classes": [{"class_id": c, "gloss": names.get(c, "?"), "accuracy": correct[c] / per_class[c],
                             "samples": per_class[c]} for c in weakest],
        "top_confusions": [{"label": names.get(a, "?"), "predicted": names.get(b, "?"), "count": n}
                           for (a, b), n in confusions.most_common(10)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, required=True, help="folder with <sample>_color.mp4 files")
    parser.add_argument("--labels", type=Path, required=True, help="CSV of <sample>,<AUTSL class id>")
    parser.add_argument("--out", type=Path, help="results CSV (default: results/ next to the videos folder)")
    parser.add_argument("--detect-every", type=int, default=1,
                        help="run the person detector on every Nth frame (1 as in the original evaluation)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--limit", type=int, help="evaluate an evenly spread subset of this many samples")
    args = parser.parse_args()

    labels = read_labels(args.labels)
    if args.limit and args.limit < len(labels):
        labels = [labels[int(i)] for i in np.linspace(0, len(labels) - 1, args.limit).round()]
    out = args.out or args.videos.parent / "results" / f"{args.videos.name}_detect{args.detect_every}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as handle:
            done = {row["sample"]: row for row in csv.DictReader(handle)}

    os.environ["SIGNTURK_226_DEVICE"] = args.device
    live = Live226()
    live.load()
    if not live.ready:
        print(live.message, file=sys.stderr)
        return 1
    names = {entry["autsl_id"]: entry["gloss"] for entry in live.vocabulary}
    print(f"226 model on {live.device}; {len(labels)} samples, {len(done)} already in {out}")

    missing = 0
    started = time.time()
    new_file = not out.exists() or out.stat().st_size == 0
    with out.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        todo = [(sample, label) for sample, label in labels if sample not in done]
        for number, (sample, label) in enumerate(todo, 1):
            video = args.videos / f"{sample}_color.mp4"
            frames = read_video(video) if video.exists() else []
            if not frames:
                missing += 1
                continue
            clip_started = time.time()
            probabilities, detect_ms, pose_ms = evaluate_clip(live, frames, args.detect_every)
            order = np.argsort(-probabilities)[:5]
            top5 = [live.vocabulary[int(i)]["autsl_id"] for i in order]
            row = {"sample": sample, "label": label, "prediction": top5[0], "correct": int(top5[0] == label),
                   "top5": " ".join(map(str, top5)), "confidence": round(float(probabilities[order[0]]), 4),
                   "frames": len(frames), "detect_ms": round(detect_ms, 1), "pose_ms": round(pose_ms, 1),
                   "seconds": round(time.time() - clip_started, 3)}
            writer.writerow(row)
            done[sample] = {key: str(value) for key, value in row.items()}
            if number % 100 == 0 or number == len(todo):
                handle.flush()
                elapsed = time.time() - started
                accuracy = np.mean([r["correct"] == "1" for r in done.values()])
                print(f"{number}/{len(todo)}  top-1 so far {accuracy:.2%}  "
                      f"{elapsed / number:.2f}s per video, about {(len(todo) - number) * elapsed / number / 60:.0f} min left")

    wanted = {sample for sample, _ in labels}
    rows = [row for sample, row in done.items() if sample in wanted]
    if not rows:
        print("no videos were found", file=sys.stderr)
        return 1
    summary = summarize(rows, names, args.detect_every)
    summary["missing_videos"] = missing
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    reported = json.loads(REPORTED.read_text(encoding="utf-8"))["metrics"]
    print(f"\n{'':14}{'measured':>10}{'reported':>10}")
    for key in ("top_1_accuracy", "top_3_accuracy", "top_5_accuracy", "macro_f1"):
        print(f"{key:14}{summary[key]:>10.4f}{reported[key]:>10.4f}")
    print(f"samples {summary['samples']}, missing videos {missing}, detection every {args.detect_every} frame(s), "
          f"detect {summary['avg_detect_ms']:.1f} ms, pose {summary['avg_pose_ms']:.1f} ms per frame")
    print("weakest:", ", ".join(f"{c['gloss']} {c['accuracy']:.0%}" for c in summary["weakest_classes"][:5]))
    print("confusions:", ", ".join(f"{c['label']}->{c['predicted']} x{c['count']}" for c in summary["top_confusions"][:5]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
