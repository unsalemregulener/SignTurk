"""Measure how the live page's recordings are read, on cached AUTSL landmarks.

    python tools/evaluate_226.py --videos AUTSL/val --labels AUTSL/validation/ground_truth.csv \\
        --detect-every 4 --cache-landmarks AUTSL/landmarks/val
    python tools/evaluate_recording.py --cache AUTSL/landmarks/val --labels AUTSL/validation/ground_truth.csv

Every AUTSL clip is turned into what the live page would record: its frames
at 12 per second, with the signer at rest for a while before and after the
sign (the clip's first and last frames, played back and forth). The table
compares the model's input taken three ways:

- autsl: 32 frames of the original 30 fps clip, as in the model card (only
  when the cache holds every frame, not with `--cache-fps 12`);
- whole: 32 frames spread over the whole recording;
- cut:   32 frames from the sign's span found by signturk_runtime/segment.py.

`--grid` tries other span settings on the recordings and prints one line
per setting; choose them on the validation split, then run the test split
once.
"""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from signturk_runtime import segment  # noqa: E402
from signturk_runtime.config import load_config, load_vocabulary  # noqa: E402
from signturk_runtime.ensemble import Ensemble  # noqa: E402
from signturk_runtime.features import build_feature_bundle, sample_indices  # noqa: E402

SOURCE_FPS = 30.0         # of the cached landmarks; cache.json in the cache folder can say otherwise
CAMERA_FPS = 12.0
FRAMES = 32
OLD_RECORDING = 32        # frames the live page recorded before
RECORDING = 48            # 4 seconds at 12 fps
REST_SECONDS = 0.1        # the start and end of an AUTSL clip that are surely at rest


def read_labels(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[0].strip(): int(row[1]) for row in csv.reader(handle) if len(row) >= 2}


def camera_frames(clip: np.ndarray, fps: float = CAMERA_FPS) -> np.ndarray:
    """The clip as a camera at `fps` would have seen it."""
    count = int(np.floor((len(clip) - 1) * fps / SOURCE_FPS)) + 1
    return clip[np.round(np.arange(count) * SOURCE_FPS / fps).astype(np.int64)]


def rest(clip: np.ndarray, count: int, at_start: bool) -> np.ndarray:
    """`count` frames of the signer at rest, from the clip's first or last frames played back and forth."""
    still_frames = max(2, int(round(REST_SECONDS * SOURCE_FPS)))
    still = clip[:still_frames] if at_start else clip[-still_frames:][::-1]
    cycle = np.concatenate([still, still[-2:0:-1]]) if len(still) > 2 else still
    order = np.resize(np.arange(len(cycle)), count)
    frames = cycle[order]
    return frames[::-1] if at_start else frames


def recording(clip: np.ndarray, lead: int, length: int) -> np.ndarray:
    """`length` camera frames: `lead` frames of rest, the sign, rest until the end (or cut short)."""
    camera = camera_frames(clip)
    parts = [rest(clip, lead, True), camera] if lead else [camera]
    frames = np.concatenate(parts)[:length]
    if len(frames) < length:
        frames = np.concatenate([frames, rest(clip, length - len(frames), False)])
    return frames


def whole(frames: np.ndarray) -> np.ndarray:
    return frames[sample_indices(len(frames), FRAMES)]


def cut(frames: np.ndarray, **options) -> np.ndarray:
    return segment.sign_clip(frames, CAMERA_FPS, FRAMES, **options)[0]


class Scorer:
    """The four streams applied to many clips at once."""

    def __init__(self):
        self.config = load_config()
        self.ensemble = Ensemble(self.config)
        self.ensemble.load()
        self.ids = np.array([entry["autsl_id"] for entry in load_vocabulary()])

    def probabilities(self, windows: list, batch: int = 512) -> np.ndarray:
        total = np.zeros((len(windows), int(self.config["num_classes"])), dtype=np.float64)
        for start in range(0, len(windows), batch):
            bundles = [build_feature_bundle(window) for window in windows[start:start + batch]]
            for name, spec in self.config["streams"].items():
                part = [getattr(b, "hand_inputs" if spec["inputs"] == "hand" else "skeleton_inputs") for b in bundles]
                inputs = [np.concatenate([b[i] for b in part]) for i in range(len(part[0]))]
                total[start:start + len(bundles)] += spec["weight"] * np.asarray(
                    self.ensemble.models[name].predict_on_batch(inputs))
        return total / total.sum(axis=1, keepdims=True)

    def score(self, windows: list, labels: np.ndarray) -> dict:
        probabilities = self.probabilities(windows)
        order = np.argsort(-probabilities, axis=1)
        top1 = self.ids[order[:, 0]]
        first = probabilities[np.arange(len(order)), order[:, 0]]
        second = probabilities[np.arange(len(order)), order[:, 1]]
        thresholds = self.config["thresholds"]
        confident = (first >= thresholds["top1"]) & (first - second >= thresholds["margin"])
        right = top1 == labels
        return {"top1": float(right.mean()),
                "top3": float(np.mean([label in self.ids[row[:3]] for label, row in zip(labels, order)])),
                "offered": float(confident.mean()),
                "offered_right": float((confident & right).sum() / max(confident.sum(), 1))}


def grid(scorer: Scorer, clips: list, truth: np.ndarray) -> int:
    """Top-1 of the cut recordings for each span setting; each distinct cut is scored once."""
    leads = (0, 12, 24)
    recordings = [(lead, index, recording(clip, lead, RECORDING)) for lead in leads for index, clip in enumerate(clips)]
    settings = list(itertools.product((0.3, 0.4, 0.5, 0.6), (0.15, 0.25, 0.4), (0.0, 0.1, 0.2, 0.3)))
    cuts, chosen = {}, {}
    for setting in settings:
        raised, away, margin = setting
        for number, (_, _, frames) in enumerate(recordings):
            span = segment.active_span(frames, CAMERA_FPS, raised=raised, away=away, margin_seconds=margin)
            key = (number, *(span or (0, len(frames) - 1)))
            chosen[setting, number] = cuts.setdefault(key, len(cuts))
    print(f"{len(cuts)} distinct cuts of {len(recordings)} recordings", flush=True)
    windows = [segment.resample(recordings[number][2][first:last + 1], FRAMES) for number, first, last in cuts]
    predicted = scorer.ids[scorer.probabilities(windows).argmax(axis=1)]
    for setting in settings:
        scores = []
        for lead in leads:
            picks = [(chosen[setting, number], index) for number, (l, index, _) in enumerate(recordings) if l == lead]
            scores.append(np.mean([predicted[cut_index] == truth[index] for cut_index, index in picks]))
        print(f"raised {setting[0]:.2f} away {setting[1]:.2f} margin {setting[2]:.1f}s  "
              + "  ".join(f"lead {lead / CAMERA_FPS:.0f}s {score:.2%}" for lead, score in zip(leads, scores))
              + f"  mean {np.mean(scores):.2%}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="folder of <sample>.npy landmarks")
    parser.add_argument("--labels", type=Path, required=True, help="CSV of <sample>,<AUTSL class id>")
    parser.add_argument("--limit", type=int, help="use an evenly spread subset of this many clips")
    parser.add_argument("--grid", action="store_true", help="try span settings instead of the table")
    parser.add_argument("--out", type=Path, help="write the table as JSON")
    args = parser.parse_args()

    global SOURCE_FPS
    info = args.cache / "cache.json"
    if info.exists():
        SOURCE_FPS = float(json.loads(info.read_text(encoding="utf-8"))["fps"])
    labels = read_labels(args.labels)
    samples = sorted(s for s in labels if (args.cache / f"{s}.npy").exists())
    if args.limit and args.limit < len(samples):
        samples = [samples[int(i)] for i in np.linspace(0, len(samples) - 1, args.limit).round()]
    if not samples:
        print("no cached clips found", file=sys.stderr)
        return 1
    clips = [np.load(args.cache / f"{s}.npy") for s in samples]
    truth = np.array([labels[s] for s in samples])
    scorer = Scorer()
    print(f"{len(samples)} clips from {args.cache}, {SOURCE_FPS:g} fps")

    if args.grid:
        return grid(scorer, clips, truth)

    rows = {}
    if SOURCE_FPS == 30:
        rows["autsl (30 fps clip, 32 frames)"] = scorer.score([whole(clip) for clip in clips], truth)
    for lead in (0, 4, 8):
        frames = [recording(clip, lead, OLD_RECORDING) for clip in clips]
        rows[f"2.7 s recording, sign after {lead / CAMERA_FPS:.1f} s, whole"] = scorer.score(
            [whole(f) for f in frames], truth)
    for lead in (0, 6, 12, 18, 24):
        frames = [recording(clip, lead, RECORDING) for clip in clips]
        rows[f"4 s recording, sign after {lead / CAMERA_FPS:.1f} s, whole"] = scorer.score(
            [whole(f) for f in frames], truth)
        rows[f"4 s recording, sign after {lead / CAMERA_FPS:.1f} s, cut"] = scorer.score(
            [cut(f) for f in frames], truth)
    camera = [camera_frames(clip) for clip in clips]
    rows["12 fps clip alone, resampled"] = scorer.score([segment.resample(f, FRAMES) for f in camera], truth)
    rows["12 fps clip alone, cut"] = scorer.score([cut(f) for f in camera], truth)

    width = max(len(name) for name in rows)
    print(f"{'':{width}}  {'top-1':>7}{'top-3':>8}{'offered':>9}{'right':>8}")
    for name, row in rows.items():
        print(f"{name:{width}}  {row['top1']:>7.2%}{row['top3']:>8.2%}{row['offered']:>9.1%}{row['offered_right']:>8.2%}")
    if args.out:
        args.out.write_text(json.dumps({"clips": len(samples), "rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
