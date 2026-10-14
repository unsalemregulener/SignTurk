"""Measure how the live page's recordings are read, on cached AUTSL landmarks.

    python tools/evaluate_226.py --videos AUTSL/val --labels AUTSL/validation/ground_truth.csv \\
        --detect-every 4 --cache-landmarks AUTSL/landmarks/val
    python tools/evaluate_recording.py --cache AUTSL/landmarks/val --labels AUTSL/validation/ground_truth.csv

Every AUTSL clip is turned into what the live page would record: its frames
at 12 per second, with the signer at rest for a while before and after the
sign (the clip's first and last frame held). Neighbouring frames are not
used for that rest: at 12 fps the second-to-last frame is often still on its
way down. The table compares the model's input taken three ways:

- autsl: 32 frames of the original 30 fps clip, as in the model card (only
  when the cache holds every frame, not with `--cache-fps 12`);
- whole: 32 frames spread over the whole recording;
- cut:   32 frames from the sign's span found by signturk_runtime/segment.py.

`--grid` tries other span settings on the recordings and prints one line
per setting; choose them on the validation split, then run the test split
once.

`--streams` measures the hands-free mode instead: clips of one signer are
joined into streams of five signs with 0.3-1.5 s of rest between them, and
signturk_runtime.segment.Segmenter reads each stream frame by frame. A found
sign counts when it covers at least half of a true sign and a true sign at
least half of it.

`--webcam` frames the clips like a laptop camera: the image ends some
shoulder widths below the shoulders, and the hands below that edge sit on
it with a low score. It reports the clip alone, a cut four-second recording
and the hands-free streams for each edge.
"""

import argparse
import csv
import itertools
import json
import sys
from collections import defaultdict
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


def read_labels(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[0].strip(): int(row[1]) for row in csv.reader(handle) if len(row) >= 2}


def camera_frames(clip: np.ndarray, fps: float = CAMERA_FPS) -> np.ndarray:
    """The clip as a camera at `fps` would have seen it."""
    count = int(np.floor((len(clip) - 1) * fps / SOURCE_FPS)) + 1
    return clip[np.round(np.arange(count) * SOURCE_FPS / fps).astype(np.int64)]


def rest(clip: np.ndarray, count: int, at_start: bool) -> np.ndarray:
    """`count` frames of the signer at rest: the clip's first or last frame held."""
    return np.repeat(clip[:1] if at_start else clip[-1:], count, axis=0)


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


CUT = {}                  # span settings for the table and the streams (--margin, --away)


def cut(frames: np.ndarray, **options) -> np.ndarray:
    return segment.sign_clip(frames, CAMERA_FPS, FRAMES, **{**CUT, **options})[0]


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


def signer_streams(clips: list, samples: list, per_stream: int = 5, seed: int = 0) -> list:
    """Streams of one signer's clips with rest between them: (frames, [(first, last, clip index)])."""
    rng = np.random.default_rng(seed)
    by_signer = defaultdict(list)
    for index, sample in enumerate(samples):
        by_signer[sample.split("_")[0]].append(index)
    streams = []
    for signer in sorted(by_signer):
        order = rng.permutation(by_signer[signer])
        for start in range(0, len(order), per_stream):
            parts, signs, length = [], [], 0
            for index in order[start:start + per_stream]:
                gap = rest(clips[index], int(rng.uniform(0.3, 1.5) * CAMERA_FPS), True)
                camera = camera_frames(clips[index])
                signs.append((length + len(gap), length + len(gap) + len(camera) - 1, int(index)))
                parts += [gap, camera]
                length += len(gap) + len(camera)
            parts.append(rest(clips[order[min(start + per_stream, len(order)) - 1]], int(CAMERA_FPS), False))
            streams.append((np.concatenate(parts), signs))
    return streams


def matches(found: tuple, true: tuple) -> bool:
    overlap = min(found[1], true[1]) - max(found[0], true[0]) + 1
    return overlap >= 0.5 * (true[1] - true[0] + 1) and overlap >= 0.5 * (found[1] - found[0] + 1)


def stream_scores(scorer: Scorer, streams: list, truth: np.ndarray, settle: float) -> dict:
    """How the Segmenter and the model read the streams with one settle time."""
    total = sum(len(signs) for _, signs in streams)
    windows, found, delays = [], [], []
    for number, (frames, _) in enumerate(streams):
        segmenter = segment.Segmenter(CAMERA_FPS, settle_seconds=settle, **CUT)
        for position, frame in enumerate(frames):
            cut_out = segmenter.push(frame)
            if cut_out is not None:
                windows.append(segment.resample(cut_out[0], FRAMES))
                found.append((number, cut_out[1], position))
    probabilities = scorer.probabilities(windows) if windows else np.zeros((0, len(scorer.ids)))
    order = np.argsort(-probabilities, axis=1)
    thresholds = scorer.config["thresholds"]
    hit, right, offered, offered_right = 0, 0, 0, 0
    used = set()
    for (number, span, position), row, probability in zip(found, order, probabilities):
        first, second = probability[row[0]], probability[row[1]]
        confident = first >= thresholds["top1"] and first - second >= thresholds["margin"]
        offered += confident
        for true_first, true_last, index in streams[number][1]:
            if (number, index) not in used and matches(span, (true_first, true_last)):
                used.add((number, index))
                hit += 1
                correct = scorer.ids[row[0]] == truth[index]
                right += correct
                offered_right += confident and correct
                delays.append((position - true_last) / CAMERA_FPS)
                break
    return {"found": hit / total, "extra": (len(found) - hit) / total, "right": right / max(hit, 1),
            "read": right / total, "offered": offered / max(len(found), 1),
            "offered_right": offered_right / max(offered, 1), "delay": float(np.median(delays)) if delays else 0.0}


STREAM_HEADER = f"{'found':>8}{'extra':>8}{'right':>8}{'read':>8}{'offered':>9}{'o.right':>9}{'delay':>8}"
STREAM_NOTE = ("found: true signs cut out; extra: cuts that match no sign, per true sign; right: of the found "
               "signs, top-1 correct; read: found and correct; offered: confident cuts; delay: median seconds "
               "from the end of the clip to the cut")


def stream_row(row: dict) -> str:
    return (f"{row['found']:>8.1%}{row['extra']:>8.1%}{row['right']:>8.1%}{row['read']:>8.1%}"
            f"{row['offered']:>9.1%}{row['offered_right']:>9.1%}{row['delay']:>7.2f}s")


def evaluate_streams(scorer: Scorer, clips: list, samples: list, truth: np.ndarray) -> int:
    streams = signer_streams(clips, samples)
    total = sum(len(signs) for _, signs in streams)
    print(f"{len(streams)} streams, {total} signs, {sum(len(f) for f, _ in streams) / CAMERA_FPS / 60:.0f} min")
    print(f"{'settle':>7}{STREAM_HEADER}")
    for settle in (0.25, 0.3, 0.4):
        print(f"{settle:>6.2f}s{stream_row(stream_scores(scorer, streams, truth, settle))}", flush=True)
    print(STREAM_NOTE)
    return 0


def framed(frames: np.ndarray, below: float) -> np.ndarray:
    """The frames as a camera sees them when its image ends `below` shoulder widths under the shoulders.

    Points lower than that edge sit on it with a low score, as RTMW places
    the parts of a person it cannot see.
    """
    out = np.array(frames, dtype=np.float32, copy=True)
    seen = (out[:, 11, 3] > 0) & (out[:, 12, 3] > 0)
    if not seen.any():
        return out
    shoulders_y = (out[seen, 11, 1] + out[seen, 12, 1]) / 2.0
    width = np.linalg.norm(out[seen, 11, :2] - out[seen, 12, :2], axis=-1)
    edge = float(np.median(shoulders_y) + below * np.median(width))
    gone = (out[..., 3] > 0) & (out[..., 1] > edge)
    out[..., 1] = np.where(gone, edge, out[..., 1])
    out[..., 3] = np.where(gone, np.minimum(out[..., 3], 0.3), out[..., 3])
    return out


def stretched(frames: np.ndarray, factor: float) -> np.ndarray:
    """y scaled by `factor` around the middle: a wide frame read with x and y on different scales."""
    out = np.array(frames, dtype=np.float32, copy=True)
    out[..., 1] = np.where(out[..., 3] > 0, 0.5 + (out[..., 1] - 0.5) * factor, 0.0)
    return out


def webcam(scorer: Scorer, clips: list, samples: list, truth: np.ndarray) -> int:
    """The clips framed like a laptop camera that cuts off the space below the shoulders."""
    camera = [camera_frames(clip) for clip in clips]
    recordings = [recording(clip, int(CAMERA_FPS), RECORDING) for clip in clips]
    print(f"{'top-1':<44}{'clip alone':>11}{'4 s, cut':>10}")
    for below in (None, 2.0, 1.5, 1.2, 1.0, 0.5):
        name = "full frame" if below is None else f"frame ends {below} shoulder widths lower"
        alone = camera if below is None else [framed(f, below) for f in camera]
        recorded = recordings if below is None else [framed(f, below) for f in recordings]
        print(f"{name:<44}{scorer.score([segment.resample(f, FRAMES) for f in alone], truth)['top1']:>11.2%}"
              f"{scorer.score([cut(f) for f in recorded], truth)['top1']:>10.2%}", flush=True)
    old = scorer.score([segment.resample(stretched(f, 1.6), FRAMES) for f in camera], truth)["top1"]
    print(f"{'640x400 read with x and y on one scale each':<44}{old:>11.2%}")
    streams = signer_streams(clips, samples)
    print(f"\nhands-free, settle 0.3 s{'':<20}{STREAM_HEADER}")
    for below in (None, 1.2, 0.5):
        name = "full frame" if below is None else f"frame ends {below} shoulder widths lower"
        moved = streams if below is None else [(framed(frames, below), signs) for frames, signs in streams]
        print(f"{name:<44}{stream_row(stream_scores(scorer, moved, truth, 0.3))}", flush=True)
    print(STREAM_NOTE)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True, help="folder of <sample>.npy landmarks")
    parser.add_argument("--labels", type=Path, required=True, help="CSV of <sample>,<AUTSL class id>")
    parser.add_argument("--limit", type=int, help="use an evenly spread subset of this many clips")
    parser.add_argument("--grid", action="store_true", help="try span settings instead of the table")
    parser.add_argument("--streams", action="store_true", help="measure the hands-free mode instead of the table")
    parser.add_argument("--webcam", action="store_true", help="measure laptop-camera framing instead of the table")
    parser.add_argument("--margin", type=float, help=f"seconds of rest kept around a cut (default {segment.MARGIN_SECONDS})")
    parser.add_argument("--away", type=float, help=f"band above rest that still counts as away (default {segment.AWAY})")
    parser.add_argument("--out", type=Path, help="write the table as JSON")
    args = parser.parse_args()
    if args.margin is not None:
        CUT["margin_seconds"] = args.margin
    if args.away is not None:
        CUT["away"] = args.away

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
    if args.streams:
        return evaluate_streams(scorer, clips, samples, truth)
    if args.webcam:
        return webcam(scorer, clips, samples, truth)

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
