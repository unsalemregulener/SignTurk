# 226-class ensemble

Classifies one isolated Turkish Sign Language sign from a clip of 32 RGB
frames. RTMW whole-body pose estimation gives 75 landmarks per frame, four
recurrent models read joint, bone, motion, geometry and hand features, and
their class probabilities are averaged with the weights 0.10 / 0.21 / 0.55 /
0.14. SignTurk's live page uses it.

## Evaluation

| AUTSL test split (official, signer-independent) | Samples | Top-1 | Top-3 | Top-5 | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reported with the trained streams | 3,742 | 94.17% | 98.88% | 99.49% | 0.940 |
| Measured with [`tools/evaluate_226.py`](../../tools/evaluate_226.py) | 3,742 | 94.39% | 98.64% | 99.47% | 0.942 |

The stream weights were chosen on the validation split and fixed before the
test run. The measurement runs every test video through the server's code:
32 evenly spaced frames, with the person detector on each of them. With the
detector on every fourth frame, as in the live mode, a 400-video subset scores
93.00% top-1 against 92.75%.

In the measured run the weakest classes are `seker` (5 of 11), `agir` (8 of 17)
and `psikoloji` (9 of 17). The most frequent confusion is `agir` read as
`hafif` (9 times), followed by `oruc` read as `okul` (5).

[`evaluation/`](evaluation/) keeps the summary of the reported evaluation:

![Per-class test accuracy and the most frequent confusions](evaluation/four_stream_error_summary.png)

There the weakest class is `hasta` (AUTSL 82) at 45.5%, and the most frequent
confusion is `siz` read as `uzgun` (9 times), followed by `eczane` read as
`devlet` (5). [`four_stream_top_confusions.csv`](evaluation/four_stream_top_confusions.csv)
lists the top pairs.

## Input pipeline

1. 32 frames evenly spaced over the clip; a shorter clip repeats its last frame.
2. YOLOX-m (trained on Human-Art) finds the person and RTMW (rtmw-dw-x-l,
   256×192) returns the 133 COCO-WholeBody keypoints. These are the models
   rtmlib's `Wholebody` uses by default, and the streams were trained on
   landmarks from them.
3. The 17 body points are placed in MediaPipe's 33 pose slots, followed by 21
   points for each hand: 75 × (x, y, 0, score), normalised to the image size.
   Points scoring below 0.05 stay zero. Left and right are the signer's own.
4. Coordinates are centred between the shoulders and divided by the shoulder
   width. Per frame the streams get the joints (300 values), bones (144),
   their frame-to-frame motion, 39 geometry values (wrist offsets, hand speed
   and distance, visibility masks, finger angles) and the 168 hand values.

The code is in [`signturk_runtime/`](../../signturk_runtime/), the stream list
and weights in [`config.json`](config.json), the vocabulary in
[`labels/`](labels/).

## Use in SignTurk

The browser records four seconds as 48 unmirrored frames at about 12 fps. The
server runs RTMW on each frame as it arrives, with the person detector on
every fourth frame. When the recording is complete,
[`segment.py`](../../signturk_runtime/segment.py) cuts the sign out of it like
an AUTSL clip, from where the higher wrist starts to rise from its rest height
to where it is back down, and resamples the cut to the 32 frames the ensemble
reads. The top sign is offered for approval when its probability is
at least 0.70 and at least 0.12 above the second. In the hands-free mode the
frames keep coming, and a sign is cut out the same way once the hands have
been back down for 0.3 s.

Landmarks are divided by the longer side of the frame, with the shorter side
centred, so x and y keep one scale as in AUTSL's square videos. Before this, a
640×400 webcam frame was read 1.6 times taller than it is.

## Recordings and the hands-free mode

[`tools/evaluate_recording.py`](../../tools/evaluate_recording.py) turns AUTSL
clips into what the live page sees: camera frames at 12 fps, with the signer
at rest before and after the sign (the clip's first and last frame held). The
test landmarks come from the 12 fps frames with the detector on every fourth
frame, as in the live mode; the validation landmarks from every 30 fps frame.
The settings were chosen on 1,536 of the 4,418 validation clips; the test
split was then run on all 3,742.

| Top-1 | Validation | Test |
| --- | ---: | ---: |
| The clip alone, 32 frames of the 30 fps video | 95.51% | 94.39%¹ |
| The clip alone at 12 fps, resampled to 32 frames | 95.51% | 93.80% |
| The clip alone at 12 fps, cut | 95.51% | 93.16% |
| The earlier 2.7 s recording, sign starting within 0.7 s | 94.40–94.66% | 89.87–91.15% |
| 4 s recording read whole, sign starting within 2 s | 88.93–89.84% | 76.27–78.81% |
| 4 s recording cut, sign starting within 2 s | 95.31–95.44% | 92.28–92.94% |

¹ `tools/evaluate_226.py`, with the detector on each of the 32 frames.

The model reads long stretches of rest badly, and more so for the test
signers. The cut removes them, and in a four-second recording it brings the
result close to the clip alone. On a clip with no rest around it, the cut
costs nothing on the validation split and 0.6 points on the test split.

The hands-free mode was measured on streams of five signs by one signer, with
0.3–1.5 s of rest between them. A sign counts as found when the cut covers at
least half of it and it covers at least half of the cut.

| Hands-free | Validation | Test |
| --- | ---: | ---: |
| Signs cut out | 98.9% | 95.4% |
| Of those, read correctly | 95.3% | 92.7% |
| Cuts that match no sign, per 100 signs | 0.7 | 2.5 |
| Offered results that are right | 98.9% | 97.9% |
| Delay after the clip's end (median) | 0.00 s | 0.08 s |

The Segmenter waits 0.3 s of rest before it cuts: on the validation split
0.25, 0.3 and 0.4 s read 94.6%, 94.2% and 93.6% of the signs, and 0.3 s
leaves one more frame for a webcam's wobble. It drops a cut with less than
0.25 s of raised hands or shorter than 0.5 s, which brought the cuts that
match no sign from 2.9 to 0.7 per 100 on the validation split.

Most missed signs ran into a neighbour: two signs with less than 0.3 s of
rest between them are read as one. The test signers also lower their hands to
a different height than they started from more often: 10.0% of the test
clips end more than 0.25 shoulder widths away, against 1.8% of the
validation clips.

### Framing

AUTSL's frame reaches about 2.4 shoulder widths below the shoulder line, so
the hands are in view at rest and in front of the chest. A laptop camera close
to the signer cuts that space off. `tools/evaluate_recording.py --webcam`
moves the bottom edge of the frame up: the hands below it sit on the edge
with a low score, as RTMW places what it cannot see.

| Test split, frame ends below the shoulders | Clip alone | 4 s recording, cut | Hands-free, signs read |
| --- | ---: | ---: | ---: |
| Full frame | 93.80% | 92.94% | 88.5% |
| 2.0 shoulder widths | 93.53% | 92.94% | |
| 1.5 shoulder widths | 92.33% | 91.82% | |
| 1.2 shoulder widths | 91.29% | 90.65% | 86.6% |
| 1.0 shoulder widths | 90.43% | 88.94% | |
| 0.5 shoulder widths | 80.55% | 64.99% | 43.7% |

The live page checks this on the camera's own frames
([`pose_quality.py`](../../signturk_runtime/pose_quality.py)) and asks the
signer to sit back or tilt the camera while less than 1.2 shoulder widths
are in view. On a close webcam the hands at rest sit on the image's edge about
0.6 shoulder widths below the shoulders, so the rest height counts from 0.45:
with the frame ending at 0.5, that lifts the cut recordings on the validation
split from 46.6% to 68.0% and changes nothing in a fuller frame. The same
clips read as 640×400 frames on two scales score 89.90% instead of 93.80%.

The first test run showed two problems, and both splits were measured again
after fixing them. `AWAY` was widened from 0.25 to 0.4 for signers whose rest
height changes; on the validation split that raises the share of signs cut
out from 97.1% to 98.4% and leaves the recording results unchanged. And the
simulated rest, which played a clip's first or last frames back and forth,
moved at 12 fps, where a neighbouring frame is often already on its way; it
now holds a single frame.

## Scope

- Isolated signs, one at a time; in the hands-free mode the hands have to
  come back down between signs.
- The camera has to see the space below the shoulders; see [Framing](#framing).
- The figures above are offline results on AUTSL. Recordings and streams are
  built from AUTSL clips, whose rest is a held frame rather than a person
  waiting in front of a webcam.
- The streams expect landmarks from the two models in the input pipeline.
  With RTMW at 384×288 and a COCO-trained YOLOX-m, the measured test top-1
  is 89.47%.
- RTMW is the heaviest step; a GPU keeps the live mode responsive.

## Files

`python tools/download_models.py` downloads the four Keras
streams and the two ONNX models (about 660 MB) from the
[`model-assets-v1`](https://github.com/unsalemregulener/SignTurk/releases/tag/model-assets-v1)
release and checks their SHA-256. YOLOX and RTMW are Apache-2.0 models from
OpenMMLab's MMPose; `model-assets.json` lists the original archives.
