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
been back down for 0.4 s.

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
| Signs cut out | 98.4% | 94.5% |
| Of those, read correctly | 95.1% | 92.8% |
| Cuts that match no sign, per 100 signs | 2.9 | 5.3 |
| Offered results that are right | 98.6% | 98.0% |
| Delay after the clip's end (median) | 0.17 s | 0.17 s |

Most missed signs ran into a neighbour: two signs with less than 0.4 s of
rest between them are read as one. The test signers also lower their hands to
a different height than they started from more often: 10.0% of the test
clips end more than 0.25 shoulder widths away, against 1.8% of the
validation clips.

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
