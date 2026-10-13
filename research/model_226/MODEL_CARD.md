# 226-class ensemble

Classifies one isolated Turkish Sign Language sign from a clip of 32 RGB
frames. RTMW whole-body pose estimation gives 75 landmarks per frame, four
recurrent models read joint, bone, motion, geometry and hand features, and
their class probabilities are averaged with the weights 0.10 / 0.21 / 0.55 /
0.14. SignTurk uses it in the 226 mode of the live page.

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

The browser records one sign as 32 unmirrored frames. The server runs RTMW on
each frame as it arrives, with the person detector on every fourth frame, and
the ensemble once the clip is complete. The top
sign is offered for approval when its probability is at least 0.70 and at
least 0.12 above the second.

## Scope

- Isolated signs, one at a time.
- The figures above are offline results on AUTSL.
- The streams expect landmarks from the two models in the input pipeline.
  With RTMW at 384×288 and a COCO-trained YOLOX-m, the measured test top-1
  is 89.47%.
- RTMW is the heaviest step; a GPU keeps the live mode responsive.

## Files

`python tools/download_models.py --bundle 226` downloads the four Keras
streams and the two ONNX models (about 660 MB) from the
[`model-assets-v1`](https://github.com/unsalemregulener/SignTurk/releases/tag/model-assets-v1)
release and checks their SHA-256. YOLOX and RTMW are Apache-2.0 models from
OpenMMLab's MMPose; `model-assets.json` lists the original archives.
