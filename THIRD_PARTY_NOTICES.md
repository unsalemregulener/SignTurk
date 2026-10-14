# Third-party notices

SignTurk's own code is released under the MIT License (see [LICENSE](LICENSE)).
The components below are included in this repository, loaded by it at run
time or installed with it. They keep their own licenses and terms.

## Included in this repository

| Component | Files | Terms |
| --- | --- | --- |
| three.js r128 | `frontend/three.min.js`, `frontend/GLTFLoader.js`, `frontend/OrbitControls.js` | MIT, Copyright 2010-2021 Three.js Authors |
| Mixamo character "Ch08" | `frontend/avatar.glb` (textures resized) | Adobe Mixamo terms of use, not the MIT License |
| YOLOX and RTMPose code adapted from [rtmlib](https://github.com/Tau-J/rtmlib) 0.0.16 | `signturk_runtime/rtm.py` | Apache-2.0, see [LICENSES/rtmlib-Apache-2.0.txt](LICENSES/rtmlib-Apache-2.0.txt); the changes are listed at the top of the file |
| Plus Jakarta Sans | `frontend/fonts/plus-jakarta-sans-*.woff2` | SIL Open Font License 1.1, see [frontend/fonts/OFL-PlusJakartaSans.txt](frontend/fonts/OFL-PlusJakartaSans.txt) |
| Lexend | `frontend/fonts/lexend-*.woff2` | SIL Open Font License 1.1, see [frontend/fonts/OFL-Lexend.txt](frontend/fonts/OFL-Lexend.txt) |
| Phosphor Icons 2.1.1 (regular and duotone paths) | `frontend/app/js/icons.js` | MIT, Copyright (c) 2023 Phosphor Icons |
| Landmark sequences derived from AUTSL | `landmarks/*.json` | see [AUTSL](#autsl) |
| Model metadata derived from AUTSL | `research/model_226/` (vocabulary, English names, evaluation summary) | see [AUTSL](#autsl) |

The avatar is included only so that the demo runs. Check Adobe's Mixamo terms
before reusing the character file on its own.

## Model weights

These files are published as release assets and downloaded by
`tools/download_models.py`.

| Files | Origin | Terms |
| --- | --- | --- |
| `226-*.keras` | trained by the project team on AUTSL | see [AUTSL](#autsl) |
| `rtmw-dw-x-l_simcc-cocktail14_256x192.onnx` | RTMW-x-l (256×192) from OpenMMLab MMPose: `end2end.onnx` from [rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip](https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.zip) | Apache-2.0 |
| `yolox_m_humanart.onnx` | YOLOX-m person detector trained on Human-Art, from OpenMMLab MMPose: `end2end.onnx` from [yolox_m_8xb8-300e_humanart-c2c7a14a.zip](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip) | Apache-2.0 |

## AUTSL

The models and the landmark sequences were derived from the AUTSL dataset:

> O. M. Sincan and H. Y. Keles, "AUTSL: A Large Scale Multi-Modal Turkish Sign
> Language Dataset and Baseline Methods," IEEE Access, 2020.

The AUTSL videos are not included in this repository. Use of the dataset, and
of data derived from it, is governed by the dataset's own terms; check them
before redistributing the derived files or using them commercially.

## Loaded from a CDN by the web interface

| Component | Version | License |
| --- | --- | --- |
| React, ReactDOM | 18.3.1 | MIT |
| @babel/standalone | 7.29.9 | MIT |

## Python packages

Installed from PyPI by `requirements.txt`, and therefore also present in the
Docker image.

| Package | License |
| --- | --- |
| FastAPI | MIT |
| Starlette | BSD-3-Clause |
| Uvicorn | BSD-3-Clause |
| Pydantic | MIT |
| python-multipart | Apache-2.0 |
| websockets | BSD-3-Clause |
| TensorFlow | Apache-2.0 |
| Keras | Apache-2.0 |
| ONNX Runtime | MIT |
| NumPy | BSD-3-Clause |
| OpenCV (opencv-python-headless) | Apache-2.0 |
| SQLAlchemy | MIT |
| psycopg | LGPL-3.0 |
| passlib | BSD |
| python-dotenv | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |

Each package's own distribution is authoritative for its license, including
the licenses of libraries bundled inside its wheels. `tools/extract_landmarks.py`
also uses MediaPipe (Apache-2.0), which is installed separately.
