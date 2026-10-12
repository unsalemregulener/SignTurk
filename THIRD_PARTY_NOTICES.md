# Third-party notices

SignTurk's own code is released under the MIT License (see [LICENSE](LICENSE)).
The components below are included in this repository, loaded by it at run
time or installed with it. They keep their own licenses and terms.

## Included in this repository

| Component | Files | Terms |
| --- | --- | --- |
| three.js r128 | `frontend/three.min.js`, `frontend/GLTFLoader.js`, `frontend/OrbitControls.js` | MIT, Copyright 2010-2021 Three.js Authors |
| Mixamo character "Ch08" | `frontend/avatar.glb` (textures resized) | Adobe Mixamo terms of use, not the MIT License |
| Landmark sequences derived from AUTSL | `landmarks/*.json` | see [AUTSL](#autsl) |
| Model metadata derived from AUTSL | `demo_assets_179/`, `research/model_226/` | see [AUTSL](#autsl) |

The avatar is included only so that the demo runs. Check Adobe's Mixamo terms
before reusing the character file on its own.

## Model weights

These files are published as release assets and downloaded by
`tools/download_models.py`.

| Files | Origin | Terms |
| --- | --- | --- |
| `live-179.keras`, `226-*.keras` | trained by the project team on AUTSL | see [AUTSL](#autsl) |
| `rtmw-dw-x-l_simcc-cocktail14.onnx` | RTMW from OpenMMLab MMPose, copied from the Hugging Face mirror [Izymka/rtmw-dw-x-l_simcc-cocktail14](https://huggingface.co/Izymka/rtmw-dw-x-l_simcc-cocktail14) | Apache-2.0 |
| `yolox_m.onnx` | YOLOX-m from Megvii, copied from the Hugging Face mirror [memescreamer/yolox-onnx](https://huggingface.co/memescreamer/yolox-onnx) | Apache-2.0 |

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
| lucide | 1.21.0 | ISC |

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
| MediaPipe | Apache-2.0 |
| ONNX Runtime | MIT |
| rtmlib | Apache-2.0 |
| NumPy | BSD-3-Clause |
| OpenCV (opencv-python, opencv-contrib-python) | Apache-2.0 |
| SQLAlchemy | MIT |
| psycopg | LGPL-3.0 |
| passlib | BSD |
| python-dotenv | BSD-3-Clause |
| huggingface_hub | Apache-2.0 |

Each package's own distribution is authoritative for its license, including
the licenses of libraries bundled inside its wheels.
