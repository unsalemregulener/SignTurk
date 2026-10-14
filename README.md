# SignTurk

Real-time Turkish Sign Language (TİD) recognition in the browser. A webcam
stream is classified sign by sign, approved words are turned into a Turkish
sentence, and a 3D avatar can play signs back. The interface and API messages
are in Turkish.

[![Live recognition of the sign "çay" (tea)](docs/images/cay-preview.gif)](docs/videos/cay.mp4)

- **Two recognition modes:** a fast 179-sign model that watches the camera
  stream continuously (MediaPipe hand landmarks), and a 226-sign ensemble that
  reads one recorded sign at a time (RTMW whole-body landmarks).
- **Approval step:** a prediction enters the sentence only after the user
  accepts it, so a wrong guess can be rejected first.
- **Turkish sentence engine:** rule-based morphology turns glosses such as
  `ben çay içmek` into `Ben çay içiyorum.`
- **3D avatar:** 226 recorded landmark sequences replayed on a rigged
  three.js character, with captions, slow mode and loop.
- **Accounts:** token login, per-user history and settings, admin pages for
  users, logs and model status.
- **Storage:** SQLite with no setup, or PostgreSQL through `DATABASE_URL`.

<img src="docs/images/avatar.jpg" alt="3D avatar signing hoşça kal" width="560">

## Live recognition

The **Canlı Çeviri** page switches between the two modes.

### Fast mode: 179 signs

1. The browser sends a JPEG frame at most every 100 ms and waits for the
   answer before sending the next one.
2. The server runs MediaPipe Hands on the frame: 2 hands × 21 points × xyz =
   126 values, with the hand MediaPipe calls "Left" first, as in the training
   data. MediaPipe names hands as if the image were mirrored, so on these
   unmirrored frames that is the signer's right hand. Frames without a hand
   are skipped; after 8 in a row the window is reset.
3. Frames with a hand are collected in a **32-frame capture window**, about
   3 seconds of signing at the full frame rate.
4. The window is resampled to **16 evenly spaced frames**, keeping the first
   and the last, because the model was trained on 16-frame sequences.
5. Each frame becomes 156 features: coordinates relative to the wrist and
   scaled by hand size (126) plus 30 finger joint angles, z-score normalised
   with the training statistics.
6. The model scores 179 signs. A prediction at or above 0.40 confidence becomes
   the approval candidate. After every prediction the window starts empty
   again.

The 32-frame window lets a whole sign fit into one prediction; it is resampled
to the 16 frames the model was trained on.

### Full mode: 226 signs

1. The user presses **İşareti kaydet**. After a 3-2-1 countdown the browser
   captures 32 unmirrored frames at about 12 fps (2.7 seconds) and streams
   them to the server as it records.
2. For each frame the server runs RTMW whole-body pose estimation inside the
   person box of a YOLOX detector, which runs on every fourth frame. Of the 133
   keypoints it keeps 17 body points and 21 points per hand: 75 landmarks with
   the signer's own left and right.
3. When the clip is complete the landmarks are centred between the shoulders,
   scaled by the shoulder width and turned into joint, bone, motion, geometry
   and hand features.
4. Four Keras models score the 226 signs and their probabilities are averaged
   with the weights 0.10, 0.21, 0.55 and 0.14.
5. The page shows the top three. The first becomes the approval candidate when
   its probability is at least 0.70 and 0.12 above the second, and any of the
   three can be approved. Barely moving hands or a badly framed signer add a
   hint next to the result.

In both modes approved words are saved to the user's history and the sentence
engine rebuilds the sentence from all approved words.

## Models

| | Fast mode | Full mode |
| --- | --- | --- |
| Vocabulary | 179 of the 226 signs in [AUTSL](#data) | all 226 signs |
| Landmarks | MediaPipe Hands, 2 × 21 points | RTMW-x-l whole body (256×192), 17 body + 2 × 21 hand points |
| Model input | 16 frames × 156 features | 32 frames; joint, bone, motion, geometry and hand streams |
| Architecture | 2 × bidirectional LSTM (256 units) → additive temporal attention → dense 512 → dense 256 | four recurrent streams, weighted average ([model card](research/model_226/MODEL_CARD.md)) |
| Accuracy | 85.65% top-1, 93.96% top-3 on the AUTSL validation split (3,512 samples) | 94.39% top-1, 98.64% top-3 on the AUTSL test split (3,742 samples) |
| Download | 63 MB | about 660 MB, including the RTMW and YOLOX ONNX models |

The fast model covers the 179 signs that were recognised most reliably during
development. Both accuracy figures are offline results on AUTSL. The full-mode
figure was measured with `tools/evaluate_226.py`, which runs the test videos
through the server's code; the [model card](research/model_226/MODEL_CARD.md)
has the details.

Weights are published in the
[`model-assets-v1`](https://github.com/unsalemregulener/SignTurk/releases/tag/model-assets-v1)
release and verified against `model-assets.json`.

## Quick start

Python 3.11 is used in the Docker image and in CI.

```bash
git clone https://github.com/unsalemregulener/SignTurk.git
cd SignTurk
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/download_models.py --bundle all   # or leave out --bundle for the fast mode only
cp .env.example .env               # set SIGNTURK_ADMIN_EMAIL and SIGNTURK_ADMIN_PASSWORD
uvicorn backend:app --port 8000
```

Open <http://127.0.0.1:8000>, register or sign in with the admin account, and
allow camera access on the **Canlı Çeviri** page. The full mode becomes
available once its models have loaded in the background; the admin
**Model Durumu** page shows its state.

With Docker (both modes; `--build-arg MODELS=live` builds without the full mode):

```bash
docker compose up --build
```

The compose file keeps the SQLite database in the `signturk-data` volume
unless `.env` sets `DATABASE_URL`.

### GPU

The full mode runs YOLOX and RTMW through ONNX Runtime. With an NVIDIA GPU,
replace the CPU build with the CUDA one; pip brings CUDA and cuDNN along:

```bash
pip uninstall -y onnxruntime
pip install -r requirements-gpu.txt
```

`/api/health` then shows `"device": "cuda"` under `model_226`. If CUDA cannot
be loaded, the server logs why and stays on the CPU. Installing
`requirements.txt` again brings the CPU build back; run the two commands
again after that.

## Configuration

Settings are read from the environment or from a `.env` file
(see [`.env.example`](.env.example)).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | empty: `local_dev.db` (SQLite) | e.g. `postgresql+psycopg://user:password@host:5432/db` |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | browser origins allowed by CORS |
| `SIGNTURK_ADMIN_EMAIL`, `SIGNTURK_ADMIN_PASSWORD` | empty | first administrator, created at startup; the password needs at least 8 characters |
| `SIGNTURK_ADMIN_NAME` | `SignTurk Admin` | display name of that account |
| `SIGNTURK_TOKEN_HOURS` | `12` | how long a sign-in stays valid |
| `SIGNTURK_ENABLE_226` | `1` | `0` skips loading the full mode |
| `SIGNTURK_226_DETECT_EVERY` | `4` | run the person detector on every Nth frame of a clip and reuse its box in between |
| `SIGNTURK_226_DEVICE` | `auto` | `cuda` or `cpu` for the full mode; `auto` uses CUDA when ONNX Runtime has it |
| `HF_TOKEN`, `SIGNAI_*` | off | optional ML layer for the sentence engine, see [`text_processing`](text_processing/README.md) |

There are no built-in accounts. Without the admin variables the application
runs with registered users only.

## API

Authenticated routes take `Authorization: Bearer <token>`; the token comes
from `/api/login`. Browsers cannot set headers on WebSockets, so the live
sockets expect it as their first message, `{"action": "auth", "token": "…"}`,
which keeps it out of URLs and access logs.

| Route | Access | Purpose |
| --- | --- | --- |
| `GET /` | public | web interface |
| `GET /avatar3d` | public | avatar page, embedded by the interface |
| `GET /api/health` | public | models, MediaPipe, database, window sizes and the state of the full mode |
| `GET /signs`, `GET /landmark/{word}` | public | avatar vocabulary and landmark sequences |
| `GET /api/dictionary`, `GET /api/dictionary/{autsl_id}` | public | the 226 signs: Turkish spelling, English (for the 179), which mode knows them |
| `POST /api/register`, `POST /api/login`, `POST /api/logout` | public | accounts |
| `WS /api/predict/live` | user | fast mode; also accepts 126 landmark values per message instead of a frame |
| `WS /api/predict/live226` | user | full mode: `{"action": "start"}`, 32 × `{"image": data_url}`, `{"action": "end"}` |
| `POST /api/predict/sequence` | user | classify 16 or 32 frames of 126 landmark values (MediaPipe's "Left" hand first) |
| `GET /api/history`, `GET /api/me/summary` | user | approved words of the current user |
| `GET /api/settings`, `POST /api/settings` | user | speech and avatar speed |
| `GET /api/model/status` | user | model summary |
| `/api/text/*` | user | sentence engine: `correct`, `correct/async`, `jobs/{id}`, `models`, `health`, `ping`, `audio/{file}` |
| `GET /api/admin/overview`, `users`, `logs`, `model` | admin | administration |
| `GET /api/debug/model-check` | admin | loading state of every fast-mode asset and its config |
| `DELETE /api/text/cache`, `DELETE /api/text/cache/{model_key}` | admin | clear the sentence engine cache |

Both live sockets accept `approve` (optionally with a `word`), `reject`,
`undo` and `reset_sentence` actions and answer with `sentence_update`.

## Project layout

```
backend.py             FastAPI app: accounts, live WebSockets, history, admin
live_pipeline.py       fast mode: window resampling and features (NumPy only)
live226.py             full mode: loading, per-frame landmarks, clip prediction
signturk_runtime/      full mode: RTMW extractor, features, ensemble, framing check
research/model_226/    full mode: stream config, vocabulary, evaluation summary, model card
landmark_smoother.py   smoothing applied to avatar landmark files
database.py, models.py SQLAlchemy engine and tables
text_processing/       Turkish sentence engine
frontend/              interface (React from a CDN), avatar page, three.js r128, avatar.glb
landmarks/             226 landmark sequences for the avatar
demo_assets_179/       fast mode metadata; the weights are downloaded
tools/                 download_models.py, extract_landmarks.py (avatar files from AUTSL), evaluate_226.py
tests/                 unit tests and the end-to-end check (backend_smoke.py)
```

## Tests

```bash
python -m unittest discover -s tests
python -m text_processing.eval --check --min-exact 0.98
ruff check .
python tests/backend_smoke.py      # needs requirements-dev.txt and the weights
```

CI runs all of them on every push and pull request, the last one with the
real models, MediaPipe and RTMW: it plays the avatar's landmark recordings
and a webcam clip through the fast mode, and the same clip through the
full-mode socket. It also checks that every weight URL is reachable, and
builds and starts the Docker image.

`tools/evaluate_226.py` measures the full mode on AUTSL videos, which are not
included. The test split takes about 75 minutes on an RTX 5070 Ti:

```bash
python tools/evaluate_226.py --videos AUTSL/test --labels AUTSL/test/ground_truth.csv
```

## Scope

- SignTurk recognises isolated signs, one at a time.
- The full mode is heavier than the fast mode; a [GPU](#gpu) keeps it
  responsive.
- The sentence engine is rule-based and covers common sentence patterns.
- `tools/extract_landmarks.py` regenerates the avatar files from the AUTSL
  training videos.

## Data

Both models and the landmark sequences were derived from the AUTSL dataset:
O. M. Sincan and H. Y. Keles, *AUTSL: A Large Scale Multi-Modal Turkish Sign
Language Dataset and Baseline Methods*, IEEE Access, 2020. The videos are not
included; their use is governed by the dataset's own terms.

## Team and role

SignTurk was built as a team graduation project. My part was the LLM/NLP and
data workflows: dataset preparation, preprocessing, validation and
sentence-generation pipeline integration. I also supported the interface and
the 3D avatar integration.

## License

The code is released under the [MIT License](LICENSE). Third-party components,
including three.js, RTMW, YOLOX, the avatar model and the AUTSL-derived data,
keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
