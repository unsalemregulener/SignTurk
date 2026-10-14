# SignTurk

Real-time Turkish Sign Language (TİD) recognition in the browser. Signs made
in front of a webcam are recognised one at a time, from a short recording or
hands-free, approved words are turned into a Turkish sentence, and a 3D avatar
can play signs back. The interface and API messages are in Turkish.

[![Live recognition of the sign "çay" (tea)](docs/images/cay-preview.gif)](docs/videos/cay.mp4)

- **226 signs:** every sign in AUTSL, recognised from RTMW whole-body
  landmarks by a four-stream ensemble: 94.39% top-1 on the AUTSL test split.
- **Approval step:** by default a prediction enters the sentence only after
  the user accepts it, so a wrong guess can be rejected first.
- **Hands-free mode:** with a GPU the page can watch the camera without a
  record button. Each sign is cut out of the stream once the hands come back
  down and waits for approval like a recorded one; the user can let confident
  results into the sentence automatically.
- **Turkish sentence engine:** rule-based morphology turns glosses such as
  `ben çay içmek` into `Ben çay içiyorum.`
- **3D avatar:** 226 recorded landmark sequences replayed on a rigged
  three.js character, with captions, slow mode and loop.
- **Accounts:** token login, per-user history and settings, admin pages for
  users, logs and model status.
- **Storage:** SQLite with no setup, or PostgreSQL through `DATABASE_URL`.

<img src="docs/images/avatar.jpg" alt="3D avatar signing hoşça kal" width="560">

## Live recognition

On the **Canlı Çeviri** page signs are read one at a time. In the **Kayıt**
mode each sign is recorded on its own:

1. The user presses **İşareti kaydet**. After a 3-2-1 countdown the browser
   captures four seconds of unmirrored frames at about 12 fps (48 frames) and
   streams them to the server as it records.
2. For each frame the server runs RTMW whole-body pose estimation inside the
   person box of a YOLOX detector, which runs on every fourth frame. Of the 133
   keypoints it keeps 17 body points and 21 points per hand: 75 landmarks with
   the signer's own left and right, divided by the longer side of the frame so
   that x and y keep one scale.
3. When the recording is complete the server cuts the sign out of it, the way
   AUTSL clips are cut: from where the higher wrist starts to rise from its
   rest height to where it is back down. The cut is resampled to 32 frames,
   and the landmarks are centred between the shoulders, scaled by the shoulder
   width and turned into joint, bone, motion, geometry and hand features.
4. Four Keras models score the 226 signs and their probabilities are averaged
   with the weights 0.10, 0.21, 0.55 and 0.14.
5. The page shows the top three. The first becomes the approval candidate when
   its probability is at least 0.70 and 0.12 above the second, and any of the
   three can be approved. Barely moving hands or a badly framed signer add a
   hint next to the result.

Approved words are saved to the user's history, and the sentence engine
rebuilds the sentence from all approved words.

In the **Sürekli** (hands-free) mode the browser keeps sending frames and the
server watches the higher wrist. When the hands rise above their rest height
and come back down for 0.3 s, the sign is cut out as in step 3, scored and
offered for approval like a recording; a twitch shorter than half a second is
ignored. With **Otomatik onay** switched on, a result that passes the
thresholds goes straight into the sentence instead, and **Geri al** removes
it. Every frame has to be processed within the camera's 83 ms, so the mode is
offered when frames are fast enough, in practice on a GPU.

The model learnt from signers filmed from further away, with the space below
their shoulders in view, where most signs are made. While the camera is on,
the page checks a few frames a second and shows a silhouette of where to sit;
until at least 1.2 shoulder widths below the shoulders are in view it asks the
signer to sit back or tilt the camera down.

### Measured on AUTSL

[`tools/evaluate_recording.py`](tools/evaluate_recording.py) turns the 3,742
clips of AUTSL's test split into what the live page sees: frames at 12 fps,
landmarks with the detector on every fourth frame, and the signer at rest
before and after the sign.

| Test split | Top-1 |
| --- | ---: |
| The clip alone at 12 fps | 93.80% |
| The earlier 2.7 s recording, sign starting within 0.7 s | 89.87–91.15% |
| 4 s recording read whole, sign starting within 2 s | 76.27–78.81% |
| 4 s recording cut, sign starting within 2 s | 92.28–92.94% |

In streams of five signs by one signer, with 0.3–1.5 s of rest between them,
the hands-free mode cut out 95.4% of the signs and read 92.7% of those
correctly; 2.5 cuts per 100 signs matched no sign. Of the results confident
enough to be offered, 97.9% were right, and a sign was cut 0.08 s (median)
after its clip ended.

How much of the space below the shoulders the camera sees matters most. With
the frame ending 1.2 shoulder widths below them, a cut recording still reads
90.65%; with 0.5, as when sitting close to a laptop camera, it reads 64.99%
and the hands-free mode finds about half of the signs. The settings were
chosen on the validation split; the
[model card](research/model_226/MODEL_CARD.md#recordings-and-the-hands-free-mode)
has both splits and the framing table.

## Model

| | |
| --- | --- |
| Vocabulary | all 226 signs in [AUTSL](#data) |
| Landmarks | RTMW-x-l whole body (256×192) with a YOLOX-m person detector; 17 body + 2 × 21 hand points |
| Model input | 32 frames; joint, bone, motion, geometry and hand streams |
| Architecture | four recurrent streams, weighted average ([model card](research/model_226/MODEL_CARD.md)) |
| Accuracy | 94.39% top-1, 98.64% top-3 on the AUTSL test split (3,742 samples) |
| Download | about 660 MB, including the RTMW and YOLOX ONNX models |

The accuracy is an offline result on AUTSL, measured with
`tools/evaluate_226.py`, which runs the test videos through the server's code;
the [model card](research/model_226/MODEL_CARD.md) has the details.

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
python tools/download_models.py
cp .env.example .env               # set SIGNTURK_ADMIN_EMAIL and SIGNTURK_ADMIN_PASSWORD
uvicorn backend:app --port 8000
```

Open <http://127.0.0.1:8000>, register or sign in with the admin account, and
allow camera access on the **Canlı Çeviri** page. The model loads in the
background when the server starts; the admin **Model Durumu** page shows its
state.

With Docker:

```bash
docker compose up --build
```

The compose file keeps the SQLite database in the `signturk-data` volume
unless `.env` sets `DATABASE_URL`.

### GPU

YOLOX and RTMW run through ONNX Runtime. With an NVIDIA GPU, replace the CPU
build with the CUDA one; pip brings CUDA and cuDNN along:

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
| `SIGNTURK_ENABLE_226` | `1` | `0` skips loading the model, for example to work on the other pages |
| `SIGNTURK_226_DETECT_EVERY` | `4` | run the person detector on every Nth frame of a recording and reuse its box in between |
| `SIGNTURK_226_DEVICE` | `auto` | `cuda` or `cpu`; `auto` uses CUDA when ONNX Runtime has it |
| `SIGNTURK_CONTINUOUS` | `auto` | hands-free mode: `auto` offers it while a frame takes at most 60 ms (on a GPU before the first frame), `on` always, `off` never |
| `HF_TOKEN`, `SIGNAI_*` | off | optional ML layer for the sentence engine, see [`text_processing`](text_processing/README.md) |

There are no built-in accounts. Without the admin variables the application
runs with registered users only.

## API

Authenticated routes take `Authorization: Bearer <token>`; the token comes
from `/api/login`. Browsers cannot set headers on WebSockets, so the live
socket expects it as its first message, `{"action": "auth", "token": "…"}`,
which keeps it out of URLs and access logs.

| Route | Access | Purpose |
| --- | --- | --- |
| `GET /` | public | web interface |
| `GET /avatar3d` | public | avatar page, embedded by the interface |
| `GET /api/health` | public | model state, device and timings, database |
| `GET /signs`, `GET /landmark/{word}` | public | avatar vocabulary and landmark sequences |
| `GET /api/dictionary`, `GET /api/dictionary/{autsl_id}` | public | the 226 signs with their Turkish spelling and English name |
| `POST /api/register`, `POST /api/login`, `POST /api/logout` | public | accounts |
| `WS /api/predict/live226` | user | `{"action": "start"}`, the frames as `{"image": data_url}`, `{"action": "end"}` |
| `GET /api/history`, `GET /api/me/summary` | user | approved words of the current user |
| `GET /api/settings`, `POST /api/settings` | user | speech and avatar speed |
| `GET /api/model/status` | user | model state, thresholds and streams |
| `/api/text/*` | user | sentence engine: `correct`, `correct/async`, `jobs/{id}`, `models`, `health`, `ping`, `audio/{file}` |
| `GET /api/admin/overview`, `users`, `logs`, `model` | admin | administration |
| `GET /api/debug/model-check` | admin | missing model files, extractor settings and model state |
| `DELETE /api/text/cache`, `DELETE /api/text/cache/{model_key}` | admin | clear the sentence engine cache |

The live socket also accepts `approve` (optionally with a `word`), `reject`,
`undo` and `reset_sentence` actions and answers with `sentence_update`.
`{"action": "continuous"}` switches it to the hands-free mode (`"on": false`
switches back, `"auto_approve": true` approves confident results): every frame
is answered with `{"type": "watching", "state": "waiting" | "signing" |
"reading"}`, and every sign cut out of the stream with a `result`. Outside a
recording, `{"image": data_url, "preview": true}` is answered with
`{"type": "framing", "ok", "hint", "space_below", "width"}`; hands-free frames
carry the same under `framing`.

## Project layout

```
backend.py             FastAPI app: accounts, live WebSocket, history, admin
live226.py             model loading, per-frame landmarks, prediction
signturk_runtime/      YOLOX and RTMW (rtm.py), extractor, features, sign cut (segment.py), ensemble, framing check
research/model_226/    stream config, vocabulary, evaluation summary, model card
landmark_smoother.py   smoothing applied to avatar landmark files
database.py, models.py SQLAlchemy engine and tables
text_processing/       Turkish sentence engine
frontend/              interface (React from a CDN), avatar page, three.js r128, avatar.glb
landmarks/             226 landmark sequences for the avatar
tools/                 download_models.py, evaluate_226.py, evaluate_recording.py, extract_landmarks.py (avatar files from AUTSL)
tests/                 unit tests, the end-to-end check (backend_smoke.py), the socket check
LICENSES/              license of the code adapted from rtmlib
```

## Tests

```bash
python -m unittest discover -s tests
python -m text_processing.eval --check --min-exact 0.98
ruff check .
python tests/backend_smoke.py      # needs requirements-dev.txt and the weights
```

CI runs all of them on every push and pull request, the last one with the
real models: it sends four-second webcam recordings through the live socket,
one of them with a second of rest before the sign. It also checks that every
weight URL is reachable, and builds and starts the Docker image, where
`tests/socket_check.py` sends a recording through uvicorn.

`tools/evaluate_226.py` measures the model on AUTSL videos, which are not
included. The test split takes about 75 minutes on an RTX 5070 Ti:

```bash
python tools/evaluate_226.py --videos AUTSL/test --labels AUTSL/test/ground_truth.csv
```

With `--cache-landmarks DIR` it saves the landmarks of every frame instead
(`--cache-fps 12` keeps the frames the live page would send), and
`tools/evaluate_recording.py --cache DIR` turns the clips into four-second
recordings with rest around the sign and compares the model's input taken
from the whole recording and from the cut; `--streams` measures the
hands-free mode. The results are in [Measured on AUTSL](#measured-on-autsl).

## Scope

- SignTurk recognises isolated signs, one at a time. In the hands-free mode
  the hands have to come back down for about a third of a second between
  signs; signs that run into each other are read as one.
- The camera has to see the signer from about the waist up; a close laptop
  camera that cuts off the hands loses most of the accuracy.
- A [GPU](#gpu) keeps the live page responsive; on a CPU a recorded sign takes
  a few seconds to process.
- The sentence engine is rule-based and covers common sentence patterns.
- `tools/extract_landmarks.py` regenerates the avatar files from the AUTSL
  training videos; it needs MediaPipe, which `requirements.txt` leaves out.

## Data

The model and the landmark sequences were derived from the AUTSL dataset:
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
including three.js, RTMW, YOLOX, the code adapted from rtmlib, the avatar model
and the AUTSL-derived data, keep their own terms; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
