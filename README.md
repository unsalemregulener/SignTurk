# SignTurk

Real-time Turkish Sign Language (TİD) recognition in the browser. A webcam
stream is classified sign by sign, approved words are turned into a Turkish
sentence, and a 3D avatar can play signs back. The interface and API messages
are in Turkish.

[![Live recognition of the sign "çay" (tea)](docs/images/cay-preview.gif)](docs/videos/cay.mp4)

- **Live recognition:** webcam frames over a WebSocket, MediaPipe Hands on the
  server, a BiLSTM + attention classifier over 179 signs.
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

## How live recognition works

1. The browser sends a JPEG frame at most every 100 ms and waits for the
   answer before sending the next one.
2. The server runs MediaPipe Hands on the frame: 2 hands × 21 points × xyz =
   126 values, the signer's left hand first. MediaPipe names hands as if the
   image were mirrored, so on these unmirrored frames its "Right" hand is the
   signer's left. Frames without a hand are skipped; after 8 in a row the
   window is reset.
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
7. Approved words are saved to the user's history and the sentence engine
   rebuilds the sentence from all approved words.

The 32-frame window lets a whole sign fit into one prediction. The model still
sees 16 frames, roughly every second frame of the window, and has not been
retrained on longer sequences.

## Model

| | |
| --- | --- |
| Vocabulary | 179 of the 226 signs in [AUTSL](#data) |
| Input | 16 frames × 156 hand features |
| Architecture | 2 × bidirectional LSTM (256 units) → additive temporal attention → dense 512 → dense 256 → softmax |
| Accuracy | 85.65% top-1, 93.96% top-3, 95.59% top-5 on the AUTSL validation split (3,512 samples) |
| Weights | `live-179.keras`, 62.6 MB, in the [`model-assets-v1`](https://github.com/unsalemregulener/SignTurk/releases/tag/model-assets-v1) release |

The validation split was also used to pick the checkpoint, so there is no
held-out test figure for this model, and live webcam accuracy has not been
measured. The vocabulary keeps the AUTSL signs that an earlier 226-class model
recognised with at least 50% validation accuracy (184), minus five that stayed
below 50% in a 184-class version of this model.

## Quick start

Python 3.11 is used in the Docker image and in CI.

```bash
git clone https://github.com/unsalemregulener/SignTurk.git
cd SignTurk
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tools/download_models.py    # downloads the weights and checks their SHA-256
cp .env.example .env               # set SIGNTURK_ADMIN_EMAIL and SIGNTURK_ADMIN_PASSWORD
uvicorn backend:app --port 8000
```

Open <http://127.0.0.1:8000>, register or sign in with the admin account, and
allow camera access on the **Canlı Çeviri** page.

With Docker:

```bash
docker compose up --build
```

## Configuration

Settings are read from the environment or from a `.env` file
(see [`.env.example`](.env.example)).

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | empty: `local_dev.db` (SQLite) | e.g. `postgresql+psycopg://user:password@host:5432/db` |
| `ALLOWED_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | browser origins allowed by CORS |
| `SIGNTURK_ADMIN_EMAIL`, `SIGNTURK_ADMIN_PASSWORD` | empty | first administrator, created at startup; the password needs at least 8 characters |
| `SIGNTURK_ADMIN_NAME` | `SignTurk Admin` | display name of that account |
| `HF_TOKEN`, `SIGNAI_*` | off | optional ML layer for the sentence engine, see [`text_processing`](text_processing/README.md) |

There are no built-in accounts. Without the admin variables the application
runs with registered users only.

## API

Authenticated routes take `Authorization: Bearer <token>`; the token comes
from `/api/login`. Browsers cannot set headers on WebSockets, so the live
socket takes it as `?token=`.

| Route | Access | Purpose |
| --- | --- | --- |
| `GET /` | public | web interface |
| `GET /avatar3d` | public | avatar page, embedded by the interface |
| `GET /api/health` | public | model, MediaPipe, database and window sizes |
| `GET /signs`, `GET /landmark/{word}` | public | avatar vocabulary and landmark sequences |
| `GET /api/dictionary`, `GET /api/dictionary/{class_id}` | public | model labels in Turkish and English |
| `POST /api/register`, `POST /api/login`, `POST /api/logout` | public | accounts |
| `WS /api/predict/live?token=…` | user | live recognition; also accepts 126 landmark values per message instead of a frame |
| `POST /api/predict/sequence` | user | classify 16 or 32 frames of 126 landmark values (signer's left hand first) |
| `GET /api/history`, `GET /api/me/summary` | user | approved words of the current user |
| `GET /api/settings`, `POST /api/settings` | user | speech and avatar speed |
| `GET /api/model/status` | user | model summary |
| `/api/text/*` | user | sentence engine: `correct`, `correct/async`, `jobs/{id}`, `models`, `health`, `ping`, `audio/{file}` |
| `GET /api/admin/overview`, `users`, `logs`, `model` | admin | administration |
| `GET /api/debug/model-check` | admin | loading state of every model asset and the model config |
| `DELETE /api/text/cache`, `DELETE /api/text/cache/{model_key}` | admin | clear the sentence engine cache |

## Project layout

```
backend.py             FastAPI app: accounts, live WebSocket, history, admin
live_pipeline.py       window resampling and feature extraction (NumPy only)
landmark_smoother.py   smoothing applied to avatar landmark files
database.py, models.py SQLAlchemy engine and tables
text_processing/       Turkish sentence engine
frontend/              interface (React from a CDN), avatar page, three.js r128, avatar.glb
landmarks/             226 landmark sequences for the avatar
demo_assets_179/       model metadata; the weights are downloaded
tools/                 download_models.py
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
real model and MediaPipe. It also checks that the weight URL is reachable, and
builds and starts the Docker image.

The end-to-end check also feeds the avatar's recordings of the 179 model
classes through the live input mapping and expects at least 85% top-1 (it
gets 90.5%). The recordings come from AUTSL training videos, so this guards the
input pipeline; it says nothing about accuracy on new signers.

## Limitations

- Only hand landmarks are used. Facial expressions and body movement, which
  carry meaning in TİD, are ignored.
- Signs are recognised one at a time; continuous signing is not segmented.
- The model was trained on AUTSL recordings. Lighting, distance and camera
  angle on a home webcam differ, and live accuracy has not been measured.
- The sentence engine is rule-based and covers common patterns; its known
  errors are listed in [`text_processing/README.md`](text_processing/README.md).
- Access tokens are kept in server memory: a restart signs everyone out, and
  several workers do not share sessions.
- History rows written by versions without per-user history have no owner.
  They count in the admin overview but appear in no user's history.
- The script that extracted, scored and smoothed the landmark files is not
  included. Each file records its source video (`source_video`) and a
  `quality_score`.

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
including three.js, the avatar model and the AUTSL-derived data, keep their own
terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
