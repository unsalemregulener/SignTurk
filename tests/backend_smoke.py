"""End-to-end check of the backend with the real model and MediaPipe.

Needs the full requirements, httpx and the downloaded weights:

    pip install -r requirements.txt httpx
    python tools/download_models.py
    python tests/backend_smoke.py

Runs against a throwaway SQLite database. Test credentials are generated at
run time and never printed. SMOKE_FAKE_MEDIAPIPE=1 skips the check that needs
MediaPipe to find hands in a real video.
"""
import base64
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ["DATABASE_URL"] = f"sqlite:///{(Path(tempfile.mkdtemp()) / 'smoke.db').as_posix()}"
ADMIN_EMAIL = "admin@example.com"
ADMIN_PASSWORD = secrets.token_urlsafe(12)
os.environ["SIGNTURK_ADMIN_EMAIL"] = ADMIN_EMAIL
os.environ["SIGNTURK_ADMIN_PASSWORD"] = ADMIN_PASSWORD

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import backend
from live_pipeline import hands_to_landmarks, resample_window
from text_processing.lexicon import restore_turkish

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


def landmark_vectors(path):
    """Model input layout for a landmark file.

    The files were extracted with MediaPipe's Tasks HandLandmarker, which
    names the signer's own hands; the live Hands solution names them the other
    way round, so left_hand is passed as its "Right".
    """
    frames = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    return np.stack([hands_to_landmarks([("Right", frame.get("left_hand")), ("Left", frame.get("right_hand"))])
                     for frame in frames])


def video_frame_data_url(path, position=0.5):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) * position))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read a frame from {path}")
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def camera_frames(path, end=0.55):
    """Unmirrored camera-panel frames of the recorded interface (BGR, 30 fps).

    docs/videos/cay.mp4 is a screen recording: the sign is made during the
    first ~55% and the camera preview on screen was mirrored, so the panel is
    cropped and flipped back to the frames the browser sent.
    """
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.flip(frame[222:612, 256:882], 1))
    capture.release()
    return frames[:max(int(len(frames) * end), 1)]


def data_url(frame):
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def camera_clip(path, count=32):
    """`count` evenly spaced camera-panel frames, as data URLs."""
    frames = camera_frames(path)
    return [data_url(frames[i]) for i in np.linspace(0, len(frames) - 1, count).round().astype(int)]


def wait_for_226(client, seconds=900):
    """The 226 models load in the background; wait until they settle."""
    deadline = time.time() + seconds
    while True:
        status = client.get("/api/health").json()["model_226"]
        if status["state"] not in ("idle", "loading") or time.time() > deadline:
            return status
        time.sleep(2)


def synthetic_hands(n):
    """Two-hand landmark vectors with a little motion."""
    base = np.random.default_rng(0).uniform(0.3, 0.7, size=126).astype(np.float32)
    return [(base + 0.01 * np.sin(i / 3.0)).tolist() for i in range(n)]


with TestClient(backend.app) as client:
    health = client.get("/api/health").json()
    check("health ok", health.get("status") == "ok", health)
    check("model loaded", health.get("model_loaded") is True)
    check("MediaPipe ready", health.get("mediapipe_ready") is True)
    check("capture window 32, model input 16 x 156",
          (health.get("capture_window"), health.get("seq_len"), health.get("feat_dim")) == (32, 16, 156), health)
    check("226 avatar signs indexed", health.get("landmark_count") == 226, health.get("landmark_count"))

    email, password = "user@example.com", secrets.token_urlsafe(12)
    r = client.post("/api/register", json={"full_name": "Test Kullanıcı", "email": email, "password": password})
    check("register", r.status_code == 200, r.text)
    r = client.post("/api/register", json={"full_name": "X", "email": "short@example.com", "password": "123"})
    check("short password rejected", r.status_code == 422)
    r = client.post("/api/login", json={"email": email, "password": password})
    user_token = r.json().get("access_token")
    check("login returns a token", r.status_code == 200 and bool(user_token))
    r = client.post("/api/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})
    admin_token = r.json().get("access_token")
    check("admin created from environment", r.status_code == 200 and r.json()["user"]["role"] == "Admin")
    U = {"Authorization": f"Bearer {user_token}"}
    A = {"Authorization": f"Bearer {admin_token}"}

    check("history needs a token", client.get("/api/history").status_code == 401)
    check("admin routes refuse users", all(
        client.get(p, headers=U).status_code == 403
        for p in ("/api/admin/overview", "/api/admin/users", "/api/admin/logs", "/api/admin/model", "/api/debug/model-check")))
    check("admin routes accept admins", client.get("/api/admin/overview", headers=A).status_code == 200)
    check("model status for users", client.get("/api/model/status", headers=U).json().get("model_loaded") is True)

    check("settings defaults", client.get("/api/settings", headers=U).json()
          == {"tts_enabled": False, "speech_rate": 1.0, "avatar_speed": 1.0})
    client.post("/api/settings", headers=U, json={"tts_enabled": True, "speech_rate": 1.4, "avatar_speed": 0.8})
    check("settings are per user", client.get("/api/settings", headers=U).json()["speech_rate"] == 1.4
          and client.get("/api/settings", headers=A).json()["speech_rate"] == 1.0)

    signs = client.get("/signs").json()
    check("avatar signs have Turkish labels", len(signs["labels"]) == 226 and signs["labels"].get("cay") == "çay")
    dictionary = client.get("/api/dictionary").json()
    check("dictionary lists 226 signs, 179 of them in the live model",
          len(dictionary) == 226 and sum(entry["live_179_id"] is not None for entry in dictionary) == 179)
    check("English labels have no underscores", all("_" not in (entry["en"] or "") for entry in dictionary))
    check("dictionary entry by AUTSL id", client.get("/api/dictionary/38").json().get("text") == "çay")

    check("text API needs a token", client.get("/api/text/models").status_code == 401)
    r = client.post("/api/text/correct", headers=U, json={"words": ["ben", "okul", "gitmek"]})
    check("text API corrects without audio", r.status_code == 200 and r.json()["sentence"] == "Ben okula gidiyorum."
          and r.json()["tts_status"] == "disabled", r.text)
    check("text cache is admin only", client.delete("/api/text/cache", headers=U).status_code == 403)

    window = resample_window(landmark_vectors(ROOT / "landmarks" / "cay.json"), 16)
    r = client.post("/api/predict/sequence", headers=U, json={"landmarks": window.tolist()})
    body = r.json()
    check("landmarks/cay.json is recognised as cay", r.status_code == 200 and body.get("label_tr") == "cay",
          [(p["label_tr"], p["confidence"]) for p in body.get("top_predictions", [])])

    # The avatar recordings come from AUTSL training videos, so this checks the
    # input pipeline end to end, not accuracy on new signers.
    model_glosses = {entry["TR"] for entry in backend.LABEL_MAP.values()}
    recordings = sorted(p for p in (ROOT / "landmarks").glob("*.json") if p.stem in model_glosses)
    correct = 0
    for path in recordings:
        window = resample_window(landmark_vectors(path), 16)
        reply = client.post("/api/predict/sequence", headers=U, json={"landmarks": window.tolist()}).json()
        correct += int(reply.get("label_tr") == path.stem)
    share = correct / max(len(recordings), 1)
    print(f"     avatar recordings of the 179 model classes: top-1 {correct}/{len(recordings)} ({share:.1%})")
    check("pipeline matches training on the avatar recordings", len(recordings) == 179 and share >= 0.85, share)
    r = client.post("/api/predict/sequence", headers=U, json={"landmarks": [[0.0] * 126] * 20})
    check("sequence of the wrong length rejected", r.status_code == 400)

    try:
        with client.websocket_connect("/api/predict/live") as ws:
            ws.receive_text()
        check("live socket refuses a missing token", False)
    except WebSocketDisconnect as exc:
        check("live socket refuses a missing token", exc.code == 1008, exc.code)

    with client.websocket_connect(f"/api/predict/live?token={user_token}") as ws:
        ws.send_text(json.dumps({"image": video_frame_data_url(ROOT / "docs" / "videos" / "cay.mp4")}))
        reply = ws.receive_json()
        check("MediaPipe processes a camera frame", reply.get("label_tr") != "MediaPipe kullanılamıyor", reply)

    # The fast mode end to end on a webcam recording: frames at 10 fps, as the
    # browser sends them, through MediaPipe and the model.
    if os.environ.get("SMOKE_FAKE_MEDIAPIPE") == "1":
        print("     skipped the cay recording: it needs the real MediaPipe")
    else:
        frames = camera_frames(ROOT / "docs" / "videos" / "cay.mp4", end=0.6)[::3]
        vectors = [v for v in (backend.extract_landmarks_from_frame(f) for f in frames)
                   if v is not None and backend.count_detected_hands(v) > 0]
        if len(vectors) >= backend.CAPTURE_WINDOW:
            window = np.stack(vectors[:backend.CAPTURE_WINDOW])
            swapped = np.concatenate([window[:, 63:], window[:, :63]], axis=1)
            for name, candidate in (("live order", window), ("hands swapped", swapped)):
                top = backend.run_best_inference(candidate)["top_predictions"]
                print(f"     cay recording, {name}: {[(p['label_tr'], p['confidence']) for p in top]}")
        first = None
        with client.websocket_connect(f"/api/predict/live?token={user_token}") as ws:
            for frame in frames:
                ws.send_text(json.dumps({"image": data_url(frame)}))
                reply = ws.receive_json()
                if reply.get("class_id", -1) >= 0:
                    first = reply
                    break
        check("fast mode recognises the cay recording",
              first is not None and first["label_tr"].strip("()?") == "cay",
              first and [(p["label_tr"], p["confidence"]) for p in first["top_predictions"]] or f"{len(vectors)} hand frames")

    with client.websocket_connect(f"/api/predict/live?token={user_token}") as ws:
        ws.send_text(json.dumps({"landmarks": [0.0] * 126}))
        check("frame without hands is skipped", ws.receive_json()["label_tr"] == "El algılanmadı")
        prediction = None
        for vector in synthetic_hands(40):
            ws.send_text(json.dumps({"landmarks": vector}))
            reply = ws.receive_json()
            if reply.get("class_id", -1) >= 0:
                prediction = reply
                break
        check("prediction after 32 frames", prediction is not None and prediction["buffer_length"] == 32, prediction)
        if prediction:
            gloss = prediction["label_tr"].strip("()?")
            check("prediction has three candidates", len(prediction["top_predictions"]) == 3)
            check("prediction carries Turkish spelling", prediction["text_tr"] == restore_turkish(gloss), prediction)

        for word in ("ben", "cay", "icmek"):
            ws.send_text(json.dumps({"action": "approve", "word": word}))
            update = ws.receive_json()
        check("approved glosses form a sentence", update.get("sentence") == "Ben çay içiyorum.", update)
        check("sentence shows Turkish spelling", update.get("display_words") == ["ben", "çay", "içmek"], update)

    history = client.get("/api/history", headers=U).json()
    check("history stores approved words", [row["result"] for row in history] == ["icmek", "cay", "ben"], history)
    check("history shows Turkish spelling", [row["text"] for row in history] == ["içmek", "çay", "ben"], history)
    check("history confidence is a percentage or '-'",
          all(row["confidence"] == "-" or row["confidence"].endswith("%") for row in history), history)
    check("history is per user", client.get("/api/history", headers=A).json() == [])
    check("summary counts approved words", client.get("/api/me/summary", headers=U).json()["translations"] == 3)

    status = wait_for_226(client)
    check("226 model ready", status["ready"], status)
    if status["ready"]:
        clip = camera_clip(ROOT / "docs" / "videos" / "cay.mp4")
        with client.websocket_connect(f"/api/predict/live226?token={user_token}") as ws:
            ws.send_text(json.dumps({"action": "start"}))
            check("226 recording starts", ws.receive_json().get("type") == "recording")
            started = time.time()
            processed = 0
            for frame in clip:
                ws.send_text(json.dumps({"image": frame}))
                processed = ws.receive_json().get("processed", processed)
            ws.send_text(json.dumps({"action": "end"}))
            result = ws.receive_json()
            elapsed = time.time() - started
            top = result.get("top", [])
            print(f"     226 on the cay clip: {[(row['gloss'], row['confidence']) for row in top]}"
                  f" | {elapsed:.1f}s for {len(clip)} frames, model {result.get('model_ms')} ms,"
                  f" feedback {result.get('feedback')!r}")
            check("226 processes every frame", processed == len(clip) == result.get("frames"), (processed, result))
            check("226 returns three candidates with Turkish spelling",
                  len(top) == 3 and all(row["text_tr"] == restore_turkish(row["gloss"]) for row in top), result)
            ws.send_text(json.dumps({"action": "approve", "word": top[0]["gloss"] if top else "cay"}))
            update = ws.receive_json()
            check("226 approval updates the sentence", update.get("type") == "sentence_update" and len(update["words"]) == 1,
                  update)
        rows = client.get("/api/history", headers=U).json()
        check("226 approval stored as Live226 with its confidence",
              rows[0]["mode"] == "Live226" and rows[0]["confidence"].endswith("%"), rows[:1])
        print("     226 extraction:", client.get("/api/health").json()["model_226"].get("avg_frame_ms"), "ms per frame")

    client.post("/api/logout", headers=U)
    check("token revoked after logout", client.get("/api/history", headers=U).status_code == 401)

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
