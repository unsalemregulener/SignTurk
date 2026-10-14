"""End-to-end check of the backend with the real 226-class models.

Needs the full requirements, httpx and the downloaded weights:

    pip install -r requirements-dev.txt
    python tools/download_models.py
    python tests/backend_smoke.py

Runs against a throwaway SQLite database. Test credentials are generated at
run time and never printed.
"""
import base64
import json
import os
import secrets
import sys
import tempfile
import time
from contextlib import contextmanager
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
from text_processing.lexicon import restore_turkish

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail and not ok else ""))


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


@contextmanager
def live_socket(client, token):
    """The live WebSocket after its token went in the first message."""
    with client.websocket_connect("/api/predict/live226") as ws:
        ws.send_text(json.dumps({"action": "auth", "token": token}))
        yield ws


def wait_for_model(client, seconds=900):
    """The models load in the background; wait until they settle."""
    deadline = time.time() + seconds
    while True:
        status = client.get("/api/health").json()["model_226"]
        if status["state"] not in ("idle", "loading") or time.time() > deadline:
            return status
        time.sleep(2)


def record(ws, clip):
    """Send one recorded sign and return the result and the number of processed frames."""
    ws.send_text(json.dumps({"action": "start"}))
    started = ws.receive_json().get("type") == "recording"
    processed = 0
    for frame in clip:
        ws.send_text(json.dumps({"image": frame}))
        processed = ws.receive_json().get("processed", processed)
    ws.send_text(json.dumps({"action": "end"}))
    return started, ws.receive_json(), processed


with TestClient(backend.app) as client:
    health = client.get("/api/health").json()
    check("health ok", health.get("status") == "ok", health)
    check("226 avatar signs indexed", health.get("landmark_count") == 226, health.get("landmark_count"))

    email, password = "user@example.com", secrets.token_urlsafe(12)
    r = client.post("/api/register", json={"full_name": "Test Kullanıcı", "email": email, "password": password})
    check("register", r.status_code == 200, r.text)
    r = client.post("/api/register", json={"full_name": "X", "email": "short@example.com", "password": "123"})
    check("short password rejected", r.status_code == 422)
    r = client.post("/api/login", json={"email": email, "password": password})
    user_token = r.json().get("access_token")
    check("login returns a token", r.status_code == 200 and bool(user_token))
    check("token lifetime is reported", r.json().get("expires_in", 0) > 0)

    spare = client.post("/api/login", json={"email": email, "password": password}).json()["access_token"]
    user_id, _ = backend._access_tokens[spare]
    backend._access_tokens[spare] = (user_id, time.time() - 1)
    check("expired token is refused", client.get("/api/history", headers={"Authorization": f"Bearer {spare}"}).status_code == 401)

    wrong = [client.post("/api/login", json={"email": "nobody@example.com", "password": "wrong-password"}).status_code
             for _ in range(6)]
    check("repeated failed logins are limited", wrong == [401] * 5 + [429], wrong)
    check("other accounts can still sign in",
          client.post("/api/login", json={"email": email, "password": password}).status_code == 200)
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
    check("model files are all present", client.get("/api/debug/model-check", headers=A).json().get("missing_files") == [])

    check("settings defaults", client.get("/api/settings", headers=U).json()
          == {"tts_enabled": False, "speech_rate": 1.0, "avatar_speed": 1.0})
    client.post("/api/settings", headers=U, json={"tts_enabled": True, "speech_rate": 1.4, "avatar_speed": 0.8})
    check("settings are per user", client.get("/api/settings", headers=U).json()["speech_rate"] == 1.4
          and client.get("/api/settings", headers=A).json()["speech_rate"] == 1.0)

    signs = client.get("/signs").json()
    check("avatar signs have Turkish labels", len(signs["labels"]) == 226 and signs["labels"].get("cay") == "çay")
    dictionary = client.get("/api/dictionary").json()
    check("dictionary lists 226 signs with English names",
          len(dictionary) == 226 and all(entry["en"] for entry in dictionary))
    check("English labels have no underscores", all("_" not in entry["en"] for entry in dictionary))
    entry = client.get("/api/dictionary/38").json()
    check("dictionary entry by AUTSL id", (entry.get("text"), entry.get("en")) == ("çay", "tea"), entry)

    check("text API needs a token", client.get("/api/text/models").status_code == 401)
    r = client.post("/api/text/correct", headers=U, json={"words": ["ben", "okul", "gitmek"]})
    check("text API corrects without audio", r.status_code == 200 and r.json()["sentence"] == "Ben okula gidiyorum."
          and r.json()["tts_status"] == "disabled", r.text)
    check("text cache is admin only", client.delete("/api/text/cache", headers=U).status_code == 403)

    status = wait_for_model(client)
    check("model ready", status["ready"], status)
    model = client.get("/api/model/status", headers=U).json()
    check("model status for users", model["model_226"]["num_classes"] == 226 and len(model["streams"]) == 4, model)

    for name, path in (("live socket refuses a missing token", "/api/predict/live226"),
                       ("token in the URL is not accepted", f"/api/predict/live226?token={user_token}")):
        try:
            with client.websocket_connect(path) as ws:
                ws.send_text(json.dumps({"action": "start"}))
                ws.receive_text()
            check(name, False)
        except WebSocketDisconnect as exc:
            check(name, exc.code == 1008, exc.code)

    if status["ready"]:
        with live_socket(client, user_token) as ws:
            for word in ("ben", "cay", "icmek"):
                ws.send_text(json.dumps({"action": "approve", "word": word}))
                update = ws.receive_json()
            check("approved glosses form a sentence", update.get("sentence") == "Ben çay içiyorum.", update)
            check("sentence shows Turkish spelling", update.get("display_words") == ["ben", "çay", "içmek"], update)

        history = client.get("/api/history", headers=U).json()
        check("history stores approved words", [row["result"] for row in history] == ["icmek", "cay", "ben"], history)
        check("history shows Turkish spelling", [row["text"] for row in history] == ["içmek", "çay", "ben"], history)
        check("history is per user", client.get("/api/history", headers=A).json() == [])
        check("summary counts approved words", client.get("/api/me/summary", headers=U).json()["translations"] == 3)

        clip = camera_clip(ROOT / "docs" / "videos" / "cay.mp4")
        default_every = backend.LIVE226.detect_every
        backend.LIVE226.reset_timings()
        with live_socket(client, user_token) as ws:
            started_ok, result, processed = record(ws, clip)
            check("recording starts", started_ok)
            top = result.get("top", [])
            timing = client.get("/api/health").json()["model_226"]
            print(f"     cay clip, detection every {default_every} frame(s): "
                  f"{[(row['gloss'], row['confidence']) for row in top]} | "
                  f"detect {timing['avg_detect_ms']} ms, pose {timing['avg_pose_ms']} ms per frame, "
                  f"model {result.get('model_ms')} ms, feedback {result.get('feedback')!r}")
            check("every frame is processed", processed == len(clip) == result.get("frames"), (processed, result))
            check("the cay recording is recognised", bool(top) and top[0]["gloss"] == "cay" and result.get("confident"),
                  result)
            check("three candidates with Turkish spelling",
                  len(top) == 3 and all(row["text_tr"] == restore_turkish(row["gloss"]) for row in top), result)
            ws.send_text(json.dumps({"action": "approve", "word": top[0]["gloss"] if top else "cay"}))
            update = ws.receive_json()
            check("approval updates the sentence", update.get("type") == "sentence_update" and len(update["words"]) == 1,
                  update)
        rows = client.get("/api/history", headers=U).json()
        check("approval stored with its confidence", rows[0]["mode"] == "Live226" and rows[0]["confidence"].endswith("%"),
              rows[:1])
        check("history confidence is a percentage or '-'",
              all(row["confidence"] == "-" or row["confidence"].endswith("%") for row in rows), rows)

        # The same clip with the other person-detection setting, for comparison
        other_every = 4 if default_every == 1 else 1
        backend.LIVE226.reset_timings()
        backend.LIVE226.detect_every = other_every
        with live_socket(client, user_token) as ws:
            _, other, _ = record(ws, clip)
        backend.LIVE226.detect_every = default_every
        timing = client.get("/api/health").json()["model_226"]
        print(f"     cay clip, detection every {other_every} frame(s): "
              f"{[(row['gloss'], row['confidence']) for row in other.get('top', [])]} | "
              f"detect {timing['avg_detect_ms']} ms, pose {timing['avg_pose_ms']} ms per frame")
        check("the other detection setting also works", len(other.get("top", [])) == 3, other)

    client.post("/api/logout", headers=U)
    check("token revoked after logout", client.get("/api/history", headers=U).status_code == 401)

failed = results.count(False)
print(f"\n{len(results) - failed}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
