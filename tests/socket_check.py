"""Open the live socket of a running server the way the browser does.

backend_smoke.py talks to the app through FastAPI's test client, which skips
uvicorn and the websockets package; this check goes through both. CI runs it
inside the Docker container, where OpenCV and websockets are installed:

    python socket_check.py --video cay.mp4 [--url http://127.0.0.1:8000]
"""

import argparse
import base64
import json
import secrets
import sys
import urllib.request

import cv2
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

FAILURES = []


def check(name, ok, detail=None):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail is not None and not ok else ""))
    if not ok:
        FAILURES.append(name)


def post(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def camera_recording(path, seconds=4.0, rest_first=0.0, fps=12):
    """`seconds` of the recorded camera panel at `fps`, unmirrored, as JPEG data URLs; the first
    `rest_first` seconds at rest (see backend_smoke.py)."""
    capture = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.flip(frame[222:612, 256:882], 1))
    capture.release()
    rest = frames[int(4.5 * 30):]
    picked = [rest[int(round(i * 30 / fps)) % len(rest)] for i in range(int(round(rest_first * fps)))]
    picked += [frames[int(round(i * 30 / fps))] for i in range(int(round((seconds - rest_first) * fps)))]
    clip = []
    for frame in picked:
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        clip.append("data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii"))
    return clip


def close_code(ws):
    try:
        ws.recv(timeout=15)
    except ConnectionClosed as closed:
        return closed.rcvd.code if closed.rcvd else None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--video", required=True, help="docs/videos/cay.mp4")
    args = parser.parse_args()
    ws_url = "ws" + args.url[len("http"):]

    email, password = f"socket-{secrets.token_hex(4)}@example.com", secrets.token_urlsafe(12)
    post(f"{args.url}/api/register", {"full_name": "Socket Check", "email": email, "password": password})
    token = post(f"{args.url}/api/login", {"email": email, "password": password})["access_token"]
    check("login over HTTP", bool(token))

    with connect(f"{ws_url}/api/predict/live226") as ws:
        ws.send(json.dumps({"action": "start"}))
        check("socket without a token is closed with 1008", close_code(ws) == 1008)

    with connect(f"{ws_url}/api/predict/live226") as ws:
        ws.send(json.dumps({"action": "auth", "token": token}))
        ws.send(json.dumps({"action": "start"}))
        check("recording starts", json.loads(ws.recv(timeout=60)).get("type") == "recording")
        for frame in camera_recording(args.video):
            ws.send(json.dumps({"image": frame}))
            json.loads(ws.recv(timeout=120))
        ws.send(json.dumps({"action": "end"}))
        result = json.loads(ws.recv(timeout=120))
        top = result.get("top", [])
        print("     cay clip:", [(row["gloss"], row["confidence"]) for row in top])
        check("the cay recording is recognised", bool(top) and top[0]["gloss"] == "cay", result)

    # Hands-free: a second of rest, then the whole video, whose sign ends in rest
    with connect(f"{ws_url}/api/predict/live226") as ws:
        ws.send(json.dumps({"action": "auth", "token": token}))
        ws.send(json.dumps({"action": "continuous"}))
        reply = json.loads(ws.recv(timeout=60))
        if not reply.get("on"):
            print("     the hands-free mode is not offered here; SIGNTURK_CONTINUOUS=on tests it anyway")
        else:
            found = []
            for frame in camera_recording(args.video, seconds=7.5, rest_first=1.0):
                ws.send(json.dumps({"image": frame}))
                if json.loads(ws.recv(timeout=120)).get("state") == "reading":
                    found.append(json.loads(ws.recv(timeout=120)))
            print("     hands-free stream:", [(r["top"][0]["gloss"], r["top"][0]["confidence"]) for r in found])
            check("the hands-free stream holds one sign, cay, waiting for approval",
                  len(found) == 1 and found[0]["top"][0]["gloss"] == "cay" and found[0].get("pending_word") == "cay",
                  found)

    print(f"{len(FAILURES)} failed" if FAILURES else "all socket checks passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
