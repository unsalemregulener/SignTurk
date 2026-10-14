"""SignTurk FastAPI application.

Serves the web interface, account and history APIs, live sign recognition
over WebSocket, sentence assembly, and the landmark data that drives the
3D avatar.
"""

import asyncio, os, json, time, traceback, uuid, logging, base64, secrets
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Load variables from a local .env file when python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from database import engine, get_db, Base, database_name
import models
from text_processing.lexicon import restore_turkish, restore_turkish_words
from live226 import Live226, prepare_frame
from rate_limit import SlidingWindowLimiter
from signturk_runtime.config import missing_files

# ── Landmark smoothing ───────────────────────────────────
try:
    from landmark_smoother import smooth_landmark_data
    SMOOTHER_AVAILABLE = True
except ImportError:
    SMOOTHER_AVAILABLE = False

# TensorFlow, loaded with the 226 ensemble, logs errors only
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ═══════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("signturk")

if not SMOOTHER_AVAILABLE:
    logger.warning("[LANDMARK] landmark_smoother not found; /landmark returns raw frames")

# ═══════════════════════════════════════════════════════════
#  PASSWORD HASHING
# ═══════════════════════════════════════════════════════════
from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

def hash_pw(pw: str) -> str:
    return pwd_context.hash(pw)

def verify_pw(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# ═══════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════
BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UI_FILE      = FRONTEND_DIR / "signturk_ui.html"
AVATAR_FILE  = FRONTEND_DIR / "avatar3d.html"

# Per-word landmark sequences that drive the 3D avatar
_candidate_landmark_dirs = [
    BASE_DIR / "landmarks",
    BASE_DIR / "dataset" / "landmarks",
]
LANDMARKS_DIR  = None
LANDMARK_INDEX = {}

# 226-class ensemble with RTMW landmarks (loaded in the background at startup)
LIVE226 = Live226()

# ═══════════════════════════════════════════════════════════
#  AVATAR LANDMARKS
# ═══════════════════════════════════════════════════════════
def load_landmark_index():
    """Index the per-word landmark files used for avatar playback."""
    global LANDMARKS_DIR, LANDMARK_INDEX

    for d in _candidate_landmark_dirs:
        if d.is_dir():
            LANDMARKS_DIR = d
            break

    if LANDMARKS_DIR:
        LANDMARK_INDEX = {
            fname[:-5].lower(): LANDMARKS_DIR / fname
            for fname in os.listdir(LANDMARKS_DIR)
            if fname.endswith(".json")
        }
        logger.info(f"[LANDMARKS] Indexed {len(LANDMARK_INDEX)} words")
    else:
        logger.warning("[LANDMARKS] Landmark directory not found")


# ═══════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════
def add_log_db(db: Session, level: str, message: str, user: str = "system"):
    log = models.Log(level=level, message=message, user=user)
    db.add(log)
    db.commit()


def decode_base64_image(data_url):
    try:
        encoded = data_url.split(",", 1)[-1] if "," in data_url else data_url
        arr = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"[DECODE] {e}")
        return None


def format_confidence(value: float) -> str:
    return f"{value * 100:.1f}%"


def parse_confidence(value: Optional[str]) -> Optional[float]:
    try:
        return float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None


def iso_time(value) -> Optional[str]:
    return value.isoformat() if value else None


# ── Schemas ───────────────────────────────────────────────
class SettingsData(BaseModel):
    tts_enabled: bool = False
    speech_rate: float = Field(1.0, ge=0.5, le=2.0)
    avatar_speed: float = Field(1.0, ge=0.25, le=3.0)

class RegisterData(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginData(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


# ═══════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await startup_event()
    yield


app = FastAPI(title="SignTurk API", version="4.0.0", lifespan=lifespan)

_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "ALLOWED_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Authentication ────────────────────────────────────────
# Bearer tokens live in memory and expire after SIGNTURK_TOKEN_HOURS; restarting
# the server signs everyone out. A multi-instance deployment needs a shared store.
_bearer = HTTPBearer(auto_error=False)
TOKEN_SECONDS = int(float(os.environ.get("SIGNTURK_TOKEN_HOURS", "12")) * 3600)
_access_tokens: Dict[str, Tuple[int, float]] = {}

# Failed logins per (IP, email) and per IP, registrations per IP
LOGIN_FAILURES = SlidingWindowLimiter(limit=5, window=600)
LOGIN_FAILURES_BY_IP = SlidingWindowLimiter(limit=20, window=600)
REGISTRATIONS_BY_IP = SlidingWindowLimiter(limit=10, window=3600)
TOO_MANY_ATTEMPTS = "Çok fazla deneme, biraz sonra tekrar deneyin"


def _issue_access_token(user_id: int) -> str:
    now = time.time()
    for expired in [t for t, (_, expires_at) in _access_tokens.items() if expires_at < now]:
        _access_tokens.pop(expired, None)
    token = secrets.token_urlsafe(32)
    _access_tokens[token] = (user_id, now + TOKEN_SECONDS)
    return token


def _user_for_token(token: Optional[str], db: Session):
    entry = _access_tokens.get(token or "")
    if not entry:
        return None
    user_id, expires_at = entry
    if expires_at < time.time():
        _access_tokens.pop(token, None)
        return None
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user or user.status != "Active":
        return None
    return user


def current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: Session = Depends(get_db),
):
    user = _user_for_token(credentials.credentials if credentials else None, db)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum açmanız gerekiyor",
        )
    return user


def admin_user(user=Depends(current_user)):
    if user.role != "Admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bu işlem için yönetici yetkisi gerekiyor",
        )
    return user

# ── Sentence engine (rule-based; ML grammar and TTS stay opt-in) ─────
try:
    from text_processing import GrammarCorrector
    SENTENCE_ENGINE = GrammarCorrector()  # use_ml=False by default
    logger.info("[TEXT] Sentence engine ready (rule-based; ML/LLM/TTS off)")
except Exception as e:
    SENTENCE_ENGINE = None
    logger.warning(f"[TEXT] Sentence engine unavailable: {e}")

try:
    from text_processing.web.router import admin_router as text_admin_router
    from text_processing.web.router import router as text_processing_router
    app.include_router(text_processing_router, dependencies=[Depends(current_user)])
    app.include_router(text_admin_router, dependencies=[Depends(admin_user)])
    logger.info("[TEXT] /api/text/* routes mounted")
except Exception as e:
    logger.warning(f"[TEXT] /api/text/* routes not mounted (optional deps?): {e}")


def assemble_sentence(words: List[str]) -> str:
    """Build a Turkish sentence from approved sign glosses (rule-based engine)."""
    cleaned = restore_turkish_words(w for w in (words or []) if w and str(w).strip())
    if not cleaned:
        return ""
    if len(cleaned) < 2:
        return ""
    if SENTENCE_ENGINE is None:
        return " ".join(cleaned)
    try:
        return SENTENCE_ENGINE.correct(cleaned)
    except Exception as e:
        logger.warning(f"[TEXT] sentence assembly failed: {e}")
        return " ".join(cleaned)


if FRONTEND_DIR.exists():
    # html=False: "/" is served by the route below, not by the static mount
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="frontend")


# ═══════════════════════════════════════════════════════════
#  ROUTES — pages and health
# ═══════════════════════════════════════════════════════════
@app.get("/")
async def root():
    if UI_FILE.exists():
        return FileResponse(str(UI_FILE))
    return JSONResponse(status_code=404, content={"error": "Arayüz dosyası bulunamadı"})


@app.get("/avatar3d")
async def avatar3d():
    if AVATAR_FILE.exists():
        return FileResponse(str(AVATAR_FILE))
    return JSONResponse(status_code=404, content={"error": "3D animasyon sayfası bulunamadı"})


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "landmark_count": len(LANDMARK_INDEX),
        "database": database_name(),
        "ui_file": UI_FILE.name,
        "ui_ready": UI_FILE.exists(),
        "stream_paths": ["/api/predict/live226"],
        "model_226": LIVE226.status(),
    }


# ═══════════════════════════════════════════════════════════
#  ROUTES — 3D avatar data
# ═══════════════════════════════════════════════════════════
@app.get("/signs")
async def list_signs():
    """List every word that has a landmark sequence."""
    words = sorted(LANDMARK_INDEX.keys())
    return {"words": words, "labels": {w: restore_turkish(w) for w in words}, "count": len(words)}

@app.get("/landmark/{word}")
async def get_landmark(word: str):
    """Return the (smoothed) landmark frames for one word."""
    key = word.lower().strip()
    path = LANDMARK_INDEX.get(key)
    if path is None:
        return Response(status_code=404, content=f"'{word}' bulunamadı")
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    if SMOOTHER_AVAILABLE and not raw_data.get("smoothed"):
        raw_data = smooth_landmark_data(raw_data)
    return Response(
        content=json.dumps(raw_data, ensure_ascii=False),
        media_type="application/json"
    )


# ═══════════════════════════════════════════════════════════
#  ROUTES — Auth
# ═══════════════════════════════════════════════════════════
@app.post("/api/register")
@app.post("/api/auth/register")
def register(data: RegisterData, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "-"
    if REGISTRATIONS_BY_IP.blocked(ip):
        return JSONResponse(status_code=429, content={"error": TOO_MANY_ATTEMPTS})
    REGISTRATIONS_BY_IP.hit(ip)
    email = data.email.lower().strip()
    if db.query(models.User).filter(models.User.email == email).first():
        return JSONResponse(status_code=400, content={"error": "Email zaten kayıtlı"})
    user = models.User(
        full_name=data.full_name.strip(), email=email,
        password_hash=hash_pw(data.password),
        role="User", status="Active", sessions=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    add_log_db(db, "Info", f"Yeni kullanıcı: {user.email}")
    return {"message": "Kayıt başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}

@app.post("/api/login")
@app.post("/api/auth/login")
def login(data: LoginData, request: Request, db: Session = Depends(get_db)):
    ip = request.client.host if request.client else "-"
    email = data.email.lower().strip()
    if LOGIN_FAILURES.blocked((ip, email)) or LOGIN_FAILURES_BY_IP.blocked(ip):
        return JSONResponse(status_code=429, content={"error": TOO_MANY_ATTEMPTS})
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or user.status != "Active" or not verify_pw(data.password, user.password_hash):
        LOGIN_FAILURES.hit((ip, email))
        LOGIN_FAILURES_BY_IP.hit(ip)
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    LOGIN_FAILURES.reset((ip, email))
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}", user.email)
    return {"message": "Giriş başarılı",
            "access_token": _issue_access_token(user.id),
            "token_type": "bearer",
            "expires_in": TOKEN_SECONDS,
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}


@app.post("/api/logout")
@app.post("/api/auth/logout")
def logout(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer)):
    if credentials:
        _access_tokens.pop(credentials.credentials, None)
    return {"message": "Çıkış yapıldı"}


# ═══════════════════════════════════════════════════════════
#  ROUTES — live recognition WebSockets
# ═══════════════════════════════════════════════════════════
SENTENCE_ACTIONS = ("approve", "reject", "undo", "reset_sentence", "clear_sentence")


AUTH_TIMEOUT_SECONDS = 10


async def _authenticate_socket(websocket: WebSocket):
    """Accept a live socket and read its first message, {"action": "auth", "token": ...}.

    Browsers cannot set headers on WebSocket requests. The token travels in the
    first message rather than the URL so it stays out of access logs. Returns
    (user id, email), or None after closing the socket.
    """
    await websocket.accept()
    token = None
    try:
        message = json.loads(await asyncio.wait_for(websocket.receive_text(), timeout=AUTH_TIMEOUT_SECONDS))
        if isinstance(message, dict) and message.get("action") == "auth":
            token = message.get("token")
    except WebSocketDisconnect:
        return None
    except Exception:
        pass
    db = next(get_db())
    try:
        user = _user_for_token(token, db)
        identity = (user.id, user.email) if user else None
    finally:
        db.close()
    if identity is None:
        await websocket.close(code=1008, reason="Oturum açmanız gerekiyor")
    return identity


class SentenceSession:
    """Approved words of one live connection.

    A prediction only becomes part of the sentence (and of History) after the
    user approves it. `candidate` is the top sign of the latest result when it
    is confident enough; `scores` keeps the confidences of that result so an
    approved alternative is stored with its own confidence.
    """

    def __init__(self, db: Session, user_id: int, mode: str):
        self.db = db
        self.user_id = user_id
        self.mode = mode
        self.session_id = uuid.uuid4().hex[:8]
        self.words: List[str] = []
        self.candidate: Optional[str] = None
        self.scores: Dict[str, float] = {}

    def present(self, scores: Dict[str, float], candidate: Optional[str] = None) -> None:
        """The result of one recorded sign replaces the previous one."""
        self.candidate = candidate
        self.scores = dict(scores)

    def pending(self) -> Dict[str, Any]:
        return {"pending_word": self.candidate,
                "pending_text": restore_turkish(self.candidate) if self.candidate else None}

    def handle(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply an approval action and return the reply for the client."""
        if action == "approve":
            word = (payload.get("word") or self.candidate or "").strip()
            if word:
                self.words.append(word)
                score = self.scores.get(word)
                self._save(word, format_confidence(score) if score is not None else "-")
            self._forget()
            return self._sentence(added=word or None)
        if action == "reject":
            self._forget()
            return {"type": "rejected"}
        if action == "undo":
            if self.words:
                self.words.pop()
            return self._sentence()
        self.words = []
        self._forget()
        return self._sentence()

    def _forget(self) -> None:
        self.candidate = None
        self.scores = {}

    def _sentence(self, **extra) -> Dict[str, Any]:
        return {"type": "sentence_update", "words": list(self.words),
                "display_words": restore_turkish_words(self.words),
                "sentence": assemble_sentence(self.words), **extra}

    def _save(self, word: str, confidence: str) -> None:
        try:
            self.db.add(models.History(user_id=self.user_id, session_id=self.session_id,
                                       mode=self.mode, result=word, confidence=confidence))
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            logger.warning(f"[DB] approved word save skipped: {e}")


MIN_CLIP_FRAMES = 8
MAX_CLIP_FRAMES = 96


@app.websocket("/api/predict/live226")
async def live226_predict(websocket: WebSocket):
    """One recorded sign at a time with the 226-class ensemble.

    After {"action": "auth", "token": ...} the client sends {"action": "start"},
    the clip's unmirrored frames as
    {"image": data_url}, then {"action": "end"}. Each frame is processed as it
    arrives ({"type": "progress"}); "end" returns {"type": "result"} with the
    top three signs. The approval actions (approve, reject, undo,
    reset_sentence) answer with {"type": "sentence_update"} or "rejected".
    """
    identity = await _authenticate_socket(websocket)
    if identity is None:
        return
    user_id, user_email = identity
    if not LIVE226.ready:
        await websocket.send_json({"type": "unavailable", **LIVE226.status()})
        await websocket.close(code=1013)
        return
    db = next(get_db())
    add_log_db(db, "Info", "226 canlı akış bağlandı", user_email)
    session = SentenceSession(db, user_id, mode="Live226")
    clip: List[np.ndarray] = []
    boxes = None
    try:
        while True:
            try:
                payload = json.loads(await websocket.receive_text())
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            action = payload.get("action")
            if action in SENTENCE_ACTIONS:
                await websocket.send_json(session.handle(action, payload))
            elif action == "start":
                clip, boxes = [], None
                await websocket.send_json({"type": "recording", "frame_count": LIVE226.frame_count})
            elif action == "end":
                if len(clip) < MIN_CLIP_FRAMES:
                    await websocket.send_json({"type": "result", "error": "Kayıt çok kısa", "frames": len(clip)})
                else:
                    result = await asyncio.to_thread(LIVE226.predict, clip)
                    top = result["top"]
                    session.present({row["gloss"]: row["confidence"] for row in top},
                                    candidate=top[0]["gloss"] if result["confident"] else None)
                    await websocket.send_json({"type": "result", "frames": len(clip), **result, **session.pending()})
                clip, boxes = [], None
            elif payload.get("image") and len(clip) < MAX_CLIP_FRAMES:
                frame = decode_base64_image(payload["image"])
                if frame is None:
                    continue
                # Detect on every Nth frame, and on every frame while nobody has been found
                found = boxes is not None and len(boxes) > 0
                reuse = boxes if found and len(clip) % LIVE226.detect_every else None
                landmarks, boxes = await asyncio.to_thread(LIVE226.landmarks, prepare_frame(frame), reuse)
                clip.append(landmarks)
                await websocket.send_json({"type": "progress", "processed": len(clip)})
    except WebSocketDisconnect:
        add_log_db(db, "Info", "226 canlı akış kapandı", user_email)
    except Exception as e:
        logger.error(f"[WS-226] Error: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"226 canlı akış hatası: {e}", user_email)
    finally:
        db.close()


# ═══════════════════════════════════════════════════════════
#  ROUTES — History / Settings / Dictionary / Admin
# ═══════════════════════════════════════════════════════════
def _history_row(r: "models.History") -> Dict[str, Any]:
    return {"id": r.id, "session_id": r.session_id, "mode": r.mode,
            "result": r.result, "text": restore_turkish(r.result), "confidence": r.confidence,
            "created_at": iso_time(r.created_at)}


def _average_confidence(rows) -> float:
    values = [v for v in (parse_confidence(r.confidence) for r in rows) if v is not None]
    return round(sum(values) / len(values), 1) if values else 0.0


def _model_status() -> Dict[str, Any]:
    return {
        "model_226": LIVE226.status(),
        "thresholds": LIVE226.config["thresholds"],
        "streams": {name: spec["weight"] for name, spec in LIVE226.config["streams"].items()},
        "landmark_words": len(LANDMARK_INDEX),
        "database": database_name(),
    }


@app.get("/api/history")
def get_history(db: Session = Depends(get_db), user=Depends(current_user)):
    rows = (db.query(models.History)
            .filter(models.History.user_id == user.id)
            .order_by(models.History.created_at.desc(), models.History.id.desc())
            .limit(100).all())
    return [_history_row(r) for r in rows]

@app.get("/api/me/summary")
def my_summary(db: Session = Depends(get_db), user=Depends(current_user)):
    rows = db.query(models.History).filter(models.History.user_id == user.id).all()
    return {"translations": len(rows),
            "avg_confidence": _average_confidence(rows),
            "sessions": user.sessions}

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db), user=Depends(current_user)):
    s = db.query(models.UserSetting).filter(models.UserSetting.user_id == user.id).first()
    if not s:
        return SettingsData().model_dump()
    return {"tts_enabled": bool(s.tts_enabled), "speech_rate": s.speech_rate,
            "avatar_speed": s.avatar_speed}

@app.post("/api/settings")
def save_settings(data: SettingsData, db: Session = Depends(get_db), user=Depends(current_user)):
    s = db.query(models.UserSetting).filter(models.UserSetting.user_id == user.id).first()
    if not s:
        s = models.UserSetting(user_id=user.id)
        db.add(s)
    for field, val in data.model_dump().items():
        setattr(s, field, val)
    db.commit()
    add_log_db(db, "Info", "Ayarlar güncellendi", user.email)
    return {"message": "Ayarlar kaydedildi", "data": data.model_dump()}

def _dictionary_entries() -> List[Dict[str, Any]]:
    """All 226 AUTSL signs with their Turkish spelling and English name."""
    return [{
        "class_id": entry["autsl_id"],
        "tr": entry["gloss"],
        "text": restore_turkish(entry["gloss"]),
        "en": entry.get("en"),
        "animation_key": entry["gloss"] if entry["gloss"] in LANDMARK_INDEX else None,
    } for entry in sorted(LIVE226.vocabulary, key=lambda e: e["autsl_id"])]

@app.get("/api/dictionary")
async def get_dictionary():
    return _dictionary_entries()

@app.get("/api/dictionary/{class_id}")
async def get_dictionary_by_id(class_id: int):
    for row in _dictionary_entries():
        if row["class_id"] == class_id:
            return row
    raise HTTPException(status_code=404, detail="Sınıf bulunamadı")

@app.get("/api/model/status")
async def model_status(_user=Depends(current_user)):
    return _model_status()

@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db), _admin=Depends(admin_user)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    return {
        "total_users": db.query(models.User).count(),
        "active_users": db.query(models.User).filter(models.User.status == "Active").count(),
        "translations": db.query(models.History).count(),
        "avg_confidence": _average_confidence(rows),
        "logs": db.query(models.Log).count(),
    }

@app.get("/api/admin/users")
def admin_users(db: Session = Depends(get_db), _admin=Depends(admin_user)):
    return [{"id": u.id, "name": u.full_name, "email": u.email,
             "role": u.role, "sessions": u.sessions, "status": u.status}
            for u in db.query(models.User).all()]

@app.get("/api/admin/logs")
def admin_logs(db: Session = Depends(get_db), _admin=Depends(admin_user)):
    return [{"id": l.id, "level": l.level, "message": l.message, "user": l.user,
             "timestamp": iso_time(l.created_at)}
            for l in db.query(models.Log).order_by(
                models.Log.created_at.desc(), models.Log.id.desc()).limit(100).all()]

@app.get("/api/admin/model")
async def admin_model(_admin=Depends(admin_user)):
    return _model_status()

@app.get("/api/debug/model-check")
async def debug_model_check(_admin=Depends(admin_user)):
    return {
        "model_226": LIVE226.status(),
        "missing_files": missing_files(LIVE226.config),
        "extractor": LIVE226.config["extractor"],
        "landmark_words": len(LANDMARK_INDEX),
        "smoother_available": SMOOTHER_AVAILABLE,
    }


# ═══════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════
def bootstrap_admin(db: Session) -> None:
    """Create the first administrator from environment variables, if set."""
    email = os.environ.get("SIGNTURK_ADMIN_EMAIL", "").strip().lower()
    password = os.environ.get("SIGNTURK_ADMIN_PASSWORD", "")
    if not email or not password:
        return
    if db.query(models.User).filter(models.User.email == email).first():
        return
    if len(password) < 8:
        raise RuntimeError("SIGNTURK_ADMIN_PASSWORD must be at least 8 characters")
    db.add(models.User(
        full_name=os.environ.get("SIGNTURK_ADMIN_NAME", "SignTurk Admin").strip(),
        email=email,
        password_hash=hash_pw(password),
        role="Admin",
        status="Active",
        sessions=0,
    ))
    db.commit()
    logger.info("[DB] Administrator account created from environment")


def ensure_history_user_column() -> None:
    """Add history.user_id to databases created before per-user history."""
    columns = {col["name"] for col in inspect(engine).get_columns("history")}
    if "user_id" not in columns:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE history ADD COLUMN user_id INTEGER"))
        logger.info("[DB] Added history.user_id column")


async def startup_event():
    Base.metadata.create_all(bind=engine)
    ensure_history_user_column()
    logger.info("[DB] Tables ready (%s)", database_name())

    db = next(get_db())
    try:
        bootstrap_admin(db)
        add_log_db(db, "Info", "Sunucu başlatıldı", "system")
    finally:
        db.close()

    load_landmark_index()
    LIVE226.start_loading()

    logger.info("=" * 60)
    logger.info("  SignTurk backend ready")
    logger.info(f"  Landmarks:  {len(LANDMARK_INDEX)} words")
    logger.info(f"  226 model:  {LIVE226.state.upper()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
