"""SignTurk FastAPI application.

Serves the web interface, account and history APIs, live sign recognition
over WebSocket, sentence assembly, and the landmark data that drives the
3D avatar.
"""

import asyncio, os, json, time, traceback, uuid, logging, base64, shutil, tempfile, secrets
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

# Load variables from a local .env file when python-dotenv is installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import cv2

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
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
from live_pipeline import (
    count_detected_hands as count_live_hands,
    hands_to_landmarks,
    preprocess_sequence as preprocess_live_sequence,
    resample_window,
)

# ── Landmark smoothing ───────────────────────────────────
try:
    from landmark_smoother import smooth_landmark_data
    SMOOTHER_AVAILABLE = True
except ImportError:
    SMOOTHER_AVAILABLE = False

# ── TensorFlow ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
try:
    tf.get_logger().setLevel("ERROR")
except AttributeError:
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

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
MODEL_DIR    = BASE_DIR / "demo_assets_179"
UI_FILE      = FRONTEND_DIR / "signturk_ui.html"
AVATAR_FILE  = FRONTEND_DIR / "avatar3d.html"

# ═══════════════════════════════════════════════════════════
#  GLOBALS — live recognition model (179 classes)
# ═══════════════════════════════════════════════════════════
MODEL        = None
LABEL_MAP    = {}
NORM_MEAN    = None
NORM_STD     = None
DEMO_CONFIG  = {}
MP_HANDS     = None

SEQ_LEN      = 16   # model input length (frames)
CAPTURE_WINDOW = 32 # frames collected from the camera, resampled to SEQ_LEN
FEAT_DIM     = 156
NUM_CLASSES  = 179
CONFIDENCE_THRESHOLD = 0.40
MODEL_LOAD_TIME  = 0.0
AVG_INFERENCE_MS = 0.0
INFERENCE_COUNT  = 0

MIN_HAND_NORM = 1e-6
MEDIAPIPE_DETECTION_CONFIDENCE = 0.15
MEDIAPIPE_TRACKING_CONFIDENCE = 0.15
MEDIAPIPE_STATIC_IMAGE_MODE = False
MEDIAPIPE_MODEL_COMPLEXITY = 1
MAX_MISSED_HAND_FRAMES = 8
LIVE_BUFFER_STRIDE = 0

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
#  MEDIAPIPE INIT
# ═══════════════════════════════════════════════════════════
def prepare_mediapipe_resource_path(mp_module):
    """Copy MediaPipe graph files to an ASCII-only path when needed.

    MediaPipe fails to open its resources when the package lives under a path
    with non-ASCII characters (common in Windows user folders).
    """
    try:
        package_root = Path(mp_module.__file__).resolve().parent
        str(package_root).encode("ascii")
        return
    except UnicodeEncodeError:
        pass
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Resource path check failed: {e}")
        return

    try:
        from mediapipe.python import solution_base

        ascii_root = Path(tempfile.gettempdir()) / "signturk_mediapipe_site"
        modules_src = package_root / "modules"
        modules_dst = ascii_root / "mediapipe" / "modules"
        hand_graph = modules_dst / "hand_landmark" / "hand_landmark_tracking_cpu.binarypb"

        if not hand_graph.exists():
            modules_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(modules_src, modules_dst, dirs_exist_ok=True)

        fake_solution = ascii_root / "mediapipe" / "python" / "solution_base.py"
        fake_solution.parent.mkdir(parents=True, exist_ok=True)
        fake_solution.touch(exist_ok=True)
        solution_base.__file__ = str(fake_solution)
        logger.info(f"[MEDIAPIPE] Using ASCII resource root: {ascii_root}")
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] ASCII resource workaround failed: {e}")


def init_mediapipe():
    """Create a MediaPipe Hands instance, trying each known import layout."""
    try:
        import mediapipe as mp
        prepare_mediapipe_resource_path(mp)
        hands = mp.solutions.hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 1 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 1 failed: {e}")

    try:
        from mediapipe.python.solutions import hands as mp_hands
        hands = mp_hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 2 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 2 failed: {e}")

    try:
        import importlib
        import mediapipe
        hands_module = getattr(mediapipe, "solutions", None)
        if hands_module is None:
            hands_module = importlib.import_module("mediapipe.solutions")
        hands = hands_module.hands.Hands(
            static_image_mode=MEDIAPIPE_STATIC_IMAGE_MODE,
            max_num_hands=2,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_TRACKING_CONFIDENCE,
        )
        hands.process(np.zeros((100, 100, 3), dtype=np.uint8))
        logger.info("[MEDIAPIPE] Strategy 3 OK")
        return hands
    except Exception as e:
        logger.warning(f"[MEDIAPIPE] Strategy 3 failed: {e}")

    logger.error("[MEDIAPIPE] All strategies failed")
    return None


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════
def load_model_assets():
    """Load the 179-class live model and its preprocessing assets."""
    global MODEL, LABEL_MAP, NORM_MEAN, NORM_STD, DEMO_CONFIG, MP_HANDS
    global SEQ_LEN, CAPTURE_WINDOW, FEAT_DIM, NUM_CLASSES, CONFIDENCE_THRESHOLD, MODEL_LOAD_TIME

    t0 = time.time()

    cfg_path = MODEL_DIR / "demo_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            DEMO_CONFIG = json.load(f)
        SEQ_LEN              = DEMO_CONFIG.get("seq_len", 16)
        CAPTURE_WINDOW       = DEMO_CONFIG.get("capture_window", 32)
        FEAT_DIM             = DEMO_CONFIG.get("feat_dim", 156)
        NUM_CLASSES          = DEMO_CONFIG.get("num_classes", 179)
        CONFIDENCE_THRESHOLD = DEMO_CONFIG.get("confidence_threshold", 0.40)
        logger.info(f"[CONFIG] seq_len={SEQ_LEN} capture_window={CAPTURE_WINDOW} feat_dim={FEAT_DIM} num_classes={NUM_CLASSES}")

    lm_path = MODEL_DIR / "label_map.json"
    if lm_path.exists():
        with open(lm_path, encoding="utf-8") as f:
            LABEL_MAP = json.load(f)
        logger.info(f"[LABELS] Loaded {len(LABEL_MAP)} classes")

    ns_path = MODEL_DIR / "norm_stats.json"
    if ns_path.exists():
        with open(ns_path, encoding="utf-8") as f:
            ns = json.load(f)
        NORM_MEAN = np.array(ns["mean"], dtype=np.float32)
        NORM_STD  = np.where(np.array(ns["std"], dtype=np.float32) < 1e-6, 1.0,
                             np.array(ns["std"], dtype=np.float32))

    model_path = MODEL_DIR / "model.keras"
    if model_path.exists():
        try:
            @tf.keras.utils.register_keras_serializable()
            class ReduceSumAxis1(tf.keras.layers.Layer):
                def call(self, inputs):
                    return tf.reduce_sum(inputs, axis=1)
                def compute_output_shape(self, input_shape):
                    return (input_shape[0], input_shape[2])

            MODEL = tf.keras.models.load_model(str(model_path), compile=False)
            MODEL.predict(np.zeros((1, SEQ_LEN, FEAT_DIM), dtype=np.float32), verbose=0)
            logger.info(f"[MODEL] Loaded {MODEL.input_shape} -> {MODEL.output_shape}")
        except Exception as e:
            logger.error(f"[MODEL] Could not load model: {e}")
    else:
        logger.warning(f"[MODEL] {model_path} missing; run: python tools/download_models.py")

    MP_HANDS = init_mediapipe()
    MODEL_LOAD_TIME = time.time() - t0
    logger.info(f"[STARTUP] Live assets loaded in {MODEL_LOAD_TIME:.2f}s")


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
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════
def extract_landmarks_from_frame(frame: np.ndarray, mirror: bool = False) -> Optional[np.ndarray]:
    """Return 126 hand coordinates in the training layout for one BGR frame."""
    if MP_HANDS is None:
        return None
    try:
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = MP_HANDS.process(rgb)
        hands = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                hands.append((hand.classification[0].label,
                              [[lm.x, lm.y, lm.z] for lm in hl.landmark]))
        return hands_to_landmarks(hands)
    except Exception as e:
        logger.error(f"[MEDIAPIPE] Frame error: {e}")
        return np.zeros(126, dtype=np.float32)


def count_detected_hands(landmarks: np.ndarray) -> int:
    return count_live_hands(landmarks, MIN_HAND_NORM)


def preprocess_sequence(raw_landmarks):
    """Resample a capture window to the model length and build the features."""
    return preprocess_live_sequence(resample_window(raw_landmarks, SEQ_LEN), NORM_MEAN, NORM_STD)


# ═══════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════
def get_label(model_index: int) -> Dict[str, Any]:
    key = str(model_index)
    if key in LABEL_MAP:
        e = LABEL_MAP[key]
        gloss = e.get("TR", f"class_{model_index}")
        return {"class_id": model_index,
                "label_tr": gloss,
                "text_tr": restore_turkish(gloss),
                "label_en": e.get("EN", f"class_{model_index}").replace("_", " ")}
    return {"class_id": model_index,
            "label_tr": f"class_{model_index}",
            "text_tr": f"class_{model_index}",
            "label_en": f"class_{model_index}"}


def run_inference(input_tensor):
    global AVG_INFERENCE_MS, INFERENCE_COUNT
    if MODEL is None:
        return None
    t0 = time.time()
    proba = MODEL.predict(input_tensor, verbose=0)
    elapsed_ms = (time.time() - t0) * 1000
    INFERENCE_COUNT += 1
    AVG_INFERENCE_MS = (AVG_INFERENCE_MS * (INFERENCE_COUNT-1) + elapsed_ms) / INFERENCE_COUNT

    top_indices = np.argsort(proba[0])[::-1][:5]
    best_idx = int(top_indices[0])
    best_conf = float(proba[0][best_idx])
    best_lbl = get_label(best_idx)

    return {
        "class_id": best_lbl["class_id"],
        "label_tr": best_lbl["label_tr"],
        "text_tr": best_lbl["text_tr"],
        "label_en": best_lbl["label_en"],
        "confidence": best_conf,
        "animation_key": best_lbl["label_tr"].lower() if best_lbl["label_tr"].lower() in LANDMARK_INDEX else "none",
        "category": "Sign",
        "above_threshold": best_conf >= CONFIDENCE_THRESHOLD,
        "top_predictions": [
            {"class_id": int(i), **get_label(int(i)),
             "confidence": round(float(proba[0][i]), 4)}
            for i in top_indices[:3]
        ],
    }


def run_best_inference(sequence: np.ndarray):
    return run_inference(preprocess_sequence(sequence))


# ═══════════════════════════════════════════════════════════
#  FRAME BUFFER — live WebSocket
# ═══════════════════════════════════════════════════════════
class FrameBuffer:
    def __init__(self, seq_len=16):
        self.seq_len = seq_len
        self.buffer: List[np.ndarray] = []

    def clear(self):
        self.buffer.clear()

    def keep_tail(self, n):
        if n <= 0:
            self.clear()
        elif len(self.buffer) > n:
            self.buffer = self.buffer[-n:]

    def add(self, landmarks):
        self.buffer.append(landmarks)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)
        return np.stack(self.buffer, axis=0) if len(self.buffer) == self.seq_len else None

    def __len__(self):
        return len(self.buffer)


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
app = FastAPI(title="SignTurk API", version="3.1.0")

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
# Bearer tokens live in memory: restarting the server signs everyone out.
# A multi-instance deployment needs signed, expiring tokens or a shared store.
_bearer = HTTPBearer(auto_error=False)
_access_tokens: Dict[str, int] = {}


def _issue_access_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    _access_tokens[token] = user_id
    return token


def _user_for_token(token: Optional[str], db: Session):
    user_id = _access_tokens.get(token or "")
    if not user_id:
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
        "model_loaded": MODEL is not None,
        "mediapipe_ready": MP_HANDS is not None,
        "landmark_count": len(LANDMARK_INDEX),
        "database": database_name(),
        "seq_len": SEQ_LEN,
        "capture_window": CAPTURE_WINDOW,
        "feat_dim": FEAT_DIM,
        "num_classes": NUM_CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ui_file": UI_FILE.name,
        "ui_ready": UI_FILE.exists(),
        "stream_paths": ["/api/predict/live", "/ws/stream", "/api/predict/live226"],
        "model_226": LIVE226.status(),
        "mediapipe_detection_threshold": MEDIAPIPE_DETECTION_CONFIDENCE,
        "mediapipe_tracking_threshold": MEDIAPIPE_TRACKING_CONFIDENCE,
        "mediapipe_static_image_mode": MEDIAPIPE_STATIC_IMAGE_MODE,
        "mediapipe_model_complexity": MEDIAPIPE_MODEL_COMPLEXITY,
        "missed_hand_grace_frames": MAX_MISSED_HAND_FRAMES,
        "live_buffer_stride": LIVE_BUFFER_STRIDE,
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
def register(data: RegisterData, db: Session = Depends(get_db)):
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
def login(data: LoginData, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == data.email.lower().strip()).first()
    if not user or user.status != "Active" or not verify_pw(data.password, user.password_hash):
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}", user.email)
    return {"message": "Giriş başarılı",
            "access_token": _issue_access_token(user.id),
            "token_type": "bearer",
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


async def _authenticate_socket(websocket: WebSocket):
    """(user id, email) for the ?token= of a WebSocket, or None after closing it.

    Browsers cannot set headers on WebSocket requests, so the token comes in
    the query string.
    """
    db = next(get_db())
    try:
        user = _user_for_token(websocket.query_params.get("token"), db)
        identity = (user.id, user.email) if user else None
    finally:
        db.close()
    if identity is None:
        await websocket.close(code=1008, reason="Oturum açmanız gerekiyor")
    return identity


class SentenceSession:
    """Approved words of one live connection.

    A prediction only becomes part of the sentence (and of History) after the
    user approves it. `candidate` is the latest confident prediction waiting
    for that decision; `scores` keeps the confidences of the latest result so
    an approved alternative is stored with its own confidence.
    """

    def __init__(self, db: Session, user_id: int, mode: str):
        self.db = db
        self.user_id = user_id
        self.mode = mode
        self.session_id = uuid.uuid4().hex[:8]
        self.words: List[str] = []
        self.candidate: Optional[str] = None
        self.scores: Dict[str, float] = {}

    def offer(self, word: str, confidence: float) -> None:
        """A confident live prediction becomes the candidate."""
        self.candidate = word
        self.scores = {word: confidence}

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


@app.websocket("/ws/stream")
@app.websocket("/api/predict/live")
async def live_predict(websocket: WebSocket):
    identity = await _authenticate_socket(websocket)
    if identity is None:
        return
    user_id, user_email = identity
    await websocket.accept()
    logger.info("[WS-LIVE] Connected")
    db = next(get_db())
    add_log_db(db, "Info", "Canlı akış bağlandı", user_email)
    buf = FrameBuffer(seq_len=CAPTURE_WINDOW)
    missed_hand_frames = 0
    session = SentenceSession(db, user_id, mode="Live")
    try:
        while True:
            incoming = await websocket.receive_text()
            landmarks = None
            image_payload = None
            mirror_frame = False
            try:
                payload = json.loads(incoming)
                action = payload.get("action") if isinstance(payload, dict) else None
                if action:
                    if action in SENTENCE_ACTIONS:
                        await websocket.send_json(session.handle(action, payload))
                    continue
                if "landmarks" in payload:
                    arr = np.array(payload["landmarks"], dtype=np.float32)
                    if arr.shape == (126,):
                        landmarks = arr
                else:
                    image_payload = (
                        payload.get("image")
                        or payload.get("frame")
                        or payload.get("data_url")
                        or payload.get("dataUrl")
                    )
                    mirror_frame = bool(payload.get("mirror", False))
            except (json.JSONDecodeError, ValueError):
                pass

            if landmarks is None:
                frame = decode_base64_image(image_payload or incoming)
                if frame is None:
                    await websocket.send_json({"class_id": -1, "label_tr": "-",
                        "label_en": "-", "confidence": 0, "animation_key": "none"})
                    continue
                landmarks = extract_landmarks_from_frame(frame, mirror=mirror_frame)
                if landmarks is None:
                    await websocket.send_json({"class_id": -1,
                        "label_tr": "MediaPipe kullanılamıyor",
                        "label_en": "Install mediapipe",
                        "confidence": 0, "animation_key": "none"})
                    continue

            hands_detected = count_detected_hands(landmarks)
            if hands_detected == 0:
                missed_hand_frames += 1
                if missed_hand_frames >= MAX_MISSED_HAND_FRAMES:
                    buf.clear()
                await websocket.send_json({
                    "class_id": -1,
                    "label_tr": "Takip korunuyor" if len(buf) else "El algılanmadı",
                    "label_en": "Tracking hold" if len(buf) else "No hand detected",
                    "confidence": 0,
                    "animation_key": "none",
                    "category": "-",
                    "hands_detected": 0,
                    "buffer_length": len(buf),
                    "missed_hand_frames": missed_hand_frames,
                })
                continue

            missed_hand_frames = 0
            sequence = buf.add(landmarks)
            if sequence is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": f"Kare toplanıyor ({len(buf)}/{CAPTURE_WINDOW})",
                    "label_en": "Collecting...",
                    "confidence": len(buf)/CAPTURE_WINDOW,
                    "animation_key": "none",
                    "category": "-",
                    "hands_detected": hands_detected,
                    "buffer_length": len(buf)})
                continue

            if MODEL is None:
                await websocket.send_json({"class_id": -1,
                    "label_tr": "Model yüklenmedi", "label_en": "Model not loaded",
                    "confidence": 0, "animation_key": "none"})
                continue

            prediction = run_best_inference(sequence)
            if prediction is None:
                continue

            result = {
                "class_id": prediction["class_id"],
                "label_tr": prediction["label_tr"] if prediction["above_threshold"]
                            else f"({prediction['label_tr']}?)",
                "text_tr": prediction["text_tr"],
                "label_en": prediction["label_en"],
                "confidence": prediction["confidence"],
                "animation_key": prediction["animation_key"],
                "category": prediction["category"],
                "top_predictions": prediction.get("top_predictions", []),
                "above_threshold": prediction["above_threshold"],
                "threshold": CONFIDENCE_THRESHOLD,
                "variant": "normal",
                "hands_detected": hands_detected,
                "buffer_length": len(buf),
            }

            if prediction["above_threshold"]:
                session.offer(prediction["label_tr"], prediction["confidence"])
            result.update(session.pending())

            await websocket.send_json(result)
            buf.keep_tail(LIVE_BUFFER_STRIDE)

    except WebSocketDisconnect:
        logger.info("[WS-LIVE] Disconnected")
        add_log_db(db, "Info", "Canlı akış kapandı", user_email)
    except Exception as e:
        logger.error(f"[WS-LIVE] Error: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"Canlı akış hatası: {e}", user_email)
    finally:
        db.close()


MIN_CLIP_FRAMES = 8
MAX_CLIP_FRAMES = 96


@app.websocket("/api/predict/live226")
async def live226_predict(websocket: WebSocket):
    """One recorded sign at a time with the 226-class ensemble.

    The client sends {"action": "start"}, the clip's unmirrored frames as
    {"image": data_url}, then {"action": "end"}. Each frame is processed as it
    arrives ({"type": "progress"}); "end" returns {"type": "result"} with the
    top three signs. Approval actions work as on /api/predict/live.
    """
    identity = await _authenticate_socket(websocket)
    if identity is None:
        return
    user_id, user_email = identity
    await websocket.accept()
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
                reuse = boxes if len(clip) % LIVE226.detect_every else None
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


@app.post("/api/predict/sequence")
async def predict_sequence(payload: Dict[str, Any], _user=Depends(current_user)):
    try:
        raw = np.array(payload["landmarks"], dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 126 or raw.shape[0] not in (SEQ_LEN, CAPTURE_WINDOW):
            return JSONResponse(status_code=400,
                content={"error": f"Beklenen ({SEQ_LEN} veya {CAPTURE_WINDOW}, 126), gelen {list(raw.shape)}"})
        result = run_best_inference(raw)
        if result is None:
            return JSONResponse(status_code=503, content={"error": "Model yüklenmedi"})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


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
        "model_loaded": MODEL is not None,
        "live_model": "Yüklü" if MODEL else "Yüklenmedi",
        "num_classes": NUM_CLASSES,
        "seq_len": SEQ_LEN,
        "capture_window": CAPTURE_WINDOW,
        "feat_dim": FEAT_DIM,
        "label_map_size": len(LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "mediapipe_ready": MP_HANDS is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "avg_inference": f"{AVG_INFERENCE_MS:.0f} ms" if INFERENCE_COUNT else "Henüz yok",
        "database": database_name(),
        "model_226": LIVE226.status(),
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
    """All 226 AUTSL signs; English names exist for the 179 in the live model."""
    live = {info.get("TR"): (int(index), info) for index, info in LABEL_MAP.items()}
    rows = []
    for entry in sorted(LIVE226.vocabulary, key=lambda e: e["autsl_id"]):
        gloss = entry["gloss"]
        index, info = live.get(gloss, (None, {}))
        rows.append({
            "class_id": entry["autsl_id"],
            "tr": gloss,
            "text": restore_turkish(gloss),
            "en": info.get("EN", "").replace("_", " ") or None,
            "live_179_id": index,
            "animation_key": gloss if gloss in LANDMARK_INDEX else None,
        })
    return rows

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
        "live_model_loaded": MODEL is not None,
        "mediapipe_ready": MP_HANDS is not None,
        "label_map_entries": len(LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "norm_stats_loaded": NORM_MEAN is not None,
        "smoother_available": SMOOTHER_AVAILABLE,
        "config": DEMO_CONFIG,
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


@app.on_event("startup")
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

    load_model_assets()
    load_landmark_index()
    LIVE226.start_loading()

    logger.info("=" * 60)
    logger.info("  SignTurk backend ready")
    logger.info(f"  Model:      {'LOADED' if MODEL else 'MISSING'}")
    logger.info(f"  Landmarks:  {len(LANDMARK_INDEX)} words")
    logger.info(f"  MediaPipe:  {'READY' if MP_HANDS else 'MISSING'}")
    logger.info(f"  226 model:  {LIVE226.state.upper()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
