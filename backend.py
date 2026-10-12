"""SignTurk FastAPI application.

Serves the web interface, account and history APIs, live sign recognition
over WebSocket, sentence assembly, and the landmark data that drives the
3D avatar.
"""

import os, json, time, traceback, uuid, logging, base64, shutil, tempfile
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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from database import engine, get_db, Base
import models

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

SEQ_LEN      = 16
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
MAX_MISSED_HAND_FRAMES = 6
LIVE_BUFFER_STRIDE = 0

# Per-word landmark sequences that drive the 3D avatar
_candidate_landmark_dirs = [
    BASE_DIR / "landmarks",
    BASE_DIR / "dataset" / "landmarks",
]
LANDMARKS_DIR  = None
LANDMARK_INDEX = {}

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
    global SEQ_LEN, FEAT_DIM, NUM_CLASSES, CONFIDENCE_THRESHOLD, MODEL_LOAD_TIME

    t0 = time.time()

    cfg_path = MODEL_DIR / "demo_config.json"
    if cfg_path.exists():
        with open(cfg_path, encoding="utf-8") as f:
            DEMO_CONFIG = json.load(f)
        SEQ_LEN              = DEMO_CONFIG.get("seq_len", 16)
        FEAT_DIM             = DEMO_CONFIG.get("feat_dim", 156)
        NUM_CLASSES          = DEMO_CONFIG.get("num_classes", 179)
        CONFIDENCE_THRESHOLD = DEMO_CONFIG.get("confidence_threshold", 0.40)
        logger.info(f"[CONFIG] seq_len={SEQ_LEN} feat_dim={FEAT_DIM} num_classes={NUM_CLASSES}")

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
    """Return 126 hand coordinates (left 63 + right 63) for one BGR frame."""
    if MP_HANDS is None:
        return None
    try:
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = MP_HANDS.process(rgb)
        left_lm = np.zeros(63, dtype=np.float32)
        right_lm = np.zeros(63, dtype=np.float32)
        if results.multi_hand_landmarks and results.multi_handedness:
            for hl, hand in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = hand.classification[0].label
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark],
                                   dtype=np.float32).flatten()
                if label == "Left":
                    left_lm = coords
                else:
                    right_lm = coords
        return np.concatenate([left_lm, right_lm])
    except Exception as e:
        logger.error(f"[MEDIAPIPE] Frame error: {e}")
        return np.zeros(126, dtype=np.float32)


def hand_is_present(hand_vec: np.ndarray) -> bool:
    return bool(np.linalg.norm(hand_vec) > MIN_HAND_NORM)


def count_detected_hands(landmarks: np.ndarray) -> int:
    if landmarks is None or landmarks.shape != (126,):
        return 0
    return int(hand_is_present(landmarks[:63])) + int(hand_is_present(landmarks[63:]))


def normalize_hands_relative(X):
    result = X.copy()
    T = result.shape[0]
    for start in [0, 63]:
        hand = result[:, start:start+63].reshape(T, 21, 3)
        hand_rel = hand - hand[:, 0:1, :]
        scale = np.linalg.norm(hand_rel[:, 9, :], axis=-1, keepdims=True)[:, :, np.newaxis]
        scale = np.where(scale < 1e-6, 1.0, scale)
        result[:, start:start+63] = (hand_rel / scale).reshape(T, 63)
    return result


def compute_finger_angles(X):
    T = X.shape[0]
    chains = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]]
    angles_all = []
    for hs in [0, 63]:
        hand = X[:, hs:hs+63].reshape(T, 21, 3)
        ha = np.zeros((T, 15), dtype=np.float32)
        idx = 0
        for chain in chains:
            for i in range(len(chain) - 1):
                a = hand[:, (0 if i == 0 else chain[i-1]), :]
                b = hand[:, chain[i], :]
                c = hand[:, chain[i+1], :]
                v1, v2 = a - b, c - b
                cos_a = np.sum(v1*v2, axis=-1) / (
                    np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
                ha[:, idx] = np.arccos(np.clip(cos_a, -1, 1))
                idx += 1
        angles_all.append(ha)
    return np.concatenate([X, np.concatenate(angles_all, axis=-1)], axis=-1)


def zscore_normalize(X):
    if NORM_MEAN is None or NORM_STD is None:
        return X
    return (X - NORM_MEAN) / NORM_STD


def preprocess_sequence(raw_landmarks):
    x = normalize_hands_relative(raw_landmarks)
    x = compute_finger_angles(x)
    x = zscore_normalize(x)
    return x[np.newaxis, :, :].astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════
def get_label(model_index: int) -> Dict[str, Any]:
    key = str(model_index)
    if key in LABEL_MAP:
        e = LABEL_MAP[key]
        return {"class_id": model_index,
                "label_tr": e.get("TR", f"class_{model_index}"),
                "label_en": e.get("EN", f"class_{model_index}")}
    return {"class_id": model_index,
            "label_tr": f"class_{model_index}",
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
        "label_en": best_lbl["label_en"],
        "confidence": best_conf,
        "animation_key": f"{best_lbl['label_en']}_sign",
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


# ── Schemas ───────────────────────────────────────────────
class SettingsData(BaseModel):
    camera: str
    voice: str
    speech_rate: float
    avatar_speed: float
    tts_enabled: bool = True
    notifications_enabled: bool = True
    avatar_enabled: bool = True
    websocket_enabled: bool = True

class RegisterData(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class LoginData(BaseModel):
    email: EmailStr
    password: str


# ═══════════════════════════════════════════════════════════
#  APP SETUP
# ═══════════════════════════════════════════════════════════
app = FastAPI(title="SignTurk API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

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
    app.include_router(text_processing_router)
    app.include_router(text_admin_router)
    logger.info("[TEXT] /api/text/* routes mounted")
except Exception as e:
    logger.warning(f"[TEXT] /api/text/* routes not mounted (optional deps?): {e}")


def assemble_sentence(words: List[str]) -> str:
    """Build a Turkish sentence from approved sign words (rule-based engine)."""
    cleaned = [w for w in (words or []) if w and str(w).strip()]
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
        "database": "Supabase PostgreSQL",
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "num_classes": NUM_CLASSES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ui_file": UI_FILE.name,
        "ui_ready": UI_FILE.exists(),
        "stream_paths": ["/api/predict/live", "/ws/stream"],
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
    return {"words": sorted(LANDMARK_INDEX.keys()), "count": len(LANDMARK_INDEX)}

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
    if db.query(models.User).filter(models.User.email == data.email.lower()).first():
        return JSONResponse(status_code=400, content={"error": "Email zaten kayıtlı"})
    user = models.User(
        full_name=data.full_name, email=data.email.lower(),
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
    user = db.query(models.User).filter(models.User.email == data.email.lower()).first()
    if not user or not verify_pw(data.password, user.password_hash):
        return JSONResponse(status_code=401, content={"error": "Hatalı email veya şifre"})
    user.sessions += 1
    db.commit()
    add_log_db(db, "Success", f"Giriş: {user.email}")
    return {"message": "Giriş başarılı",
            "user": {"id": user.id, "full_name": user.full_name,
                     "email": user.email, "role": user.role}}


# ═══════════════════════════════════════════════════════════
#  ROUTES — live recognition WebSocket
# ═══════════════════════════════════════════════════════════
@app.websocket("/ws/stream")
@app.websocket("/api/predict/live")
async def live_predict(websocket: WebSocket):
    await websocket.accept()
    logger.info("[WS-LIVE] Connected")
    db = next(get_db())
    add_log_db(db, "Info", "Canlı akış bağlandı")
    buf = FrameBuffer(seq_len=SEQ_LEN)
    missed_hand_frames = 0
    # A prediction only becomes part of the sentence (and of History) after
    # the user approves it. last_candidate holds the latest confident
    # prediction that is waiting for that decision.
    session_id = uuid.uuid4().hex[:8]
    sentence_words: List[str] = []
    last_candidate: Optional[str] = None
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
                    # ── approval control channel ───────────────
                    if action == "approve":
                        word = (payload.get("word") or last_candidate or "").strip()
                        if word:
                            sentence_words.append(word)
                            try:
                                db.add(models.History(
                                    session_id=session_id, mode="Live",
                                    result=word, confidence="approved",
                                ))
                                db.commit()
                            except Exception as e:
                                db.rollback()
                                logger.warning(f"[DB] approved word save skipped: {e}")
                        last_candidate = None
                        await websocket.send_json({
                            "type": "sentence_update",
                            "words": list(sentence_words),
                            "sentence": assemble_sentence(sentence_words),
                            "added": word or None,
                        })
                    elif action == "reject":
                        last_candidate = None
                        await websocket.send_json({"type": "rejected"})
                    elif action == "undo":
                        if sentence_words:
                            sentence_words.pop()
                        await websocket.send_json({
                            "type": "sentence_update",
                            "words": list(sentence_words),
                            "sentence": assemble_sentence(sentence_words),
                        })
                    elif action in ("reset_sentence", "clear_sentence"):
                        sentence_words = []
                        last_candidate = None
                        await websocket.send_json({
                            "type": "sentence_update", "words": [], "sentence": "",
                        })
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
                    "label_tr": f"Kare toplanıyor ({len(buf)}/{SEQ_LEN})",
                    "label_en": "Collecting...",
                    "confidence": len(buf)/SEQ_LEN,
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

            # Remember the latest confident prediction as the approval
            # candidate; nothing is stored until the user approves it.
            if prediction["above_threshold"]:
                last_candidate = prediction["label_tr"]
            result["pending_word"] = last_candidate

            await websocket.send_json(result)
            buf.keep_tail(LIVE_BUFFER_STRIDE)

    except WebSocketDisconnect:
        logger.info("[WS-LIVE] Disconnected")
        add_log_db(db, "Info", "Canlı akış kapandı")
    except Exception as e:
        logger.error(f"[WS-LIVE] Error: {e}\n{traceback.format_exc()}")
        add_log_db(db, "Warning", f"Canlı akış hatası: {e}")
    finally:
        db.close()


@app.post("/api/predict/sequence")
async def predict_sequence(payload: Dict[str, Any]):
    try:
        raw = np.array(payload["landmarks"], dtype=np.float32)
        if raw.shape != (SEQ_LEN, 126):
            return JSONResponse(status_code=400,
                content={"error": f"Beklenen ({SEQ_LEN}, 126), gelen {list(raw.shape)}"})
        result = run_best_inference(raw)
        if result is None:
            return JSONResponse(status_code=503, content={"error": "Model yüklenmedi"})
        return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════
#  ROUTES — History / Settings / Dictionary / Admin
# ═══════════════════════════════════════════════════════════
@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    return [{"id": r.session_id, "mode": r.mode, "result": r.result,
             "conf": r.confidence,
             "time": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else ""}
            for r in rows]

@app.get("/api/settings")
def get_settings(db: Session = Depends(get_db)):
    s = db.query(models.Setting).first()
    if not s:
        return {"camera": "Default Camera", "voice": "Female Voice",
                "speech_rate": 1.0, "avatar_speed": 1.0,
                "tts_enabled": True, "notifications_enabled": True,
                "avatar_enabled": True, "websocket_enabled": True}
    return {"camera": s.camera, "voice": s.voice,
            "speech_rate": s.speech_rate, "avatar_speed": s.avatar_speed,
            "tts_enabled": s.tts_enabled, "notifications_enabled": s.notifications_enabled,
            "avatar_enabled": s.avatar_enabled, "websocket_enabled": s.websocket_enabled}

@app.post("/api/settings")
def save_settings(data: SettingsData, db: Session = Depends(get_db)):
    s = db.query(models.Setting).first()
    if not s:
        s = models.Setting(id=1)
        db.add(s)
    for field, val in data.dict().items():
        setattr(s, field, val)
    db.commit()
    add_log_db(db, "Info", "Ayarlar güncellendi", "admin")
    return {"message": "Ayarlar kaydedildi", "data": data.dict()}

@app.get("/api/dictionary")
async def get_dictionary():
    return [{"classId": int(k), "tr": v.get("TR", "?"), "en": v.get("EN", "?"),
             "animation": f"{v.get('EN','unknown')}_sign", "category": "Sign"}
            for k, v in sorted(LABEL_MAP.items(), key=lambda x: int(x[0]))]

@app.get("/api/dictionary/{class_id}")
async def get_dictionary_by_id(class_id: int):
    info = get_label(class_id)
    return {"classId": info["class_id"], "tr": info["label_tr"],
            "en": info["label_en"], "animation": f"{info['label_en']}_sign"}

@app.get("/api/avatar/{class_id}")
async def get_avatar(class_id: int):
    info = get_label(class_id)
    return {"class_id": info["class_id"], "label_tr": info["label_tr"],
            "label_en": info["label_en"], "animation_key": f"{info['label_en']}_sign"}

@app.get("/api/admin/overview")
def admin_overview(db: Session = Depends(get_db)):
    rows = db.query(models.History).order_by(
        models.History.created_at.desc()).limit(100).all()
    confs = []
    for r in rows:
        try:
            confs.append(float(r.confidence.replace("%", "")))
        except ValueError:
            pass
    return {
        "total_users": db.query(models.User).count(),
        "active_users": db.query(models.User).filter(models.User.status == "Active").count(),
        "translations": db.query(models.History).count(),
        "avg_confidence": round(sum(confs)/len(confs), 1) if confs else 0.0,
    }

@app.get("/api/admin/users")
def admin_users(db: Session = Depends(get_db)):
    return [{"name": u.full_name, "email": u.email,
             "role": u.role, "sessions": u.sessions, "status": u.status}
            for u in db.query(models.User).all()]

@app.get("/api/admin/logs")
def admin_logs(db: Session = Depends(get_db)):
    return [{"level": l.level, "message": l.message, "user": l.user,
             "when": l.created_at.strftime("%Y-%m-%d %H:%M") if l.created_at else ""}
            for l in db.query(models.Log).order_by(
                models.Log.created_at.desc()).limit(100).all()]

@app.get("/api/admin/model")
async def admin_model():
    return {
        "live_model": "Yüklü" if MODEL else "Yüklenmedi",
        "num_classes": NUM_CLASSES,
        "label_map_size": len(LABEL_MAP),
        "landmark_words": len(LANDMARK_INDEX),
        "mediapipe_ready": MP_HANDS is not None,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "avg_inference": f"{AVG_INFERENCE_MS:.0f} ms" if INFERENCE_COUNT else "Henüz yok",
        "database": "Supabase PostgreSQL",
    }

@app.get("/api/debug/model-check")
async def debug_model_check():
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


@app.on_event("startup")
async def startup_event():
    Base.metadata.create_all(bind=engine)
    logger.info("[DB] Tables ready")

    db = next(get_db())
    try:
        bootstrap_admin(db)
        if db.query(models.Setting).count() == 0:
            db.add(models.Setting(id=1))
            db.commit()
        add_log_db(db, "Info", "Sunucu başlatıldı", "system")
    finally:
        db.close()

    load_model_assets()
    load_landmark_index()

    logger.info("=" * 60)
    logger.info("  SignTurk backend ready")
    logger.info(f"  Model:      {'LOADED' if MODEL else 'MISSING'}")
    logger.info(f"  Landmarks:  {len(LANDMARK_INDEX)} words")
    logger.info(f"  MediaPipe:  {'READY' if MP_HANDS else 'MISSING'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
