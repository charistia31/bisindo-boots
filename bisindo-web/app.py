# -*- coding: utf-8 -*-
"""
BISINDO Web - Flask + SocketIO Backend
Deploy-ready untuk Railway
"""

import os
import sys
import warnings
import pickle
import base64
import numpy as np

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import tensorflow as tf
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SEQ_LEN         = 30
FEATURE_DIM     = 126
CONF_THRESHOLD  = 0.40
SMOOTH_WINDOW   = 5
DEBOUNCE_FRAMES = 20

MODEL_IMAGE_PATH   = os.path.join(BASE_DIR, "model_image.h5")
MODEL_VIDEO_PATH   = os.path.join(BASE_DIR, "model_video.h5")
ENCODER_IMAGE_PATH = os.path.join(BASE_DIR, "label_encoder_image.pkl")
ENCODER_VIDEO_PATH = os.path.join(BASE_DIR, "label_encoder_video.pkl")

# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "bisindo-secret-key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024,
                    ping_timeout=60, ping_interval=25,
                    transports=["websocket", "polling"])

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────
def load_model_safe(model_path, encoder_path, name):
    if not os.path.exists(model_path):
        print(f"[WARN] {name}: model not found at {model_path}")
        return None, None, "NOT_FOUND"
    if not os.path.exists(encoder_path):
        print(f"[WARN] {name}: encoder not found at {encoder_path}")
        return None, None, "ENCODER_NOT_FOUND"
    try:
        # Handle TF version mismatch untuk LSTM time_major argument
        class SafeLSTM(tf.keras.layers.LSTM):
            def __init__(self, *args, **kwargs):
                kwargs.pop('time_major', None)
                super().__init__(*args, **kwargs)
        try:
            model = tf.keras.models.load_model(model_path, compile=False,
                        custom_objects={'LSTM': SafeLSTM})
        except Exception:
            model = tf.keras.models.load_model(model_path, compile=False)
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        print(f"[OK] {name} loaded. Classes: {le.classes_}")
        return model, le, "SUCCESS"
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return None, None, "LOAD_ERROR"

print("=" * 50)
print("BISINDO Web — Loading Models...")
print("=" * 50)
print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] Files in BASE_DIR: {os.listdir(BASE_DIR)}")
print(f"[DEBUG] MODEL_IMAGE_PATH: {MODEL_IMAGE_PATH} exists={os.path.exists(MODEL_IMAGE_PATH)}")
print(f"[DEBUG] MODEL_VIDEO_PATH: {MODEL_VIDEO_PATH} exists={os.path.exists(MODEL_VIDEO_PATH)}")
print(f"[DEBUG] ENCODER_IMAGE_PATH: {ENCODER_IMAGE_PATH} exists={os.path.exists(ENCODER_IMAGE_PATH)}")
print(f"[DEBUG] ENCODER_VIDEO_PATH: {ENCODER_VIDEO_PATH} exists={os.path.exists(ENCODER_VIDEO_PATH)}")
model_img, le_img, status_img = load_model_safe(MODEL_IMAGE_PATH, ENCODER_IMAGE_PATH, "IMAGE")
model_vid, le_vid, status_vid = load_model_safe(MODEL_VIDEO_PATH, ENCODER_VIDEO_PATH, "VIDEO")
print("=" * 50)

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

# Satu instance hands per session (thread-safe via per-sid state)
def make_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.55,
        model_complexity=1
    )

# ─────────────────────────────────────────────
#  PER-CLIENT SESSION STATE
# ─────────────────────────────────────────────
sessions = {}

def get_session(sid):
    if sid not in sessions:
        sessions[sid] = {
            "hands":        make_hands(),
            "mode":         "IMAGE",
            "prob_buffer":  deque(maxlen=SMOOTH_WINDOW),
            "label_buffer": deque(maxlen=SMOOTH_WINDOW),
            "translation":  "",
            "debounce":     0,
            "sequence":     [],
            "collecting":   False,
            "no_hand_count": 0,
            "last_label":   None,
            "current_label": "---",
            "current_conf": 0.0,
        }
    return sessions[sid]

# ─────────────────────────────────────────────
#  LANDMARK EXTRACTION
# ─────────────────────────────────────────────
def extract_landmarks(frame, hands_obj):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands_obj.process(rgb)
    rgb.flags.writeable = True

    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if res.multi_hand_landmarks and res.multi_handedness:
        for idx, hand_lm in enumerate(res.multi_hand_landmarks):
            label  = res.multi_handedness[idx].classification[0].label
            coords = np.array([[p.x, p.y, p.z] for p in hand_lm.landmark], dtype=np.float32)
            wrist  = coords[0]
            coords = coords - wrist
            scale  = np.linalg.norm(coords[9])
            if scale > 1e-6:
                coords /= scale
            flat = coords.flatten()
            if label == "Left":
                left  = flat
            else:
                right = flat

    return np.concatenate([left, right]), res

def draw_landmarks_on_frame(frame, res):
    if res.multi_hand_landmarks:
        for hand_lm in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
    return frame

# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    available_modes = []
    if model_img is not None:
        available_modes.append("IMAGE")
    if model_vid is not None:
        available_modes.append("VIDEO")
    return render_template("index.html",
                           available_modes=available_modes,
                           image_classes=list(le_img.classes_) if le_img else [],
                           video_classes=list(le_vid.classes_) if le_vid else [])

# ─────────────────────────────────────────────
#  SOCKETIO EVENTS
# ─────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    sid = request_sid()
    get_session(sid)
    print(f"[+] Client connected: {sid}")
    emit("server_info", {
        "image_available": model_img is not None,
        "video_available": model_vid is not None,
        "image_classes": list(le_img.classes_) if le_img else [],
        "video_classes": list(le_vid.classes_) if le_vid else [],
    })

@socketio.on("disconnect")
def on_disconnect():
    sid = request_sid()
    if sid in sessions:
        try:
            sessions[sid]["hands"].close()
        except Exception:
            pass
        del sessions[sid]
    print(f"[-] Client disconnected: {sid}")

@socketio.on("set_mode")
def on_set_mode(data):
    sid   = request_sid()
    s     = get_session(sid)
    mode  = data.get("mode", "IMAGE")
    if mode == "IMAGE" and model_img is None:
        emit("error", {"msg": "Model gambar tidak tersedia"})
        return
    if mode == "VIDEO" and model_vid is None:
        emit("error", {"msg": "Model video tidak tersedia"})
        return
    s["mode"]          = mode
    s["sequence"]      = []
    s["collecting"]    = False
    s["prob_buffer"].clear()
    s["label_buffer"].clear()
    s["no_hand_count"] = 0
    s["last_label"]    = None
    s["current_label"] = "---"
    s["current_conf"]  = 0.0
    s["debounce"]      = 0
    emit("mode_changed", {"mode": mode})

@socketio.on("clear_translation")
def on_clear():
    sid = request_sid()
    s   = get_session(sid)
    s["translation"] = ""
    s["last_label"]  = None
    emit("translation_update", {"text": ""})

@socketio.on("frame")
def on_frame(data):
    sid = request_sid()
    s   = get_session(sid)

    # Decode base64 frame dari browser
    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr   = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception as e:
        return

    frame = cv2.flip(frame, 1)
    combined, res = extract_landmarks(frame, s["hands"])
    frame         = draw_landmarks_on_frame(frame, res)

    hand_detected = res.multi_hand_landmarks is not None
    mode          = s["mode"]

    # ── MODE GAMBAR ──
    if mode == "IMAGE" and model_img is not None:
        if hand_detected:
            s["no_hand_count"] = 0
            inp  = combined.reshape(1, FEATURE_DIM)
            pred = model_img.predict(inp, verbose=0)[0]

            s["prob_buffer"].append(pred)
            avg_pred   = np.mean(s["prob_buffer"], axis=0)
            idx        = int(np.argmax(avg_pred))
            conf       = float(avg_pred[idx])
            pred_label = le_img.inverse_transform([idx])[0]

            s["label_buffer"].append(pred_label)
            voted = max(set(s["label_buffer"]), key=s["label_buffer"].count)

            s["current_conf"]  = conf
            s["current_label"] = voted if conf >= CONF_THRESHOLD else "?" + voted

            if conf >= CONF_THRESHOLD and s["debounce"] == 0 and voted != s["last_label"]:
                s["translation"] += voted
                s["last_label"]   = voted
                s["debounce"]     = DEBOUNCE_FRAMES
                emit("translation_update", {"text": s["translation"]})
        else:
            s["no_hand_count"] += 1
            if s["no_hand_count"] > 10:
                s["current_label"] = "---"
                s["current_conf"]  = 0.0
                s["prob_buffer"].clear()
                s["label_buffer"].clear()
            if s["no_hand_count"] > 20:
                s["last_label"] = None

    # ── MODE VIDEO ──
    elif mode == "VIDEO" and model_vid is not None:
        if hand_detected:
            s["no_hand_count"] = 0
            if not s["collecting"]:
                s["collecting"] = True
                s["sequence"]   = []

            s["sequence"].append(combined)
            seq_progress = min(len(s["sequence"]), SEQ_LEN)

            if len(s["sequence"]) >= SEQ_LEN:
                inp  = np.array(s["sequence"][:SEQ_LEN], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
                pred = model_vid.predict(inp, verbose=0)[0]

                s["prob_buffer"].append(pred)
                avg_pred   = np.mean(s["prob_buffer"], axis=0)
                idx        = int(np.argmax(avg_pred))
                conf       = float(avg_pred[idx])
                pred_label = le_vid.inverse_transform([idx])[0]

                s["label_buffer"].append(pred_label)
                voted = max(set(s["label_buffer"]), key=s["label_buffer"].count)

                s["current_conf"]  = conf
                s["current_label"] = voted if conf >= CONF_THRESHOLD else "?" + voted

                if conf >= CONF_THRESHOLD and s["debounce"] == 0 and voted != s["last_label"]:
                    s["translation"] += voted + " "
                    s["last_label"]   = voted
                    s["debounce"]     = DEBOUNCE_FRAMES * 2
                    emit("translation_update", {"text": s["translation"]})

                s["sequence"] = s["sequence"][SEQ_LEN // 2:]

            emit("seq_progress", {"current": seq_progress, "total": SEQ_LEN})
        else:
            s["no_hand_count"] += 1
            seq_len = len(s["sequence"])
            emit("seq_progress", {"current": seq_len, "total": SEQ_LEN})

            if s["no_hand_count"] > 15:
                if s["collecting"] and seq_len >= SEQ_LEN:
                    inp  = np.array(s["sequence"][:SEQ_LEN], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
                    pred = model_vid.predict(inp, verbose=0)[0]
                    idx  = int(np.argmax(pred))
                    conf = float(pred[idx])
                    pred_label = le_vid.inverse_transform([idx])[0]
                    s["current_conf"]  = conf
                    s["current_label"] = pred_label if conf >= CONF_THRESHOLD else "?" + pred_label
                    if conf >= CONF_THRESHOLD and s["debounce"] == 0 and pred_label != s["last_label"]:
                        s["translation"] += pred_label + " "
                        s["last_label"]   = pred_label
                        s["debounce"]     = DEBOUNCE_FRAMES * 2
                        emit("translation_update", {"text": s["translation"]})

                s["collecting"]    = False
                s["sequence"]      = []
                s["prob_buffer"].clear()
                s["label_buffer"].clear()

    # Debounce
    if s["debounce"] > 0:
        s["debounce"] -= 1

    # Encode frame kembali ke JPEG untuk ditampilkan di browser
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    emit("prediction", {
        "label":        s["current_label"],
        "conf":         round(s["current_conf"] * 100, 1),
        "hand_detected": hand_detected,
        "mode":         mode,
        "frame":        frame_b64,
    })

# Helper untuk mendapatkan SID dari context SocketIO
from flask import request as flask_request

def request_sid():
    return flask_request.sid

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Starting BISINDO Web on port {port}...")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)