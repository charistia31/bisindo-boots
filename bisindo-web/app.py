# -*- coding: utf-8 -*-
"""
BISINDO Web - Flask + SocketIO Backend
Two-user room system: User1 (signer) <-> User2 (listener + speech)
"""
import eventlet
eventlet.monkey_patch()
import os
import sys
import warnings
import pickle
import base64
import numpy as np


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, render_template, request as flask_request
from flask_socketio import SocketIO, emit, join_room, leave_room
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
                    max_http_buffer_size=10 * 1024 * 1024)

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
model_img, le_img, status_img = load_model_safe(MODEL_IMAGE_PATH, ENCODER_IMAGE_PATH, "IMAGE")
model_vid, le_vid, status_vid = load_model_safe(MODEL_VIDEO_PATH, ENCODER_VIDEO_PATH, "VIDEO")
print("=" * 50)

# ─────────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

def make_hands():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.55,
        model_complexity=1
    )

# ─────────────────────────────────────────────
#  ROOM MANAGEMENT
#  rooms = { room_code: { "user1": sid, "user2": sid, "translation": "", "speech": "" } }
# ─────────────────────────────────────────────
rooms = {}

def get_or_create_room(room_code):
    if room_code not in rooms:
        rooms[room_code] = {
            "user1": None,
            "user2": None,
            "translation": "",   # isyarat -> teks (dari user1)
            "speech": "",        # suara -> teks (dari user2)
        }
    return rooms[room_code]

def find_room_by_sid(sid):
    for code, room in rooms.items():
        if room["user1"] == sid or room["user2"] == sid:
            return code, room
    return None, None

# ─────────────────────────────────────────────
#  PER-CLIENT SESSION STATE
# ─────────────────────────────────────────────
sessions = {}

def get_session(sid):
    if sid not in sessions:
        sessions[sid] = {
            "hands":         make_hands(),
            "role":          None,   # "user1" or "user2"
            "room_code":     None,
            "mode":          "IMAGE",
            "prob_buffer":   deque(maxlen=SMOOTH_WINDOW),
            "label_buffer":  deque(maxlen=SMOOTH_WINDOW),
            "debounce":      0,
            "sequence":      [],
            "collecting":    False,
            "no_hand_count": 0,
            "last_label":    None,
            "current_label": "---",
            "current_conf":  0.0,
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
    return render_template("lobby.html")

@app.route("/user1")
def user1_page():
    available_modes = []
    if model_img is not None:
        available_modes.append("IMAGE")
    if model_vid is not None:
        available_modes.append("VIDEO")
    return render_template("user1.html",
                           available_modes=available_modes,
                           image_classes=list(le_img.classes_) if le_img else [],
                           video_classes=list(le_vid.classes_) if le_vid else [])

@app.route("/user2")
def user2_page():
    return render_template("user2.html")

# ─────────────────────────────────────────────
#  ROUTE — ESP32 HARDWARE BUTTON
#  GET /esp32/mode?room=ROOMCODE&mode=IMAGE|VIDEO
#  Called by esp32_bridge.py (Serial → HTTP)
# ─────────────────────────────────────────────
@app.route("/esp32/mode")
def esp32_mode():
    room_code = flask_request.args.get("room", "").strip().upper()
    mode      = flask_request.args.get("mode", "IMAGE").strip().upper()

    if mode not in ("IMAGE", "VIDEO"):
        return {"ok": False, "error": "mode harus IMAGE atau VIDEO"}, 400

    if not room_code:
        return {"ok": False, "error": "room_code wajib diisi"}, 400

    if room_code not in rooms:
        return {"ok": False, "error": f"Room {room_code} tidak ditemukan"}, 404

    room = rooms[room_code]
    user1_sid = room.get("user1")

    if not user1_sid or user1_sid not in sessions:
        return {"ok": False, "error": "User 1 belum terhubung ke room ini"}, 404

    s = sessions[user1_sid]

    # Validate model availability
    if mode == "IMAGE" and model_img is None:
        return {"ok": False, "error": "Model gambar tidak tersedia di server"}, 503
    if mode == "VIDEO" and model_vid is None:
        return {"ok": False, "error": "Model video tidak tersedia di server"}, 503

    # Update session state (same logic as on_set_mode)
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

    # Broadcast mode_changed to the whole room (User1 + User2 both receive it)
    socketio.emit("mode_changed", {"mode": mode, "source": "esp32"}, room=room_code)

    label = "Gambar" if mode == "IMAGE" else "Video"
    print(f"[ESP32] Room {room_code} → Mode {label} (via hardware button)")
    return {"ok": True, "room": room_code, "mode": mode}

# ─────────────────────────────────────────────
#  SOCKETIO — ROOM / ROLE EVENTS
# ─────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    sid = flask_request.sid
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
    sid = flask_request.sid
    # Clean up room membership
    room_code, room = find_room_by_sid(sid)
    if room:
        role = "user1" if room["user1"] == sid else "user2"
        room[role] = None
        # Notify the other user
        socketio.emit("partner_disconnected", {"role": role}, room=room_code)
        print(f"[-] {role} left room {room_code}")

    if sid in sessions:
        try:
            sessions[sid]["hands"].close()
        except Exception:
            pass
        del sessions[sid]
    print(f"[-] Client disconnected: {sid}")

@socketio.on("join_room_as")
def on_join_room(data):
    """
    data: { room_code: str, role: "user1"|"user2" }
    """
    sid       = flask_request.sid
    s         = get_session(sid)
    role      = data.get("role")
    room_code = data.get("room_code", "default").strip().upper()

    if not room_code:
        room_code = "DEFAULT"

    room = get_or_create_room(room_code)

    # Check if role already taken
    if role == "user1" and room["user1"] and room["user1"] != sid:
        emit("room_error", {"msg": "User 1 sudah ada di room ini. Coba kode room lain."})
        return
    if role == "user2" and room["user2"] and room["user2"] != sid:
        emit("room_error", {"msg": "User 2 sudah ada di room ini. Coba kode room lain."})
        return

    # Leave previous room if any
    prev_code, prev_room = find_room_by_sid(sid)
    if prev_code and prev_code != room_code:
        prev_role = "user1" if prev_room["user1"] == sid else "user2"
        prev_room[prev_role] = None
        leave_room(prev_code)

    # Join new room
    room[role]       = sid
    s["role"]        = role
    s["room_code"]   = room_code

    join_room(room_code)

    print(f"[ROOM] {role} joined room '{room_code}' (sid={sid})")

    # Emit success to joiner
    emit("room_joined", {
        "room_code":   room_code,
        "role":        role,
        "translation": room["translation"],
        "speech":      room["speech"],
        "partner_online": (room["user2" if role == "user1" else "user1"] is not None),
    })

    # Notify partner
    partner_sid = room["user2" if role == "user1" else "user1"]
    if partner_sid:
        socketio.emit("partner_connected", {"role": role}, room=room_code, skip_sid=sid)

# ─────────────────────────────────────────────
#  SOCKETIO — USER1 EVENTS (isyarat)
# ─────────────────────────────────────────────
@socketio.on("set_mode")
def on_set_mode(data):
    sid  = flask_request.sid
    s    = get_session(sid)
    mode = data.get("mode", "IMAGE")
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
    sid       = flask_request.sid
    s         = get_session(sid)
    room_code = s.get("room_code")

    if room_code and room_code in rooms:
        rooms[room_code]["translation"] = ""

    s["last_label"] = None
    emit("translation_update", {"text": ""})

    # Also tell user2 in the room
    if room_code:
        socketio.emit("sign_translation", {"text": ""}, room=room_code, skip_sid=sid)

@socketio.on("frame")
def on_frame(data):
    sid       = flask_request.sid
    s         = get_session(sid)
    room_code = s.get("room_code")

    try:
        img_data = base64.b64decode(data["image"].split(",")[1])
        np_arr   = np.frombuffer(img_data, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
    except Exception:
        return

    frame = cv2.flip(frame, 1)
    combined, res = extract_landmarks(frame, s["hands"])
    frame         = draw_landmarks_on_frame(frame, res)

    hand_detected = res.multi_hand_landmarks is not None
    mode          = s["mode"]

    translation_updated = False

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
                if room_code and room_code in rooms:
                    rooms[room_code]["translation"] += voted
                    new_text = rooms[room_code]["translation"]
                else:
                    new_text = voted
                s["last_label"]  = voted
                s["debounce"]    = DEBOUNCE_FRAMES
                translation_updated = True
                emit("translation_update", {"text": new_text})
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
            s["collecting"]    = True
            s["sequence"].append(combined)

            if len(s["sequence"]) > SEQ_LEN * 3:
                s["sequence"] = s["sequence"][-SEQ_LEN:]

            seq_len = len(s["sequence"])
            emit("seq_progress", {"current": min(seq_len, SEQ_LEN), "total": SEQ_LEN})

            if seq_len >= SEQ_LEN:
                inp  = np.array(s["sequence"][-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
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
                    if room_code and room_code in rooms:
                        rooms[room_code]["translation"] += voted + " "
                        new_text = rooms[room_code]["translation"]
                    else:
                        new_text = voted + " "
                    s["last_label"]  = voted
                    s["debounce"]    = DEBOUNCE_FRAMES * 2
                    translation_updated = True
                    emit("translation_update", {"text": new_text})

                s["sequence"] = s["sequence"][SEQ_LEN // 2:]

        else:
            s["no_hand_count"] += 1
            seq_len = len(s["sequence"])
            emit("seq_progress", {"current": min(seq_len, SEQ_LEN), "total": SEQ_LEN})

            if s["no_hand_count"] > 15:
                if s["collecting"] and seq_len >= SEQ_LEN:
                    inp  = np.array(s["sequence"][-SEQ_LEN:], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
                    pred = model_vid.predict(inp, verbose=0)[0]
                    idx  = int(np.argmax(pred))
                    conf = float(pred[idx])
                    pred_label = le_vid.inverse_transform([idx])[0]
                    s["current_conf"]  = conf
                    s["current_label"] = pred_label if conf >= CONF_THRESHOLD else "?" + pred_label
                    if conf >= CONF_THRESHOLD and s["debounce"] == 0 and pred_label != s["last_label"]:
                        if room_code and room_code in rooms:
                            rooms[room_code]["translation"] += pred_label + " "
                            new_text = rooms[room_code]["translation"]
                        else:
                            new_text = pred_label + " "
                        s["last_label"]  = pred_label
                        s["debounce"]    = DEBOUNCE_FRAMES * 2
                        translation_updated = True
                        emit("translation_update", {"text": new_text})

                s["collecting"]    = False
                s["sequence"]      = []
                s["prob_buffer"].clear()
                s["label_buffer"].clear()
                s["current_label"] = "---"
                s["current_conf"]  = 0.0

    # Debounce countdown
    if s["debounce"] > 0:
        s["debounce"] -= 1

    # Broadcast new translation to user2 in same room
    if translation_updated and room_code and room_code in rooms:
        new_text = rooms[room_code]["translation"]
        socketio.emit("sign_translation", {"text": new_text}, room=room_code, skip_sid=sid)

    # Encode frame
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()

    emit("prediction", {
        "label":         s["current_label"],
        "conf":          round(s["current_conf"] * 100, 1),
        "hand_detected": hand_detected,
        "mode":          mode,
        "frame":         frame_b64,
    })

# ─────────────────────────────────────────────
#  SOCKETIO — USER2 EVENTS (speech-to-text)
# ─────────────────────────────────────────────
@socketio.on("speech_result")
def on_speech_result(data):
    """
    User2 mengirim hasil speech-to-text ke User1.
    data: { text: str, is_final: bool }
    """
    sid       = flask_request.sid
    s         = get_session(sid)
    room_code = s.get("room_code")
    text      = data.get("text", "")
    is_final  = data.get("is_final", False)

    if is_final and room_code and room_code in rooms:
        rooms[room_code]["speech"] = text

    if room_code:
        # Send to user1 in the room
        socketio.emit("speech_update", {
            "text":     text,
            "is_final": is_final,
        }, room=room_code, skip_sid=sid)

@socketio.on("clear_speech")
def on_clear_speech():
    sid       = flask_request.sid
    s         = get_session(sid)
    room_code = s.get("room_code")

    if room_code and room_code in rooms:
        rooms[room_code]["speech"] = ""

    emit("speech_update", {"text": "", "is_final": True})
    if room_code:
        socketio.emit("speech_update", {"text": "", "is_final": True}, room=room_code, skip_sid=sid)

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 Starting BISINDO Web on port {port}...")
    socketio.run(app, host="0.0.0.0", port=port, debug=False)