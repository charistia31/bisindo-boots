# -*- coding: utf-8 -*-
"""
=============================================================
  BISINDO Real-Time Predictor - FINAL VERSION (FIXED)
  Mendukung: Mode Gambar (alfabet) & Mode Video (kata)

  KONTROL:
    [I] = Mode Gambar (alfabet/statis)
    [V] = Mode Video  (kata bergerak)
    [C] = Clear teks terjemahan
    [Q] = Keluar
=============================================================
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import os
import sys
import warnings
from collections import deque

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Fix encoding terminal Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# -----------------------------------------
#  KONFIGURASI PATH - OS.PATH (SANGAT ROBUST)
# -----------------------------------------
# Memastikan direktori selalu mengarah ke tempat file python ini berada
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

SEQ_LEN           = 30
FEATURE_DIM       = 126
CONF_THRESHOLD    = 0.40
SMOOTH_WINDOW     = 5
DEBOUNCE_FRAMES   = 20

MODEL_IMAGE_PATH   = os.path.join(BASE_DIR, "model_image.h5")
MODEL_VIDEO_PATH   = os.path.join(BASE_DIR, "model_video.h5")
ENCODER_IMAGE_PATH = os.path.join(BASE_DIR, "label_encoder_image.pkl")
ENCODER_VIDEO_PATH = os.path.join(BASE_DIR, "label_encoder_video.pkl")

print("="*60)
print("BISINDO REAL-TIME TRANSLATOR - LOADING MODELS")
print("="*60)
print(f"📁 Base Directory: {BASE_DIR}")
print("-"*60)

# Warna UI (BGR)
COLOR_IMAGE_MODE  = (90, 40, 10)
COLOR_VIDEO_MODE  = (10, 70, 10)
COLOR_PANEL_BOT   = (15, 15, 15)
COLOR_WHITE       = (255, 255, 255)
COLOR_GREEN       = (80, 255, 100)
COLOR_YELLOW      = (0, 230, 255)
COLOR_ORANGE      = (0, 165, 255)
COLOR_RED         = (50, 50, 220)
COLOR_GRAY        = (160, 160, 160)
COLOR_TEAL        = (200, 220, 0)

# -----------------------------------------
#  LOAD MODEL DENGAN ERROR HANDLING DETAIL
# -----------------------------------------
def load_model_safe(model_path, encoder_path, model_name):
    """Load model dengan error handling yang memisahkan error NotFound vs LoadError"""
    print(f"\n🔍 Checking {model_name}...")
    
    # Check model file
    if not os.path.exists(model_path):
        print(f"   ❌ Model file NOT FOUND: {model_path}")
        return None, None, "NOT_FOUND"
    
    print(f"   ✅ Model file exists: {model_path}")
    
    # Check encoder file
    if not os.path.exists(encoder_path):
        print(f"   ❌ Encoder file NOT FOUND: {encoder_path}")
        return None, None, "ENCODER_NOT_FOUND"
    
    print(f"   ✅ Encoder file exists: {encoder_path}")
    
    # Try to load
    try:
        print(f"   ⏳ Loading model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print(f"   ✅ Model loaded successfully")
        
        print(f"   ⏳ Loading encoder...")
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        print(f"   ✅ Encoder loaded successfully")
        
        print(f"   🎯 Classes: {le.classes_}")
        return model, le, "SUCCESS"
        
    except Exception as e:
        print(f"   ❌ ERROR loading {model_name}: {type(e).__name__}")
        print(f"   💬 Details: {str(e)}")
        return None, None, "LOAD_ERROR"

# -----------------------------------------
#  MEDIAPIPE SETUP
# -----------------------------------------
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.55,
    model_complexity=1
)

# -----------------------------------------
#  EKSTRAK LANDMARK
# -----------------------------------------
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = hands.process(rgb)
    rgb.flags.writeable = True

    left  = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)

    if res.multi_hand_landmarks and res.multi_handedness:
        for idx, hand_lm in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[idx].classification[0].label
            lm    = hand_lm.landmark
            coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            wrist  = coords[0]
            coords = coords - wrist
            scale  = np.linalg.norm(coords[9])
            if scale > 1e-6:
                coords = coords / scale
            flat = coords.flatten()
            if label == "Left":
                left  = flat
            else:
                right = flat

    combined = np.concatenate([left, right])
    return combined, res

def draw_hands(frame, res):
    if res.multi_hand_landmarks:
        for hand_lm in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

# -----------------------------------------
#  UI DRAWING HELPERS
# -----------------------------------------
def draw_top_bar(frame, mode, model_img, model_vid, hand_detected):
    h, w = frame.shape[:2]
    bar_color = COLOR_IMAGE_MODE if mode == "IMAGE" else COLOR_VIDEO_MODE

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 38), bar_color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    mode_label = "MODE: GAMBAR (Alfabet)" if mode == "IMAGE" else "MODE: VIDEO (Kata)"
    hand_str   = "  |  Tangan: OK" if hand_detected else "  |  Tangan: --"
    cv2.putText(frame, mode_label + hand_str, (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_WHITE, 2, cv2.LINE_AA)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 38), (w, 72), (25, 25, 25), -1)
    cv2.addWeighted(overlay2, 0.70, frame, 0.30, 0, frame)

    def draw_key(x, key_char, desc, available=True, active=False):
        key_col  = COLOR_GREEN if active else (COLOR_YELLOW if available else COLOR_GRAY)
        desc_col = COLOR_WHITE if available else COLOR_GRAY
        cv2.rectangle(frame, (x, 42), (x + 22, 66), key_col, 1)
        cv2.putText(frame, key_char, (x + 4, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, key_col, 2, cv2.LINE_AA)
        cv2.putText(frame, desc, (x + 26, 61),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, desc_col, 1, cv2.LINE_AA)

    draw_key(10,  "1", "Gambar",  model_img is not None, mode == "IMAGE")
    draw_key(130, "2", "Video",   model_vid is not None, mode == "VIDEO")
    draw_key(250, "C", "Clear",   True,  False)
    draw_key(340, "Q", "Keluar",  True,  False)

def draw_translation_bar(frame, text):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 72), (w, 108), (8, 35, 8), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    label = "Terjemahan:"
    cv2.putText(frame, label, (10, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEAL, 1, cv2.LINE_AA)

    max_chars = 55
    display   = text[-max_chars:] if len(text) > max_chars else text
    display   = display if display else "(kosong - tekan C untuk clear)"
    cv2.putText(frame, display, (120, 98),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, COLOR_GREEN, 2, cv2.LINE_AA)

def draw_bottom_result(frame, label, conf, mode):
    h, w = frame.shape[:2]
    PANEL_H = 110

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - PANEL_H), (w, h), COLOR_PANEL_BOT, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.line(frame, (0, h - PANEL_H), (w, h - PANEL_H), (60, 60, 60), 1)

    if conf >= CONF_THRESHOLD:
        lbl_color = COLOR_GREEN
    elif conf >= CONF_THRESHOLD * 0.6:
        lbl_color = COLOR_YELLOW
    else:
        lbl_color = COLOR_ORANGE

    cv2.putText(frame, label, (22, h - 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.9, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(frame, label, (22, h - 52),
                cv2.FONT_HERSHEY_DUPLEX, 1.9, lbl_color, 3, cv2.LINE_AA)

    bar_x, bar_y = 22, h - 35
    bar_w_max    = w - 44
    bar_filled   = int(min(conf, 1.0) * bar_w_max)
    bar_color    = lbl_color

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 12), (50, 50, 50), -1)
    if bar_filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_filled, bar_y + 12), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + 12), (80, 80, 80), 1)

    conf_text = f"Conf: {conf*100:.1f}%"
    cv2.putText(frame, conf_text, (bar_x + bar_w_max + 6, bar_y + 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_GRAY, 1, cv2.LINE_AA)

def draw_sequence_bar(frame, current, total):
    h, w = frame.shape[:2]
    bar_x  = 10
    bar_y  = h - 128
    bar_w  = w - 20
    filled = int((min(current, total) / total) * bar_w)

    cv2.putText(frame, f"Seq: {current}/{total}", (bar_x, bar_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEAL, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (50, 50, 50), -1)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 10), COLOR_TEAL, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (80, 80, 80), 1)

def draw_notify(frame, msg, color=COLOR_YELLOW):
    h, w = frame.shape[:2]
    size, _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    x = (w - size[0]) // 2
    y = h // 2
    cv2.putText(frame, msg, (x + 2, y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, msg, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

# -----------------------------------------
#  HELPER: RESET STATE
# -----------------------------------------
def reset_state(prob_buffer, label_buffer):
    prob_buffer.clear()
    label_buffer.clear()
    return [], False, 0, None, "---", 0.0

# -----------------------------------------
#  MAIN
# -----------------------------------------
def main():
    # Load Models dengan status akurat
    model_img, le_img, status_img = load_model_safe(MODEL_IMAGE_PATH, ENCODER_IMAGE_PATH, "IMAGE (Alfabet)")
    model_vid, le_vid, status_vid = load_model_safe(MODEL_VIDEO_PATH, ENCODER_VIDEO_PATH, "VIDEO (Kata)")

    print("\n" + "="*60)
    
    if model_img is None and model_vid is None:
        print("\n[ERROR KRITIS] Tidak ada model yang berhasil dimuat!")
        print(f"\n💡 Pastikan file-file berikut ada di folder:")
        print(f"   {BASE_DIR}")
        print("\n Jika file belum ada, jalankan:")
        print("   python train_model.py")
        print("="*60)
        sys.exit(1)

    # Tentukan mode awal
    mode = "IMAGE" if model_img is not None else "VIDEO"
    print(f"✅ Mode awal: {mode}")
    print("🎮 Kontrol:")
    print("   [1] Mode Gambar  [2] Mode Video  [C] Clear  [Q] Keluar")
    print("="*60 + "\n")

    # Setup kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka!")
        sys.exit(1)

    # State variables
    prob_buffer   = deque(maxlen=SMOOTH_WINDOW)
    label_buffer  = deque(maxlen=SMOOTH_WINDOW)
    translation   = ""
    debounce      = 0
    notify_msg    = ""
    notify_timer  = 0
    notify_color  = COLOR_YELLOW

    sequence      = []
    collecting    = False
    no_hand_count = 0
    last_label    = None
    current_label = "---"
    current_conf  = 0.0

    print("🎥 Kamera aktif. Mulai deteksi gesture...\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        combined, res = extract_landmarks(frame)
        draw_hands(frame, res)

        hand_detected = res.multi_hand_landmarks is not None

        # ==============================
        #  MODE GAMBAR (statis / image)
        # ==============================
        if mode == "IMAGE" and model_img is not None:
            if hand_detected:
                no_hand_count = 0
                inp  = combined.reshape(1, FEATURE_DIM)
                pred = model_img.predict(inp, verbose=0)[0]

                prob_buffer.append(pred)
                avg_pred   = np.mean(prob_buffer, axis=0)
                idx        = int(np.argmax(avg_pred))
                conf       = float(avg_pred[idx])
                pred_label = le_img.inverse_transform([idx])[0]

                label_buffer.append(pred_label)
                voted = max(set(label_buffer), key=label_buffer.count)

                current_conf  = conf
                current_label = voted if conf >= CONF_THRESHOLD else "?" + voted

                if conf >= CONF_THRESHOLD and debounce == 0 and voted != last_label:
                    translation += voted
                    last_label   = voted
                    debounce     = DEBOUNCE_FRAMES

            else:
                no_hand_count += 1
                if no_hand_count > 10:
                    current_label = "---"
                    current_conf  = 0.0
                    prob_buffer.clear()
                    label_buffer.clear()
                if no_hand_count > 20:
                    last_label = None

        # ==============================
        #  MODE VIDEO (sequence / video)
        # ==============================
        elif mode == "VIDEO" and model_vid is not None:
            if hand_detected:
                no_hand_count = 0

                if not collecting:
                    collecting = True
                    sequence   = []

                sequence.append(combined)
                draw_sequence_bar(frame, len(sequence), SEQ_LEN)

                if len(sequence) >= SEQ_LEN:
                    inp  = np.array(sequence[:SEQ_LEN], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
                    pred = model_vid.predict(inp, verbose=0)[0]

                    prob_buffer.append(pred)
                    avg_pred   = np.mean(prob_buffer, axis=0)
                    idx        = int(np.argmax(avg_pred))
                    conf       = float(avg_pred[idx])
                    pred_label = le_vid.inverse_transform([idx])[0]

                    label_buffer.append(pred_label)
                    voted = max(set(label_buffer), key=label_buffer.count)

                    current_conf  = conf
                    current_label = voted if conf >= CONF_THRESHOLD else "?" + voted

                    if conf >= CONF_THRESHOLD and debounce == 0 and voted != last_label:
                        translation += voted + " "
                        last_label   = voted
                        debounce     = DEBOUNCE_FRAMES * 2

                    # Sliding window overlap 50%
                    sequence = sequence[SEQ_LEN // 2:]

            else:
                no_hand_count += 1
                draw_sequence_bar(frame, len(sequence), SEQ_LEN)

                if no_hand_count > 15:
                    if collecting and len(sequence) >= SEQ_LEN:
                        inp  = np.array(sequence[:SEQ_LEN], dtype=np.float32).reshape(1, SEQ_LEN, FEATURE_DIM)
                        pred = model_vid.predict(inp, verbose=0)[0]
                        idx  = int(np.argmax(pred))
                        conf = float(pred[idx])
                        pred_label = le_vid.inverse_transform([idx])[0]
                        current_conf  = conf
                        current_label = pred_label if conf >= CONF_THRESHOLD else "?" + pred_label
                        if conf >= CONF_THRESHOLD and debounce == 0 and pred_label != last_label:
                            translation += pred_label + " "
                            last_label   = pred_label
                            debounce     = DEBOUNCE_FRAMES * 2

                    collecting = False
                    sequence   = []
                    prob_buffer.clear()
                    label_buffer.clear()

        # Debounce counter
        if debounce > 0:
            debounce -= 1

        # Draw UI
        draw_top_bar(frame, mode, model_img, model_vid, hand_detected)
        draw_translation_bar(frame, translation)
        draw_bottom_result(frame, current_label, current_conf, mode)

        if notify_timer > 0:
            draw_notify(frame, notify_msg, notify_color)
            notify_timer -= 1

        cv2.imshow("BISINDO Real-Time Translator", frame)
        key = cv2.waitKey(1) & 0xFF

        # Keyboard handler
        if key in (ord('q'), ord('Q'), 27):
            break

        elif key in (ord('1'), ord('1')):
            if model_img is not None:
                mode = "IMAGE"
                sequence, collecting, no_hand_count, last_label, current_label, current_conf = \
                    reset_state(prob_buffer, label_buffer)
                notify_msg   = ">> Mode GAMBAR (Alfabet) aktif"
                notify_color = COLOR_YELLOW
                notify_timer = 60
            else:
                if status_img == "NOT_FOUND":
                    notify_msg = "!! model_image.h5 tidak ada"
                elif status_img == "LOAD_ERROR":
                    notify_msg = "!! model_image.h5 corrupt/error"
                else:
                    notify_msg = "!! encoder tidak ditemukan"
                notify_color = COLOR_RED
                notify_timer = 90

        elif key in (ord('2'), ord('2')):
            if model_vid is not None:
                mode = "VIDEO"
                sequence, collecting, no_hand_count, last_label, current_label, current_conf = \
                    reset_state(prob_buffer, label_buffer)
                notify_msg   = ">> Mode VIDEO (Kata) aktif"
                notify_color = COLOR_GREEN
                notify_timer = 60
            else:
                # Disini kita beritahu error yang sebenarnya
                if status_vid == "NOT_FOUND":
                    notify_msg = "!! model_video.h5 tidak ada"
                elif status_vid == "LOAD_ERROR":
                    notify_msg = "!! model_video.h5 corrupt/error"
                else:
                    notify_msg = "!! encoder video tidak ada"
                notify_color = COLOR_RED
                notify_timer = 90

        elif key in (ord('c'), ord('C')):
            translation  = ""
            last_label   = None
            notify_msg   = "Terjemahan dikosongkan"
            notify_color = COLOR_TEAL
            notify_timer = 45

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n[INFO] Program selesai.")

if __name__ == "__main__":
    main()