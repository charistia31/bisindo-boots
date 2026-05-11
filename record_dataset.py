"""
=============================================================
  BISINDO Dataset Recorder - FINAL VERSION
  Mendukung: Gambar (Huruf/Alfabet) & Video (Kata Bergerak)
=============================================================
  KONTROL:
    MODE:
      [1] = Mode Gambar (statis/alfabet)
      [2] = Mode Video (kata bergerak)

    REKAM:
      [SPACE] = Mulai rekam (di kedua mode)
      [ESC]   = Keluar

    KELAS HURUF (Mode Gambar):
      [A-Z]   = Pilih huruf

    KELAS KATA (Mode Video):
      Ketik nama kata lewat input terminal lalu tekan Enter

  STRUKTUR DATA:
    data/images/<KELAS>/sample_N.npy  → frame tunggal (126,)
    data/videos/<KELAS>/sample_N.npy  → sequence (SEQ_LEN, 126)
=============================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import os
import sys
import threading

# ─────────────────────────────────────────
#  KONFIGURASI
# ─────────────────────────────────────────
NUM_SAMPLES_IMG   = 30    # jumlah sample per kelas (gambar)
NUM_SAMPLES_VID   = 10    # jumlah sample per kelas (video)
SEQ_LEN           = 30    # panjang sequence video (frame)
COUNTDOWN_FRAMES  = 30    # countdown sebelum rekam (≈1 detik @30fps)

IMAGE_DATA_DIR    = os.path.join("data", "images")
VIDEO_DATA_DIR    = os.path.join("data", "videos")

os.makedirs(IMAGE_DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────
#  MEDIAPIPE
# ─────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style   = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
    model_complexity=1
)

# ─────────────────────────────────────────
#  HELPER: Ekstrak Landmark
# ─────────────────────────────────────────
def extract_landmarks(frame):
    """
    Kembalikan (left_126, right_126, hasil_mp, frame_rgb)
    Koordinat di-normalize relatif ke wrist (titik 0) agar
    background/posisi kamera tidak mempengaruhi fitur.
    """
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

            # Ambil koordinat mentah
            coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)

            # ── Normalisasi 1: relatif ke wrist ──
            wrist  = coords[0]
            coords = coords - wrist  # titik 0 jadi origin

            # ── Normalisasi 2: scale oleh jarak wrist→middle_MCP (titik 9) ──
            scale = np.linalg.norm(coords[9])
            if scale > 1e-6:
                coords = coords / scale

            flat = coords.flatten()  # 63 nilai

            if label == "Left":
                left  = flat
            else:
                right = flat

    combined = np.concatenate([left, right])  # 126 nilai
    return combined, res

def draw_hands(frame, res):
    if res.multi_hand_landmarks:
        for hand_lm in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_lm,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )

# ─────────────────────────────────────────
#  HELPER: Overlay UI
# ─────────────────────────────────────────
def put_bar(frame, text, y=30, color=(30, 30, 30), alpha=0.5):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, 0), (w, y + 10), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

def put_center(frame, text, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)
    x = (w - size[0]) // 2
    y = (h + size[1]) // 2
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 5)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 1.5, color, 3)

# ─────────────────────────────────────────
#  INPUT NAMA KATA (thread terpisah)
# ─────────────────────────────────────────
video_class_input = {"value": None, "ready": False}

def ask_video_class():
    print("\n[VIDEO MODE] Masukkan nama kata/kelas (contoh: halo, terima_kasih): ", end="", flush=True)
    val = input().strip().upper().replace(" ", "_")
    if val:
        video_class_input["value"] = val
    video_class_input["ready"] = True

# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka!")
        sys.exit(1)

    mode          = "IMAGE"   # "IMAGE" atau "VIDEO"
    current_class = None
    status_msg    = ""
    asking_input  = False

    print("=" * 55)
    print("  BISINDO Dataset Recorder")
    print("  [1] Mode Gambar   [2] Mode Video")
    print("  [A-Z] Pilih huruf (Mode Gambar)")
    print("  [SPACE] Rekam     [ESC] Keluar")
    print("=" * 55)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        combined, res = extract_landmarks(frame)
        draw_hands(frame, res)

        # ── UI Bar ──
        hand_detected = res.multi_hand_landmarks is not None
        hand_color    = (0, 200, 0) if hand_detected else (0, 0, 200)

        mode_text = f"Mode: {'GAMBAR (1/2)' if mode == 'IMAGE' else 'VIDEO  (1/2)'}  |  Kelas: {current_class or '-'}  |  Tangan: {'OK' if hand_detected else 'TIDAK TERDETEKSI'}"
        put_bar(frame, mode_text, y=28, color=(20, 20, 60))

        if status_msg:
            h = frame.shape[0]
            cv2.putText(frame, status_msg, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # ── Cek input kata (video mode) ──
        if asking_input and video_class_input["ready"]:
            asking_input  = False
            current_class = video_class_input["value"]
            video_class_input["ready"] = False
            video_class_input["value"] = None
            if current_class:
                status_msg = f"Kelas video dipilih: {current_class} | Tekan SPACE untuk rekam"
            else:
                status_msg = "Input kosong, coba lagi tekan V"

        cv2.imshow("BISINDO Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        # ── Ganti Mode ──
        if key == ord('1'):
            mode          = "IMAGE"
            current_class = None
            status_msg    = "Mode GAMBAR aktif. Tekan A-Z pilih kelas."
            print("\n[MODE] Gambar (Alfabet)")

        elif key == ord('2'):
            mode          = "VIDEO"
            current_class = None
            status_msg    = "Mode VIDEO aktif. Ketik nama kata di terminal..."
            print("\n[MODE] Video (Kata Bergerak)")
            asking_input  = True
            video_class_input["ready"] = False
            t = threading.Thread(target=ask_video_class, daemon=True)
            t.start()

        # ── Pilih kelas huruf ──
        elif mode == "IMAGE" and ord('A') <= key <= ord('Z'):
            current_class = chr(key)
            status_msg    = f"Kelas: {current_class} | Tekan SPACE rekam"
            print(f"[KELAS] {current_class}")

        elif mode == "IMAGE" and ord('a') <= key <= ord('z'):
            current_class = chr(key - 32)
            status_msg    = f"Kelas: {current_class} | Tekan SPACE rekam"
            print(f"[KELAS] {current_class}")

        # ── REKAM ──
        elif key == ord(' ') and current_class:
            if mode == "IMAGE":
                record_image(cap, current_class)
                status_msg = f"Selesai rekam gambar kelas {current_class}"
            else:
                record_video(cap, current_class)
                status_msg = f"Selesai rekam video kelas {current_class}"

        elif key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n[INFO] Recorder ditutup.")

# ─────────────────────────────────────────
#  REKAM GAMBAR (statis)
# ─────────────────────────────────────────
def record_image(cap, cls):
    folder   = os.path.join(IMAGE_DATA_DIR, cls)
    os.makedirs(folder, exist_ok=True)
    start_id = len([f for f in os.listdir(folder) if f.endswith(".npy")])
    collected = 0

    print(f"\n[REKAM GAMBAR] Kelas: {cls} | Target: {NUM_SAMPLES_IMG} sample")

    # Countdown
    for cd in range(COUNTDOWN_FRAMES, 0, -1):
        ret, frm = cap.read()
        if not ret: break
        frm = cv2.flip(frm, 1)
        _, res = extract_landmarks(frm)
        draw_hands(frm, res)
        put_center(frm, f"Siap... {cd // 10 + 1}", color=(0, 255, 255))
        cv2.imshow("BISINDO Recorder", frm)
        cv2.waitKey(33)

    while collected < NUM_SAMPLES_IMG:
        ret, frm = cap.read()
        if not ret: break

        frm = cv2.flip(frm, 1)
        combined, res = extract_landmarks(frm)
        draw_hands(frm, res)

        h = frm.shape[0]
        put_bar(frm, f"[GAMBAR] {cls}  {collected+1}/{NUM_SAMPLES_IMG}", color=(60, 20, 20))
        cv2.putText(frm, "Tahan posisi tangan", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.imshow("BISINDO Recorder", frm)
        cv2.waitKey(33)

        # Hanya simpan jika tangan terdeteksi
        if res.multi_hand_landmarks:
            np.save(
                os.path.join(folder, f"sample_{start_id + collected}.npy"),
                combined.astype(np.float32)
            )
            collected += 1
            print(f"  Sample {collected}/{NUM_SAMPLES_IMG} disimpan", end="\r")

    print(f"\n[OK] {collected} sample gambar kelas '{cls}' disimpan.")

# ─────────────────────────────────────────
#  REKAM VIDEO (bergerak)
# ─────────────────────────────────────────
def record_video(cap, cls):
    folder   = os.path.join(VIDEO_DATA_DIR, cls)
    os.makedirs(folder, exist_ok=True)
    start_id = len([f for f in os.listdir(folder) if f.endswith(".npy")])

    print(f"\n[REKAM VIDEO] Kelas: {cls} | Target: {NUM_SAMPLES_VID} sample x {SEQ_LEN} frame")

    for s in range(NUM_SAMPLES_VID):
        # ── Countdown antara sample ──
        for cd in range(COUNTDOWN_FRAMES, 0, -1):
            ret, frm = cap.read()
            if not ret: break
            frm = cv2.flip(frm, 1)
            _, res = extract_landmarks(frm)
            draw_hands(frm, res)
            put_bar(frm, f"[VIDEO] {cls} | Sample {s+1}/{NUM_SAMPLES_VID}", color=(20, 60, 20))
            put_center(frm, f"Siap {cd // 10 + 1}...", color=(0, 255, 255))
            cv2.imshow("BISINDO Recorder", frm)
            cv2.waitKey(33)

        # ── Rekam sequence ──
        sequence = []
        frame_count = 0

        while frame_count < SEQ_LEN:
            ret, frm = cap.read()
            if not ret: break

            frm = cv2.flip(frm, 1)
            combined, res = extract_landmarks(frm)
            draw_hands(frm, res)

            sequence.append(combined.astype(np.float32))
            frame_count += 1

            progress = int((frame_count / SEQ_LEN) * 200)
            h, w = frm.shape[:2]
            cv2.rectangle(frm, (10, h - 30), (10 + 200, h - 10), (50, 50, 50), -1)
            cv2.rectangle(frm, (10, h - 30), (10 + progress, h - 10), (0, 220, 0), -1)
            put_bar(frm, f"[VIDEO] {cls} | S{s+1}/{NUM_SAMPLES_VID} | F{frame_count}/{SEQ_LEN}", color=(20, 60, 20))
            cv2.imshow("BISINDO Recorder", frm)
            cv2.waitKey(33)

        if len(sequence) == SEQ_LEN:
            arr = np.array(sequence, dtype=np.float32)  # (SEQ_LEN, 126)
            np.save(os.path.join(folder, f"sample_{start_id + s}.npy"), arr)
            print(f"  Sample {s+1}/{NUM_SAMPLES_VID} disimpan")
        else:
            print(f"  [WARN] Sample {s+1} tidak lengkap, dilewati.")

    print(f"[OK] {NUM_SAMPLES_VID} sample video kelas '{cls}' selesai.")

# ─────────────────────────────────────────
if __name__ == "__main__":
    main()