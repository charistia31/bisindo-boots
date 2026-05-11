# -*- coding: utf-8 -*-
"""
=============================================================
  BISINDO Model Trainer - FINAL VERSION
  Mendukung: Gambar (statis) & Video (sequence bergerak)

  Output:
    model_image.h5          -> model untuk huruf/alfabet
    model_video.h5          -> model untuk kata bergerak
    label_encoder_image.pkl -> label encoder gambar
    label_encoder_video.pkl -> label encoder video
=============================================================
"""

import os
import sys
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# Fix encoding untuk terminal Windows (cp1252)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# -----------------------------------------
#  KONFIGURASI
# -----------------------------------------
IMAGE_DATA_DIR = os.path.join("data", "images")
VIDEO_DATA_DIR = os.path.join("data", "videos")

SEQ_LEN     = 30    # harus sama dengan record_dataset.py
FEATURE_DIM = 126   # 63 (left) + 63 (right)

EPOCHS      = 150
BATCH_SIZE  = 16
TEST_SIZE   = 0.2

# -----------------------------------------
#  AUGMENTASI DATA
# -----------------------------------------
def augment_sample(x, noise_std=0.01, flip_prob=0.5, scale_range=(0.9, 1.1)):
    """
    Augmentasi ringan agar model robust terhadap variasi:
    - Gaussian noise kecil
    - Flip horizontal (kiri<->kanan)
    - Scale acak
    """
    # Gaussian noise
    x = x + np.random.normal(0, noise_std, x.shape).astype(np.float32)

    # Flip horizontal: tukar left (0:63) <-> right (63:126)
    if x.ndim == 1:  # (126,)
        if np.random.rand() < flip_prob:
            x = np.concatenate([x[63:], x[:63]])
    else:            # (SEQ_LEN, 126)
        if np.random.rand() < flip_prob:
            x = np.concatenate([x[:, 63:], x[:, :63]], axis=1)

    # Scale
    scale = np.random.uniform(*scale_range)
    x = x * scale

    return x.astype(np.float32)

def augment_dataset(X, y, multiplier=3):
    """Perbanyak dataset dengan augmentasi"""
    X_aug, y_aug = [X], [y]
    for _ in range(multiplier - 1):
        X_tmp = np.array([augment_sample(x) for x in X])
        X_aug.append(X_tmp)
        y_aug.append(y)
    return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

# -----------------------------------------
#  LOAD DATA GAMBAR
#  Mendukung dua format folder:
#    Baru : data/images/A/sample_0.npy  (shape 126,)
#    Lama : data/A/sample_0.npy         (shape 20x126 atau 126,)
# -----------------------------------------
def load_image_data():
    X, y = [], []

    # Kumpulkan semua direktori sumber yang valid
    search_dirs = []
    if os.path.isdir(IMAGE_DATA_DIR):
        search_dirs.append(IMAGE_DATA_DIR)
    # Folder lama: data/ langsung berisi A/, B/, C/ dst
    legacy_dir = "data"
    if os.path.isdir(legacy_dir):
        for d in os.listdir(legacy_dir):
            full = os.path.join(legacy_dir, d)
            # Hanya folder huruf/kelas tunggal (bukan 'images' atau 'videos')
            if os.path.isdir(full) and d not in ("images", "videos"):
                search_dirs.append(full)
                break  # tandai bahwa legacy ada
        # Kumpulkan semua subfolder legacy (A, B, C, ...)
        # reset dan kumpulkan ulang dengan benar
        search_dirs = []
        if os.path.isdir(IMAGE_DATA_DIR):
            search_dirs.append(("new", IMAGE_DATA_DIR))
        has_legacy = any(
            os.path.isdir(os.path.join(legacy_dir, d)) and d not in ("images", "videos")
            for d in os.listdir(legacy_dir)
            if os.path.isdir(os.path.join(legacy_dir, d))
        )
        if has_legacy:
            search_dirs.append(("legacy", legacy_dir))

    # --- Format baru: data/images/<cls>/ ---
    found_new = False
    if os.path.isdir(IMAGE_DATA_DIR):
        classes_new = sorted([
            d for d in os.listdir(IMAGE_DATA_DIR)
            if os.path.isdir(os.path.join(IMAGE_DATA_DIR, d))
        ])
        for cls in classes_new:
            path = os.path.join(IMAGE_DATA_DIR, cls)
            for f in os.listdir(path):
                if not f.endswith(".npy"):
                    continue
                data = np.load(os.path.join(path, f))
                if data.shape == (FEATURE_DIM,):
                    X.append(data)
                    y.append(cls)
                    found_new = True
                elif data.ndim == 2 and data.shape[1] == FEATURE_DIM:
                    X.append(data[data.shape[0] // 2])
                    y.append(cls)
                    found_new = True

    # --- Format lama: data/<cls>/ (A, B, C, ...) ---
    legacy_dir = "data"
    if os.path.isdir(legacy_dir):
        classes_leg = sorted([
            d for d in os.listdir(legacy_dir)
            if os.path.isdir(os.path.join(legacy_dir, d))
            and d not in ("images", "videos")
            and len(d) <= 3  # folder kelas biasanya 1-3 karakter (A, HALO, dll)
        ])
        for cls in classes_leg:
            path = os.path.join(legacy_dir, cls)
            for f in os.listdir(path):
                if not f.endswith(".npy"):
                    continue
                data = np.load(os.path.join(path, f))
                if data.shape == (FEATURE_DIM,):
                    X.append(data)
                    y.append(cls)
                elif data.ndim == 2 and data.shape[1] == FEATURE_DIM:
                    # Data lama: sequence (20, 126) -> ambil frame tengah
                    X.append(data[data.shape[0] // 2])
                    y.append(cls)

    if not X:
        print("\n[INFO] Tidak ada data gambar ditemukan.")
        print("  Cek folder: data/images/<kelas>/ atau data/<kelas>/")
        return None, None, None

    classes_all = sorted(list(set(y)))
    print(f"\n[GAMBAR] Kelas: {classes_all} | Total sample: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y), classes_all

# -----------------------------------------
#  LOAD DATA VIDEO
# -----------------------------------------
def load_video_data():
    X, y = [], []
    if not os.path.isdir(VIDEO_DATA_DIR):
        return None, None, None

    classes = sorted([
        d for d in os.listdir(VIDEO_DATA_DIR)
        if os.path.isdir(os.path.join(VIDEO_DATA_DIR, d))
    ])
    if not classes:
        return None, None, None

    print(f"\n[VIDEO] Kelas ditemukan: {classes}")

    for cls in classes:
        path = os.path.join(VIDEO_DATA_DIR, cls)
        for f in os.listdir(path):
            if not f.endswith(".npy"):
                continue
            data = np.load(os.path.join(path, f))
            if data.ndim == 2 and data.shape == (SEQ_LEN, FEATURE_DIM):
                X.append(data)
                y.append(cls)
            elif data.ndim == 2 and data.shape[0] != SEQ_LEN:
                # Resize sequence ke SEQ_LEN dengan interpolasi
                indices = np.linspace(0, data.shape[0] - 1, SEQ_LEN).astype(int)
                X.append(data[indices])
                y.append(cls)

    print(f"[VIDEO] Total sample: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y), classes

# -----------------------------------------
#  ARSITEKTUR MODEL GAMBAR (MLP + BN)
#  Input: (126,)
# -----------------------------------------
def build_image_model(input_dim, num_classes):
    inp = tf.keras.Input(shape=(input_dim,), name="input")

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.35)(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inp, out, name="BISINDO_Image")
    return model

# -----------------------------------------
#  ARSITEKTUR MODEL VIDEO (BiLSTM + CNN temporal)
#  Input: (SEQ_LEN, 126)
# -----------------------------------------
def build_video_model(seq_len, feat_dim, num_classes):
    inp = tf.keras.Input(shape=(seq_len, feat_dim), name="input")

    # -- 1D Conv temporal feature extraction --
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # -- Bidirectional LSTM --
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True,
                                          dropout=0.2, recurrent_dropout=0.1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                          dropout=0.2, recurrent_dropout=0.1))(x)
    x = layers.BatchNormalization()(x)

    # -- Attention (global weighted average) --
    # Skor attention per timestep
    att = layers.Dense(1, activation="tanh")(x)          # (batch, T, 1)
    att = layers.Flatten()(att)                           # (batch, T)
    att = layers.Activation("softmax")(att)               # (batch, T)
    att = layers.RepeatVector(128)(att)                   # (batch, 128, T)
    att = layers.Permute([2, 1])(att)                     # (batch, T, 128)
    x   = layers.Multiply()([x, att])
    x   = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(x)  # (batch, 128)

    # -- Dense head --
    x = layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = tf.keras.Model(inp, out, name="BISINDO_Video")
    return model

# -----------------------------------------
#  TRAIN MODEL
# -----------------------------------------
def train(X, y, classes, model_fn, model_path, encoder_path, label):
    print(f"\n{'='*50}")
    print(f"  Training Model {label}")
    print(f"{'='*50}")
    print(f"  Shape X  : {X.shape}")
    print(f"  Kelas    : {classes}")
    print(f"  Jumlah   : {len(classes)}")

    # -- Label Encoding --
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    y_enc = le.transform(y)

    with open(encoder_path, "wb") as f:
        pickle.dump(le, f)
    print(f"  Encoder disimpan -> {encoder_path}")

    # -- Augmentasi --
    print("  Augmentasi data (3x)...")
    X_aug, y_aug = augment_dataset(X, y_enc, multiplier=3)
    print(f"  Setelah augmentasi: {X_aug.shape[0]} sample")

    # -- Split --
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_aug, y_aug,
        test_size=TEST_SIZE,
        stratify=y_aug,
        random_state=42
    )
    print(f"  Train: {len(X_tr)}  Val: {len(X_val)}")

    # -- Class Weight (handle imbalance) --
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weight_dict = dict(enumerate(cw))

    # -- Build Model --
    if label == "GAMBAR":
        model = model_fn(X.shape[1], len(classes))
    else:
        model = model_fn(X.shape[1], X.shape[2], len(classes))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    # -- Callbacks --
    cb = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=0
        )
    ]

    # -- Fit --
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        class_weight=class_weight_dict,
        verbose=1
    )

    # -- Evaluasi --
    best_val_acc = max(history.history["val_accuracy"])
    print(f"\n[HASIL] {label} Best Val Accuracy: {best_val_acc*100:.2f}%")
    print(f"[SAVED] Model -> {model_path}")

    return model

# -----------------------------------------
#  MAIN
# -----------------------------------------
def main():
    print("=" * 55)
    print("  BISINDO Trainer - FINAL VERSION")
    print("=" * 55)

    trained_any = False

    # -- Train Model Gambar --
    X_img, y_img, cls_img = load_image_data()
    if X_img is not None and len(cls_img) >= 2:
        train(
            X_img, y_img, cls_img,
            build_image_model,
            "model_image.h5",
            "label_encoder_image.pkl",
            "GAMBAR"
        )
        trained_any = True
    elif X_img is not None:
        print(f"\n[SKIP] Data gambar hanya {len(cls_img)} kelas, minimal 2 kelas untuk training.")
    else:
        print("\n[INFO] Tidak ada data gambar ditemukan di data/images/")

    # -- Train Model Video --
    X_vid, y_vid, cls_vid = load_video_data()
    if X_vid is not None and len(cls_vid) >= 2:
        train(
            X_vid, y_vid, cls_vid,
            build_video_model,
            "model_video.h5",
            "label_encoder_video.pkl",
            "VIDEO"
        )
        trained_any = True
    elif X_vid is not None:
        print(f"\n[SKIP] Data video hanya {len(cls_vid)} kelas, minimal 2 kelas untuk training.")
    else:
        print("\n[INFO] Tidak ada data video ditemukan di data/videos/")

    if not trained_any:
        print("\n[ERROR] Tidak ada data yang bisa ditraining.")
        print("  Jalankan record_dataset.py terlebih dahulu.")
        sys.exit(1)

    print("\n[SELESAI] Training selesai!")

if __name__ == "__main__":
    main()