# BISINDO Hand Gesture Translator - FINAL VERSION

## Penerjemah Hand Gesture BISINDO Real-Time (LSTM-CNN + Vision IoT)

---

## STRUKTUR FOLDER

```
BISINDO/
├── record_dataset.py       ← Rekam data gambar & video
├── train_model.py          ← Training model gambar & video
├── predict_realtime.py     ← Prediksi real-time dual mode
├── data/
│   ├── images/             ← Data gambar (huruf statis)
│   │   ├── A/sample_0.npy
│   │   └── ...
│   └── videos/             ← Data video (kata bergerak)
│       ├── HALO/sample_0.npy
│       └── ...
├── model_image.h5           ← Model huruf (dibuat setelah train)
├── model_video.h5           ← Model kata (dibuat setelah train)
├── label_encoder_image.pkl
└── label_encoder_video.pkl
```

---

## CARA PAKAI

### 1. Rekam Dataset

```bash
python record_dataset.py
```

**Mode Gambar (Huruf Alfabet):**

- Tekan `[I]` → aktifkan mode gambar
- Tekan `[A-Z]` → pilih huruf
- Tekan `[SPACE]` → mulai rekam 30 sample
- Tahan posisi tangan hingga selesai

**Mode Video (Kata Bergerak):**

- Tekan `[V]` → aktifkan mode video
- Ketik nama kata di terminal (contoh: `HALO`) lalu Enter
- Tekan `[SPACE]` → mulai rekam 30 sample × 30 frame
- Lakukan gerakan kata setiap kali countdown muncul

---

### 2. Training

```bash
python train_model.py
```

- Otomatis melatih model gambar (jika ada data di `data/images/`)
- Otomatis melatih model video (jika ada data di `data/videos/`)
- Membutuhkan minimal **2 kelas** untuk masing-masing mode
- Hasil: `model_image.h5`, `model_video.h5`

---

### 3. Prediksi Real-Time

```bash
python predict_realtime.py
```

**Kontrol:**
| Tombol | Fungsi |
|--------|--------|
| `[1]` | Mode Gambar (huruf statis) |
| `[2]` | Mode Video (kata bergerak) |
| `[C]` | Clear teks terjemahan |
| `[Q]` | Keluar |

---

## TIPS AKURASI TINGGI

1. **Pencahayaan**: Pastikan tangan terang, hindari backlit (cahaya dari belakang)
2. **Background**: Rekam di berbagai background berbeda untuk robustness
3. **Jumlah sample**: Semakin banyak sample = semakin akurat. Target minimal 50+ sample/kelas
4. **Jarak**: Rekam di berbagai jarak tangan dari kamera
5. **Sudut**: Variasikan sudut tangan saat rekam

## DEPENDENCY

```bash
pip install opencv-python mediapipe tensorflow scikit-learn numpy
```
