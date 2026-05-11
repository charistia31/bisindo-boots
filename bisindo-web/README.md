# BISINDO Web — Deploy ke Railway

## Struktur File

```
bisindo-web/
├── app.py                    ← Flask + SocketIO backend
├── templates/
│   └── index.html            ← Frontend web
├── model_image.h5            ← ⚠️ Kamu harus copy ke sini
├── model_video.h5            ← ⚠️ Kamu harus copy ke sini
├── label_encoder_image.pkl   ← ⚠️ Kamu harus copy ke sini
├── label_encoder_video.pkl   ← ⚠️ Kamu harus copy ke sini
├── requirements.txt
├── Procfile
├── railway.toml
└── .gitignore
```

## Langkah Deploy

### 1. Copy file model ke folder ini

Dari folder BISINDO_FINAL, copy 4 file ini ke folder `bisindo-web/`:
```
model_image.h5
model_video.h5
label_encoder_image.pkl
label_encoder_video.pkl
```

### 2. Buat GitHub repo baru

```bash
cd bisindo-web
git init
git add .
git commit -m "Initial deploy BISINDO Web"
git branch -M main
git remote add origin https://github.com/USERNAME/bisindo-web.git
git push -u origin main
```

> ⚠️ File .h5 dan .pkl cukup besar. Pastikan total < 100MB (GitHub free limit).
> Jika lebih besar, gunakan Git LFS:
> ```bash
> git lfs install
> git lfs track "*.h5"
> git lfs track "*.pkl"
> git add .gitattributes
> ```

### 3. Deploy ke Railway

1. Buka https://railway.app dan login
2. Klik **"New Project"** → **"Deploy from GitHub repo"**
3. Pilih repo `bisindo-web`
4. Railway otomatis detect `Procfile` dan mulai build
5. Setelah build selesai, klik **"Generate Domain"** untuk dapat URL publik

### 4. Cek Logs

Jika ada error, buka tab **"Logs"** di Railway dashboard untuk debug.

## Cara Pakai Web

| Tombol | Keyboard | Fungsi |
|--------|----------|--------|
| [I] Gambar | `I` | Mode huruf alfabet statis |
| [V] Video  | `V` | Mode kata bergerak |
| [C] Clear  | `C` | Hapus terjemahan |
| ▶ Aktifkan Kamera | — | Izinkan akses webcam browser |

## Catatan Penting

- Webcam diakses dari **browser user** (bukan server), jadi bisa jalan di cloud
- Server hanya memproses landmark & prediksi, bukan video penuh
- Gunakan HTTPS agar browser mengizinkan akses kamera (Railway otomatis HTTPS)
- Disarankan 1 pengguna sekaligus karena model TF tidak thread-safe untuk multi-user berat
