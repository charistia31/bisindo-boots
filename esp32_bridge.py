#!/usr/bin/env python3
"""
esp32_bridge.py — BISINDO ESP32 Hardware Button Bridge
=======================================================
Membaca output Serial dari ESP32 dan meneruskan perubahan mode
ke server Flask via HTTP GET.

Cara pakai:
  python esp32_bridge.py --port COM3 --room ROOMCODE
  python esp32_bridge.py --port /dev/ttyUSB0 --room ABC123 --server http://localhost:5000

Pastikan pyserial sudah terinstall:
  pip install pyserial requests
"""

import argparse
import time
import sys

try:
    import serial
except ImportError:
    print("[ERROR] pyserial tidak ditemukan. Jalankan: pip install pyserial")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("[ERROR] requests tidak ditemukan. Jalankan: pip install requests")
    sys.exit(1)


# ── Mapping output ESP32 → mode server ──────────────────────────────────────
MODE_MAP = {
    "mode gambar": "IMAGE",
    "mode video":  "VIDEO",
}


def send_mode(server_url: str, room: str, mode: str) -> bool:
    """Kirim perintah mode ke Flask server."""
    url = f"{server_url}/esp32/mode"
    try:
        resp = requests.get(url, params={"room": room, "mode": mode}, timeout=3)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            print(f"[OK] Mode {mode} dikirim ke room {room}")
            return True
        else:
            print(f"[WARN] Server response: {data}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Tidak dapat terhubung ke server {server_url}")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def run_bridge(port: str, baud: int, room: str, server_url: str):
    print(f"[BRIDGE] Membuka port Serial {port} @ {baud} baud...")
    print(f"[BRIDGE] Target room : {room}")
    print(f"[BRIDGE] Server URL  : {server_url}")
    print(f"[BRIDGE] Tekan Ctrl+C untuk berhenti\n")

    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Tunggu ESP32 reset selesai
        ser.flushInput()
        print("[BRIDGE] Koneksi Serial berhasil. Menunggu sinyal tombol...\n")
    except serial.SerialException as e:
        print(f"[ERROR] Gagal membuka port {port}: {e}")
        sys.exit(1)

    last_mode = None  # Hindari pengiriman duplikat berurutan

    try:
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip().lower()
                if not line:
                    continue

                print(f"[SERIAL] '{line}'")

                server_mode = MODE_MAP.get(line)
                if server_mode is None:
                    # Baris tidak dikenal (boot log, dll.) — abaikan
                    continue

                if server_mode == last_mode:
                    # Sama seperti sebelumnya, tidak perlu dikirim ulang
                    print(f"[SKIP] Mode sudah {server_mode}, tidak dikirim ulang")
                    continue

                ok = send_mode(server_url, room, server_mode)
                if ok:
                    last_mode = server_mode

            except UnicodeDecodeError:
                continue
            except serial.SerialException as e:
                print(f"[ERROR] Serial error: {e}")
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n[BRIDGE] Dihentikan oleh pengguna.")
    finally:
        ser.close()
        print("[BRIDGE] Port Serial ditutup.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BISINDO ESP32 Bridge — Serial ke Flask HTTP"
    )
    parser.add_argument(
        "--port", "-p",
        required=True,
        help="Port Serial ESP32, contoh: COM3 (Windows) atau /dev/ttyUSB0 (Linux/Mac)"
    )
    parser.add_argument(
        "--room", "-r",
        required=True,
        help="Kode room BISINDO yang ingin dikontrol (harus sama dengan kode di browser)"
    )
    parser.add_argument(
        "--baud", "-b",
        type=int,
        default=115200,
        help="Baud rate Serial (default: 115200)"
    )
    parser.add_argument(
        "--server", "-s",
        default="http://localhost:5000",
        help="URL Flask server (default: http://localhost:5000)"
    )

    args = parser.parse_args()
    run_bridge(
        port=args.port,
        baud=args.baud,
        room=args.room.strip().upper(),
        server_url=args.server.rstrip("/"),
    )