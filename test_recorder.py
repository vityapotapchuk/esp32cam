"""Тест рекордера без камери: підміняємо detector.snapshot() синтетичними
кадрами, запускаємо ClipRecorder._record і перевіряємо MP4 + SQLite."""
import sqlite3
import sys
import time

import numpy as np

sys.path.insert(0, "/home/server/esp32cam")
import dashboard  # noqa: E402

# --- підміняємо detector: 640x480 рухомий патерн ---------------------------
class FakeDet:
    def snapshot(self):
        f = getattr(self, "_f", 0)
        self._f = f + 1
        img = np.zeros((480, 640, 3), np.uint8)
        img[:, :, 0] = (f * 7) % 255
        img[:, :, 2] = (f * 13) % 255
        return dashboard.cv2.imencode(".jpg", img)[1].tobytes()

dashboard.detector = FakeDet()

# --- скорочуємо кліп до 3 с для швидкості тесту ----------------------------
dashboard.CLIP_SECONDS = 3

rec = dashboard.ClipRecorder()
t0 = time.time()
rec._record(["person", "dog"])
dt = time.time() - t0
print(f"record took {dt:.1f}s")

# --- перевірка файлу і БД ---------------------------------------------------
last = rec.last_clip
assert last, "last_clip не заповнений"
import os
p = dashboard.CLIPS_DIR / last["path"]
assert p.exists() and p.stat().st_size > 0, "mp4 файл порожній"
print("file:", p, p.stat().st_size, "bytes")

con = sqlite3.connect(dashboard.DB_PATH)
con.row_factory = sqlite3.Row
row = con.execute("SELECT * FROM clips ORDER BY id DESC LIMIT 1").fetchone()
assert row is not None, "запис у БД відсутній"
print("db row:", dict(row))
assert row["frames"] > 0 and row["duration"] >= 2.5, "короткий кліп"

# --- ffprobe: тривалість і кодек -------------------------------------------
import imageio_ffmpeg
ff = imageio_ffmpeg.get_ffmpeg_exe()
out = os.popen(f"{ff} -i {p} -f null - 2>&1").read()
for line in out.splitlines():
    if "Duration" in line or "Video" in line:
        print(line.strip())
assert "h264" in out, "кліп не H.264"
assert "yuv420p" in out, "пікселі не yuv420p"
print("TEST-OK")