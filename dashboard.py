#!/usr/bin/env python3
"""Realtime-дашборд для ESP32-CAM: MJPEG-стрім з рамками детекції + керування.

Запуск:  ~/cv-venv/bin/python dashboard.py   (порт 8084)

- Фоновий потік тягне MJPEG-стрім камери (detect.MjpegGrabber),
  проганяє кадри крізь YOLOv8n (models/yolov8n.onnx), малює рамки
  навколо людей/тварин і рахує рух (frame diff).
- /            — UI: живий стрім з рамками, статус, керування, журнал подій
- /stream.mjpg — анотований MJPEG (рамки навколо об'єктів)
- /api/state   — весь realtime-стан у JSON (опитується UI щосекунди)
- /api/control — POST {var, val} -> проксі на камеру /control?var=..&val=..
- /api/frame_annotated — останній анотований JPEG (для тестів/скріншотів)
"""
import collections
import json
import os
import sqlite3
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, Response, jsonify, request, send_file

from detect import (MODEL_PATH, STREAM_URL, TARGET_CLASSES, MjpegGrabber,
                    enhance, letterbox, postprocess)
import motion_config
import activity

CAM = "http://192.168.1.186"
PORT = 8084
JPEG_QUALITY = 80

# --- дебаунс подій детекції -----------------------------------------------
# Хибні спалахи (крісло при зміні освітлення) живуть 1-2 кадри (~0.2 с).
# Подія appear/change створюється лише коли кандидат протримався
# CONFIRM_FRAMES кадрів підряд І щонайменше CONFIRM_SECONDS секунд;
# clear — після CLEAR_FRAMES кадрів підряд без об'єктів.
CONFIRM_FRAMES = 3
CONFIRM_SECONDS = 0.3
CLEAR_FRAMES = 2

# --- рекордер кліпів (10 с, H.264 MP4, SQLite-каталог) ---------------------
CLIPS_DIR = Path(__file__).with_name("clips")
DB_PATH = Path(__file__).with_name("clips.db")
CLIP_SECONDS = 10
# Оригінали кліпів лишаються в clips/ — внутрішня пам'ять карти (SD, mmcblk0).
# У Telegram йде тільки легка версія (clips/light/, 360p, високий CRF).
LIGHT_DIR = CLIPS_DIR / "light"
LIGHT_HEIGHT = 360
LIGHT_CRF = "32"
# Журнал руху — ОКРЕМА файл-база (каталог кліпів живе в clips.db)
JOURNAL_DB = Path(__file__).with_name("journal.db")
# статичний ffmpeg з libx264 (imageio-ffmpeg у cv-venv): системного
# H.264-енкодера на Pi немає, cv2.VideoWriter('avc1') не відкривається
import imageio_ffmpeg  # noqa: E402
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

from telegram_notify import deliver, send_message  # noqa: E402
import event_log  # noqa: E402

FRAMESIZE_NAMES = {0: "qqvga", 1: "qqvga2", 2: "qcif", 3: "cif", 4: "qvga",
                   5: "vga", 6: "svga", 7: "xga", 8: "sxga", 9: "uxga",
                   10: "hd", 11: "fhd"}

app = Flask(__name__)

# --- файловий лог -----------------------------------------------------------
# journald на цьому хості не персистентний ("No journal files were found"),
# тому stdout сервісу губиться. Пишемо важливі події рекордера у файл.
LOG_PATH = Path(__file__).with_name("dashboard.log")


def _log(msg):
    try:
        with LOG_PATH.open("a") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')} {msg}\n")
    except Exception:
        pass


def _safe_json(text):
    """Парсить JSON, лікуючи нецитовані значення типу 4MB (баг прошивки)."""
    import re
    return json.loads(re.sub(
        r':\s*([0-9]*[A-Za-z_][A-Za-z0-9_.]*)\s*([,}])', r':"\1"\2', text))


# ---------------------------------------------------------------- detector
class Detector(threading.Thread):
    """YOLO-інференс над стрімом: рамки + метрики + журнал подій."""

    def __init__(self, grabber):
        super().__init__(daemon=True)
        self.grabber = grabber
        self.lock = threading.Lock()
        self.annotated = None          # останній анотований JPEG
        self.dets = []                 # [{cls, conf, box:[x1,y1,x2,y2]}]
        self.fps = 0.0                 # fps інференсу
        self.frames = 0
        self.motion = 0.0              # % змінених пікселів (EMA)
        self.events = []               # журнал появи/зникнення
        self.last_classes = set()
        self.error = None
        self.stopped = False
        self.enabled = True            # вимикач детекції (кнопка на панелі)
        # --- дебаунс: кандидат-класи, що чекають підтвердження ---
        self._cand = set()             # класи-кандидати на appear/change
        self._cand_n = 0               # скільки кадрів підряд кандидат тримається
        self._cand_t0 = 0.0            # час першого кадру кандидата
        self._clear_n = 0              # кадри підряд без об'єктів
        # стан графіка (оновлюється в run() з motion_schedule.json)
        self.sched_active = True
        self.sched_reason = "schedule-on"
        self.sched_profile = "weekday"

    def run(self):
        # 4 дефолтні intra-op потоки ORT крутяться по ~100% CPU навіть
        # між інференсами (spin-wait) — 2 потоки + без spinning вистачає.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        sess = ort.InferenceSession(MODEL_PATH, opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        fps_hist = collections.deque(maxlen=30)
        prev_gray = None
        ema = 0.0

        while not self.stopped:
            # Графік роботи (motion_schedule.json) читається тут, у циклі:
            # master-перемикач має пріоритет над сіткою, а ручний enabled —
            # над обома. Розклад діє без браузера й без рестарту сервісу.
            sched = motion_config.load()
            sched_on, sched_reason, sched_profile = motion_config.active_now(sched)
            with self.lock:
                self.sched_active = sched_on
                self.sched_reason = sched_reason
                self.sched_profile = sched_profile
            # Вдома нікого (охорона) — детектор працює й поза графіком:
            # всі детекції йдуть у Telegram. Ручний вимикач — важливіший.
            away = activity_log.alerts_on()
            if not self.enabled or (not sched_on and not away):
                if self.enabled:
                    activity_log.tick(time.time())   # таймер «5 хв нікого» іде й на паузі
                # Детектор вимкнено кнопкою або графіком: YOLO-інференс не
                # рахуємо (звільняємо CPU), але кадр оновлюємо — живий стрім
                # продовжується, просто без рамок і подій.
                frame = self.grabber.read()
                if frame is not None:
                    try:
                        ok, jpg = cv2.imencode(
                            ".jpg", enhance(frame),
                            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                        with self.lock:
                            if ok:
                                self.annotated = jpg.tobytes()
                            self.dets = []
                            self.fps = 0.0
                            self.motion = 0.0
                            self.last_classes = set()
                    except Exception:
                        pass
                time.sleep(0.2)
                continue
            frame = self.grabber.read()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                frame = enhance(frame)

                # --- рух: frame diff на зменшеному сірому кадрі
                gray = cv2.GaussianBlur(
                    cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               (160, 120)), (5, 5), 0)
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    m = float((diff > 25).mean() * 100)
                    ema = 0.7 * ema + 0.3 * m if ema else m
                prev_gray = gray

                # --- YOLO
                t0 = time.time()
                inp, r, dx, dy = letterbox(frame)
                blob = inp[:, :, ::-1].transpose(2, 0, 1)[None]
                blob = blob.astype(np.float32) / 255.0
                out = sess.run(None, {inp_name: blob})[0]
                raw = postprocess(out, r, dx, dy, frame.shape)
                dt = time.time() - t0
                fps_hist.append(1.0 / dt if dt > 0 else 0)

                # --- фільтри з графіка: чутливість, мін. розмір, зона ------
                # postprocess ріже за сталим CONF_THRES=0.35; тут звужуємо
                # (або розширюємо) поріг за повзунком чутливості, відсіюємо
                # дрібні об'єкти та все, що поза зоною детекції.
                conf_thres = motion_config.conf_thres(sched)
                min_area = sched["min_area_pct"] / 100.0
                zone = motion_config.zone_points(sched)
                if conf_thres > 0.35 or min_area > 0 or zone:
                    raw = self._apply_filters(raw, frame.shape, conf_thres,
                                              min_area, zone)

                # --- рамки
                vis = frame
                h_img, w_img = vis.shape[:2]
                for (bx, conf, c) in raw:
                    color = (0, 255, 255) if c == 0 else (0, 170, 255)
                    x1, y1, x2, y2 = map(int, bx)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    # кутові маркери — краще видно на стрімі
                    L = max(10, (x2 - x1 + y2 - y1) // 14)
                    t = 3
                    for (cx, cy, ddx, ddy) in (
                            (x1, y1, 1, 1), (x2, y1, -1, 1),
                            (x1, y2, 1, -1), (x2, y2, -1, -1)):
                        cv2.line(vis, (cx, cy),
                                 (cx + ddx * L, cy), color, t)
                        cv2.line(vis, (cx, cy),
                                 (cx, cy + ddy * L), color, t)
                    label = f"{TARGET_CLASSES[c]} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    ytop = max(y1 - th - 8, 0)
                    cv2.rectangle(vis, (x1, ytop),
                                  (x1 + tw + 6, ytop + th + 6), color, -1)
                    cv2.putText(vis, label, (x1 + 3, ytop + th + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
                ok, jpg = cv2.imencode(
                    ".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                dets = [{"cls": TARGET_CLASSES[c], "conf": round(float(cf), 2),
                         "box": [int(v) for v in bx]}
                        for bx, cf, c in raw]
                classes = {d["cls"] for d in dets}
                now = time.time()
                confirmed, gone = self._debounce(classes, now)

                with self.lock:
                    if ok:
                        self.annotated = jpg.tobytes()
                    self.dets = dets
                    self.fps = round(float(np.mean(fps_hist)), 1) if fps_hist else 0.0
                    self.frames += 1
                    self.motion = round(ema, 1)
                    # подія — лише після підтвердження дебаунсом
                    if confirmed and classes != self.last_classes:
                        etype = "appear" if not self.last_classes else "change"
                        self._event(now, etype, sorted(classes), dets)
                        self.last_classes = set(classes)
                    elif gone and self.last_classes:
                        self._event(now, "clear", [])
                        self.last_classes = set()
                # журнал дій (прийшов/пішов, кіт поїв…) — на кожному кадрі,
                # бо «пішов» визначається саме відсутністю об'єктів
                try:
                    activity_log.update(now, dets, vis.shape,
                                        jpg.tobytes() if ok else None)
                except Exception as e:
                    _log(f"[activity] {type(e).__name__}: {e}")
            except Exception as e:  # не вмирати від разових збоїв
                self.error = str(e)
                time.sleep(0.5)

    def _apply_filters(self, raw, shape, conf_thres, min_area, zone):
        """Відсіяти детекції за чутливістю, мін. площею та зоною.

        raw — [(box, conf, cls)] з postprocess. zone — полігон у
        нормалізованих координатах 0..1 (або None = весь кадр).
        Об'єкт лишається, якщо його центр усередині зони.
        """
        h, w = shape[:2]
        frame_area = float(h * w) or 1.0
        out = []
        for (bx, conf, c) in raw:
            if conf < conf_thres:
                continue
            x1, y1, x2, y2 = map(float, bx)
            if (max(0.0, x2 - x1) * max(0.0, y2 - y1)) / frame_area < min_area:
                continue
            if zone:
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                if not motion_config.point_in_polygon(cx, cy, zone):
                    continue
            out.append((bx, conf, c))
        return out

    def _debounce(self, classes, now):
        """Дебаунс подій. classes — множина класів поточного кадру.

        Повертає (confirmed, gone): confirmed=True — класи протримались
        CONFIRM_FRAMES кадрів підряд і CONFIRM_SECONDS секунд (appear/change);
        gone=True — CLEAR_FRAMES кадрів підряд без об'єктів (clear).
        Викликається на кожному кадрі детекції.
        """
        if classes:
            if classes == self._cand:
                self._cand_n += 1
            else:
                self._cand = set(classes)
                self._cand_n = 1
                self._cand_t0 = now
            self._clear_n = 0
        else:
            self._clear_n += 1
            self._cand = set()
            self._cand_n = 0
        confirmed = (bool(classes) and self._cand_n >= CONFIRM_FRAMES
                     and now - self._cand_t0 >= CONFIRM_SECONDS)
        gone = not classes and self._clear_n >= CLEAR_FRAMES
        return confirmed, gone

    def _event(self, ts, etype, classes, dets=None):
        self.events.append({"t": ts, "type": etype, "classes": classes})
        del self.events[:-50]
        # початок руху (поява/зміна об'єктів) — у журнал (окрема БД + файл),
        # clear не тригерить ні журнал, ні запис.
        # Telegram — лише коли вдома нікого (охорона в activity.py).
        if etype in ("appear", "change") and classes:
            alert = activity_log.alerts_on(ts)
            jid = journal_add(ts, classes)
            event_log.write("motion", ts, dets, kind=etype, alert=alert,
                            journal_id=jid)
            recorder.trigger(classes, jid, alert=alert)

    def snapshot(self):
        with self.lock:
            return self.annotated


# ------------------------------------------------------------ clip recorder
def db_init():
    """Схема каталогу кліпів."""
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,            -- unix-час початку кліпу
        classes TEXT NOT NULL,       -- JSON-список класів у кліпі
        path TEXT NOT NULL,          -- відносний шлях у clips/
        size INTEGER NOT NULL,       -- байт
        duration REAL NOT NULL,      -- секунд (фактична)
        frames INTEGER NOT NULL)""")
    con.commit()
    con.close()


def db_add(ts, classes, path, size, duration, frames):
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO clips (ts, classes, path, size, duration, frames) "
        "VALUES (?,?,?,?,?,?)",
        (ts, json.dumps(classes), path, size, duration, frames))
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def db_list(limit=100):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM clips ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------- журнал руху (окрема БД)
def journal_init():
    """Схема журналу руху в journal.db — окрема файл-база.

    start_ts — unix-час ПОЧАТКУ руху (момент події appear/change),
    start_local — те саме локальним часом 'YYYY-MM-DD HH:MM:SS' (дата + час).
    """
    con = sqlite3.connect(JOURNAL_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS motion_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_ts REAL NOT NULL,      -- unix-час початку руху
        start_local TEXT NOT NULL,   -- локальні дата і час початку руху
        classes TEXT NOT NULL,       -- JSON-список класів на початку руху
        clip_id INTEGER,             -- id кліпу в clips.db (якщо записано)
        clip_path TEXT,              -- ім'я файлу оригіналу в clips/
        light_path TEXT,             -- ім'я файлу легкої версії в clips/light/
        tg_sent INTEGER NOT NULL DEFAULT 0)""")
    con.commit()
    con.close()


def journal_add(start_ts, classes):
    """Записати початок руху; повернути id рядка журналу."""
    con = sqlite3.connect(JOURNAL_DB)
    cur = con.execute(
        "INSERT INTO motion_log (start_ts, start_local, classes) "
        "VALUES (?,?,?)",
        (start_ts,
         datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
         json.dumps(classes)))
    con.commit()
    jid = cur.lastrowid
    con.close()
    return jid


def journal_update_clip(jid, clip_id, clip_path, light_path):
    """Після запису кліпу прив'язати його (оригінал + легка версія) до події."""
    con = sqlite3.connect(JOURNAL_DB)
    con.execute(
        "UPDATE motion_log SET clip_id=?, clip_path=?, light_path=? "
        "WHERE id=?", (clip_id, clip_path, light_path, jid))
    con.commit()
    con.close()


def journal_mark_sent(jid):
    con = sqlite3.connect(JOURNAL_DB)
    con.execute("UPDATE motion_log SET tg_sent=1 WHERE id=?", (jid,))
    con.commit()
    con.close()


def journal_list(limit=100):
    con = sqlite3.connect(JOURNAL_DB)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM motion_log ORDER BY start_ts DESC LIMIT ?",
        (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


class ClipRecorder:
    """Пише 10-секундні H.264 MP4-кліпи з анотованих кадрів детектора.

    Тригер — подія детекції (appear/change/clear). Кліп починається з моменту
    тригера і триває CLIP_SECONDS; кадри за цей час пишуться в тимчасовий
    .mp4 через pipe у статичний ffmpeg (libx264, yuv420p — грає в браузерах),
    потім реєструється в SQLite. Поки йде запис — нові тригери ігноруються
    (один кліп за раз, інакше на Pi 2 кадри одночасно не потягнути).

    Самовідновлення: робочий потік може померти або зависнути (напр. на
    write() у ffmpeg). Тоді прапорець writing лишався True назавжди, і
    trigger() мовчки відкидав усі події — саме це сталося 30.09. Тепер
    trigger() перевіряє, чи живий потік і чи не вичерпано жорсткий ліміт
    HARD_TIMEOUT, і за потреби піднімає новий потік.
    """

    HARD_TIMEOUT = CLIP_SECONDS + 120   # жорсткий ліміт на один кліп, с

    def __init__(self):
        self.lock = threading.Lock()
        self.queue = []            # [класи, journal_id], що чекають на запис
        self.writing = False       # зараз пишеться кліп
        self.writing_since = 0.0   # коли почався поточний запис
        self.last_clip = None      # {id, ts, classes, path, size, duration, frames}
        self.error = None
        self._worker = None        # робочий потік
        self._proc = None          # поточний ffmpeg (щоб убити при скиданні)
        CLIPS_DIR.mkdir(exist_ok=True)
        LIGHT_DIR.mkdir(exist_ok=True)
        db_init()
        journal_init()

    def start(self):
        self._ensure_worker()

    def _ensure_worker(self, force=False):
        """Підняти робочий потік, якщо він помер (або примусово)."""
        w = self._worker
        if force or w is None or not w.is_alive():
            w = threading.Thread(target=self._loop, daemon=True,
                                 name="clip-recorder")
            self._worker = w
            w.start()

    def trigger(self, classes, journal_id=None, alert=True):
        """Викликається з Detector._event — поставити класи в чергу."""
        with self.lock:
            w = self._worker
            dead = w is None or not w.is_alive()
            stuck = (self.writing
                     and time.time() - self.writing_since > self.HARD_TIMEOUT)
            stale = None
            if dead or stuck:
                if self.writing or self.queue:
                    self.error = ("watchdog: потік рекордера помер"
                                  if dead else "watchdog: скинуто завислий запис")
                    _log(f"[recorder] {self.error}")
                self.writing = False
                self.queue = []
                stale = self._proc
                self._proc = None
            if self.writing or self.queue:
                return
            self.queue = [(list(classes), journal_id, alert)]
        if stale is not None:
            try:
                stale.kill()
            except Exception:
                pass
        self._ensure_worker(force=stuck)

    def status(self):
        with self.lock:
            return {"writing": self.writing, "pending": list(self.queue),
                    "last": dict(self.last_clip) if self.last_clip else None,
                    "error": self.error}

    def _loop(self):
        while True:
            with self.lock:
                item = self.queue.pop(0) if self.queue else None
                self.writing = item is not None
                if item is not None:
                    self.writing_since = time.time()
            if item is None:
                time.sleep(0.25)
                continue
            try:
                classes, jid, alert = item
                self._record(classes, jid, alert)
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
                _log(f"[recorder] {self.error}")
            finally:
                with self.lock:
                    self.writing = False
                    self._proc = None

    def _record(self, classes, journal_id=None, alert=True):
        t0 = time.time()
        stamp = datetime.fromtimestamp(t0).strftime("%Y%m%d_%H%M%S")
        name = f"clip_{stamp}.mp4"
        tmp = CLIPS_DIR / (name + ".part")
        final = CLIPS_DIR / name
        light_name = "light_" + name
        light_tmp = LIGHT_DIR / (light_name + ".part")
        light_final = LIGHT_DIR / light_name
        # -re не потрібен: кадри йдуть у реальному часі; -use_wallclock
        # як timestamps дав би змінний fps, тому фіксуємо 10 fps на вході
        proc = subprocess.Popen(
            [FFMPEG, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", "{0}x{1}".format(*self._size()), "-r", "10",
             "-i", "pipe:0",
             "-c:v", "libx264", "-preset", "ultrafast", "-threads", "2",
             "-crf", "26",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-f", "mp4", str(tmp)],
            stdin=subprocess.PIPE)
        self._proc = proc
        # Watchdog: якщо ffmpeg перестав споживати stdin, proc.stdin.write()
        # блокується НАЗАВЖДИ (саме це застрягло 09-09) — вбиваємо ffmpeg,
        # write() кине BrokenPipeError, цикл завершиться, finally прибере.
        watchdog = threading.Timer(CLIP_SECONDS + 60, proc.kill)
        watchdog.daemon = True
        watchdog.start()
        frames = 0
        deadline = t0 + CLIP_SECONDS
        while time.time() < deadline:
            jpg = detector.snapshot()
            if jpg is not None:
                img = cv2.imdecode(
                    np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    try:
                        proc.stdin.write(img.tobytes())
                        frames += 1
                    except (BrokenPipeError, OSError):
                        break
            time.sleep(0.1)
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            self.error = "ffmpeg not exited in 30s — killed"
        finally:
            watchdog.cancel()
        duration = time.time() - t0
        duration = time.time() - t0
        if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg exit {proc.returncode}")
        tmp.rename(final)
        # --- легка версія для Telegram: 360p, високий CRF, без звуку ---
        light_name_db = None
        try:
            subprocess.run(
                [FFMPEG, "-y", "-loglevel", "error", "-i", str(final),
                 "-vf", f"scale=-2:{LIGHT_HEIGHT}", "-c:v", "libx264",
                 "-preset", "ultrafast", "-threads", "2", "-crf", LIGHT_CRF,
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 "-an", "-f", "mp4", str(light_tmp)],
                check=True, timeout=120)
            light_tmp.rename(light_final)
            light_name_db = light_name
        except Exception as e:
            light_tmp.unlink(missing_ok=True)
            print(f"[recorder] light transcode failed: {e}", flush=True)
        cid = db_add(t0, classes, name, final.stat().st_size,
                     round(duration, 1), frames)
        # --- журнал: прив'язати кліп (оригінал + легка версія) до події ---
        if journal_id:
            journal_update_clip(journal_id, cid, name, light_name_db)
        with self.lock:
            self.last_clip = {"id": cid, "ts": t0, "classes": classes,
                              "path": name, "size": final.stat().st_size,
                              "duration": round(duration, 1),
                              "frames": frames, "light": light_name_db}
            self.error = None
        print(f"[recorder] {name}: {frames} frames, "
              f"{final.stat().st_size} B, light={light_name_db}, {classes}",
              flush=True)
        # --- Telegram: тільки ЛЕГКА версія кліпу ---
        from telegram_notify import TOKEN as TG_TOKEN, CHAT_ID as TG_CHAT
        if alert and light_name_db and TG_TOKEN and TG_CHAT:
            stamp = datetime.fromtimestamp(t0).strftime("%d.%m.%Y %H:%M:%S")
            deliver(light_final,
                    caption=f"🚨 Вдома нікого · рух: {', '.join(classes)}\n"
                            f"🕒 {stamp} · {duration:.0f} с · "
                            f"{light_final.stat().st_size // 1024} КБ (light)")
            if journal_id:
                journal_mark_sent(journal_id)

    @staticmethod
    def _size():
        """Розмір кадру анотованого стріму (всі кадри однакові)."""
        jpg = detector.snapshot()
        if jpg is None:
            return 640, 480
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return img.shape[1], img.shape[0]


recorder = ClipRecorder()
def _on_activity(ev):
    """Кожна подія журналу дій — у файл; зміна охорони — ще й у Telegram."""
    etype = "guard" if ev.get("kind") in ("away", "home") else "activity"
    event_log.write(etype, ev.get("ts"), None, kind=ev.get("kind"),
                    actor=ev.get("actor"), zone=ev.get("zone"),
                    duration=ev.get("duration"), count=ev.get("count"),
                    text=ev.get("text"), snap=ev.get("snap"))
    if etype == "guard":
        msg = ("🔒 Вдома нікого вже 5 хв — надсилаю всі детекції руху"
               if ev["kind"] == "away" else
               "🏠 Хтось удома — сповіщення про рух вимкнено")
        threading.Thread(target=send_message, args=(msg,), daemon=True).start()


activity_log = activity.ActivityLog(on_event=_on_activity)
recorder.start()


# ------------------------------------------------------------ camera poller
class CameraPoller(threading.Thread):
    """Раз на 2 с опитує /status і /settings камери."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.status = {}
        self.settings = {}
        self.online = False
        self.last_ok = 0.0
        self.stopped = False

    def run(self):
        while not self.stopped:
            st, se, ok = {}, {}, False
            try:
                r = requests.get(f"{CAM}/status", timeout=3)
                st = _safe_json(r.text)
                r = requests.get(f"{CAM}/settings", timeout=3)
                se = _safe_json(r.text)
                ok = bool(st) and bool(se)
            except Exception as e:
                print(f"[poller] {type(e).__name__}: {e}", flush=True)
            with self.lock:
                self.status, self.settings, self.online = st, se, ok
                if ok:
                    self.last_ok = time.time()
            time.sleep(2)

    def read(self):
        with self.lock:
            return dict(self.status), dict(self.settings), self.online


# ------------------------------------------------------------------ state
grabber = MjpegGrabber(STREAM_URL)
grabber.start()
detector = Detector(grabber)
poller = CameraPoller()
poller.start()
detector.start()


def state_payload():
    status, settings, online = poller.read()
    with detector.lock:
        dets = list(detector.dets)
        events = list(detector.events)
        fps, frames, motion = detector.fps, detector.frames, detector.motion
        det_error = detector.error
        sched_active = detector.sched_active
        sched_reason = detector.sched_reason
        sched_profile = detector.sched_profile
    frame_age = None
    if grabber.frame is not None:
        frame_age = 0  # останній кадр у пам'яті; вік не критичний
    return {
        "cam": {"online": online, "status": status, "settings": settings,
                "last_ok_ago_s": round(time.time() - poller.last_ok, 1)
                if poller.last_ok else None},
        "stream": {"url": STREAM_URL, "connected": grabber.is_alive()
                   and grabber.frame is not None,
                   "clients": status.get("stream_clients")},
        "detector": {"fps": fps, "frames": frames, "motion_pct": motion,
                     "detections": dets, "error": det_error,
                     "enabled": detector.enabled,
                     "schedule": {"active": sched_active,
                                  "reason": sched_reason,
                                  "profile": sched_profile}},
        "recorder": recorder.status(),
        "home": {**activity_log.guard_state(),
                 "alerts": activity_log.alerts_on()},
        "events": events,
    }


# ------------------------------------------------------------------ routes
HTML = Path(__file__).with_name("dashboard.html").read_text()


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


@app.route("/stream.mjpg")
def stream():
    def gen():
        last = None
        while True:
            jpg = detector.snapshot()
            if jpg is not None and jpg != last:
                last = jpg
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpg + b"\r\n")
            time.sleep(0.05)
    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    return jsonify(state_payload())


@app.route("/api/clips")
def api_clips():
    return jsonify({"clips": db_list(100)})


@app.route("/api/journal")
def api_journal():
    """Журнал руху: дата+час початку руху, класи, кліпи, статус Telegram."""
    return jsonify({"journal": journal_list(200)})


@app.route("/api/activity")
def api_activity():
    """Журнал дій: «прийшов додому», «кіт поїв»… — новіші першими.

    ?before=<ts> — сторінка старіших подій. До кожної — clip_id кліпу,
    що почався в межах 15 с від події (якщо є).
    """
    try:
        before = float(request.args.get("before") or 0) or None
        limit = max(1, min(500, int(request.args.get("limit") or 200)))
    except ValueError:
        return jsonify({"ok": False, "error": "bad before/limit"}), 400
    rows = activity.db_list(limit, before)
    clips = [(c["ts"], c["id"]) for c in db_list(1000)]
    for r in rows:
        near = [cid for ts, cid in clips if -15 <= ts - r["ts"] <= 15]
        r["clip_id"] = near[0] if near else None
    return jsonify({"ok": True, "activity": rows})


@app.route("/api/home", methods=["GET"])
def api_home_get():
    """Охорона: home | leaving | away, alerts — чи шлемо рух у Telegram."""
    return jsonify({"ok": True, **activity_log.guard_state(),
                    "alerts": activity_log.alerts_on()})


@app.route("/api/journal/file")
def api_journal_file():
    """Файловий журнал за день: ?day=YYYY-MM-DD (за замовчуванням — сьогодні)."""
    day = request.args.get("day") or datetime.now().strftime("%Y-%m-%d")
    try:
        datetime.strptime(day, "%Y-%m-%d")
    except ValueError:
        return jsonify({"ok": False, "error": "day must be YYYY-MM-DD"}), 400
    return jsonify({"ok": True, "day": day, "events": event_log.read_day(day)})


@app.route("/api/activity/snap/<name>")
def api_activity_snap(name):
    p = activity.snap_path(name)
    if p is None:
        return ("no such picture", 404)
    return send_file(p, mimetype="image/jpeg", max_age=86400)


@app.route("/api/activity/zones", methods=["GET"])
def api_zones_get():
    return jsonify({"ok": True, **activity.load_zones()})


@app.route("/api/activity/zones", methods=["POST"])
def api_zones_set():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "JSON object expected"}), 400
    try:
        cfg = activity.save_zones(data)
    except OSError as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    return jsonify({"ok": True, **cfg})


@app.route("/api/clips/<int:cid>/video")
def api_clip_video(cid):
    row = next((r for r in db_list(1000) if r["id"] == cid), None)
    if not row:
        return ("кліп не знайдено", 404)
    p = CLIPS_DIR / row["path"]
    if not p.exists():
        return ("файл кліпу відсутній", 404)
    return send_file(p, mimetype="video/mp4",
                     conditional=True, download_name=row["path"])


@app.route("/api/clips/record", methods=["POST"])
def api_clip_record():
    """Ручний тригер запису (кнопка «Записати зараз» на панелі)."""
    classes = request.get_json(silent=True) or {}
    recorder.trigger(classes.get("classes") or ["manual"])
    return jsonify({"queued": True})


@app.route("/api/detector/toggle", methods=["POST"])
def api_detector_toggle():
    """Увімкнути/вимкнути YOLO-детекцію (кнопка на панелі «Детекція руху»).

    POST {"enabled": true|false}. При вимкненні інференс не рахується
    (CPU звільняється), стрім продовжує йти без рамок, подій і запису.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data.get("enabled"), bool):
        return jsonify({"ok": False, "error": "потрібен enabled: bool"}), 400
    detector.enabled = data["enabled"]
    return jsonify({"ok": True, "enabled": detector.enabled})


@app.route("/api/control", methods=["POST"])
def api_control():
    data = request.get_json(silent=True) or {}
    var, val = data.get("var"), data.get("val")
    if not var or val is None:
        return jsonify({"ok": False, "error": "потрібні var і val"}), 400
    try:
        r = requests.get(f"{CAM}/control", params={"var": var, "val": val},
                         timeout=5)
        return jsonify({"ok": r.ok, "response": r.text}), r.status_code
    except requests.RequestException as e:
        return jsonify({"ok": False, "error": str(e)}), 502


@app.route("/api/frame_annotated")
def api_frame():
    jpg = detector.snapshot()
    if jpg is None:
        return ("стрім ще не підключився", 503)
    return Response(jpg, mimetype="image/jpeg")


@app.route("/api/snapshot")
def api_snapshot():
    try:
        r = requests.get(f"{CAM}/capture", timeout=8)
        return Response(r.content, mimetype="image/jpeg")
    except requests.RequestException as e:
        return (str(e), 502)


@app.route("/api/schedule", methods=["GET"])
def api_schedule_get():
    """Поточний графік детекції + що діє просто зараз.

    Читає motion_schedule.json — той самий файл, з яким звіряється
    Detector-потік, тому UI бачить рівно те, що бачить детектор.
    """
    return jsonify(motion_config.snapshot())


@app.route("/api/schedule", methods=["POST"])
def api_schedule_set():
    """Зберегти графік у motion_schedule.json (атомарний запис).

    Детектор підхоплює зміни на наступному кадрі — рестарт не потрібен.
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"ok": False, "error": "потрібен JSON-об'єкт"}), 400
    try:
        motion_config.save(data)
    except OSError as e:
        return jsonify({"ok": False, "error": f"не вдалось записати: {e}"}), 500
    return jsonify({"ok": True, **motion_config.snapshot()})


@app.route("/api/schedule/reset", methods=["POST"])
def api_schedule_reset():
    """Скинути графік до дефолту (детекція цілодобово)."""
    try:
        motion_config.save(motion_config.default_config())
    except OSError as e:
        return jsonify({"ok": False, "error": f"не вдалось записати: {e}"}), 500
    return jsonify({"ok": True, **motion_config.snapshot()})


if __name__ == "__main__":
    print(f"Dashboard on http://0.0.0.0:{PORT}  (stream: {STREAM_URL})")
    app.run(host="0.0.0.0", port=PORT, threaded=True,
            debug=False, use_reloader=False)