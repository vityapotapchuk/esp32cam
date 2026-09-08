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
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, Response, jsonify, request

from detect import (MODEL_PATH, STREAM_URL, TARGET_CLASSES, MjpegGrabber,
                    enhance, letterbox, postprocess)

CAM = "http://192.168.1.186"
PORT = 8084
JPEG_QUALITY = 80

FRAMESIZE_NAMES = {0: "qqvga", 1: "qqvga2", 2: "qcif", 3: "cif", 4: "qvga",
                   5: "vga", 6: "svga", 7: "xga", 8: "sxga", 9: "uxga",
                   10: "hd", 11: "fhd"}

app = Flask(__name__)


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

    def run(self):
        sess = ort.InferenceSession(MODEL_PATH,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        fps_hist = collections.deque(maxlen=30)
        prev_gray = None
        ema = 0.0

        while not self.stopped:
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
                with self.lock:
                    if ok:
                        self.annotated = jpg.tobytes()
                    self.dets = dets
                    self.fps = round(float(np.mean(fps_hist)), 1) if fps_hist else 0.0
                    self.frames += 1
                    self.motion = round(ema, 1)
                    if classes and not self.last_classes:
                        self._event(now, "appear", sorted(classes))
                    elif classes and classes != self.last_classes:
                        self._event(now, "change", sorted(classes))
                    elif not classes and self.last_classes:
                        self._event(now, "clear", [])
                    self.last_classes = classes
            except Exception as e:  # не вмирати від разових збоїв
                self.error = str(e)
                time.sleep(0.5)

    def _event(self, ts, etype, classes):
        self.events.append({"t": ts, "type": etype, "classes": classes})
        del self.events[:-50]

    def snapshot(self):
        with self.lock:
            return self.annotated


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
                     "detections": dets, "error": det_error},
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


if __name__ == "__main__":
    print(f"Dashboard on http://0.0.0.0:{PORT}  (stream: {STREAM_URL})")
    app.run(host="0.0.0.0", port=PORT, threaded=True,
            debug=False, use_reloader=False)