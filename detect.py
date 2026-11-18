#!/usr/bin/env python3
"""Детекція людей і тварин з MJPEG-стріму ESP32-CAM.

Використання:
  ~/cv-venv/bin/python detect.py --duration 30            # лог детекцій у консоль
  ~/cv-venv/bin/python detect.py --duration 30 --save out # зберегти анотовані кадри
"""
import argparse
import collections
import io
import json
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

STREAM_URL = "http://192.168.1.186:81/stream"
MODEL_PATH = __file__.rsplit("/", 1)[0] + "/models/yolov8n.onnx"

# COCO: person + усі тварини
TARGET_CLASSES = {
    0: "person", 14: "bird", 15: "cat", 16: "dog", 17: "horse",
    18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe",
}
CONF_THRES = 0.35
NMS_THRES = 0.45
IMGSZ = 640
FLIP180 = False  # орієнтацію задає сама камера (vflip/hmirror у HomeOS), інакше Boxes показує картинку перевернутою
GAMMA = 1.8     # програмне висвітлення тіней (None = вимкнути)


# LUT рахується ОДИН раз на імпорті (раніше — на кожному кадрі:
# 256-елементний Python-цикл по 5.5 fps просто так).
_GAMMA_LUT = np.array([((i / 255.0) ** (1.0 / GAMMA)) * 255
                       for i in range(256)]).clip(0, 255).astype(np.uint8)


def enhance(img):
    if not GAMMA:
        return img
    return cv2.LUT(img, _GAMMA_LUT)


class MjpegGrabber(threading.Thread):
    """Тягне MJPEG-стрім у фоновому потоці, тримає останній декодований кадр."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

    def run(self):
        while not self.stopped:
            try:
                req = urllib.request.Request(self.url)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    buf = b""
                    for chunk in iter(lambda: resp.read(4096), b""):
                        buf += chunk
                        # JPEG: від SOI (FFD8) до EOI (FFD9)
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9", start + 2)
                        if start != -1 and end != -1:
                            jpg = buf[start:end + 2]
                            buf = buf[end + 2:]
                            img = cv2.imdecode(
                                np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                            if img is not None:
                                with self.lock:
                                    self.frame = img
            except Exception as e:
                sys.stderr.write(f"[grabber] {e}; reconnect in 2s\n")
                time.sleep(2)

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            img = self.frame.copy()
        return cv2.flip(img, -1) if FLIP180 else img

    def stop(self):
        self.stopped = True


def letterbox(img, new=IMGSZ):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), 114, np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


def postprocess(output, r, dx, dy, orig_shape):
    """output: (1, 84, N) -> список [x1,y1,x2,y2,score,cls] в координатах оригіналу."""
    pred = output[0].T  # (N, 84)
    scores = pred[:, 4:]
    cls_ids = scores.argmax(axis=1)
    confs = scores.max(axis=1)
    keep = confs > CONF_THRES
    pred, confs, cls_ids = pred[keep], confs[keep], cls_ids[keep]
    if len(pred) == 0:
        return []
    # xywh (центр) -> xyxy, знімаємо letterbox
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w / 2 - dx) / r
    y1 = (cy - h / 2 - dy) / r
    x2 = (cx + w / 2 - dx) / r
    y2 = (cy + h / 2 - dy) / r
    oh, ow = orig_shape[:2]
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    idx = cv2.dnn.NMSBoxes(
        boxes.tolist(), confs.tolist(), CONF_THRES, NMS_THRES)
    out = []
    for i in np.array(idx).flatten():
        c = int(cls_ids[i])
        if c in TARGET_CLASSES:
            bx = boxes[i].astype(int)
            bx[[0, 2]] = bx[[0, 2]].clip(0, ow)
            bx[[1, 3]] = bx[[1, 3]].clip(0, oh)
            out.append((bx, float(confs[i]), c))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=30)
    ap.add_argument("--save", metavar="DIR", help="зберігати анотовані кадри")
    ap.add_argument("--url", default=STREAM_URL)
    args = ap.parse_args()

    sess = ort.InferenceSession(
        MODEL_PATH, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    grabber = MjpegGrabber(args.url)
    grabber.start()
    print(f"Очікую перший кадр зі {args.url} ...")
    t0 = time.time()
    while grabber.read() is None:
        if time.time() - t0 > 15:
            sys.exit("Кадр не отримано за 15 с — стрім недоступний?")
        time.sleep(0.1)
    print("Стрім підключено.")

    if args.save:
        import os
        os.makedirs(args.save, exist_ok=True)

    fps_hist = collections.deque(maxlen=30)
    end = time.time() + args.duration
    n_saved = 0
    last_log = 0.0
    while time.time() < end:
        frame = grabber.read()
        if frame is None:
            time.sleep(0.05)
            continue
        frame = enhance(frame)
        t1 = time.time()
        inp, r, dx, dy = letterbox(frame)
        blob = inp[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        output = sess.run(None, {inp_name: blob})[0]
        dets = postprocess(output, r, dx, dy, frame.shape)
        dt = time.time() - t1
        fps_hist.append(1.0 / dt if dt > 0 else 0)

        if dets and time.time() - last_log > 1.0:
            last_log = time.time()
            parts = [f"{TARGET_CLASSES[c]} {conf:.2f}" for _, conf, c in dets]
            print(f"[{time.strftime('%H:%M:%S')}] {', '.join(parts)}")
        if args.save and dets and n_saved < 20:
            vis = frame.copy()
            for (bx, conf, c) in dets:
                cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
                cv2.putText(vis, f"{TARGET_CLASSES[c]} {conf:.2f}",
                            (bx[0], max(bx[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            n_saved += 1
            cv2.imwrite(f"{args.save}/det_{n_saved:03d}.jpg", vis)

    fps = np.mean(fps_hist) if fps_hist else 0
    print(json.dumps({
        "avg_inference_fps": round(float(fps), 1),
        "frames_processed": len(fps_hist),
        "saved": n_saved,
    }))


if __name__ == "__main__":
    main()