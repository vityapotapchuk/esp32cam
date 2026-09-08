#!/usr/bin/env python3
"""Санітарна перевірка детектора: еталонне зображення + живий кадр зі стріму."""
import sys
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/server/esp32cam")
from detect import MjpegGrabber, letterbox, postprocess, TARGET_CLASSES

sess = ort.InferenceSession(
    "/home/server/esp32cam/models/yolov8n.onnx",
    providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name


def run(img, label):
    inp, r, dx, dy = letterbox(img)
    blob = inp[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {inp_name: blob})[0]
    dets = postprocess(out, r, dx, dy, img.shape)
    print(f"{label}: {len(dets)} детекцій")
    for bx, conf, c in dets:
        print(f"  {TARGET_CLASSES[c]} {conf:.2f} box={bx.tolist()}")
    return dets


# 1) Еталон: bus.jpg (людина + автобус, відомий тест YOLO)
urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", "/tmp/bus.jpg")
bus = cv2.imread("/tmp/bus.jpg")
run(bus, "bus.jpg (еталон)")

# 2) Живий кадр зі стріму
g = MjpegGrabber("http://192.168.1.186:81/stream")
g.start()
import time
t0 = time.time()
while g.read() is None and time.time() - t0 < 15:
    time.sleep(0.1)
frame = g.read()
if frame is None:
    sys.exit("стрім недоступний")
cv2.imwrite("/home/server/esp32cam/live.jpg", frame)
print(f"live.jpg збережено, {frame.shape[1]}x{frame.shape[0]}")
run(frame, "live-кадр")