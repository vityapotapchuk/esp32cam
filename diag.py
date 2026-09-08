#!/usr/bin/env python3
"""Діагностика: сирі детекції на живому кадрі при низькому порозі."""
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/server/esp32cam")
import detect as D

sess = ort.InferenceSession(
    "/home/server/esp32cam/models/yolov8n.onnx",
    providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

# Перевизначаємо поріг на низький для діагностики
D.CONF_THRES = 0.05

g = D.MjpegGrabber("http://192.168.1.186:81/stream")
g.start()
t0 = time.time()
while g.read() is None and time.time() - t0 < 15:
    time.sleep(0.1)
frame = g.read()
if frame is None:
    sys.exit("стрім недоступний")

# Кілька кадрів: середній по 5 знімах зменшить шум
dets_all = []
for i in range(5):
    f = g.read()
    if f is None:
        continue
    inp, r, dx, dy = D.letterbox(f)
    blob = inp[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {inp_name: blob})[0]
    dets = D.postprocess(out, r, dx, dy, f.shape)
    dets_all.extend(dets)
    time.sleep(0.3)

print(f"усього сирих детекцій (conf>0.05, 5 кадрів): {len(dets_all)}")
from collections import Counter
cnt = Counter(D.TARGET_CLASSES[c] for _, _, c in dets_all)
print("по класах:", dict(cnt))
best = {}
for bx, conf, c in dets_all:
    name = D.TARGET_CLASSES[c]
    if name not in best or conf > best[name][0]:
        best[name] = (conf, bx.tolist())
for name, (conf, bx) in sorted(best.items(), key=lambda kv: -kv[1][0]):
    print(f"  найкраща {name}: conf={conf:.2f} box={bx}")