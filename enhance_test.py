#!/usr/bin/env python3
"""Чи покращує програмне висвітлення детекцію в темряві?"""
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, "/home/server/esp32cam")
import detect as D

D.CONF_THRES = 0.05
sess = ort.InferenceSession(
    "/home/server/esp32cam/models/yolov8n.onnx",
    providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

g = D.MjpegGrabber("http://192.168.1.186:81/stream")
g.start()
t0 = time.time()
while g.read() is None and time.time() - t0 < 15:
    time.sleep(0.1)
time.sleep(1)

# Гамма-LUT: яскравіші тіні
def gamma_lut(gamma):
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)]).clip(0, 255).astype(np.uint8)
    return lut

LUT = gamma_lut(1.8)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def variants(f):
    yield "оригінал", f
    yield "гамма1.8", cv2.LUT(f, LUT)
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    yield "clahe", cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    yield "гамма+clahe", cv2.cvtColor(
        cv2.cvtColor(cv2.LUT(f, LUT), cv2.COLOR_BGR2LAB), cv2.COLOR_LAB2BGR)

def infer(f):
    inp, r, dx, dy = D.letterbox(f)
    blob = inp[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    out = sess.run(None, {inp_name: blob})[0]
    return D.postprocess(out, r, dx, dy, f.shape)

# по 5 кадрів на варіант
from collections import Counter
results = {}
for name, _ in variants(g.read()):
    cnt = Counter()
    best = 0.0
    for i in range(5):
        f = g.read()
        if f is None:
            continue
        for vname, vf in variants(f):
            if vname != name:
                continue
            for _, conf, c in infer(vf):
                cnt[D.TARGET_CLASSES[c]] += 1
                best = max(best, conf)
    results[name] = (sum(cnt.values()), dict(cnt), round(best, 2))

for name, (total, cnt, best) in results.items():
    print(f"{name:14s} детекцій: {total:3d}  {cnt}  max_conf={best}")