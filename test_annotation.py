#!/usr/bin/env python3
"""Перевірка малювання рамок: синтетичний кадр + блок анотації з dashboard.py."""
import cv2
import numpy as np

TARGET_CLASSES = ["person", "cat", "dog"]

# сірий кадр 640x480
frame = np.full((480, 640, 3), 90, dtype=np.uint8)
raw = [((100, 120, 300, 400), 0.91, 0),   # person
       ((380, 60, 560, 200), 0.74, 1)]    # cat

vis = frame
for (bx, conf, c) in raw:
    color = (0, 255, 255) if c == 0 else (0, 170, 255)
    x1, y1, x2, y2 = map(int, bx)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    L = max(10, (x2 - x1 + y2 - y1) // 14)
    t = 3
    for (cx, cy, ddx, ddy) in (
            (x1, y1, 1, 1), (x2, y1, -1, 1),
            (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(vis, (cx, cy), (cx + ddx * L, cy), color, t)
        cv2.line(vis, (cx, cy), (cx, cy + ddy * L), color, t)
    label = f"{TARGET_CLASSES[c]} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    ytop = max(y1 - th - 8, 0)
    cv2.rectangle(vis, (x1, ytop), (x1 + tw + 6, ytop + th + 6), color, -1)
    cv2.putText(vis, label, (x1 + 3, ytop + th + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

cv2.imwrite("/home/server/esp32cam/frames/_annotation_test.jpg", vis)
print("saved, shape:", vis.shape)