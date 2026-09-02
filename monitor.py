#!/usr/bin/env python3
"""Серійний монітор для ESP32-CAM з апаратним ресетом через DTR/RTS."""
import serial, time, sys

port = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)

# Ресет у run-режим: IO0=HIGH (dtr False), імпульс EN (rts True -> False)
port.dtr = False
port.rts = True
time.sleep(0.1)
port.rts = False
time.sleep(0.05)
port.reset_input_buffer()

end = time.time() + 12
while time.time() < end:
    data = port.read(1024)
    if data:
        sys.stdout.write(data.decode('utf-8', errors='replace'))
        sys.stdout.flush()
port.close()