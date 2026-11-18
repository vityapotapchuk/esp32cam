#!/usr/bin/env python3
"""Файловий журнал детектора: час + об'єкти кожної події.

Один файл на день: journal/YYYY-MM-DD.jsonl, один рядок = одна подія (JSON).
Файли лише дописуються, тому їх можна читати будь-чим (tail -f, jq, Excel
через імпорт JSON) навіть під час роботи детектора.

Поля рядка:
    time     "2026-11-17 19:42:05" — локальний час
    ts       1763426525.31         — unix-час
    type     motion   — початок руху (YOLO побачив об'єкти)
             activity — подія журналу дій (came home, cat ate…)
             guard    — охорона: away (вдома нікого) / home (хтось удома)
    objects  [{"cls": "person", "conf": 0.81, "box": [x1, y1, x2, y2]}, …]
    kind, text, actor, zone, duration — для activity/guard
    alert    true — подію відправлено/відправляється в Telegram (вдома нікого)

Ті самі події лежать і в journal.db (motion_log, activity_log) — там їх
читає HomeOS; цей файл — простий архів для людини.
"""
import json
import threading
import time
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "journal"
_lock = threading.Lock()


def path_for(ts):
    return LOG_DIR / (datetime.fromtimestamp(ts).strftime("%Y-%m-%d") + ".jsonl")


def write(etype, ts=None, objects=None, **fields):
    """Дописати подію. Ніколи не кидає винятків — журнал не має валити детектор."""
    ts = time.time() if ts is None else float(ts)
    row = {"time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
           "ts": round(ts, 2), "type": etype, "objects": objects or []}
    row.update({k: v for k, v in fields.items() if v is not None})
    try:
        with _lock:
            LOG_DIR.mkdir(exist_ok=True)
            with path_for(ts).open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        pass
    return row


def read_day(day):
    """Події за день 'YYYY-MM-DD' (старіші першими)."""
    p = LOG_DIR / f"{day}.jsonl"
    out = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out
