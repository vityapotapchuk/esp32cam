#!/usr/bin/env python3
"""Відправка кліпів детектора руху в Telegram (sendVideo).

Токен/чат читаються з .env поруч із модулем (TELEGRAM_BOT_TOKEN,
TELEGRAM_CHAT_ID); змінні оточення мають пріоритет, якщо задані.

Надійність: невдалі відправки пишуться в .tg_pending і повторюються
при наступній спробі; успішні — логуються в .tg_sent.log (аудит).
"""
import os
import socket
import time
from pathlib import Path

import requests
import urllib3.util.connection as urllib3_cn

# api.telegram.org має IPv6, який на цьому хості обривається —
# примусово IPv4 для всіх з'єднань urllib3 (requests).
urllib3_cn.allowed_gai_family = lambda: socket.AF_INET


def _load_env():
    env = Path(__file__).with_name(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


_load_env()

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
API = f"https://api.telegram.org/bot{TOKEN}" if TOKEN else None
MAX_BYTES = 49 * 1024 * 1024  # ліміт Bot API для sendVideo — 50 МБ

SENT_LOG = Path(__file__).with_name(".tg_sent.log")
PENDING = Path(__file__).with_name(".tg_pending")


def configured():
    return bool(TOKEN and CHAT_ID)


def _log_sent(name, size):
    with SENT_LOG.open("a") as f:
        f.write(f"{int(time.time())}\t{name}\t{size}\n")


def _load_pending():
    if not PENDING.exists():
        return []
    return [l.strip() for l in PENDING.read_text().splitlines() if l.strip()]


def _save_pending(items):
    if items:
        PENDING.write_text("\n".join(items) + "\n")
    else:
        PENDING.unlink(missing_ok=True)


def send_video(path, caption=""):
    """Відправляє MP4-файл у Telegram. True — успіх."""
    if not configured():
        print("[tg] TELEGRAM_BOT_TOKEN/CHAT_ID не задані — пропускаю",
              flush=True)
        return False
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        print(f"[tg] файл відсутній/порожній: {p}", flush=True)
        return False
    size = p.stat().st_size
    if size > MAX_BYTES:
        print(f"[tg] файл завеликий ({size} B) — пропускаю", flush=True)
        return False
    try:
        with p.open("rb") as f:
            r = requests.post(
                f"{API}/sendVideo",
                data={"chat_id": CHAT_ID, "caption": caption[:1024],
                      "supports_streaming": "true"},
                files={"video": (p.name, f, "video/mp4")},
                timeout=120)
        ok = r.ok and r.json().get("ok", False)
        if ok:
            print(f"[tg] sent {p.name} ({size} B)", flush=True)
            _log_sent(p.name, size)
        else:
            print(f"[tg] sendVideo fail: {r.status_code} {r.text[:200]}",
                  flush=True)
        return ok
    except requests.RequestException as e:
        print(f"[tg] sendVideo error: {e}", flush=True)
        return False


def deliver(path, caption=""):
    """Відправити кліп; при невдачі — у чергу повтору. Потім retry."""
    ok = send_video(path, caption)
    if not ok and Path(path).exists():
        items = _load_pending()
        items.append(f"{path}\t{caption}")
        _save_pending(items)
        print(f"[tg] у черзі повтору: {path}", flush=True)
    _retry_pending()
    return ok


def _retry_pending():
    """Повторити невдалі відправки; успішні прибрати з черги."""
    remaining = []
    for item in _load_pending():
        path, _, caption = item.partition("\t")
        if not send_video(path, caption):
            remaining.append(item)
        else:
            print(f"[tg] retry ok: {path}", flush=True)
    _save_pending(remaining)