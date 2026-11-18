#!/usr/bin/env python3
"""Графік роботи детекції руху — конфіг, який читає сам детектор.

Файл motion_schedule.json лежить поруч із dashboard.py. Detector-потік
звіряється з ним на кожному кадрі (з кешем за mtime), тому розклад
працює без відкритого браузера і без рестарту сервісу: достатньо
записати файл — наступний кадр уже бачить нові правила.

Структура конфігу:
{
  "version": 1,
  "enabled": true,                   # master-перемикач, пріоритет над сіткою
  "day_profile": ["weekday", ...],   # 7 днів, Пн..Нд -> ім'я профілю
  "profiles": {
     "weekday": {"label": "Будні",   "grid": [[0|1] * 24] * 7},
     "weekend": {"label": "Вихідні", "grid": [[0|1] * 24] * 7},
     "custom":  {"label": "Власний", "grid": [[0|1] * 24] * 7}
  },
  "sensitivity": 73,                 # 0..100 -> поріг впевненості YOLO
  "min_area_pct": 0.4,               # мін. площа рамки, % площі кадру
  "zone": null,                     # null = весь кадр, або {"points": [[x,y],...]} 0..1
  "day_windows": null               # null = використовувати grid вище;
                                    # інакше — 7 інтервалів "HH:MM" на день
}

day_windows — спрощений графік «початок/кінець» на кожен день тижня
(Пн..Нд). Кожен елемент: null (день вимкнено) або
{"start": "HH:MM", "end": "HH:MM"}. Якщо end < start — інтервал
переходить через північ (22:00 -> 06:00). Якщо start == end — день
вимкнено. Коли day_windows задано, він має пріоритет над profiles/grid;
коли null — працює старий сітковий графік (сумісність зі старою
вкладкою «Camera / Motion»).
"""
import json
import os
import threading
import time

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "motion_schedule.json")

DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"]
PROFILE_NAMES = ("weekday", "weekend", "custom")

# Поріг впевненості YOLO з повзунка чутливості.
# sens=0 -> 0.90 (лише дуже впевнені детекції), sens=100 -> 0.15.
# Дефолт 73 -> 0.3525, тобто історичний CONF_THRES = 0.35 з detect.py.
CONF_MAX = 0.90
CONF_MIN = 0.15

_lock = threading.Lock()
_cache = {"mtime": None, "cfg": None, "checked": 0.0}


def _blank_grid():
    return [[1] * 24 for _ in range(7)]


def default_config():
    """Дефолт: детекція увімкнена цілодобово — поведінка не змінюється,
    доки користувач сам не звузить графік."""
    return {
        "version": 1,
        "enabled": True,
        "day_profile": ["weekday"] * 5 + ["weekend"] * 2,
        "profiles": {
            "weekday": {"label": "Будні", "grid": _blank_grid()},
            "weekend": {"label": "Вихідні", "grid": _blank_grid()},
            "custom": {"label": "Власний", "grid": _blank_grid()},
        },
        "sensitivity": 73,
        "min_area_pct": 0.4,
        "zone": None,
        "day_windows": None,
    }


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _parse_hhmm(s):
    """'HH:MM' -> хвилини від півночі, або None якщо формат невірний."""
    if not isinstance(s, str):
        return None
    parts = s.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return h * 60 + m


def _fmt_hhmm(minutes):
    minutes = int(minutes) % (24 * 60)
    return "%02d:%02d" % (minutes // 60, minutes % 60)


def _normalize_windows(raw):
    """Привести day_windows до 7 валідних інтервалів або None.

    None означає «не задано» -> працює старий сітковий графік.
    Кожен день: null (вимкнено) або {"start","end"} у "HH:MM".
    """
    if not isinstance(raw, list):
        return None
    out = []
    for d in range(7):
        item = raw[d] if d < len(raw) else None
        if not isinstance(item, dict):
            out.append(None)
            continue
        start = _parse_hhmm(item.get("start"))
        end = _parse_hhmm(item.get("end"))
        if start is None or end is None:
            out.append(None)
            continue
        out.append({"start": _fmt_hhmm(start), "end": _fmt_hhmm(end)})
    return out


def _window_active(win, hour, minute):
    """Чи попадає (hour, minute) в інтервал. Підтримує перехід через північ."""
    if not win:
        return False
    start = _parse_hhmm(win.get("start"))
    end = _parse_hhmm(win.get("end"))
    if start is None or end is None:
        return False
    if start == end:
        return False  # нульова довжина = день вимкнено
    now = hour * 60 + minute
    if start < end:
        return start <= now < end
    # через північ: 22:00 -> 06:00
    return now >= start or now < end


def normalize(raw):
    """Привести довільний вхід до валідного конфігу.

    Ніколи не кидає: будь-яке сміття замінюється дефолтом. Це важливо,
    бо файл читає детектор у циклі — поганий конфіг не має його вбити.
    """
    base = default_config()
    if not isinstance(raw, dict):
        return base

    out = {"version": 1}
    out["enabled"] = bool(raw.get("enabled", base["enabled"]))

    # профілі
    profiles = {}
    src_profiles = raw.get("profiles")
    if not isinstance(src_profiles, dict):
        src_profiles = {}
    for name in PROFILE_NAMES:
        src = src_profiles.get(name)
        if not isinstance(src, dict):
            src = {}
        label = src.get("label")
        if not isinstance(label, str) or not label.strip():
            label = base["profiles"][name]["label"]
        grid_src = src.get("grid")
        grid = []
        for d in range(7):
            row_src = grid_src[d] if isinstance(grid_src, list) and d < len(grid_src) else None
            row = []
            for h in range(24):
                v = row_src[h] if isinstance(row_src, list) and h < len(row_src) else 1
                row.append(1 if v else 0)
            grid.append(row)
        profiles[name] = {"label": label.strip()[:40], "grid": grid}
    out["profiles"] = profiles

    # який профіль діє в кожен день тижня
    dp_src = raw.get("day_profile")
    day_profile = []
    for d in range(7):
        name = dp_src[d] if isinstance(dp_src, list) and d < len(dp_src) else None
        day_profile.append(name if name in PROFILE_NAMES else base["day_profile"][d])
    out["day_profile"] = day_profile

    # чутливість
    try:
        sens = int(raw.get("sensitivity", base["sensitivity"]))
    except (TypeError, ValueError):
        sens = base["sensitivity"]
    out["sensitivity"] = _clamp(sens, 0, 100)

    # мінімальний розмір об'єкта
    try:
        area = float(raw.get("min_area_pct", base["min_area_pct"]))
    except (TypeError, ValueError):
        area = base["min_area_pct"]
    out["min_area_pct"] = round(_clamp(area, 0.0, 100.0), 2)

    # зона детекції: полігон у нормалізованих координатах 0..1
    zone = raw.get("zone")
    clean_zone = None
    if isinstance(zone, dict):
        pts = zone.get("points")
        if isinstance(pts, list) and len(pts) >= 3:
            clean = []
            for p in pts[:32]:
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    try:
                        clean.append([round(_clamp(float(p[0]), 0.0, 1.0), 4),
                                      round(_clamp(float(p[1]), 0.0, 1.0), 4)])
                    except (TypeError, ValueError):
                        clean = []
                        break
                else:
                    clean = []
                    break
            if len(clean) >= 3:
                clean_zone = {"points": clean}
    out["zone"] = clean_zone

    # спрощений графік «початок/кінець» на кожен день (нова вкладка)
    out["day_windows"] = _normalize_windows(raw.get("day_windows"))
    return out


def load(force=False):
    """Прочитати конфіг. Кеш скидається за mtime, перевірка — не частіше
    одного разу на секунду (детектор кличе це на кожному кадрі)."""
    now = time.time()
    with _lock:
        if not force and _cache["cfg"] is not None and now - _cache["checked"] < 1.0:
            return _cache["cfg"]
        _cache["checked"] = now
        try:
            mtime = os.stat(CONFIG_PATH).st_mtime
        except OSError:
            mtime = None
        if not force and _cache["cfg"] is not None and mtime == _cache["mtime"]:
            return _cache["cfg"]
        cfg = None
        if mtime is not None:
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    cfg = normalize(json.load(f))
            except (OSError, ValueError):
                cfg = None
        if cfg is None:
            cfg = default_config()
        _cache["cfg"] = cfg
        _cache["mtime"] = mtime
        return cfg


def save(cfg):
    """Атомарний запис: детектор ніколи не побачить напівзаписаний файл."""
    clean = normalize(cfg)
    tmp = CONFIG_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CONFIG_PATH)
    with _lock:
        _cache["cfg"] = clean
        _cache["mtime"] = os.stat(CONFIG_PATH).st_mtime
        _cache["checked"] = time.time()
    return clean


def conf_thres(cfg=None):
    """Поріг впевненості YOLO з повзунка чутливості."""
    cfg = cfg or load()
    sens = cfg["sensitivity"]
    return round(CONF_MAX - (CONF_MAX - CONF_MIN) * (sens / 100.0), 4)


def active_now(cfg=None, ts=None):
    """Чи має детекція працювати зараз.

    Повертає (active, reason, profile):
      active  — bool, рахувати інференс чи ні
      reason  — 'master-off' | 'schedule-off' | 'schedule-on'
      profile — ім'я профілю, що діє сьогодні
    Master-перемикач має пріоритет над сіткою.
    """
    cfg = cfg or load()
    if not cfg["enabled"]:
        return False, "master-off", cfg["day_profile"][(ts or time.localtime()).tm_wday]
    lt = time.localtime(ts) if ts is not None else time.localtime()
    day = lt.tm_wday          # 0 = понеділок
    hour = lt.tm_hour
    profile = cfg["day_profile"][day]

    # Спрощений графік (вкладка «Графік детекції») має пріоритет над сіткою.
    windows = cfg.get("day_windows")
    if windows is not None:
        on = _window_active(windows[day], hour, lt.tm_min)
        return on, ("schedule-on" if on else "schedule-off"), profile

    grid = cfg["profiles"][profile]["grid"]
    on = bool(grid[day][hour])
    return on, ("schedule-on" if on else "schedule-off"), profile


def point_in_polygon(x, y, points):
    """Променевий тест: чи точка (x,y) усередині полігона points.

    Координати нормалізовані 0..1. Використовується для зони детекції.
    """
    inside = False
    n = len(points)
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            xcross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xcross:
                inside = not inside
        j = i
    return inside


def zone_points(cfg=None):
    """Точки зони або None (уся площа кадру)."""
    cfg = cfg or load()
    z = cfg.get("zone")
    return z["points"] if z else None


def snapshot():
    """Стан для API/UI: конфіг + що діє просто зараз."""
    cfg = load(force=True)
    active, reason, profile = active_now(cfg)
    lt = time.localtime()
    return {
        "config": cfg,
        "state": {
            "active": active,
            "reason": reason,
            "profile": profile,
            "profile_label": cfg["profiles"][profile]["label"],
            "day": lt.tm_wday,
            "hour": lt.tm_hour,
            "conf_thres": conf_thres(cfg),
            "now": time.strftime("%Y-%m-%d %H:%M:%S", lt),
        },
        "path": CONFIG_PATH,
    }
