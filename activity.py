#!/usr/bin/env python3
"""Журнал дій у кімнаті: з сирих детекцій YOLO — людські події.

Детектор бачить лише «person / dog / cat у кадрі». Цей модуль перетворює
потік детекцій на події, які можна читати як щоденник:

    18:02  Person came home            (з'явилась у зоні «двері»)
    18:40  Cat ate · 2 min             (кіт пробув біля миски)
    19:15  Person left home · 1 h 13 min

Як це працює
------------
* Групи: person — люди; pet — усі тварини (YOLO часто бачить кота як
  dog/bird/horse — це той самий кіт). Ім'я тварини береться з конфігу.
* Присутність групи: з'явилась, коли група трималась у кадрі ARRIVE_S
  секунд; пішла, коли її не було GONE_S секунд (YOLO пропускає кадри —
  без цього журнал засмічувався б).
* Точка об'єкта — середина нижньої сторони рамки («ноги»): вона краще
  за центр показує, де людина чи кіт стоїть на підлозі.
* Зони (activity_zones.json) — полігони 0..1, які власник малює в HomeOS:
    door   — вхідні двері: поява тут = «прийшла додому»,
             зникнення тут = «пішла з дому»;
    bowl   — миска: кіт пробув ≥ min_s → «кіт поїв»;
    water  — вода → «кіт попив»;
    litter — лоток → «кіт сходив у лоток»;
    place  — будь-яке інше місце (диван, стіл): «X біля <назва>».
* Візит у зону: група є в зоні; закінчується, коли її немає в зоні
  ZONE_GAP_S секунд. Подія пишеться в кінці візиту, з тривалістю.

Режим «вдома нікого» (охорона)
------------------------------
Людина вийшла через двері (leave_home) → стан «leaving». Якщо за
AWAY_AFTER_S (5 хв) людина не з'явилась знову → стан «away»: детектор
шле в Telegram кожну детекцію руху. Щойно людина знову вдома → стан
«home», сповіщення вимикаються (з запасом ALERT_GRACE_S, щоб кліп
того, хто зайшов, теж пішов у Telegram). Стан лежить у home_state.json
і переживає рестарт сервісу.

Події лежать у journal.db, таблиця activity_log; до кожної — JPEG-кадр
у activity/ (останні MAX_SNAPS файлів).
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZONES_PATH = HERE / "activity_zones.json"
JOURNAL_DB = HERE / "journal.db"
SNAP_DIR = HERE / "activity"
MAX_SNAPS = 3000

PET_CLASSES = {"cat", "dog", "bird", "horse", "sheep", "cow", "bear", "teddy bear"}
ARRIVE_S = {"person": 1.0, "pet": 2.0}     # скільки триматись, щоб «з'явитись»
GONE_S = {"person": 20.0, "pet": 40.0}     # скільки не бачити, щоб «піти»
ZONE_GAP_S = 8.0                           # пауза, після якої візит у зону скінчився

AWAY_AFTER_S = 300.0                       # скільки нікого після «пішов з дому» → охорона
ALERT_GRACE_S = 60.0                       # ще стільки шлемо після повернення людини
GUARD_PATH = HERE / "home_state.json"

KINDS = ("door", "bowl", "water", "litter", "place")
DEFAULT_MIN_S = {"door": 0, "bowl": 15, "water": 5, "litter": 20, "place": 30}


# ------------------------------------------------------------------ зони
def default_zones():
    return {"version": 1, "pet_name": "cat", "zones": []}


def normalize_zones(raw):
    """Перевірити і почистити конфіг зон (приходить з HomeOS)."""
    out = default_zones()
    if not isinstance(raw, dict):
        return out
    name = str(raw.get("pet_name") or "cat").strip()[:24]
    out["pet_name"] = name or "cat"
    zones = []
    for i, z in enumerate(raw.get("zones") or []):
        if not isinstance(z, dict) or len(zones) >= 12:
            continue
        kind = z.get("kind") if z.get("kind") in KINDS else "place"
        pts = []
        for p in z.get("points") or []:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            pts.append([round(min(1.0, max(0.0, x)), 4), round(min(1.0, max(0.0, y)), 4)])
        if len(pts) < 3 or len(pts) > 32:
            continue
        try:
            min_s = int(z.get("min_s", DEFAULT_MIN_S[kind]))
        except (TypeError, ValueError):
            min_s = DEFAULT_MIN_S[kind]
        label = str(z.get("name") or kind).strip()[:32] or kind
        zid = "".join(ch for ch in str(z.get("id") or "") if ch.isalnum() or ch in "-_")[:24]
        zones.append({"id": zid or f"z{i + 1}", "name": label, "kind": kind,
                      "min_s": max(0, min(3600, min_s)), "points": pts})
    out["zones"] = zones
    return out


_cache = {"mtime": None, "cfg": None}


def load_zones():
    """Конфіг зон; файл перечитується лише коли змінився (детектор кличе щокадру)."""
    try:
        mtime = ZONES_PATH.stat().st_mtime
    except OSError:
        return default_zones()
    if _cache["mtime"] != mtime:
        try:
            _cache["cfg"] = normalize_zones(json.loads(ZONES_PATH.read_text()))
        except (OSError, ValueError):
            _cache["cfg"] = default_zones()
        _cache["mtime"] = mtime
    return _cache["cfg"]


def save_zones(raw):
    cfg = normalize_zones(raw)
    tmp = ZONES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    os.replace(tmp, ZONES_PATH)
    return cfg


def point_in_polygon(x, y, pts):
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi:
            inside = not inside
        j = i
    return inside


# ------------------------------------------------------------------ база
def db_init():
    con = sqlite3.connect(JOURNAL_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS activity_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL NOT NULL,            -- unix-час події (для візитів — початок)
        local TEXT NOT NULL,         -- 'YYYY-MM-DD HH:MM:SS'
        kind TEXT NOT NULL,          -- arrive|appear|leave_home|leave|eat|drink|litter|visit
        actor TEXT NOT NULL,         -- person | pet
        zone TEXT,                   -- назва зони (якщо є)
        duration REAL,               -- секунд (візит / скільки був удома)
        count INTEGER,               -- скільки об'єктів групи було в кадрі
        text TEXT NOT NULL,          -- готовий текст англійською
        snap TEXT)                   -- ім'я JPEG у activity/""")
    con.execute("CREATE INDEX IF NOT EXISTS activity_ts ON activity_log(ts)")
    con.commit()
    con.close()


def db_list(limit=200, before=None):
    con = sqlite3.connect(JOURNAL_DB)
    con.row_factory = sqlite3.Row
    if before:
        rows = con.execute("SELECT * FROM activity_log WHERE ts < ? ORDER BY ts DESC LIMIT ?",
                           (float(before), limit)).fetchall()
    else:
        rows = con.execute("SELECT * FROM activity_log ORDER BY ts DESC LIMIT ?",
                           (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


def snap_path(name):
    """Безпечний шлях до кадру події (лише файли з activity/)."""
    if not name or "/" in name or "\\" in name or not name.endswith(".jpg"):
        return None
    p = SNAP_DIR / name
    return p if p.is_file() else None


def fmt_dur(s):
    s = int(round(s or 0))
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min" if m else f"{h} h"


# ------------------------------------------------------------------ трекер
class _Presence:
    """Стан однієї групи (person / pet) у кадрі."""

    def __init__(self):
        self.present = False
        self.cand_since = None      # коли група вперше з'явилась (ще не підтверджена)
        self.first_pts = []         # точки на початку появи
        self.since = 0.0            # коли підтвердили появу
        self.last_seen = 0.0
        self.last_pts = []          # точки в останньому кадрі, де група була
        self.last_jpg = None        # кадр, де група була востаннє
        self.max_count = 0


class _Visit:
    def __init__(self, ts, jpg):
        self.start = ts
        self.last_in = ts
        self.jpg = jpg


class ActivityLog:
    """Отримує детекції кожного кадру і пише події в activity_log."""

    def __init__(self, on_event=None):
        self.lock = threading.Lock()
        self.groups = {"person": _Presence(), "pet": _Presence()}
        self.visits = {}            # (zone_id, group) -> _Visit
        self.on_event = on_event    # callback(dict) — напр. для Telegram
        self.recent = []            # останні події (для /api/state)
        self.guard = self._load_guard()   # {"state": home|leaving|away, "since", "away_end"}
        SNAP_DIR.mkdir(exist_ok=True)
        db_init()

    # -- охорона: «вдома нікого»
    @staticmethod
    def _load_guard():
        try:
            g = json.loads(GUARD_PATH.read_text())
            if g.get("state") in ("home", "leaving", "away"):
                return {"state": g["state"], "since": float(g.get("since") or 0),
                        "away_end": float(g.get("away_end") or 0)}
        except (OSError, ValueError, TypeError):
            pass
        return {"state": "home", "since": 0.0, "away_end": 0.0}

    def _save_guard(self):
        try:
            tmp = GUARD_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.guard))
            os.replace(tmp, GUARD_PATH)
        except OSError:
            pass

    def _set_guard(self, state, ts):
        if state == "home" and self.guard["state"] == "away":
            self.guard["away_end"] = ts
        self.guard["state"] = state
        self.guard["since"] = ts
        self._save_guard()

    def _guard_step(self, ts, events):
        """Оновити стан охорони; повертає службові події (away / home)."""
        out = []
        person = self.groups["person"]
        state = self.guard["state"]
        for ev in events:
            if ev["kind"] == "leave_home" and ev["actor"] == "person" and not person.present:
                if state != "away":
                    self._set_guard("leaving", ev["ts"])
                    state = "leaving"
        if person.present:
            if state == "away":
                self._set_guard("home", ts)
                out.append({"ts": ts, "kind": "home", "actor": "person",
                            "text": "Someone is home · motion alerts off", "jpg": None})
            elif state == "leaving":
                self._set_guard("home", ts)          # повернулась до 5 хв — тихо
        elif state == "leaving" and ts - self.guard["since"] >= AWAY_AFTER_S:
            self._set_guard("away", ts)
            out.append({"ts": ts, "kind": "away", "actor": "person",
                        "text": "Nobody home · motion alerts on", "jpg": None})
        return out

    def tick(self, ts):
        """Без кадрів (детектор на паузі за графіком) — лише перевірити таймер охорони."""
        with self.lock:
            out = self._guard_step(ts, [])
        for ev in out:
            self._store(ev)
        return out

    def alerts_on(self, ts=None):
        """Чи слати рух у Telegram зараз: вдома нікого (або щойно повернулись)."""
        ts = time.time() if ts is None else ts
        with self.lock:
            g = self.guard
            return g["state"] == "away" or ts - g["away_end"] <= ALERT_GRACE_S

    def guard_state(self):
        with self.lock:
            return dict(self.guard)

    # -- допоміжне
    @staticmethod
    def _group(cls):
        if cls == "person":
            return "person"
        return "pet" if cls in PET_CLASSES else None

    @staticmethod
    def _foot(box, w, h):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0 / w, y2 / float(h))

    def _zones_of(self, pt, zones):
        return [z for z in zones if point_in_polygon(pt[0], pt[1], z["points"])]

    def _at_door(self, pts, zones):
        doors = [z for z in zones if z["kind"] == "door"]
        return any(point_in_polygon(p[0], p[1], z["points"]) for p in pts for z in doors)

    def _name(self, group, cfg, count=1):
        if group == "person":
            return "Person" if count <= 1 else f"{count} people"
        pet = cfg.get("pet_name") or "cat"
        return pet.capitalize() if count <= 1 else f"{count} {pet}s"

    # -- головне: кожен кадр
    def update(self, ts, dets, shape, jpg=None):
        """dets — [{"cls", "conf", "box"}]; shape — (h, w, ...); jpg — байти кадру."""
        h, w = shape[:2]
        cfg = load_zones()
        zones = cfg["zones"]
        pts = {"person": [], "pet": []}
        for d in dets:
            g = self._group(d.get("cls"))
            if g:
                pts[g].append(self._foot(d["box"], w, h))
        out = []
        with self.lock:
            for g, st in self.groups.items():
                out += self._presence(ts, g, st, pts[g], jpg, cfg, zones)
            out += self._zone_visits(ts, pts, jpg, cfg, zones)
            out += self._guard_step(ts, out)
        for ev in out:
            self._store(ev)
        return out

    def _presence(self, ts, g, st, pts, jpg, cfg, zones):
        events = []
        if pts:
            if not st.present:
                if st.cand_since is None or ts - st.last_seen > 3.0:
                    st.cand_since = ts
                    st.first_pts = list(pts)
                    st.max_count = 0
                if ts - st.cand_since >= ARRIVE_S[g]:
                    st.present = True
                    st.since = st.cand_since
                    door = self._at_door(st.first_pts, zones)
                    kind = "arrive" if door else "appear"
                    name = self._name(g, cfg, len(pts))
                    text = (f"{name} came home" if door and g == "person"
                            else f"{name} came in" if door
                            else f"{name} in the room")
                    events.append({"ts": st.since, "kind": kind, "actor": g,
                                   "count": len(pts), "text": text, "jpg": jpg})
            st.last_seen = ts
            st.last_pts = list(pts)
            st.max_count = max(st.max_count, len(pts))
            if jpg is not None:
                st.last_jpg = jpg
        elif st.present and ts - st.last_seen >= GONE_S[g]:
            st.present = False
            st.cand_since = None
            stay = st.last_seen - st.since
            door = self._at_door(st.last_pts, zones)
            name = self._name(g, cfg, st.max_count)
            if door:
                kind = "leave_home"
                text = f"{name} left home" if g == "person" else f"{name} went out the door"
            else:
                kind = "leave"
                text = f"{name} left the room"
            events.append({"ts": st.last_seen, "kind": kind, "actor": g, "count": st.max_count,
                           "duration": stay, "text": text, "jpg": st.last_jpg})
            st.last_jpg = None
        elif not st.present and st.cand_since is not None and ts - st.last_seen > 3.0:
            st.cand_since = None    # мигнуло і зникло — не подія
        return events

    def _zone_visits(self, ts, pts, jpg, cfg, zones):
        events = []
        live = set()
        for z in zones:
            if z["kind"] == "door":
                continue
            for g in ("person", "pet"):
                if any(point_in_polygon(p[0], p[1], z["points"]) for p in pts[g]):
                    key = (z["id"], g)
                    live.add(key)
                    v = self.visits.get(key)
                    if v is None:
                        self.visits[key] = _Visit(ts, jpg)
                    else:
                        v.last_in = ts
                        if jpg is not None and ts - v.start < 3:
                            v.jpg = jpg     # кадр з початку візиту — там видно, хто
        ids = {z["id"]: z for z in zones}
        for key, v in list(self.visits.items()):
            z = ids.get(key[0])
            if z is None:
                del self.visits[key]
                continue
            if key in live or ts - v.last_in < ZONE_GAP_S:
                continue
            del self.visits[key]
            dur = v.last_in - v.start
            if dur < max(1, z["min_s"]):
                continue
            g = key[1]
            name = self._name(g, cfg)
            kind, text = "visit", f"{name} at the {z['name'].lower()}"
            if g == "pet" and z["kind"] == "bowl":
                kind, text = "eat", f"{name} ate"
            elif g == "pet" and z["kind"] == "water":
                kind, text = "drink", f"{name} drank water"
            elif g == "pet" and z["kind"] == "litter":
                kind, text = "litter", f"{name} used the litter box"
            events.append({"ts": v.start, "kind": kind, "actor": g, "zone": z["name"],
                           "duration": dur, "text": text, "jpg": v.jpg})
        return events

    # -- запис
    def _store(self, ev):
        snap = None
        jpg = ev.pop("jpg", None)
        if jpg:
            snap = "act_%s%03d_%s.jpg" % (datetime.fromtimestamp(ev["ts"]).strftime("%Y%m%d_%H%M%S_"),
                                          int(ev["ts"] * 1000) % 1000, ev["kind"])
            try:
                (SNAP_DIR / snap).write_bytes(jpg)
                self._prune()
            except OSError:
                snap = None
        ev["snap"] = snap
        ev["local"] = datetime.fromtimestamp(ev["ts"]).strftime("%Y-%m-%d %H:%M:%S")
        try:
            con = sqlite3.connect(JOURNAL_DB)
            cur = con.execute(
                "INSERT INTO activity_log (ts, local, kind, actor, zone, duration, count, text, snap) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ev["ts"], ev["local"], ev["kind"], ev["actor"], ev.get("zone"),
                 ev.get("duration"), ev.get("count"), ev["text"], snap))
            con.commit()
            ev["id"] = cur.lastrowid
            con.close()
        except sqlite3.Error:
            pass
        self.recent.append(ev)
        del self.recent[:-20]
        if self.on_event:
            try:
                self.on_event(dict(ev))
            except Exception:
                pass

    def _prune(self):
        files = sorted(SNAP_DIR.glob("act_*.jpg"))
        for f in files[:-MAX_SNAPS]:
            f.unlink(missing_ok=True)
