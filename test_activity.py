"""Тести журналу дій: прийшов/пішов, кіт поїв, мигання YOLO не дає подій.

    .venv/bin/python -m unittest test_activity
"""
import json
import tempfile
import unittest
from pathlib import Path

import activity

SHAPE = (480, 640, 3)
DOOR = {"id": "door", "name": "Front door", "kind": "door",
        "points": [[0, 0.6], [0.3, 0.6], [0.3, 1], [0, 1]]}
BOWL = {"id": "bowl", "name": "Bowl", "kind": "bowl", "min_s": 15,
        "points": [[0.7, 0.7], [0.9, 0.7], [0.9, 0.9], [0.7, 0.9]]}


def det(cls, fx, fy, size=0.1):
    """Детекція з «ногами» в точці (fx, fy) у частках кадру."""
    w, h = SHAPE[1], SHAPE[0]
    x1, x2 = (fx - size / 2) * w, (fx + size / 2) * w
    y2 = fy * h
    return {"cls": cls, "conf": 0.8, "box": [x1, y2 - size * h, x2, y2]}


class ActivityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        activity.ZONES_PATH = d / "zones.json"
        activity.JOURNAL_DB = d / "j.db"
        activity.SNAP_DIR = d / "snaps"
        activity._cache.update(mtime=None, cfg=None)
        activity.save_zones({"pet_name": "cat", "zones": [DOOR, BOWL]})
        self.log = activity.ActivityLog()
        self.t = 1000.0

    def tearDown(self):
        self.tmp.cleanup()

    def run_frames(self, seconds, dets, fps=4):
        out = []
        for _ in range(int(seconds * fps)):
            self.t += 1.0 / fps
            out += self.log.update(self.t, dets, SHAPE, b"jpg")
        return out

    def kinds(self):
        return [r["kind"] for r in reversed(activity.db_list())]

    def test_came_home_and_left(self):
        self.run_frames(2, [det("person", 0.15, 0.9)])      # у дверях
        self.run_frames(30, [det("person", 0.5, 0.7)])      # у кімнаті
        self.run_frames(2, [det("person", 0.1, 0.85)])      # назад до дверей
        self.run_frames(25, [])
        rows = list(reversed(activity.db_list()))
        self.assertEqual([r["kind"] for r in rows], ["arrive", "leave_home"])
        self.assertEqual(rows[0]["text"], "Person came home")
        self.assertGreater(rows[1]["duration"], 30)
        self.assertTrue(rows[0]["snap"])

    def test_room_without_door(self):
        self.run_frames(5, [det("person", 0.5, 0.7)])
        self.run_frames(25, [])
        self.assertEqual(self.kinds(), ["appear", "leave"])

    def test_flicker_is_not_an_event(self):
        self.run_frames(0.5, [det("person", 0.5, 0.7)])
        self.run_frames(10, [])
        self.assertEqual(self.kinds(), [])

    def test_short_gaps_keep_presence(self):
        for _ in range(5):
            self.run_frames(3, [det("person", 0.5, 0.7)])
            self.run_frames(5, [])                           # YOLO загубив на 5 с
        self.run_frames(25, [])
        self.assertEqual(self.kinds(), ["appear", "leave"])

    def test_cat_ate(self):
        self.run_frames(40, [det("dog", 0.8, 0.8)])         # YOLO каже dog — це кіт
        self.run_frames(50, [])
        rows = list(reversed(activity.db_list()))
        self.assertEqual([r["kind"] for r in rows], ["appear", "eat", "leave"])
        self.assertEqual(rows[1]["text"], "Cat ate")
        self.assertGreaterEqual(rows[1]["duration"], 39)

    def test_cat_passing_bowl_is_not_eating(self):
        self.run_frames(5, [det("cat", 0.8, 0.8)])
        self.run_frames(20, [det("cat", 0.5, 0.5)])
        self.assertNotIn("eat", self.kinds())

    def test_zones_are_cleaned(self):
        cfg = activity.normalize_zones({"pet_name": "", "zones": [
            {"name": "x", "kind": "evil", "points": [[0, 0], [2, 0], [1, -1]]},
            {"name": "two points", "kind": "bowl", "points": [[0, 0], [1, 1]]},
            "junk"]})
        self.assertEqual(cfg["pet_name"], "cat")
        self.assertEqual(len(cfg["zones"]), 1)
        z = cfg["zones"][0]
        self.assertEqual(z["kind"], "place")
        self.assertEqual(z["points"][1], [1.0, 0.0])
        self.assertEqual(z["points"][2], [1.0, 0.0])

    def test_snap_path_is_safe(self):
        self.assertIsNone(activity.snap_path("../journal.db"))
        self.assertIsNone(activity.snap_path("x.png"))
        json.dumps(activity.load_zones())


if __name__ == "__main__":
    unittest.main()
