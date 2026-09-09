#!/usr/bin/env python3
"""Тести дебаунсу подій детектора (хибні спалахи на кріслі).

Перевіряє контракт рівня run(): послідовність кадрів -> список подій.
Запуск: ~/cv-venv/bin/python test_debounce.py
"""
import time

from dashboard import CONFIRM_FRAMES, CONFIRM_SECONDS, CLEAR_FRAMES, Detector


def make_det():
    d = Detector.__new__(Detector)  # без запуску потоків
    d._cand = set()
    d._cand_n = 0
    d._cand_t0 = 0.0
    d._clear_n = 0
    return d


def events(seq, dt=0.2):
    """Прогнати кадри через _debounce + edge-логіку run().

    Повертає список подій [("appear"|"change"|"clear", classes), ...] —
    саме те, що потрапило б у журнал і Telegram.
    """
    d = make_det()
    last = set()
    evs = []
    t = time.time()
    for classes in seq:
        t += dt
        cs = set(classes) if classes else set()
        confirmed, gone = d._debounce(cs, t)
        if confirmed and cs != last:
            evs.append(("appear" if not last else "change", tuple(sorted(cs))))
            last = cs
        elif gone and last:
            evs.append(("clear", ()))
            last = set()
    return evs


def test_chair_flash():
    """Хибний спалах на 1-2 кадри (крісло при зміні світла) — БЕЗ подій."""
    evs = events([("person",), ("person",), (), (), ("person",), ()])
    assert evs == [], f"спалах пройшов як подія: {evs}"
    print(f"OK: спалах {CONFIRM_FRAMES - 1} кадрів відфільтровано")


def test_sustained_person():
    """Справжня людина (тримається) — рівно одна подія appear.

    dt=0.2 с (~5 fps інференсу на Pi): 3 кадри покривають 0.4 с >= 0.3 с.
    """
    evs = events([("person",)] * 6, dt=0.2)
    assert evs == [("appear", ("person",))], f"очікувалась одна appear: {evs}"
    print("OK: стійкий person -> рівно одна подія appear")


def test_fast_frames():
    """Часовий гейт: 6 кадрів поспіль по 0.05 с (0.25 с сумарно < 0.3 с) —
    ще не підтверджено, CONFIRM_SECONDS фільтрує."""
    evs = events([("person",)] * 6, dt=0.05)
    assert evs == [], f"часовий гейт не спрацював: {evs}"
    print(f"OK: кадри швидші за {CONFIRM_SECONDS} с відфільтровані часом")


def test_clear():
    """Зникнення об'єкта — clear після 2 порожніх кадрів, не після першого."""
    evs = events([("person",)] * 4 + [(), (), ()])
    assert evs == [("appear", ("person",)), ("clear", ())], f"неочікувано: {evs}"
    print(f"OK: clear після {CLEAR_FRAMES} порожніх кадрів")


def test_flicker():
    """Мерехтіння person/порожньо — жодних подій в обидва боки."""
    evs = events([("person",), (), ("person",), (), ("person",), ()])
    assert evs == [], f"мерехтіння породило події: {evs}"
    print("OK: мерехтіння повністю відфільтроване")


def test_change():
    """person -> person+cat (стійко) — одна appear + одна change."""
    evs = events([("person",)] * 4 + [("cat", "person")] * 4)
    assert evs == [("appear", ("person",)), ("change", ("cat", "person"))], \
        f"неочікувано: {evs}"
    print("OK: стійка зміна класів -> одна подія change")


if __name__ == "__main__":
    test_chair_flash()
    test_sustained_person()
    test_fast_frames()
    test_clear()
    test_flicker()
    test_change()
    print("\nУсі тести дебаунсу пройшли.")