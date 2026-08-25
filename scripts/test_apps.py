#!/usr/bin/env python3
"""test_apps.py — the screen apps, without needing the screen.

    ./venv/bin/python scripts/test_apps.py            checks only
    ./venv/bin/python scripts/test_apps.py --render   also writes PNGs

Renders every app headlessly (SDL_VIDEODRIVER=dummy) at the panel's real
1920x1080, so a broken draw() is caught here rather than by Harsh looking at
a black screen. Then exercises the parts that have no pixels: the time
parser, the schedule store, and the rule that a safety face outranks an app.

It leaves the real state/schedule.json alone -- everything runs against a
temporary state directory.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame                                            # noqa: E402

import app_control                                       # noqa: E402
import config                                            # noqa: E402

SIZE = config.FACE_DISPLAY_SIZE
OUT = Path(tempfile.gettempdir()) / "friday-apps"

_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {label}"
          + (f"   {detail}" if detail else ""))
    if not condition:
        _failures.append(label)


def use_temp_state() -> None:
    """Point the schedule and command files at a scratch directory."""
    tmp = Path(tempfile.mkdtemp(prefix="friday-apps-"))
    app_control.STATE_DIR = tmp
    app_control.SCHEDULE_FILE = tmp / "schedule.json"
    app_control.COMMAND_FILE = tmp / "app_command.json"
    app_control.STATUS_FILE = tmp / "app_status.json"


# =====================================================================

def test_parsing() -> None:
    print("\ntime parsing")
    now = datetime(2026, 8, 25, 21, 0)
    cases = {
        "8 AM": "08:00", "8:30 pm": "20:30", "19:45": "19:45",
        "tomorrow at 7": "07:00", "7": "07:00", "11 pm": "23:00",
    }
    for text, expected in cases.items():
        got = datetime.fromtimestamp(app_control.parse_when(text, now))
        check(f"{text!r} -> {expected}", got.strftime("%H:%M") == expected,
              got.strftime("%a %H:%M"))
    check("a time already gone rolls to tomorrow",
          datetime.fromtimestamp(
              app_control.parse_when("8 AM", now)).day == 26)

    print("\nduration parsing")
    for text, expected in {"10 minutes": 600, "1h30m": 5400,
                           "90 seconds": 90, "an hour and a half": 5400,
                           "10": 600, "two minutes": 120}.items():
        got = app_control.parse_duration(text)
        check(f"{text!r} -> {expected}s", got == expected, f"{got}s")

    for bad in ("", "sometime", "the day after the thing"):
        try:
            app_control.parse_when(bad)
            check(f"{bad!r} is refused", False)
        except app_control.ScheduleError:
            check(f"{bad!r} is refused", True)


def test_schedule() -> None:
    print("\nthe schedule")
    use_temp_state()
    alarm = app_control.add_alarm("8 AM", "school")
    timer = app_control.add_timer("10 minutes", "pasta")
    check("both are stored", len(app_control.pending()) == 2)
    check("soonest first", app_control.pending()[0]["id"] == timer["id"])
    check("labels survive", alarm["label"] == "school")

    check("nothing is due yet", app_control.due() == [])

    # Due, and due so long ago it should be dropped instead of ringing.
    schedule = app_control.read_schedule()
    schedule["timers"][0]["at"] = time.time() - 2
    schedule["alarms"][0]["at"] = time.time() - 10_000
    app_control.write_schedule(schedule)
    ready = app_control.due()
    check("the just-due timer rings", len(ready) == 1
          and ready[0]["id"] == timer["id"])
    check("the long-missed alarm is dropped, not rung",
          all(i["id"] != alarm["id"] for i in app_control.pending()))

    app_control.mark_fired(timer["id"])
    check("a fired timer is spent", app_control.pending() == [])

    app_control.add_timer("5 minutes")
    check("cancel all clears it", "cancelled" in app_control.cancel("all"))
    check("cancelling nothing is not an error",
          "no" in app_control.cancel("timer9"))


def test_render(save: bool) -> None:
    print("\nrendering at %dx%d" % SIZE)
    pygame.display.init()
    pygame.display.set_mode((1, 1))
    surface = pygame.Surface(SIZE)

    import apps
    registry = apps.load()
    check("every app registered", set(registry) ==
          set(app_control.APP_NAMES), ", ".join(sorted(registry)))

    app_control.add_timer("10 minutes", "pasta")
    cases = {
        "clock": {},
        "timer": {},
        "alarm": {"label": "school", "at": time.time(), "silent": True},
        "weather": {"weather": {"place": "Pune", "condition": "rain",
                                "temp_c": 24, "high_c": 29, "low_c": 22,
                                "wind_kph": 11, "humidity": 78}},
    }
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
    for name, kwargs in cases.items():
        app = registry[name](SIZE, **kwargs)
        drew_something = False
        for frame in range(4):                    # a few frames of animation
            app.update(1 / 30)
            surface.fill((0, 0, 0))
            app.draw(surface)
            lit = pygame.transform.average_color(surface)[:3]
            drew_something = drew_something or sum(lit) > 0
        check(f"{name} draws", drew_something, f"mean rgb {lit}")
        if save:
            path = OUT / f"{name}.png"
            pygame.image.save(surface, str(path))
            print(f"        {path}")
        app.close()

    print("\nthe manager")
    manager = apps.Manager(SIZE)
    check("launch reports back", "clock" in manager.launch("clock").lower()
          or manager.running == "clock")
    check("one app at a time", manager.launch("weather") is not None
          and manager.running == "weather")
    check("a nonexistent app is a sentence, not a crash",
          "no" in manager.launch("toaster").lower())
    check("the face is still up after that", manager.running == "weather")
    manager.dismiss()
    check("dismiss returns the screen", manager.running is None)

    manager.launch("alarm", label="test", at=time.time(), silent=True)
    check("a ringing alarm is urgent", manager.urgent is True)
    manager.launch("clock")
    check("a clock is not", manager.urgent is False)
    manager.dismiss()

    print("\nthe safety rule")
    from face_director import SAFETY_FACES
    check("scared and dizzy are the safety faces",
          SAFETY_FACES == {"scared", "dizzy"}, ", ".join(sorted(SAFETY_FACES)))
    import friday_face
    check("the face yields to safety in code",
          "SAFETY_FACES and not self.apps.urgent"
          in Path(friday_face.__file__).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--render", action="store_true",
                    help=f"also write PNGs to {OUT}")
    args = ap.parse_args()

    test_parsing()
    test_schedule()
    test_render(args.render)

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all good")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
