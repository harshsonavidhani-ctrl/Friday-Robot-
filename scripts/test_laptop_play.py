#!/usr/bin/env python3
"""Friday's own play on the laptop and "I miss my laptop" (laptop_play.py,
2026-10-20). Every gate, with the laptop, the clock and the scene faked: no
network, nothing opened, the real state files untouched.

    ./venv/bin/python scripts/test_laptop_play.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402

_TMP = Path(tempfile.mkdtemp())
config.LAPTOP_SEEN_FILE = _TMP / "laptop_seen.json"
config.LAPTOP_APPS_FILE = _TMP / "laptop_apps.json"
config.LAPTOP_AGENT_FILE = _TMP / "secrets" / "laptop_agent.json"
config.RATE_LIMIT_FILE = _TMP / "rate_limits.json"
config.LAPTOP_AGENT_FILE.parent.mkdir(parents=True)
config.LAPTOP_AGENT_FILE.write_text(json.dumps({"url": "http://127.0.0.1:9", "token": "t" * 40}))
config.LAPTOP_PLAY, config.LAPTOP_MISS = True, True
config.LAPTOP_PLAY_HOURS = (0, 24)
config.LAPTOP_MISS_HOURS = (0, 24)

import laptop_play as lp                                 # noqa: E402
import laptop_research as lr                             # noqa: E402

lp.STATE_FILE = _TMP / "laptop_play.json"
lp._person_seen_within = lambda seconds, now: False     # not the live camera
lp._seen_things = lambda now: []
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


IDLE = {"v": {"idle_s": 900.0, "app": "explorer.exe"}}
lr.idle = lambda: IDLE["v"]
NOW = time.time()


def hello(at: float | None) -> None:
    if at is None:
        config.LAPTOP_SEEN_FILE.unlink(missing_ok=True)
    else:
        config.LAPTOP_SEEN_FILE.write_text(json.dumps(
            {"ip": "192.168.1.5", "port": 8765, "via": "direct", "at": at}))


def reset(**st) -> None:
    lp.STATE_FILE.write_text(json.dumps(st))


lr._save_apps([{"name": "WhatsApp", "via": "store"}, {"name": "Minecraft Launcher", "via": "desktop"},
               {"name": "Steam", "via": "desktop"}, {"name": "Hollow Knight", "via": "steam"}])
rng = random.Random(4)

print("playing on her own")
hello(NOW - 5)
reset()
item = lp.next_item(NOW, last_heard=NOW - 60, rng=rng)
check("laptop on, untouched 15 min, he spoke a minute ago -> she plays",
      item is not None and item["kind"] == "laptop_play" and item["action"], str(item))
check("...and says what she is doing", bool(item and item["say"]))
IDLE["v"] = {"idle_s": 30.0, "app": "chrome.exe"}
reset()
check("he used the laptop 30 s ago -> she never interrupts",
      lp.next_item(NOW, NOW - 60, rng) is None)
IDLE["v"] = None
check("an old helper (no idle time) -> she cannot tell, so she never plays",
      lp.next_item(NOW, NOW - 60, rng) is None)
IDLE["v"] = {"idle_s": 900.0, "app": "explorer.exe"}
check("nobody around (silent 2 h, no person seen) -> nothing",
      lp.next_item(NOW, NOW - 7200, rng) is None)
reset(last_play=NOW - 600)
check("played 10 min ago -> not again within the hour",
      lp.next_item(NOW, NOW - 60, rng) is None)
config.LAPTOP_PLAY_HOURS = (9, 10)
reset()
night = time.mktime(time.strptime("2026-10-20 02:00", "%Y-%m-%d %H:%M"))
hello(night - 5)
check("at 2 a.m. -> nothing", lp.next_item(night, night - 60, rng) is None)
config.LAPTOP_PLAY_HOURS = (0, 24)
config.LAPTOP_PLAY = False
hello(NOW - 5)
check("LAPTOP_PLAY False -> never on her own", lp.next_item(NOW, NOW - 60, rng) is None)
config.LAPTOP_PLAY = True

kinds = {lp.choose(NOW, random.Random(i))["action"]["do"] for i in range(40)}
check("she sometimes opens a game, sometimes searches", kinds == {"launch", "search"}, str(kinds))
games = {lp.choose(NOW, random.Random(i))["action"].get("name") for i in range(60)} - {None}
check("...and the only programs she opens are GAMES on his list (not Steam, not WhatsApp)",
      games and games <= {"Minecraft Launcher", "Hollow Knight"}, str(games))

print("missing the laptop")
hello(None)
reset(last_seen_on=NOW - 3700)
item = lp.next_item(NOW, NOW - 60, rng)
check("off for just over an hour and he is here -> 'I miss my laptop'",
      item is not None and item["kind"] == "laptop_miss" and "laptop" in item["say"], str(item))
reset(last_seen_on=NOW - 1800)
check("off for only 30 min -> nothing", lp.next_item(NOW, NOW - 60, rng) is None)
reset(last_seen_on=NOW - 3 * 3600, last_miss=NOW - 3600)
check("said it an hour ago -> not again yet", lp.next_item(NOW, NOW - 60, rng) is None)
reset()
config.LAPTOP_MISS_HOURS = (9, 10)
reset(last_seen_on=night - 3 * 3600)
check("'miss' keeps its own hours: at 2 a.m. -> nothing", lp.next_item(night, night - 60, rng) is None)
config.LAPTOP_MISS_HOURS = (0, 24)
reset()
check("first time ever (no record) counts as just seen -> nothing",
      lp.next_item(NOW, NOW - 60, rng) is None)
config.LAPTOP_MISS = False
reset(last_seen_on=NOW - 3 * 3600)
check("LAPTOP_MISS False -> never", lp.next_item(NOW, NOW - 60, rng) is None)
config.LAPTOP_MISS = True

print("perform: marks it, opens nothing without a reachable laptop (no exceptions)")
reset()
lp.perform({"kind": "laptop_miss", "say": "x"})
check("a miss is remembered (no repeat for LAPTOP_MISS_GAP_S)",
      json.loads(lp.STATE_FILE.read_text()).get("last_miss", 0) > NOW - 5)

print("through ReminderWatch: initiative still decides")
import initiative                                        # noqa: E402
import reminders                                         # noqa: E402
reminders.due_unsaid = lambda now: []
import day_calendar                                      # noqa: E402
day_calendar.heads_up_due = lambda now: []
import environments                                      # noqa: E402
environments.pending = lambda: []
lr.finished = lambda: []
config.BRIEFING = False
hello(time.time() - 5)
reset()
watch = reminders.ReminderWatch()
watch.heard_harsh()
initiative.decide = lambda kind, have_something=True, idle_for=None: (initiative.VOICE, kind)
got = watch.step()
check("the watch offers her play, said between turns", got is not None
      and got["kind"] == "laptop_play", str(got))
check("...and not twice while it is being said", watch.step() is None)
watch.missed(got)
initiative.decide = lambda kind, have_something=True, idle_for=None: (initiative.FACE, "he is talking")
check("he is talking -> nothing is offered", watch.step() is None)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
