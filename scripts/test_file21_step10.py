#!/usr/bin/env python3
"""File 21 step 10: curiosity -- she asks to go exploring (spec 4.15).

    ./venv/bin/python scripts/test_file21_step10.py

Network blocked; no robot; a fake clock. The spec's six (quiet hours, "no" ->
2 h, silence -> 30 min, auto announces, not localized / no home anchor ->
never, off -> never) plus the score, the gates this build adds, the question
as a job the tick never waits on, and a "yes" that runs a whole mission in
the simulated house.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import area_store as st        # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import curiosity as cu         # noqa: E402
from exploration_mission import durable                 # noqa: E402
from exploration_mission import mission_manager as M    # noqa: E402
from exploration_mission import speech                  # noqa: E402
from harness import TMP, World                          # noqa: E402
import make_maps as mm                                  # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


# The store stamps missions with the wall clock; here that is the world's
# fake clock, so "hours since the last mission" is measured in one time.
CLOCK: list = [None]
_real_now_iso = st.now_iso


def _fake_now_iso() -> str:
    if CLOCK[0] is None:
        return _real_now_iso()
    return dt.datetime.fromtimestamp(CLOCK[0].wall()).astimezone().isoformat(
        timespec="seconds")


st.now_iso = _fake_now_iso


def at(hh: int, mm_: int = 0, day: int = 1) -> float:
    """A local wall time on 2026-10-<day>."""
    return dt.datetime(2026, 10, day, hh, mm_).timestamp()


ASK = speech.render("curiosity_ask")[0]
AUTO = speech.render("curiosity_auto")[0]
ROOM = mm.closed_room()
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]
HOME = "home"


def home_world(mode: str = "ask", start_hh: int = 11, **cur) -> World:
    """Booted, one mission finished from this spot (home anchor + "kitchen"),
    back at the start, nobody talking for an hour. Curiosity's own state is
    then what each test sets."""
    w = World(["save it as kitchen"], world=ROOM, path=ROOM_PATH,
              cfg={"curiosity": dict(mode=mode, **cur)})
    w.clock.t = at(start_hh)
    CLOCK[0] = w.clock
    w.mgr.boot()
    w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
    w.mgr.start()
    w.run()
    # The idle home look (step 9) places her before curiosity looks at all.
    w.run(lambda x: x.mgr.placed == "home", max_s=60)
    w.a.talk.quiet = 3600.0
    return w


def eager(w: World, frontier_m: float = 12.0) -> None:
    """The last mission left `frontier_m` of reachable edge (a time cap or a
    'that's enough'); with >= 24 h since it, the score reaches 0.8."""
    w.mgr.curiosity.mission_ended(HOME, "earlier", frontier_m, w.clock.wall())


def later(w: World, seconds: float, ticks: int = 4) -> None:
    """Jump the clock (she was idle all along), then tick so a check runs and
    its result is read."""
    w.clock.advance(seconds)
    for _ in range(ticks):
        w.step()


def asked(w: World) -> int:
    return w.voice.said.count(ASK)


def starts(w: World) -> list[dict]:
    out = []
    for m in w.store.missions(HOME, limit=50):
        out += [e for e in (m.get("events") or []) if e.get("kind") == "start"]
    return out


# =============================================================================
print("--- the score (spec 4.15) ---")

W = (0.5, 0.3, 0.2)
check("all three terms full = 1.0",
      abs(cu.score(10.0, 24.0, 4, 4, W) - 1.0) < 1e-9)
check("5 m of frontier, 12 h, 1 of 4 areas unnamed = 0.25 + 0.15 + 0.05",
      abs(cu.score(5.0, 12.0, 1, 4, W) - 0.45) < 1e-9, str(cu.score(5.0, 12.0, 1, 4, W)))
check("clamped: 40 m and 100 h count as full, never more",
      abs(cu.score(40.0, 100.0, 0, 0, W) - 0.8) < 1e-9)
check("no areas: the third term is 0 (not a division by zero)",
      cu.score(0.0, 0.0, 0, 0, W) == 0.0)
check("never a mission (None): the hours term is full",
      abs(cu.score(0.0, None, 0, 0, W) - 0.3) < 1e-9)
check("an unknown / negative frontier is 0", cu.score(None, 0.0, 0, 0, W) == 0.0
      and cu.score(-3.0, 0.0, 0, 0, W) == 0.0)
check("the config's numbers are the spec's", (lambda c: c.threshold == 0.8
      and list(c.weights) == [0.5, 0.3, 0.2] and c.min_idle_min == 10
      and c.min_quiet_conversation_min == 5 and list(c.quiet_hours) == ["22:00", "08:00"]
      and c.no_cooldown_h == 2 and c.silence_cooldown_min == 30
      and c.check_period_s == 60)(mcfg.load().curiosity))
check("⛔ curiosity SHIPS OFF (config.py), like AGENCY_MOTION",
      mcfg.load().curiosity.mode == "off")

print("--- quiet hours ---")
Q = ["22:00", "08:00"]
for (hh, mm_), quiet in (((22, 0), True), ((23, 30), True), ((3, 0), True),
                         ((7, 59), True), ((8, 0), False), ((12, 0), False),
                         ((21, 59), False)):
    check(f"{hh:02d}:{mm_:02d} is {'inside' if quiet else 'outside'} 22:00-08:00",
          cu.in_quiet_hours(dt.datetime(2026, 10, 1, hh, mm_), Q) is quiet)
check("a window that does not wrap (13:00-14:00)",
      cu.in_quiet_hours(dt.datetime(2026, 10, 1, 13, 30), ["13:00", "14:00"])
      and not cu.in_quiet_hours(dt.datetime(2026, 10, 1, 14, 0), ["13:00", "14:00"]))
check("start == end is no quiet hours at all",
      not cu.in_quiet_hours(dt.datetime(2026, 10, 1, 9, 0), ["09:00", "09:00"]))

print("--- the answer ---")
for text, want in (("yes", "yes"), ("Yes please.", "yes"), ("haan", "yes"),
                   ("haan chalo", "yes"), ("sure", "yes"), ("okay", "yes"),
                   ("Okay.", "yes"), ("go ahead", "yes"), ("explore karo", "yes"),
                   ("no", "no"), ("nahi", "no"), ("not now", "no"),
                   ("baad mein", "no"), ("maybe later", "no"), ("stop", "no"),
                   ("that's enough", "no"), ("don't", "no"),
                   (None, "silence"), ("", "silence"), ("umm", "silence"),
                   ("thank you", "silence"), ("what time is it", "silence")):
    check(f"{text!r} -> {want}", cu.classify(text) == want, cu.classify(text))

# =============================================================================
print("--- the gates, one at a time (why_not) ---")

cfg = mcfg.load({"curiosity": {"mode": "ask"}})
C = cu.Curiosity(cfg.curiosity, lambda env: TMP / "gates" / env / "curiosity.json")
NOON = at(12)


def facts(**kw) -> cu.Facts:
    base = dict(mode="ask", has_home=True, idle_s=3600.0, quiet_s=3600.0, busy=None,
                can_listen=True, frontier_m=12.0, hours_since_mission=30.0,
                stale_unnamed=0, areas=1)
    base.update(kw)
    return cu.Facts(**base)


check("everything open -> ask", C.why_not(facts(), "g", NOON) is None)
check("mode off -> never", C.why_not(facts(mode="off"), "g", NOON) == "off")
check("no home anchor -> never (the first mission is started by voice)",
      "home anchor" in (C.why_not(facts(has_home=False), "g", NOON) or ""))
check("the robot is busy -> never",
      C.why_not(facts(busy="a run has the robot"), "g", NOON) == "a run has the robot")
check("quiet hours -> never", C.why_not(facts(), "g", at(23)) == "quiet hours")
check("idle 9 min -> not yet",
      "idle" in (C.why_not(facts(idle_s=540.0), "g", NOON) or ""))
check("someone talked 4 min ago -> not yet",
      "talked" in (C.why_not(facts(quiet_s=240.0), "g", NOON) or ""))
check("nobody ever marked a turn (no heard.json) -> never: unknown is not quiet",
      "conversation clock" in (C.why_not(facts(quiet_s=None), "g", NOON) or ""))
check("ask mode with nobody able to hear the answer -> never",
      "hear" in (C.why_not(facts(can_listen=False), "g", NOON) or ""))
check("...but auto mode needs no listener",
      C.why_not(facts(mode="auto", can_listen=False), "g", NOON) is None)
check("score 0.75 < 0.8 -> not yet",
      "score" in (C.why_not(facts(frontier_m=9.0), "g", NOON) or ""))
C.answered("g", "no", NOON)
check("a 'no' -> a 2 h cooldown",
      abs(C.cooldown_left_s("g", NOON) - 7200.0) < 1e-6
      and "cooldown" in (C.why_not(facts(), "g", NOON + 7190) or "")
      and C.why_not(facts(), "g", NOON + 7201) is None)
C.answered("g", "silence", NOON)
check("silence -> a 30 min cooldown",
      abs(C.cooldown_left_s("g", NOON) - 1800.0) < 1e-6
      and C.why_not(facts(), "g", NOON + 1801) is None)
C2 = cu.Curiosity(cfg.curiosity, lambda env: TMP / "gates" / env / "curiosity.json")
check("the cooldown is on disk: a restart does not reset it",
      abs(C2.cooldown_left_s("g", NOON) - 1800.0) < 1e-6)
p = TMP / "gates" / "g" / "curiosity.json"
p.write_text("{ torn")
check("a torn curiosity file falls back to its .bak (durable.py)",
      C2.load("g").get("next_ok_at") is not None)
check("...the .bak is one save behind: the 'no' before the silence",
      [a["outcome"] for a in C2.load("g")["answers"]] == ["no"])
C2.answered("g", "yes", NOON)
check("the last answers are kept, newest last",
      [a["outcome"] for a in C2.load("g")["answers"]] == ["no", "yes"])

# =============================================================================
print("--- off -> never ---")

w = home_world(mode="off")
eager(w)
said = len(w.voice.said)
for hours in (1, 6, 30):
    later(w, hours * 3600)
check("off: 37 hours idle, eager score, nobody talking: never a word",
      len(w.voice.said) == said and asked(w) == 0)
check("...and never even a check (no job, no LiDAR look)",
      w.mgr._curious_job is None and w.mgr.curiosity.last == {})

# =============================================================================
print("--- ask: a yes starts a whole mission, exactly as 'explore' ---")

w = home_world(mode="ask")
eager(w)
listened = len(w.voice.listened)
dialog_seen: list = []
real_listen = w.voice.listen_once


def listening(timeout_s, _w=w):
    dialog_seen.append(_w.mgr.context().dialog)
    return real_listen(timeout_s)


w.voice.listen_once = listening
w.voice.replies.append("haan")
later(w, 10 * 60)                                       # 10 min idle, only 1 h since
check("10 min after the mission: not yet (score too low with 0 h since)",
      asked(w) == 0 and "score" in (w.mgr.curiosity.last.get("why") or ""),
      str(w.mgr.curiosity.last))
later(w, 30 * 3600)
check("30 h later: she asks the spec's line, once",
      asked(w) == 1, str(w.voice.said[-3:]))
check("...listened for the spec's 10 s",
      w.voice.listened[listened:] == [10.0], str(w.voice.listened))
check("...while asking, the voice router sees a 'curiosity' dialog",
      dialog_seen == ["curiosity"], str(dialog_seen))
check("...'haan' -> MISSION_INIT", w.mgr.state in ("MISSION_INIT", "EXPLORE"),
      w.mgr.state)
w.run()
s = starts(w)
check("...the mission ran to the end and she is IDLE again", w.mgr.state == "IDLE")
check("...recorded as started by curiosity", s and s[0].get("by") == "curiosity"
      and s[-1].get("by") == "voice", str(s))
check("...from the home spot, so it adopted the home frame (at_home)",
      s[0].get("at_home") is True, str(s[0]))
check("...the kitchen was recognised, not asked again; one area in all",
      [a["name"] for a in w.areas()] == ["kitchen"], str(w.areas()))
check("...and this mission finished the room: 0 m of frontier left on record",
      w.mgr.curiosity.frontier_m(HOME) == 0.0, str(w.mgr.curiosity.load(HOME)))
said = len(w.voice.said)
later(w, 48 * 3600)
check("...so 48 h later the score is only the hours term: no question",
      len(w.voice.said) == said, str(w.mgr.curiosity.last))

print("--- ask: 'no' -> 2 h, silence -> 30 min ---")

w = home_world(mode="ask")
eager(w)
w.voice.replies.append("no")
later(w, 30 * 3600)
check("asked, 'no': no mission", asked(w) == 1 and w.mgr.state == "IDLE"
      and len(starts(w)) == 1)
for minutes in (30, 45, 40):                            # +30, +75, +115 min
    later(w, minutes * 60)
check("...not asked again within 2 h (checked at +30, +75, +115 min)",
      asked(w) == 1, str(w.mgr.curiosity.last))
w2 = home_world(mode="ask")
eager(w2)
w2.voice.replies.append("nahi")
later(w2, 30 * 3600)
later(w2, 119 * 60)
n119 = asked(w2)
w2.voice.replies.append(None)                           # silence this time
later(w2, 2 * 60)
check("'nahi' -> asked again only after 2 h (not at 1 h 59)",
      n119 == 1 and asked(w2) == 2, f"{n119} {asked(w2)}")
later(w2, 29 * 60)
n29 = asked(w2)
w2.voice.replies.append("no")
later(w2, 2 * 60)
check("silence -> asked again only after 30 min (not at 29)",
      n29 == 2 and asked(w2) == 3, f"{n29} {asked(w2)}")
check("...three questions, no mission started by any of them",
      len(starts(w2)) == 1 and w2.mgr.state == "IDLE")

print("--- the cooldown survives a reboot ---")
w2.power_cut(off_s=60)
w2.mgr.boot()
w2.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
later(w2, 60 * 60)
check("rebooted 1 h after a 'no': still not asked", asked(w2) == 3,
      str(w2.mgr.curiosity.last))

print("--- 'stop' while she asks is a no ---")
w = home_world(mode="ask")
eager(w)


def stopped(timeout_s, _w=w):
    _w.mgr.halt()                       # what intents.handle does with "stop"
    return None


w.voice.listen_once = stopped
later(w, 30 * 3600)
check("'stop' during the question -> no mission, the 2 h cooldown",
      asked(w) == 1 and len(starts(w)) == 1
      and abs(w.mgr.curiosity.cooldown_left_s(HOME, w.clock.wall()) - 7200) < 5,
      str(w.mgr.curiosity.load(HOME)))

# =============================================================================
print("--- auto: announces, then starts ---")

w = home_world(mode="auto")
eager(w)
listened = len(w.voice.listened)
later(w, 30 * 3600)
check("auto: says the spec's line", AUTO in w.voice.said, str(w.voice.said[-3:]))
check("...asks nothing and listens to nothing",
      asked(w) == 0 and len(w.voice.listened) == listened)
w.run()
check("...and the mission runs to the end", len(starts(w)) == 2
      and starts(w)[0].get("by") == "curiosity" and w.mgr.state == "IDLE")

w = home_world(mode="auto")
eager(w)
later(w, 30 * 3600)
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=60)
w.mgr.halt()
w.run(lambda x: x.mgr.state == "PAUSED", max_s=10)
check("...every stop still works on an auto mission (stop -> PAUSED)",
      w.mgr.state == "PAUSED", w.mgr.state)

# =============================================================================
print("--- not localized / no home anchor -> never ---")

w = home_world(mode="ask")
eager(w)
w.sim.truth = (2.0, 3.2, 1.0)                           # carried off the home spot
w.sim.orders.clear()
said = len(w.voice.said)
later(w, 30 * 3600)
check("carried away from the home spot: the LiDAR look fails -> no question",
      len(w.voice.said) == said
      and "not localized" in (w.mgr.curiosity.last.get("why") or ""),
      str(w.mgr.curiosity.last))
check("...and the look never moved her", not w.orders("start", "hold", "return"))
w.sim.truth = (3.0, 2.5, 0.0)                           # put back
w.voice.replies.append("no")
later(w, 70)
check("...put back on the home spot: she asks at the next check", asked(w) == 1)

w = World([], world=ROOM, path=ROOM_PATH, cfg={"curiosity": {"mode": "auto"}})
w.clock.t = at(11)
CLOCK[0] = w.clock
w.a.talk.quiet = 3600.0
w.mgr.boot()
w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
w.mgr.curiosity.mission_ended(HOME, "none", 50.0, w.clock.wall())
said = len(w.voice.said)
for hours in (1, 24, 72):
    later(w, hours * 3600)
check("first ever: no home anchor -> never, even in auto with a full score",
      len(w.voice.said) == said and not starts(w)
      and "home anchor" in (w.mgr.curiosity.last.get("why") or ""),
      str(w.mgr.curiosity.last))

# =============================================================================
print("--- the other gates, in the whole manager ---")

w = home_world(mode="ask")
eager(w)
w.clock.t = at(23, 0, day=3)                            # 23:00, two days on
for _ in range(3):
    w.step()
check("quiet hours (23:00): no question", asked(w) == 0
      and w.mgr.curiosity.last.get("why") == "quiet hours")
w.voice.replies.append("no")
w.clock.t = at(8, 1, day=4)
later(w, 0)
check("...08:01: she asks", asked(w) == 1)

w = home_world(mode="ask")
eager(w)
w.a.talk.quiet = None
later(w, 30 * 3600)
check("no heard.json (the voice hook has not marked a turn): never",
      asked(w) == 0 and "conversation" in (w.mgr.curiosity.last.get("why") or ""))
w.a.talk.quiet = 120.0
later(w, 70)
check("he talked 2 min ago: not yet", asked(w) == 0
      and "talked" in (w.mgr.curiosity.last.get("why") or ""))
w.a.talk.quiet = 3600.0
w.a.talk.mode_quiet = True
later(w, 70)
check("quiet mode: never", asked(w) == 0
      and w.mgr.curiosity.last.get("why") == "quiet mode")
w.a.talk.mode_quiet = False
w.a.nav.journey = "kitchen"
later(w, 70)
check("a File 14 journey has the robot: never", asked(w) == 0
      and "journey" in (w.mgr.curiosity.last.get("why") or ""))
w.a.nav.journey = ""
w.sim.is_running = True
later(w, 70)
check("an RL run already out (the dashboard's button): never", asked(w) == 0
      and "run has the robot" in (w.mgr.curiosity.last.get("why") or ""))
w.sim.is_running = False
w.voice.listening = False
later(w, 70)
check("nobody able to hear (the real VoiceAdapter until step 11): never asks",
      asked(w) == 0 and "hear" in (w.mgr.curiosity.last.get("why") or ""))
check("the real VoiceAdapter says so", ad.VoiceAdapter().can_listen() is False)
w.voice.listening = True
w.mgr.interrupted = True
later(w, 70)
check("a power-cut mission waiting for continue / go back: never", asked(w) == 0
      and "interrupted" in (w.mgr.curiosity.last.get("why") or ""))
w.mgr.interrupted = False
w.voice.replies.append("no")
later(w, 70)
check("...every gate open again: she asks", asked(w) == 1)

w = home_world(mode="ask")
eager(w)
checks = []
real_check = w.mgr._curiosity_check
w.mgr._curiosity_check = lambda seen: (checks.append(w.clock.wall()), real_check(seen))[1]
for _ in range(1200):                                   # 2 min of 10 Hz ticks
    w.step()
check("checked at most once per check_period_s (60 s): 2 in 2 min of ticks",
      len(checks) == 2, str(len(checks)))

# =============================================================================
print("--- the question is a job: the tick never waits on it ---")


def paced(w: World, until, max_real_s: float = 5.0) -> list[float]:
    """Tick the way run_forever does (a pause between ticks, so the job's
    thread gets the CPU) until `until(w)`; returns each tick's ms."""
    ms = []
    t_end = time.monotonic() + max_real_s
    while time.monotonic() < t_end and not until(w):
        t0 = time.perf_counter()
        w.step()
        ms.append((time.perf_counter() - t0) * 1000.0)
        time.sleep(0.005)
    return ms


w = home_world(mode="ask")
eager(w)
w.mgr.runner = M.ThreadRunner()
release = threading.Event()


def slow_listen(timeout_s):
    release.wait(10.0)
    return "no"


w.voice.listen_once = slow_listen
w.clock.advance(30 * 3600)
paced(w, lambda x: x.mgr._curious_stage == "asking")
ms = paced(w, lambda x: False, max_real_s=0.5)          # she is waiting for him
asking = w.mgr._curious_stage
release.set()
paced(w, lambda x: x.mgr._curious_job is None)
check("while she waits for an answer the 10 Hz tick keeps running",
      asking == "asking" and len(ms) > 20 and max(ms) < 50.0,
      f"stage={asking} ticks={len(ms)} max={max(ms) if ms else 0:.1f} ms")
check("...and the answer is read when it comes",
      w.mgr._curious_job is None and w.mgr.curiosity.last.get("outcome") == "no",
      str(w.mgr.curiosity.last))

w = home_world(mode="ask")
eager(w)
w.mgr.runner = M.ThreadRunner()
gate = threading.Event()
w.voice.listen_once = lambda t: (gate.wait(10.0), "yes")[1]
w.clock.advance(30 * 3600)
paced(w, lambda x: x.mgr._curious_stage == "asking")
w.mgr.start()                                           # he says "explore" meanwhile
paced(w, lambda x: x.mgr.state != "IDLE")
state_before = w.mgr.state
gate.set()
paced(w, lambda x: x.mgr._curious_job is None)
paced(w, lambda x: x.mgr.state == "EXPLORE")
w.mgr.runner = M.InlineRunner()
w.run()
check("'explore' while she asks: ONE mission (his), the late 'yes' starts nothing",
      state_before == "MISSION_INIT" and len(starts(w)) == 2
      and starts(w)[0].get("by") == "voice", f"{state_before} {starts(w)}")

# =============================================================================
print("--- what the last mission left (the score's first term) ---")

w = World(["save it as hall"], world=mm.three_rooms(), cfg={"curiosity": {"mode": "ask"}})
w.clock.t = at(11)
CLOCK[0] = w.clock
w.mgr.boot()
w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 40, max_s=120)
w.mgr.finish()                                          # "that's enough"
w.run()
left = w.mgr.curiosity.frontier_m(HOME)
check("a mission ended early in the three-room house leaves frontier on record",
      left is not None and left > 1.0, f"{left} {w.mgr.curiosity.load(HOME)}")
snap = w.mgr.snapshot()
check("the dashboard snapshot carries curiosity's mode, stage and last decision",
      snap["curiosity"]["mode"] == "ask" and "last" in snap["curiosity"])

# =============================================================================
print("--- adapters: the conversation clock ---")

d = TMP / "talk"
d.mkdir(parents=True, exist_ok=True)
clk = ad.FakeClock(start=at(12))
talk = ad.TalkAdapter(heard_file=d / "heard.json", speaking_file=d / "speaking.json",
                      clock=clk)
check("no heard.json -> None (unknown)", talk.quiet_s() is None)
talk.mark_heard()
clk.advance(400)
check("mark_heard, 400 s later -> 400", abs(talk.quiet_s() - 400) < 1e-6)
(d / "speaking.json").write_text(json.dumps({"state": "speaking", "at": at(12) + 300,
                                             "until": at(12) + 350}))
check("her own voice counts: she stopped speaking 50 s ago -> 50",
      abs(talk.quiet_s() - 50) < 1e-6, str(talk.quiet_s()))
(d / "heard.json").write_text("{ torn")
check("a torn heard.json is unknown, never quiet", talk.quiet_s() is None)
check("mark_heard leaves no temp file behind",
      not [f for f in d.iterdir() if f.name.endswith(".tmp")])
check("nothing here wrote to the real state folder",
      not (ROOT / "state" / "exploration_mission" / "heard.json").exists())

# =============================================================================
print("--- nothing else changed ---")
import subprocess                                       # noqa: E402
diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--",
                       "rl", "slam", "exploration_mode.py",
                       "mcp_server/tools/explore.py", "main.py", "offline.py"],
                      capture_output=True, text=True).stdout.split()
check("rl/, slam/, exploration_mode.py, explore.py, main.py, offline.py untouched",
      diff == [], str(diff))

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
