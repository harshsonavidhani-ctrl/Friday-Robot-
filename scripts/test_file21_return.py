#!/usr/bin/env python3
"""File 21: home on the map she built (2026-10-02, Harsh: "map path first").

    ./venv/bin/python scripts/test_file21_return.py

The first real mission came home the RL run's way (aim at the start, dodge
locally) and was pinned 1 m out. Now exploring ends by PARKING the run, and
File 14's engine drives her home on the mission's own map through the return
ladder. Checked here:
  * whole missions in the simulated house: the run is parked before the first
    engine attempt and never asked for its own way back; the ladder climbs on
    failure; HELP with the true distance; no engine -> the old order;
  * nav_engine.EngineHost with File 14, the robot and the run faked: the run
    must be gone before she drives, she is found on the mission map (or it
    refuses), File 14 tracks on a COPY, File 14's own FAILSAFE record is
    cleared, back-ups and spins are guarded MCP moves;
  * environment_watch leaves a mission's run alone.
Network blocked; no robot; no sound.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import nav_engine as ne        # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


ROOM = mm.closed_room()
PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


class DrivingNav(ad.FakeNav):
    """FakeNav whose ARRIVED really puts her at the goal in the simulated
    world, so the arrival LOOK (LiDAR vs the start's signature) is honest."""

    def __init__(self, sim, **kw):
        super().__init__(**kw)
        self.sim = sim

    def attempt(self, goal_xy, waypoints=None):
        upcoming = self.script[0] if self.script else "FAILED"
        ok = super().attempt(goal_xy, waypoints)
        if ok and upcoming == "ARRIVED":
            s = self.sim.start_pose                     # the start frame's origin
            gx, gy = goal_xy
            self.sim.truth = (s[0] + gx * math.cos(s[2]) - gy * math.sin(s[2]),
                              s[1] + gx * math.sin(s[2]) + gy * math.cos(s[2]), s[2])
        return ok


def world(script, engine=True, **kw):
    # This file tests the MAP-FIRST order explicitly (config may say either).
    cfg = kw.pop("cfg", {"return": {"mode": "map_first"}})
    w = World(["kitchen"], world=ROOM, path=PATH, cfg=cfg, **kw)
    w.a.nav = DrivingNav(w.sim, during_run=False, clock=w.clock, script=script,
                         engine=engine)
    w.mgr.a.nav = w.a.nav
    return w


# =============================================================================
print("--- whole missions: home on the mission's map, first ---")

w = world(["ARRIVED"])
w.mgr.start()
end = w.run()
m = w.mission()
orders = [o[0] for o in w.sim.orders]
check("exploring ends -> the run is PARKED ('stop'), never asked for its own way back",
      "stop" in orders and "return" not in orders, str(w.sim.orders))
first_attempt = next((i for i, c in enumerate(w.a.nav.calls) if c[0] == "attempt"), None)
check("...and File 14's first attempt drives to the START (0, 0) on her map",
      first_attempt is not None and tuple(w.a.nav.calls[first_attempt][1]) == (0.0, 0.0),
      str(w.a.nav.calls[:3]))
check("...she says she is going back (the parked run says nothing now)",
      "Going back to where I started." in w.voice.said, str(w.voice.said[-4:]))
check("...arrived on the first attempt: returned, rung 0, then the look matched",
      end == "IDLE" and m["status"] == "returned" and m["rth_attempts"] == 1
      and m["rungs_used"][0]["rung"] == 0 and m["rungs_used"][-1].get("verified") is True,
      str(m["rungs_used"]))
check("...'I've reached where I started. Exploration saved.'",
      any("reached where I started" in s for s in w.voice.said))
ev = [e["kind"] for e in m["events"]]
check("...logged as a return on the map", "return_on_map" in ev, str(ev[-8:]))

w = world(["FAILED", "FAILED", "ARRIVED"])
w.mgr.start()
end = w.run()
m = w.mission()
rungs = [r["rung"] for r in m["rungs_used"] if isinstance(r.get("rung"), int)]
check("blocked twice, then home: the ladder climbs (wait, clear obstacles) and arrives",
      end == "IDLE" and m["status"] == "returned" and rungs[:3] == [0, 1, 2],
      str(m["rungs_used"]))

# --- THE LAST METRE BY THE START'S VIEW (2026-10-19: forward and back at one spot,
# 0.47-0.93 m from the start, 8 NO_ROUTE). Within mm.CLOSE_APPROACH_M she looks at
# the start by its own view first; the ladder is unchanged after it.
import exploration_mission.mission_manager as mmgr      # noqa: E402

w = world([])                                           # a path attempt would fail
w.a.nav.approach_outcome = "ARRIVED"                    # this engine CAN step on...
w.a.nav.on_approach = lambda: setattr(w.sim, "truth", tuple(w.sim.start_pose))  # ...onto it
w.mgr.start()
end = w.run()
m = w.mission()
ev = [e["kind"] for e in m["events"]]
check("close to the start: she looks by its view first, no path attempt needed",
      end == "IDLE" and m["status"] == "returned"
      and not any(c[0] == "attempt" for c in w.a.nav.calls) and "close_approach" in ev,
      f"{m['status']} calls={w.a.nav.calls[:4]} ev={ev[-6:]}")
check("...and the arrival is VERIFIED by the start's view",
      m["rungs_used"] and m["rungs_used"][-1].get("verified") is True, str(m["rungs_used"][-2:]))

PATH_OUT = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.2, 2.9)]   # ends 0.9 m out
w = World(["kitchen"], world=ROOM, path=PATH_OUT, cfg={"return": {"mode": "map_first"}})
w.a.nav = DrivingNav(w.sim, during_run=False, clock=w.clock, script=[], engine=True)
w.a.nav.approach_outcome = "FAILED"                     # stepping on cannot do it
w.mgr.a.nav = w.a.nav
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=3000)
m = w.mission()
ev = [e["kind"] for e in m["events"]]
kinds = [c[0] for c in w.a.nav.calls]
check("...the view cannot get her there: NOT 'returned' -- the ladder and HELP go on",
      w.mgr.state == "HELP_WAIT" and m.get("status") != "returned"
      and "close_approach_failed" in ev and "attempt" in kinds,
      f"state={w.mgr.state} status={m.get('status')} calls={kinds[:12]}")
check("...and at most CLOSE_APPROACH_TRIES close approaches in one way home",
      ev.count("close_approach") <= mmgr.CLOSE_APPROACH_TRIES, str(ev.count("close_approach")))

w = world([])                                           # every attempt fails
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=3000)
said = " ".join(w.voice.said)
check("never arrives -> HELP with the distance, then HELP_WAIT (no endless trying)",
      w.mgr.state == "HELP_WAIT" and "I'm stuck about" in said
      and "from where I started" in said, said[-300:])

w = world(["ARRIVED"], engine=False)
w.mgr.start()
w.run()
check("no engine (outside friday-mcp) -> the old order: the run's own way back",
      ("return", w.mission()["id"]) in w.sim.orders or
      any(o[0] == "return" for o in w.sim.orders), str(w.sim.orders))

w = world(["ARRIVED"], cfg={"return": {"mode": "run_first"}})
w.mgr.start()
w.run()
check("return.mode = run_first -> the run's own way back first, as before",
      any(o[0] == "return" for o in w.sim.orders), str(w.sim.orders))

# =============================================================================
print("--- the engine (File 14, the robot and the run faked) ---")

import nav_state_machine as nsm                          # noqa: E402
import nav_voice                                         # noqa: E402
from mcp_server import hardware                          # noqa: E402
from mcp_server.tools import autonomy, nav_tools         # noqa: E402
import nav_follower                                      # noqa: E402
from exploration_mission import service                  # noqa: E402


class Voice:
    def __init__(self): self.said = []
    def say(self, key, *a, **k): self.said.append(key); return key
    def say_text(self, text): self.said.append(text); return text


VOICE = Voice()
nav_voice.VOICE = VOICE
ne.SETTLE_S = 0.0
ne.RUN_GONE_WAIT_S = 0.3
running = [False]
autonomy._running = lambda: running[0]
nav_tools._reclaim_if_run_gone = lambda: None
cleared = []
nsm.clear_failsafe = lambda: cleared.append(1)


class Robot:
    """The real GuardedRobot refuses any move with no fresh look first
    (embodiment.check); this one does too."""
    def __init__(self): self.moves, self.looked = [], False
    def arm(self): return 0
    async def observe(self): self.looked = True
    async def drive_seconds(self, direction, seconds):
        if not self.looked:
            return "REFUSED — Friday has not looked at the room yet. Call observe() first"
        self.looked = False
        self.moves.append((direction, round(seconds, 3)))
        return f"moved {direction}"


ROBOTS: list = []


def make_robot():
    r = Robot()
    ROBOTS.append(r)
    return r


nav_follower.GuardedRobot = make_robot
SCRIPT: list = []
DRIVES: list = []
CORRIDORS: list = []          # what each leg was handed as her trail (2026-10-20)


class Controller:
    def __init__(self, robot=None, voice=None):
        self.robot, self.voice, self.obstacles = robot, voice, "old layer"

    async def return_to(self, xy, mapper, label="", corridor=None):
        DRIVES.append((tuple(xy), label, mapper))
        CORRIDORS.append(corridor)
        res = SCRIPT.pop(0) if SCRIPT else "arrived"
        if res == "arrived":
            mapper.pose = (xy[0], xy[1], mapper.pose[2])
            return nsm.ReturnReport(arrived=True)
        if res == "failsafe":
            self.voice.say("failsafe")
            return nsm.ReturnReport(state=nsm.NavState.FAILSAFE, arrived=False,
                                    why="stuck twice")
        return nsm.ReturnReport(arrived=False, why="no route")


nsm.NavController = Controller
og = mm.to_occupancy_grid(ROOM.grid)
og.scans = 30
og_bytes = og.logodds.copy()
FIX = [(1.0, 0.5, 0.2)]


class Slam:
    def __init__(self): self.calls = []
    def localise_on(self, grid, scan, near, radius):
        self.calls.append((tuple(near), radius))
        return FIX[0]


class Lidar:
    def __init__(self, scan=None): self.scan = scan or {0.0: 2000, 180.0: 600, 200.0: 450}
    def sweep(self): return dict(self.scan)


SLAM = Slam()
MGR = SimpleNamespace(analysis=SimpleNamespace(map=SimpleNamespace(og=og)),
                      pose=(1.1, 0.4, 0.25), mission={"id": "m1"},
                      a=SimpleNamespace(lidar=Lidar(), slam=SLAM))
service._manager = MGR


def run(action, **args):
    host = ne.EngineHost()

    async def go():
        res = host.request(action, args)
        if res.get("started"):
            await nav_tools._JOURNEY
        return res
    return host, asyncio.run(go())


def again(host, action, **args):
    async def go():
        res = host.request(action, args)
        if res.get("started"):
            await nav_tools._JOURNEY
        return res
    return asyncio.run(go())


host, res = run("attempt", goal=[0.0, 0.0])
check("attempt: started, then ARRIVED", res["started"] and host.outcome == "ARRIVED",
      f"{res} {host.outcome} {host.why}")
check("...she was found on the MISSION map near her last tracked pose (1.5 m)",
      SLAM.calls[0] == ((1.1, 0.4), 1.5), str(SLAM.calls))
check("...File 14 drove to the start on a tracker seeded with that fix",
      DRIVES[-1][0] == (0.0, 0.0) and DRIVES[-1][1] == "where I started")
check("...on a COPY: the mission's own map is untouched",
      DRIVES[-1][2].grid is not og and (og.logodds == og_bytes).all())
check("...with File 14's own arrival / failsafe lines silenced (the mission speaks)",
      isinstance(Controller(voice=ne._QuietVoice(VOICE)).voice, ne._QuietVoice)
      and ne._QuietVoice(VOICE).say("failsafe") == ""
      and ne._QuietVoice(VOICE).say("blocked") == "blocked")
check("...distance_to / pose come from the tracker",
      host.distance_to((0.0, 0.0)) == 0.0 and host.pose()[:2] == (0.0, 0.0))

DRIVES.clear()
host, res = run("attempt", goal=[0.0, 0.0], waypoints=[[0.5, 0.3], [0.2, 0.1]])
check("retrace: each breadcrumb is a leg, then the start",
      [d[0] for d in DRIVES] == [(0.5, 0.3), (0.2, 0.1), (0.0, 0.0)]
      and host.outcome == "ARRIVED", str([d[0] for d in DRIVES]))
check("...and every retrace leg may use the strip she DROVE (her trail, from "
      "where she stands), which the margin can close",
      len(CORRIDORS) >= 3 and all(c is not None and len(c) == 4 and
                                  [tuple(p) for p in c[1:]] == [(0.5, 0.3), (0.2, 0.1), (0.0, 0.0)]
                                  for c in CORRIDORS[-3:]), str(CORRIDORS[-3:]))
CORRIDORS.clear()

SCRIPT[:] = ["failsafe"]
cleared.clear()
host, res = run("attempt", goal=[0.0, 0.0])
check("a plain attempt (no trail) hands File 14 no corridor",
      CORRIDORS and all(c is None for c in CORRIDORS), str(CORRIDORS))
check("File 14 FAILSAFE inside an attempt -> FAILED for the ladder, its record "
      "cleared (a later 'try again' must not redo it)",
      host.outcome == "FAILED" and cleared == [1], f"{host.outcome} {cleared}")

FIX[0] = None
host, res = run("attempt", goal=[0.0, 0.0])
check("not sure where she is on the mission map -> REFUSED, nothing driven",
      host.outcome == "REFUSED" and "not sure" in host.why, host.why)
FIX[0] = (1.0, 0.5, 0.2)

running[0] = True
host, res = run("attempt", goal=[0.0, 0.0])
check("the run still out -> REFUSED (File 14 never drives beside it)",
      host.outcome == "REFUSED" and "still out" in host.why, host.why)
running[0] = False

host, res = run("attempt", goal=[0.0, 0.0])
host.busy, host.action = True, "attempt"                 # mid-drive
check("one action at a time: a second one while busy is refused, never queued",
      host.request("spin", {})["started"] is False and host.action == "attempt")
host.busy = False
ROBOTS.clear()
host, _ = run("attempt", goal=[0.0, 0.0])
res = again(host, "back_up", m=0.25)
mv = ROBOTS[-1].moves if ROBOTS else []
check("back_up 0.25 m -> ONE guarded backward move of 0.25/0.45 s",
      res["started"] and mv and mv[-1][0] == "backward"
      and abs(mv[-1][1] - round(0.25 / 0.45, 3)) < 1e-6, str(mv))
res = again(host, "spin")
check("spin -> one guarded left turn of 90 deg at the measured 75 deg/s, then a look",
      res["started"] and ROBOTS[-1].moves[-1] == ("left", round(90 / 75, 3)),
      str(ROBOTS[-1].moves))
check("an unknown action is refused", host.request("dance", {})["started"] is False)

# --- the 16:15 live test's three failures ---
print("--- what the 2026-10-02 16:15 test found ---")
DRIVES.clear()
host, _ = run("attempt", goal=[0.0, 0.0])
hardware.ABORT.set()                                    # File 14's FAILSAFE left it set
first_ctrl = host.controller
again(host, "attempt", goal=[0.0, 0.0])
check("a stop flag left set by File 14's give-up no longer ends the next rung "
      "('stop was called' in 0.6 s): the attempt is re-armed and drives",
      not hardware.ABORT.is_set() and len(DRIVES) == 2 and host.outcome == "ARRIVED",
      f"{hardware.ABORT.is_set()} {len(DRIVES)} {host.outcome}")
check("...and each attempt gets a FRESH File 14 controller (its stuck allowance "
      "starts over)", host.controller is not first_ctrl)

SLAM.calls.clear()
MGR.last_matched_pose = (0.3, -0.2, 0.0)
run("attempt", goal=[0.0, 0.0])
check("she is looked for round the LAST POSE THE TRACKER WAS SURE OF, 1.5 m only "
      "(no 3 m fallback: it found a look-alike 2 m away)",
      SLAM.calls == [((0.3, -0.2), 1.5)], str(SLAM.calls))
del MGR.last_matched_pose

answers = [None, None, None, (0.2, 0.1, 0.0)]           # found after the 4th sweep
class TurningSlam(Slam):
    def localise_on(self, grid, scan, near, radius):
        self.calls.append((tuple(near), radius))
        return answers.pop(0) if answers else None
MGR.a.slam = TurningSlam()
ROBOTS.clear()
host = ne.EngineHost()
res = again(host, "spin")
turns = [m for m in ROBOTS[-1].moves if m[0] == "left"] if ROBOTS else []
check("not sure where she is -> the look-again rung turns a quarter at a time and "
      "looks after each, until found", host.outcome == "DONE" and len(turns) == 3
      and host.mapper is not None, f"{host.outcome} {host.why} turns={turns}")
answers[:] = []
host = ne.EngineHost()
ROBOTS.clear()
again(host, "spin")
check("...never found -> FAILED after a full turn (4 quarters), never a guess",
      host.outcome == "FAILED" and len(ROBOTS[-1].moves) == 4 and host.mapper is None,
      f"{host.outcome} {ROBOTS[-1].moves if ROBOTS else None}")
MGR.a.slam = SLAM

import queue as _q                                       # noqa: E402
import time as _t                                        # noqa: E402
import tts                                               # noqa: E402
class QVoice:
    def __init__(self): self._queue = _q.Queue()
    def say_text(self, text): self._queue.put(text); return text
qv = QVoice()
qv._queue.put("Sorry, I didn't catch that. What should I call this area?")
speaking_until = [_t.time() + 0.6]
tts.quiet_at = lambda: speaking_until[0]
asked_at = []
class Asker:
    def listen_once(self, timeout_s): asked_at.append(_t.time()); return "kitchen"
    def can_listen(self): return True
va = ad.VoiceAdapter(voice=qv, asker=Asker())
import threading as _th                                  # noqa: E402
t0 = _t.time()
_th.Timer(0.4, lambda: qv._queue.get()).start()         # the worker starts speaking
heard = va.listen_once(10.0)
check("her re-ask is SAID and out of the room before the mic opens (16:18 she "
      "heard her own 'Sorry, I didn't catch that')",
      heard == "kitchen" and asked_at and asked_at[0] >= speaking_until[0] - 0.05,
      f"opened {asked_at[0] - t0 if asked_at else None:.2f}s, quiet at "
      f"{speaking_until[0] - t0:.2f}s")

# MissionNav: the ladder's side
calls = []
nav = ne.MissionNav(lidar=Lidar({0.0: 3000, 170.0: 800, 185.0: 500, 90.0: 300}),
                    host=host, call=lambda action, **a: (calls.append((action, a)),
                                                          {"started": True})[1])
check("MissionNav.attempt sends the goal and the waypoints",
      nav.attempt((0.0, 0.0), [(0.5, 0.2)]) and calls[-1] == (
          "attempt", {"goal": [0.0, 0.0], "waypoints": [[0.5, 0.2]]}), str(calls))
check("rear_clear_m: the nearest return within +-30 deg BEHIND her (not beside)",
      nav.rear_clear_m() == 0.5, str(nav.rear_clear_m()))
check("go-to with no mission stays File 14's (MissionNav.go refuses)",
      nav.go({"name": "kitchen"}) is False and nav.available_during_run() is False)

# =============================================================================
print("--- the ladder waits for a busy engine (2026-10-02 22:45) ---")
from exploration_mission import rth_ladder as rl, config as mcfg   # noqa: E402
clk = ad.FakeClock()
fn = ad.FakeNav(clock=clk, script=["ARRIVED"])
lad = rl.Ladder(fn, rl.LadderConfig.from_config(mcfg.load()), lambda *a, **k: None,
                {"goal": (0.0, 0.0), "trail": [], "grid": None, "inflate_m": 0.3,
                 "body_radius_m": 0.21, "obstacles": None})
lad.begin(clk.monotonic())
fn._busy, fn._pending, fn._until = True, "SPUN", clk.monotonic() + 20.0   # a spin still turning
lad._attempt(clk.monotonic())
check("an attempt while the engine is still busy WAITS (not 'refused' at once)",
      lad.phase == "engine_busy" and not lad.attempts, lad.phase)
out = None
for _ in range(300):
    clk.advance(0.1)
    out = lad.tick(clk.monotonic())
    if out:
        break
check("...and starts as soon as the engine is free, then arrives",
      out == "arrived" and [a.result for a in lad.attempts] == ["arrived"],
      f"{out} {[a.result for a in lad.attempts]}")
check("the look-again rung may take up to 75 s (a full turn, a look per quarter)",
      rl.MANEUVER_TIMEOUT_S >= 75.0)

# =============================================================================
print("--- arrival: onto the start by its own view (the engine's 'approach') ---")
from exploration_mission import rth_ladder as _rl                # noqa: E402
VIEWS: list = []
HEADINGS: list = []                     # the gyro heading each look was given
_real_from_start = _rl.from_start


def _scripted_from_start(sweeps, sig, max_range=8.0, heading_rad=None):
    HEADINGS.append(heading_rad)
    return ({"pose": VIEWS.pop(0), "why": "scripted"} if VIEWS and VIEWS[0] is not None
            else (VIEWS.pop(0) if VIEWS else None,
                  {"pose": None, "why": "nothing like it"})[1])


_rl.from_start = _scripted_from_start
MGR.mission = {"id": "m1", "start_signature": [1.0] * 360}


def approach(views, robot=None):
    VIEWS[:] = list(views)
    ROBOTS.clear()
    h = ne.EngineHost()
    if robot is not None:
        h.robot = robot
    res = again(h, "approach")
    r = robot if robot is not None else (ROBOTS[-1] if ROBOTS else None)
    return h, res, ([] if r is None else r.moves)


# Her pose in the START's frame, as the start's view would say after each move.
h, res, moves = approach([(1.0, 0.1, 0.0),             # start behind: turn (capped)
                          (1.0, 0.1, -2.1),            # nearly facing: turn the rest
                          (1.0, 0.0, math.pi),         # facing it: step
                          (0.65, 0.0, math.pi),        # step
                          (0.30, 0.0, math.pi),        # a shorter last step
                          (0.05, 0.0, math.pi)])       # on it: face the way she set off
check("approach: turn toward the start, short steps with a look after each, then "
      "turn to face the way she set off -> ARRIVED",
      res.get("started") and h.outcome == "ARRIVED"
      and [m[0] for m in moves] == ["right", "right", "forward", "forward", "forward",
                                    "right"], f"{h.outcome} {h.why} {moves}")
check("...the turn toward a start behind her is capped at 120 deg, a step at 0.35 m",
      abs(moves[0][1] - 120.0 / 75.0) < 0.01
      and abs(moves[2][1] - ne.APPROACH_STEP_M / 0.45) < 0.01, str(moves[:3]))
check("...and File 14's tracker is dropped (she moved without it)",
      h.mapper is None)


class Refuses:
    def __init__(self): self.moves = []
    def arm(self): return 0
    async def observe(self): pass
    async def drive_seconds(self, direction, seconds):
        self.moves.append((direction, seconds))
        return "REFUSED -- something is 210 mm ahead"


h, res, moves = approach([(0.8, 0.0, math.pi)], robot=Refuses())
check("approach: a guarded step refused (something in the way) -> FAILED, no "
      "second try at it", h.outcome == "FAILED" and "REFUSED" in h.why
      and len(moves) == 1, f"{h.outcome} {h.why} {moves}")
h, res, moves = approach([(0.5, 0.0, math.pi), (0.8, 0.0, math.pi),
                          (1.1, 0.0, math.pi), (0.2, 0.0, math.pi)])
check("approach: each step leaves her FURTHER from the start -> stops after the "
      "second, never wanders", h.outcome == "FAILED" and "further" in h.why
      and len(moves) == 2, f"{h.outcome} {h.why} {moves}")
h, res, moves = approach([None])
check("approach: the start's view finds nothing like it -> FAILED, no move",
      h.outcome == "FAILED" and not moves, f"{h.outcome} {h.why}")


# --- the GYRO (2026-10-09): her heading narrows each look, turns are measured ---
class GyroRobot(Robot):
    def __init__(self, answers=True):
        super().__init__()
        self.turns, self.answers = [], answers

    async def turn_measured(self, degrees_left):
        if not self.answers:
            return None                              # no gyro on the board
        if not self.looked:
            return "REFUSED — Friday has not looked at the room yet. Call observe() first"
        self.looked = False
        self.turns.append(round(degrees_left, 1))
        return f"Turned {abs(degrees_left):.0f} degrees (measured by the gyro)."


class Heading:
    def __init__(self, values): self.values = list(values)
    def now(self): return self.values.pop(0) if self.values else None


MGR.heading = Heading([3.0, 2.9, 2.8])
HEADINGS.clear()
gr = GyroRobot()
h, res, moves = approach([(1.0, 0.1, 0.0), (1.0, 0.0, math.pi), (0.05, 0.0, math.pi)],
                         robot=gr)
check("approach with the gyro: every look is given her gyro heading",
      HEADINGS == [3.0, 2.9, 2.8], str(HEADINGS))
check("...the turn toward the start and the final face are MEASURED turns (signed, "
      "+ = left), the step is the only timed move -> ARRIVED",
      h.outcome == "ARRIVED" and gr.turns == [-120.0, -180.0]
      and [m[0] for m in moves] == ["forward"], f"{h.outcome} {h.why} {gr.turns} {moves}")
gr = GyroRobot(answers=False)
h, res, moves = approach([(1.0, 0.1, 0.0), (1.0, 0.0, math.pi), (0.05, 0.0, math.pi)],
                         robot=gr)
check("...a board with no gyro: the same approach with timed pivots, as before",
      h.outcome == "ARRIVED" and [m[0] for m in moves] == ["right", "forward", "right"]
      and abs(moves[0][1] - 120.0 / 75.0) < 0.01, f"{h.outcome} {moves}")
MGR.heading = Heading([])
HEADINGS.clear()
h, res, moves = approach([(0.05, 0.0, 0.0)])
check("...no heading the gyro can vouch for -> the look searches every heading (None)",
      h.outcome == "ARRIVED" and HEADINGS == [None], str(HEADINGS))
del MGR.heading

MGR.mission = {"id": "m1"}
h, res, moves = approach([(0.8, 0.0, math.pi)])
check("approach: no start view saved -> REFUSED, no move",
      h.outcome == "REFUSED" and not moves, f"{h.outcome} {h.why}")
check("MissionNav.approach_start() asks the engine for 'approach'",
      ne.MissionNav(call=lambda action, **a: {"started": action == "approach"})
      .approach_start() is True)
_rl.from_start = _real_from_start

# =============================================================================
print("--- environment_watch leaves a mission's run alone ---")
import environment_watch                                 # noqa: E402
service._manager = SimpleNamespace(mission={"id": "m"}, state="RETURN_TO_START")
check("a mission busy -> environment_watch does not start its own drive home",
      environment_watch._mission_owns_the_run() is True)
service._manager = SimpleNamespace(mission=None, state="IDLE")
check("...an idle mission (or none) -> it does, as before",
      environment_watch._mission_owns_the_run() is False)
service._manager = None
check("...no mission service at all -> as before",
      environment_watch._mission_owns_the_run() is False)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
