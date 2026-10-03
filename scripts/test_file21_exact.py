#!/usr/bin/env python3
"""File 21, built exactly as the file (Harsh, 2026-10-03: "make exactly as
file ... dont change the exploration behaviour ... just follow the pattern in
the file").

    ./venv/bin/python scripts/test_file21_exact.py

U1 -- spec 3.2 / 4.11 / 4.12: the way home is File 14's planner on the map she
built, the HELD run driving it (rl/way_back.py drive orders), the ladder round
it, HELP last. Checked here:
  * the run's side: a drive order is checked; exploring and the plain hold are
    unchanged; a path is steered at with the way back's own command; every
    safety stage below still has the last word (a wall ahead, the rear gate);
    a mission hold keeps the run's own way back from ending the run;
  * the engine (exploration_mission/run_drive.py) in the simulated house: it
    plans through the doors, arrives, fails a stall and marks a runtime
    obstacle, refuses a goal with no route, turns a full circle, backs up,
    reads the rear, notices a silent tracker, lets go of the hold;
  * whole missions: home by path, the ladder climbing through the held run,
    HELP with the file's numbers (30 s x 20), and the run parked at the end.
Network blocked; no robot; no sound.
"""

from __future__ import annotations

import math
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import way_back as wb                                   # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import mission_manager as M    # noqa: E402
from exploration_mission import run_drive as rd         # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402
import simworld as sw                                   # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file21-exact-"))
_results: list[bool] = []

# THE RUN'S REAL FILES. A ControlLoop's run() loads state/rl_map.npz and, unable
# to place itself on it with no LiDAR, MOVES IT ASIDE -- the first draft of this
# suite did exactly that to the dashboard's map (2026-10-03 15:37, put back the
# same minute). Every run() here is isolated, and this proves it afterwards.
REAL = sorted(p for p in (ROOT / "state").glob("rl_*") if p.is_file()) + [
    ROOT / "state" / "explore_mode.json"]
REAL_BEFORE = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in REAL if p.exists()}
ARCHIVES_BEFORE = set((ROOT / "state").glob("map-archive-*"))


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


# =============================================================================
print("--- U1, the run's side: the drive order (rl/way_back.py) ---")

check("a path order is read, its tolerance clamped",
      wb.drive_order({"kind": "path", "id": "a", "path": [[1, 2], [3, 4]], "tol": 9})
      == {"id": "a", "kind": "path", "path": [(1.0, 2.0), (3.0, 4.0)], "tol": 1.0})
check("garbage is no order (= the plain hold: she stands still)",
      all(wb.drive_order(x) is None for x in (
          None, "path", {"kind": "fly"}, {"kind": "path", "path": []},
          {"kind": "path", "path": [["nan", 1]]}, {"kind": "path"},
          {"kind": "path", "path": [[1]]})))
check("turn and back orders", wb.drive_order({"kind": "turn", "dir": -3})["dir"] == -1.0
      and wb.drive_order({"kind": "back"})["kind"] == "back")

hp = TMP / "hold.json"
h = wb.Hold(path=hp)
ad.atomic_write_json(hp, {"until": time.time() + 3.0, "why": "asking"})
h.poll()
check("a plain hold carries no drive", h.active() and h.driving() is None)
ad.atomic_write_json(hp, {"until": time.time() + 3.0, "why": "x",
                          "drive": {"kind": "turn", "id": "t1", "dir": 1}})
h.poll()
check("a hold with an order: active AND driving", h.active()
      and (h.driving() or {}).get("kind") == "turn")
ad.atomic_write_json(hp, {"until": time.time() - 1.0, "why": "x",
                          "drive": {"kind": "turn", "id": "t1"}})
h.poll()
check("a lapsed hold drives nothing (the mission stopped watching)",
      not h.active() and h.driving() is None)


class FakeTracker:
    """The start frame = the run frame shifted by `off` (no rotation)."""

    def __init__(self, off=(0.0, 0.0, 0.0)):
        self.off = off

    def home_rl(self):
        return self.off

    def to_rl(self, xy):
        p = wb._compose(self.off, (xy[0], xy[1], 0.0))
        return (p[0], p[1])

    def submit(self, *a, **k):
        pass

    def stop(self):
        pass

    def snapshot(self):
        return {}


md = wb.MissionDrive()
tr = FakeTracker((1.0, 0.0, 0.0))
order = {"id": "p1", "kind": "path", "path": [(2.0, 0.0), (2.0, 2.0)], "tol": 0.3}
v, om, why = md.command(order, tr, (0.0, 0.0, 0.0), 1.6, 0.1, -0.15)
check("a point ahead: drive at it (the way back's cruise), labelled as the mission's",
      v > 0.2 and abs(om) < 0.1 and why.startswith("mission path 1/2"), why)
v, om, why = md.command(order, tr, (3.0, 0.0, 0.0), 1.6, 0.1, -0.15)
check("passed the first point: heading for the next, turning to it in place",
      md.i == 1 and v == 0.0 and abs(om) == 1.6, f"{md.i} {v} {om} {why}")
v, om, why = md.command(order, tr, (3.0, 2.05, math.pi / 2), 1.6, 0.1, -0.15)
check("at the end of the path: zero", (v, om) == (0.0, 0.0), why)
v, om, why = md.command({"id": "p2", "kind": "path", "path": [(5.0, 0.0)], "tol": 0.3},
                        tr, (0.0, 0.0, 0.0), 1.6, 0.1, -0.15)
check("a new order starts from its first point", md.i == 0 and v > 0)
check("turn: in place at full turn rate",
      md.command({"id": "t", "kind": "turn", "dir": -1.0}, tr, (0, 0, 0), 1.6, 0.1,
                 -0.15)[:2] == (0.0, -1.6))
check("back: straight back at the run's reverse speed",
      md.command({"id": "b", "kind": "back"}, tr, (0, 0, 0), 1.6, 0.1, -0.15)[:2]
      == (-0.15, 0.0))
check("no start fix yet: stands still",
      md.command(order, None, (0, 0, 0), 1.6, 0.1, -0.15)[:2] == (0.0, 0.0))

# ---- the real ControlLoop ------------------------------------------------------
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot)

rcfg = RobotConfig()
rcfg.tof_enabled = False
rcfg.cam_enabled = False


def new_loop():
    slots = {"lidar": SensorSlot("lidar", rcfg.lidar_deadline, True),
             "tof": SensorSlot("tof", rcfg.tof_deadline, True)}
    loop = ControlLoop(rcfg, PolicyRunner(HERE / "exploration_model.onnx", rcfg, 1),
                       NullMotorDriver(rcfg), slots)
    loop.hold.path = TMP / f"loop_hold_{id(loop)}.json"
    # The run's frame starts where the loop's pose does ((12, 12, 0), the
    # middle of its map); the start frame's origin is that pose.
    loop.way_back = wb.WayBack(FakeTracker(tuple(float(v) for v in loop.map.pose)))
    return loop, slots


open_scan = np.full(rcfg.n_beams, 3.0, np.float32)
loop, slots = new_loop()
rows = []
for _ in range(5):
    slots["lidar"].publish(open_scan)
    rows.append(loop.tick())
check("no hold: exploring as before (no mission gate, nothing counted)",
      all(r["gate"] not in ("mission_hold", "mission_drive") for r in rows)
      and "mission_drive_ticks" not in loop.stats)
ad.atomic_write_json(loop.hold.path, {"until": time.time() + 3.0, "why": "asking"})
loop.hold.poll()
slots["lidar"].publish(open_scan)
row = loop.tick()
check("a plain hold: still the zero tick it always was",
      row["gate"] == "mission_hold" and row["v"] == 0.0 and row["omega"] == 0.0)


def drive_ticks(loop, slots, order, scan, n=15):
    ad.atomic_write_json(loop.hold.path, {"until": time.time() + 3.0, "why": "m",
                                          "drive": order})
    loop.hold.poll()
    out = []
    for _ in range(n):
        slots["lidar"].publish(scan)
        out.append(loop.tick())
        time.sleep(0.01)
    return out


rows = drive_ticks(loop, slots, {"id": "d1", "kind": "path", "path": [[3.0, 0.0]],
                                 "tol": 0.3}, open_scan)
recs = [r for r in rows if r["gate"] != "mission_hold"]
check("a path order: the tick runs in full and she drives at it",
      len(recs) == len(rows) and max(r["v"] for r in rows) > 0.05
      and loop.stats.get("mission_drive_ticks", 0) >= len(rows),
      str([(r["gate"], round(r["v"], 2)) for r in rows[-3:]]))
check("...the command is labelled the mission's in the decision record",
      any("mission" in str(s) for s in getattr(loop.last_record, "stages", [])
          or [str(loop.last_record.__dict__)]))

wall = open_scan.copy()
wall[:6] = 0.12
wall[-6:] = 0.12
loop2, slots2 = new_loop()
rows = drive_ticks(loop2, slots2, {"id": "d2", "kind": "path", "path": [[3.0, 0.0]],
                                   "tol": 0.3}, wall)
check("a wall 12 cm ahead: the safety chain still says no (never forward)",
      max(r["v"] for r in rows) <= 0.0, str([round(r["v"], 2) for r in rows]))

rear = open_scan.copy()
n = rcfg.n_beams
rear[n // 2 - 4: n // 2 + 5] = 0.10
loop3, slots3 = new_loop()
rows = drive_ticks(loop3, slots3, {"id": "b1", "kind": "back"}, rear)
check("a back-up with something right behind: the rear gate refuses it",
      min(r["v"] for r in rows) >= 0.0, str([round(r["v"], 2) for r in rows]))
loop4, slots4 = new_loop()
rows = drive_ticks(loop4, slots4, {"id": "b2", "kind": "back"}, open_scan)
check("...and with the rear clear she backs up", min(r["v"] for r in rows) < 0.0,
      str([round(r["v"], 2) for r in rows]))

# A mission hold keeps the run's own way back from ending the run under it.
# run() ISOLATED as rl/tests/test_way_back.py's whole_run is: no world map (it
# would load and set aside the real state/rl_map.npz), no state file, no
# blackbox dump, no report.
from world import WorldMap                                          # noqa: E402
loop5, slots5 = new_loop()
loop5.map_enabled = False
loop5.planner = None
loop5.world = WorldMap(map_file=TMP / "w5" / "map.npz", max_range=rcfg.lidar_max_range)
loop5.state_file = None
loop5.blackbox.dump = lambda _p: None
loop5.report = lambda: None
loop5.way_back = wb.WayBack(FakeTracker(tuple(float(v) for v in loop5.map.pose)))
loop5.way_back.begin("time")                    # its own way back, already home
ad.atomic_write_json(loop5.hold.path, {"until": time.time() + 30.0, "why": "asking"})
loop5.hold.poll()
loop5.hold.start = lambda: None                 # the test refreshes it by hand
loop5.tick = (lambda f: (lambda: (slots5["lidar"].publish(open_scan), f())[1]))(loop5.tick)
loop5.shutdown = lambda: None
t = threading.Thread(target=loop5.run, kwargs={"duration": 0.5, "verbose": False},
                     daemon=True)
t.start()
t.join(1.5)
held_alive = t.is_alive()
hp5 = loop5.hold.path
hp5.unlink()
loop5.hold.poll()
t.join(3.0)
check("held: the run's way back waits -- it does not 'arrive' and end the run",
      held_alive, "the run ended while the mission held it")
check("...released: its own way back carries on and ends it as before",
      not t.is_alive() and loop5.way_back.outcome == "arrived", loop5.way_back.outcome)
loop5._stop.set()


# =============================================================================
print("--- U1, the engine: File 14's planner, the held run driving ---")

cfg = mcfg.load({})
house = mm.three_rooms()
OG = mm.to_occupancy_grid(house.grid)


def engine_world(start=(10.0, 3.0)):
    clock = ad.FakeClock()
    sim = sw.SimRun(clock, house, [start, start], start_delay_s=0.0)
    sim.start(30)
    # Its frame starts where she stands; the tests speak in the house's frame,
    # so the start pose is the house's origin with no turn.
    sim.start_pose = (0.0, 0.0, 0.0)
    sim.truth = (start[0], start[1], 0.0)
    eng = rd.RunDrive(sim, lambda: OG, cfg, clock, M.InlineRunner(), 0.20)
    return clock, sim, eng


def run_until(clock, sim, eng, pred, max_s=120.0, dt=0.1):
    t = 0.0
    while t < max_s:
        clock.advance(dt)
        sim.step(dt)
        eng.step(clock.monotonic())
        if pred():
            return True
        t += dt
    return False


def engaged(start=(10.0, 3.0)):
    clock, sim, eng = engine_world(start)
    sim.step(0.1)                               # her first fix
    eng.engage(clock.monotonic())
    eng.step(clock.monotonic())
    return clock, sim, eng


clock, sim, eng = engaged((10.6, 4.4))           # room C, top right corner
check("engaged: she is held still, nothing driving",
      sim.held() is not None and "drive" not in sim.held())
check("available: the run is out and she has a fix", eng.available())
goal = (2.0, 2.0)                               # room A, bottom left: two doors away
check("an attempt starts", eng.attempt(goal))
eng.step(clock.monotonic())
order = (sim.held() or {}).get("drive") or {}
path = order.get("path") or []


def crosses_wall(points) -> list:
    """Every leg sampled every 2 cm against the true walls."""
    occ_, g = house.grid.occupied(), house.grid
    bad = []
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        n = max(int(math.hypot(bx - ax, by - ay) / 0.02), 1)
        for k in range(n + 1):
            x, y = ax + (bx - ax) * k / n, ay + (by - ay) * k / n
            if occ_[g.world_to_cell(x, y)]:
                bad.append((round(x, 2), round(y, 2)))
    return bad


route = [tuple(sim.truth[:2])] + [tuple(p) for p in path]
check("File 14's planner found a route that turns (not the straight line through "
      "the wall) and no leg of it crosses a wall",
      order.get("kind") == "path" and len(path) >= 2 and not crosses_wall(route),
      f"{path} {crosses_wall(route)[:3]}")
truth_track = []
ok = run_until(clock, sim, eng, lambda: (truth_track.append(sim.truth[:2]) or True)
               and not eng.busy())
occ = house.grid.occupied()
hit = [p for p in truth_track if occ[house.grid.world_to_cell(*p)]]
check("...the held run drove it and ARRIVED (within tolerance, held 2 s)",
      ok and eng.outcome() == "ARRIVED" and eng.distance_to(goal) <= 0.35,
      f"{eng.outcome()} {eng.distance_to(goal)}")
check("...never through a wall", not hit, str(hit[:3]))
check("...and stands still again after", "drive" not in (sim.held() or {}))

# a stall: she stops moving (wedged), the attempt fails, a runtime obstacle is marked
clock, sim, eng = engaged()
eng.attempt((2.5, 3.0))
run_until(clock, sim, eng, lambda: sim.drive_ticks > 20)
sim.freeze_until = clock.monotonic() + 1e6
t0 = clock.monotonic()
run_until(clock, sim, eng, lambda: not eng.busy(), max_s=60)
check("wedged: no progress for progress_timeout_s (20 s) -> STALLED",
      eng.outcome() == "STALLED" and 19.0 <= clock.monotonic() - t0 <= 25.0,
      f"{eng.outcome()} after {clock.monotonic() - t0:.1f} s")
check("...a runtime obstacle marks the blocked way (File 14's RuntimeObstacles)",
      len(eng.obstacles) == 1)
eng.clear_runtime_obstacles()
check("...rung 2 clears it (never the map)", eng.obstacles == [])

clock, sim, eng = engaged()
eng.attempt((20.0, 20.0))                       # outside the house: no route
eng.step(clock.monotonic())
check("no route on the map: NO_ROUTE, never a guess at the direction",
      not eng.busy() and eng.outcome() == "NO_ROUTE")

eng.spin_relocalise()
run_until(clock, sim, eng, lambda: not eng.busy(), max_s=40)
check("rung 3: one full turn, then done", eng.outcome() == "DONE"
      and eng.log[-1]["kind"] == "turn" and eng.log[-1]["s"] < rd.SPIN_MAX_S,
      str(eng.log[-1]))
x0 = sim.truth[:2]
eng.back_up(0.25)
run_until(clock, sim, eng, lambda: not eng.busy(), max_s=20)
moved = math.hypot(sim.truth[0] - x0[0], sim.truth[1] - x0[1])
check("rung 4: backs up about 0.25 m", eng.outcome() == "DONE" and 0.2 <= moved <= 0.45,
      f"{moved:.2f} m")
eng._scan = {180: 900, 170: 800, 195: 1200, 0: 300}
check("rear clearance from the sweep behind her (+-30 deg)", eng.rear_clear_m() == 0.8)
eng._scan = {0: 300, 90: 400}
check("...nothing measured behind: None (not 'clear')", eng.rear_clear_m() is None)

eng.attempt((2.5, 3.0))
run_until(clock, sim, eng, lambda: sim.drive_ticks > 5)
sim.publishing = False
run_until(clock, sim, eng, lambda: not eng.busy(), max_s=40)
check("the tracker goes silent while driving: the attempt fails (LOST)",
      eng.outcome() == "LOST")
eng.disengage()
check("disengage lets go of the run", sim.held() is None)


# =============================================================================
print("--- U1, whole missions: home by path ---")

check("the mode ships as the file's pattern ('path') with the file's HELP numbers",
      cfg["return"].mode == "path" and cfg["return"].help_retry_s == 30
      and cfg["return"].help_max_cycles == 20)

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.mgr.state == "RETURN_TO_START", max_s=30)
check("'go back': the run is HELD with a path, not stopped, not told to return",
      w.mgr.state == "RETURN_TO_START" and not w.orders("return")
      and ("stop",) not in w.sim.orders
      and ((w.sim.held() or {}).get("drive") or {}).get("kind") == "path",
      str(w.sim.orders[-4:]))
end = w.run()
m = w.mission()
check("...she arrives by the path, the run is parked, the mission is 'returned'",
      end == "IDLE" and m["status"] == "returned" and w.sim.drive_ticks > 0
      and ("stop",) in w.sim.orders and (m.get("home_error_m") or 9) <= 0.35,
      f"{m['status']} {m.get('home_error_m')} {w.sim.orders[-3:]}")
check("...and SHE says she has arrived (the run's own line never came)",
      "I've reached where I started. Exploration saved." in w.voice.said,
      str(w.voice.said[-3:]))
check("...'Going back to where I started.' said once",
      w.voice.said.count("Going back to where I started.") == 1, str(w.voice.said))

w = World(["hall"])
w.mgr.start(minutes=0.5)                       # the run's own timer ends exploring
end = w.run()
m = w.mission()
check("the run's own timer: held at once, then home by path (not its beeline)",
      end == "IDLE" and m["status"] == "returned" and w.sim.drive_ticks > 0
      and w.sim.phase == "returning", f"{m['status']} {w.sim.phase}")

# the ladder climbs through the held run: first attempt wedged, rung 1 frees it
w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.sim.drive_ticks > 10, max_s=40)
w.sim.freeze_until = w.clock.monotonic() + 24.0       # wedged for 24 s
end = w.run()
rungs = [a for a in w.mission()["rungs_used"] if a.get("rung") != "arrival"]
check("wedged: the attempt fails (no progress), rung 1 waits, the next attempt "
      "arrives", end == "IDLE" and w.mission()["status"] == "returned"
      and [r["rung"] for r in rungs][:2] == [0, 1] and rungs[0]["result"] == "failed",
      str(rungs))

# everything fails: HELP every 30 s, help_max_cycles rounds (20 in the file;
# 3 here, step 8 counts the 20), then HELP_WAIT, the run held
w = World(["hall", "kitchen", "bedroom"], cfg={"return": {"help_max_cycles": 3}})
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.sim.drive_ticks > 10, max_s=40)
w.sim.freeze_until = w.clock.monotonic() + 1e9
end = w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=3600)
said = " | ".join(w.voice.said)
check("nothing works: HELP with the distance, rungs 1-6 again each round, then "
      "HELP_WAIT after help_max_cycles rounds",
      w.mgr.state == "HELP_WAIT" and "I'm stuck about" in said
      and "I'll wait here." in said and w.mgr.ladder.cycle >= 3, said[-200:])
check("...the run is still HELD there (zero on the wire, no autonomy)",
      w.sim.held() is not None and "drive" not in w.sim.held()
      and ("stop",) not in w.sim.orders)
w.sim.freeze_until = -1e9
w.mgr._return_again(w.clock.monotonic())
end = w.run()
check("'Friday, come back' restarts the ladder on the same tracked map: home",
      end == "IDLE" and w.mission()["status"] == "returned", w.mission()["status"])

w = World(["hall", "kitchen", "bedroom"], cfg={"return": {"help_max_cycles": 2}})
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.sim.drive_ticks > 10, max_s=40)
w.sim.freeze_until = w.clock.monotonic() + 1e9
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=3600)
w.run(lambda x: ("stop",) in x.sim.orders, max_s=M.HELP_WAIT_KEEP_S + 30)
check("HELP_WAIT parks the held run after HELP_WAIT_KEEP_S (it costs a core)",
      ("stop",) in w.sim.orders and not w.mgr.drive.engaged)

# =============================================================================
print("--- nothing of the real run's touched ---")
after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in REAL if p.exists()}
check("state/rl_* and explore_mode.json exactly as they were (none moved, none written)",
      after == REAL_BEFORE, str(set(REAL_BEFORE) ^ set(after)))
check("no map was set aside", set((ROOT / "state").glob("map-archive-*")) == ARCHIVES_BEFORE)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
