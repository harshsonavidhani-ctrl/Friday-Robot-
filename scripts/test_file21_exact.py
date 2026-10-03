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
    ROOT / "state" / "explore_mode.json",
    ROOT / "state" / "exploration_mission" / "pause.json"]
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
    # ...and its PAUSE file: Hold took that from the REAL folder when built.
    loop.hold.pause_path = loop.hold.path.with_name(f"pause_{id(loop)}.json")
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
print("--- U2 (spec 4.2): FRONTIER_ASSIST when the RL finds nothing new ---")


def events(w, kind):
    return [e for e in (w.mission() or {}).get("events", []) if e["kind"] == kind]


w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 12)
check("exploring normally: no assist, the run never held for one",
      w.mgr.assist is None and not events(w, "frontier_assist"))
w.sim.rl_stall_until = w.clock.monotonic() + 1e9          # the RL finds nothing new
t0 = w.t
w.run(lambda x: x.mgr.assist is not None, max_s=200)
t_assist = w.t
w.run(lambda x: ((x.sim.held() or {}).get("drive") or {}).get("kind") == "path"
      or x.mgr.assist is None, max_s=2)
order = (w.sim.held() or {}).get("drive") or {}
check("the map grew < 0.5 m2 in 60 s with a frontier left: FRONTIER_ASSIST -- the RL "
      "is held and File 14's planner drives her to the best frontier",
      order.get("kind") == "path" and 55.0 <= t_assist - t0 <= 140.0,
      f"{t_assist - t0:.0f} s, {order}")
w.run(lambda x: x.mgr.assist is None, max_s=120)
last = (w.mgr.assists or [{}])[-1]
check("...it ends at the frontier, once that frontier is explored, or for a room's "
      "question (which comes first, 4.6); exploring goes on",
      last.get("why") in ("arrived", "explored", "asking")
      and (w.sim.held() is None or w.mgr.state == "ASK_LABEL"), str(last))
grew_before = w.mgr.explored_m2
w.run(lambda x: len(x.mgr.assists) >= 3
      or x.mgr.state not in ("EXPLORE", "ASK_LABEL", "TARGET_NAV"), max_s=900)
done = [a for a in w.mgr.assists if a["why"] in ("arrived", "explored")]
check("RL_DRIVE <-> FRONTIER_ASSIST again and again while frontiers are left, at "
      "least one reaching its frontier, and the map grows",
      len(w.mgr.assists) >= 2 and done and w.mgr.explored_m2 >= grew_before,
      f"{w.mgr.assists} {w.mgr.state}")

# a frontier she cannot reach: fails twice, then it is blacklisted for the mission
w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 12)
w.sim.rl_stall_until = w.clock.monotonic() + 1e9
w.run(lambda x: x.mgr.assist is not None, max_s=200)
first = tuple(w.mgr.assist["centroid"])
w.sim.freeze_until = w.clock.monotonic() + 1e9            # wedged: nothing moves
w.run(lambda x: len(x.mgr.assists) >= 2 or x.mgr.state != "EXPLORE", max_s=400)
rows = w.mgr.assists
check("wedged on the way: no progress in 20 s -> one failure for that frontier, the "
      "RL gets its turn, then the second failure blacklists it",
      len(rows) >= 2 and rows[0]["why"] == "stalled" and rows[1]["blacklisted"]
      and w.mgr.blacklist.blocks(first), str(rows))

# the hand-over is gap-free: ending an assist for a question leaves her HELD
w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 12)
w.sim.rl_stall_until = w.clock.monotonic() + 1e9
w.run(lambda x: x.mgr.assist is not None, max_s=200)
w.mgr._end_assist(w.clock.monotonic(), "asking", release=False)
held = w.sim.held() or {}
check("an assist ended for a question / a go-to / the end leaves her on the PLAIN "
      "hold -- never a gap for the RL, never a drive order left behind",
      w.mgr.assist is None and held and "drive" not in held, str(held))


# =============================================================================
print("--- U2 (spec 3.3): 'go to X' in the middle of a mission ---")


def target_for(w, name):
    a = w.store.by_name("home", name)
    return {"kind": "area", "name": a["name"], "x": a["center_x"], "y": a["center_y"],
            "radius_m": a["radius_m"], "area": a}


def world_xy(w, xy):
    p = M.mission_map.compose(w.sim.start_pose, (xy[0], xy[1], 0.0))
    return p[:2]


w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: len(x.areas()) >= 2 and x.mgr.state == "EXPLORE", max_s=300)
trail_before, bl_before = w.mgr.trail, w.mgr.blacklist
hall = target_for(w, "hall")
n = len(w.voice.said)
w.mgr.go_to("hall", hall)
w.run(lambda x: x.mgr.state == "TARGET_NAV", max_s=5)
check("'go to the hall' mid-mission: TARGET_NAV, the run held with a path, "
      "File 14's line", w.mgr.state == "TARGET_NAV"
      and ((w.sim.held() or {}).get("drive") or {}).get("kind") == "path"
      and "Heading to hall." in w.voice.said[n:], str(w.voice.said[n:]))
w.run(lambda x: x.mgr.state != "TARGET_NAV", max_s=200)
hx, hy = world_xy(w, (hall["x"], hall["y"]))
d = math.hypot(w.sim.truth[0] - hx, w.sim.truth[1] - hy)
check("...she gets there and says so, then EXPLORE again with the RL driving",
      "I have arrived at hall." in w.voice.said[n:] and w.mgr.state in ("EXPLORE", "ASK_LABEL")
      and d <= 0.5, f"{d:.2f} m {w.voice.said[n:]} {w.mgr.state}")
check("...the snapshot is kept: the same trail (loop-cut as it should be), the "
      "same blacklist, the same mission -- nothing reset",
      w.mgr.trail is trail_before and w.mgr.blacklist is bl_before
      and len(w.mgr.trail.crumbs) > 0 and w.mgr.mission is not None)

n = len(w.voice.said)
w.mgr.go_to("desk", {"kind": "landmark", "name": "desk", "x": 1.0, "y": 1.0})
w.run(lambda x: False, max_s=1.0)
check("a place saved in ANOTHER frame (the saved map's, this mission has its own): "
      "'I don't know where desk is yet.' -- never driven to a wrong address",
      "I don't know where desk is yet." in w.voice.said[n:] and w.mgr.state != "TARGET_NAV",
      str(w.voice.said[n:]))

w.run(lambda x: x.mgr.state == "EXPLORE", max_s=60)
kitchen = target_for(w, "kitchen")
n = len(w.voice.said)
w.mgr.go_to("kitchen", kitchen)
w.run(lambda x: x.mgr.state == "TARGET_NAV", max_s=5)
w.sim.freeze_until = w.clock.monotonic() + 1e9
w.run(lambda x: x.mgr.state != "TARGET_NAV", max_s=200)
check("wedged on the way: one re-plan, then 'I cannot find a route to kitchen.' and "
      "'Continuing exploration.' -- back to EXPLORE, the RL let go",
      "I cannot find a route to kitchen." in w.voice.said[n:]
      and "Continuing exploration." in w.voice.said[n:] and w.mgr.state == "EXPLORE"
      and w.sim.held() is None
      and [e for e in events(w, "go_to_done")][-1]["outcome"] != "ARRIVED",
      str(w.voice.said[n:]))
w.sim.freeze_until = -1e9

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: len(x.areas()) >= 2 and x.mgr.state == "EXPLORE", max_s=300)
hall = target_for(w, "hall")
w.mgr.go_to("hall", hall)
w.run(lambda x: x.mgr.state == "TARGET_NAV", max_s=5)
w.run(lambda x: x.sim.drive_ticks > 15, max_s=30)
w.mgr.halt()
w.run(lambda x: x.mgr.state == "PAUSED", max_s=5)
w.run(lambda x: False, max_s=40)                            # paused a while
w.mgr.resume()
w.run(lambda x: x.mgr.state != "PAUSED", max_s=5)
check("stop during a go-to, then continue: the same journey picks up (a pause is "
      "not a failure)", w.mgr.state == "TARGET_NAV", w.mgr.state)
w.run(lambda x: x.mgr.state != "TARGET_NAV", max_s=200)
check("...and arrives", [e for e in events(w, "go_to_done")][-1]["outcome"] == "ARRIVED")


# =============================================================================
print("--- U3 (spec 4.8): 'Friday stop' pauses a mission, 'continue' carries on ---")

ph = TMP / "pause" / "hold.json"
ph.parent.mkdir(parents=True, exist_ok=True)
h = wb.Hold(path=ph)
check("no pause file: not paused", not h.paused())
pp = ph.with_name(wb.PAUSE_NAME)
ad.atomic_write_json(pp, {"at": time.time(), "by": "voice"})
h.poll_pause()
check("a person's pause: held -- and it has no expiry", h.paused() and h.active())
ad.atomic_write_json(ph, {"until": time.time() + 3.0, "why": "m",
                          "drive": {"kind": "path", "id": "x", "path": [[1, 1]]}})
h.poll()
check("...a drive order beside it is NOT driven: the person's pause wins",
      h.driving() is None and h.active())
pp.write_text("{torn")
h.poll_pause()
check("...a torn read keeps the pause (never dropped on a half-written file)", h.paused())
pp.unlink()
h.poll_pause()
check("cleared by the mission: the pause is over", not h.paused())
ad.atomic_write_json(pp, {"at": time.time() - 3600.0, "by": "voice"})
h2 = wb.Hold(path=ph)
h2.poll_pause()
check("a pause written before this run began is ignored (an old pause never "
      "freezes a new run)", not h2.paused())
pp.unlink()

loop6, slots6 = new_loop()
ad.atomic_write_json(loop6.hold.pause_path, {"at": time.time(), "by": "voice"})
loop6.hold.poll_pause()
ad.atomic_write_json(loop6.hold.path, {"until": time.time() + 3.0, "why": "m",
                                       "drive": {"kind": "path", "id": "y",
                                                 "path": [[15.0, 12.0]]}})
loop6.hold.poll()
rows = []
for _ in range(5):
    slots6["lidar"].publish(open_scan)
    rows.append(loop6.tick())
check("the run's tick while paused: zero on the wire, whatever else is asked",
      all(r["gate"] == "mission_hold" and r["v"] == 0.0 and r["omega"] == 0.0
          for r in rows))
# Paused past PAUSE_MAX_S: the run ENDS standing still (isolated run(), as above).
loop7, slots7 = new_loop()
loop7.map_enabled = False
loop7.planner = None
loop7.state_file = None
loop7.blackbox.dump = lambda _p: None
loop7.report = lambda: None
loop7.shutdown = lambda: None
loop7.hold.start = lambda: None
loop7.hold.pause_path = loop7.hold.path.with_name("pause7.json")
ad.atomic_write_json(loop7.hold.pause_path, {"at": time.time(), "by": "voice"})
loop7.hold.poll_pause()
loop7.hold._paused_since = time.monotonic() - wb.PAUSE_MAX_S - 1.0
t7 = threading.Thread(target=loop7.run, kwargs={"duration": 60.0, "verbose": False},
                      daemon=True)
t7.start()
t7.join(3.0)
check("paused for over 30 min with nobody saying continue: the run ENDS (she never "
      "sets off again on her own)", not t7.is_alive())
loop7._stop.set()

from exploration_mission import voice_link as vl            # noqa: E402
vdir = TMP / "vl"
check("friday-brain writes the pause straight to the run's folder",
      vl.pause_run("voice", folder=vdir) and (vdir / vl.PAUSE_NAME).exists()
      and vl.PAUSE_NAME == wb.PAUSE_NAME)
ad.atomic_write_json(vdir / vl.STATE_NAME, {"state": "EXPLORE"})
check("...and reads the mission's published state", vl.mission_state(vdir) == "EXPLORE"
      and "EXPLORE" in vl.RUN_STATES and "IDLE" not in vl.RUN_STATES)

# --- the mission ---
w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
check("the mission publishes its state for friday-brain",
      "EXPLORE" in w.sim.states_published, str(w.sim.states_published))
w.sim.pause("voice")                    # what friday-brain writes on "Friday stop"
w.mgr.halt()
w.run(lambda x: x.mgr.state == "PAUSED", max_s=5)
here = w.sim.truth
n = len(w.voice.said)
w.run(lambda x: False, max_s=90)        # far longer than any hold's 3 s
check("'Friday stop' while exploring: PAUSED, she says so, and stays put -- the "
      "pause never lapses", w.mgr.state == "PAUSED" and w.sim.truth == here
      and "Stopping. Say continue when you're ready." in w.voice.said[-3:]
      and ("stop",) not in w.sim.orders, f"{w.mgr.state} {w.voice.said[-3:]}")
w.mgr.resume()
w.run(lambda x: x.mgr.state != "PAUSED", max_s=5)
w.run(lambda x: False, max_s=10)
check("'continue': 'Continuing exploration.', the pause is cleared, the RL drives on",
      w.mgr.state in ("EXPLORE", "ASK_LABEL") and not w.sim.paused()
      and w.sim.truth != here and "Continuing exploration." in w.voice.said,
      f"{w.mgr.state} {w.sim.orders[-3:]}")
end = w.run()
check("...and the mission finishes as normal", end == "IDLE"
      and w.mission()["status"] == "returned", w.mission()["status"])

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.sim.drive_ticks > 15, max_s=60)
w.sim.pause("voice")
w.mgr.halt()
w.run(lambda x: x.mgr.state == "PAUSED", max_s=5)
here = w.sim.truth
w.run(lambda x: False, max_s=30)
check("'Friday stop' on the way home: paused, standing", w.sim.truth == here
      and w.mgr.paused_from in ("RETURN_TO_START", None))
w.mgr.resume()
end = w.run()
check("...'continue': the way home picks up again and she arrives",
      end == "IDLE" and w.mission()["status"] == "returned" and not w.sim.paused(),
      w.mission()["status"])

w = World(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.sim.pause("voice")
w.mgr.halt()
w.run(lambda x: x.mgr.state == "PAUSED", max_s=5)
w.mgr.finish()                          # "go back" / "stop exploring" while paused
end = w.run()
check("paused, then 'go back': summary, home, the pause cleared on the way",
      end == "IDLE" and w.mission()["status"] == "returned" and not w.sim.paused())

w = World(["hall", "kitchen", "bedroom"], cfg={"return": {"help_max_cycles": 1}})
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 15)
w.mgr.finish()
w.run(lambda x: x.sim.drive_ticks > 10, max_s=40)
w.sim.freeze_until = w.clock.monotonic() + 1e9
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=3600)
w.sim.pause("voice")
w.mgr.halt()
w.run(lambda x: x.mgr.state == "IDLE", max_s=10)
check("'Friday stop' in HELP_WAIT: IDLE (the file's rule), and the mission's run "
      "ends with it -- never left out to explore on its own",
      w.mgr.state == "IDLE" and ("stop",) in w.sim.orders and not w.sim.paused()
      and w.mission()["status"] == "help_wait", w.mgr.state)

w = World([])
w.run(lambda x: x.mgr.state == "IDLE", max_s=20)
check("the state is published from boot on -- RELOCALIZE -> IDLE assigns it "
      "directly, and friday-brain must still see IDLE (found live 2026-10-04 00:07)",
      w.sim.states_published[-1:] == ["IDLE"], str(w.sim.states_published))

# A power cut leaves pause.json behind: cleared at boot -- unless a run is out.
w = World([])
w.sim.pause("voice")
w.sim.is_running = False
w.mgr.boot(w.clock.monotonic())
check("boot after a power cut: the dead run's pause is cleared",
      not w.sim.paused())
w = World([])
w.sim.pause("voice")
w.sim.is_running = True
w.mgr.boot(w.clock.monotonic())
check("...but if a run is still out (friday-mcp restarted under it) she STAYS "
      "paused: a restart never undoes a person's stop", w.sim.paused())

# --- friday-brain's side (main.py), the run faked ---
import asyncio                                                   # noqa: E402
import config                                                    # noqa: E402
import main                                                      # noqa: E402


class Turn:
    def __init__(self):
        self.notes, self.result = [], None

    def note(self, text, dash=True):
        self.notes.append(text)

    def answered_by(self, *a):
        pass

    def done(self, text=""):
        self.result = text


calls = []
real = (vl.pause_run, main._mission_claims, main._narration_off, main._run_standing_since)


async def fake_claims(session, spoken, turn, announce=True):
    calls.append(("claims", spoken))
    return True


try:
    vl.pause_run = lambda by="voice", folder=None: calls.append(("pause", by)) or True
    main._mission_claims = fake_claims
    main._narration_off = lambda *a, **k: None
    main._run_standing_since = lambda t0: True
    tr = Turn()
    ok = asyncio.run(main._mission_pause(None, tr))
    check("main: the pause file FIRST, then the mission told, then the run's "
          "confirmation -> paused, no full stop", ok and calls[:2] == [
              ("pause", "voice"), ("claims", "stop")] and "paused" in (tr.result or ""),
          str(calls))
    main._run_standing_since = lambda t0: False
    calls.clear()
    t0 = time.monotonic()
    ok = asyncio.run(main._mission_pause(None, Turn()))
    check("main: the run never confirms it stopped -> False within ~2 s: the caller "
          "does the FULL stop (a stop never depends on the new path)",
          ok is False and time.monotonic() - t0 <= main.MISSION_PAUSE_CONFIRM_S + 0.5)
finally:
    (vl.pause_run, main._mission_claims, main._narration_off,
     main._run_standing_since) = real

real_root = config.ROOT
try:
    config.ROOT = TMP / "fakeroot"
    (config.ROOT / "state").mkdir(parents=True, exist_ok=True)
    rlpath = config.ROOT / "state" / "rl_autonomy.json"
    t_before = time.time() - 1.0
    ad.atomic_write_json(rlpath, {"REASON_CODE": "mission_hold",
                                  "FINAL_EXECUTED_ACTION": {"v": 0.0, "omega": 0.0}})
    yes = main._run_standing_since(t_before)
    ad.atomic_write_json(rlpath, {"REASON_CODE": "navigator",
                                  "FINAL_EXECUTED_ACTION": {"v": 0.2, "omega": 0.0}})
    no = main._run_standing_since(t_before)
    old = main._run_standing_since(time.time() + 5.0)
    check("main reads the run's own status: zero + mission_hold = standing; moving, "
          "or a status older than the stop, is not", yes and not no and not old)
finally:
    config.ROOT = real_root
src = (ROOT / "main.py").read_text()
check("main: the wake word stays on while a mission's run is out",
      "_mission_run_state() is None)" in src)


# =============================================================================
print("--- U4 (spec 3.1 + 4.8): 'this is home' puts her home, and her rooms, on the saved map ---")

HOUSE_OG = mm.to_occupancy_grid(mm.three_rooms().grid)   # the "saved SLAM map"
START = sw.THREE_ROOM_TOUR[0]                             # (2.5, 3.0): the sim's start


def map_world(replies):
    w = World(replies)
    w.a.slam.load = lambda: HOUSE_OG                      # the saved map = the house
    w.run(lambda x: x.mgr.state == "IDLE", max_s=20)     # booted
    return w


def file10_home(w, xy, age_s=0.0):
    """What File 10's remember_place("home") leaves: the place, just seen."""
    w.a.places.upsert("home", xy[0], xy[1], 0.5, "test")
    w.a.places.rows["home"]["last_seen"] = w.a.clock.wall() - age_s


def this_is_home(w):
    w.mgr.set_home(with_place=True)
    w.run(lambda x: False, max_s=1.0)


from exploration_mission import intents                          # noqa: E402
check("'this is home' / 'yahan home hai' are SET_HOME WITH the saved map; 'set home "
      "here' is SET_HOME without it",
      all(intents.is_home_phrase(s) for s in ("this is home", "Friday, this is home",
                                             "yahan home hai"))
      and not intents.is_home_phrase("this is the kitchen")
      and [i.args for i in intents.candidates("this is home") if i.name == "SET_HOME"]
      == [{"with_place": True}]
      and [i.args for i in intents.candidates("set home here") if i.name == "SET_HOME"]
      == [{"with_place": False}])

w = map_world(["hall", "kitchen", "bedroom"])
file10_home(w, START)
this_is_home(w)
a = w.store.home_anchor("home")
# 0.20 m: the RAY-CAST sweeps' own near-face bias -- the matcher polished from
# the TRUE pose lands at (2.40, 3.10) on this map, 14 cm out (File 14's tests
# measure the same, 16 cm). The polish reaches that optimum; the lattice alone
# was 18 cm out.
check("'this is home' right after File 10 saved home here: the mission's start anchor "
      "goes ON the saved map, polished, and she confirms",
      a is not None and a["frame"] == "map"
      and math.hypot(a["x"] - START[0], a["y"] - START[1]) <= 0.20
      and "I've marked this spot as my starting point." in w.voice.said,
      f"{a and (a['frame'], round(a['x'], 3), round(a['y'], 3))}")
calls = []
real_lo = w.a.slam.localise_on
w.a.slam.localise_on = lambda og, *a_, **k: (calls.append(1) if og is HOUSE_OG
                                             else None) or real_lo(og, *a_, **k)
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=30)
T = w.mgr.T or (9, 9, 9)
check("a mission from that spot is ON the saved map from its first move (frame 'map', "
      "the start where she stands) -- and never searched the SAVED map to get there "
      "(only the anchor's own sweep, spec 4.9)",
      w.mgr.frame_label == "map" and math.hypot(T[0] - START[0], T[1] - START[1]) <= 0.25
      and calls == [], f"{w.mgr.frame_label} {T} searches={len(calls)}")
end = w.run()
hall = w.a.places.find_exact("hall")
truth = sw.ROOM_CENTRES["A"]
check("...every room it names is on the saved map AND a place at once: 'go to the "
      "hall' works for File 14 after the mission (your point 3)",
      end == "IDLE" and all(x["frame"] == "map" for x in w.areas()) and hall is not None
      and math.hypot(hall["x"] - truth[0], hall["y"] - truth[1]) <= 0.8,
      f"{[x['frame'] for x in w.areas()]} {hall}")
n = len(w.voice.said)
w.run(lambda x: False, max_s=0.1)
w.mgr.start()
w.run()
check("a second mission from home: the saved rooms are recognised -- no question",
      "What should I call it?" not in " ".join(w.voice.said[n:]) and len(w.areas()) == 3,
      str(w.voice.said[n:]))

# THE DANGER the first draft had: a home that merely EXISTS, she is elsewhere,
# three look-alike rooms -> "placed" 7 m away. Never again.
w = map_world([])
file10_home(w, (9.8, 3.0), age_s=3600.0)                  # saved an hour ago, room C
this_is_home(w)
a = w.store.home_anchor("home")
check("File 10's home NOT just saved here (an hour old, in another room): never placed "
      "on the map -- the anchor keeps its own frame", a is not None
      and a["frame"].startswith("home:"), str(a and a["frame"]))
w = map_world([])
file10_home(w, (9.8, 3.0))
w.mgr.set_home(with_place=False)                          # "set home here"
w.run(lambda x: False, max_s=1.0)
check("'set home here' (no File 10 save with it): its own frame, no search",
      w.store.home_anchor("home")["frame"].startswith("home:"))
w = map_world(["hall", "kitchen", "bedroom"])
file10_home(w, (9.8, 3.0))                                 # home exists, she is in room A
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=30)
check("a mission start with a home elsewhere: never placed on the map (own frame)",
      w.mgr.frame_label is None, str(w.mgr.frame_label))

w = map_world(["hall", "kitchen", "bedroom"])
w.mgr.start()
w.run()
frame1 = w.areas()[0]["frame"]
file10_home(w, START)
n = len(w.voice.said)
w.run(lambda x: False, max_s=0.1)
this_is_home(w)                                            # back at the start spot
w.mgr.start()
w.run()
check("rooms named BEFORE 'this is home' move onto the saved map with the old anchor "
      "(she stood at it), become places, and are not asked again",
      frame1.startswith("mission:") and all(x["frame"] == "map" for x in w.areas())
      and len(w.areas()) == 3 and w.a.places.find_exact("kitchen") is not None
      and "What should I call it?" not in " ".join(w.voice.said[n:]),
      f"{[x['frame'] for x in w.areas()]} {w.voice.said[n:]}")
src = (ROOT / "main.py").read_text()
i10, imis = src.find('args={"name": "home"}'), src.find(
    "await _mission_claims(session, spoken, turn, announce=False)\n        return \"home\"")
check("main: her permanent home (File 10) FIRST, then the mission; only in IDLE/PAUSED",
      0 < i10 < imis and '_mission_state_now() in ("", "IDLE", "PAUSED")' in src)


print("--- U4 (spec 4.7 point 9): a name suggested from what she saw ---")
w = World(["yes", "hall", "bedroom"])
w.a.detector = ad.FakeDetector(running=True)
w.a.detector.frames = [[("refrigerator", 0.82)], [("refrigerator", 0.77), ("cup", 0.9)]]
w.mgr.start()
w.run(lambda x: x.mgr.questions >= 1 and x.mgr.state != "ASK_LABEL", max_s=200)
said = " ".join(w.voice.said)
check("a detector that is ALREADY running saw a fridge in the room: 'It looks like a "
      "kitchen', and 'yes' saves it", "It looks like a kitchen." in said
      and any(a["name"] == "kitchen" for a in w.areas()), said[-300:])
w = World(["hall"])
w.mgr.start()
w.run(lambda x: x.mgr.questions >= 1 and x.mgr.state != "ASK_LABEL", max_s=200)
check("no detector running (scene.py pauses during a run): the plain question, never "
      "a detector loaded for it", "It looks like" not in " ".join(w.voice.said))
autof = TMP / "rl_autonomy.json"
ad.atomic_write_json(autof, {"OBJECTS": [{"label": "bed", "score": 0.9}],
                             "OBJECTS_FRAMES": 7})
st_ = ad.RunAdapter(state_dir=TMP / "ra", autonomy_file=autof).status()
check("the run's own detector, when it runs, is read from its published state",
      st_["objects"] == [("bed", 0.9)] and st_["objects_frames"] == 7)


class FakeDet:
    enabled, frames = True, 3

    def snapshot(self):
        from friday_vision import Detection
        return [Detection("bed", 0.91, 0.0, 0.2, None)]


loop8, slots8 = new_loop()
slots8["lidar"].publish(open_scan)
loop8.tick()
loop8.detector = FakeDet()
objs = (loop8.autonomy_state() or {}).get("OBJECTS")
loop8.detector = None
check("rl: the run publishes its detector's labels (read-only), nothing when none runs",
      objs == [{"label": "bed", "score": 0.91}]
      and "OBJECTS" not in (loop8.autonomy_state() or {}))


print("--- U4: curiosity asks, and she listens 0.3 s after she stops ---")
check("curiosity ships 'ask' (4.15): she asks, and goes only after his yes",
      cfg.curiosity.mode == "ask")
check("main waits listen_after_tts_s (0.3 s, 4.7) after her voice ends",
      '.get("listen_after_tts_s", 0.3)' in src and cfg.voice.listen_after_tts_s == 0.3)


# =============================================================================
print("--- nothing of the real run's touched ---")
after = {p: (p.stat().st_mtime_ns, p.stat().st_size) for p in REAL if p.exists()}
check("state/rl_* and explore_mode.json exactly as they were (none moved, none written)",
      after == REAL_BEFORE, str(set(REAL_BEFORE) ^ set(after)))
check("no map was set aside", set((ROOT / "state").glob("map-archive-*")) == ARCHIVES_BEFORE)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
