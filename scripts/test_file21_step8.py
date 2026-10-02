#!/usr/bin/env python3
"""File 21 step 8: the return ladder, HELP, and the arrival check.

    ./venv/bin/python scripts/test_file21_step8.py

Network blocked; no robot. File 14's engine is a scripted fake
(adapters.FakeNav): each attempt's outcome is given in advance and takes fake
clock time. The second half runs whole missions in the simulated house, where
the run's own way back is made to fail so the ladder has to take over.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402
from exploration_mission import home_anchor as ha       # noqa: E402
from exploration_mission import rth_ladder as rl        # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402
import synth                                            # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


CFG = rl.LadderConfig.from_config(mcfg.load())


def ladder(script, grid=None, trail=None, **nav_kw):
    clock = ad.FakeClock()
    nav = ad.FakeNav(clock=clock, script=script, **nav_kw)
    said: list = []
    lad = rl.Ladder(nav, CFG, lambda key, **f: said.append((key, f)),
                    {"goal": (0.0, 0.0), "trail": trail or [], "grid": grid,
                     "inflate_m": 0.30, "body_radius_m": 0.21, "obstacles": None})
    lad.begin(clock.monotonic())
    return lad, nav, clock, said


def climb(lad, clock, first=None, max_s=2000.0, until=None):
    """The run's way back fails (unless `first` says otherwise); then tick."""
    out = lad.external_result(bool(first), 1.5, clock.monotonic())
    t = 0.0
    while out is None and t < max_s and not (until and until(lad)):
        clock.advance(0.1)
        t += 0.1
        out = lad.tick(clock.monotonic())
    return out, t


def kinds(nav):
    return [c[0] for c in nav.calls]


# =============================================================================
print("--- the ladder (spec 4.11) ---")

lad, nav, clock, said = ladder([])
out = lad.external_result(True, 0.22, clock.monotonic())
check("the run's own way back arrived: done at rung 0, File 14 never used",
      out == "arrived" and nav.calls == [] and lad.rungs_used()
      == [{"rung": 0, "result": "arrived", "distance_m": 0.22, "duration_s": 0.0,
           "cycle": 0}])

lad, nav, clock, said = ladder(["ARRIVED"])
out, t = climb(lad, clock)
check("one failure -> rung 1 (wait 5 s) -> the next attempt succeeds",
      out == "arrived" and [a.rung for a in lad.attempts] == [0, 1]
      and [a.result for a in lad.attempts] == ["failed", "arrived"]
      and kinds(nav) == ["attempt"], str(lad.rungs_used()))
check("...it waited the 5 s before trying (plus the 4 s drive)", 8.9 <= t <= 9.5, str(t))
check("...and said nothing: rung 1 is silent", said == [])

# A corridor map for rungs 5 and 6: the start is (0, 0) in the start frame.
g = gu.Grid(np.full((120, 60), gu.OCC, np.int8), 0.05, (-1.0, -1.5))
g.occ[4:116, 10:50] = gu.FREE                           # x -0.8..4.8, y -1.0..1.0
trail = [(i * 0.25, 0.0, float(i)) for i in range(13)]  # 0 .. 3 m along x

lad, nav, clock, said = ladder([], grid=g, trail=trail, position=(3.0, 0.1, 0.0))
out, t = climb(lad, clock, until=lambda x: x.in_help)   # every attempt fails
check("scripted failures walk the rungs 1 -> 6 in order",
      kinds(nav) == ["attempt", "clear", "attempt", "spin", "attempt", "backup",
                     "attempt", "attempt", "attempt", "stop"], str(kinds(nav)))
check("...an attempt follows every rung, numbered 0..6",
      [a.rung for a in lad.attempts] == [0, 1, 2, 3, 4, 5, 6]
      and all(a.result == "failed" for a in lad.attempts), str(lad.rungs_used()))
check("...rung 2 clears File 14's runtime obstacles BEFORE its attempt",
      kinds(nav).index("clear") < [i for i, k in enumerate(kinds(nav))
                                   if k == "attempt"][1])
check("...rung 4 backs up the configured 0.25 m",
      ("backup", 0.25) in nav.calls)
check("...she speaks only on rungs 3, 5, 6 and HELP",
      [k for k, _ in said] == ["rung3", "rung5", "rung6", "help"], str(said))
check("...the help line carries the distance",
      said[-1][0] == "help" and abs(said[-1][1]["d"] - 3.0) < 0.05, str(said[-1]))
lad0, nav0, clock0, said0 = ladder([])                  # no mission map at all
climb(lad0, clock0, until=lambda x: x.in_help)
check("with no map there is no relaxed goal to find: rung 6 is skipped, not faked",
      [(a.rung, a.result) for a in lad0.attempts][-1] == (6, "skipped")
      and ("rung6", {}) in said0)
check("...and after rung 6 she is in HELP, with the engine stopped",
      lad.in_help and lad.phase == "help_pause" and nav.calls[-1] == ("stop",))

lad, nav, clock, said = ladder([], rear_m=0.20)         # something 20 cm behind
out, t = climb(lad, clock, until=lambda x: x.in_help)
check("rear blocked (0.20 m < 0.35 m): rung 4 is skipped, no back-up, no attempt",
      "backup" not in kinds(nav) and [(a.rung, a.result) for a in lad.attempts][4]
      == (4, "skipped") and [a.rung for a in lad.attempts] == [0, 1, 2, 3, 4, 5, 6],
      str(lad.rungs_used()))
lad, nav, clock, said = ladder([], rear_m=None)         # rear LiDAR blind
climb(lad, clock, until=lambda x: x.in_help)
check("rear unknown (no reading) is not 'clear': skipped too",
      "backup" not in kinds(nav))

lad, nav, clock, said = ladder(["FAILED"] * 4 + ["ARRIVED"], grid=g, trail=trail,
                               position=(3.0, 0.1, 0.0))
out, t = climb(lad, clock)
five = [c for c in nav.calls if c[0] == "attempt"][4]
check("the planner keeps failing but the breadcrumbs work: rung 5 arrives",
      out == "arrived" and lad.attempts[-1].rung == 5
      and lad.attempts[-1].result == "arrived", str(lad.rungs_used()))
check("...rung 5 hands File 14 the backtracked waypoints (a straight trail is "
      "two: where she is, and the start)", five[2] == 2, str(five))
check("...and said 'Retracing my path.'", ("rung5", {}) in said)

blocked = gu.Grid(g.occ.copy(), g.resolution, g.origin)
bx, by = blocked.world_to_cell(0.0, 0.0)
blocked.occ[bx - 3:bx + 4, by - 3:by + 4] = gu.OCC      # a 35 cm box ON the start
rg = rl.relaxed_goal(blocked, (0.0, 0.0), (3.0, 0.1), 0.30, 0.35, 0.05)
check("a big box on the start: nothing free within 0.35 m of it -> no relaxed goal",
      rg is None, str(rg))
box15 = gu.Grid(g.occ.copy(), g.resolution, g.origin)
box15.occ[bx - 1:bx + 2, by - 1:by + 2] = gu.OCC        # a 15 cm box on the start
rg = rl.relaxed_goal(box15, (0.0, 0.0), (3.0, 0.1), 0.30, 0.35, 0.05)
check("a 15 cm box on the start: the nearest cell she can stand on, 0.30-0.35 m "
      "out, clear of the box by File 14's inflation",
      rg is not None and 0.30 <= math.hypot(*rg) <= 0.35
      and gu.traversable(box15, 0.30)[box15.world_to_cell(*rg)], str(rg))
small = gu.Grid(g.occ.copy(), g.resolution, g.origin)
small.occ[bx, by] = gu.OCC                              # one cell, 0.30 m inflation
rg = rl.relaxed_goal(small, (0.0, 0.0), (3.0, 0.1), 0.05, 0.35, 0.05)
check("one occupied cell on the start: the ring search takes the NEAREST free, "
      "reachable cell -- a neighbour, not the start's own cell",
      rg is not None and 0.0 < math.hypot(*rg) <= 0.10
      and small.world_to_cell(*rg) != (bx, by)
      and not small.occupied()[small.world_to_cell(*rg)], str(rg))
rg0 = rl.relaxed_goal(g, (0.0, 0.0), (3.0, 0.1), 0.30, 0.35, 0.05)
check("a free start: the relaxed goal is the start's own cell",
      rg0 is not None and math.hypot(*rg0) <= 0.05, str(rg0))
cut = gu.Grid(g.occ.copy(), g.resolution, g.origin)
cx, _ = cut.world_to_cell(1.5, 0.0)
cut.occ[cx:cx + 2, :] = gu.OCC                          # a wall across the corridor
check("a free start she cannot REACH is no goal",
      rl.relaxed_goal(cut, (0.0, 0.0), (3.0, 0.1), 0.05, 0.35, 0.05) is None)
lad, nav, clock, said = ladder(["FAILED"] * 5 + ["ARRIVED"], grid=small, trail=trail,
                               position=(3.0, 0.1, 0.0))
lad.plan["inflate_m"] = 0.05
out, t = climb(lad, clock)
six = [c for c in nav.calls if c[0] == "attempt"][5]
check("rung 6 drives to that relaxed goal, not the blocked start, and arrives",
      out == "arrived" and lad.attempts[-1].rung == 6 and six[1] != (0.0, 0.0)
      and math.hypot(*six[1]) <= 0.35 and ("rung6", {}) in said, str(six))

lad, nav, clock, said = ladder(["REFUSED", "ARRIVED"])
out, t = climb(lad, clock)
check("an attempt File 14 REFUSES (no match) counts as a failure: next rung",
      out == "arrived" and [(a.rung, a.result) for a in lad.attempts]
      == [(0, "failed"), (1, "refused"), (2, "arrived")], str(lad.rungs_used()))
lad, nav, clock, said = ladder(["ARRIVED"], attempt_s=10_000.0)
out, t = climb(lad, clock, until=lambda x: len(x.attempts) >= 2)
check("an attempt that never ends is stopped and counted as a timeout",
      lad.attempts[1].result == "timeout" and ("stop",) in nav.calls
      and lad.attempts[1].duration_s >= rl.ATTEMPT_TIMEOUT_S, str(lad.rungs_used()))


# =============================================================================
print("--- HELP: bounded, never silent ---")

lad, nav, clock, said = ladder([])
out, t = climb(lad, clock, max_s=40_000)
helps = [i for i, (k, _) in enumerate(said) if k == "help"]
check("everything fails: HELP cycles, then HELP_WAIT", out == "help_wait")
check("...after exactly help_max_cycles (20) more goes at rungs 1-6",
      lad.cycle == 20 and max(a.cycle for a in lad.attempts) == 20, str(lad.cycle))
check("...each cycle re-runs rungs 1-6 (6 attempts a cycle, back-up included)",
      sum(1 for a in lad.attempts if a.cycle == 7) == 6
      and kinds(nav).count("backup") == 21, f"{kinds(nav).count('backup')}")
check("...the help line is repeated every 3rd cycle, not every retry",
      len(helps) == 1 + 6, f"{len(helps)} help lines")
check("...and the rung lines are said once, never again in HELP",
      [k for k, _ in said].count("rung3") == 1 and [k for k, _ in said].count("rung5") == 1)
check("...30 s apart: about 20 cycles x (30 s + the rungs' own time)", t > 20 * 30, str(t))
lad, nav, clock, said = ladder(["FAILED"] * 6 + ["FAILED"] * 8 + ["ARRIVED"])
out, t = climb(lad, clock, max_s=5000)
check("a HELP cycle that gets through arrives (she was freed on cycle 2)",
      out == "arrived" and lad.attempts[-1].cycle == 2, str(lad.rungs_used()[-2:]))
lad, nav, clock, said = ladder([])
climb(lad, clock, max_s=40_000)
nav.script = ["ARRIVED"]
out = lad.restart(clock.monotonic())
while out is None:
    clock.advance(0.1)
    out = lad.tick(clock.monotonic())
check("'Friday, come back' restarts the ladder from a plain attempt",
      out == "arrived" and lad.attempts[-1].rung == 0 and not lad.in_help)

lad, nav, clock, said = ladder([], engine=False)
out = lad.external_result(False, 1.4, clock.monotonic())
check("no engine to climb with (File 14 not wired): straight to HELP_WAIT, one "
      "help line, no ten minutes of pretend attempts",
      out == "help_wait" and said == [("help", {"d": 1.4})] and len(lad.attempts) == 1
      and "attempt" not in kinds(nav), str(said))
check("...the distance in that line is the run's last known one, never 0.0",
      said[0][1]["d"] == 1.4)


# =============================================================================
print("--- arrival (spec 4.12) ---")

ac = rl.ArrivalCheck(0.35, 0.10, 2.0)
check("outside tolerance: not arrived", not ac.update(0.50, 0.0))
check("inside tolerance, but only just: not yet", not ac.update(0.30, 1.0))
check("still inside 2 s later: arrived", ac.update(0.30, 3.0))
ac = rl.ArrivalCheck(0.35, 0.10, 2.0)
seq = [(0.30, 0.0), (0.38, 0.5), (0.33, 1.0), (0.41, 1.5), (0.34, 2.1)]
res = [ac.update(d, t) for d, t in seq]
check("jitter around 0.35 m (0.30, 0.38, 0.33, 0.41, 0.34) does not reset the dwell",
      res == [False, False, False, False, True], str(res))
ac = rl.ArrivalCheck(0.35, 0.10, 2.0)
res = [ac.update(d, t) for d, t in [(0.30, 0.0), (0.50, 1.0), (0.30, 1.5), (0.30, 2.5),
                                     (0.30, 3.6)]]
check("really moving away (beyond 0.45 m) DOES reset it",
      res == [False, False, False, False, True], str(res))
ac = rl.ArrivalCheck(0.35, 0.10, 2.0)
ac.update(0.30, 0.0)
check("an unknown distance changes nothing", ac.update(None, 1.0) is False
      and ac.update(None, 2.5) is True)

walls = synth.room_walls()
start_sig = ha.signature([synth.ray_cast(walls, (1.2, 1.4, 0.3))], 8.0)
same = rl.verify([synth.ray_cast(walls, (1.25, 1.45, 1.1), noise_m=0.01,
                                 rng=np.random.default_rng(1)) for _ in range(5)],
                 list(start_sig))
check("the look at arrival: back at the start (5 cm off, turned) -> verified",
      same["verified"] and abs(same["shift_deg"] - 46) <= 3, str(same))
other = rl.verify([synth.ray_cast(walls, (3.4, 3.0, 0.3))], list(start_sig))
check("...somewhere else -> not verified", not other["verified"], str(other))
check("...no sweep, or no start signature -> not verified, no crash",
      not rl.verify([], list(start_sig))["verified"]
      and not rl.verify([{0: 1000}], None)["verified"])
stored = [None if not math.isfinite(v) else float(v) for v in start_sig]
check("...the start signature as stored (NaN as null) works",
      rl.verify([synth.ray_cast(walls, (1.2, 1.4, 0.3))], stored)["verified"])


# =============================================================================
print("--- whole missions: the run's way back fails ---")

ROOM = mm.closed_room()
PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


def world(script, engine=True, **kw):
    # These missions test the run's OWN way back first, then the ladder: the
    # "run_first" order. Since 2026-10-02 the default is "map_first" (File 14
    # on the mission's map from the start), tested in test_file21_return.py.
    cfg = kw.pop("cfg", {})
    cfg = {**cfg, "return": {**cfg.get("return", {}), "mode": "run_first"}}
    w = World(["kitchen"], world=ROOM, path=PATH, cfg=cfg, **kw)
    w.a.nav = ad.FakeNav(during_run=False, clock=w.clock, script=script,
                         engine=engine)
    w.mgr.a.nav = w.a.nav
    return w


w = world([])
w.mgr.start()
end = w.run()
m = w.mission()
check("the run's own way back arrives: returned, one attempt, rung 0",
      end == "IDLE" and m["status"] == "returned" and m["rth_attempts"] == 1
      and m["rungs_used"][0]["rung"] == 0 and m["rungs_used"][0]["result"] == "arrived",
      str(m["rungs_used"]))
check("...the look at arrival matched the start: verified",
      m["rungs_used"][-1] == {"rung": "arrival", "verified": True}
      and any(e["kind"] == "arrival_check" and e.get("verified") for e in m["events"]),
      str(m["rungs_used"]))
check("...the hardware was reclaimed from the ended run before looking",
      ("stop",) in w.sim.orders and not w.a.nav.calls)
check("...'I've reached where I started. Exploration saved.' said after saving",
      w.voice.said[-1] == "I've reached where I started. Exploration saved.")

w = world(["FAILED", "ARRIVED"], fail_return_at_m=1.2)
w.mgr.start()
end = w.run()
m = w.mission()
check("the run gives up 1.2 m out: the ladder takes over and gets her home",
      end == "IDLE" and m["status"] == "returned"
      and [(r["rung"], r["result"]) for r in m["rungs_used"][:3]]
      == [(0, "failed"), (1, "failed"), (2, "arrived")], str(m["rungs_used"]))
check("...rth_attempts and rungs_used are in the mission row",
      m["rth_attempts"] == 3 and len(m["rungs_used"]) == 4)
check("...the ladder's goal is the start: (0, 0) in the mission's frame",
      all(c[1] == (0.0, 0.0) for c in w.a.nav.calls if c[0] == "attempt"))

w = world([], fail_return_at_m=1.2)
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP", max_s=600)
check("every rung fails: the mission is in HELP and says how far she is",
      w.mgr.state == "HELP" and any(s.startswith("I'm stuck about ")
                                    and s.endswith("Please help me or clear the way.")
                                    for s in w.voice.said), str(w.voice.said[-3:]))
check("...the router sees HELP as a mission state (stop still works)",
      w.mgr.context().mission_active)
w.a.nav.script = ["ARRIVED"]
end = w.run(max_s=200)
check("...cleared by a person: the next HELP cycle gets through and she arrives",
      end == "IDLE" and w.mission()["status"] == "returned"
      and w.mission()["rungs_used"][-2]["cycle"] == 1, str(w.mission()["rungs_used"][-3:]))

w = world([], fail_return_at_m=1.2)
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=40_000)
check("never cleared: after 20 cycles she parks in HELP_WAIT with one last line",
      w.mgr.state == "HELP_WAIT" and w.mission()["status"] == "help_wait"
      and w.voice.said[-1] == "I'll wait here. Say Friday, come back, to try again.")
check("...and does nothing more on her own",
      w.run(lambda x: False, max_s=120) == "HELP_WAIT")
w.a.nav.script = ["ARRIVED"]
w.mgr.finish()                                          # "Friday, come back"
end = w.run(max_s=100)
check("'Friday, come back' restarts the ladder and she arrives",
      end == "IDLE" and w.mission()["status"] == "returned", str(w.states[-4:]))

w = world([], fail_return_at_m=1.2)
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP", max_s=600)
w.mgr.halt()                                            # "Friday stop"
w.step()
check("'Friday stop' in HELP: no more tries, IDLE, saved as help_wait",
      w.mgr.state == "IDLE" and w.mission()["status"] == "help_wait")

w = world([], engine=False, fail_return_at_m=1.2)
w.mgr.start()
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=300)
check("File 14 not wired (today's real adapter): the help line, then HELP_WAIT -- "
      "at once, not after ten minutes",
      w.mgr.state == "HELP_WAIT" and w.voice.said[-2].startswith("I'm stuck about 1.")
      and w.voice.said[-1].startswith("I'll wait here."), str(w.voice.said[-3:]))

w = world(["ARRIVED"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "RETURN_TO_START")
real_sweep = w.a.lidar.sweep
w.a.lidar.sweep = lambda: synth.grid_ray_cast(
    mm.three_rooms().grid.occupied(), (0.0, 0.0), 0.05, (9.8, 3.0, 0.0))
end = w.run()
m = w.mission()
check("the pose says arrived but the view does not match (the room changed): "
      "ONE spin + relocalise, then arrived all the same, verified = false",
      end == "IDLE" and m["status"] == "returned"
      and m["rungs_used"][-1] == {"rung": "arrival", "verified": False}
      and [c[0] for c in w.a.nav.calls].count("spin") == 1, str(m["rungs_used"])
      + str(w.a.nav.calls))
check("...'Checking where I am.' was said for that spin, once",
      w.voice.said.count("Checking where I am.") == 1)
checks = [e for e in m["events"] if e["kind"] == "arrival_check"]
check("...two looks logged, never a third", len(checks) == 2
      and checks[0].get("retry") and not checks[1].get("verified"), str(checks))

w = World(["kitchen"], world=ROOM, path=PATH)
w.a.power = ad.FakePower(low=None)
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and x.t > 8)
w.a.power._low = True
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=10)
check("power low: straight to HELP_WAIT with the battery line, run stopped",
      w.mgr.state == "HELP_WAIT" and w.voice.said[-1]
      == "My battery is low, so I'm stopping here." and ("stop",) in w.sim.orders
      and w.mission()["stop_reason"] == "power_low")
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
