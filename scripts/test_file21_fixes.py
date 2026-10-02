#!/usr/bin/env python3
"""File 21, the fixes after the first real mission (2026-10-02).

    ./venv/bin/python scripts/test_file21_fixes.py

What went wrong on the robot, as whole missions in the simulated house:
  1. runs stopped friday-brain, so nothing could hear a room's name
     (rl/modes/confident_explore.env now quiets the face only);
  2. the 2-minute cap ended exploring with frontiers left in the room she was
     in -> no question, the room saved unnamed (the final ask now includes the
     room she is standing in, complete or not, saved as incomplete);
  3. HELP_WAIT could only be left by restarting friday-mcp ("Friday, stop" and
     a new "explore" now end it).
Network blocked; no robot.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import intents                 # noqa: E402
from exploration_mission import service                 # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


ASK = "I've finished exploring this area. What should I call it?"
ROOM = mm.closed_room()
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]


def booted(w: World) -> World:
    w.mgr.boot()
    w.run(lambda x: x.mgr.booted and x.mgr.state == "IDLE", max_s=60)
    return w


# =============================================================================
print("--- 1. runs keep friday-brain up (it hears the answers) ---")
env = (ROOT / "rl" / "modes" / "confident_explore.env").read_text()
quiet = [ln for ln in env.splitlines() if ln.startswith("MODE_QUIET_UNITS=")]
check("confident_explore.env quiets the face only, not friday-brain",
      quiet == ['MODE_QUIET_UNITS="friday-face.service"'], str(quiet))

# =============================================================================
print("--- 2. the time cap ends exploring: she still asks about her room ---")
# Three rooms, a 30 s cap: the run's own timer starts its way back while she
# is still in room A with frontiers left -- last night's ending.
w = booted(World(["kitchen"], world=mm.three_rooms(),
                 cfg={"explore": {"mission_cap_min": 0.5}}))
w.mgr.start()
w.run()
m = w.mission()
asked = w.voice.said.count(ASK)
areas = w.areas()
check("the run's timer ended it (time_cap)", m["stop_reason"] == "time_cap",
      str(m["stop_reason"]))
check("...and she asked about the room she was in before going home",
      asked == 1 and any(a["name"] == "kitchen" for a in areas),
      f"asked {asked}, areas {[(a['name'], a['complete']) for a in areas]}")
from types import SimpleNamespace                       # noqa: E402
from exploration_mission import segmentation as seg     # noqa: E402
lp = seg.LabelParams()
room = SimpleNamespace(key=7, area_m2=9.0, complete=False, area_id=None)
dw = SimpleNamespace(inside_s={7: 3.0}, exited_at={}, current=7)
check("the rule: an INCOMPLETE room she is standing in qualifies at the end "
      "(final ask) ...", seg.why_not(room, dw, 100.0, lp, final=True) == [],
      str(seg.why_not(room, dw, 100.0, lp, final=True)))
check("...but never mid-run (frontiers left = keep exploring it)",
      "not complete" in seg.why_not(room, dw, 100.0, lp, final=False))
other = SimpleNamespace(key=8, area_m2=9.0, complete=False, area_id=None)
check("...and an incomplete room she is NOT in is still not asked at the end",
      "not complete" in seg.why_not(other, dw, 100.0, lp, final=True))
fake = SimpleNamespace(segments=[room])
choice = seg.choose_label_target(fake, dw, 100.0, None, True, lp, ignore_cooldown=True)
check("...and it is recorded as incomplete, not passed off as complete",
      choice is not None and choice.complete is False, str(choice))
check("...the run was HELD for the question (it stood still while she asked)",
      any(o[0] == "hold" and "asking" in str(o) for o in w.sim.orders),
      str([o for o in w.sim.orders if o[0] == "hold"][:3]))
check("...and she still got home", w.mgr.state == "IDLE"
      and m["status"] in ("returned", "arrived"), f"{w.mgr.state} {m['status']}")
said = " ".join(w.voice.said)
check("the summary names the room", "kitchen" in said.split("Exploration complete")[-1]
      or "kitchen" in said, said[-300:])

# =============================================================================
print("--- 3. HELP_WAIT has ways out ---")


def help_world() -> World:
    """A one-room mission whose way back gives up 1.1 m out -> HELP_WAIT."""
    w = booted(World(["study"], world=ROOM, path=ROOM_PATH, fail_return_at_m=1.5))
    w.mgr.start()
    w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=400)
    return w


w = help_world()
check("the way back gave up: HELP_WAIT", w.mgr.state == "HELP_WAIT", w.mgr.state)
service._manager, service._router = w.mgr, None
r = service.voice("stop")
w.step()
check("'Friday, stop' in HELP_WAIT -> the mission closes, IDLE",
      r["handled"] and w.mgr.state == "IDLE" and w.mgr.mission is None,
      f"{r} {w.mgr.state}")
check("...the mission stays saved as help_wait (nothing lost)",
      w.mission()["status"] == "help_wait")

w = help_world()
service._manager, service._router = w.mgr, None
w.sim.truth = (3.0, 2.5, 0.0)                           # someone carried her back
n_before = len(w.store.missions("home"))
r = service.voice("explore")
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=60)
check("'explore' in HELP_WAIT -> a NEW mission starts (no restart needed)",
      r["handled"] and len(w.store.missions("home")) == n_before + 1
      and w.mgr.state == "EXPLORE", f"{r} {w.mgr.state}")
w.run()

w = help_world()
service._manager, service._router = w.mgr, None
r = service.start_mission(2)
w.run(lambda x: x.mgr.state in ("MISSION_INIT", "EXPLORE"), max_s=10)
check("the dashboard's Explore button in HELP_WAIT -> a new mission",
      r["ok"] and w.mgr.state in ("MISSION_INIT", "EXPLORE"), f"{r} {w.mgr.state}")

ro = intents.Router(type("A", (), {"__getattr__": lambda s, n: (lambda *a, **k: True)})(),
                    resolve_area=lambda n: None)
check("'explore' is still refused while she is actually exploring",
      ro.handle("explore", intents.Context(state="EXPLORE")) is True)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
