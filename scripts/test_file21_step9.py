#!/usr/bin/env python3
"""File 21 step 9: boot restore, and a mission a power cut interrupted.

    ./venv/bin/python scripts/test_file21_step9.py

Network blocked; no robot. A "power cut" in these tests is what it is on the
robot: the manager and the run vanish mid-mission with nothing closed, and a
new manager comes up on the same temp database and files. Claude Code never
reboots the Pi: the real reboot / power-cut check is Harsh's H6.
"""

from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import boot                    # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import durable                 # noqa: E402
from exploration_mission import intents                 # noqa: E402
from exploration_mission import mission_map as mmap     # noqa: E402
from exploration_mission import service                 # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402
from harness import World                               # noqa: E402
import make_maps as mm                                  # noqa: E402
import numpy as np                                      # noqa: E402
import simworld as sw                                   # noqa: E402
import synth                                            # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


ROOM = mm.closed_room()
ROOM_PATH = [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0), (2.0, 3.0), (3.0, 2.5)]
MOTION = ("start", "hold", "return", "stop")


FIRST: dict = {}


def booted(w: World, max_s: float = 120.0) -> str:
    w.mgr.boot()
    FIRST["state"] = w.mgr.state            # before any tick
    return w.run(lambda x: x.mgr.state == "IDLE" and x.mgr.booted, max_s=max_s)


def finished_mission(**kw) -> World:
    """A whole one-room mission, run to the end; she is back at the start."""
    w = World(["save it as kitchen"], world=ROOM, path=ROOM_PATH, **kw)
    w.mgr.start()
    w.run()
    return w


def saved_map_world(distinct: bool) -> World:
    """A house with its saved SLAM map. `distinct`: rooms of different sizes
    (fits one way only); otherwise three identical rooms (look-alikes)."""
    world = mm.uneven_rooms() if distinct else mm.three_rooms()
    path = sw.UNEVEN_TOUR if distinct else sw.THREE_ROOM_TOUR
    w = World(["save it as hall", "isko kitchen bolo", "ye bedroom hai"],
              world=world, path=path)
    w.a.slam.map_file.parent.mkdir(parents=True, exist_ok=True)
    mm.to_occupancy_grid(w.sim.world.grid).save(w.a.slam.map_file)   # her saved map
    del w.a.slam.localise                   # the real one-sweep search ...
    w.a.slam.localise = lambda scan: None   # ... which refuses at the start
    return w


# =============================================================================
print("--- boot with data, standing at home (spec 4.14 a) ---")

w = finished_mission()
w.power_cut()                                           # a clean reboot, in effect
said_before = len(w.voice.said)
end = booted(w)
check("BOOT_RESTORE -> RELOCALIZE -> IDLE", FIRST["state"] == "BOOT_RESTORE"
      and w.seen() == ["RELOCALIZE", "IDLE"], f"{FIRST} {w.states}")
check("placed by the HOME CHECK: the view matches the saved start signature",
      w.mgr.placed == "home" and w.mgr.localized, str(w.mgr.place))
check("...areas and the anchor survived the reboot",
      [a["name"] for a in w.areas()] == ["kitchen"]
      and w.store.home_anchor("home") is not None)
check("...nothing was interrupted, so she says nothing at boot",
      len(w.voice.said) == said_before and not w.mgr.interrupted)
check("...and boot never drives: no start, hold, return or stop was ordered",
      not w.orders(*MOTION), str(w.sim.orders))
for _ in range(900):                                    # 90 s idle
    w.step()
check("...90 s later still IDLE, still nothing ordered",
      w.mgr.state == "IDLE" and not w.orders(*MOTION))

print("--- the chain, link by link (no motion in any) ---")

cfg = mcfg.load()
sweep_home = w.a.lidar.sweep()
anchor = w.store.home_anchor("home")
loaded = {"anchor": anchor, "checkpoint": None}


class Slam:
    def __init__(self, fix):
        self.fix, self.calls = fix, 0

    def localise(self, scan):
        self.calls += 1
        return self.fix


slam = Slam((9.0, 9.0, 0.0))
r = boot.relocalise([sweep_home] * 3, loaded, slam, cfg)
check("(a) matches: placed 'home' and File 14's search is never even run",
      r["placed"] == "home" and slam.calls == 0 and r["localized"])
w.sim.truth = (2.0, 3.2, 1.0)                           # carried somewhere else
sweep_away = w.a.lidar.sweep()
T = (10.0, 5.0, 0.3)
cp = {"anchor": list(T), "pose": [1.0, 0.5, 0.2], "state": "EXPLORE"}
was = mmap.compose(T, (1.0, 0.5, 0.2))
near = (was[0] + 0.2, was[1] - 0.1, was[2])
r = boot.relocalise([sweep_away], {"anchor": anchor, "checkpoint": cp}, Slam(near), cfg)
check("moved: (a) fails -> (b) the global fix agrees with the last checkpoint",
      r["placed"] == "checkpoint" and r["pose"] == near, str(r.get("how")))
r = boot.relocalise([sweep_away], {"anchor": anchor, "checkpoint": cp},
                    Slam((0.0, 0.0, 0.0)), cfg)
check("...(c) a sure fix somewhere else -> placed 'global'", r["placed"] == "global")
r = boot.relocalise([sweep_away], {"anchor": anchor, "checkpoint": None},
                    Slam((0.0, 0.0, 0.0)), cfg)
check("...no checkpoint to agree with -> 'global' too", r["placed"] == "global")
r = boot.relocalise([sweep_away], {"anchor": anchor, "checkpoint": cp}, Slam(None), cfg)
check("...(d) File 14 is not sure either -> not placed, but the LiDAR works",
      r["placed"] is None and r["localized"] is True)
r = boot.relocalise([], {"anchor": anchor, "checkpoint": cp}, Slam((1, 1, 0)), cfg)
check("...no sweeps at all -> not localized (she cannot mark a start)",
      r["placed"] is None and r["localized"] is False)
check("the checkpoint's pose on the saved map is only known for an anchored mission",
      boot.checkpoint_pose_on_map({"anchor": None, "pose": [1, 1, 0]}) is None
      and boot.checkpoint_pose_on_map(cp) == was)

w.power_cut()
w.a.slam.localise = lambda scan: None
booted(w)
check("booted away from home with File 14 unsure: not placed, nothing ordered",
      w.mgr.placed is None and w.mgr.localized and not w.orders(*MOTION))
router = intents.Router(w.mgr, lambda n: w.store.by_name("home", n),
                        w.a.places.find_exact, monotonic=w.clock.monotonic)
check("...NOT PLACED DOES NOT REFUSE A MISSION: 'explore' still starts one "
      "(her areas will stay in its own frame)",
      router.handle("explore", w.mgr.context()) and w.step() == "MISSION_INIT"
      and "I'm not sure where I am" not in " ".join(w.voice.said[-2:]))
w.mgr.halt()
w.run(max_s=5)

w = finished_mission()
w.power_cut()
w.sim.truth = (2.0, 3.2, 1.0)
w.a.slam.localise = lambda scan: None
booted(w)
check("moved while off: not placed", w.mgr.placed is None)
w.sim.truth = (3.0, 2.5, 0.0)                            # put back home
w.run(lambda x: x.mgr.placed == "home", max_s=cfg.boot.home_check_period_s + 10)
check("...put back on her home spot: the 30 s home re-check places her, "
      "still without moving", w.mgr.placed == "home" and not w.orders(*MOTION))


# =============================================================================
print("--- a power cut in the middle of exploring (spec 4.14.5) ---")

w = World(["save it as hall", "isko kitchen bolo", "bedroom"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.mgr.state == "EXPLORE")
w.run(lambda x: x.t > 45)                               # on into room B
old_id = w.mgr.mission["id"]
crumbs_before = len(w.mgr.trail)
pose_before = w.mgr.pose
w.mgr._checkpoint(w.clock.monotonic())                  # the 10 s checkpoint lands
truth_at_cut = w.sim.truth
w.power_cut(rest_of_path=sw.THREE_ROOM_TOUR[9:])
check("after the cut the disk still says a mission was running",
      w.store.load_checkpoint("home")[0]["state"] == "EXPLORE"
      and w.store.mission(old_id)["status"] == "running")
booted(w)
check("boot notices: the mission is marked interrupted, and kept whole",
      w.mgr.interrupted and w.store.mission(old_id)["status"] == "interrupted"
      and [a["name"] for a in w.areas()] == ["hall"])
check("...she says so, with the count: 'I saved 1 area. Say continue ...'",
      w.voice.said[-1] == "My last exploration was interrupted. I saved 1 area. "
                          "Say continue to resume, or explore to start again.",
      w.voice.said[-1])
for _ in range(600):
    w.step()
check("...and DOES NOT MOVE: a minute later, IDLE, nothing ordered",
      w.mgr.state == "IDLE" and not w.orders(*MOTION) and w.sim.truth == truth_at_cut)
check("the router knows a mission is waiting",
      w.mgr.context().interrupted and not w.mgr.context().mission_active)

router = intents.Router(w.mgr, lambda n: w.store.by_name("home", n),
                        w.a.places.find_exact, monotonic=w.clock.monotonic)
router.handle("Friday, continue", w.mgr.context())
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=30)
check("'continue': the same mission is running again (not a new one)",
      w.mgr.mission["id"] == old_id and w.store.mission(old_id)["status"] == "running"
      and w.orders("start") and "Continuing exploration." in w.voice.said)
check("...the trail, the saved areas and the counters came back from the checkpoint",
      len(w.mgr.trail) >= crumbs_before and len(w.mgr.new_area_ids) == 1
      and w.mgr.distance_m > 5)
true_pose = (w.sim.truth[0] - 2.5, w.sim.truth[1] - 3.0)  # the mission began at (2.5, 3)
check("...she was FOUND on the mission's own saved map: although she had turned "
      "90 deg since the last checkpoint, new fixes land in the OLD mission's frame "
      "(within 15 cm of the truth)",
      math.hypot(w.mgr.pose[0] - true_pose[0], w.mgr.pose[1] - true_pose[1]) < 0.15,
      f"{w.mgr.pose} vs {true_pose}")
ev = [e for e in w.store.mission(old_id)["events"]
      if e["kind"] == "resumed_after_interruption"]
check("...the resume is in the event log with the offset it used", len(ev) == 1
      and len(ev[0]["offset"]) == 3)
end = w.run(lambda x: x.mgr.state in ("HELP_WAIT", "IDLE") and x.t > 5, max_s=400)
names = sorted(a["name"] for a in w.areas())
check("...she finishes the house: the other rooms get their names too",
      names == ["bedroom", "hall", "kitchen"], str(names))
hall = w.store.by_name("home", "hall")
bed = w.store.by_name("home", "bedroom")
check("...old and new areas line up in ONE frame (hall at the start, bedroom "
      "about 7 m along)", abs(hall["center_x"]) < 1.0 and 6.0 < bed["center_x"] < 8.5,
      f"{hall['center_x']:.1f}, {bed['center_x']:.1f}")
far = math.hypot(*pose_before[:2])
last = [s for s in w.voice.said if s.startswith("I'm stuck about ")]
check("...the RUN's way back ends where she RESUMED, not at the mission's start: "
      "with no File 14 engine she says how far she really is and waits",
      w.mgr.state == "HELP_WAIT" and last
      and abs(float(last[-1].split("about ")[1].split(" ")[0]) - far) < 0.8
      and w.store.mission(old_id)["status"] == "help_wait", f"{last[-1:]} vs {far:.1f}")

print("--- ...with File 14 wired, the same resumed mission gets home ---")
w = World(["save it as hall", "kitchen", "bedroom"],
          cfg={"return": {"mode": "map_first"}})       # the map-first order
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 45)
w.mgr._checkpoint(w.clock.monotonic())
w.power_cut(rest_of_path=sw.THREE_ROOM_TOUR[9:])
w.a.nav = ad.FakeNav(during_run=False, clock=w.clock, script=["ARRIVED"])
w.mgr.a.nav = w.a.nav
booted(w)
w.mgr.resume()
end = w.run(max_s=500)
m = w.mission()
# Since 2026-10-02 the way home is MAP FIRST: the run is parked when exploring
# ends and File 14's first attempt drives to the mission's REAL start -- the
# resumed run's own start (the resume point) is never the goal.
attempts = [n for n in w.a.nav.calls if n[0] == "attempt"]
check("continue -> explore -> the run is PARKED, and File 14's first attempt "
      "takes her to the mission's real start -> returned",
      end == "IDLE" and m["status"] == "returned"
      and [(r["rung"], r.get("result")) for r in m["rungs_used"]
           if isinstance(r.get("rung"), int)][:1] == [(0, "arrived")]
      and attempts and attempts[0][1] == (0.0, 0.0)
      and not any(o[0] == "return" for o in w.sim.orders),
      f"{m['rungs_used']} {attempts[:1]} {w.sim.orders[-4:]}")

print("--- she cannot be found on the old map: never a guess ---")
w = World(["save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 45)
w.mgr._checkpoint(w.clock.monotonic())
old_id = w.mgr.mission["id"]
w.power_cut(rest_of_path=sw.THREE_ROOM_TOUR[9:])
w.store.map_file("home").unlink()                       # the saved map did not survive
booted(w)
w.mgr.resume()
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=30)
check("'continue' with no saved mission map: a NEW mission from where she stands, "
      "not the old frame on a stale pose", w.mgr.mission["id"] != old_id
      and w.mgr.frame_offset is None and w.voice.said[-1].startswith(
          "Starting exploration."), str(w.voice.said[-2:]))
check("...the interrupted mission and its area stay saved",
      w.store.mission(old_id)["status"] == "interrupted"
      and w.store.by_name("home", "hall") is not None)
w = World(["save it as hall", "kitchen"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 45)
w.mgr._checkpoint(w.clock.monotonic())
w.power_cut()
w.a.slam.localise_on = lambda og, scan, near=None, radius=None: None   # not sure
booted(w)
w.mgr.finish()                                          # "go back"
w.run(lambda x: x.t > x.states[-1][0] + 3, max_s=10)
check("'go back' when she cannot be placed in the old frame: refused, and the "
      "mission is still waiting", w.mgr.state == "IDLE" and w.mgr.interrupted
      and w.voice.said[-1] == "I can't do that right now." and not w.orders(*MOTION))

print("--- 'explore' and 'go back' instead of 'continue' ---")
w = World(["save it as hall", "kitchen", "study"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 40)
w.mgr._checkpoint(w.clock.monotonic())
old_id = w.mgr.mission["id"]
w.power_cut(rest_of_path=sw.THREE_ROOM_TOUR[9:])
booted(w)
router = intents.Router(w.mgr, lambda n: w.store.by_name("home", n),
                        w.a.places.find_exact, monotonic=w.clock.monotonic)
router.handle("explore", w.mgr.context())
check("'explore' with one waiting: 'Continue the last exploration or start a new "
      "one?'", w.voice.said[-1] == "Continue the last exploration or start a new one?")
router.handle("start a new one", w.mgr.context())
w.run(lambda x: x.mgr.state == "EXPLORE", max_s=30)
check("...'start a new one': a NEW mission; the old one stays saved as interrupted",
      w.mgr.mission["id"] != old_id
      and w.store.mission(old_id)["status"] == "interrupted"
      and not w.mgr.interrupted and w.mgr.frame_offset is None)
check("...and its area is still there", w.store.by_name("home", "hall") is not None)

w = World(["save it as hall", "kitchen"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 40)
w.mgr._checkpoint(w.clock.monotonic())
far = math.hypot(*w.mgr.pose[:2])
w.power_cut()
booted(w)
w.mgr.finish()                                          # "Friday, go back"
w.run(lambda x: x.mgr.state == "HELP_WAIT", max_s=20)
check("'go back' after a power cut: no run is started; with no engine she says "
      "the distance to the mission's start and waits",
      w.mgr.state == "HELP_WAIT" and not w.orders("start")
      and any(s.startswith("I'm stuck about ") and abs(float(
          s.split("about ")[1].split(" ")[0]) - far) < 0.4 for s in w.voice.said),
      f"{w.voice.said[-2:]} (far {far:.1f})")
w = World(["save it as hall", "kitchen"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 40)
w.mgr._checkpoint(w.clock.monotonic())
w.power_cut()
w.a.nav = ad.FakeNav(during_run=False, clock=w.clock, script=["ARRIVED"])
w.mgr.a.nav = w.a.nav
booted(w)
w.mgr.finish()
end = w.run(max_s=60)
check("...with File 14 wired, 'go back' drives to the interrupted mission's start",
      end == "IDLE" and w.mission()["status"] == "returned"
      and w.a.nav.calls[0][:2] == ("attempt", (0.0, 0.0)), str(w.a.nav.calls))

print("--- cut while she was already returning, and with a label pending ---")
w = World(["save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "RETURN_TO_START")
w.run(lambda x: x.t > x.states[-1][0] + 5)
w.power_cut()
booted(w)
check("interrupted mid-return: boot still says so",
      w.mgr.interrupted and "interrupted" in w.voice.said[-1])
w.mgr.resume()
w.run(lambda x: x.mgr.state in ("HELP_WAIT", "IDLE"), max_s=30)
check("...'continue' resumes the RETURN, not the exploring: no new run",
      not w.orders("start") and w.mgr.state == "HELP_WAIT")

w = World(["Friday stop", "save it as hall", "kitchen", "bedroom"])
w.mgr.start()
w.run(lambda x: x.mgr.state == "PAUSED")                # stopped mid-question
w.power_cut(rest_of_path=sw.THREE_ROOM_TOUR[3:])
booted(w)
check("cut while paused with a label pending: nothing was saved, boot says 0 areas",
      w.areas() == [] and "I saved 0 areas." in w.voice.said[-1])
w.mgr.resume()
w.run(lambda x: len(x.areas()) == 1, max_s=60)
check("...'continue': the pending label is asked again first thing, and saved",
      [a["name"] for a in w.areas()] == ["hall"], str(w.voice.said[-4:]))


# =============================================================================
print("--- map reset, corrupt checkpoint, first boot ever ---")

w = saved_map_world(distinct=True)                      # areas end up on the saved map
w.mgr.start()
w.run()
w.power_cut()
w.a.slam.map_id = lambda read_only=False: "map-REBUILT"
n_said = len(w.voice.said)
booted(w)
check("the map id changed while she was off: her saved-map areas are stale, "
      "never deleted", [a["status"] for a in w.areas()] == ["stale"] * 3
      and w.store.by_name("home", "kitchen") is not None)
check("...and she says it once: 'My map was reset ...'",
      w.voice.said[n_said:] == ["My map was reset, so I need to re-explore my "
                                "saved areas."], str(w.voice.said[n_said:]))
w.power_cut()
w.a.slam.map_id = lambda read_only=False: "map-REBUILT"
n_said = len(w.voice.said)
booted(w)
check("...the next boot does not say it again", w.voice.said[n_said:] == [])

w = World(["save it as hall", "kitchen"])
w.mgr.start()
w.run(lambda x: len(x.areas()) == 1 and x.t > 40)
w.mgr._checkpoint(w.clock.monotonic())
w.mgr._checkpoint(w.clock.monotonic() + 1)              # so a .bak exists
w.power_cut()
cpf = w.store.checkpoint_file("home")
cpf.write_text('{"mission_id": "abc", "sta')             # torn by the power cut
booted(w)
check("a checkpoint torn by the power cut: the .bak is loaded and the "
      "interrupted mission is still found", w.mgr.interrupted
      and "interrupted" in w.voice.said[-1])
w.power_cut()
cpf.write_text("garbage")
durable.bak_path(cpf).write_text("garbage")
n_said = len(w.voice.said)
booted(w)
check("both copies corrupt: she boots all the same, to IDLE, saying nothing",
      w.mgr.state == "IDLE" and w.mgr.booted and not w.mgr.interrupted
      and w.voice.said[n_said:] == [])
durable.atomic_write_json(cpf, {"schema_version": 99, "state": "EXPLORE"})
w.power_cut()
booted(w)
check("a checkpoint from a newer version is ignored, not a crash",
      w.mgr.state == "IDLE" and not w.mgr.interrupted)

w = World([], world=ROOM, path=ROOM_PATH)
booted(w)
check("the first boot ever (nothing saved): IDLE, not placed, nothing said, "
      "nothing ordered", w.mgr.state == "IDLE" and w.mgr.placed is None
      and w.voice.said == [] and not w.orders(*MOTION) and w.mgr.localized)

w = World([], world=ROOM, path=ROOM_PATH, sweeps=False,
          cfg={"boot": {"slam_wait_s": 3}})
booted(w)
check("the LiDAR never comes up within slam_wait_s: booted, but not localized",
      w.mgr.state == "IDLE" and w.mgr.booted and not w.mgr.localized)
router = intents.Router(w.mgr, lambda n: None, monotonic=w.clock.monotonic)
router.handle("explore", w.mgr.context())
check("...'explore' is refused: 'I'm not sure where I am. Please place me at my "
      "home spot.'", w.voice.said[-1]
      == "I'm not sure where I am. Please place me at my home spot."
      and not w.orders("start"))
w.a.lidar = w.sim.lidar()                               # the LiDAR comes up
w.mgr.a.lidar = w.a.lidar
w.run(lambda x: x.mgr.localized, max_s=cfg.boot.home_check_period_s + 10)
check("...once the LiDAR is up, the periodic look notices and she can start",
      w.mgr.localized)


# =============================================================================
print("--- placed late: the whole mission map fitted onto the saved map ---")


w = saved_map_world(distinct=True)
w.mgr.start()
w.run(lambda x: x.mgr.state == "EXPLORE" and len(x.areas()) == 1)
check("not placed at the start: the first area is saved in the mission's own "
      "frame, not a landmark yet", w.areas()[0]["frame"].startswith("mission:")
      and not w.a.places.all())
end = w.run()
areas = {a["name"]: a for a in w.areas()}
check("at the END the whole mission map fits the saved map: every area is moved "
      "into the saved map's frame", end == "IDLE" and len(areas) == 3
      and all(a["frame"] == "map" for a in areas.values()),
      str({n: a["frame"] for n, a in areas.items()}))
ROOMS_X = {"hall": (1.0, 4.5), "kitchen": (4.5, 7.0), "bedroom": (7.0, 11.5)}
where = {n: (round(areas[n]["center_x"], 2), round(areas[n]["center_y"], 2))
         for n in ROOMS_X if n in areas}
check("...at the right places on it: each centre lies inside its real room",
      len(where) == 3 and all(ROOMS_X[n][0] < x < ROOMS_X[n][1] and 1.0 < y < 5.0
                              for n, (x, y) in where.items()), str(where))
check("...and they are File 14 landmarks now: 'go to the kitchen' has somewhere "
      "to go", sorted(p["name"] for p in w.a.places.all())
      == ["bedroom", "hall", "kitchen"])
ha_row = w.store.home_anchor("home")
check("...the home anchor moved to the saved map's frame too, at the real start",
      ha_row["frame"] == "map" and math.hypot(ha_row["x"] - 2.5, ha_row["y"] - 3.0) < 0.4,
      str((ha_row["frame"], ha_row["x"], ha_row["y"])))
check("...logged on the mission", any(e["kind"] == "anchored_late" and e["areas"] == 3
                                      for e in w.mission()["events"]))
w = saved_map_world(distinct=False)         # three identical rooms
w.mgr.start()
end = w.run()
check("a house of identical rooms (it fits two ways): the fit REFUSES, and the "
      "areas stay in the mission's frame -- named, kept, not landmarks",
      end == "IDLE" and len(w.areas()) == 3
      and all(a["frame"].startswith("mission:") for a in w.areas())
      and not w.a.places.all(), str([a["frame"][:8] for a in w.areas()]))


# =============================================================================
print("--- File 14's search on the real saved map (read only) ---")

real_slam = ad.SlamAdapter()
og = real_slam.load()
if og is None:
    check("real map present", False, "state/map.npz missing")
else:
    rg = gu.from_occupancy_grid(og)
    trav = np.argwhere(gu.traversable(rg, config.NAV_INFLATE_M))
    rng = np.random.default_rng(7)
    picks = trav[rng.choice(len(trav), size=6, replace=False)]
    wide, near_ok, crashed, wrong, near_wrong = 0, 0, 0, 0, 0
    for ix, iy in picks:
        px, py = rg.cell_to_world(int(ix), int(iy))
        scan = synth.grid_ray_cast(rg.occupied(), rg.origin, rg.resolution,
                                   (px, py, 0.7))
        try:
            g = real_slam.localise(scan)
            n = real_slam.localise_on(og, scan, (px, py), 1.5)
        except Exception as exc:                        # noqa: BLE001
            crashed += 1
            print(f"        {type(exc).__name__}: {exc}")
            continue
        right_g = g is not None and math.hypot(g[0] - px, g[1] - py) < 0.3
        right_n = (n is not None and math.hypot(n[0] - px, n[1] - py) < 0.3
                   and abs((n[2] - 0.7 + math.pi) % (2 * math.pi) - math.pi) < 0.2)
        wide += int(right_g)
        near_ok += int(right_n)
        wrong += int(g is not None and not right_g)
        near_wrong += int(n is not None and not right_n)
    print(f"        6 real spots: whole-map search sure and right at {wide}; "
          f"searched within 1.5 m of the truth, sure and right at {near_ok}; "
          f"sure and WRONG: {wrong}")
    check("the REAL adapter runs File 14's search without error (the test doubles "
          "had hidden a crash here)", crashed == 0)
    print("        (the whole-map search's sure-and-wrong answers are File 14's own; "
          "it is why the mission never places its start from one sweep)")
    check("restricted to 1.5 m round her last pose (what resuming uses) it is "
          "NEVER wrong: it answers right or refuses", near_wrong == 0, str(near_wrong))
    check("...and it is never worse than the whole-map search", near_ok >= wide,
          f"{near_ok} vs {wide}")


# =============================================================================
print("--- the launcher: a thread in friday-mcp, shipped OFF ---")

# Harsh turned the service ON on 2026-10-02; these check the switch itself,
# whichever way the file has it, by setting it here.
check("the switch is a plain bool in config.py",
      isinstance(config.EXPLORATION_MISSION_SERVICE, bool))
_was = config.EXPLORATION_MISSION_SERVICE
config.EXPLORATION_MISSION_SERVICE = False
check("switched off, start() does nothing at all",
      service.enabled() is False and service.start() is None and service.get() is None)
config.EXPLORATION_MISSION_SERVICE = _was
w = World([], world=ROOM, path=ROOM_PATH)
w.clock = ad.Clock()                                    # real time for the thread
w.a.clock = w.clock
w.sim.clock = w.clock
mgr = service.start(adapters=w.a, store=w.store, cfg=w.cfg, force=True)
deadline = time.time() + 30
while time.time() < deadline and not (mgr and mgr.booted):
    time.sleep(0.1)
check("switched on (fakes): the mission thread starts, boots and idles",
      mgr is not None and mgr.booted and mgr.state == "IDLE"
      and service.get() is mgr and any(t.name == "friday-mission"
                                       for t in threading.enumerate()))
check("...a second start() is the same manager, not a second thread",
      service.start(adapters=w.a, store=w.store, cfg=w.cfg, force=True) is mgr
      and sum(t.name == "friday-mission" for t in threading.enumerate()) == 1)
service.stop()
check("...and stop() ends it", service.get() is None
      and not any(t.name == "friday-mission" for t in threading.enumerate()))


class Boom:
    def __getattr__(self, name):
        raise RuntimeError("adapters exploded")


check("a mission that cannot start costs the mission, never the server",
      service.start(adapters=Boom(), store=Boom(), cfg=w.cfg, force=True) is None)
server = (ROOT / "mcp_server" / "server.py").read_text()
check("mcp_server/server.py starts it beside the other watchers, guarded",
      "mission_service.start()" in server
      and server.index("environment_watch.start()") < server.index(
          "mission_service.start()"))
diff = subprocess.run(["git", "diff", "-U0", "HEAD", "--", "mcp_server/server.py",
                       "rl", "slam", "exploration_mode.py",
                       ":(exclude)rl/modes/*"],     # mode config, his OK
                      capture_output=True, text=True, cwd=str(ROOT)).stdout
removed = [ln for ln in diff.splitlines()
           if ln.startswith("-") and not ln.startswith("---")]
check("existing code: the launcher block only ADDS lines; rl/, slam/ untouched "
      "by this step", not removed and "rl/" not in diff and "slam/" not in diff,
      "\n".join(removed))
real_db = Path(config.MEMORY_DB)
import sqlite3                                          # noqa: E402
# Since the switch went on (2026-10-02) the REAL mission created its tables --
# after a sqlite backup(). What matters here: that backup exists, and these
# tests never wrote to that database (every World uses a temp one).
backups = sorted(real_db.parent.glob(real_db.name + ".bak-*-file21"))
has_tables = sqlite3.connect(f"file:{real_db}?mode=ro", uri=True).execute(
    "SELECT 1 FROM sqlite_master WHERE name='missions'").fetchone()
check("the real memory.db got its exploration tables only after a backup",
      (not has_tables) or bool(backups), str(backups))
check("these tests never opened the real memory.db for writing",
      all(Path(s).resolve() != real_db.resolve()
          for s in (World([]).store.db_path,)))
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
