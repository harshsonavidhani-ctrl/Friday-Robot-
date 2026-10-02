#!/usr/bin/env python3
"""Home saved at boot, and exploring ends at home. No motors, no LiDAR.

    ./venv/bin/python scripts/test_home_dock.py

Places go to a temp memory.db, the navigation bundle is not written, and the
journey that would drive home is replaced by a recorder -- nothing here can
move the robot.
"""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-home-"))
config.MEMORY_DB = TMP / "memory.db"

import home_dock                                        # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
import offline                                          # noqa: E402

nsm._bundle = lambda *a, **k: None                      # never touch the real bundle

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def clear_home() -> None:
    from slam import places
    places.forget("home")


# ---------------------------------------------------------------------------
print("\nhome is saved at boot — once, and only from a sure fix")
clear_home()
why = home_dock.save_boot_home({"state": "matched", "sure": False,
                                "pose": [1.0, 2.0, 0.5],
                                "why": "fits, two look-alike spots"})
check("an AMBIGUOUS boot fix never saves home",
      nsm.home_place() is None and "not certain" in why, why)
why = home_dock.save_boot_home({"state": "unknown", "why": "no LiDAR sweep"})
check("no sweep: no home", nsm.home_place() is None, why)
why = home_dock.save_boot_home({"state": "matched", "sure": True,
                                "pose": [1.25, -0.5, math.radians(30)]})
row = nsm.home_place()
check("a sure boot fix saves home at the boot pose",
      row is not None and abs(row["x"] - 1.25) < 1e-6
      and abs(row["y"] + 0.5) < 1e-6, why)
check("...with its heading kept",
      abs(home_dock.home_heading() - math.radians(30)) < 1e-3)
why = home_dock.save_boot_home({"state": "matched", "sure": True,
                                "pose": [3.0, 3.0, 0.0]})
row = nsm.home_place()
check("an existing home is PERMANENT: the next boot does not move it",
      abs(row["x"] - 1.25) < 1e-6 and "kept" in why, why)
clear_home()
why = home_dock.save_boot_home({"state": "no_map"})
row = nsm.home_place()
check("a brand-new map: home is its origin (0, 0, 0)",
      row is not None and row["x"] == 0.0 and row["y"] == 0.0
      and home_dock.home_heading() == 0.0, why)
config.HOME_SAVE_AT_BOOT = False
clear_home()
check("HOME_SAVE_AT_BOOT = False: nothing saved",
      home_dock.save_boot_home({"state": "no_map"}) == "off"
      and nsm.home_place() is None)
config.HOME_SAVE_AT_BOOT = True
home_dock.save_boot_home({"state": "no_map"})

# ---------------------------------------------------------------------------
print("\nexploring ends at home — only when the run ended on its own")
for status, tail, want, label in (
        ({"state": "completed", "exit_code": 0}, "I am back where I started.",
         True, "timer ran out, came back"),
        ({"state": "finished"}, "", True, "finished (nowhere left to explore)"),
        ({"state": "stopped", "stopped_because": "stopped by hand"}, "",
         False, "a person stopped it"),
        ({"state": "completed", "exit_code": 0}, "... stopped by hand", False,
         "the log says stopped by hand"),
        ({"state": "completed", "exit_code": 1}, "", False, "the run failed"),
        ({"state": "interrupted"}, "", False, "cut off by a power loss")):
    go, why = home_dock.wants_home_after(status, tail)
    check(f"{label}: {'go home' if want else 'stay'}", go is want, why)
clear_home()
go, why = home_dock.wants_home_after({"state": "completed", "exit_code": 0})
check("no home saved: stay", go is False and "no home" in why, why)
home_dock.save_boot_home({"state": "no_map"})
config.EXPLORE_ENDS_AT_HOME = False
check("EXPLORE_ENDS_AT_HOME = False: stay",
      home_dock.wants_home_after({"state": "completed"})[0] is False)
config.EXPLORE_ENDS_AT_HOME = True

# ---------------------------------------------------------------------------
print("\nthe hand-over tool: every guard, then one ordinary journey")
from mcp_server import hardware                         # noqa: E402
from mcp_server.tools import autonomy, nav_tools        # noqa: E402

started = []
nav_tools._start_journey = lambda label, coro: (started.append(label),
                                                coro.close())
nav_tools._reclaim_if_run_gone = lambda: None
nav_tools._controller = lambda: (object(), None)
autonomy._running = lambda: False
nav_tools._journey_running = lambda: ""
hardware.ABORT.clear()


def call() -> str:
    return asyncio.run(nav_tools.return_home_after_explore())


said = call()
check("all clear: one journey called 'home' starts", started == ["home"], said)
started.clear()
hardware.ABORT.set()
check("a person said stop: no journey", "stop" in call() and not started)
hardware.ABORT.clear()
autonomy._running = lambda: True
check("a run still holds the robot: no journey",
      "run" in call() and not started)
autonomy._running = lambda: False
nav_tools._journey_running = lambda: "the kitchen"
check("another journey is driving: no journey",
      "busy" in call() and not started)
nav_tools._journey_running = lambda: ""
clear_home()
check("no home: no journey", "no home" in call() and not started)

# ---------------------------------------------------------------------------
print("\nafter the run: wait for the LiDAR (live 21:32: 31 usable beams)")
config.HOME_AFTER_EXPLORE_SETTLE_S = 0.0
home_dock.save_boot_home({"state": "no_map"})
real_sleep = asyncio.sleep
nav_tools.asyncio_sleep_calls = []


class FakeMapper:
    def __init__(self, x, y):
        self.pose = (x, y, 0.0)


class FakeController:
    def __init__(self):
        self.got = "not called"

    async def go_home(self, mapper=None, **kw):
        self.got = mapper
        return nsm.ReturnReport(arrived=False, why="fake")


answers = [(None, "my LiDAR did not answer, so I cannot find myself."),
           (None, "only 31 usable beams in that sweep"),
           (FakeMapper(2.0, 0.0), "")]
nsm.relocalise = lambda allow_moves=True, prior=None: _pop()


async def _pop():
    return answers.pop(0)


orig_sleep = asyncio.sleep
asyncio.sleep = lambda s: orig_sleep(0)
said_lines = []
nav_voice_say = nav_tools.nav_voice.VOICE.say_text
nav_tools.nav_voice.VOICE.say_text = lambda t: said_lines.append(t)
nav_tools.nav_voice.VOICE.say = lambda *a: said_lines.append(a[0])
ctl = FakeController()
asyncio.run(nav_tools._home_after_explore(ctl))
check("two bad sweeps, then a good one: it waited and drove",
      isinstance(ctl.got, FakeMapper) and not answers, str(ctl.got))
check("...handing the fix to the engine (no second relocalisation)",
      ctl.got.pose == (2.0, 0.0, 0.0))
answers[:] = [(None, "only 31 usable beams in that sweep")] * 5
ctl = FakeController()
said_lines.clear()
report = asyncio.run(nav_tools._home_after_explore(ctl))
check("the LiDAR never recovers: she stays and SAYS so, no drive, no FAILSAFE",
      ctl.got == "not called" and not report.arrived
      and any("LiDAR" in l for l in said_lines), str(said_lines))
answers[:] = [(FakeMapper(0.1, 0.1), "")]
ctl = FakeController()
said_lines.clear()
home_dock.home_heading = lambda: None
report = asyncio.run(nav_tools._home_after_explore(ctl))
check("already within HOME_ARRIVE_M: 'I'm home', no drive",
      ctl.got == "not called" and report.arrived and "I'm home." in said_lines)
asyncio.sleep = orig_sleep

# ---------------------------------------------------------------------------
print("\nlook-alikes: the one near where the run started wins (live 21:39)")


class Fix:
    def __init__(self, pose, runner_up, candidate=None):
        self.pose, self.runner_up, self.candidate = pose, runner_up, candidate


live = Fix((-1.125, 1.075, -0.384), (-0.225, 0.775, 2.862))
start = (-0.175, 0.675, 0.6)
check("21:39's fix: the runner-up 0.11 m from the start is chosen",
      nsm._near_prior(live, start) == (-0.225, 0.775, 2.862))
check("no prior: the top score, exactly as before",
      nsm._near_prior(live, None) is None)
check("neither near the start: no override",
      nsm._near_prior(Fix((3, 3, 0), (4, 4, 0)), start) is None)
check("the top score IS near: kept",
      nsm._near_prior(Fix((-0.2, 0.7, 0.1), (3, 3, 0)), start) == (-0.2, 0.7, 0.1))
check("an ambiguous fix (no pose) still offers its candidates",
      nsm._near_prior(Fix(None, (5, 5, 0), candidate=(-0.1, 0.6, 1.0)), start)
      == (-0.1, 0.6, 1.0))
start_file = TMP / "run_start_fix.json"
nav_tools._run_start_file = lambda: start_file
import time as _t                                      # noqa: E402
start_file.write_text(json.dumps({"pose": [-0.175, 0.675, 0.2], "at": _t.time()}))
check("way-back arrived + a fresh start note -> a prior",
      nav_tools._run_start_prior(True) == (-0.175, 0.675, config.HOME_START_PRIOR_M))
check("way-back did NOT arrive -> no prior (she may be anywhere)",
      nav_tools._run_start_prior(False) is None)
start_file.write_text(json.dumps({"pose": [0, 0, 0], "at": _t.time() - 5 * 3600}))
check("a start note hours old -> no prior", nav_tools._run_start_prior(True) is None)

# ---------------------------------------------------------------------------
print("\nre-planning: the same thing, closer each time, is an obstacle (21:52)")
why = ("blocked: forward STOPPED after 0.2s of 0.6s — something is 109 mm "
       "away — inside the 200 mm bump guard (LiDAR)")
check("the distance is read from the block message", nsm._mm_in(why) == 109.0)
spots = [(0.10, 0.50, 190.0)]
check("190 then 109 mm at the same spot: an obstacle",
      nsm.repeat_block(spots, (0.12, 0.52), 109.0))
check("a block somewhere else: not (yet)",
      not nsm.repeat_block(spots, (1.0, 1.0), 100.0))
check("same spot but FARTHER (a person moving off): not",
      not nsm.repeat_block(spots, (0.1, 0.5), 400.0))
check("no distance in the message: not (the old walk-past rule)",
      not nsm.repeat_block(spots, (0.1, 0.5), None))

# ---------------------------------------------------------------------------
print("\nfacing the saved heading on arrival")
config.AGENCY_TURN_DEG_PER_S = 75.0
turn = home_dock.heading_turn(0.0, math.radians(90))
check("90° to the left: turn left ~1.2 s", turn[0] == "left"
      and abs(turn[1] - 1.2) < 0.01, str(turn))
turn = home_dock.heading_turn(0.0, math.radians(-60))
check("60° to the right: turn right ~0.8 s", turn[0] == "right"
      and abs(turn[1] - 0.8) < 0.01, str(turn))
check("within 25°: no turn", home_dock.heading_turn(0.0, math.radians(20)) is None)
turn = home_dock.heading_turn(math.radians(170), math.radians(-170))
check("wraps the short way round (170° -> -170° is 20°)", turn is None, str(turn))
turn = home_dock.heading_turn(0.0, math.radians(179))
check("a half-turn is one pivot, capped", turn[1] <= config.HOME_FACE_MAX_S)

# ---------------------------------------------------------------------------
print("\nvoice: dock is home")
for said in ("go home", "Friday, go to your dock", "return to the dock",
             "dock yourself", "ghar chalo"):
    intent = offline.understand(said, strict=True)
    check(f"{said!r} -> nav_home", intent is not None
          and intent.name == "nav_home", str(intent))

# ---------------------------------------------------------------------------
print("\nthe hard rule: exploration itself is unchanged")
diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--stat", "HEAD", "--",
                       "rl", "slam", "exploration_mode.py",
                       "mcp_server/tools/explore.py",
                       # mode CONFIG, changed with Harsh's OK (2026-10-02:
                       # runs keep friday-brain up for File 21); rl/ code stays
                       ":(exclude)rl/modes/*"],
                      capture_output=True, text=True).stdout.strip()
check("rl/, slam/, exploration_mode.py, explore.py: no changes", diff == "",
      diff)

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
