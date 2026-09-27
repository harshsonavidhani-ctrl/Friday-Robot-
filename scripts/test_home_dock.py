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
                       "mcp_server/tools/explore.py"],
                      capture_output=True, text=True).stdout.strip()
check("rl/, slam/, exploration_mode.py, explore.py: no changes", diff == "",
      diff)

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
