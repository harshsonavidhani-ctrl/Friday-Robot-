#!/usr/bin/env python3
"""File 14 step 13 — the acceptance run. Eight scenarios, end to end.

    scripts/test_file14_acceptance.py

This is NOT a replacement for the three unit suites, which test the parts.
This walks the eight things step 13 asks for, in order, as journeys: timer
expires -> route -> drive -> arrive -> stop, and so on. A part can be correct
while the journey is broken, and the pause/resume item is exactly that case —
every piece of it passed its own test while `pause_exploration` was reaching
for a `.__wrapped__` that does not exist and silently not pausing anything.

WHAT IS SIMULATED AND WHAT IS NOT. The chassis and the scan matcher are
simulated (see scripts/test_nav_follower.py for why). Everything else is
real: the real planner on a real grid, the real state machine, the real MCP
tool functions, the real `slam/places.py` against a temporary database, the
real git working tree for the diff check, and the real running dashboard for
the panel check.

⚠ IT WRITES NOTHING REAL. Places go to a temporary sqlite file and the
attempt log to a temporary directory — the first version of the suite wrote
forty imaginary journeys onto Harsh's Navigation panel before anyone noticed.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                           # noqa: E402
import nav_follower                                     # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
from test_nav_follower import FakeMapper, FakeRobot, FakeWorld   # noqa: E402
from test_nav_home import BlockingRobot, TempPlaces, _grid       # noqa: E402

_LOG_DIR = tempfile.TemporaryDirectory()
nsm.NAV_LOG = Path(_LOG_DIR.name) / "nav_log.jsonl"

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0
_items: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    global _failures
    if not ok:
        _failures += 1
    print(f"    [{PASS if ok else FAIL}] {label}"
          f"{'  — ' + detail if detail else ''}")
    return ok


def item(n: int, title: str) -> None:
    print(f"\n{n}. {title}")


def _arrival(goal):
    """Arrival with no wall-clock dwell — the sim runs ticks in microseconds."""
    return nav_follower.ArrivalMonitor(goal, cell_m=0.05, dwell_s=0.0,
                                       min_samples=2)


# --------------------------------------------------------------------------

def one_return_to_home() -> None:
    item(1, "Return-to-Home: timer expiry → path → drive → arrival → stop")
    with TempPlaces():
        nsm.set_home((2.0, 2.0, 0.0))
        world = FakeWorld(pose=(0.5, 2.0, 0.0))
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)
        robot = FakeRobot(world)
        c = nsm.NavController(robot=robot, minutes=5.0)

        c.exploring(now=0.0)
        check("it is exploring, clock running",
              c.state is nsm.NavState.EXPLORING and c.timer.running)
        check("before expiry it stays put",
              c.tick(now=60.0) is nsm.NavState.EXPLORING)
        check("the timer expiring flips it to RETURN_TO_HOME",
              c.tick(now=300.0) is nsm.NavState.RETURN_TO_HOME)

        report = asyncio.run(c.go_home(mapper=mapper, max_ticks=80,
                                       arrival=_arrival((2.0, 2.0))))
        check("a path was computed", report.waypoints >= 2,
              f"{report.waypoints} waypoints over {report.planned_m:.2f} m")
        check("it drove", (report.follow or {}).get("driven_m", 0) > 0,
              f"{report.follow.get('driven_m')} m driven")
        check("it arrived", report.arrived, report.why)
        check("state is ARRIVED", report.state is nsm.NavState.ARRIVED)
        check("and it really is home",
              np.hypot(world.x - 2.0, world.y - 2.0) < 0.5,
              f"{np.hypot(world.x - 2.0, world.y - 2.0):.2f} m from home")
        check("it stopped commanding once there",
              report.follow["ticks"] < 80, f"{report.follow['ticks']} ticks")
        _items.append(("Return-to-Home end to end", report.arrived))


def two_target_navigation() -> None:
    item(2, "Target Navigation: command → lookup → drive → arrival → action → log")
    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.0, 2.0, 0.0))
        world = FakeWorld(pose=(0.5, 2.0, 0.0))
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)
        c = nsm.NavController(robot=FakeRobot(world))

        # What the brain parses a voice command into: a name and an action.
        name, action = "the kitchen", "tell me what you can see"
        row, why = nsm.resolve_landmark(name)
        check("the landmark is found by name", row is not None, why)

        report = asyncio.run(c.go_to_landmark(
            name, action=action, mapper=mapper, max_ticks=80,
            arrival=_arrival((2.0, 2.0))))
        check("it arrived", report.arrived, report.why)
        check("the queued action came back for the brain to perform",
              report.action == action, report.action)

        rows = nsm.recent_attempts()
        check("the attempt was logged", bool(rows), str(rows[:1]))
        check("with target, outcome and time taken",
              rows and rows[0]["destination"] == "the kitchen"
              and rows[0]["arrived"] is True and rows[0]["seconds"] >= 0,
              str(rows[0]) if rows else "")
        check("the visit was recorded against the place",
              (places.find("the kitchen") or {}).get("visits", 0) >= 1)
        _items.append(("Target Navigation end to end",
                       report.arrived and bool(rows)))


def three_pause_and_resume() -> None:
    item(3, "Target-nav mid-exploration: pause, then RESUME (not restart)")
    from mcp_server.tools import autonomy

    calls: list[tuple] = []
    real_running, real_stop, real_start = (
        autonomy._running, autonomy.stop_exploration, autonomy.start_exploration)
    out = [True]
    autonomy._running = lambda: out[0]

    def fake_stop():
        calls.append(("stop",))
        out[0] = False
        return "stopped the run and took my wheels back"

    def fake_start(minutes=None, thinking=None):
        calls.append(("start", minutes))
        out[0] = True
        return f"exploring for {minutes} minutes"

    autonomy.stop_exploration, autonomy.start_exploration = fake_stop, fake_start
    try:
        with TempPlaces():
            from slam import places
            places.remember_here("the kitchen", (2.0, 2.0, 0.0))
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            c = nsm.NavController(robot=FakeRobot(world), minutes=20.0)
            c.exploring()                    # real clock: ~20 min remaining

            paused = c.pause_exploration()
            check("exploration was actually stopped",
                  ("stop",) in calls and nsm._stopped_ok(paused["stopped"]),
                  paused["stopped"])
            check("a snapshot was taken", c.snapshot is not None)
            check("it records the in-progress target as a hint",
                  "target_xy" in paused)
            check("and is honest that frontiers are recomputed",
                  "recomputed" in paused["note"])

            report = asyncio.run(c.go_to_landmark(
                "the kitchen", action="", mapper=mapper, max_ticks=80,
                arrival=_arrival((2.0, 2.0))))
            check("the target-nav run finished", report.arrived, report.why)

            resumed = c.resume_exploration()
            check("exploration was resumed", resumed["resumed"], str(resumed))
            starts = [c for c in calls if c[0] == "start"]
            check("it resumed for the REMAINING time, not a fresh 20 minutes",
                  starts and starts[0][1] is not None
                  and 19.0 < starts[0][1] < 20.0,
                  f"asked for {starts[0][1]:.2f} of 20 minutes"
                  if starts else "")
            check("the state machine is exploring again",
                  c.state is nsm.NavState.EXPLORING)
            check("the snapshot is consumed, not left to go stale",
                  c.snapshot is None)
            _items.append(("Pause and resume exploration",
                           resumed["resumed"] and bool(starts)))
    finally:
        autonomy._running = real_running
        autonomy.stop_exploration = real_stop
        autonomy.start_exploration = real_start


def four_transient_block() -> None:
    item(4, "Transient block: no replanning storm")
    saved = nsm.BLOCK_RECHECK_S
    nsm.BLOCK_RECHECK_S = 0.0
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            robot = BlockingRobot(world, clears_after=1)   # walks away
            c = nsm.NavController(robot=robot)
            report = asyncio.run(c.go_home(mapper=mapper, max_ticks=6,
                                           max_attempts=2))
            check("it rechecked before re-planning", robot.checks > 0,
                  f"{robot.checks} rechecks")
            check("it saw the block", report.blocks_seen > 0)
            check("recorded NO obstacle, because it cleared",
                  report.blocks_confirmed == 0 and len(c.obstacles) == 0)
            check("and did not replan more than the budget allows",
                  report.attempts <= 2, f"{report.attempts} attempts")
            _items.append(("Transient block, no storm",
                           report.blocks_confirmed == 0 and robot.checks > 0))
    finally:
        nsm.BLOCK_RECHECK_S = saved


def five_persistent_block() -> None:
    item(5, "Persistent block: remembered and planned around")
    saved = nsm.BLOCK_RECHECK_S
    nsm.BLOCK_RECHECK_S = 0.0
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            robot = BlockingRobot(world)                  # never clears
            c = nsm.NavController(robot=robot)
            report = asyncio.run(c.go_home(mapper=mapper, max_ticks=6,
                                           max_attempts=3))
            check("the block was confirmed", report.blocks_confirmed > 0,
                  f"{report.blocks_confirmed} confirmed")
            check("it went into the runtime layer", len(c.obstacles) > 0,
                  str(c.obstacles.as_dict()["points"][:2]))

            # The layer must actually change what the planner may use.
            import path_planner
            clear = path_planner.preprocess(mapper.grid)
            around = path_planner.preprocess(mapper.grid,
                                             extra_blocked=c.obstacles.points())
            check("and the next plan has to go around it",
                  around.plannable_cells < clear.plannable_cells,
                  f"{clear.plannable_cells} -> {around.plannable_cells} cells")
            check("nothing was written to the saved map",
                  not (config.ROOT / "state" / "map.npz.tmp").exists())
            _items.append(("Persistent block planned around",
                           report.blocks_confirmed > 0 and len(c.obstacles) > 0))
    finally:
        nsm.BLOCK_RECHECK_S = saved


def six_failsafe() -> None:
    item(6, "FAILSAFE when no path exists: halts, does not retry forever")
    with TempPlaces():
        nsm.set_home((1.0, 1.0, 0.0))
        world = FakeWorld()
        robot = BlockingRobot(world)

        class Unmapped:
            grid = _grid()                 # all unknown: nothing plannable
            pose = (0.2, 0.2, 0.0)

        c = nsm.NavController(robot=robot)
        report = asyncio.run(c.go_home(mapper=Unmapped(), max_attempts=3))
        check("it did not arrive", not report.arrived)
        check("state is FAILSAFE", report.state is nsm.NavState.FAILSAFE)
        check("it halted the wheels", robot.halted)
        check("it stopped at the replanning budget, not indefinitely",
              report.attempts == 3, f"{report.attempts} attempts")
        check("and it says what is actually wrong",
              "nowhere on my map I can plan" in report.why, report.why)
        check("the failure is logged for the dashboard",
              any(not r["arrived"] for r in nsm.recent_attempts()))
        _items.append(("FAILSAFE halts",
                       report.state is nsm.NavState.FAILSAFE and robot.halted))


def seven_exploration_untouched() -> None:
    item(7, "The exploration/SLAM files are unmodified")
    watched = ["slam/", "rl/", "exploration_mode.py",
               "mcp_server/tools/explore.py", "motor.py", "memory.py",
               "config.py", "mcp_server/safety.py"]

    # ⚠ "THE TREE IS CLEAN" IS THE WRONG QUESTION, and asking it gave a false
    # failure the first time this ran. Other sessions work in this repo and
    # leave their own uncommitted changes — `config.py` carries file 12's
    # INITIATIVE block right now. What must be proved is narrower and is the
    # thing the rule actually says: that FILE 14 did not modify these. So a
    # dirty file is inspected, and it only passes if its diff contains none of
    # this module's vocabulary.
    fingerprint = ("nav_", "landmark", "path_planner", "pursuit", "speck",
                   "failsafe", "return_home", "astar", "navigate_to",
                   "nav_follower", "nav_state_machine", "RuntimeObstacles")
    ok = True
    for path in watched:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", path],
            cwd=ROOT, capture_output=True, text=True).stdout.strip()
        tracked = [ln for ln in status.splitlines() if not ln.startswith("??")]
        if not tracked:
            print(f"      · {path}: unchanged")
            continue
        diff = subprocess.run(["git", "diff", "--", path], cwd=ROOT,
                              capture_output=True, text=True).stdout
        added = "\n".join(ln for ln in diff.splitlines()
                           if ln.startswith("+") and not ln.startswith("+++"))
        mine = [w for w in fingerprint if w.lower() in added.lower()]
        if mine:
            ok = False
            check(f"{path} was modified by file 14", False, f"found {mine}")
        else:
            lines = len([ln for ln in added.splitlines()])
            print(f"      · {path}: dirty, but NOT by file 14 "
                  f"({lines} added lines, none navigation-related)")

    check("no watched file carries a file-14 change", ok)

    # And the strongest form for the files that must be pristine full stop.
    must_be_clean = ["slam/", "rl/", "exploration_mode.py",
                     "mcp_server/tools/explore.py", "motor.py",
                     "mcp_server/safety.py"]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--"] + must_be_clean,
        cwd=ROOT, capture_output=True, text=True).stdout.strip()
    tracked = [ln for ln in dirty.splitlines() if not ln.startswith("??")]
    ok &= check("the exploration stack itself is byte-for-byte untouched",
                not tracked, "; ".join(tracked) or
                "slam/ rl/ exploration_mode.py explore.py motor.py safety.py")
    _items.append(("Exploration/SLAM untouched", ok))


def eight_dashboard_live() -> None:
    item(8, "Both dashboard panels show live, correct data")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = f"https://127.0.0.1:{config.DASHBOARD_PORT}/api/nav"
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=6) as r:
            data = json.loads(r.read().decode())
    except Exception as exc:                 # noqa: BLE001
        check(f"the dashboard answers on {url}", False, str(exc))
        _items.append(("Dashboard panels live", False))
        return

    ok = check("/api/nav answers", data.get("ok") is True)
    frame = data.get("frame", {})
    ok &= check("the Target Navigation panel has landmarks",
                bool(data.get("landmarks")),
                ", ".join(m["name"] for m in data.get("landmarks", [])))
    ok &= check("their positions are real map coordinates",
                all("u" in m for m in data["landmarks"]) if frame.get("ok")
                else True,
                f"survey map {frame.get('size_m')} m")
    ok &= check("it states which frame they are in, not the Map panel's",
                "survey map" in data.get("note", ""))
    ok &= check("the RTH panel reports the last return",
                "last_return" in data,
                str(data.get("last_return") or "none yet — she has not driven"))
    ok &= check("the attempt log is exposed", isinstance(data.get("attempts"), list),
                f"{len(data.get('attempts', []))} rows")

    with urllib.request.urlopen(
            f"https://127.0.0.1:{config.DASHBOARD_PORT}/", context=ctx,
            timeout=6) as r:
        page = r.read().decode()
    ok &= check("the panel is in the served page", 'id="panel-nav"' in page)
    ok &= check("the page still serves the older panels too",
                'id="panel-map"' in page and 'id="panel-explore"' in page)
    _items.append(("Dashboard panels live", bool(ok)))


def main() -> int:
    print("FILE 14 — STEP 13 ACCEPTANCE RUN")
    print("=" * 64)
    one_return_to_home()
    two_target_navigation()
    three_pause_and_resume()
    four_transient_block()
    five_persistent_block()
    six_failsafe()
    seven_exploration_untouched()
    eight_dashboard_live()

    print("\n" + "=" * 64)
    for name, ok in _items:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 64)
    print(f"{'ALL SCENARIOS PASSED' if not _failures else f'{_failures} CHECKS FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
