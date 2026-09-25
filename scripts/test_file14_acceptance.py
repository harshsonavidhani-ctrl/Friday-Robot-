#!/usr/bin/env python3
"""File 14 steps 14 and 18 — the acceptance run, end to end.

Scenarios 1-9 are the original step-13 journeys. 10-13 (2026-09-23) are the
updated spec's additions: an offline voice errand with no model anywhere,
stuck told apart from blocked, exploration ending early when its frontiers run
out, and FAILSAFE -> a person moves her -> "try again".

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
import re
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                           # noqa: E402
import nav_follower                                     # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
import math                                             # noqa: E402
from test_nav_follower import FakeMapper, FakeRobot, FakeWorld   # noqa: E402
from test_nav_home import BlockingRobot, TempPlaces, _grid       # noqa: E402

_LOG_DIR = tempfile.TemporaryDirectory()
nsm.NAV_LOG = Path(_LOG_DIR.name) / "nav_log.jsonl"
nsm.NAV_MODE_FILE = Path(_LOG_DIR.name) / "nav_mode.json"
nsm.PENDING_FILE = Path(_LOG_DIR.name) / "nav_round_trip.json"
nsm.FAILSAFE_FILE = Path(_LOG_DIR.name) / "nav_failsafe.json"
nsm.EARLY_RETURN_FILE = Path(_LOG_DIR.name) / "nav_early_return.json"
# ⚠ AN EXPLORE TRIP NOW GROWS THE SAVED MAP (map_growth.py). Every trip here
# must grow a scratch copy, never state/map.npz.
import shutil                                           # noqa: E402
_REAL_MAP = Path(config.SLAM_MAP_FILE)
config.SLAM_MAP_FILE = Path(_LOG_DIR.name) / "map.npz"
if _REAL_MAP.exists():
    shutil.copy2(_REAL_MAP, config.SLAM_MAP_FILE)

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
    # The ONE file-14 line allowed outside file 14's own files (2026-09-23):
    # the offline navigation tools join config.COMMAND_FRESH_TOOLS, so a
    # spoken "go to the kitchen" goes stale like every other move. It has to
    # live in config.py (one config file, CLAUDE.md PART 2). Only lines that
    # are exactly that are excused; anything else navigation-shaped still
    # fails.
    excused = {"config.py": re.compile(
        r'^\+\s*(?:# file 14\'s offline navigation orders.*|'
        r'(?:"(?:nav_go|nav_home|nav_come_back|try_again|'
        r'explore_and_return)",?\s*)+\)?)\s*$')}
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
        allow = excused.get(path)
        added = "\n".join(ln for ln in diff.splitlines()
                           if ln.startswith("+") and not ln.startswith("+++")
                           and not (allow and allow.match(ln)))
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


def nine_round_trip() -> None:
    item(9, 'Round trip: "go to the kitchen" → goes → comes back to the start')
    import nav_follower
    from mcp_server.tools import autonomy

    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.5, 2.0, 0.0))
        start = (0.5, 2.0, 0.0)
        world = FakeWorld(pose=start)
        mapper = FakeMapper(world)
        mapper.grid = _grid((160, 160), free=True)
        c = nsm.NavController(robot=FakeRobot(world))

        report = asyncio.run(c.go_to_landmark(
            "the kitchen", action="say you are there", mapper=mapper,
            return_to_start=True, max_ticks=200,
            arrival=_arrival((2.5, 2.0))))
        check("she reached the kitchen", report.arrived, report.why)
        check("the action still came back for the brain",
              report.action == "say you are there")
        check("she came back to the spot she set off from",
              report.returned_to_start,
              str(report.inbound.get("why", ""))[:70])
        back = math.hypot(world.x - start[0], world.y - start[1])
        check("and is really there", back < 0.5, f"{back:.2f} m from the start")
        check("both legs are in the log",
              len([r for r in nsm.recent_attempts()
                   if r["destination"] in ("the kitchen",
                                           "where I set off from")]) >= 2)

        # And the armed form, which is what "explore and come back" uses.
        nsm.arm_return_to_start(start, why="exploration")
        check("an exploration return can be armed",
              nsm.pending_return() is not None)
        world2 = FakeWorld(pose=(2.5, 2.0, 0.0))
        mapper2 = FakeMapper(world2)
        mapper2.grid = _grid((160, 160), free=True)
        c2 = nsm.NavController(robot=FakeRobot(world2))
        armed = asyncio.run(c2.return_to_armed_start(
            mapper=mapper2, max_ticks=200, arrival=_arrival(start[:2])))
        check("and firing it brings her back", armed.arrived, armed.why)
        check("the arming is consumed", nsm.pending_return() is None)
        _items.append(("Round trip there and back",
                       report.returned_to_start and armed.arrived))


# --------------------------------------------------------------------------
# 2026-09-23: offline navigation, stuck vs blocked, frontier exhaustion and
# the way out of FAILSAFE -- driven through the REAL tool functions, with the
# chassis, the scan matcher and the exploration process simulated.
# --------------------------------------------------------------------------

# A simulated chassis that drives as fast as the calibration says it does
# (state/calibration.json: 45.2 cm/s, 75 deg/s, less the sim's 30% loss).
# The older FakeWorld default is six times slower than the calibration, which
# the follower rightly reads as a stall -- fine for scenarios about stalls,
# wrong for ones that are not.
_CALIBRATED = {"cm_s": 45.0, "deg_s": 75.0}


class _SimController(nsm.NavController):
    """The real controller, with the sim's instant arrival test on every leg."""

    async def _drive_to(self, goal_xy, label, mode, mapper=None,
                        max_attempts=3, **kw):
        kw.setdefault("max_ticks", 80)
        kw.setdefault("arrival", _arrival(goal_xy))
        return await super()._drive_to(goal_xy, label, mode, mapper,
                                       max_attempts, **kw)


class _ToolRig:
    """Point nav_tools at a simulated robot, voice and exploration process."""

    def __init__(self, world, mapper, running=False):
        import nav_voice
        from mcp_server.tools import autonomy, nav_tools
        self.nav_voice, self.autonomy, self.nav_tools = (nav_voice, autonomy,
                                                         nav_tools)
        self.world, self.mapper = world, mapper
        self.voice = nav_voice.Voice(speak=lambda _t: None, background=False)
        self.running = running
        self.calls: list = []
        self.relocalised_at: list[int] = []

    def __enter__(self):
        a, nt = self.autonomy, self.nav_tools
        self.saved = (nt._controller, nsm.relocalise, self.nav_voice.VOICE,
                      a._running, a.stop_exploration, a.start_exploration,
                      nt._trip_start)
        robot = FakeRobot(self.world)

        async def halt():
            self.calls.append("halt")
            return "stopped"

        robot.halt = halt

        def controller():
            return _SimController(robot=robot, voice=self.voice), robot

        async def relocalise():
            self.relocalised_at.append(len(self.world.commands))
            return self.mapper, ""

        def stop():
            self.calls.append("stop_exploration")
            self.running = False
            return "stopped the run and took my wheels back"

        def start(minutes=None, thinking=None):
            self.calls.append(("start_exploration", minutes))
            self.running = True
            return f"exploring for {minutes} minutes"

        async def trip_start(_robot):
            # Through nsm.relocalise, looked up at call time, so a test that
            # swaps in a refusal still refuses the trip.
            mapper, why = await nsm.relocalise()
            return (None, why) if mapper is None else (_FakeTracker(mapper),
                                                       "")

        nt._controller = controller
        nt._trip_start = trip_start
        nsm.relocalise = relocalise
        self.nav_voice.VOICE = self.voice
        a._running = lambda: self.running
        a.stop_exploration, a.start_exploration = stop, start
        return self

    def __exit__(self, *exc):
        a, nt = self.autonomy, self.nav_tools
        (nt._controller, nsm.relocalise, self.nav_voice.VOICE,
         a._running, a.stop_exploration, a.start_exploration,
         nt._trip_start) = self.saved
        return False


class _FakeTracker:
    """trip_tracker.TripTracker's shape over a FakeMapper, sure of its start.

    The real tracker, on the real Mapper and the real map, is tested in
    scripts/test_trip_tracker.py; this rig tests the TRIP around it.
    """

    def __init__(self, mapper):
        self.mapper = mapper
        self.start = tuple(float(v) for v in mapper.pose)

    def explorer_mapper(self):
        return self.mapper

    def commanded(self, moved_m=0.0, turned_rad=0.0):
        pass

    def note_sweep(self, scan):
        pass

    async def drain(self):
        pass

    def close(self):
        pass

    def best(self):
        return self.mapper, self.start, True

    def as_dict(self):
        return {"fake": True}


def _line(key, dest=""):
    import nav_voice
    return nav_voice.line(key, dest)


def _in_order(said, lines) -> bool:
    """`lines` appear in `said` in this order (other lines may come between)."""
    it = iter(said)
    return all(any(s == line for s in it) for line in lines)


def ten_offline_voice_errand() -> None:
    item(10, "Offline: \"go to the kitchen and tell me the time and come "
             "back\" with no model at all")
    import offline
    from mcp_server.tools import nav_tools
    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.0, 2.0, 0.0))
        world = FakeWorld(pose=(0.5, 2.0, 0.0), **_CALIBRATED)
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)
        said = "Friday, go to the kitchen and tell me the time and come back"
        intent = offline.understand(said, strict=True)
        check("the order is understood locally, even with a brain available",
              intent is not None and intent.name == "nav_go"
              and intent.name in offline._REFLEX_INTENTS,
              str(intent and (intent.name, intent.args)))
        check("the place, the action and the way back are all parsed",
              intent.args == {"place": "the kitchen", "come_back": True,
                              "then": "tell me the time"}, str(intent.args))
        check("main.py will not read the tool's answer out a second time",
              not intent.speak_result and not intent.say)

        with _ToolRig(world, mapper) as rig:
            async def run():
                import time as _t
                t0 = _t.monotonic()
                reply = await nav_tools.nav_go(**intent.args)
                took = _t.monotonic() - t0
                await nav_tools._JOURNEY
                return reply, took

            reply, took = asyncio.run(run())
            check("the tool answers at once, inside the offline timeout",
                  took < 1.0 and reply == _line("setting_off", "the kitchen"),
                  f"{took:.2f} s: {reply!r}")
            spoken = rig.voice.said
            check("she set off, arrived, told the time THERE, then came back",
                  len(spoken) >= 4
                  and spoken[0] == _line("setting_off", "the kitchen")
                  and spoken[1] == _line("arrived", "the kitchen")
                  and spoken[2].lower().startswith(("it's", "it is"))
                  and spoken[3] == _line("arrived_start"), str(spoken))
            check("and really is back where she started",
                  math.hypot(world.x - 0.5, world.y - 2.0) < 0.5,
                  f"{math.hypot(world.x - 0.5, world.y - 2.0):.2f} m away")
            check("every sentence came from the fixed table or the clock",
                  all(s in _all_lines() or s.lower().startswith(("it's",
                                                                 "it is"))
                      for s in spoken), str(spoken))
        _items.append(("Offline voice errand, no model",
                       len(spoken) >= 4 and spoken[3] == _line("arrived_start")))


def _all_lines() -> set[str]:
    import nav_voice
    out = set()
    for key in nav_voice.LINES:
        for dest in ("home", "the kitchen", "the desk", "where I set off from",
                     ""):
            out.add(nav_voice.line(key, dest))
    return out


def eleven_stuck_is_not_blocked() -> None:
    item(11, "Stuck vs blocked: recovery before any re-plan, and told apart "
             "in the log and out loud")
    import nav_voice
    saved = nsm.BLOCK_RECHECK_S
    nsm.BLOCK_RECHECK_S = 0.0
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))

            class Wedged(FakeWorld):
                def drive(self, direction, seconds):
                    if direction == "backward":
                        self.dead = False
                    super().drive(direction, seconds)

            stuck_world = Wedged(pose=(0.5, 2.0, 0.0), cm_s=45.0, deg_s=75.0,
                                 dead=True, match_noise_m=0.01)
            stuck_map = FakeMapper(stuck_world)
            stuck_map.grid = _grid(free=True)
            stuck_voice = nav_voice.Voice(speak=lambda _t: None,
                                          background=False)
            stuck = asyncio.run(_SimController(
                robot=FakeRobot(stuck_world), voice=stuck_voice).go_home(
                mapper=stuck_map))
            stuck_row = nsm.recent_attempts(1)[0]

            block_world = FakeWorld(pose=(0.5, 2.0, 0.0))
            block_map = FakeMapper(block_world)
            block_map.grid = _grid(free=True)
            block_voice = nav_voice.Voice(speak=lambda _t: None,
                                          background=False)
            blocker = BlockingRobot(block_world, clears_after=1)
            asyncio.run(_SimController(robot=blocker, voice=block_voice)
                        .go_home(mapper=block_map, max_attempts=2))
            block_row = nsm.recent_attempts(1)[0]

            dirs = [d for d, _ in stuck_world.commands]
            check("stuck: the manoeuvre ran BEFORE the only re-plan",
                  "backward" in dirs and stuck.replans == 1
                  and stuck.recoveries == 1, f"{stuck.replans} re-plans, "
                  f"{stuck.recoveries} recoveries, {dirs[:10]}")
            check("stuck: and she got there", stuck.arrived, stuck.why)
            check("the log tells them apart",
                  stuck_row["trouble"] == "stuck" and stuck_row["stucks"] == 1
                  and block_row["trouble"] == "blocked"
                  and block_row["stucks"] == 0
                  and block_row["blocks_seen"] >= 1,
                  f"stuck={stuck_row['trouble']} blocked={block_row['trouble']}")
            check("and so does she",
                  _line("stuck") in stuck_voice.said
                  and _line("blocked") not in stuck_voice.said
                  and _line("blocked") in block_voice.said
                  and _line("stuck") not in block_voice.said,
                  f"{stuck_voice.said} / {block_voice.said}")
            check("a block was never answered with a reverse",
                  not blocker.world.commands
                  or "backward" not in [d for d, _ in blocker.world.commands])
            _items.append(("Stuck vs blocked",
                           stuck.arrived and stuck_row["trouble"] == "stuck"
                           and block_row["trouble"] == "blocked"))
    finally:
        nsm.BLOCK_RECHECK_S = saved


class _TripRig:
    """The explore trip's own pieces, simulated: the explorer and its robot.

    The real Mission is tested in scripts/test_mission.py; what is tested here
    is the TRIP around it -- mark, explore on one map, stop means stop, come
    back to the mark. The fake explorer moves the simulated chassis somewhere
    else and reports why it stopped, exactly the shape Mission.run returns.
    """

    def __init__(self, rig, end_pose, reason="time budget",
                 person_stopped=False):
        from mcp_server.tools import nav_tools
        self.nt, self.rig = nav_tools, rig
        self.end_pose, self.reason = end_pose, reason
        self.person_stopped = person_stopped
        self.ran = []

    def __enter__(self):
        nt, rig, trip = self.nt, self.rig, self

        class Explorer:
            def stopped(self):
                return trip.person_stopped

        class Mission:
            def __init__(self, robot):
                pass

            async def run(self, mapper, budget_m=None, budget_s=None):
                trip.ran.append((budget_m, budget_s))
                rig.world.x, rig.world.y, rig.world.theta = trip.end_pose
                mapper.update({0: 1500})
                return {"stopped_because": trip.reason, "driven_m": 2.4,
                        "floor_m2": 6.5}

        self.saved = (nt._mission, nt._explorer_robot)
        nt._mission = lambda robot: Mission(robot)
        nt._explorer_robot = lambda: Explorer()
        return self

    def __exit__(self, *exc):
        (self.nt._mission, self.nt._explorer_robot) = self.saved
        return False


def _trip(minutes=2.0):
    from mcp_server.tools import nav_tools

    async def go():
        said = await nav_tools.explore_and_return(minutes=minutes)
        result = await nav_tools._JOURNEY
        return said, result
    return asyncio.run(go())


def twelve_explore_trip() -> None:
    item(12, "\"Friday, explore\": mark the spot with the LiDAR, explore on "
             "ONE tracked map, come back to the mark")
    import offline
    with TempPlaces():
        world = FakeWorld(pose=(0.5, 2.0, 0.0), **_CALIBRATED)
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)
        intent = offline.understand("Friday, explore for 2 minutes",
                                    strict=True)
        check("plain \"explore\" is the trip that comes back, offline",
              intent is not None and intent.tool == "explore_and_return"
              and intent.name in offline._REFLEX_INTENTS
              and not intent.local, str(intent and (intent.name, intent.tool)))
        with _ToolRig(world, mapper) as rig, \
                _TripRig(rig, end_pose=(2.3, 1.2, 1.0)) as trip:
            said, back = _trip(2.0)
            check("she says she is exploring and will come back",
                  said == _line("exploring", "2 minutes"), said)
            check("she marked the spot BEFORE exploring",
                  rig.relocalised_at[:1] == [0], str(rig.relocalised_at))
            check("the explorer got a time limit from the command",
                  trip.ran and trip.ran[0][1] == 120.0, str(trip.ran))
            check("and she came back to the mark",
                  back is not None and back.arrived
                  and math.hypot(world.x - 0.5, world.y - 2.0) < 0.5,
                  f"{math.hypot(world.x - 0.5, world.y - 2.0):.2f} m away")
            check("without finding herself again: one map, tracked all along",
                  len(rig.relocalised_at) == 1, str(rig.relocalised_at))
            check("she said so",
                  _in_order(rig.voice.said,
                            [_line("setting_off", "where I set off from"),
                             _line("arrived_start")]), str(rig.voice.said))
            check("and the trip's map was kept: the saved map grew",
                  any(t.startswith("My map grew by") for t in rig.voice.said),
                  str(rig.voice.said[-1:]))
            home = nsm.home_place()
            check("with no home stored, the start spot became home (the "
                  "prompt's boot pose), and she said so",
                  home is not None
                  and math.hypot(home["x"] - 0.5, home["y"] - 2.0) < 0.3
                  and _line("home_set") in rig.voice.said, str(home))
        _items.append(("Explore trip comes back",
                       back is not None and back.arrived))

        # Nothing left to explore: back early, and says why.
        world2 = FakeWorld(pose=(0.5, 2.0, 0.0), **_CALIBRATED)
        mapper2 = FakeMapper(world2)
        mapper2.grid = _grid(free=True)
        with _ToolRig(world2, mapper2) as rig, \
                _TripRig(rig, end_pose=(2.0, 1.0, 0.5),
                         reason="no frontiers left"):
            _, back = _trip(5.0)
            legs = [c for c in world2.commands]
            check("a room already mapped from the start spot is TOURED, not "
                  "abandoned (2026-09-24 00:20: 25 cm and home)",
                  _line("tour") in rig.voice.said and len(legs) > 10,
                  f"{len(legs)} moves; said {rig.voice.said[:3]}")
            check("once she has been everywhere, she heads back early",
                  _line("frontiers_done") in rig.voice.said
                  and back.arrived, str(rig.voice.said[-3:]))
            check("and comes back to the mark",
                  math.hypot(world2.x - 0.5, world2.y - 2.0) < 0.5,
                  f"{math.hypot(world2.x - 0.5, world2.y - 2.0):.2f} m away")
            early = nsm.last_early_return()
            check("the early return is noted for the dashboard",
                  early and early.get("reason") == "toured the whole room",
                  str(early))
            check("tour legs are not logged as navigation trips",
                  not any(r["destination"] == "a new spot"
                          for r in nsm.recent_attempts(50)))


def thirteen_try_again_after_being_moved() -> None:
    item(13, "FAILSAFE, a person moves her, \"try again\": relocalise first, "
             "fresh allowance, same goal")
    import offline
    from mcp_server.tools import nav_tools
    with TempPlaces():
        nsm.set_home((2.0, 2.0, 0.0))
        nsm.clear_failsafe()

        # 1. Wedged for good on the way home: she stops and asks for help.
        stuck_world = FakeWorld(pose=(0.5, 2.0, 0.0), cm_s=45.0, deg_s=75.0,
                                dead=True, match_noise_m=0.01)
        stuck_map = FakeMapper(stuck_world)
        stuck_map.grid = _grid(free=True)
        with _ToolRig(stuck_world, stuck_map) as rig:
            async def first():
                await nav_tools.nav_home()
                await nav_tools._JOURNEY
            asyncio.run(first())
            record = nsm.failsafe_record()
            check("she ended in FAILSAFE and wrote down what she was doing",
                  record is not None and record["mode"] == "return_to_home",
                  str(record))
            check("with the stuck allowance spent",
                  record and record["recoveries"] == nsm.MAX_STUCK_RECOVERIES)
            check("FAILSAFE halted the wheels", "halt" in rig.calls)
            check("nothing lets her out on her own: the record just waits",
                  nsm.failsafe_record() is not None)

        # 2. A person lifts her clear and sets her down somewhere else.
        free_world = FakeWorld(pose=(1.0, 1.4, 0.6), **_CALIBRATED)
        free_map = FakeMapper(free_world)
        free_map.grid = _grid(free=True)
        intent = offline.understand("Friday, try again", strict=True)
        check("\"try again\" is understood locally",
              intent is not None and intent.tool == "try_again")
        with _ToolRig(free_world, free_map) as rig:
            async def second():
                reply = await nav_tools.try_again()
                await nav_tools._JOURNEY
                return reply
            reply = asyncio.run(second())
            check("it answers at once", reply == _line("try_again"), reply)
            check("it relocalised before a single wheel turned",
                  rig.relocalised_at[:1] == [0], str(rig.relocalised_at))
            check("from where she was PUT, not where she stopped",
                  free_world.commands
                  and math.hypot(free_world.x - 2.0, free_world.y - 2.0) < 0.5,
                  f"now {math.hypot(free_world.x - 2.0, free_world.y - 2.0):.2f}"
                  " m from home")
            row = nsm.recent_attempts(1)[0]
            check("it retried the mode that failed, and arrived",
                  row["destination"] == "home" and row["arrived"]
                  and row["retried"], str(row))
            check("the record is gone", nsm.failsafe_record() is None)
            check("she said so, in order",
                  rig.voice.said[:1] == [_line("try_again")]
                  and rig.voice.said[-1] == _line("arrived_home"),
                  str(rig.voice.said))
            _items.append(("FAILSAFE, moved, try again",
                           row["arrived"] and row["retried"]))


def fourteen_trip_edge_cases() -> None:
    item(14, "Explore trip: the OLD map only (no fresh map), a person's stop, "
             "and an old trip-map record")
    from mcp_server.tools import nav_tools
    with TempPlaces():
        # 1. She cannot be placed on the saved map even after moving to tell
        #    two look-alike spots apart: she does NOT explore -- there is no
        #    fresh map to fall back on any more (Harsh, 2026-09-25).
        world = FakeWorld(pose=(1.0, 1.0, 0.0), **_CALIBRATED)
        with _ToolRig(world, None) as rig, \
                _TripRig(rig, end_pose=(2.6, 2.2, 2.0)) as trip:
            async def cannot_tell():
                rig.relocalised_at.append(len(world.commands))
                return None, ("even after moving, two places on my map "
                              "still look alike")
            nsm.relocalise = cannot_tell
            said, result = _trip(2.0)
            check("she does not set off on a guess",
                  result is None and not trip.ran and not world.commands,
                  f"explorer ran: {bool(trip.ran)}, moves: {world.commands[:3]}")
            check("and no fresh map exists to fall back on",
                  not hasattr(nav_tools, "_working_map"))

        # 2. A person says stop while she explores: she stays stopped.
        world2 = FakeWorld(pose=(0.5, 2.0, 0.0), **_CALIBRATED)
        mapper2 = FakeMapper(world2)
        mapper2.grid = _grid(free=True)
        with _ToolRig(world2, mapper2) as rig, \
                _TripRig(rig, end_pose=(2.2, 1.4, 1.0),
                         reason="stop was called", person_stopped=True):
            _, result = _trip(2.0)
            check("after \"stop\" she does NOT drive back",
                  not world2.commands
                  and math.hypot(world2.x - 2.2, world2.y - 1.4) < 0.1,
                  str(world2.commands[:3]))
            check("and says nothing about coming back",
                  _line("setting_off", "where I set off from")
                  not in rig.voice.said, str(rig.voice.said))

        # 3. A FAILSAFE record left from a trip on a fresh map -- the kind the
        #    23:55 trip on 2026-09-24 wrote. That map no longer exists, so
        #    "try again" says so and clears it; it never drives to those
        #    coordinates on the saved map.
        world3 = FakeWorld(pose=(0.5, 2.0, 0.0), **_CALIBRATED)
        with _ToolRig(world3, None) as rig:
            nsm.write_failsafe({"mode": "returning_to_start",
                                "label": "where I set off from",
                                "goal_xy": [0.0, 0.0], "why": "lost",
                                "at": time.time(), "frame": "trip",
                                "paused_exploration": False,
                                "resume_minutes": None, "recoveries": 0})

            async def retry():
                await nav_tools.try_again()
                return await nav_tools._JOURNEY
            asyncio.run(retry())
            check("an old trip-map record is never driven to",
                  not world3.commands
                  and rig.voice.said[-1] == _line("trip_map_gone")
                  and nsm.failsafe_record() is None, str(rig.voice.said[-2:]))
        _items.append(("Explore trip edge cases", True))


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
    nine_round_trip()
    ten_offline_voice_errand()
    eleven_stuck_is_not_blocked()
    twelve_explore_trip()
    thirteen_try_again_after_being_moved()
    fourteen_trip_edge_cases()

    print("\n" + "=" * 64)
    for name, ok in _items:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 64)
    print(f"{'ALL SCENARIOS PASSED' if not _failures else f'{_failures} CHECKS FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
