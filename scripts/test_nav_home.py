#!/usr/bin/env python3
"""Test nav_state_machine.py and landmarks.py — file 14 steps 6-7.

    scripts/test_nav_home.py

No hardware, no motors, and NO WRITES TO THE REAL DATABASE. `slam/places.py`
talks to `memory.db()`, which is Friday's live memory — three processes have
it open and it holds things she was actually told. So the place-store tests
run against a temporary sqlite file swapped in for the duration, and the swap
is undone in a finally block. A test that names "the kitchen" in the real
database would still be there tomorrow, and `go_to` would believe it.
"""

from __future__ import annotations

import asyncio
import math
import sys
import json
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
import landmarks                                        # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0

# ⚠ THE ATTEMPT LOG IS REAL AND THE DASHBOARD READS IT. `_drive_to` appends a
# row to `state/nav_log.jsonl` on every attempt, so the simulated drives below
# were writing fake navigation history onto Harsh's Navigation panel — forty
# rows of "returned home, failsafe" for journeys that never happened. Anything
# in this file that drives must write somewhere else, so the redirect is done
# once, here, for the whole run rather than per test.
_LOG_DIR = tempfile.TemporaryDirectory()
nsm.NAV_LOG = Path(_LOG_DIR.name) / "nav_log.jsonl"
# The same for every other file navigation writes. A leaked FAILSAFE record is
# worse than a fake log row: "try again" on the real robot would act on it.
nsm.FAILSAFE_FILE = Path(_LOG_DIR.name) / "nav_failsafe.json"
nsm.EARLY_RETURN_FILE = Path(_LOG_DIR.name) / "nav_early_return.json"
nsm.PENDING_FILE = Path(_LOG_DIR.name) / "nav_round_trip.json"
nsm.NAV_MODE_FILE = Path(_LOG_DIR.name) / "nav_mode.json"


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


class TempPlaces:
    """Point memory.db() at a throwaway file, and put it back afterwards."""

    def __enter__(self):
        import memory
        self.memory = memory
        self.dir = tempfile.TemporaryDirectory()
        self.saved_path = config.MEMORY_DB if hasattr(config, "MEMORY_DB") else None
        memory.close()
        self.patched = Path(self.dir.name) / "test_memory.db"
        self._orig = memory.db
        conn = {}

        def db():
            import sqlite3
            if "c" not in conn:
                c = sqlite3.connect(self.patched, check_same_thread=False)
                c.row_factory = sqlite3.Row
                c.execute("""CREATE TABLE IF NOT EXISTS places (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    x REAL NOT NULL, y REAL NOT NULL,
                    radius_m REAL NOT NULL DEFAULT 0.75,
                    kind TEXT NOT NULL DEFAULT 'place', note TEXT,
                    created_at REAL NOT NULL, last_seen REAL,
                    visits INTEGER NOT NULL DEFAULT 0)""")
                c.commit()
                conn["c"] = c
            return conn["c"]

        memory.db = db
        return self

    def __exit__(self, *exc):
        self.memory.db = self._orig
        self.dir.cleanup()
        return False


def _grid(shape=(120, 120), free=False) -> OccupancyGrid:
    g = OccupancyGrid(0.05)
    g.logodds = np.zeros(shape, np.float32)      # all unknown
    g.origin = np.zeros(2, np.float64)
    g._empty = False
    if free:
        g.logodds[:] = -1.0
    return g


# --------------------------------------------------------------------------
# step 6
# --------------------------------------------------------------------------

def test_timer() -> None:
    print("\n— the home timer —")
    t = nsm.HomeTimer(minutes=10.0)
    check("it does not run until started", not t.running)
    check("an unstarted timer never expires", not t.expired(now=1e9))
    check("remaining is infinite before it starts",
          t.remaining_s(now=0.0) == float("inf"))

    t.start(now=0.0)
    check("it runs once started", t.running)
    check("not expired at nine minutes", not t.expired(now=9 * 60))
    check("remaining counts down",
          abs(t.remaining_s(now=9 * 60) - 60.0) < 1e-9,
          f"{t.remaining_s(now=9 * 60):.1f} s")
    check("expired at ten", t.expired(now=10 * 60))
    check("remaining floors at zero", t.remaining_s(now=99 * 60) == 0.0)
    t.stop()
    check("stopping it stops it", not t.running and not t.expired(now=1e9))

    check("the duration is configurable",
          nsm.HomeTimer(minutes=2.0).minutes == 2.0)


def test_state_transitions() -> None:
    print("\n— the state machine —")
    r = nsm.NavController(robot=object(), minutes=5.0)
    check("it starts idle", r.state is nsm.NavState.IDLE)
    check("idle does not transition on its own",
          r.tick(now=1e9) is nsm.NavState.IDLE)

    r.exploring(now=0.0)
    check("exploring starts the clock", r.state is nsm.NavState.EXPLORING
          and r.timer.running)
    check("it stays exploring before the timer fires",
          r.tick(now=60.0) is nsm.NavState.EXPLORING)
    check("EXPLORING -> RETURN_TO_HOME on expiry",
          r.tick(now=5 * 60) is nsm.NavState.RETURN_TO_HOME)
    check("and it does not bounce back",
          r.tick(now=5 * 60 + 1) is nsm.NavState.RETURN_TO_HOME)
    check("the state serialises for telemetry",
          nsm.NavState.RETURN_TO_HOME.value == "return_to_home")
    check("status reports the remaining time",
          r.status(now=0.0)["minutes"] == 5.0)


def test_home_is_a_place() -> None:
    print("\n— where home is —")
    with TempPlaces():
        from slam import places
        check("no home before one is set", nsm.home_place() is None)
        check("distance_home is None with no home",
              nsm.distance_home((1.0, 1.0)) is None)

        nsm.set_home((1.5, 2.0, 0.0), note="by the door")
        row = nsm.home_place()
        check("home is stored", row is not None and row["name"] == "home")
        check("in the places table every other tool already reads",
              any(p["name"] == "home" for p in places.all()))
        check("distance_home measures from it",
              abs(nsm.distance_home((1.5, 5.0)) - 3.0) < 1e-6,
              f"{nsm.distance_home((1.5, 5.0)):.2f} m")
        check("it is marked as a home, not a generic place",
              row.get("kind") == "home", str(row.get("kind")))


def test_run_refuses_rather_than_guessing() -> None:
    print("\n— it refuses rather than driving to a stale coordinate —")
    with TempPlaces():
        r = nsm.NavController(robot=object())
        report = asyncio.run(r.go_home(mapper=object()))
        check("no home stored: it says so and does not move",
              not report.arrived and "do not know where home is" in report.why,
              report.why)
        check("and the state says failed", report.state is nsm.NavState.FAILSAFE)

        nsm.set_home((1.0, 1.0, 0.0))

        class Untrusted:
            grid = _grid(free=True)
            pose = (0.2, 0.2, 0.0)

        # A mapper is passed in, so relocalisation is the caller's promise —
        # but an unreachable goal must still be refused, not approximated.
        walled = _grid()                        # all unknown: nothing plannable
        blocked = Untrusted()
        blocked.grid = walled
        report2 = asyncio.run(nsm.NavController(robot=object()).go_home(mapper=blocked))
        check("an unmapped room is refused, not approximated",
              not report2.arrived
              and "nowhere on my map I can plan" in report2.why,
              report2.why)
        check("it reports the grid it planned on",
              "plannable_cells" in report2.grid)


def test_run_plans_and_follows() -> None:
    print("\n— a real return, on a simulated chassis —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld

    with TempPlaces():
        nsm.set_home((2.0, 2.0, 0.0))
        world = FakeWorld(pose=(0.5, 2.0, 0.0))
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)          # a wide open, fully mapped room

        # dwell_s=0: the real monitor wants a second of WALL CLOCK inside
        # tolerance, and this simulation runs eighty ticks in milliseconds.
        # The dwell rule itself is tested against injected clocks in
        # test_nav_follower.py; what is under test here is the plumbing.
        import nav_follower
        r = nsm.NavController(robot=FakeRobot(world))
        report = asyncio.run(r.go_home(
            mapper=mapper, max_ticks=80,
            arrival=nav_follower.ArrivalMonitor((2.0, 2.0), cell_m=0.05,
                                                dwell_s=0.0, min_samples=2)))
        check("it planned a route", report.waypoints >= 2,
              f"{report.waypoints} waypoints, {report.planned_m:.2f} m")
        check("it arrived", report.arrived, report.why)
        check("the state machine ends in ARRIVED",
              report.state is nsm.NavState.ARRIVED)
        check("and it is really near home",
              math.hypot(world.x - 2.0, world.y - 2.0) < 0.5,
              f"{math.hypot(world.x - 2.0, world.y - 2.0):.2f} m away")
        check("the follow report is carried through",
              "ticks" in report.follow and report.follow["ticks"] > 0)


# --------------------------------------------------------------------------
# step 7
# --------------------------------------------------------------------------

def test_area_watch() -> None:
    print("\n— noticing somewhere new —")
    g = _grid()
    watch = landmarks.AreaWatch(min_area_m2=1.0)
    g.logodds[10:30, 10:30] = -1.0               # 1 m x 1 m known at the start
    check("the first look is the baseline, not a discovery",
          watch.update(g) is None)

    g.logodds[40:80, 40:80] = -1.0               # 2 m x 2 m = 4 m^2, new
    found = watch.update(g)
    check("a big new contiguous area is a discovery", found is not None)
    check("its area is right", found and abs(found.area_m2 - 4.0) < 0.05,
          f"{found.area_m2:.2f} m²" if found else "")
    check("its centroid is in the middle of it",
          found and abs(found.centroid[0] - 3.0) < 0.1
          and abs(found.centroid[1] - 3.0) < 0.1,
          str(found.centroid) if found else "")
    check("the radius covers roughly what was found",
          found and landmarks.MIN_RADIUS_M <= found.radius_m <= landmarks.MAX_RADIUS_M,
          f"{found.radius_m:.2f} m" if found else "")
    check("nothing new the second time", watch.update(g) is None)

    g.logodds[0:3, 0:3] = -1.0                   # a few cells of edge trimming
    check("edge trimming is not a discovery", watch.update(g) is None)

    # Scattered specks that ADD UP to the threshold are still not a region.
    scatter = _grid()
    w2 = landmarks.AreaWatch(min_area_m2=1.0)
    scatter.logodds[5, 5] = -1.0
    w2.update(scatter)
    for i in range(30):
        scatter.logodds[i * 3, 60] = -1.0
    check("a scatter of cells is not a room", w2.update(scatter) is None)


def test_area_watch_survives_map_growth() -> None:
    """The grid reallocates as it grows; 'new' must not mean 'reindexed'."""
    print("\n— when the map grows underneath it —")
    g = _grid((60, 60))
    g.logodds[10:50, 10:50] = -1.0
    watch = landmarks.AreaWatch(min_area_m2=1.0)
    watch.update(g)

    bigger = _grid((100, 100))
    bigger.origin = np.array([-1.0, -1.0])       # grown down and left
    shift = int(round(1.0 / bigger.cell))
    bigger.logodds[10 + shift:50 + shift, 10 + shift:50 + shift] = -1.0
    check("the same free space after a grow is not a discovery",
          watch.update(bigger) is None)


def test_inside_known_place() -> None:
    print("\n— not naming the same room twice —")
    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.0, 2.0, 0.0), radius_m=1.0)
        check("standing in it counts as inside",
              landmarks.inside_known_place((2.3, 2.1)) is not None)
        check("just outside the radius still counts, by the margin",
              landmarks.inside_known_place((3.2, 2.0)) is not None)
        check("well away does not",
              landmarks.inside_known_place((8.0, 8.0)) is None)


def test_naming_flow() -> None:
    print("\n— asking, and only saving a real answer —")
    with TempPlaces():
        asked: list[str] = []

        def ask(q):
            asked.append(q)
            return "the workshop"

        namer = landmarks.LandmarkNamer(ask=ask, min_area_m2=1.0)
        g = _grid()
        g.logodds[10:20, 10:20] = -1.0
        asyncio.run(namer.consider(g, (0.75, 0.75, 0.0)))     # baseline
        g.logodds[40:80, 40:80] = -1.0
        result = asyncio.run(namer.consider(g, (3.0, 3.0, 0.0)))

        check("it asked", result.asked and asked, str(asked[:1]))
        check("the question says how big the area is",
              "square metres" in asked[0], asked[0])
        check("it saved the answer", result.saved, result.why)
        check("under the name given", result.name == "the workshop",
              result.name)
        check("with a map session id",
              result.place.get("map_session_id", "").startswith("map-"),
              str(result.place))
        check("and a radius that came from the area",
              result.place["radius_m"] > 0)

        from slam import places
        row = places.find("the workshop")
        check("it is in the places table go_to reads", row is not None)
        check("the session id round-trips out of the note",
              landmarks.session_of(row) == result.place["map_session_id"],
              str(landmarks.session_of(row)))


def test_no_answer_saves_nothing() -> None:
    print("\n— a name she did not hear is not a name —")
    with TempPlaces():
        g = _grid()
        for answer in ("", None, "   ", "no", "I don't know", 42):
            namer = landmarks.LandmarkNamer(ask=lambda q, a=answer: a,
                                            min_area_m2=1.0)
            grid = _grid()
            grid.logodds[5:15, 5:15] = -1.0
            asyncio.run(namer.consider(grid, (0.5, 0.5, 0.0)))
            grid.logodds[40:80, 40:80] = -1.0
            res = asyncio.run(namer.consider(grid, (3.0, 3.0, 0.0)))
            check(f"{answer!r} saves nothing", not res.saved, res.why)

        from slam import places
        check("nothing at all was written", not places.all(),
              f"{len(places.all())} rows")


def test_no_ask_means_no_save() -> None:
    print("\n— the process that cannot listen —")
    with TempPlaces():
        namer = landmarks.LandmarkNamer(ask=None, min_area_m2=1.0)
        g = _grid()
        g.logodds[5:15, 5:15] = -1.0
        asyncio.run(namer.consider(g, (0.5, 0.5, 0.0)))
        g.logodds[40:80, 40:80] = -1.0
        res = asyncio.run(namer.consider(g, (3.0, 3.0, 0.0)))
        check("it notices but does not invent a name", not res.saved)
        check("and says why", "nothing here can ask" in res.why, res.why)
        check("it did not pretend to ask", not res.asked)


def test_overwrite_needs_confirmation() -> None:
    print("\n— it will not silently move a place you named —")
    with TempPlaces():
        from slam import places
        places.remember_here("the study", (9.0, 9.0, 0.0), radius_m=0.75)

        found = landmarks.Discovery(4.0, (3.0, 3.0), 1600, 1.1)
        namer = landmarks.LandmarkNamer(ask=lambda q: "the study",
                                        confirm=lambda q: "no",
                                        min_area_m2=1.0)
        res = asyncio.run(namer.save("the study", (3.0, 3.0, 0.0), found))
        check("a refused confirmation leaves it alone", not res.saved, res.why)
        moved = places.find("the study")
        check("the original is untouched",
              abs(moved["x"] - 9.0) < 1e-9 and abs(moved["y"] - 9.0) < 1e-9,
              f"({moved['x']}, {moved['y']})")

        yes = landmarks.LandmarkNamer(ask=lambda q: "the study",
                                      confirm=lambda q: "yes please",
                                      min_area_m2=1.0)
        res2 = asyncio.run(yes.save("the study", (3.0, 3.0, 0.0), found))
        check("an agreed confirmation moves it", res2.saved, res2.why)
        check("and reports what it moved", res2.overwrote.get("moved_m", 0) > 8,
              str(res2.overwrote))
        now = places.find("the study")
        check("the row really moved", abs(now["x"] - 3.0) < 1e-9)

        silent = landmarks.LandmarkNamer(ask=None, min_area_m2=1.0)
        res3 = asyncio.run(silent.save("the study", (5.0, 5.0, 0.0), found))
        check("with nothing to confirm with, it does not overwrite",
              not res3.saved, res3.why)


def test_yes_and_name_parsing() -> None:
    print("\n— hearing the answer correctly —")
    for text in ("yes", "Yes please", "yeah", "go ahead", "sure", "ok"):
        check(f"{text!r} is yes", landmarks._is_yes(text))
    for text in ("no", "No, leave it", "don't", "maybe", "", "what?",
                 "the kitchen"):
        check(f"{text!r} is not yes", not landmarks._is_yes(text))

    for raw, want in (("the kitchen", "the kitchen"),
                      ("Call it the hallway", "the hallway"),
                      ("it's the back bedroom", "the back bedroom"),
                      ("  The  Study  ", "the study"),
                      ("let's call this the landing", "the landing")):
        got = landmarks._clean(raw)
        check(f"{raw!r} -> {want!r}", got == want, got)
    for raw in ("no", "I don't know", "",
                "well I suppose you could call it something like the kitchen"):
        check(f"{raw!r} is not a name", landmarks._clean(raw) == "")


def test_map_session_rotates_when_the_map_is_rebuilt() -> None:
    print("\n— which map a landmark belongs to —")
    saved = landmarks._SESSION_FILE
    with tempfile.TemporaryDirectory() as tmp:
        landmarks._SESSION_FILE = Path(tmp) / "map_session.json"
        try:
            class G:
                scans = 100
            first = landmarks.map_session_id(G())
            check("an id is minted", first.startswith("map-"), first)
            check("the same map keeps it", landmarks.map_session_id(G()) == first)

            class Grown:
                scans = 400
            check("a map that grew keeps it",
                  landmarks.map_session_id(Grown()) == first)

            time.sleep(1.05)          # the id is second-resolution
            class Rebuilt:
                scans = 3
            check("a rebuilt map gets a new one",
                  landmarks.map_session_id(Rebuilt()) != first)
        finally:
            landmarks._SESSION_FILE = saved



# --------------------------------------------------------------------------
# steps 8-9
# --------------------------------------------------------------------------

def test_resolve_never_guesses() -> None:
    print("\n— it will not drive on a near miss —")
    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.0, 2.0, 0.0))
        places.remember_here("the back bedroom", (5.0, 5.0, 0.0))

        for asked in ("the kitchen", "kitchen", "The Kitchen", "  kitchen  "):
            row, why = nsm.resolve_landmark(asked)
            check(f"{asked!r} resolves exactly", row is not None
                  and row["name"] == "the kitchen", why)

        row, why = nsm.resolve_landmark("bedroom")
        check("a substring does NOT resolve", row is None, str(row))
        check("but it is offered back as a question",
              "did you mean" in why.lower() and "the back bedroom" in why, why)

        row, why = nsm.resolve_landmark("the garage")
        check("an unknown name is refused", row is None)
        check("and she says what she does know",
              "the kitchen" in why and "the back bedroom" in why, why)

        row, why = nsm.resolve_landmark("")
        check("an empty name is refused", row is None, why)


def test_target_nav_refuses_unknown() -> None:
    print("\n— TARGET_NAV with a name she does not know —")
    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.0, 2.0, 0.0))
        c = nsm.NavController(robot=object())
        report = asyncio.run(c.go_to_landmark("the cellar", action="say hello"))
        check("it does not drive", not report.arrived)
        check("it ends in FAILSAFE", report.state is nsm.NavState.FAILSAFE)
        check("the destination is echoed back", report.destination == "the cellar")
        check("and the queued action is not lost",
              report.action == "say hello", report.action)


def test_both_modes_share_one_engine() -> None:
    print("\n— RETURN_TO_HOME and TARGET_NAV are the same engine —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
    import nav_follower

    with TempPlaces():
        from slam import places
        nsm.set_home((2.0, 2.0, 0.0))
        places.remember_here("the kitchen", (2.0, 2.0, 0.0))

        def drive(which, name=None):
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            c = nsm.NavController(robot=FakeRobot(world))
            arrival = nav_follower.ArrivalMonitor((2.0, 2.0), cell_m=0.05,
                                                  dwell_s=0.0, min_samples=2)
            if which == "home":
                rep = asyncio.run(c.go_home(mapper=mapper, max_ticks=80,
                                            arrival=arrival))
            else:
                rep = asyncio.run(c.go_to_landmark(
                    name, action="say what you see", mapper=mapper,
                    max_ticks=80, arrival=arrival))
            return rep, world

        home_rep, home_world = drive("home")
        targ_rep, targ_world = drive("target", "the kitchen")

        check("returning home arrives", home_rep.arrived, home_rep.why)
        check("target nav arrives", targ_rep.arrived, targ_rep.why)
        check("they report different states on the way in",
              home_rep.destination == "home"
              and targ_rep.destination == "the kitchen")
        check("both end in ARRIVED",
              home_rep.state is nsm.NavState.ARRIVED
              and targ_rep.state is nsm.NavState.ARRIVED)
        check("both got to the same place, because it is the same engine",
              abs(home_world.x - targ_world.x) < 0.4,
              f"{home_world.x:.2f} vs {targ_world.x:.2f}")
        check("the action rides along with target nav only",
              targ_rep.action == "say what you see" and home_rep.action == "")


def test_action_is_returned_not_performed() -> None:
    print("\n— the action comes back, it is not performed here —")
    src = (Path(__file__).resolve().parent.parent
           / "mcp_server" / "tools" / "nav_tools.py").read_text()
    import ast
    tree = ast.parse(src)
    refs = {ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    check("the nav tool never speaks", not any(
        r.startswith("tts.") or r.endswith(".speak") for r in refs), str(
            sorted(r for r in refs if "speak" in r or "tts" in r)))
    check("and never sets the face", not any("set_face" in r for r in refs))
    check("it hands the action back as action_pending",
          "action_pending" in src)


def test_replan_only_when_replanning_can_help() -> None:
    print("\n— re-planning, and when not to —")
    check("a human stop is final", nsm._is_final("stop was called"))
    check("a refusal is final", nsm._is_final("refused: no fresh observation"))
    stall = ("the last 3 moves delivered under 25% of what they asked for — "
             "the wheels are not driving")
    check("a stall is NOT final — it gets a recovery first (step 4)",
          not nsm._is_final(stall))
    check("a stall is recognised as stuck", nsm.is_stuck(stall))
    check("and stuck is not blocked", not nsm.is_block(stall))
    check("a block is not stuck",
          not nsm.is_stuck("REFUSED: something is 150 mm ahead, inside the "
                           "200 mm stop distance"))
    check("an unmatched pose is final",
          nsm._is_final("I do not know where I am"))
    check("a budget is final", nsm._is_final("time budget"))
    check("running out of path is NOT final",
          not nsm._is_final("in tolerance, but still correcting"))
    check("no route is NOT final", not nsm._is_final("nothing passable connects"))


def test_replan_loop_retries_and_gives_up() -> None:
    print("\n— the re-plan loop —")
    with TempPlaces():
        nsm.set_home((1.0, 1.0, 0.0))

        class Blocked:
            grid = _grid()                 # all unknown: never plannable
            pose = (0.2, 0.2, 0.0)

        c = nsm.NavController(robot=object())
        rep = asyncio.run(c.go_home(mapper=Blocked(), max_attempts=3))
        check("it tried more than once", rep.attempts == 3, str(rep.attempts))
        check("and counted the re-plans", rep.replans == 2, str(rep.replans))
        check("then gave up in FAILSAFE", rep.state is nsm.NavState.FAILSAFE)
        check("saying what is actually wrong",
              "nowhere on my map I can plan" in rep.why, rep.why)

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
        world = FakeWorld(pose=(0.5, 2.0, 0.0))
        mapper = FakeMapper(world)
        mapper.grid = _grid(free=True)
        nsm.set_home((2.0, 2.0, 0.0))
        stopper = FakeRobot(world, stop_after=1)
        rep2 = asyncio.run(nsm.NavController(robot=stopper).go_home(
            mapper=mapper, max_ticks=30))
        check("a human stop is never retried", rep2.attempts == 1,
              f"{rep2.attempts} attempts — a stop was retried")
        check("and it ends in FAILSAFE", rep2.state is nsm.NavState.FAILSAFE)


def test_snapshot_is_honest_about_what_it_keeps() -> None:
    print("\n— pausing exploration —")
    snap = nsm.ExplorationSnapshot.capture(remaining_minutes=7.5)
    check("it captures whatever telemetry is on disk", snap.taken_at > 0)
    check("it keeps the time left", snap.remaining_minutes == 7.5)
    d = snap.as_dict()
    check("and says plainly that frontiers are recomputed, not restored",
          "recomputed from the saved map" in d["note"], d["note"])
    check("the in-progress target is carried as a hint",
          "target_xy" in d)

    c = nsm.NavController(robot=object())
    check("resuming with nothing paused says so",
          c.resume_exploration()["resumed"] is False)


def test_tools_are_registered() -> None:
    print("\n— the MCP tools —")
    from mcp_server import registry
    registry.load_tool_modules()
    names = {name for _, name, _ in registry.REGISTRY}
    for wanted in ("navigate_to", "return_home", "nav_status"):
        check(f"{wanted} is registered", wanted in names)
    check("go_to is untouched and still registered", "go_to" in names)
    check("nothing registered twice",
          len(names) == len({n for _, n, _ in registry.REGISTRY}))

    descriptions = {name: desc for _, name, desc in registry.REGISTRY}
    check("navigate_to tells the model it does not guess at names",
          "guess" in descriptions["navigate_to"].lower(),
          descriptions["navigate_to"][:80])



# --------------------------------------------------------------------------
# steps 10-11
# --------------------------------------------------------------------------

class BlockingRobot:
    """A fake robot that refuses to drive, and may or may not clear."""

    def __init__(self, world, clears_after: int | None = None) -> None:
        self.world = world
        self.clears_after = clears_after
        self.checks = 0
        self.halted = False
        self.sweeps = 0

    async def sweep(self):
        self.sweeps += 1
        return {0: 1500}

    async def observe(self):
        pass

    async def drive_seconds(self, direction, seconds):
        return ("REFUSED: something is 150 mm ahead, inside the 200 mm "
                "stop distance (tof)")

    async def clearance(self):
        self.checks += 1
        if self.clears_after is not None and self.checks >= self.clears_after:
            return 1800.0, False
        return 150.0, True

    async def halt(self):
        self.halted = True
        return "stopped"

    def say(self, text):
        pass

    def stopped(self):
        return False


def test_runtime_obstacles_stay_out_of_the_map() -> None:
    print("\n— the runtime obstacle layer —")
    o = nsm.RuntimeObstacles(ttl_s=10.0)
    check("it starts empty", len(o) == 0 and o.points(now=0.0).shape == (0, 2))
    o.add((1.0, 2.0), now=0.0)
    o.add((3.0, 4.0), now=0.0)
    check("points come back as world metres", o.points(now=1.0).shape == (2, 2))
    check("they expire", o.points(now=99.0).shape == (0, 2))

    src = (Path(__file__).resolve().parent.parent / "nav_state_machine.py").read_text()
    import ast
    refs = {ast.unparse(n) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute)}
    check("nothing here ever saves a grid",
          not any(r.endswith(".save") for r in refs),
          str(sorted(r for r in refs if "save" in r)))

    # And it must actually change what the planner may use.
    import path_planner
    g = _grid(free=True)
    clear = path_planner.preprocess(g)
    with_block = path_planner.preprocess(g, extra_blocked=np.array([[3.0, 3.0]]))
    check("a runtime obstacle removes plannable cells",
          with_block.plannable_cells < clear.plannable_cells,
          f"{clear.plannable_cells} -> {with_block.plannable_cells}")


def test_block_classification() -> None:
    print("\n— what kind of stop was that —")
    for why in ("refused: something is 150 mm ahead, inside the 200 mm stop distance (tof)",
                "refused: something is 180 mm away — inside the 250 mm bump guard (LiDAR)",
                "refused: no range sensor can see ahead — refusing to drive forward blind",
                "refused: no fresh observation"):
        check(f"{why[:40]}… is a block", nsm.is_block(why))
    for why in ("stop was called", "time budget",
                "refused: cannot do distances yet — forward_cm_per_s is 0",
                "the last 3 moves delivered under 25% — not driving"):
        check(f"{why[:40]}… is not a block", not nsm.is_block(why))


def test_obstacle_position() -> None:
    print("\n— where the blocking thing is —")
    x, y = nsm.obstacle_ahead((0.0, 0.0, 0.0), 400.0)
    check("it is ahead along the heading", abs(x - 0.4) < 1e-9 and abs(y) < 1e-9,
          f"({x:.2f}, {y:.2f})")
    x, y = nsm.obstacle_ahead((1.0, 1.0, math.pi / 2), 500.0)
    check("it follows the heading round",
          abs(x - 1.0) < 1e-9 and abs(y - 1.5) < 1e-9, f"({x:.2f}, {y:.2f})")
    x, y = nsm.obstacle_ahead((0.0, 0.0, 0.0), None)
    check("an unreadable clearance still places it at arm's length",
          x >= config.NAV_ROBOT_RADIUS_M, f"{x:.2f}")


def test_goal_cell_occupied() -> None:
    print("\n— a goal she cannot stand on —")
    import path_planner
    g = _grid(free=True)
    g.logodds[40:44, 40:44] = 3.0                # a block at (2, 2)
    prep = path_planner.preprocess(g)
    cell, moved = nsm.snap_goal(prep, (42, 42))
    check("the goal moves to the nearest place she fits",
          cell is not None and moved > 0, f"moved {moved:.2f} m")
    check("and not far", moved <= nsm.GOAL_SNAP_M, f"{moved:.2f} m")
    check("the snapped cell is plannable", prep.plannable[cell])

    ok_cell, zero = nsm.snap_goal(prep, (10, 10))
    check("a reachable goal is not moved at all",
          ok_cell == (10, 10) and zero == 0.0)

    sealed = path_planner.preprocess(_grid())    # all unknown
    none_cell, far = nsm.snap_goal(sealed, (10, 10))
    check("nowhere to stand at all is refused, not approximated",
          none_cell is None)


def test_transient_block_does_not_storm() -> None:
    print("\n— someone walks past —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeWorld

    saved = nsm.BLOCK_RECHECK_S
    nsm.BLOCK_RECHECK_S = 0.0                    # do not really sleep
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            robot = BlockingRobot(world, clears_after=1)
            c = nsm.NavController(robot=robot)
            rep = asyncio.run(c.go_home(mapper=mapper, max_ticks=6,
                                        max_attempts=2))
            check("it rechecked before replanning", robot.checks > 0,
                  f"{robot.checks} rechecks")
            check("it saw a block", rep.blocks_seen > 0, str(rep.blocks_seen))
            check("but recorded no obstacle, because it walked away",
                  rep.blocks_confirmed == 0 and len(c.obstacles) == 0,
                  f"{rep.blocks_confirmed} confirmed")
    finally:
        nsm.BLOCK_RECHECK_S = saved


def test_persistent_block_is_remembered_and_planned_around() -> None:
    print("\n— something that is really there —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeWorld

    saved = nsm.BLOCK_RECHECK_S
    nsm.BLOCK_RECHECK_S = 0.0
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            world = FakeWorld(pose=(0.5, 2.0, 0.0))
            mapper = FakeMapper(world)
            mapper.grid = _grid(free=True)
            robot = BlockingRobot(world)          # never clears
            c = nsm.NavController(robot=robot)
            rep = asyncio.run(c.go_home(mapper=mapper, max_ticks=6,
                                        max_attempts=3))
            check("the block was confirmed", rep.blocks_confirmed > 0,
                  str(rep.blocks_confirmed))
            check("and went into the runtime layer", len(c.obstacles) > 0,
                  str(c.obstacles.as_dict()))
            check("the report carries it", rep.obstacles.get("count", 0) > 0)
            check("it ends in FAILSAFE, not trying forever",
                  rep.state is nsm.NavState.FAILSAFE)
            check("FAILSAFE halted the wheels", robot.halted)
            check("and it did not exceed the replanning budget",
                  rep.attempts <= 3, str(rep.attempts))
    finally:
        nsm.BLOCK_RECHECK_S = saved


def test_nav_log_round_trip() -> None:
    print("\n— the attempt log —")
    saved = nsm.NAV_LOG
    with tempfile.TemporaryDirectory() as tmp:
        nsm.NAV_LOG = Path(tmp) / "nav_log.jsonl"
        try:
            rep = nsm.ReturnReport(state=nsm.NavState.ARRIVED, arrived=True,
                                   destination="the kitchen", seconds=12.5,
                                   attempts=1, at=time.time())
            nsm._log_attempt(rep)
            rows = nsm.recent_attempts()
            check("one row written", len(rows) == 1, str(rows))
            check("with the destination", rows[0]["destination"] == "the kitchen")
            check("the outcome", rows[0]["arrived"] is True)
            check("and how long it took", rows[0]["seconds"] == 12.5)

            nsm.NAV_LOG.write_text('{"broken": ')       # a torn line
            check("a torn line does not break the reader",
                  nsm.recent_attempts() == [])
        finally:
            nsm.NAV_LOG = saved


def test_dashboard_panel_is_read_only() -> None:
    print("\n— the dashboard panel —")
    from dashboard import nav_view
    data = nav_view.current()
    check("it returns something", data.get("ok") is True)
    check("it says which frame the landmarks are in",
          "survey map" in data["note"] and "Map panel" in data["note"],
          data["note"])
    check("the states are listed for the UI",
          "return_to_home" in data["states"] and "failsafe" in data["states"])
    for mark in data["landmarks"]:
        if "u" in mark:
            check(f"{mark['name']} has a position on the plan",
                  0.0 <= mark["u"] <= 1.0 or not mark["on_map"],
                  f"u={mark['u']} v={mark['v']}")

    src = (Path(__file__).resolve().parent.parent
           / "dashboard" / "nav_view.py").read_text()
    import ast
    refs = {ast.unparse(n) for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Attribute)}
    check("the view never writes",
          not any(r.endswith((".save", ".write_text", ".remember_here"))
                  for r in refs), str(sorted(refs))[:120])

    server = (Path(__file__).resolve().parent.parent
              / "dashboard" / "server.py").read_text()
    check("the GET route is registered", '"/api/nav", api_nav)' in server)
    # ⚠ THIS PANEL USED TO HAVE NO POST, and the test asserted that. Harsh
    # asked for the dashboard to trigger target navigation too, so it has one
    # now — but the reason for the old rule has not gone away, so what is
    # checked is the thing that actually keeps it safe: the POST goes through
    # the MCP server, never the motors.
    check("the POST route exists now Harsh asked for it",
          'add_post("/api/nav"' in server)
    import ast
    post = server[server.index("async def api_nav_post"):
                  server.index("async def api_explore_post")]
    check("it never opens the motors itself",
          "hardware.motors" not in post and "Motors(" not in post)
    check("every action goes out through the MCP server",
          "robot.call(" in post, "")
    check("and it will not invent a destination",
          "landmark_name" in post and "which place?" in post)



# --------------------------------------------------------------------------
# round trip — "go to the kitchen and come back to where you started"
# --------------------------------------------------------------------------

def test_round_trip_mode_persists() -> None:
    print("\n— the round-trip mode flag —")
    saved = nsm.NAV_MODE_FILE
    with tempfile.TemporaryDirectory() as tmp:
        nsm.NAV_MODE_FILE = Path(tmp) / "nav_mode.json"
        try:
            check("off by default", not nsm.round_trip_enabled())
            check("turning it on saves", nsm.set_round_trip(True)["saved"])
            check("and it reads back", nsm.round_trip_enabled())
            check("turning it off saves", nsm.set_round_trip(False)["saved"])
            check("and it reads back off", not nsm.round_trip_enabled())
            nsm.NAV_MODE_FILE.write_text("{ broken")
            check("a corrupt file reads as off, not as a crash",
                  not nsm.round_trip_enabled())
        finally:
            nsm.NAV_MODE_FILE = saved


def test_round_trip_goes_and_comes_back() -> None:
    print("\n— she goes to the kitchen and comes back —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
    import nav_follower

    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.5, 2.0, 0.0))
        start = (0.5, 2.0, 0.0)
        world = FakeWorld(pose=start)
        mapper = FakeMapper(world)
        mapper.grid = _grid((160, 160), free=True)
        c = nsm.NavController(robot=FakeRobot(world))

        report = asyncio.run(c.go_to_landmark(
            "the kitchen", mapper=mapper, return_to_start=True,
            max_ticks=200,
            arrival=nav_follower.ArrivalMonitor((2.5, 2.0), cell_m=0.05,
                                                dwell_s=0.0, min_samples=2)))
        check("she got to the kitchen", report.arrived, report.why)
        check("the trip was a round trip", report.round_trip)
        check("the start pose was captured before setting off",
              report.start_pose
              and abs(report.start_pose[0] - start[0]) < 0.2,
              str(report.start_pose))
        check("she came back", report.returned_to_start,
              str(report.inbound.get("why", ""))[:90])
        check("and is really back where she began",
              math.hypot(world.x - start[0], world.y - start[1]) < 0.5,
              f"{math.hypot(world.x - start[0], world.y - start[1]):.2f} m "
              "from the start")
        check("the way back is reported separately from the way out",
              "why" in report.inbound and report.inbound["destination"]
              == "where I set off from", str(report.inbound)[:80])


def test_round_trip_off_means_stay() -> None:
    print("\n— with the mode off she stays put —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
    import nav_follower

    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.5, 2.0, 0.0))
        world = FakeWorld(pose=(0.5, 2.0, 0.0))
        mapper = FakeMapper(world)
        mapper.grid = _grid((160, 160), free=True)
        c = nsm.NavController(robot=FakeRobot(world))
        report = asyncio.run(c.go_to_landmark(
            "the kitchen", mapper=mapper, return_to_start=False,
            max_ticks=200,
            arrival=nav_follower.ArrivalMonitor((2.5, 2.0), cell_m=0.05,
                                                dwell_s=0.0, min_samples=2)))
        check("she arrived", report.arrived, report.why)
        check("it was not a round trip", not report.round_trip)
        check("she stayed at the kitchen",
              abs(world.x - 2.5) < 0.5,
              f"x={world.x:.2f} (kitchen is 2.5)")


def test_getting_there_survives_failing_to_come_back() -> None:
    """Arriving is what he asked for. Not getting back does not undo it.

    The outbound leg is driven normally and the inbound one is then given a
    robot that refuses everything. Trying to make ONE simulated chassis
    succeed and then fail at a chosen moment meant guessing where in the
    approach the refusal would land, which tested the guess rather than the
    guarantee.
    """
    print("\n— she got there but could not get back —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
    import nav_follower

    with TempPlaces():
        from slam import places
        places.remember_here("the kitchen", (2.5, 2.0, 0.0))
        start_pose = (0.5, 2.0, 0.0)
        world = FakeWorld(pose=start_pose)
        mapper = FakeMapper(world)
        mapper.grid = _grid((160, 160), free=True)

        c = nsm.NavController(robot=FakeRobot(world))
        report = asyncio.run(c.go_to_landmark(
            "the kitchen", mapper=mapper, return_to_start=False,
            max_ticks=200,
            arrival=nav_follower.ArrivalMonitor((2.5, 2.0), cell_m=0.05,
                                                dwell_s=0.0, min_samples=2)))
        check("the outbound leg arrived", report.arrived, report.why)

        # Now the way back, with a chassis that has stopped answering.
        c.robot = FakeRobot(world, refuse=True)
        asyncio.run(c._return_to_start(mapper, start_pose, report,
                                       max_ticks=20))
        check("the outbound leg is STILL reported as arrived", report.arrived,
              "coming back failing must not un-arrive her at the kitchen")
        check("but coming back is reported as failed",
              not report.returned_to_start)
        check("and the sentence says both",
              "could not get back" in report.why, report.why[-80:])
        check("the inbound leg has its own report",
              report.inbound.get("destination") == "where I set off from",
              str(report.inbound.get("why", ""))[:60])


def test_armed_return_for_exploration() -> None:
    print("\n— the return armed before an exploration run —")
    saved = nsm.PENDING_FILE
    with tempfile.TemporaryDirectory() as tmp:
        nsm.PENDING_FILE = Path(tmp) / "nav_round_trip.json"
        try:
            check("nothing armed to begin with", nsm.pending_return() is None)
            armed = nsm.arm_return_to_start((1.5, 2.5, 0.3), why="exploration")
            check("arming writes it down", armed["armed"])
            row = nsm.pending_return()
            check("and it reads back", row and abs(row["x"] - 1.5) < 1e-9)
            check("with why it was armed", row["why"] == "exploration")

            nsm.clear_pending()
            check("clearing clears it", nsm.pending_return() is None)

            # A stale one must never fire — it is a pose from another session.
            nsm.arm_return_to_start((9.0, 9.0, 0.0))
            stale = json.loads(nsm.PENDING_FILE.read_text())
            stale["at"] = time.time() - nsm.PENDING_TTL_S - 10
            nsm.PENDING_FILE.write_text(json.dumps(stale))
            check("a stale armed return is dropped, not driven to",
                  nsm.pending_return() is None)
        finally:
            nsm.PENDING_FILE = saved


def test_armed_return_drives_back() -> None:
    print("\n— and firing it brings her back —")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_nav_follower import FakeMapper, FakeRobot, FakeWorld
    import nav_follower

    saved = nsm.PENDING_FILE
    with tempfile.TemporaryDirectory() as tmp:
        nsm.PENDING_FILE = Path(tmp) / "nav_round_trip.json"
        try:
            with TempPlaces():
                # She "explored" away from her start and is now at (2.5, 2.0).
                nsm.arm_return_to_start((0.5, 2.0, 0.0), why="exploration")
                world = FakeWorld(pose=(2.5, 2.0, 0.0))
                mapper = FakeMapper(world)
                mapper.grid = _grid((160, 160), free=True)
                c = nsm.NavController(robot=FakeRobot(world))
                report = asyncio.run(c.return_to_armed_start(
                    mapper=mapper, max_ticks=200,
                    arrival=nav_follower.ArrivalMonitor(
                        (0.5, 2.0), cell_m=0.05, dwell_s=0.0, min_samples=2)))
                check("she drove back", report.arrived, report.why)
                check("to the armed spot",
                      math.hypot(world.x - 0.5, world.y - 2.0) < 0.5,
                      f"{math.hypot(world.x - 0.5, world.y - 2.0):.2f} m away")
                check("the arming is consumed so it cannot fire twice",
                      nsm.pending_return() is None)
                check("nothing armed means nothing happens",
                      not asyncio.run(
                          c.return_to_armed_start(mapper=mapper)).arrived)
        finally:
            nsm.PENDING_FILE = saved


def test_round_trip_tools_registered() -> None:
    print("\n— the round-trip tools —")
    from mcp_server import registry
    registry.load_tool_modules()
    entries = {name: desc for _, name, desc in registry.REGISTRY}
    for wanted in ("set_round_trip", "explore_and_return", "come_back"):
        check(f"{wanted} is registered", wanted in entries)
    check("explore_and_return tells the model when to use it",
          "come back" in entries["explore_and_return"].lower())
    # The docstring wraps, so "round trip" can arrive as "round\ntrip".
    flat = " ".join(entries["set_round_trip"].lower().split())
    check("set_round_trip lists the words Harsh actually says",
          all(w in flat for w in ("come back", "there and back", "round trip")))
    check("navigate_to now takes return_to_start",
          "return_to_start" in entries["navigate_to"])


# --------------------------------------------------------------------------
# 2026-09-23: stuck is not blocked, the voice, the way out of FAILSAFE, and
# exploration that runs out of places to go
# --------------------------------------------------------------------------

def _sim():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_nav_follower
    return test_nav_follower


def _quiet_voice():
    import nav_voice
    return nav_voice.Voice(speak=lambda _text: None, background=False)


def _arrive(goal):
    import nav_follower
    return nav_follower.ArrivalMonitor(goal, cell_m=0.05, dwell_s=0.0,
                                       min_samples=2)


def _wedged(pose=(0.5, 2.0, 0.0), frees_on_reverse=True):
    """A chassis whose wheels do nothing -- until it is backed off, maybe.

    Rates match the calibrated drivetrain so the recovery manoeuvre is its
    real size; `efficiency` keeps the usual 30% loss.
    """
    sim = _sim()

    class Wedged(sim.FakeWorld):
        def drive(self, direction, seconds):
            if direction == "backward" and frees_on_reverse:
                self.dead = False
            super().drive(direction, seconds)

    # 1 cm matcher noise: the real tracker's standstill scatter, where the
    # default 2 cm would read a dead chassis as "moved" one time in ten.
    return Wedged(pose=pose, cm_s=45.0, deg_s=75.0, dead=True,
                  match_noise_m=0.01)


def _halting_robot(world):
    sim = _sim()

    class Halting(sim.FakeRobot):
        halted = False

        async def halt(self):
            self.halted = True
            return "stopped"

    return Halting(world)


def _line(key, dest=""):
    import nav_voice
    return nav_voice.line(key, dest)


def test_voice_lines_match_the_exploration() -> None:
    print("\n— what navigation says (step 10) —")
    import nav_voice
    lines = nav_voice.LINES
    check("every line is a plain statement ending in a full stop",
          all(t.endswith(".") and "!" not in t for t in lines.values()))
    check("every line is short enough to say in one breath",
          all(len(t) <= 80 for t in lines.values()),
          max(lines.values(), key=len))
    check("stuck and blocked never share a sentence",
          lines["stuck"] != lines["blocked"])
    check("the stuck line is the exploration's own sentence",
          lines["stuck"] == "I am unable to make progress. Initiating "
                            "recovery.")
    check("recovery names where she is going, like exploration names its job",
          _line("recovered", "the kitchen")
          == "Recovery successful. Continuing to the kitchen.")
    check("\"I have arrived at home\" is not said",
          nav_voice.arrival_key("home") == "arrived_home"
          and _line("arrived_home") == "I am home.")
    v = _quiet_voice()
    check("a repeated block is announced once, not three times",
          v.say("blocked", now=0.0) and not v.say("blocked", now=3.0)
          and v.say("blocked", now=9.0))
    check("an arrival is never throttled",
          v.say("arrived", "the desk", now=0.0)
          and v.say("arrived", "the desk", now=0.5))
    check("the engine is silent unless it is handed a voice",
          nsm.NavController().voice is nav_voice.SILENT)
    failing = nav_voice.Voice(speak=lambda _t: 1 / 0, background=False)
    check("a voice that cannot speak never ends a drive",
          failing.say("arrived_home") == "I am home.")


def test_stuck_recovers_before_replanning() -> None:
    print("\n— stuck: a recovery manoeuvre first, THEN a new plan (step 4) —")
    with TempPlaces():
        nsm.set_home((2.0, 2.0, 0.0))
        world = _wedged(frees_on_reverse=True)
        sim = _sim()
        mapper = sim.FakeMapper(world)
        mapper.grid = _grid(free=True)
        voice = _quiet_voice()
        c = nsm.NavController(robot=_halting_robot(world), voice=voice)
        rep = asyncio.run(c.go_home(mapper=mapper, max_ticks=80,
                                    arrival=_arrive((2.0, 2.0))))
        check("the stall was seen", rep.stucks_seen == 1, str(rep.stucks_seen))
        check("one recovery, and it freed her",
              rep.recoveries == 1 and rep.recoveries_freed == 1,
              f"{rep.recoveries} tried, {rep.recoveries_freed} freed")
        dirs = [d for d, _ in world.commands]
        check("the recovery backed off first, then turned",
              "backward" in dirs
              and dirs[dirs.index("backward") + 1] in ("left", "right"),
              str(dirs[:12]))
        check("the re-plan came AFTER the recovery, not instead of it",
              rep.replans == 1, f"{rep.replans} re-plans")
        check("and then she got there", rep.arrived, rep.why)
        check("it is logged as STUCK, not as blocked",
              rep.trouble == "stuck" and rep.blocks_seen == 0)
        row = nsm.recent_attempts(1)[0]
        check("the log row says stuck and counts the recovery",
              row["trouble"] == "stuck" and row["stucks"] == 1
              and row["recoveries"] == 1 and row["blocks_seen"] == 0, str(row))
        said = voice.said
        check("she said stuck, then recovered, then arrived — in that order",
              said == [_line("stuck"), _line("recovered", "home"),
                       _line("arrived_home")], str(said))


def test_stuck_is_capped_then_failsafe() -> None:
    print("\n— stuck for good: capped recoveries, then FAILSAFE (step 13) —")
    with TempPlaces():
        nsm.clear_failsafe()
        nsm.set_home((2.0, 2.0, 0.0))
        world = _wedged(frees_on_reverse=False)
        sim = _sim()
        mapper = sim.FakeMapper(world)
        mapper.grid = _grid(free=True)
        robot = _halting_robot(world)
        voice = _quiet_voice()
        c = nsm.NavController(robot=robot, voice=voice)
        rep = asyncio.run(c.go_home(mapper=mapper, max_ticks=80))
        check("it ends in FAILSAFE", rep.state is nsm.NavState.FAILSAFE)
        check(f"it tried to free itself exactly {nsm.MAX_STUCK_RECOVERIES} "
              "times, never more",
              rep.recoveries == nsm.MAX_STUCK_RECOVERIES
              and [d for d, _ in world.commands].count("backward")
              == nsm.MAX_STUCK_RECOVERIES,
              f"{rep.recoveries} recoveries")
        check("a chassis that never moved is never reported freed",
              rep.recoveries_freed == 0
              and _line("recovered", "home") not in voice.said,
              str(voice.said))
        check("FAILSAFE halted the wheels", robot.halted)
        check("it never called being stuck \"blocked\"",
              _line("blocked") not in voice.said and rep.blocks_seen == 0)
        check("it told a person it needs help, last",
              voice.said and voice.said[-1] == _line("failsafe"),
              str(voice.said))
        rec = nsm.failsafe_record()
        check("and left a record for \"try again\"",
              rec is not None and rec["mode"] == "return_to_home"
              and rec["label"] == "home" and rec["goal_xy"] == [2.0, 2.0]
              and rec["recoveries"] == nsm.MAX_STUCK_RECOVERIES, str(rec))
        check("the reason says she tried and it did not work",
              "did not work" in rep.why, rep.why)


def test_try_again_relocalises_resets_and_resumes() -> None:
    print("\n— \"try again\": relocalise, fresh allowance, same goal (17) —")
    sim = _sim()
    real = nsm.relocalise
    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            nsm.write_failsafe({"mode": "return_to_home", "label": "home",
                                "goal_xy": [2.0, 2.0], "why": "stuck",
                                "at": time.time(), "paused_exploration": False,
                                "resume_minutes": None, "recoveries": 2})
            # Somebody picked her up and put her down somewhere else, still
            # wedged against the same kind of thing.
            world = _wedged(pose=(0.8, 1.2, 0.4), frees_on_reverse=True)
            moved = sim.FakeMapper(world)
            moved.grid = _grid(free=True)
            seen = {}

            async def relocalise():
                seen["commands_before"] = len(world.commands)
                seen["recoveries_then"] = c.recoveries
                return moved, ""

            nsm.relocalise = relocalise
            voice = _quiet_voice()
            c = nsm.NavController(robot=_halting_robot(world), voice=voice)
            c.recoveries = nsm.MAX_STUCK_RECOVERIES      # the spent allowance
            rep = asyncio.run(c.retry_failsafe(max_ticks=80,
                                               arrival=_arrive((2.0, 2.0))))
            check("it relocalised BEFORE moving a wheel",
                  seen.get("commands_before") == 0, str(seen))
            check("the recovery counter was reset before anything else",
                  seen.get("recoveries_then") == 0, str(seen))
            check("it drove from where she is NOW, not where she stopped",
                  world.commands and rep.arrived, rep.why)
            check("the fresh allowance was needed and used",
                  rep.recoveries == 1 and rep.recoveries_freed == 1,
                  f"{rep.recoveries} recoveries")
            check("it resumed the mode that failed", rep.destination == "home"
                  and rep.state is nsm.NavState.ARRIVED)
            check("the record is gone once the retry started",
                  nsm.failsafe_record() is None)
            check("the attempt is logged as a retry",
                  nsm.recent_attempts(1)[0]["retried"] is True)
            check("she said she was trying again, first",
                  voice.said[0] == _line("try_again"), str(voice.said[:2]))

            # Nothing waiting.
            rep2 = asyncio.run(nsm.NavController().retry_failsafe())
            check("with nothing waiting it does nothing and says so",
                  "nothing waiting" in rep2.why.lower(), rep2.why)

            # Still lost after the human intervention.
            nsm.write_failsafe({"mode": "target_nav", "label": "the desk",
                                "goal_xy": [1.0, 1.0], "why": "no route",
                                "at": time.time(), "paused_exploration": False,
                                "resume_minutes": None, "recoveries": 1})

            async def lost():
                return None, "I can see the room but cannot match it."

            nsm.relocalise = lost
            quiet = _quiet_voice()
            idle = _sim().FakeWorld()
            rep3 = asyncio.run(nsm.NavController(
                robot=_halting_robot(idle), voice=quiet).retry_failsafe())
            check("if she still cannot place herself she does not drive",
                  rep3.state is nsm.NavState.FAILSAFE and not idle.commands)
            check("and the record is back, still waiting for a person",
                  (nsm.failsafe_record() or {}).get("label") == "the desk")
            check("she says she is lost, then that she needs help",
                  quiet.said[-2:] == [_line("lost"), _line("failsafe")],
                  str(quiet.said))
            nsm.clear_failsafe()
    finally:
        nsm.relocalise = real


def test_try_again_resumes_exploration() -> None:
    print("\n— \"try again\" when exploring is what failed —")
    from mcp_server.tools import autonomy
    sim = _sim()
    real = (nsm.relocalise, autonomy._running, autonomy.stop_exploration,
            autonomy.start_exploration)
    calls = []

    async def relocalise():
        calls.append("relocalise")
        m = sim.FakeMapper(sim.FakeWorld())
        m.grid = _grid(free=True)
        return m, ""

    def start(minutes=None, thinking=None):
        calls.append(("start", minutes))
        return f"exploring for {minutes} minutes"

    try:
        nsm.relocalise = relocalise
        autonomy._running = lambda: False
        autonomy.start_exploration = start
        nsm.write_failsafe({"mode": "exploring", "label": "exploration",
                            "goal_xy": None, "why": "could not restart",
                            "at": time.time(), "paused_exploration": True,
                            "resume_minutes": 7.5, "recoveries": 0})
        rep = asyncio.run(nsm.NavController(
            voice=_quiet_voice()).retry_failsafe())
        check("relocalised first, then restarted the run",
              calls[:2] == ["relocalise", ("start", 7.5)], str(calls))
        check("with the minutes it still had, not a fresh run",
              ("start", 7.5) in calls)
        check("and is exploring again", rep.state is nsm.NavState.EXPLORING)
        check("the record is consumed", nsm.failsafe_record() is None)
    finally:
        (nsm.relocalise, autonomy._running, autonomy.stop_exploration,
         autonomy.start_exploration) = real
        nsm.clear_failsafe()


def test_frontier_watch_debounces() -> None:
    print("\n— frontier exhaustion, and not a momentary zero (step 16) —")
    import os
    tmp = tempfile.TemporaryDirectory()
    path = Path(tmp.name) / "rl_autonomy.json"

    def write(tick, exhausted=True, frontiers=0, error="", age=0.0):
        path.write_text(json.dumps({"TICK": tick, "EXPLORATION_TARGET": {
            "exhausted": exhausted, "frontiers": frontiers, "error": error}}))
        when = time.time() - age
        os.utime(path, (when, when))

    w = nsm.FrontierWatch(path=path)
    write(10)
    check("one reading of \"none left\" is not the end", not w.update(now=0.0))
    for tick, now in ((15, 2.0), (20, 4.0), (25, 6.0)):
        write(tick)
        fired = w.update(now=now)
    check("four readings inside six seconds is still too quick", not fired,
          str(w.as_dict()))
    write(30)
    check("held over enough readings AND enough time, it is the end",
          w.update(now=8.5))

    w = nsm.FrontierWatch(path=path)
    results = []
    for tick, (exhausted, frontiers) in enumerate(
            [(True, 0), (True, 0), (False, 14), (True, 0), (True, 0),
             (True, 0)]):
        write(tick, exhausted, frontiers)
        results.append(w.update(now=tick * 2.0))
    check("a zero between normal readings starts the count again",
          not any(results), str(w.as_dict()))

    w = nsm.FrontierWatch(path=path)
    stale = []
    for tick in range(8):
        write(tick, age=30.0)
        stale.append(w.update(now=tick * 3.0))
    check("a stale file (no run writing it) never triggers", not any(stale))

    w = nsm.FrontierWatch(path=path)
    frozen = []
    for n in range(8):
        write(99)
        frozen.append(w.update(now=n * 3.0))
    check("a tick that is not moving never counts", not any(frozen))

    w = nsm.FrontierWatch(path=path)
    errs = []
    for tick in range(8):
        write(tick, error="no reachable space yet")
        errs.append(w.update(now=tick * 3.0))
    check("a planner that could not plan is not a finished house",
          not any(errs))
    check("a missing file is not an empty house",
          not nsm.FrontierWatch(path=Path(tmp.name) / "nope.json").update())
    tmp.cleanup()


def test_bundle_keeps_data_with_its_map() -> None:
    print("\n— navigation data saved WITH its map (nav_bundle) —")
    import hashlib
    import shutil
    import nav_bundle

    real_map = config.SLAM_MAP_FILE
    before = hashlib.sha256(real_map.read_bytes()).hexdigest() \
        if real_map.exists() else None
    tmp = tempfile.TemporaryDirectory()
    live = Path(tmp.name) / "map.npz"          # a stand-in live map
    if real_map.exists():
        shutil.copy2(real_map, live)
    else:
        live.write_bytes(b"not really a map")
    config.SLAM_MAP_FILE = live
    # Its own state directory: the suite-wide one already holds a bundle with
    # every earlier test's places in its history.
    saved_paths = {k: getattr(nsm, k) for k in
                   ("NAV_LOG", "FAILSAFE_FILE", "EARLY_RETURN_FILE",
                    "PENDING_FILE", "NAV_MODE_FILE")}
    for k, path in saved_paths.items():
        setattr(nsm, k, Path(tmp.name) / "state" / path.name)
    try:
        with TempPlaces():
            folder = nav_bundle.bundle_dir()
            check("the bundle follows the redirected state dir, never the "
                  "real one", str(folder).startswith(str(nsm.NAV_LOG.parent)),
                  str(folder))
            nsm.set_home((1.0, 1.0, 0.0))
            m = nav_bundle.manifest()
            check("saving home saved the bundle", m is not None
                  and m["places"] == ["home"], str(m and m["places"]))
            check("the map inside it IS the map the places belong to",
                  (folder / "map.npz").read_bytes() == live.read_bytes()
                  and json.loads((folder / "places.json").read_text())
                  ["map_sha256"] == m["map_sha256"])

            nsm.write_failsafe({"mode": "target_nav", "label": "home",
                                "goal_xy": [1, 1], "why": "t",
                                "at": time.time()})
            check("a FAILSAFE record is saved with it",
                  (folder / "nav_failsafe.json").exists())
            nsm.clear_failsafe()
            check("and cleared from it once no longer waiting",
                  not (folder / "nav_failsafe.json").exists())
            check("records and trips alone do not pile up history",
                  not (folder / "history").exists()
                  or not any((folder / "history").iterdir()))

            from slam import places
            places.remember_here("the kitchen", (2.0, 0.5, 0.0))
            nav_bundle.save("place saved: the kitchen")
            hist = sorted((folder / "history").iterdir())
            check("a new place archives the old pair first",
                  len(hist) == 1 and json.loads(
                      (hist[0] / "places.json").read_text())["places"][0]
                  ["name"] == "home", str([h.name for h in hist]))
            check("and the bundle now holds both places",
                  nav_bundle.manifest()["places"] == ["home", "the kitchen"])

            live.write_bytes(live.read_bytes() + b"-changed")
            st = nav_bundle.status()
            check("a changed live map is noticed", not st["same_map_as_live"])
            said = nav_bundle.restore()
            check("restore refuses to put places onto a different map",
                  "--force" in said, said)
            changed = live.read_bytes()
            said = nav_bundle.restore(force=True)
            check("with --force it puts the bundle's map back",
                  live.read_bytes() == (folder / "map.npz").read_bytes(), said)
            kept = [h for h in (folder / "history").iterdir()
                    if h.name.endswith("live-before-restore")]
            check("after archiving the live map it replaced",
                  kept and (kept[0] / "map.npz").read_bytes() == changed)
    finally:
        config.SLAM_MAP_FILE = real_map
        for k, path in saved_paths.items():
            setattr(nsm, k, path)
        tmp.cleanup()
    after = hashlib.sha256(real_map.read_bytes()).hexdigest() \
        if real_map.exists() else None
    check("the REAL map was never written", before == after)


def test_refused_restart_is_not_a_run() -> None:
    print("\n— a refused exploration restart is not a run —")
    check("JSON started:false is a failure",
          nsm.start_failed('{"started": false, "why": "I am already '
                           'exploring. Say stop to end it."}'))
    check("JSON started:true is not",
          not nsm.start_failed('{"started": true, "minutes": 5}'))
    check("the old wording still counts",
          nsm.start_failed("FAILED to start") and nsm.start_failed(
              "could not restart: boom"))
    from mcp_server.tools import autonomy
    real = autonomy.start_exploration
    autonomy.start_exploration = lambda minutes=None, thinking=None: (
        '{"started": false, "why": "systemctl refused"}')
    try:
        c = nsm.NavController(voice=_quiet_voice())
        c.snapshot = nsm.ExplorationSnapshot(remaining_minutes=5.0)
        out = c.resume_exploration()
        check("resume says it did not resume",
              out["resumed"] is False and c.state is nsm.NavState.FAILSAFE,
              str(out))
        check("and leaves a record so \"try again\" can restart it",
              (nsm.failsafe_record() or {}).get("mode") == "exploring")
    finally:
        autonomy.start_exploration = real
        nsm.clear_failsafe()


def test_already_there_is_arrived() -> None:
    print("\n— already at the goal is an arrival, not \"no route\" —")
    sim = _sim()
    with TempPlaces():
        nsm.set_home((1.0, 1.0, 0.0))
        world = sim.FakeWorld(pose=(1.05, 0.98, 2.0))
        mapper = sim.FakeMapper(world)
        mapper.grid = _grid(free=True)
        voice = _quiet_voice()
        rep = asyncio.run(nsm.NavController(
            robot=_halting_robot(world), voice=voice).go_home(mapper=mapper))
        check("she is home without moving",
              rep.arrived and rep.state is nsm.NavState.ARRIVED
              and not world.commands, rep.why)
        check("and says so, not that there is no route",
              voice.said == [_line("arrived_home")], str(voice.said))
        check("nothing is left waiting for help", nsm.failsafe_record() is None)


def test_lost_on_the_map_looks_again() -> None:
    print("\n— lost on the map she is driving on: search it again, carry on —")
    from types import SimpleNamespace
    sim = _sim()

    class Flaky(sim.FakeMapper):
        """Loses the match once, on its first look AFTER A MOVE -- as after a
        turn that did not happen the way it was commanded (2026-09-24 23:55).
        After a move, not standing still: since 2026-09-25 a standstill sweep
        keeps the last verified pose, because nothing moved."""

        lost_for = None

        def update(self, scan, moved_m=0.0, turned_rad=0.0, now=None):
            out = super().update(scan, moved_m, turned_rad, now)
            # Lost on the sweep after the move AND the look after it: a
            # post-move miss the next look confirms is merely a fresh fix.
            if (moved_m or turned_rad) and self.lost_for is None:
                self.lost_for = 2
            if self.lost_for:
                self.lost_for -= 1
                self.pose_trusted = False
            else:
                self.pose_trusted = True
            return out

    real = nsm.locate
    calls = []

    async def found(robot, grid=None, voice=None, allow_moves=True):
        calls.append(grid)
        return SimpleNamespace(pose=world.measured(), pose_trusted=True), ""

    try:
        with TempPlaces():
            nsm.set_home((2.0, 2.0, 0.0))
            world = sim.FakeWorld(pose=(0.5, 2.0, 0.0), cm_s=45.0, deg_s=75.0)
            mapper = Flaky(world)
            mapper.grid = _grid(free=True)
            nsm.locate = found
            voice = _quiet_voice()
            rep = asyncio.run(nsm.NavController(
                robot=_halting_robot(world), voice=voice).go_home(
                mapper=mapper, max_ticks=80, arrival=_arrive((2.0, 2.0))))
            check("she searched the SAME map again, once",
                  len(calls) == 1 and calls[0] is mapper.grid, str(len(calls)))
            check("and carried on home", rep.arrived, rep.why)
            check("saying she lost her place, then found it",
                  _in_order(voice.said, [_line("refinding"),
                                         _line("refound", "home"),
                                         _line("arrived_home")]),
                  str(voice.said))

            async def not_found(robot, grid=None, voice=None,
                                allow_moves=True):
                calls.append(grid)
                return None, "two places on my map still look alike"

            nsm.locate = not_found
            calls.clear()
            world2 = sim.FakeWorld(pose=(0.5, 2.0, 0.0), cm_s=45.0, deg_s=75.0)
            mapper2 = Flaky(world2)
            mapper2.grid = _grid(free=True)
            robot2 = _halting_robot(world2)
            rep2 = asyncio.run(nsm.NavController(
                robot=robot2, voice=_quiet_voice()).go_home(
                mapper=mapper2, max_ticks=80, arrival=_arrive((2.0, 2.0))))
            check("if she still cannot find herself, FAILSAFE — not a guess",
                  rep2.state is nsm.NavState.FAILSAFE and robot2.halted,
                  rep2.why[:80])
            check("and the search is capped",
                  len(calls) <= nsm.MAX_REFINDS, f"{len(calls)} searches")
            nsm.clear_failsafe()
    finally:
        nsm.locate = real


def _in_order(said, lines) -> bool:
    it = iter(said)
    return all(any(s == line for s in it) for line in lines)


def test_offline_tools_registered() -> None:
    print("\n— the offline navigation tools —")
    from mcp_server import registry
    registry.load_tool_modules()
    names = {name for _, name, _ in registry.REGISTRY}
    check("nav_go, nav_home, nav_come_back and try_again are tools",
          {"nav_go", "nav_home", "nav_come_back", "try_again"} <= names)
    check("they are all fresh-order gated like every other move",
          {"nav_go", "nav_home", "nav_come_back", "try_again",
           "explore_and_return"} <= set(config.COMMAND_FRESH_TOOLS))


def main() -> int:
    print("nav_state_machine.py + landmarks.py + nav_tools.py — file 14 steps 6-11")
    test_timer()
    test_state_transitions()
    test_home_is_a_place()
    test_run_refuses_rather_than_guessing()
    test_run_plans_and_follows()
    test_area_watch()
    test_area_watch_survives_map_growth()
    test_inside_known_place()
    test_naming_flow()
    test_no_answer_saves_nothing()
    test_no_ask_means_no_save()
    test_overwrite_needs_confirmation()
    test_yes_and_name_parsing()
    test_map_session_rotates_when_the_map_is_rebuilt()
    test_resolve_never_guesses()
    test_target_nav_refuses_unknown()
    test_both_modes_share_one_engine()
    test_action_is_returned_not_performed()
    test_replan_only_when_replanning_can_help()
    test_replan_loop_retries_and_gives_up()
    test_snapshot_is_honest_about_what_it_keeps()
    test_tools_are_registered()
    test_runtime_obstacles_stay_out_of_the_map()
    test_block_classification()
    test_obstacle_position()
    test_goal_cell_occupied()
    test_transient_block_does_not_storm()
    test_persistent_block_is_remembered_and_planned_around()
    test_nav_log_round_trip()
    test_dashboard_panel_is_read_only()
    test_round_trip_mode_persists()
    test_round_trip_goes_and_comes_back()
    test_round_trip_off_means_stay()
    test_getting_there_survives_failing_to_come_back()
    test_armed_return_for_exploration()
    test_armed_return_drives_back()
    test_round_trip_tools_registered()
    test_voice_lines_match_the_exploration()
    test_stuck_recovers_before_replanning()
    test_stuck_is_capped_then_failsafe()
    test_try_again_relocalises_resets_and_resumes()
    test_try_again_resumes_exploration()
    test_frontier_watch_debounces()
    test_bundle_keeps_data_with_its_map()
    test_refused_restart_is_not_a_run()
    test_lost_on_the_map_looks_again()
    test_already_there_is_arrived()
    test_offline_tools_registered()
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
