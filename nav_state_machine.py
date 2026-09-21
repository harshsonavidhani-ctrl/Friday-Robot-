"""nav_state_machine.py — file 14 steps 6, 9 and 10: the shared navigation machine.

                    ┌──────────────── timer expires ──────────┐
                    │                                         v
    IDLE ──> EXPLORING ──(a target-nav command)──> TARGET_NAV / RETURN_TO_HOME
                ^                                        │
                │                                        ├──> ARRIVED
          resume_exploration()                           ├──> BLOCKED_REPLANNING ──┐
                │                                        └──> FAILSAFE (halts)     │
                └────────────────────────────────────────────────────┘<────────────┘

WHAT THIS OWNS. The six states, the home timer, the re-plan loop, the runtime
obstacle layer, and pause/resume of exploration. It owns no geometry and does
no driving of its own: `path_planner` plans, `nav_follower` drives, and both
of those go out through `safety.guarded_move` like everything else.

⚠ RETURN_TO_HOME AND TARGET_NAV ARE THE SAME ENGINE. `_drive_to(goal, label,
mode)` is the whole of it; the two differ only in what triggers entry and what
happens on arrival. If a third destination ever needs navigating to, it calls
`_drive_to` — it does not get a second copy of it.

⚠ IT NEVER TRUSTS A STORED POSE, AND THIS IS THE WHOLE POINT OF STEP 6.
Friday's map persists and grows across runs, but her POSITION in it does not:
`state/map.npz` carries no pose, `map_here` starts every run at (0, 0, 0), and
somebody may well have picked her up between runs. A stored coordinate is
therefore a fact about where she WAS, and the only way to turn it into a fact
about where she IS is to look at the room and match it against the map. So
every run begins by relocalising, and REFUSES if the match fails rather than
falling back on the last known pose. See
[[friday-map-relocalisation-ratchet]] for what the alternative cost the RL
stack, and `mcp_server/tools/places_tools.py`, whose `_placed_mapper()` does
exactly this job already and is reused here rather than rewritten.

WHERE HOME IS. A row in the `places` table called "home" -- the same store
`remember_place` / `go_to` / `list_places` already use, so home is a landmark
like any other and every existing tool can see it. It is NOT `Mapper.home`,
which is only wherever this run happened to start, and it is not a separate
file that could disagree with the places table.

⚠ THE TIMER IS NOT A SAFETY MECHANISM. It expiring means "start driving
home", not "stop". Stopping is `stop()` and the reflex, which are unaffected
by anything here. FAILSAFE is the one that halts, and it waits for a human
rather than retrying past the replanning budget.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

import config
import nav_follower
import path_planner

log = logging.getLogger("friday.nav_state")

# The place row that means home. Lower case because `places.remember_here`
# normalises names that way, so "Home" and "home" are the same row.
HOME_PLACE_NAME = "home"

# Configurable, and a default rather than a rule: `ReturnHome(minutes=...)`
# overrides it, and so does `HomeTimer.minutes` at runtime. Twenty minutes is
# roughly the exploration run length that has been useful so far.
RETURN_AFTER_MIN = 20.0

# Below this there is no point resuming: an exploration run of twelve seconds
# costs a process start, a map load and a relocalisation to do nothing. If the
# interrupt used up what was left of the time, the timer has effectively
# expired and the right next move is to go home, not to explore for no time.
RESUME_MIN_MINUTES = 0.5

# --- round trip: go and do the thing, then come back here ------------------
#
# Harsh, 2026-09-21: "friday go explore the home it explores and like come to
# the position where it started and like then i commanded it go to the kitchen
# then it should go to the kitchen and return to where it started".
#
# ⚠ "WHERE IT STARTED" IS NOT HOME, and conflating the two would be the easy
# mistake. `home` is a place somebody deliberately named and it stays put for
# weeks. The start of a round trip is wherever she happened to be standing
# when the command arrived -- the middle of the hall, next to his chair -- and
# it is valid for exactly that one errand. So it is captured per run, in the
# map's frame, from a RELOCALISED pose, and it is never written to the places
# table where it would pollute the names she knows.
NAV_MODE_FILE = config.ROOT / "state" / "nav_mode.json"

# Coming back is a second drive with its own budget, not a continuation of the
# first. Half the outbound budget: the way back is over ground she has just
# driven, so if it needs more than that something has changed and stopping to
# say so beats grinding.
RETURN_BUDGET_FRACTION = 0.5

# She must actually have gone somewhere before coming back is meaningful.
# Under this and the round trip is skipped -- the errand ended where it began.
ROUND_TRIP_MIN_M = 0.40


def round_trip_enabled() -> bool:
    """Is round-trip mode on? Persisted, so voice and dashboard agree."""
    try:
        return bool(json.loads(NAV_MODE_FILE.read_text()).get("round_trip"))
    except (OSError, ValueError, AttributeError):
        return False


def set_round_trip(on: bool) -> dict:
    """Turn round-trip mode on or off. Survives a restart."""
    state = {"round_trip": bool(on), "changed_at": time.time()}
    try:
        NAV_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = NAV_MODE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(NAV_MODE_FILE)
    except OSError as exc:                           # noqa: BLE001
        log.warning("could not write %s: %s", NAV_MODE_FILE, exc)
        return {"round_trip": bool(on), "saved": False, "why": str(exc)}
    log.info("round trip mode %s", "ON" if on else "off")
    return {"round_trip": bool(on), "saved": True}


class NavState(str, Enum):
    """Where the navigation half of the robot currently thinks it is.

    A str Enum so it serialises into telemetry and the dashboard without a
    converter, and compares equal to its own name in a JSON blob.

    RETURN_TO_HOME and TARGET_NAV are the same engine. They differ only in
    what put the robot into them and what happens when it gets there, which
    is why `_drive_to` takes both as a parameter rather than existing twice.
    """

    IDLE = "idle"
    EXPLORING = "exploring"
    RETURN_TO_HOME = "return_to_home"
    TARGET_NAV = "target_nav"
    BLOCKED_REPLANNING = "blocked_replanning"
    RETURNING_TO_START = "returning_to_start"
    ARRIVED = "arrived"
    FAILSAFE = "failsafe"


@dataclass
class HomeTimer:
    """How long she may keep exploring before she is sent home.

    Counts wall clock from `start()`, not ticks: a run that is stalled against
    a wall is exactly the run that most needs the timer to fire, and a
    tick-counting timer stops counting when the robot stops moving.
    """

    minutes: float = RETURN_AFTER_MIN
    started_at: float | None = None

    def start(self, now: float | None = None) -> None:
        self.started_at = time.monotonic() if now is None else now

    def stop(self) -> None:
        self.started_at = None

    @property
    def running(self) -> bool:
        return self.started_at is not None

    def elapsed_s(self, now: float | None = None) -> float:
        if self.started_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, now - self.started_at)

    def remaining_s(self, now: float | None = None) -> float:
        if self.started_at is None:
            return float("inf")
        return max(0.0, self.minutes * 60.0 - self.elapsed_s(now))

    def expired(self, now: float | None = None) -> bool:
        return self.running and self.remaining_s(now) <= 0.0


def home_place() -> dict | None:
    """The stored home, or None if nobody has set one."""
    from slam import places
    return places.find(HOME_PLACE_NAME)


def set_home(pose, note: str | None = None) -> dict:
    """Name the robot's current position home, in the map's frame.

    The caller must pass a RELOCALISED pose. This deliberately does not fetch
    one itself: saving home from an unverified pose is the one mistake that
    poisons every later return, because from then on she drives confidently to
    the wrong address and nothing about the run looks wrong.
    """
    from slam import places
    return places.remember_here(HOME_PLACE_NAME, pose, kind="home", note=note)


async def relocalise():
    """Where am I, really? Returns (mapper, "") or (None, why-not).

    This is `places_tools._placed_mapper()`, unchanged and unwrapped: it loads
    the saved map, folds `config.MISSION_SETTLE_SWEEPS` sweeps in from a
    standstill, and refuses if the scan matcher cannot place her. Reused
    rather than reimplemented so that "where am I" has exactly one answer on
    this robot, and `go_to` and return-to-home cannot drift apart.
    """
    from mcp_server.tools.places_tools import _placed_mapper
    return await _placed_mapper()


@dataclass
class ReturnReport:
    state: NavState = NavState.IDLE
    arrived: bool = False
    why: str = ""
    destination: str = "home"
    action: str = ""            # queued for the brain to perform on arrival
    attempts: int = 0
    replans: int = 0
    goal_moved_m: float = 0.0
    seconds: float = 0.0
    at: float = 0.0
    round_trip: bool = False
    returned_to_start: bool = False
    start_pose: list[float] = field(default_factory=list)
    inbound: dict = field(default_factory=dict)
    blocks_seen: int = 0
    blocks_confirmed: int = 0
    obstacles: dict = field(default_factory=dict)
    home: list[float] = field(default_factory=list)
    pose: list[float] = field(default_factory=list)
    waypoints: int = 0
    planned_m: float = 0.0
    follow: dict = field(default_factory=dict)
    grid: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["state"] = self.state.value
        d["planned_m"] = round(self.planned_m, 2)
        return d


class NavController:
    """The shared state machine: one engine, several reasons to run it.

    Deliberately NOT a thread and not a loop that owns the robot. It is driven
    by whoever is already running -- the exploration run, the dashboard, an MCP
    tool -- calling `tick()` to learn whether the timer has fired and then
    awaiting `run()`. Anything that owned a thread here would be a second thing
    competing for the hardware lease with the exploration run that is already
    holding it.
    """

    def __init__(self, robot=None, minutes: float = RETURN_AFTER_MIN,
                 timer: HomeTimer | None = None) -> None:
        self.robot = robot
        self.timer = timer or HomeTimer(minutes=minutes)
        self.state = NavState.IDLE
        self.snapshot: ExplorationSnapshot | None = None
        self.obstacles = RuntimeObstacles()

    # ---- the state half ---------------------------------------------------

    def exploring(self, now: float | None = None) -> None:
        """Called when an exploration run starts. Starts the clock."""
        self.state = NavState.EXPLORING
        self.timer.start(now)

    def tick(self, now: float | None = None) -> NavState:
        """Advance the state machine. The only transition it makes by itself.

        EXPLORING + expired timer -> RETURNING. Nothing else moves on its own:
        arriving and failing are decided by `run()`, which knows what actually
        happened, and IDLE is where a caller puts it.
        """
        if self.state is NavState.EXPLORING and self.timer.expired(now):
            log.info("home timer expired after %.1f min", self.timer.minutes)
            self.state = NavState.RETURN_TO_HOME
        return self.state

    def status(self, now: float | None = None) -> dict:
        return {
            "state": self.state.value,
            "timer_running": self.timer.running,
            "minutes": self.timer.minutes,
            "remaining_s": (None if not self.timer.running
                            else round(self.timer.remaining_s(now), 1)),
            "home": home_place(),
        }

    # ---- the driving half -------------------------------------------------

    # ---- pause / resume (step 9) ------------------------------------------

    def pause_exploration(self) -> dict:
        """Snapshot what exploration is doing, then stop the run.

        ⚠ WHAT CAN AND CANNOT BE RESTORED, stated plainly because the
        difference decides the design. The exploration run is a SEPARATE
        PROCESS (`stop_exploration`'s own docstring says so), and its frontier
        list lives in that process's memory. Reaching into it would mean
        editing `rl/` or `exploration_mode.py`, which are read-only to this
        file.

        That turns out to matter much less than it sounds, because THE
        FRONTIER LIST IS NOT STATE. It is a pure function of the occupancy
        grid and the pose -- `slam/frontier.py::find(grid, distances, pose)`
        computes it from scratch every cycle, and the grid is saved when the
        run ends. So resuming recomputes the same frontiers from the same map
        rather than restoring a stale copy of them, which is both simpler and
        more correct: a frontier snapshotted ten minutes ago may have been
        explored by the target-nav drive that interrupted it.

        What genuinely is lost is the heuristic state: which frontier it had
        committed to, and the blacklist of ones it gave up on. Those are
        captured here from telemetry and handed back as a HINT, and the honest
        position is that nothing currently consumes that hint -- doing so
        would mean changing the explorer. It is recorded so the loss is
        visible in the report rather than silent.
        """
        self.snapshot = ExplorationSnapshot.capture(
            remaining_minutes=(self.timer.remaining_s() / 60.0
                               if self.timer.running else None))
        # ⚠ CALL THE PLAIN FUNCTION, NOT A `.__wrapped__` OF IT.
        # `registry.tool` registers an async `invoke` in REGISTRY and then
        # `return fn` — the module attribute stays the ORIGINAL sync function
        # and has no `__wrapped__`. An earlier version here reached for one,
        # which raised AttributeError straight into the except below and
        # reported "could not stop exploration" while the run carried on
        # driving. A pause that does not pause is worse than no pause: the
        # exploration process and the navigation run would both be steering.
        stopped = ""
        try:
            from mcp_server.tools import autonomy
            if autonomy._running():
                stopped = autonomy.stop_exploration()
                if not _stopped_ok(stopped):
                    log.error("pause did NOT stop exploration: %s", stopped)
            else:
                stopped = "no run was out"
        except Exception as exc:                 # noqa: BLE001
            stopped = f"could not stop exploration: {exc}"
            log.error("%s", stopped)
        out = self.snapshot.as_dict()
        out["stopped"] = stopped
        log.info("paused exploration: %s", out)
        return out

    @staticmethod
    def _paused_ok(said: str) -> bool:
        return _stopped_ok(said)

    def resume_exploration(self) -> dict:
        """Restart exploration for whatever was left of its time."""
        if self.snapshot is None:
            return {"resumed": False, "why": "nothing was paused"}
        minutes = self.snapshot.remaining_minutes
        if minutes is not None and minutes < RESUME_MIN_MINUTES:
            self.snapshot = None
            self.state = NavState.RETURN_TO_HOME
            return {"resumed": False,
                    "why": (f"only {minutes * 60:.0f} s of the exploration "
                            "budget was left when I was interrupted — that is "
                            "not worth starting a run for, so the timer counts "
                            "as expired and I should go home instead."),
                    "state": self.state.value}
        said = ""
        try:
            from mcp_server.tools import autonomy
            said = autonomy.start_exploration(minutes=minutes)
        except Exception as exc:                 # noqa: BLE001
            return {"resumed": False, "why": f"could not restart: {exc}"}
        self.state = NavState.EXPLORING
        if minutes:
            self.timer.minutes = minutes
        self.timer.start()
        out = {"resumed": True, "minutes": minutes, "said": said,
               "recomputed_from_map": True,
               "hint": self.snapshot.as_dict()}
        self.snapshot = None
        log.info("resumed exploration: %s", out)
        return out

    # ---- the driving half -------------------------------------------------

    async def go_home(self, mapper=None, **kw) -> ReturnReport:
        """RETURN_TO_HOME: the timer fired, or somebody asked."""
        home = home_place()
        if home is None:
            report = ReturnReport(state=NavState.FAILSAFE, destination="home",
                                  why="I do not know where home is. Stand "
                                      "where you want home to be and say "
                                      "\u201cremember this as home\u201d.")
            self.state = report.state
            return report
        report = await self._drive_to((home["x"], home["y"]), "home",
                                      NavState.RETURN_TO_HOME, mapper, **kw)
        report.home = [round(home["x"], 3), round(home["y"], 3)]
        if report.arrived:
            from slam import places
            places.visited(HOME_PLACE_NAME)
        return report

    async def go_to_landmark(self, name: str, action: str = "",
                             mapper=None, return_to_start: bool | None = None,
                             **kw) -> ReturnReport:
        """TARGET_NAV: drive to a landmark she already knows by name.

        The landmark must be one she KNOWS. No fuzzy match, no nearest guess:
        see `resolve_landmark`.

        `return_to_start` makes it a ROUND TRIP: remember where she is
        standing now, drive there, and come back to this spot afterwards.
        None means "whatever the mode says" (`round_trip_enabled()`), so the
        dashboard toggle and a spoken "and come back" reach the same place.
        """
        row, why = resolve_landmark(name)
        if row is None:
            report = ReturnReport(state=NavState.FAILSAFE, destination=name,
                                  why=why, action=action)
            self.state = report.state
            return report

        if mapper is None:
            mapper, whynot = await relocalise()
            if mapper is None:
                report = ReturnReport(state=NavState.FAILSAFE,
                                      destination=row["name"], action=action,
                                      why=f"I am not setting off: {whynot}")
                self.state = report.state
                return report

        # ⚠ CAPTURED BEFORE THE FIRST WHEEL TURNS, from the relocalised pose.
        # Read it back afterwards and it is wherever she ENDED, which is the
        # one place she does not need directions to.
        start = tuple(float(v) for v in mapper.pose)

        report = await self._drive_to((row["x"], row["y"]), row["name"],
                                      NavState.TARGET_NAV, mapper, **kw)
        report.action = action
        if report.arrived:
            from slam import places
            places.visited(row["name"])

        wants = (round_trip_enabled() if return_to_start is None
                 else bool(return_to_start))
        report.round_trip = wants
        report.start_pose = [round(v, 3) for v in start]
        if wants and report.arrived:
            await self._return_to_start(mapper, start, report, **kw)
        return report

    async def return_to_armed_start(self, mapper=None, **kw) -> ReturnReport:
        """Drive back to a start pose armed earlier by something else.

        This is the second half of "go and explore, then come back": the
        exploration run is a separate process, so the leg that started the
        errand and the leg that finishes it do not share memory. The pose was
        written down by `arm_return_to_start`; this picks it up, and CLEARS IT
        FIRST so that a failure cannot leave it armed to fire again later at a
        pose that is by then even staler.
        """
        row = pending_return()
        if row is None:
            report = ReturnReport(state=self.state,
                                  why="nothing is waiting to be returned from")
            return report
        clear_pending()
        report = await self._drive_to((row["x"], row["y"]),
                                      "where I set off from",
                                      NavState.RETURNING_TO_START, mapper, **kw)
        report.round_trip = True
        report.returned_to_start = report.arrived
        report.start_pose = [round(row["x"], 3), round(row["y"], 3)]
        return report

    async def _return_to_start(self, mapper, start, report: ReturnReport,
                               **kw) -> None:
        """The way back. A second drive, with its own budget and its own state.

        Deliberately runs AFTER the outbound leg is reported as arrived rather
        than being folded into it, because the two have different meanings.
        "I got to the kitchen but could not get back" is a true and useful
        sentence, and a single merged result could not say it -- it would
        report the whole errand as a failure and lose the fact that the thing
        he asked for actually happened.
        """
        went = math.hypot(mapper.pose[0] - start[0], mapper.pose[1] - start[1])
        if went < ROUND_TRIP_MIN_M:
            report.returned_to_start = True
            report.inbound = {"skipped": f"only went {went:.2f} m — already here"}
            return

        budget = kw.pop("budget_m", config.GOTO_BUDGET_M) * RETURN_BUDGET_FRACTION
        kw.pop("budget_s", None)

        # ⚠ THE OUTBOUND ARRIVAL MONITOR IS FOR THE OUTBOUND GOAL. Passing it
        # through would have the way back declaring "arrived" the moment she
        # is near the KITCHEN, which is where she already is -- so she would
        # report coming home without moving. Its SETTINGS are worth keeping
        # (a caller who loosened the dwell meant it for both legs); its goal
        # is not.
        outbound = kw.pop("arrival", None)
        if outbound is not None:
            kw["arrival"] = nav_follower.ArrivalMonitor(
                start[:2], cell_m=mapper.grid.cell,
                tolerance_m=outbound.tolerance, loose_m=outbound.loose,
                dwell_s=outbound.dwell_s, min_samples=outbound.min_samples,
                settle_extra_s=outbound.settle_extra_s)
        back = await self._drive_to(start[:2], "where I set off from",
                                    NavState.RETURNING_TO_START, mapper,
                                    budget_m=max(budget, 1.0), **kw)
        report.returned_to_start = back.arrived
        report.inbound = back.as_dict()
        # The ERRAND's state is what the caller asked about. Coming back
        # failing does not un-arrive her at the kitchen.
        self.state = (NavState.ARRIVED if back.arrived
                      else NavState.FAILSAFE)
        if not back.arrived:
            report.why = (f"{report.why} — but I could not get back to where "
                          f"I set off from: {back.why}")

    async def _drive_to(self, goal_xy, label: str, mode: NavState,
                        mapper=None, max_attempts: int = 3,
                        **follow_kw) -> ReturnReport:
        """The engine both modes share: relocalise, plan, follow, re-plan.

        ⚠ WHY THERE IS A RE-PLAN LOOP AT ALL. One plan followed to its end is
        a plan made against the map as it was BEFORE the drive, and the map is
        the thing that changes most while she drives. `slam/navigate.py`
        solves this by re-planning after every leg; the follower here drives a
        whole smoothed path, so the equivalent is to re-plan when following it
        did not finish the job. The grid is re-read from the mapper each
        attempt, so attempt two is planned on everything attempt one saw.

        NOT EVERY FAILURE IS WORTH RETRYING, and getting that wrong turns one
        problem into a robot repeating it. A human stop, a refusal from the
        guard, a drivetrain that is not driving and a pose that cannot be
        matched are all conditions another identical attempt cannot improve,
        so they go straight to FAILSAFE. Only "no route" and "ran out of path"
        are re-planned, because those are the ones a changed map can fix.
        """
        report = ReturnReport(state=mode, destination=label)
        report.at = time.time()
        started = time.monotonic()
        self.state = mode

        if mapper is None:
            mapper, why = await relocalise()
            if mapper is None:
                report.state, report.why = NavState.FAILSAFE, (
                    f"I am not setting off: {why}")
                self.state = report.state
                return report
        report.pose = [round(float(v), 3) for v in mapper.pose]

        robot = self.robot or nav_follower.GuardedRobot()
        follower = nav_follower.Follower(robot)

        for attempt in range(1, max_attempts + 1):
            report.attempts = attempt
            if attempt > 1:
                self.state = report.state = NavState.BLOCKED_REPLANNING
                report.replans += 1

            # The runtime layer is handed to the planner as extra blocked
            # world points. It inflates exactly like a wall, and it is the
            # ONLY thing that makes this plan differ from the last one when
            # the map itself has not changed.
            blocked = self.obstacles.points()
            prep = path_planner.preprocess(mapper.grid, extra_blocked=blocked)
            report.grid = prep.stats()
            report.obstacles = self.obstacles.as_dict()

            ix, iy = prep.grid.cells_of(
                np.array([[mapper.pose[0], mapper.pose[1]],
                          [goal_xy[0], goal_xy[1]]]))
            # ⚠ TWO DIFFERENT FAILURES THAT MUST NOT SHARE A SENTENCE. "There
            # is nowhere on this map I can plan through at all" and "the goal
            # specifically is not somewhere I fit" have different causes and
            # different fixes, and an early version reported the first as
            # "the nearest spot is 0.0 m away", which is not a fact about
            # anything.
            if not prep.plannable.any():
                report.why = (
                    "there is nowhere on my map I can plan a route through — "
                    "either almost nothing is mapped yet, or everything free "
                    "is within "
                    f"{prep.inflate_m:.2f} m of something solid. Map more "
                    "first.")
                if attempt >= max_attempts:
                    break
                continue

            goal_cell, moved = snap_goal(prep, (int(ix[1]), int(iy[1])))
            report.goal_moved_m = round(moved, 2)
            if goal_cell is None:
                report.why = (
                    f"{label} is not somewhere I can stand, and the nearest "
                    f"place I could is {moved:.1f} m away — too far to call it "
                    f"{label}.")
                if attempt >= max_attempts:
                    break
                continue

            cells, prep = path_planner.plan(prep, (int(ix[0]), int(iy[0])),
                                            goal_cell)
            if len(cells) < 2:
                report.why = (f"I can see where {label} is but nothing "
                              "passable connects here to it on the map as it "
                              "stands. I am not going to drive in its general "
                              "direction.")
                if attempt >= max_attempts:
                    break
                continue

            world = path_planner.cells_to_world(cells, prep.grid)
            report.waypoints = len(world)
            report.planned_m = float(np.sum(np.hypot(np.diff(world[:, 0]),
                                                     np.diff(world[:, 1]))))

            result = await follower.follow(mapper, world, goal_xy=goal_xy,
                                           **follow_kw)
            report.follow = result.as_dict()
            report.pose = result.pose or report.pose
            report.arrived = result.arrived
            report.why = result.why
            if result.arrived:
                report.state = NavState.ARRIVED
                self.state = report.state
                report.seconds = round(time.monotonic() - started, 1)
                log.info("%s: arrived in %d attempt(s)", label, attempt)
                _log_attempt(report)
                return report

            # ---- step 10: what KIND of stop was that? ---------------------
            if is_block(result.why):
                report.blocks_seen += 1
                self.state = report.state = NavState.BLOCKED_REPLANNING
                still, clearance = await self._still_blocked(robot)
                if not still:
                    # It walked away. Re-plan on the unchanged map, which is
                    # nearly free, rather than recording a ghost obstacle.
                    log.info("%s: the block cleared on recheck", label)
                    continue
                report.blocks_confirmed += 1
                self.obstacles.add(obstacle_ahead(mapper.pose, clearance))
                report.obstacles = self.obstacles.as_dict()
                log.info("%s: block confirmed, %d runtime obstacle(s)",
                         label, len(self.obstacles))
                continue

            if _is_final(result.why):
                break

        # ---- FAILSAFE: stop, and wait for a person ------------------------
        # No further autonomous attempt beyond the replanning budget above.
        # The halt is belt and braces -- the follower is not commanding
        # anything by this point -- but "FAILSAFE means the wheels are off"
        # should be true because something made it true, not because nothing
        # happened to be running.
        report.seconds = round(time.monotonic() - started, 1)
        report.state = NavState.FAILSAFE
        self.state = report.state
        try:
            halt = getattr(robot, "halt", None)
            if halt is not None:
                await halt()
        except Exception as exc:                 # noqa: BLE001
            log.warning("could not halt on failsafe: %s", exc)
        log.info("%s: %s", label, report.as_dict())
        _log_attempt(report)
        return report

    async def _still_blocked(self, robot) -> tuple[bool, float | None]:
        """Wait, look again, and say whether the thing is still there.

        Two rechecks a couple of seconds apart, because the case being
        separated out is a person walking past, and a person walking past is
        gone by the second one. A chair is not.
        """
        clearance = None
        for _ in range(BLOCK_RECHECKS):
            if BLOCK_RECHECK_S:
                await asyncio.sleep(BLOCK_RECHECK_S)
            try:
                await robot.observe()
                clearance, blocked = await robot.clearance()
            except Exception as exc:             # noqa: BLE001
                log.warning("could not recheck clearance: %s", exc)
                return True, clearance
            if not blocked:
                return False, clearance
        return True, clearance


def _stopped_ok(said: str) -> bool:
    """Did `stop_exploration` actually stop something?

    Its own failure mode is a readable string, not an exception, so the only
    way to know is to read it. Anything that says it failed counts as not
    stopped — and a pause that silently did not pause leaves two things
    driving the same robot.
    """
    low = (said or "").lower()
    if not low:
        return False
    return not any(word in low for word in
                   ("failed", "could not", "error", "refused"))


# --- step 10: being blocked, and what kind of blocked it is ----------------

# ⚠ DO NOT REPLAN THE INSTANT SOMETHING APPEARS. Harsh walking across the
# LiDAR is gone again in under a second, and a planner that re-plans on sight
# of him produces a replanning storm: every plan is invalidated by the same
# person, each replan costs a Dijkstra and a sweep, and the robot stands there
# recomputing instead of waiting two seconds for him to pass. So a block is
# assumed TRANSIENT until it has survived a recheck.
BLOCK_RECHECK_S = 2.0
BLOCK_RECHECKS = 2

# How long a confirmed obstacle stays in the runtime layer. Long enough to
# plan around it several times, short enough that a chair that was moved does
# not haunt the map for the rest of the run. It is never written to the SLAM
# map -- see RuntimeObstacles.
BLOCK_TTL_S = 90.0

# How far out to look for a free cell when the goal itself is not plannable.
# Small on purpose: a goal that has to move a metre is not that goal any more.
GOAL_SNAP_M = 0.60


# Refusals that mean "something is in the way right now", as opposed to
# "this request was never going to work". Matched against the guard's own
# wording in mcp_server/safety.py.
_BLOCK_WORDS = ("stop distance", "bump guard", "no range sensor can see",
                "no fresh observation", "cannot read the range sensors")


def is_block(why: str) -> bool:
    """Is this refusal something that might simply walk away?"""
    low = (why or "").lower()
    return any(word in low for word in _BLOCK_WORDS)


class RuntimeObstacles:
    """Things that blocked her during THIS run. Never part of the map.

    ⚠ THIS DOES NOT TOUCH `state/map.npz`, AND THAT IS THE POINT. The static
    map is a survey: it says what the room is. A person standing in a doorway
    is not what the room is, and writing him into it would mean Friday plans
    around a ghost of him tomorrow -- and would need ten contradicting sweeps
    to clear him again, because the log-odds grid is deliberately slow to
    change its mind. `slam/transient.py` makes the same separation for the
    same reason, one sweep at a time; this one remembers across re-plans
    inside a single navigation run, which is the timescale replanning needs.

    Points go in as world metres and come out as the `extra_blocked` array
    `path_planner.preprocess` already accepts and inflates exactly like walls.
    """

    def __init__(self, ttl_s: float = BLOCK_TTL_S) -> None:
        self.ttl_s = float(ttl_s)
        self._points: list[tuple[float, float, float]] = []   # x, y, when

    def add(self, xy, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._points.append((float(xy[0]), float(xy[1]), now))

    def expire(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        before = len(self._points)
        self._points = [p for p in self._points if now - p[2] < self.ttl_s]
        return before - len(self._points)

    def points(self, now: float | None = None) -> np.ndarray:
        self.expire(now)
        if not self._points:
            return np.empty((0, 2))
        return np.array([[p[0], p[1]] for p in self._points])

    def __len__(self) -> int:
        return len(self._points)

    def as_dict(self) -> dict:
        return {"count": len(self._points), "ttl_s": self.ttl_s,
                "points": [[round(p[0], 2), round(p[1], 2)]
                           for p in self._points[-12:]]}


def obstacle_ahead(pose, clearance_mm: float | None) -> tuple[float, float]:
    """Where the thing in front of her probably is, in world metres.

    A range reading is a distance along the nose, so the obstacle is that far
    along the current heading. It is an estimate and it is treated as one: it
    goes into the runtime layer, where being a few centimetres out costs a
    slightly wrong detour, not a collision -- the clearance interrupt is still
    what actually stops her.
    """
    reach = (config.SAFETY_STOP_MM if clearance_mm is None
             else float(clearance_mm)) / 1000.0
    reach = max(reach, config.NAV_ROBOT_RADIUS_M)
    return (pose[0] + reach * math.cos(pose[2]),
            pose[1] + reach * math.sin(pose[2]))


def snap_goal(prep, cell, radius_m: float = GOAL_SNAP_M):
    """The goal cell, or the nearest plannable one within `radius_m`.

    Step 10's "goal cell occupied" case. It happens constantly and innocently:
    a goal saved against the middle of a room is fine, but a goal saved next
    to the sofa is inside the inflation of the sofa, and refusing it would
    mean Friday cannot be sent to half the places she has been taught.
    Returns (cell or None, how far it moved in metres).
    """
    nx, ny = prep.plannable.shape
    ix, iy = int(cell[0]), int(cell[1])
    if 0 <= ix < nx and 0 <= iy < ny and prep.plannable[ix, iy]:
        return (ix, iy), 0.0
    if not prep.plannable.any():
        return None, 0.0
    legal = np.argwhere(prep.plannable)
    away = np.hypot(legal[:, 0] - ix, legal[:, 1] - iy) * prep.cell_m
    best = int(np.argmin(away))
    if away[best] > radius_m:
        return None, float(away[best])
    return (int(legal[best, 0]), int(legal[best, 1])), float(away[best])


# Reasons another identical attempt cannot improve on. ⚠ ALL LOWER CASE: the
# text being searched is lower-cased first, and a capital letter here silently
# stops a reason matching. That is not a cosmetic bug — "do not know where I
# am" failing to match meant an unmatched pose was RE-PLANNED rather than
# failing safe, i.e. the one situation where driving again is driving blind.
_FINAL_REASONS = ("stop was called", "refused:", "not driving",
                  "do not know where i am", "budget")


def _is_final(why: str) -> bool:
    low = (why or "").lower()
    return any(reason in low for reason in _FINAL_REASONS)


# A round trip whose outbound leg is ANOTHER PROCESS -- an exploration run --
# cannot be held in memory, because the thing that has to fire the return is
# not the thing that started it and may not even be the same process by then.
# So the start pose is written down, and anything that notices exploration has
# finished can pick it up. It is cleared the moment it is acted on; a stale
# one would send her "back" to a room she left yesterday.
PENDING_FILE = config.ROOT / "state" / "nav_round_trip.json"

# How long a pending return stays valid. Past this the pose it holds is a
# coordinate from a map session that may not exist any more, and driving to it
# would be the confident-wrong-address failure this whole module avoids.
PENDING_TTL_S = 6 * 3600.0


def arm_return_to_start(pose, why: str = "exploration") -> dict:
    """Write down where she set off from, for something else to act on later."""
    row = {"x": float(pose[0]), "y": float(pose[1]),
           "theta": float(pose[2]) if len(pose) > 2 else 0.0,
           "at": time.time(), "why": why}
    try:
        PENDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = PENDING_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(row))
        tmp.replace(PENDING_FILE)
    except OSError as exc:                           # noqa: BLE001
        log.warning("could not arm the return: %s", exc)
        return {"armed": False, "why": str(exc)}
    log.info("armed a return to (%.2f, %.2f) after %s", row["x"], row["y"], why)
    return {"armed": True, **row}


def pending_return() -> dict | None:
    """The armed return, if there is one and it is still fresh."""
    try:
        row = json.loads(PENDING_FILE.read_text())
    except (OSError, ValueError):
        return None
    if time.time() - float(row.get("at", 0)) > PENDING_TTL_S:
        log.warning("a pending return was too old to trust; dropping it")
        clear_pending()
        return None
    return row


def clear_pending() -> None:
    try:
        PENDING_FILE.unlink()
    except OSError:
        pass


NAV_LOG = config.ROOT / "state" / "nav_log.jsonl"
NAV_LOG_KEEP = 200


def _log_attempt(report: "ReturnReport") -> None:
    """One line per navigation attempt, for the dashboard to read back.

    JSONL and append-only, the same shape `state/dashboard_events.jsonl`
    already uses, so a half-written line costs one row rather than the file.
    Trimmed on write rather than by a cron: this is the only writer, and a log
    nobody prunes is a log that eventually fills the SD card.
    """
    row = {"at": report.at or time.time(), "destination": report.destination,
           "state": report.state.value, "arrived": report.arrived,
           "seconds": report.seconds, "attempts": report.attempts,
           "replans": report.replans, "blocks": report.blocks_confirmed,
           "planned_m": round(report.planned_m, 2),
           "driven_m": (report.follow or {}).get("driven_m", 0.0),
           "why": report.why}
    try:
        NAV_LOG.parent.mkdir(parents=True, exist_ok=True)
        with NAV_LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        lines = NAV_LOG.read_text().splitlines()
        if len(lines) > NAV_LOG_KEEP * 2:
            tmp = NAV_LOG.with_suffix(".tmp")
            tmp.write_text("\n".join(lines[-NAV_LOG_KEEP:]) + "\n")
            tmp.replace(NAV_LOG)
    except OSError as exc:                       # noqa: BLE001
        log.warning("could not write %s: %s", NAV_LOG, exc)


def recent_attempts(limit: int = 20) -> list[dict]:
    """The last navigation attempts, newest first. Read-only."""
    try:
        lines = NAV_LOG.read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines[-limit * 2:]):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                              # a torn final line
        if len(out) >= limit:
            break
    return out


def resolve_landmark(name: str) -> tuple[dict | None, str]:
    """A landmark by name, or (None, what to say). NEVER a fuzzy match.

    `slam/places.py::find` is already careful -- exact, then "the" dropped,
    then a substring only when exactly one place matches -- but that last step
    is still a guess, and step 8 asks for none. So this uses the first two
    rules only, and when a substring WOULD have matched it offers the name
    back as a question instead of driving several metres on it.
    """
    from slam import places

    wanted = " ".join((name or "").strip().lower().split())
    if not wanted:
        return None, "I need the name of somewhere I know."

    rows = places.all()
    if not rows:
        return None, ("I do not know anywhere by name yet. Stand somewhere "
                      "and say \u201cremember this as the kitchen\u201d.")

    bare = wanted[4:] if wanted.startswith("the ") else wanted
    for row in rows:
        if row["name"] == wanted or row["name"] in (bare, f"the {bare}"):
            return row, ""

    known = ", ".join(r["name"] for r in rows)
    near = [r["name"] for r in rows if bare in r["name"] or r["name"] in bare]
    if len(near) == 1:
        return None, (f"I do not know anywhere called \u201c{name}\u201d. "
                      f"Did you mean \u201c{near[0]}\u201d? I know: {known}.")
    return None, (f"I do not know anywhere called \u201c{name}\u201d. "
                  f"I know: {known}.")


@dataclass
class ExplorationSnapshot:
    """What exploration was doing when it was interrupted.

    Read from the files the run already writes -- `state/rl_autonomy.json` for
    the live picture and `state/explore_mode.json` for the run itself -- and
    never from the run's memory, which is in another process.
    """

    taken_at: float = 0.0
    target_xy: list[float] | None = None
    target_age_s: float | None = None
    frontiers: int | None = None
    blacklisted: int | None = None
    tick: int | None = None
    explored_m2: float | None = None
    remaining_minutes: float | None = None
    run: dict = field(default_factory=dict)

    @classmethod
    def capture(cls, remaining_minutes: float | None = None
                ) -> "ExplorationSnapshot":
        snap = cls(taken_at=time.time(), remaining_minutes=remaining_minutes)
        try:
            live = json.loads(
                (config.ROOT / "state" / "rl_autonomy.json").read_text())
            target = live.get("EXPLORATION_TARGET") or {}
            snap.target_xy = target.get("target_xy")
            snap.target_age_s = target.get("target_age_s")
            snap.frontiers = target.get("frontiers")
            snap.blacklisted = target.get("blacklisted")
            snap.tick = live.get("TICK")
            snap.explored_m2 = (live.get("MAP_STATE") or {}).get("explored_m2")
        except (OSError, ValueError, AttributeError):
            pass
        try:
            snap.run = json.loads(
                (config.ROOT / "state" / "explore_mode.json").read_text())
        except (OSError, ValueError):
            pass
        return snap

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["note"] = ("frontiers are recomputed from the saved map on resume, "
                     "not restored: they are a pure function of it")
        return d


def distance_home(pose) -> float | None:
    """How far the given pose is from stored home, or None if none is stored.

    Harsh asked for "distance from a saved home" on 2026-09-13
    ([[friday-people-and-wake-requests]]); this is that number, straight line,
    in the map's frame. It is NOT the driving distance -- `path_planner` gives
    that, and it is always longer.
    """
    home = home_place()
    if home is None:
        return None
    return math.hypot(pose[0] - home["x"], pose[1] - home["y"])


# The step 6 name, kept so anything written against it keeps working.
ReturnHome = NavController
