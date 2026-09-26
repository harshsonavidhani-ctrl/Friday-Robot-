"""nav_tools.py — file 14 step 8: "take me to the kitchen", as a tool.

    navigate_to("the kitchen", action="tell me what you can see")
    return_home()
    nav_status()

WHY THIS IS NOT `go_to`. `places_tools.go_to` already drives to a named place,
and it stays exactly as it is. It runs `slam/navigate.py`: one Dijkstra per
leg, turn-then-drive to each waypoint. These tools run the file 14 engine
instead -- despeckled grid, A* with a clearance gradient, a smoothed path, a
pure-pursuit follower and an arrival test that has to HOLD -- plus two things
`go_to` does not have: a re-plan loop around a failed attempt, and an action
queued for after arrival. Both tools are registered; neither replaces the
other, and `go_to` remains the one with the longer service record.

⚠ THE ACTION IS RETURNED, NOT PERFORMED, and that is a direction-of-dependency
fact rather than a shortcut. The brain calls this server; this server cannot
call the brain. So "when you get there, tell me what you see" comes back in
the result as `action_pending`, and the brain -- which is mid-turn waiting on
this call -- performs it through file 08's ordinary reply/emotion pipeline,
which is where speaking and face changes are supposed to come from.

⚠ WHAT THE DRIVE ITSELF SAYS IS DIFFERENT, since 2026-09-23. Step 10 wants
the transitions announced -- blocked, stuck, recovered, arrived, stopped for
help -- and Harsh ruled that none of navigation may need a model. Those are
therefore FIXED LINES from `nav_voice`, spoken through `tts.speak()` (so the
wake detector still knows it is her voice), from whichever drive is running.
They are the only speech in this file, and none of it is composed by a model.

⚠ NO FUZZY MATCHING. Step 8 is explicit and it is right: a near-miss string
match sends a robot several metres across a flat to somewhere nobody asked
for. `nav_state_machine.resolve_landmark` accepts an exact name or the same
name with "the" added or dropped, and otherwise refuses -- offering the close
name back as a question instead of driving on it.
"""

from __future__ import annotations

import json
import logging
import math
import time

import config
import nav_state_machine as nsm
import nav_voice
from mcp_server import hardware
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.nav")


def _controller():
    """A controller holding a guarded robot, armed against a stale abort.

    It gets the REAL voice: every drive, whoever started it, says the same
    fixed lines at the same transitions (step 10). The engine itself defaults
    to silence so the test suites never make a sound.
    """
    import nav_follower

    _reclaim_if_run_gone()
    robot = nav_follower.GuardedRobot()
    hardware.ABORT.clear()
    robot.arm()
    return nsm.NavController(robot=robot, voice=nav_voice.VOICE), robot


def _reclaim_if_run_gone() -> None:
    """Take back a lease whose exploration run has already ended.

    A run that finishes on its own does not return its lease (only
    stop_exploration() does), and the lease runs 300 s past the run. Every
    navigation command in that window was refused with "the range sensors are
    in use by an autonomous run" although no run was out. With `_running()`
    False the run's unit and processes are gone, so nothing holds the ports.
    """
    from mcp_server.tools import autonomy

    if hardware.lease_active() > 0.0 and not autonomy._running():
        log.info("the exploration run has ended; taking the hardware back")
        hardware.lease_return()


@tool
async def navigate_to(landmark_name: str, action: str = "",
                      return_to_start: bool | None = None,
                      budget_m: float | None = None,
                      budget_seconds: float | None = None) -> str:
    """Drive to somewhere I know by name, then optionally do something there.

    I DRIVE MYSELF when you call this. I work out where I am on my map, plan a
    route around what I know, smooth it, and follow it — re-measuring where I
    am from the LiDAR every step rather than trusting how far the wheels were
    told to go. I scan before every move and the safety interrupt is live
    throughout. Call stop() and I stop.

    I refuse rather than guess if I do not know the name, cannot work out
    where I am, or cannot find a route.

    If you give me an `action`, I do NOT do it myself — I bring it back to you
    in `action_pending` when I arrive, and you perform it in your reply so it
    goes through the normal voice and face pipeline.

    Args:
        landmark_name: the exact name, like "the kitchen". I do not guess at
            near matches.
        action: what to do on arrival, in your own words — "say what you see",
            "tell Harsh you are there". Left empty, I just stop.
        return_to_start: True to come back to where I am standing now once I
            get there — use it when Harsh says "and come back", "then come
            back here", or "there and back". False to stay there. Leave it
            out and I use whatever round-trip mode is set to.
        budget_m: how far I may drive trying.
        budget_seconds: how long I may take.
    """
    controller, robot = _controller()
    kw = {}
    if budget_m is not None:
        kw["budget_m"] = float(budget_m)
    if budget_seconds is not None:
        kw["budget_s"] = float(budget_seconds)

    report = await controller.go_to_landmark(
        landmark_name, action=action, return_to_start=return_to_start, **kw)
    out = report.as_dict()
    out["action_pending"] = action if (report.arrived and action) else ""
    if report.round_trip:
        out["came_back"] = report.returned_to_start
    if report.arrived and action:
        out["next"] = ("I am there. Do the queued action now, in your reply — "
                       "do not call this tool again.")
    log.info("navigate_to %s: %s", landmark_name, report.why)
    return json.dumps(out, indent=2)


@tool
async def return_home() -> str:
    """Go back to the place called home and stop there.

    Home is an ordinary saved place: stand where you want it and say
    "remember this as home". I refuse rather than guess if no home is saved,
    if I cannot place myself on my map, or if nothing passable connects here
    to it.
    """
    controller, robot = _controller()
    report = await controller.go_home()
    log.info("return_home: %s", report.why)
    return json.dumps(report.as_dict(), indent=2)


@tool
def nav_status() -> str:
    """Where my navigation is up to: the state, the home timer, and home.

    Read-only. Says which of exploring / returning home / driving to a target
    I am in, how long is left on the return-home timer, and where home is.
    """
    controller = nsm.NavController()
    state = controller.status()
    state["known_places"] = _known()
    state["round_trip"] = nsm.round_trip_enabled()
    state["waiting_to_come_back"] = nsm.pending_return()
    return json.dumps(state, indent=2)


def _known() -> list[str]:
    try:
        from slam import places
        return [row["name"] for row in places.all()]
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not list places: %s", exc)
        return []


# =====================================================================
# ROUND TRIP — go and do the thing, then come back to where you started
# =====================================================================
# Harsh, 2026-09-21: "friday go explore the home it explores and like come to
# the position where it started and like then i commanded it go to the kitchen
# then it should go to the kitchen and return to where it started".
#
# Target navigation is driven here, in one call, so its way back happens
# inline. Exploring and coming back is an EXPLORE TRIP (below): explored and
# driven home by this server on one map, so the start spot is never lost.


@tool
def set_round_trip(on: bool = True) -> str:
    """Turn "and come back afterwards" on or off for everything I'm sent to do.

    Use this when Harsh says come back, return to where you started, round
    trip, there and back, or "always come back" — and with on=False for stay
    there, don't come back, or "wait for me there".

    With it ON, anywhere I am sent I remember the exact spot I was standing
    when you told me, and I drive back to it when I am done. That spot is NOT
    my saved home — it is wherever I happened to be when you asked, and it
    only counts for that one errand.

    Args:
        on: True to come back afterwards, False to stay where I am sent.
    """
    state = nsm.set_round_trip(bool(on))
    if not state.get("saved"):
        return f"I could not remember that setting: {state.get('why')}"
    return ("Round trip is ON — wherever you send me, I will come back to "
            "where I was standing when you asked."
            if on else
            "Round trip is off — I will stay where you send me.")


# =====================================================================
# EXPLORE TRIP — "Friday, explore" always comes back to where it started
# =====================================================================
# Harsh, 2026-09-24: "if it got command to explore and come back it should
# first use lidar to save the spot it started then explore and then back to
# the spot where it started" -- and "i say friday explore after exploring
# should come back where it started". Asked how, he chose: track her whole
# trip.
#
# ⚠ WHY NOT THE RL RUN ANY MORE. Exploring in a separate process (the RL run)
# meant finding herself TWICE on the saved navigation map: once to note the
# start, once afterwards to plan the way back. That map is 13 m2 from 30
# sweeps (2026-09-19), and in the hostel room two spots on it looked alike
# (1.11 against 1.06), so she refused to set off; and a two-minute run takes
# her past its edge, so the way back could not be planned either.
#
# SO THE TRIP NEVER LETS GO OF WHERE SHE IS. slam/mission.py explores inside
# this server with ONE Mapper tracking her the whole way, and the same Mapper
# drives her back: the start spot is a point on the map she has been measured
# on continuously, not something to rediscover.
#   * ALWAYS on the OLD saved map (Harsh, 2026-09-25: "it should not use
#     fresh map -- mapping the position as well as all the things in old map
#     only"). A copy in memory, grown as she explores. When two spots on it
#     look alike she moves a little to tell them apart (nsm.locate); when
#     she still cannot, she does not set off.
# AND THE SAVED MAP GROWS (Harsh, 2026-09-24: "map grows while exploring and
# it should use the map"): after the trip, map_growth saves the grown copy
# over state/map.npz, keeping every earlier version in the navigation
# bundle's history. The RL exploration map is never written.
#
# A PERSON'S STOP ENDS THE TRIP where it is. "Stop" is never followed by a
# drive back that nobody asked for.

# A trip drives in short legs; this is a ceiling on distance, not a target.
TRIP_METRES_PER_MINUTE = 6.0
TRIP_DEFAULT_MINUTES = 2.0

_EXPLORED_OUT = ("no frontiers left", "nothing reachable left")


def _mission(robot):
    from slam.mission import Mission
    return Mission(robot)


def _explorer_robot():
    """The same six-call robot explore_room drives through the guard."""
    from mcp_server.tools.explore import _Robot
    robot = _Robot()
    robot._trips = robot.reflex_trips()      # a baseline, not a stop
    return robot


@tool
async def explore_and_return(minutes: float | None = None,
                             thinking: bool | None = None) -> str:
    """Explore on my own, then come back to the exact spot I started from.

    What "Friday, explore" and "explore and come back" do. Returns at once
    and runs in the background; I announce everything myself, so do not
    repeat my reply. I first look with the LiDAR and mark this spot on my
    saved map -- moving a little if two places on it look alike, and not
    setting off at all if I still cannot tell -- then explore while tracking
    where I am the whole way, drive back to the mark, and add what I saw to
    my saved map. Say stop and I stop where I am, without coming back.

    Args:
        minutes: how long to explore before coming back. Defaults to 2.
        thinking: ignored -- this explorer never uses a model.
    """
    if _journey_running():
        return _busy_reply()
    try:
        minutes = float(minutes) if minutes else TRIP_DEFAULT_MINUTES
    except (TypeError, ValueError):
        minutes = TRIP_DEFAULT_MINUTES
    minutes = max(0.5, min(minutes, 30.0))
    nsm.clear_failsafe()
    _start_journey("exploring", _explore_trip(minutes))
    return nav_voice.VOICE.say("exploring", _minutes_phrase(minutes))


async def _trip_start(robot):
    """(TripTracker, "") or (None, why): where she is, from one look.

    On the saved map only. Two look-alike spots are both kept and followed;
    only a map that fits nowhere at all stops the trip before it starts.
    """
    import anyio.to_thread
    import trip_tracker
    from slam.grid import OccupancyGrid

    try:
        grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
    except (OSError, ValueError) as exc:
        return None, f"I have no saved map ({exc})"
    import trip_recorder

    scan = await robot.sweep()
    trip_recorder.note("sweep", src="start", scan=scan)
    if not scan:
        return None, "my LiDAR did not answer"
    hyps, certain, why = await anyio.to_thread.run_sync(
        trip_tracker.start_hypotheses, grid, scan)
    trip_recorder.note("start", hypotheses=hyps, certain=certain, why=why)
    if not hyps:
        return None, f"nothing on my map fits what I see: {why}"
    log.info("explore trip: on the saved map at %s%s",
             [[round(float(v), 2) for v in h] for h in hyps],
             "" if certain else " (two look-alikes; following both)")
    return trip_tracker.TripTracker(grid, hyps, certain), ""


def _minutes_phrase(minutes: float) -> str:
    if abs(minutes - round(minutes)) < 1e-6:
        n = int(round(minutes))
        return f"{n} minute" + ("" if n == 1 else "s")
    return f"{minutes:g} minutes"


async def _explore_trip(minutes: float):
    """Mark the spot, explore on one tracked map, come back to the mark."""
    # 1. THE MARK. She looks with the LiDAR, before a wheel turns, ON THE OLD
    # MAP ONLY (Harsh, 2026-09-25: "it should not use fresh map"), and sets
    # off: "it just sees, it should explore". Where two spots on the map look
    # alike -- most of the hostel room, measured 2026-09-25 -- she neither
    # refuses nor shuffles about first; trip_tracker follows both, and the
    # explorer's own moves tell them apart.
    _reclaim_if_run_gone()
    import trip_recorder

    trip_recorder.start(_recordings_dir())
    try:
        return await _recorded_trip(minutes)
    finally:
        trip_recorder.stop()


def _recordings_dir():
    """Next to the nav log, so the test suites record into their temp dirs."""
    return nsm.NAV_LOG.parent / "trip_recordings"


async def _recorded_trip(minutes: float):
    import asyncio

    hardware.ABORT.clear()
    robot = _explorer_robot()
    tracker, why = await _trip_start(robot)
    if tracker is None:
        log.info("explore trip: not setting off — %s", why)
        nav_voice.VOICE.say("lost")
        return None

    # WHAT SHE SAW (Harsh, 2026-09-25: "remember the objects it saw and ...
    # speak what all it saw during exploration"). The camera watcher,
    # scene.py, keeps running through the trip; this collects what it names,
    # remembers it and says it at the very end -- however the trip ends:
    # back at the mark, stopped by a person, or stuck and asking for help.
    seen = _sightings()
    watcher = asyncio.create_task(_watch_objects(seen))
    try:
        return await _trip_body(minutes, robot, tracker)
    finally:
        watcher.cancel()
        nav_voice.VOICE.say_text(seen.finish())


async def _trip_body(minutes: float, robot, tracker):
    """Explore, tour, come back to the mark, grow the map, save home."""
    import trip_recorder
    import trip_tracker

    frame = "saved"

    # 2. EXPLORE. The explorer runs exactly as it always has, on its own
    # copy of the saved map; the tracker only listens to its moves and sweeps.
    deadline = time.monotonic() + minutes * 60.0
    # PlannerTurns: the explorer plans left as positive and the turn tool
    # takes left as negative, so without it every turn was driven mirrored.
    report = await _mission(trip_tracker.ListeningRobot(
        trip_tracker.PlannerTurns(robot), tracker)).run(
        tracker.explorer_mapper(),
        budget_m=max(config.MISSION_BUDGET_M,
                     minutes * TRIP_METRES_PER_MINUTE),
        budget_s=minutes * 60.0)
    why = report.get("stopped_because", "")
    log.info("explore trip: exploring ended (%s), %.2f m driven, %.1f m2",
             why, report.get("driven_m", 0.0), report.get("floor_m2", 0.0))
    # From here on she drives on the TRACKER's map, back to its mark -- once
    # it has caught up with every sweep the explorer took.
    try:
        await tracker.drain()
    finally:
        tracker.close()
    mapper, start, certain = tracker.best()
    log.info("explore trip: tracked %s", tracker.as_dict())
    if getattr(tracker, "hyps", None):
        # ASK THE SAVED MAP BEFORE THE TOUR, certain or not: on 2026-09-26
        # 17:19 the tracker was "certain" and 0.7 m out in position (heading
        # right) from the hand-over on. Two agreeing sure fixes either pick
        # a look-alike (near one) or re-anchor the leader's whole frame --
        # its pose AND the start mark -- onto the map's.
        # A SURE start: a mismatch now is drift, so only her pose moves. A
        # look-alike start: the whole frame may be off, so the mark moves too.
        fix = await nsm.two_sure_fixes(robot)
        picked = None
        if fix is not None and getattr(tracker, "started_certain", False):
            # Two agreeing SURE fixes are the safeguard, not the distance:
            # 17:48 they agreed at margin 1.67 that she was 0.7 m off after
            # 0.5 m of exploring, and a 0.6 m cap refused them silently.
            picked = tracker.confirmed_by(fix, max_jump_m=DRIFT_FIX_MAX_M)
            if picked is not None:
                mapper, start = picked
                log.info("explore trip: the saved map corrects her to %s",
                         [round(float(v), 2) for v in mapper.pose])
            else:
                log.info("explore trip: the saved map puts her over %.1f m "
                         "from her tracking; not taken", DRIFT_FIX_MAX_M)
        elif fix is not None:
            mapper, start = tracker.reanchor(fix)
            log.info("explore trip: re-anchored on the saved map at %s; the "
                     "start mark moves with her to %s",
                     [round(float(v), 2) for v in mapper.pose],
                     [round(float(v), 2) for v in start])
        if fix is None:
            log.info("explore trip: the saved map is not sure where she is "
                     "here; keeping the trip's own tracking")
    # Is the trip's frame the saved map's? A sure start, or a re-anchor just
    # now, says yes; then a later loss is drift and the mark stays put.
    anchored = bool(getattr(tracker, "started_certain", True)) or (
        fix is not None if getattr(tracker, "hyps", None) else True)
    # On the mapper, which the tour legs and the way home share (see
    # NavController._drive_to): a later re-anchor of an unanchored frame
    # records its correction there, for the start mark to follow.
    mapper.frame_anchored = anchored
    mapper.frame_deltas = []
    trip_recorder.note("tracked", report=report, tracker=tracker.as_dict())

    # 3. A PERSON SAID STOP: stay stopped. What she mapped is still kept.
    if why == "stop was called" or robot.stopped():
        log.info("explore trip: stopped by a person — not driving back")
        await _grow_map(mapper, frame, trusted=certain and bool(
            getattr(mapper, "pose_trusted", False)))
        return report

    # 4. THE ROOM IS ALREADY MAPPED, BUT HE ASKED FOR MINUTES: tour it.
    # 2026-09-24 00:20 in the hostel: the LiDAR saw the whole 11 m2 room from
    # the start spot, the mapping explorer found no frontiers at all, and
    # "explore for 2 minutes" came back after 25 cm. So with time left she
    # drives to the reachable spot farthest from everywhere she has been,
    # again and again, on the same tracked map, until the time is up or the
    # room has been toured.
    controller, _ = _controller()           # a fresh stop baseline
    controller.frame = frame
    toured_out = False
    if why in _EXPLORED_OUT and time.monotonic() < deadline - TOUR_MIN_LEG_S:
        outcome = await _tour(controller, mapper, start, deadline)
        if outcome == "stopped":
            log.info("explore trip: stopped by a person — not driving back")
            await _grow_map(mapper, frame, trusted=bool(
                getattr(mapper, "pose_trusted", False)))
            return report
        toured_out = outcome == "toured"

    # 5. BACK TO THE MARK, on the map she was tracked on.
    if toured_out or (why in _EXPLORED_OUT
                      and time.monotonic() < deadline - TOUR_MIN_LEG_S):
        nsm.note_explored_out("toured the whole room" if toured_out else why,
                              report.get("floor_m2"))
        nav_voice.VOICE.say("frontiers_done")
    else:
        nav_voice.VOICE.say("setting_off", "where I set off from")
    # A tour leg that re-anchored an unanchored frame moved her into the
    # saved map's frame; the start mark, set in the old one, follows.
    for delta in getattr(mapper, "frame_deltas", []) or []:
        start = nsm.apply_delta(delta, start)
        log.info("explore trip: the start mark follows the re-anchor to %s",
                 [round(float(v), 2) for v in start])
    mapper.frame_deltas = []
    controller, _ = _controller()           # a fresh baseline for the way home
    controller.frame = frame
    import nav_follower
    # Harsh, 2026-09-26: "when it says heading back early then it should go
    # where it started... then stop". On 17:39 she re-found herself on the
    # saved map and drove on, then ran out of attempts after one stall. The
    # drive home now has room for more: a whole-map re-anchor costs ~5 s.
    back = await controller.return_to(start[:2], mapper, refind_near=True,
                                      max_attempts=TRIP_HOME_ATTEMPTS,
                                      max_refinds=TRIP_HOME_REFINDS,
                                      arrival=nav_follower.ArrivalMonitor(
                                          start[:2],
                                          tolerance_m=TRIP_RETURN_TOL_M,
                                          loose_m=TRIP_RETURN_LOOSE_M))
    log.info("explore trip: back — %s", back.why)
    if back.arrived and await _check_home(controller, mapper, start, tracker):
        # The saved map says she is not quite there: one more short drive.
        controller, _ = _controller()
        controller.frame = frame
        back = await controller.return_to(
            start[:2], mapper, arrival=nav_follower.ArrivalMonitor(
                start[:2], tolerance_m=TRIP_RETURN_TOL_M,
                loose_m=TRIP_RETURN_LOOSE_M))
        log.info("explore trip: back, after the map's correction — %s",
                 back.why)
    if back.arrived:
        faced = await _face(controller, mapper, float(start[2]))
        log.info("explore trip: %s the way she set off",
                 "facing" if faced else "could not turn to face")

    # 6. THE MAP GROWS (Harsh, 2026-09-24: "map grows while exploring and it
    # should use the map"). Only from a trip that knows where it is: home
    # safely, or at least still matched.
    # Never from a trip whose start stayed a look-alike: its map would be
    # merged into the saved one at whichever spot it happened to follow.
    # And only from one that GOT BACK: "the last sweep still matched" let a
    # trip that ended stuck, its heading 75 degrees out after a cut-short
    # turn, add 11.5 m2 of rotated walls (2026-09-25 22:43, undone).
    grown = await _grow_map(mapper, frame,
                            trusted=certain and bool(back.arrived))

    # 7. A HOME TO RETURN TO. The prompt's return-to-home goes to "the stored
    # home / boot pose", and none had ever been stored. The first exploration
    # that knows where its start is ON THE SAVED MAP makes that spot home --
    # never a spot on a throw-away working map that did not fit.
    # ⚠ Only from a start the whole-map search was sure of ON ITS OWN, never
    # one the trip's evidence settled: replayed, the 23:14 trip called itself
    # certain of the WRONG look-alike, and the next trip, starting where it
    # ended, was placed confidently at the other one.
    if (back.arrived and getattr(tracker, "started_certain", False)
            and nsm.home_place() is None):
        where = None
        if frame == "saved":
            where = start
        elif grown and grown.get("grew") and grown.get("fitted"):
            where = tuple(grown["fitted"]["pose"])
        if where is not None:
            nsm.set_home((float(where[0]), float(where[1]), 0.0),
                         note="where my first exploration started")
            nav_voice.VOICE.say("home_set")
    return back


# "Back where I started" means THE spot. Harsh, 2026-09-25: "the place where
# Friday stops she assumes her starting position" -- the trips ended about
# 0.14 m off and turned, inside the general 0.28 m arrival tolerance, so she
# announced she was back without moving. The tracker holds her to ~5-12 cm,
# so the way back asks for 12 cm, relaxing to 20 cm only if it cannot settle.
TRIP_RETURN_TOL_M = 0.12
# After a SURE start, how far two agreeing sure fixes may move her pose.
DRIFT_FIX_MAX_M = 1.5
TRIP_HOME_ATTEMPTS = 6
TRIP_HOME_REFINDS = 4
TRIP_RETURN_LOOSE_M = 0.20

# ⚠ HER TRACKER DRIFTS OVER A TRIP, THE SAVED MAP DOES NOT. Trip 2026-09-26
# 00:09 came back 13 cm from the mark by her own reckoning; a sure whole-map
# fix afterwards put her 18 cm off and facing 48 degrees away from the way
# she set off, so "facing the way she set off" faced somewhere else. When the
# start was placed on the saved map with no look-alike, the same whole-map
# search is asked again on arrival, and a sure answer that agrees with the
# tracker to within HOME_CHECK_MAX_JUMP_M (so a look-alike cannot hijack it)
# replaces the tracked pose before the last correction and the turn.
HOME_CHECK_MAX_JUMP_M = 0.60


async def _check_home(controller, mapper, start, tracker) -> bool:
    """Correct the pose from the saved map. True if she should drive again."""
    if not getattr(tracker, "started_certain", False):
        return False
    fix = await _home_fix(controller.robot)
    if fix is None:
        log.info("explore trip: the saved map is not sure where she is at the "
                 "mark, so her own tracking stands")
        return False
    jump = math.hypot(fix[0] - float(mapper.pose[0]),
                      fix[1] - float(mapper.pose[1]))
    turn = math.degrees(abs((fix[2] - float(mapper.pose[2]) + math.pi)
                            % (2 * math.pi) - math.pi))
    if jump > HOME_CHECK_MAX_JUMP_M:
        log.info("explore trip: the map places her %.2f m from where she "
                 "thinks she is -- a look-alike, not a correction", jump)
        return False
    log.info("explore trip: the map corrects her by %.2f m and %.0f deg",
             jump, turn)
    mapper.pose = tuple(float(v) for v in fix)
    mapper.pose_trusted = True
    away = math.hypot(fix[0] - float(start[0]), fix[1] - float(start[1]))
    return away > TRIP_RETURN_TOL_M


async def _home_fix(robot):
    """A sure, polished whole-map fix of where she stands, or None."""
    return await nsm.sure_fix(robot)


# Back on the spot, she turns to face the way she set off: Harsh, 2026-09-25,
# watched her say "I am back where I started" at a spot that did not look like
# the start -- within the 0.28 m arrival tolerance, facing somewhere else.
FACE_TOLERANCE_DEG = 15.0
FACE_MAX_PIVOTS = 8
# A pivot measured under this turned nothing; this many in a row, stop.
FACE_IDLE_DEG = 3.0
FACE_IDLE_PIVOTS = 2


async def _face(controller, mapper, heading: float) -> bool:
    """Pivot, measured, until she faces `heading` (map radians). True if so."""
    import nav_follower

    robot = controller.robot
    follower = nav_follower.Follower(robot)
    follower._held_trust = bool(getattr(mapper, "pose_trusted", False))
    idle = 0
    for _ in range(FACE_MAX_PIVOTS):
        if robot.stopped():
            return False
        error = math.degrees(nav_follower._wrap(heading
                                                - float(mapper.pose[2])))
        if abs(error) <= FACE_TOLERANCE_DEG:
            return True
        step = max(-nav_follower.PIVOT_MAX_DEG,
                   min(nav_follower.PIVOT_MAX_DEG, error))
        done = await follower.step(mapper, "left" if step > 0 else "right",
                                   max(nav_follower.PIVOT_MIN_S,
                                       follower.rates.seconds_for_turn(step)))
        if done.refused or not getattr(mapper, "pose_trusted", False):
            return False
        # Pivots that do not move her are humming, not turning (2026-09-26
        # 00:35: -26, -17, -16, -9, -7, -7, 0, 0, 0 degrees as the motor
        # battery ran down). Two in a row and she stops asking.
        idle = idle + 1 if abs(done.turned_deg) < FACE_IDLE_DEG else 0
        if idle >= FACE_IDLE_PIVOTS:
            log.info("explore trip: the wheels are not turning her any more")
            return False
    return False


def _sightings():
    """What she sees on this trip: read from scene.py's file, never the camera."""
    import trip_objects
    return trip_objects.TripSightings(
        camera_reason=_camera_reason,
        report_file=nsm.NAV_LOG.parent / "trip_objects.json")


def _camera_reason():
    import scene
    return scene._why_not_now()


async def _watch_objects(seen) -> None:
    """Collect every new frame the camera watcher names, until cancelled."""
    import asyncio

    quiet_since = time.monotonic()
    while True:
        if seen.poll():
            quiet_since = time.monotonic()
        elif time.monotonic() - quiet_since > 2.5 * config.SCENE_PERIOD_S:
            # Ask why WHILE it is quiet: afterwards the reason may be gone.
            seen.note_camera()
        await asyncio.sleep(config.SCENE_BRIDGE_POLL_S)


# ⚠ OFF (2026-09-25 23:2x). Twice in one evening a trip grew state/map.npz
# and made it WORSE: at 22:43 from a wrong heading (11.5 m2 of rotated walls)
# and at 23:18 from a trip that was certain and got back (+8.5 m2, after which
# 0 of 4 trip start sweeps could be placed on it, against 2 of 4 before). Both
# were undone from the bundle history. Harsh, the same evening: "the map saved
# should be used ... not change the map". The code stays; this is the switch.
GROW_SAVED_MAP = False


async def _grow_map(mapper, frame: str, trusted: bool):
    """Fold the trip's map into the saved one, off the event loop.

    The fit is a whole-map search (a few seconds on the Pi), so it runs on a
    worker thread: `stop()` must still be answerable while it thinks.
    """
    if not GROW_SAVED_MAP:
        log.info("explore trip: the saved map is kept exactly as it is")
        return {"grew": False, "why": "the saved map is not changed by trips"}
    import anyio.to_thread
    import map_growth

    try:
        out = await anyio.to_thread.run_sync(map_growth.grow, mapper.grid,
                                             frame, trusted)
    except Exception as exc:                     # noqa: BLE001
        log.warning("explore trip: the map did not grow: %s", exc)
        return None
    log.info("explore trip: map %s", out)
    added = float(out.get("added_m2") or 0.0)
    if out.get("grew") and added >= 0.5:
        n = max(1, int(round(added)))
        nav_voice.VOICE.say("map_grew",
                            f"{n} square metre" + ("" if n == 1 else "s"))
    elif out.get("fitted") is False:
        nav_voice.VOICE.say("map_not_fitted")
    return out


# A tour leg needs at least this long to be worth starting: a sweep per step
# at ~1 Hz makes even a short leg ten-odd seconds.
TOUR_MIN_LEG_S = 12.0
# Two legs in a row that got nowhere -- blocked, wedged, no route -- and the
# tour is over; the drive home, with its full recovery, is what comes next.
TOUR_MAX_FAILED_LEGS = 2


async def _tour(controller, mapper, start, deadline: float) -> str:
    """Drive around an already-mapped room until the time is up.

    Returns "time" (ran out of time), "toured" (been everywhere reachable),
    "stuck" (legs kept failing), or "stopped" (a person said stop).
    """
    visited = [tuple(start[:2]), tuple(float(v) for v in mapper.pose[:2])]
    failed = 0
    announced = False
    while time.monotonic() < deadline - TOUR_MIN_LEG_S:
        if controller.robot.stopped():
            return "stopped"
        goal = nsm.tour_goal(mapper.grid, mapper.pose, visited,
                             extra_blocked=controller.obstacles.points())
        if goal is None:
            return "toured"
        if not announced:
            nav_voice.VOICE.say("tour")
            announced = True
        leg = await controller._drive_to(
            goal, "a new spot", nsm.NavState.EXPLORING, mapper,
            max_attempts=2, touring=True, refind_near=True,
            budget_s=max(TOUR_MIN_LEG_S, deadline - time.monotonic()))
        if leg.why == "stop was called" or controller.robot.stopped():
            return "stopped"
        # Wherever she got to counts as visited, and so does the goal: a spot
        # she could not reach is not one to keep choosing.
        visited.append(tuple(float(v) for v in mapper.pose[:2]))
        visited.append(tuple(goal))
        failed = 0 if leg.arrived else failed + 1
        log.info("explore trip: tour leg to (%.2f, %.2f) %s", goal[0],
                 goal[1], "arrived" if leg.arrived else f"gave up: {leg.why}")
        if failed >= TOUR_MAX_FAILED_LEGS:
            return "stuck"
    return "time"


@tool
async def come_back() -> str:
    """Drive back to where I set off from, if I noted a spot and have not returned.

    Use this if I explored and did not come back — because I was restarted, or
    the run ended oddly — or if Harsh just says come back, return, or come
    back here after sending me somewhere.
    """
    row = nsm.pending_return()
    if row is None:
        return ("I have not got a spot noted to come back to. Say “go to the "
                "kitchen and come back” and I will note one.")
    controller, _ = _controller()
    report = await controller.return_to_armed_start()
    return json.dumps(report.as_dict(), indent=2)


# ===========================================================================
# OFFLINE NAVIGATION — every behaviour here runs with no model at all.
#
# Harsh, 2026-09-23: "no ai will be used all behaviour in offline model".
# `offline.py` matches "go to the kitchen", "go home", "come back" and "try
# again" and calls the four tools below -- online or off, the brain is never
# asked. Everything they say is a fixed line from `nav_voice`.
#
# ⚠ THEY RETURN AT ONCE AND DRIVE IN THE BACKGROUND. A journey takes minutes
# and the offline path gives one tool call `STAGE_TOOL_TIMEOUT_S` (25 s). So
# each tool checks what it can check immediately (is the place known, is a
# journey already under way), says so, and hands the drive to ONE task in this
# server. `stop()` still ends it: ABORT is
# what the follower checks before every move.
#
# ⚠ THEY SPEAK FOR THEMSELVES, through the one navigation voice, so that the
# first sentence ("Heading to the kitchen.") and every later one ("Path
# blocked. Replanning my route.") come out of the same queue in order, never
# on top of each other from two processes. The offline intents therefore do
# not read the result out again.
# ===========================================================================

_JOURNEY: "asyncio.Task | None" = None
_JOURNEY_LABEL = ""

def _journey_running() -> str:
    """The destination of the journey under way, or "" if there is none."""
    if _JOURNEY is not None and not _JOURNEY.done():
        return _JOURNEY_LABEL
    return ""


def _start_journey(label: str, coro) -> None:
    import asyncio

    global _JOURNEY, _JOURNEY_LABEL
    _JOURNEY_LABEL = label
    _JOURNEY = asyncio.create_task(_run_journey(label, coro))


async def _run_journey(label: str, coro):
    """Await one journey, and make sure nothing it raises leaves the wheels on.

    Returns the journey's report (None if it failed), so whoever awaits the
    task -- a test, or a later caller -- can see how it ended.
    """
    import asyncio

    try:
        report = await coro
        log.info("journey to %s: %s", label,
                 getattr(report, "why", report))
        return report
    except asyncio.CancelledError:
        raise
    except Exception as exc:                     # noqa: BLE001
        log.error("journey to %s failed: %s", label, exc)
        try:
            from mcp_server.tools import movement
            import anyio.to_thread
            await anyio.to_thread.run_sync(movement.stop)
        except Exception as stop_exc:            # noqa: BLE001
            log.error("could not stop after the failure: %s", stop_exc)
        nav_voice.VOICE.say("failsafe")


async def _pause_exploration(controller) -> bool:
    """Step 9: stop a live exploration run before driving, keeping its time.

    Returns True if a run was paused (and so should be resumed on arrival).
    Waits for the run to actually release the robot -- the lease coming back
    is `_running()` going False -- because driving while it still steers is
    two things steering one robot.
    """
    import asyncio

    from mcp_server.tools import autonomy

    if not autonomy._running():
        return False
    out = controller.pause_exploration()
    for _ in range(60):
        if not autonomy._running():
            break
        await asyncio.sleep(0.5)
    if autonomy._running():
        raise RuntimeError(f"exploration did not stop: {out.get('stopped')}")
    return True


def _resume_after(controller, paused: bool, report) -> None:
    """Step 9: pick exploration back up once the errand has ARRIVED.

    Not after FAILSAFE: that waits for a person (step 13), and the paused
    run's time is in the failsafe record for "try again" to resume.
    """
    if not paused or report.state is not nsm.NavState.ARRIVED:
        return
    out = controller.resume_exploration()
    if out.get("resumed"):
        nav_voice.VOICE.say("resuming")
    log.info("resume after the errand: %s", out)


def _arrival_action(then: str):
    """An offline command to run on arrival, as a coroutine function, or None.

    Only commands that neither move her nor need the spine qualify -- the
    time, the battery, the weather card, the clock. It is parsed again AT
    arrival, so "tell me the time" tells the time it is when she gets there.
    """
    import offline

    first = offline.understand(then) if then else None
    if first is None or first.local or first.name in _NOT_ON_ARRIVAL:
        return None

    async def act() -> None:
        from mcp_server.registry import REGISTRY

        intent = offline.understand(then)
        if intent is None:
            return
        answer = ""
        for tool_name, args in intent.plan():
            invoke = next((fn for fn, name, _ in REGISTRY
                           if name == tool_name), None)
            if invoke is None:
                log.warning("no tool %r for the arrival action", tool_name)
                return
            answer = str(await invoke(**args))
        spoken = (offline.speakable(intent.name, answer) if intent.speak_result
                  else intent.say)
        if spoken:
            nav_voice.VOICE.say_text(spoken)

    return act


# Offline intents that must never run as an arrival action: anything that
# moves her, starts a run, or is itself navigation.
_NOT_ON_ARRIVAL = frozenset({
    "stop", "stop_exploring", "explore", "around", "backward", "forward",
    "left", "right", "go_left", "go_right", "go_forward", "go_backward",
    "speed", "nav_go", "nav_home", "nav_come_back", "try_again",
    "explore_return",
})


async def _errand(controller, name: str, come_back: bool | None, act):
    paused = await _pause_exploration(controller)
    report = await controller.go_to_landmark(
        name, return_to_start=come_back, after_arrival=act)
    _resume_after(controller, paused, report)
    return report


async def _home_errand(controller):
    paused = await _pause_exploration(controller)
    report = await controller.go_home()
    _resume_after(controller, paused, report)
    return report


def _busy_reply() -> str:
    busy = _journey_running()
    if busy == "exploring":
        return (nav_voice.VOICE.say("busy_exploring")
                or nav_voice.line("busy_exploring"))
    return nav_voice.VOICE.say("busy", busy) or nav_voice.line("busy", busy)


@tool
async def nav_go(place: str, come_back: bool | None = None,
                 then: str = "") -> str:
    """Offline voice command: drive to a place I know. Returns at once.

    Called by the offline command matcher for "go to the kitchen", "kitchen
    chalo", "go to the desk and come back". I announce everything myself —
    setting off, blocks, getting stuck, arriving — so do not repeat my reply.
    For a brain-driven errand whose result you need, use navigate_to instead.

    Args:
        place: the exact name of a saved place. Never guessed at.
        come_back: True to return to this spot afterwards; left out, the
            round-trip setting decides.
        then: an offline command to carry out on arrival, like "tell me the
            time". Anything that moves me is not allowed there.
    """
    if _journey_running():
        return _busy_reply()
    row, why = nsm.resolve_landmark(place)
    if row is None:
        return nav_voice.VOICE.say_text(why)
    act = _arrival_action(then)
    # A person giving a new order has dealt with whatever the last failure
    # was, as surely as by saying "try again"; the old record would otherwise
    # send the NEXT "try again" somewhere nobody now wants to go.
    nsm.clear_failsafe()
    controller, _ = _controller()
    _start_journey(row["name"], _errand(controller, row["name"], come_back,
                                        act))
    said = nav_voice.VOICE.say("setting_off", row["name"])
    if then and act is None:
        said += " " + nav_voice.VOICE.say_text(
            "I cannot do the rest of that on my own, so I will just go "
            "there.")
    return said


@tool
async def nav_home() -> str:
    """Offline voice command: drive to the saved place called home. Returns at once.

    Called by the offline command matcher for "go home" / "ghar chalo". I
    announce everything myself, so do not repeat my reply.
    """
    if _journey_running():
        return _busy_reply()
    if nsm.home_place() is None:
        return nav_voice.VOICE.say_text(
            "I do not know where home is. Stand where you want it and say "
            "remember this as home.")
    nsm.clear_failsafe()
    controller, _ = _controller()
    _start_journey("home", _home_errand(controller))
    return nav_voice.VOICE.say("setting_off", "home")


@tool
async def nav_come_back() -> str:
    """Offline voice command: drive back to the spot I set off from. Returns at once.

    Called by the offline command matcher for "come back" / "wapas aao". I
    announce everything myself, so do not repeat my reply.
    """
    if _journey_running():
        return _busy_reply()
    if nsm.pending_return() is None:
        return nav_voice.VOICE.say_text(
            "I have not got a spot noted to come back to.")
    nsm.clear_failsafe()
    controller, _ = _controller()
    _start_journey("where I set off from", controller.return_to_armed_start())
    return nav_voice.VOICE.say("setting_off", "where I set off from")


@tool
async def try_again() -> str:
    """"Friday, try again" — the only way out of a navigation FAILSAFE.

    Use this ONLY when Harsh says try again / resume / phir se try karo after
    I have stopped and asked for help. I work out where I am from scratch
    first (I may have been moved), start with a fresh allowance of stuck
    recoveries, and then do again whatever failed: exploring, going home, or
    going to a place. I never do this by myself. Returns at once; I announce
    everything myself.
    """
    if _journey_running():
        return _busy_reply()
    record = nsm.failsafe_record()
    if record is None:
        return nav_voice.VOICE.say("nothing_waiting")
    controller, _ = _controller()
    label = record.get("label") or "my destination"
    _start_journey(label, controller.retry_failsafe(record))
    # retry_failsafe says the "trying again" line itself, first thing.
    return nav_voice.line("try_again")
