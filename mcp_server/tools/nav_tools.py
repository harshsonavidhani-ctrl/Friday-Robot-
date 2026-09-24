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
#   * On the OLD saved map when she can be placed on it (Harsh: use the old
#     one) -- a copy in memory, grown as she explores.
#   * Otherwise the start spot is the origin of a fresh working map. She
#     cannot lose the origin of a map she is standing on.
# AND THE SAVED MAP GROWS (Harsh, 2026-09-24: "map grows while exploring and
# it should use the map"): after the trip, map_growth folds the trip's map
# into state/map.npz -- the copy directly, or the working map fitted onto it
# first -- keeping every earlier version in the navigation bundle's history.
# The RL exploration map is never written.
#
# A PERSON'S STOP ENDS THE TRIP where it is. "Stop" is never followed by a
# drive back that nobody asked for.

# A trip drives in short legs; this is a ceiling on distance, not a target.
TRIP_METRES_PER_MINUTE = 6.0
TRIP_DEFAULT_MINUTES = 2.0

# The last trip's map, kept so "try again" after a FAILSAFE on the way back
# can still find the start: its coordinates belong to this map only.
_TRIP: dict = {}

_EXPLORED_OUT = ("no frontiers left", "nothing reachable left")


def _working_map():
    """A fresh map whose origin is the spot she is standing on."""
    from slam.mapper import Mapper
    return Mapper(pose=(0.0, 0.0, 0.0))


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
    repeat my reply. I first look with the LiDAR and mark this spot -- on my
    saved map if I can place myself on it, otherwise on a fresh map of my
    own -- then explore while tracking where I am the whole way, then drive
    back to the mark. Say stop and I stop where I am, without coming back.

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


def _minutes_phrase(minutes: float) -> str:
    if abs(minutes - round(minutes)) < 1e-6:
        n = int(round(minutes))
        return f"{n} minute" + ("" if n == 1 else "s")
    return f"{minutes:g} minutes"


async def _explore_trip(minutes: float):
    """Mark the spot, explore on one tracked map, come back to the mark."""
    # 1. THE MARK. From the LiDAR, before a wheel turns.
    # ONE look. An ambiguous room looks the same two seconds later, and the
    # working map below is a complete answer, not a degraded one.
    _reclaim_if_run_gone()
    mapper = await _relocalise_when_ready(tries=1)
    frame = "saved"
    if mapper is None:
        frame = "trip"
        mapper = _working_map()
        nav_voice.VOICE.say("working_map")
    start = tuple(float(v) for v in mapper.pose)
    _TRIP.clear()
    _TRIP.update(mapper=mapper, start=start, frame=frame, at=time.time())
    log.info("explore trip: marked %s on the %s map, exploring %.1f min",
             [round(v, 2) for v in start], frame, minutes)

    # 2. EXPLORE, tracked the whole way on the same map.
    deadline = time.monotonic() + minutes * 60.0
    hardware.ABORT.clear()
    robot = _explorer_robot()
    report = await _mission(robot).run(
        mapper, budget_m=max(config.MISSION_BUDGET_M,
                             minutes * TRIP_METRES_PER_MINUTE),
        budget_s=minutes * 60.0)
    why = report.get("stopped_because", "")
    log.info("explore trip: exploring ended (%s), %.2f m driven, %.1f m2",
             why, report.get("driven_m", 0.0), report.get("floor_m2", 0.0))

    # 3. A PERSON SAID STOP: stay stopped. What she mapped is still kept.
    if why == "stop was called" or robot.stopped():
        log.info("explore trip: stopped by a person — not driving back")
        await _grow_map(mapper, frame,
                        trusted=bool(getattr(mapper, "pose_trusted", False)))
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
    controller, _ = _controller()           # a fresh baseline for the way home
    controller.frame = frame
    back = await controller.return_to(start[:2], mapper)
    log.info("explore trip: back — %s", back.why)

    # 6. THE MAP GROWS (Harsh, 2026-09-24: "map grows while exploring and it
    # should use the map"). Only from a trip that knows where it is: home
    # safely, or at least still matched.
    grown = await _grow_map(mapper, frame, trusted=bool(
        back.arrived or getattr(mapper, "pose_trusted", False)))

    # 7. A HOME TO RETURN TO. The prompt's return-to-home goes to "the stored
    # home / boot pose", and none had ever been stored. The first exploration
    # that knows where its start is ON THE SAVED MAP makes that spot home --
    # never a spot on a throw-away working map that did not fit.
    if back.arrived and nsm.home_place() is None:
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


async def _grow_map(mapper, frame: str, trusted: bool):
    """Fold the trip's map into the saved one, off the event loop.

    The fit is a whole-map search (a few seconds on the Pi), so it runs on a
    worker thread: `stop()` must still be answerable while it thinks.
    """
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
            max_attempts=1, touring=True,
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


async def _relocalise_when_ready(tries: int = 4, wait_s: float = 3.0):
    """Relocalise once the range sensors are answering again. Mapper or None.

    Straight after a run hands the hardware back, the LiDAR reopens lazily on
    first use and its first sweep can come back empty. A few short retries
    cover that; if it still cannot place her, None goes to the engine, which
    tries once more and then stops in FAILSAFE with the start spot recorded
    -- so "try again" can still bring her back.
    """
    import asyncio

    why = ""
    for attempt in range(1, tries + 1):
        mapper, why = await nsm.relocalise()
        if mapper is not None:
            return mapper
        log.info("relocalise, try %d/%d: %s", attempt, tries, why)
        if attempt < tries:
            await asyncio.sleep(wait_s)
    log.warning("could not place myself on the saved map: %s", why)
    return None


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
    if record.get("frame") == "trip":
        # The goal is on the trip's own map, which only this process holds.
        _start_journey(label, _retry_on_trip_map(controller, record))
    else:
        _start_journey(label, controller.retry_failsafe(record))
    # retry_failsafe says the "trying again" line itself, first thing.
    return nav_voice.line("try_again")


# Standstill sweeps to re-find her on the trip's map before a retry. Few,
# because the pose has not been lost -- only perhaps nudged by the person who
# helped her.
TRIP_REFRESH_SWEEPS = 3


async def _retry_on_trip_map(controller, record):
    """"Try again" after a FAILSAFE on an explore trip's own map.

    Relocalising on the SAVED map would be wrong here: the goal is a point on
    the map the trip built in memory. So she re-finds herself on THAT map
    with a few sweeps; if she was carried too far for it to match, she says
    she is lost and keeps waiting, exactly as the saved-map retry does. If
    this server has restarted since the trip, that map is gone, and
    retry_failsafe says so rather than guess.
    """
    mapper = _TRIP.get("mapper")
    if mapper is not None:
        for _ in range(TRIP_REFRESH_SWEEPS):
            try:
                scan = await controller.robot.sweep()
            except Exception as exc:             # noqa: BLE001
                log.warning("could not sweep before retrying: %s", exc)
                scan = None
            if scan:
                mapper.update(scan)
        if not mapper.pose_trusted:
            controller.voice.say("lost")
            nsm.write_failsafe(dict(record, at=time.time(),
                                    why="I could not find myself on the map "
                                        "of that trip."))
            controller.voice.say("failsafe")
            return None
    return await controller.retry_failsafe(record, mapper=mapper)
