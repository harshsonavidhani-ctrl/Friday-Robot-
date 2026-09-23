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
# ⚠ THE OUTBOUND LEG OF "GO AND EXPLORE" IS A DIFFERENT PROCESS. `explore_room`
# and `start_exploration` hand the robot to a separate exploration run that
# holds the hardware lease; this server does not drive during it and must not
# try. So the return is ARMED (the start pose is written down) and a watcher
# task fires it when the run ends. Target navigation needs none of that --
# that leg is driven here, in one call, so the return happens inline.

_WATCHER: "asyncio.Task | None" = None


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


@tool
async def explore_and_return(minutes: float | None = None,
                             thinking: bool | None = None) -> str:
    """Go and explore on my own, then come back to where I started.

    Use this when Harsh says explore and come back, go look around and return,
    map the place and come back here, or asks me to explore while round trip
    is on. It is `start_exploration` plus a promise: I note the exact spot I
    am standing on my map before I set off, and when the run finishes I drive
    back to it.

    I work out where I am first and REFUSE to set off if I cannot place myself
    on my map — a start position I am not sure of is one I cannot return to.

    Args:
        minutes: how long to explore. Defaults to 20.
        thinking: True to also think about where to go while I drive.
    """
    import asyncio

    global _WATCHER

    # FIRST, WHERE AM I -- from the LiDAR, against the saved map. This is the
    # spot she comes back to, so a lease left over from an earlier run must
    # not be what stops her looking.
    _reclaim_if_run_gone()
    mapper, why = await nsm.relocalise()
    if mapper is None:
        return (f"I am not setting off: {why} I would not be able to find my "
                "way back to this spot.")
    start = tuple(float(v) for v in mapper.pose)
    armed = nsm.arm_return_to_start(start, why="exploration")
    if not armed.get("armed"):
        return f"I could not note down where I am starting from: {armed['why']}"

    from mcp_server.tools import autonomy
    said = autonomy.start_exploration(minutes=minutes, thinking=thinking)
    if nsm.start_failed(said):
        # Nothing set off, so there is nothing to come back from: arming a
        # return (and a watcher) here would send her "back" after somebody
        # else's run ended.
        nsm.clear_pending()
        return said
    _note_run_clock(minutes)

    if _WATCHER is None or _WATCHER.done():
        _WATCHER = asyncio.create_task(_watch_then_return())
    return (f"{said}\n\nI have noted where I am standing "
            f"(x={start[0]:.2f} y={start[1]:.2f}) and I will drive back to "
            "this spot when the run finishes.")


async def _watch_then_return(poll_s: float = 2.0,
                             watch: "nsm.FrontierWatch | None" = None,
                             lease_wait_s: float = 30.0) -> None:
    """Wait for the exploration run to end, then bring her home to her spot.

    ⚠ IT WAITS FOR THE RUN TO ACTUALLY BE GONE before doing anything. The
    exploration process holds the hardware lease, and two things driving one
    robot is the failure this whole module is arranged to avoid. `_running()`
    going False is the lease being back.

    STEP 16: IT ALSO ENDS THE RUN EARLY when the run has genuinely run out of
    frontiers -- the explorer's own `exhausted` flag and frontier count, read
    from the telemetry the run already writes, held over several fresh
    readings (`nsm.FrontierWatch`). That is the same transition the run's own
    timer triggers, only sooner: stop the run, wait for the lease, come back.

    If this server restarts mid-run the task dies with it and the return stays
    ARMED on disk — `nav_status` reports it and `come_back()` fires it by hand.
    That is a deliberately visible loose end rather than a silent one.
    """
    import asyncio

    from mcp_server.tools import autonomy

    watch = watch or nsm.FrontierWatch()
    deadline = time.monotonic() + 6 * 3600.0
    try:
        await asyncio.sleep(poll_s)
        while autonomy._running() and time.monotonic() < deadline:
            if watch.update():
                row = nsm.note_early_return(watch)
                log.info("frontiers exhausted %s — ending the run early", row)
                nav_voice.VOICE.say("frontiers_done")
                said = autonomy.stop_exploration()
                if not nsm._stopped_ok(said):
                    log.error("could not end the run early: %s", said)
                waited = 0.0
                while autonomy._running() and waited < lease_wait_s:
                    await asyncio.sleep(0.5)
                    waited += 0.5
                break
            await asyncio.sleep(poll_s)
        if autonomy._running():
            # The run would not stop. Driving now would be two things
            # steering one robot, so the return stays armed for come_back().
            log.error("exploration is still running; the return stays armed")
            return
        if nsm.pending_return() is None:
            return                       # somebody already brought her back
        if _journey_running():
            log.info("another journey is driving; the return stays armed")
            return
        log.info("exploration finished — returning to the start pose")
        _claim_journey("where I set off from")
        # ⚠ TAKE THE HARDWARE BACK FIRST. A run that ends ON ITS OWN leaves
        # its lease behind -- only stop_exploration() returns it -- and the
        # lease is the run's length plus a 300 s margin. 2026-09-23 23:22: the
        # two-minute run finished, this fired, and the first sweep of the way
        # back was refused ("the range sensors are in use by an autonomous run
        # for another 252s"). `_running()` is False here, so the run's unit
        # and processes are gone and nothing else holds the ports: the same
        # reasoning stop_exploration() uses to return it.
        _reclaim_if_run_gone()
        nav_voice.VOICE.say("setting_off", "where I set off from")
        mapper = await _relocalise_when_ready()
        controller, _ = _controller()
        report = await controller.return_to_armed_start(mapper=mapper)
        log.info("return after exploring: %s", report.why)
    except asyncio.CancelledError:
        raise
    except Exception as exc:             # noqa: BLE001
        log.error("the return after exploring failed: %s", exc)


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
        log.info("relocalise after the run, try %d/%d: %s", attempt, tries,
                 why)
        if attempt < tries:
            await asyncio.sleep(wait_s)
    log.warning("could not place myself after the run: %s", why)
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
# server -- the same pattern as `_WATCHER`. `stop()` still ends it: ABORT is
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

# When navigation itself started an exploration run, and for how long, so a
# pause can hand back the time that was left. The run's own state file does
# not record its length.
_RUN_CLOCK: dict = {}


def _note_run_clock(minutes: float | None) -> None:
    _RUN_CLOCK.clear()
    _RUN_CLOCK.update({"started": time.monotonic(),
                       "minutes": float(minutes or nsm.RETURN_AFTER_MIN)})


def _journey_running() -> str:
    """The destination of the journey under way, or "" if there is none."""
    if _JOURNEY is not None and not _JOURNEY.done():
        return _JOURNEY_LABEL
    return ""


def _claim_journey(label: str) -> None:
    """Mark the CURRENT task as the journey (used by the exploration watcher)."""
    import asyncio

    global _JOURNEY, _JOURNEY_LABEL
    _JOURNEY, _JOURNEY_LABEL = asyncio.current_task(), label


def _start_journey(label: str, coro) -> None:
    import asyncio

    global _JOURNEY, _JOURNEY_LABEL
    _JOURNEY_LABEL = label
    _JOURNEY = asyncio.create_task(_run_journey(label, coro))


async def _run_journey(label: str, coro) -> None:
    """Await one journey, and make sure nothing it raises leaves the wheels on."""
    import asyncio

    try:
        report = await coro
        log.info("journey to %s: %s", label,
                 getattr(report, "why", report))
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
    if _RUN_CLOCK:
        controller.timer.minutes = _RUN_CLOCK["minutes"]
        controller.timer.started_at = _RUN_CLOCK["started"]
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
