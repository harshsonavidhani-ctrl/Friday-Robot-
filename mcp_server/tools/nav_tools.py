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
which is where speaking and face changes are supposed to come from. A tool
that spoke on its own would bypass `reply_contract` and the face arbiter, and
would also speak from the wrong process.

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
from mcp_server import hardware
from mcp_server.registry import tool

log = logging.getLogger("friday.tools.nav")


def _controller():
    """A controller holding a guarded robot, armed against a stale abort."""
    import nav_follower

    robot = nav_follower.GuardedRobot()
    hardware.ABORT.clear()
    robot.arm()
    return nsm.NavController(robot=robot), robot


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
    if "FAILED" in said.upper() or "could not" in said.lower():
        nsm.clear_pending()
        return said

    if _WATCHER is None or _WATCHER.done():
        _WATCHER = asyncio.create_task(_watch_then_return())
    return (f"{said}\n\nI have noted where I am standing "
            f"(x={start[0]:.2f} y={start[1]:.2f}) and I will drive back to "
            "this spot when the run finishes.")


async def _watch_then_return(poll_s: float = 5.0) -> None:
    """Wait for the exploration run to end, then bring her home to her spot.

    ⚠ IT WAITS FOR THE RUN TO ACTUALLY BE GONE before doing anything. The
    exploration process holds the hardware lease, and two things driving one
    robot is the failure this whole module is arranged to avoid. `_running()`
    going False is the lease being back.

    If this server restarts mid-run the task dies with it and the return stays
    ARMED on disk — `nav_status` reports it and `come_back()` fires it by hand.
    That is a deliberately visible loose end rather than a silent one.
    """
    import asyncio

    from mcp_server.tools import autonomy

    deadline = time.monotonic() + 6 * 3600.0
    try:
        await asyncio.sleep(poll_s)
        while autonomy._running() and time.monotonic() < deadline:
            await asyncio.sleep(poll_s)
        if nsm.pending_return() is None:
            return                       # somebody already brought her back
        log.info("exploration finished — returning to the start pose")
        controller, _ = _controller()
        report = await controller.return_to_armed_start()
        log.info("return after exploring: %s", report.why)
    except asyncio.CancelledError:
        raise
    except Exception as exc:             # noqa: BLE001
        log.error("the return after exploring failed: %s", exc)


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
