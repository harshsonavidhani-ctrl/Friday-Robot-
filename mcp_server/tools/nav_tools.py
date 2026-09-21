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
        budget_m: how far I may drive trying.
        budget_seconds: how long I may take.
    """
    controller, robot = _controller()
    kw = {}
    if budget_m is not None:
        kw["budget_m"] = float(budget_m)
    if budget_seconds is not None:
        kw["budget_s"] = float(budget_seconds)

    report = await controller.go_to_landmark(landmark_name, action=action, **kw)
    out = report.as_dict()
    out["action_pending"] = action if (report.arrived and action) else ""
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
    return json.dumps(state, indent=2)


def _known() -> list[str]:
    try:
        from slam import places
        return [row["name"] for row in places.all()]
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not list places: %s", exc)
        return []
