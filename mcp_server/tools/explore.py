"""explore.py — step 9: Friday explores a room on her own.

    explore_room()                 conservative defaults from config
    explore_room(budget_m=4)       a shorter leash

⚠ THIS IS THE ONLY TOOL IN THE PROJECT THAT DRIVES THE ROBOT WITHOUT A HUMAN
CHOOSING EACH MOVE. Everything about it is built around that.

WHAT IT DOES NOT DO. It does not touch the motors. Every move goes out through
movement.turn_degrees and movement.move_distance -- the same decorated tools
the brain and the dashboard call -- so each one runs embodiment.check (has she
actually looked?) and safety.guarded_move (the clearance interrupt, the
duration cap, the 100 ms re-check while moving). There is no fast path here
and there must never be one: a tool that drives autonomously is exactly the
wrong place to start trusting its own judgement about clearance.

HOW IT STOPS. Five ways, and any one of them is enough:

  * stop() -- hardware.ABORT is checked between every single move.
  * Three budgets: metres driven, wall-clock seconds, and decisions taken.
  * A refusal from the movement tools ends the run rather than being retried.
  * A TURN THAT DID NOT HAPPEN. This chassis can energise, buzz, and not break
    traction, and when it does odometry still reports the turn -- so the robot
    would plan everything after from a heading that is a lie. slam/mission.py
    compares the sweep before each turn with the sweep after and stops if the
    world did not move. See its docstring; this is the failure Harsh caught by
    watching, that no software noticed.

THE MAP STARTS EMPTY, ON PURPOSE. Continuing the saved map would mean knowing
where she is in it, and she does not -- relocalising into an old map is a
different problem and pretending to solve it would put her confidently in the
wrong room. Each run builds its own map and saves it at the end.
"""

from __future__ import annotations

import json
import logging

import anyio.to_thread

import config
import tts
from mcp_server import hardware, reflex
from mcp_server.registry import tool
from mcp_server.tools import mapping, movement, perception
from slam.mapper import Mapper
from slam.mission import Mission

log = logging.getLogger("friday.tools.explore")


class _Robot:
    """The mission's view of the real robot: six calls, no shortcuts.

    ⚠ EVERY TOOL CALL GOES THROUGH A WORKER THREAD, and that is a safety
    requirement rather than a performance one. registry.tool leaves the plain
    SYNCHRONOUS function callable and only the registered wrapper is async --
    so calling movement.turn_degrees() directly here would run a blocking
    serial move on the event loop, and stop() could not be answered until it
    finished. The whole point of hardware.ABORT is that it interrupts rather
    than queues. anyio.to_thread.run_sync is what the registry itself does,
    for exactly this reason.
    """

    def __init__(self) -> None:
        self._trips = 0

    async def sweep(self) -> dict[int, int]:
        return await anyio.to_thread.run_sync(mapping._sweep)

    async def observe(self) -> None:
        # Cheap -- range sensors only, no camera -- and it is what the
        # movement gate wants before it will allow anything to move.
        await anyio.to_thread.run_sync(perception.scan_environment)

    async def turn(self, degrees: float) -> str:
        return await anyio.to_thread.run_sync(movement.turn_degrees, degrees)

    async def forward(self, cm: float) -> str:
        return await anyio.to_thread.run_sync(movement.move_distance, cm)

    def say(self, text: str) -> None:
        try:
            tts.speak(text)
        except Exception as exc:            # noqa: BLE001
            # Losing her voice must not end a run that is otherwise going
            # fine; the line is still in the report either way.
            log.warning("could not speak %r: %s", text, exc)

    def reflex_trips(self) -> int:
        """How many times the safety interrupt has fired since boot.

        The mission uses this to tell a SAFETY TRIP from a HUMAN STOP. Both
        set hardware.ABORT, and treating them the same ended the first real
        run after one cycle: she drove 1.49 m, something came within 234 mm,
        the reflex did its job, and the mission reported "stop was called" and
        gave up. One of those is a reason to turn away and re-plan. The other
        is a reason to stop and stay stopped.
        """
        return int(reflex.status().get("trips_since_start", 0))

    def stopped(self) -> bool:
        """A HUMAN asked for stop -- never recovered from.

        Both stop() and the reflex set hardware.ABORT, and there is no
        separate flag for "a person did this". Rather than add one to
        safety.py -- which is devguard-protected and is the last file that
        should grow state to suit a caller -- the two are told apart from what
        is already there: the reflex bumps its trip counter when it fires and
        a human calling stop() does not. So ABORT set with no new trip is a
        person.

        guarded_move clears ABORT when it starts, so a set flag outside a move
        genuinely means something interrupted rather than a stale bit.
        """
        tripped = self.reflex_trips()
        was, self._trips = self._trips, tripped
        if not hardware.ABORT.is_set():
            return False
        if tripped != was:
            # A trip we have not accounted for yet. CLEAR IT, having noted it:
            # the reflex has already stopped the motors, and the flag exists to
            # interrupt a move in flight, not to latch. Left set, it makes the
            # NEXT check -- with the counter no longer moving -- look exactly
            # like a person calling stop, which is how a recovered run ended
            # with "stop was called" that nobody called. guarded_move clears
            # this at the start of every move anyway, so clearing it between
            # moves changes nothing except that the mission can now tell the
            # two apart.
            hardware.ABORT.clear()
            return False
        return True


@tool
async def explore_room(budget_m: float | None = None,
                       budget_seconds: float | None = None) -> str:
    """Explore this room on my own, mapping it as I go, then stop and report.

    I DRIVE MYSELF when you call this. I pick where to go, turn and drive
    there under the usual safety interrupt, and narrate what I am doing. Call
    stop() at any time and I stop between moves.

    I stop by myself when: there is nothing reachable left to explore, I run
    out of distance, time or decisions, a move is refused, or a turn I asked
    for does not actually happen.

    Args:
        budget_m: how far I may drive in total, in metres.
        budget_seconds: how long I may take.

    Returns JSON: why I stopped, how far I drove, how much floor I mapped,
    how many loops I closed, what I said, and every turn with how much the
    world actually moved for it.
    """
    budget_m = config.MISSION_BUDGET_M if budget_m is None else float(budget_m)
    budget_seconds = (config.MISSION_BUDGET_S if budget_seconds is None
                      else float(budget_seconds))
    if budget_m <= 0 or budget_seconds <= 0:
        return "a budget of zero means there is nothing to do. Refusing."

    hardware.ABORT.clear()
    robot = _Robot()
    robot._trips = robot.reflex_trips()   # baseline, not a stop
    mapper = Mapper(pose=(0.0, 0.0, 0.0))
    report = await Mission(robot).run(mapper, budget_m=budget_m,
                                         budget_s=budget_seconds)
    try:
        mapper.grid.save()
        report["map_saved"] = str(config.SLAM_MAP_FILE)
    except Exception as exc:                # noqa: BLE001
        report["map_saved"] = f"could not save the map: {exc}"
    log.info("explore_room finished: %s", report["stopped_because"])
    return json.dumps(report, indent=2)
