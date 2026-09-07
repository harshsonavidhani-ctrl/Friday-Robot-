"""navigate.py — step 9 §9: drive to a place Friday already knows by name.

    "go to the door"  ->  places.find("door")  ->  a route on the occupancy
                          grid  ->  short verified legs, re-planned each time

⚠ THIS IS THE SECOND TOOL THAT DRIVES WITHOUT A HUMAN CHOOSING EACH MOVE, and
it is built on the same rules as the first. It does not touch the motors. Every
move goes out through `movement.turn_degrees` / `movement.move_distance` via the
caller's robot object, so each one still runs `embodiment.check` (has she
actually looked?) and `safety.guarded_move` (the clearance interrupt, the
duration cap, the 100 ms re-check while moving). There is no fast path here and
there must never be one.

WHY IT RE-PLANS AFTER EVERY LEG instead of following one route to the end.
`DistanceField` is a snapshot of what the map said at one moment, and the map
is the thing that changes most while she drives -- a route planned across a
region she has never seen is a guess about cells nobody has looked at. Driving
a stale plan is how a robot walks confidently into the one chair that was not
on the map yet. Re-planning costs one Dijkstra per leg over a grid that is
already in memory, and it means an obstacle discovered on leg three is routed
around on leg four rather than driven into.

WHAT IT DOES NOT DO. It does not relocalise. The caller hands in a `Mapper`
whose pose it trusts, exactly as `mission.py` does, and if that pose is wrong
this drives confidently to the wrong address. Placing her in a saved map is a
different problem -- see `ControlLoop.relocalise` in rl/pi_deploy.py, which
solves it for the RL stack and is deliberately not reached into from here.

HOW IT STOPS. Six ways, and any one of them is enough:

  * stop() -- checked between every single move.
  * Three budgets: metres driven, wall clock, and legs taken.
  * A refusal from the movement tools ends the run rather than being retried.
  * A TURN THAT DID NOT HAPPEN, by the same sweep comparison mission.py uses.
    This chassis can energise, buzz, and not break traction while odometry
    still reports the turn -- so everything planned afterwards would come off
    a heading that is a lie.
  * NO ROUTE. An unreachable destination is reported, never approximated by
    driving in its general direction.
"""

from __future__ import annotations

import logging
import math
import time

import config
from slam.mission import scan_shift_mm
from slam.plan import DistanceField, path_to_moves

log = logging.getLogger("friday.navigate")


class Navigator:
    """One "go to X" run, with the same brakes as an exploration mission."""

    def __init__(self, robot, narrator=None) -> None:
        self.robot = robot
        self.narrator = narrator
        self.said: list[str] = []
        self.events: list[dict] = []

    def _say(self, line: str | None) -> None:
        if line:
            self.said.append(line)
            self.robot.say(line)

    async def _fold(self, mapper, moved_m: float = 0.0,
                    turned_rad: float = 0.0):
        scan = await self.robot.sweep()
        if scan:
            mapper.update(scan, moved_m, turned_rad)
        return scan

    async def _turn(self, mapper, degrees: float) -> tuple[bool, str]:
        """Turn, and prove it happened. Returns (ok, reason-if-not).

        The verified retry is mission.py's, for the failure it found on the
        real robot: a turn straight after a forward drive is the one that
        stalls (-27.7 deg moved 0 mm, -52.1 deg moved 1 mm, both immediately
        after a drive). A retry is only safe BECAUSE it is verified -- if the
        first attempt genuinely did nothing, turning again lands the commanded
        angle exactly once. Exactly one retry: a chassis that needs two is not
        one to keep asking.
        """
        for attempt in (1, 2):
            # scan_environment first: the movement gate will not allow a move
            # until she has actually looked, and that gate is the point.
            await self.robot.observe()
            before = await self.robot.sweep()
            said = await self.robot.turn(degrees)
            if "REFUSED" in said.upper():
                return False, f"refused: {said}"
            after = await self.robot.sweep()
            shift = scan_shift_mm(before, after)
            self.events.append({"turn_deg": degrees, "scan_shift_mm": shift,
                                "attempt": attempt})
            if shift is None or shift >= config.PIVOT_MIN_SCAN_SHIFT_MM:
                await self._fold(mapper, turned_rad=math.radians(degrees))
                return True, ""
        return False, ("the wheels did not turn, twice — the heading would be "
                       "a guess from here on")

    async def go(self, mapper, target_xy, name: str = "there",
                 arrive_m: float | None = None,
                 budget_m: float | None = None,
                 budget_s: float | None = None,
                 max_legs: int | None = None) -> dict:
        """Drive to `target_xy`, re-planning every leg. Never raises."""
        arrive_m = config.GOTO_ARRIVE_M if arrive_m is None else arrive_m
        budget_m = config.GOTO_BUDGET_M if budget_m is None else budget_m
        budget_s = config.GOTO_BUDGET_S if budget_s is None else budget_s
        max_legs = config.GOTO_MAX_LEGS if max_legs is None else max_legs

        target = (float(target_xy[0]), float(target_xy[1]))
        started = time.monotonic()
        driven = 0.0
        legs = 0
        why = "budget"
        arrived = False

        # Look before deciding anything, standing still, so no odometry is
        # claimed for the settling sweeps.
        for _ in range(config.MISSION_SETTLE_SWEEPS):
            await self._fold(mapper)

        def remaining() -> float:
            return math.hypot(target[0] - mapper.pose[0],
                              target[1] - mapper.pose[1])

        self._say(f"Going to {name}. It is about "
                  f"{remaining():.1f} metres away.")

        while True:
            if self.robot.stopped():
                why = "stop was called"
                break
            if remaining() <= arrive_m:
                arrived, why = True, "arrived"
                break
            if legs >= max_legs:
                why = "leg budget"
                break
            if driven >= budget_m:
                why = "distance budget"
                break
            if time.monotonic() - started >= budget_s:
                why = "time budget"
                break

            legs += 1
            field = DistanceField(mapper.grid, (mapper.pose[0], mapper.pose[1]))
            path = field.path_to(target)
            if len(path) < 2:
                # NOT approximated by driving roughly that way. An unreachable
                # address is a fact worth reporting; a robot that heads in the
                # general direction of one is a robot driving at a wall.
                why = ("no route — nothing passable connects here to "
                       f"{name} on the map as it stands")
                break

            moves = path_to_moves(path, mapper.pose[2])
            if not moves:
                why = "no route (empty plan)"
                break

            # ONE leg per plan. The rest of the route is recomputed next time
            # round against a map that now includes whatever this leg revealed.
            turn_deg, forward_m = moves[0]
            forward_m = min(forward_m, config.MISSION_MAX_LEG_M)

            if abs(turn_deg) >= config.MISSION_MIN_TURN_DEG:
                ok, reason = await self._turn(mapper, turn_deg)
                if not ok:
                    why = reason
                    break
                if self.robot.stopped():
                    why = "stop was called"
                    break

            if forward_m > 0.01:
                await self.robot.observe()
                said = await self.robot.forward(forward_m * 100.0)
                if "REFUSED" in said.upper():
                    why = f"refused: {said}"
                    break
                await self._fold(mapper, moved_m=forward_m)
                driven += forward_m

        report = {
            "arrived": arrived,
            "place": name,
            "why": why,
            "legs": legs,
            "driven_m": round(driven, 2),
            "seconds": round(time.monotonic() - started, 1),
            "remaining_m": round(remaining(), 2),
            "pose": [round(v, 3) for v in mapper.pose],
            "said": self.said,
        }
        self._say("Here we are." if arrived
                  else f"I could not get to {name}: {why}.")
        log.info("go_to %s: %s", name, report)
        return report
