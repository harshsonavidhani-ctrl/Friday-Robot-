"""mission.py — step 9: the loop that actually explores, and knows when to stop.

    report = await Mission(robot).run(mapper, budget_m=8.0, budget_s=180.0)

WHAT THIS IS. slam/explore.py DECIDES where to go next; nothing until now
DROVE there. This is that missing half, and it is deliberately written against
a small interface rather than against the hardware:

    robot.sweep()               -> {bearing: mm}
    robot.observe()             -> None        (satisfies the movement gate)
    robot.turn(degrees)         -> str         (what the tool said)
    robot.forward(cm)           -> str
    robot.say(text)             -> None
    robot.stopped()             -> bool        (has a HUMAN hit stop)
    robot.reflex_trips()        -> int         (safety interrupts so far)

So the whole mission is testable with a simulated robot and no motors, and the
real one is a thin adapter in mcp_server/tools/. slam/ still knows nothing
about serial ports, which is the invariant that keeps the safety interrupt
un-bypassable: every move here goes out through the MCP movement tools, which
run embodiment.check and safety.guarded_move, and neither can be overridden
from in here.

⚠ IT VERIFIES ITS OWN TURNS, and this is the part that matters most.

A commanded turn on this chassis does not always happen -- the motors can be
energised, buzz, and not break traction. When that happens odometry still says
"+30 degrees", the scan matcher cannot see the error because the truth is
outside its search window, and the robot then plans its next move from a
heading that is a lie. Everything downstream is wrong and nothing reports it.
Harsh caught this by watching; software did not.

So after every turn the mission compares the sweep before with the sweep after
and asks how much the world actually moved. A real pivot moves it by hundreds
of millimetres; a pivot that did not happen leaves it at sensor noise. If the
wheels did not turn, the mission STOPS. It does not retry harder, and it does
not carry on hoping -- a robot navigating on a heading it only believes in is
exactly the thing that drives into furniture.

A SAFETY TRIP IS NOT THE SAME AS BEING TOLD TO STOP, and conflating them
ended the first real run after one cycle. The reflex loop fires when something
is inside the stop distance ahead; that is the robot working, not the mission
failing, and the right answer is to turn away and re-plan rather than give up
where it stands. A human calling stop() means stop, always, immediately, and
is never recovered from. reflex.trips_since_start is what tells them apart.

Recoveries are capped. A robot that backs away and re-plans forever in a
corner it cannot leave is a robot that has hung, so after
MISSION_MAX_RECOVERIES it says so and stops.

BUDGETS ARE HARD AND THERE ARE THREE. Distance, wall-clock, and cycles. Any
one of them ending stops the run. A loop that explores "until finished" is one
bad frontier away from running until the battery dies.
"""

from __future__ import annotations

import math
import time

import config
from slam.explore import Explorer
from slam.narration import Narrator


def scan_shift_mm(before: dict, after: dict) -> float | None:
    """Median absolute range change between two sweeps, over shared bearings.

    The only measure here that does not come from the estimator, which is
    precisely why it is the one trusted to say whether the robot moved.
    """
    shared = [abs(after[b] - before[b]) for b in before if b in after]
    if len(shared) < 40:
        return None
    shared.sort()
    return float(shared[len(shared) // 2])


class Mission:
    """One autonomous exploration run, with the brakes built in."""

    def __init__(self, robot, explorer: Explorer | None = None,
                 narrator: Narrator | None = None) -> None:
        self.robot = robot
        self.explorer = explorer or Explorer()
        self.narrator = narrator or Narrator()
        self.said: list[str] = []
        self.events: list[dict] = []

    def _say(self, line: str | None) -> None:
        if line:
            self.said.append(line)
            self.robot.say(line)

    async def _fold(self, mapper, moved_m=0.0, turned_rad=0.0):
        scan = await self.robot.sweep()
        if scan:
            mapper.update(scan, moved_m, turned_rad)
        return scan

    async def run(self, mapper, budget_m: float | None = None,
                  budget_s: float | None = None,
                  max_cycles: int | None = None) -> dict:
        budget_m = config.MISSION_BUDGET_M if budget_m is None else budget_m
        budget_s = config.MISSION_BUDGET_S if budget_s is None else budget_s
        max_cycles = (config.MISSION_MAX_CYCLES if max_cycles is None
                      else max_cycles)

        started = time.monotonic()
        driven = 0.0
        cycles = 0
        recoveries = 0
        trips = self.robot.reflex_trips()
        stopped_because = "budget"

        # A map to match against before deciding anything. Standing still, so
        # no odometry is claimed.
        for _ in range(config.MISSION_SETTLE_SWEEPS):
            await self._fold(mapper)

        while True:
            if self.robot.stopped():
                stopped_because = "stop was called"
                break
            if cycles >= max_cycles:
                stopped_because = "cycle budget"
                break
            if driven >= budget_m:
                stopped_because = "distance budget"
                break
            if time.monotonic() - started >= budget_s:
                stopped_because = "time budget"
                break

            cycles += 1
            before_area = mapper.grid.free_mask().sum() * mapper.grid.cell ** 2

            closed = mapper.try_loop_closure()
            decision = self.explorer.decide(mapper)
            if decision.done:
                self._say(self.narrator.consider(decision=decision,
                                                 closure=closed,
                                                 now=time.monotonic()))
                stopped_because = decision.reason or "finished"
                break

            self._say(self.narrator.consider(decision=decision, closure=closed,
                                             now=time.monotonic()))

            for turn_deg, forward_m in decision.moves:
                if self.robot.stopped():
                    stopped_because = "stop was called"
                    break

                if abs(turn_deg) >= config.MISSION_MIN_TURN_DEG:
                    await self.robot.observe()
                    before = await self.robot.sweep()
                    said = await self.robot.turn(turn_deg)
                    if "REFUSED" in said.upper():
                        stopped_because = f"refused: {said}"
                        break
                    after = await self.robot.sweep()
                    shift = scan_shift_mm(before, after)
                    retried = False

                    # ONE VERIFIED RETRY. Measured on the real robot: a turn
                    # straight after a forward drive is the one that stalls.
                    # Fourteen pivots back to back in
                    # scripts/calibrate_turn.py never missed, including 45 and
                    # 60 degrees; two autonomous runs both stalled on the turn
                    # that followed a drive (-27.7 deg moved 0 mm, -52.1 deg
                    # moved 1 mm) right after a large turn had worked fine.
                    #
                    # A retry is only safe BECAUSE it is verified. If the
                    # first attempt genuinely did nothing -- and 0 or 1 mm is
                    # nothing -- then turning again lands the commanded angle
                    # exactly once, not twice. Without the sweep check this
                    # would be a coin flip on the heading, which is worse than
                    # stopping. Exactly one retry: a chassis that needs two is
                    # not a chassis to keep asking.
                    if (shift is not None
                            and shift < config.PIVOT_MIN_SCAN_SHIFT_MM):
                        retried = True
                        await self.robot.observe()
                        before = await self.robot.sweep()
                        said = await self.robot.turn(turn_deg)
                        if "REFUSED" in said.upper():
                            stopped_because = f"refused: {said}"
                            break
                        after = await self.robot.sweep()
                        shift = scan_shift_mm(before, after)

                    self.events.append({"turn_deg": turn_deg,
                                        "scan_shift_mm": shift,
                                        "retried": retried})
                    if (shift is not None
                            and shift < config.PIVOT_MIN_SCAN_SHIFT_MM):
                        # The wheels did not turn. Everything after this would
                        # be planned from a heading that is not true.
                        stopped_because = ("the wheels did not turn, twice "
                                           f"-- the world moved {shift:.0f} mm "
                                           f"for a {turn_deg:+.0f} degree "
                                           "command")
                        self._say("something is stopping me turning, "
                                  "so I am going to stand still")
                        break
                    await self._fold(mapper, 0.0, math.radians(turn_deg))

                # Clamped, never skipped: the planner may want a 2 m run and
                # she may only take it a bit at a time. See MISSION_MAX_LEG_M
                # -- there are things on this floor that nothing on this robot
                # can see until it is touching them.
                forward_m = min(forward_m, config.MISSION_MAX_LEG_M)
                if forward_m > 0.01:
                    if self.robot.stopped():
                        stopped_because = "stop was called"
                        break
                    # ⚠ CHECKED PER LEG, NOT PER CYCLE. Checking only at the
                    # top of the loop lets a cycle that drives two legs
                    # overshoot by a whole cycle's worth -- measured at 7.09 m
                    # against a 6.0 m budget, 18% over. A budget that can be
                    # exceeded is not a budget, and this one exists to stop
                    # her ending up somewhere she cannot get back from.
                    if driven + forward_m > budget_m:
                        stopped_because = "distance budget"
                        break
                    if time.monotonic() - started >= budget_s:
                        stopped_because = "time budget"
                        break
                    await self.robot.observe()
                    said = await self.robot.forward(forward_m * 100.0)
                    now_trips = self.robot.reflex_trips()
                    if "REFUSED" in said.upper():
                        stopped_because = f"refused: {said}"
                        break
                    # Something got close and the reflex stopped her. She
                    # moved SOME of the way, so the odometry claim is now a
                    # guess -- fold the sweep in with no motion claimed and
                    # let the matcher place her, rather than asserting a
                    # distance the interrupt cut short.
                    if now_trips > trips:
                        trips = now_trips
                        recoveries += 1
                        await self._fold(mapper)
                        if recoveries > config.MISSION_MAX_RECOVERIES:
                            stopped_because = ("boxed in -- the safety "
                                               "interrupt keeps firing")
                            self._say("I keep finding things in the way, "
                                      "so I am going to stop here")
                            break
                        self._say("something is close, backing off and "
                                  "trying another way")
                        await self.robot.observe()
                        away = await self.robot.turn(
                            config.MISSION_RECOVER_TURN_DEG)
                        if "REFUSED" not in away.upper():
                            await self._fold(mapper, 0.0, math.radians(
                                config.MISSION_RECOVER_TURN_DEG))
                        break            # re-plan from where she actually is
                    driven += forward_m
                    await self._fold(mapper, forward_m, 0.0)
            gained = (mapper.grid.free_mask().sum() * mapper.grid.cell ** 2
                      - before_area)
            self.explorer.note(mapper, gained)
            self._say(self.narrator.consider(gained_m2=gained,
                                             now=time.monotonic()))
            # A recovery is not a reason to end the run -- it is a reason to
            # re-plan, which is what the next cycle does. Only a real stop
            # reason ends it.
            if stopped_because not in ("budget",):
                break

        return {
            "stopped_because": stopped_because,
            "cycles": cycles,
            "driven_m": round(driven, 2),
            "seconds": round(time.monotonic() - started, 1),
            "floor_m2": round(float(mapper.grid.free_mask().sum())
                              * mapper.grid.cell ** 2, 1),
            "wall_cells": int(mapper.grid.occupied_mask().sum()),
            "closures": mapper.closures,
            "recoveries": recoveries,
            "said": self.said,
            "turns": self.events,
        }
