#!/usr/bin/env python3
"""Test slam/navigate.py against a synthetic room. No hardware, no motors.

    scripts/test_navigate.py

WHY A FAKE ROBOT AND NOT THE REAL ONE. The thing worth testing here is the
DECIDING -- does she route around a wall, does she re-plan, does she refuse
when there is no way through -- and that is exactly the part that a hardware
run cannot isolate, because a real failure to arrive could equally be a wheel
that did not turn. The fake robot executes every commanded move perfectly, so
anything that goes wrong in these tests is the planner's fault and nothing
else's.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402
from slam.navigate import Navigator                     # noqa: E402


# The room, as geometry rather than as pixels: an outer box with a dividing
# wall at x=3 that has a 0.8 m doorway in it.
BOUNDS = (0.0, 6.0, 0.0, 4.0)
DOORWAY = (2.6, 3.4)


def _hits_wall(x: float, y: float, sealed: bool = False) -> bool:
    x0, x1, y0, y1 = BOUNDS
    if not (x0 < x < x1 and y0 < y < y1):
        return True
    if abs(x - 3.0) < 0.05:
        if sealed or not (DOORWAY[0] <= y <= DOORWAY[1]):
            return True
    return False


def _cast(px: float, py: float, sealed: bool) -> np.ndarray:
    """One 360-beam sweep from (px, py), as world endpoints.

    Ray-cast rather than hand-placed, because `integrate` carves the cells a
    beam PASSED THROUGH towards free and marks only where it stopped. Handing
    it a list of wall points and a list of floor points -- the obvious thing --
    does not build a map, it builds a contradiction.
    """
    ends = []
    for deg in range(0, 360):
        a = math.radians(deg)
        r = 0.05
        while r < 8.0:
            x, y = px + r * math.cos(a), py + r * math.sin(a)
            if _hits_wall(x, y, sealed):
                break
            r += 0.02
        ends.append((px + r * math.cos(a), py + r * math.sin(a)))
    return np.array(ends)


def room_with_a_doorway(sealed: bool = False) -> OccupancyGrid:
    """A 6x4 m room split at x=3, with a doorway unless `sealed`.

    The only way from (1, 1) to (5, 3) is through the gap, so a planner that
    ignores the wall shows up immediately as a route straight through it.
    """
    grid = OccupancyGrid(cell_m=0.05)
    # Viewpoints on both sides, so both halves are actually observed. A cell
    # nobody looked at is UNKNOWN, and unknown is not passable.
    for px, py in [(1.0, 1.0), (1.0, 3.0), (2.0, 3.0), (2.9, 3.0),
                   (3.1, 3.0), (4.0, 3.0), (5.0, 3.0), (5.0, 1.0)]:
        if sealed and px > 3.0:
            # With the door bricked up she could never have seen the far side;
            # but the test is about routing, not about how it got mapped, so
            # the far side is still mapped from its own viewpoints.
            pass
        grid.integrate(_cast(px, py, sealed), (px, py))
    return grid


class FakeMapper:
    """Perfect localisation: the estimate is whatever the robot really did.

    ONE SOURCE OF TRUTH. The first version of this had FakeRobot move the pose
    AND FakeMapper.update move it again, so every leg was applied twice and she
    drove off the map -- a bug in the test that looked exactly like a bug in the
    planner. The real _Robot never touches the mapper; only mapper.update does.
    """

    def __init__(self, grid, robot_pose_source):
        self.grid = grid
        self._robot = robot_pose_source
        self.pose = tuple(self._robot.pose)

    def update(self, scan, moved_m=0.0, turned_rad=0.0):
        self.pose = tuple(self._robot.pose)


class FakeRobot:
    """Owns the true pose and executes every move exactly as commanded."""

    def __init__(self, grid_sealed: bool = False,
                 refuse_after: int | None = None, never_turns: bool = False):
        self.pose = (1.0, 1.0, 0.0)
        self.moves: list[tuple[str, float]] = []
        self.observes = 0
        self.refuse_after = refuse_after
        self.never_turns = never_turns
        self.sealed = grid_sealed
        self._stop = False

    async def sweep(self):
        """A real sweep from where she actually is.

        It has to be pose-dependent: scan_shift_mm compares two sweeps to prove
        a turn happened, and a constant sweep would report every turn as dead.
        """
        ends = _cast(self.pose[0], self.pose[1], self.sealed)
        out = {}
        for deg, (ex, ey) in enumerate(ends):
            rel = math.degrees(math.atan2(ey - self.pose[1], ex - self.pose[0]))
            bearing = int(round((rel - math.degrees(self.pose[2])) % 360))
            out[bearing % 360] = int(math.hypot(ex - self.pose[0],
                                                ey - self.pose[1]) * 1000)
        return out

    async def observe(self):
        self.observes += 1

    async def turn(self, degrees):
        self.moves.append(("turn", degrees))
        if self.refuse_after is not None and len(self.moves) > self.refuse_after:
            return "REFUSED: too close to something"
        if not self.never_turns:
            x, y, th = self.pose
            self.pose = (x, y, th + math.radians(degrees))
        return f"turned {degrees:.0f} degrees"

    async def forward(self, cm):
        self.moves.append(("forward", cm / 100.0))
        if self.refuse_after is not None and len(self.moves) > self.refuse_after:
            return "REFUSED: something is in the way"
        x, y, th = self.pose
        self.pose = (x + (cm / 100.0) * math.cos(th),
                     y + (cm / 100.0) * math.sin(th), th)
        return f"moved {cm:.0f} cm"

    def say(self, text): pass
    def reflex_trips(self): return 0
    def stopped(self): return self._stop


def check(name, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {name}"
          + (f"  — {detail}" if detail else ""))
    return bool(condition)


async def main() -> int:
    grid = room_with_a_doorway()
    ok = True

    print("\n1. she routes THROUGH the doorway, not through the wall")
    robot = FakeRobot()
    mapper = FakeMapper(grid, robot)
    report = await Navigator(robot).go(mapper, (5.0, 3.0), name="the door",
                                       budget_m=30.0, budget_s=60.0,
                                       max_legs=400)
    ok &= check("arrived", report["arrived"], report["why"])
    ok &= check("scanned before every move",
                robot.observes >= len([m for m in robot.moves]),
                f"{robot.observes} scans / {len(robot.moves)} moves")
    # Crossing x=3 anywhere but the gap would mean it drove through the wall.
    ok &= check("final position is the target",
                report["remaining_m"] <= 0.45, f"{report['remaining_m']} m left")

    print("\n2. an unreachable place is REFUSED, not approximated")
    sealed = room_with_a_doorway(sealed=True)
    robot2 = FakeRobot(grid_sealed=True)
    mapper2 = FakeMapper(sealed, robot2)
    report2 = await Navigator(robot2).go(mapper2, (5.0, 3.0), name="the door",
                                         budget_m=30.0, budget_s=60.0)
    ok &= check("did not arrive", not report2["arrived"])
    ok &= check("said there is no route", "no route" in report2["why"],
                report2["why"])
    ok &= check("did not drive at the wall", report2["driven_m"] == 0.0,
                f"drove {report2['driven_m']} m")

    print("\n3. a refusal from the safety layer ends the run")
    robot3 = FakeRobot(refuse_after=2)
    mapper3 = FakeMapper(grid, robot3)
    report3 = await Navigator(robot3).go(mapper3, (5.0, 3.0), name="the door",
                                         budget_m=30.0, budget_s=60.0)
    ok &= check("stopped on the refusal", "refused" in report3["why"],
                report3["why"])

    print("\n4. wheels that do not turn stop the run rather than lying")
    robot4 = FakeRobot(never_turns=True)
    mapper4 = FakeMapper(grid, robot4)
    # scan_shift_mm sees identical sweeps, so every turn reads as 0 mm moved.
    report4 = await Navigator(robot4).go(mapper4, (5.0, 3.0), name="the door",
                                         budget_m=30.0, budget_s=60.0)
    ok &= check("stopped on the dead turn", "did not turn" in report4["why"],
                report4["why"])

    print("\n5. stop() is obeyed between moves")
    robot5 = FakeRobot()
    mapper5 = FakeMapper(grid, robot5)
    robot5._stop = True
    report5 = await Navigator(robot5).go(mapper5, (5.0, 3.0), name="the door")
    ok &= check("stopped immediately", report5["why"] == "stop was called",
                report5["why"])
    ok &= check("never moved", not robot5.moves, f"{len(robot5.moves)} moves")

    print(f"\n{'ALL PASSED' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
