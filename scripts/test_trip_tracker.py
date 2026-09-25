#!/usr/bin/env python3
"""Tests for trip_tracker.py — tracking an explore trip without changing it.

    ./venv/bin/python scripts/test_trip_tracker.py

No hardware, no motors. The world is the REAL saved map (ray-cast sweeps) and
the matcher is the REAL slam.mapper.Mapper: the fakes elsewhere ignore motion
priors, which is the whole subject here. The saved map is loaded, never saved;
a hash check at the end proves it.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402

import config                                        # noqa: E402
import path_planner                                  # noqa: E402
import trip_tracker as tt                            # noqa: E402
from slam.grid import OccupancyGrid                  # noqa: E402
from slam.mapper import Mapper                       # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


def cast(grid, pose, rng=None, max_m=6.0, step_m=0.02) -> dict[int, int]:
    """The sweep the LiDAR would return at `pose`, with a little range noise."""
    occ = grid.occupied_mask()
    nx, ny = occ.shape
    ox, oy = grid.origin
    out: dict[int, int] = {}
    for bearing in range(0, 360):
        if any(lo <= bearing <= hi for lo, hi in config.LIDAR_SELF_MASK):
            continue
        world = pose[2] + math.radians(bearing - config.LIDAR_FORWARD_DEG)
        dx, dy = math.cos(world), math.sin(world)
        r = 0.10
        while r < max_m:
            cx = int((pose[0] + dx * r - ox) / grid.cell)
            cy = int((pose[1] + dy * r - oy) / grid.cell)
            if not (0 <= cx < nx and 0 <= cy < ny):
                break
            if occ[cx, cy]:
                noise = rng.gauss(0.0, 8.0) if rng else 0.0
                out[bearing] = int(r * 1000 + noise)
                break
            r += step_m
    return out


class SimRobot:
    """The mission's six calls, on a drivetrain that does not always obey.

    `turn_plan` is the fraction of each commanded turn that really happens,
    in order: 1.0 obeys, 0.5 slips halfway, 0.0 buzzes and stays put. Forward
    moves stop short of anything plannable-map says is solid.
    """

    def __init__(self, grid, pose, turn_plan=(1.0,), drive_plan=(1.0,),
                 seed=1) -> None:
        self.grid = grid
        self.prep = path_planner.preprocess(grid)
        self.truth = tuple(float(v) for v in pose)
        self.turn_plan = list(turn_plan)
        self.drive_plan = list(drive_plan)
        self.turns = 0
        self.drives = 0
        self.rng = random.Random(seed)
        self.calls: list[str] = []

    def _next(self, plan, i):
        return plan[i % len(plan)]

    async def sweep(self):
        self.calls.append("sweep")
        return cast(self.grid, self.truth, self.rng)

    async def observe(self):
        self.calls.append("observe")

    async def turn(self, degrees):
        self.calls.append("turn")
        f = self._next(self.turn_plan, self.turns)
        self.turns += 1
        x, y, th = self.truth
        th = (th + math.radians(degrees) * f + math.pi) % (2 * math.pi) - math.pi
        self.truth = (x, y, th)
        return f"turned {degrees:+.0f} degrees"

    async def forward(self, cm):
        self.calls.append("forward")
        f = self._next(self.drive_plan, self.drives)
        self.drives += 1
        x, y, th = self.truth
        want = cm / 100.0 * f
        went = 0.0
        while went + 0.02 <= want:
            nx_, ny_ = (x + (went + 0.02) * math.cos(th),
                        y + (went + 0.02) * math.sin(th))
            ix, iy = self.prep.grid.cells_of(np.array([[nx_, ny_]]))
            if not self.prep.plannable[int(ix[0]), int(iy[0])]:
                break
            went += 0.02
        self.truth = (x + went * math.cos(th), y + went * math.sin(th), th)
        return f"moved forward about {cm:.0f} cm"

    def say(self, text):
        pass

    def stopped(self):
        return False

    def reflex_trips(self):
        return 0


def _open_spot(grid, rng):
    prep = path_planner.preprocess(grid)
    cells = np.argwhere(prep.plannable)
    cx, cy = cells[rng.randrange(len(cells))]
    return ((cx + 0.5) * grid.cell + grid.origin[0],
            (cy + 0.5) * grid.cell + grid.origin[1],
            rng.uniform(-math.pi, math.pi))


def _err(pose, truth):
    d = math.hypot(pose[0] - truth[0], pose[1] - truth[1])
    a = abs((pose[2] - truth[2] + math.pi) % (2 * math.pi) - math.pi)
    return d, math.degrees(a)


def _way_home(pose, start):
    """The trip home as the robot would see it: (distance, bearing from her nose)."""
    dx, dy = start[0] - pose[0], start[1] - pose[1]
    bearing = math.atan2(dy, dx) - pose[2]
    return (math.hypot(dx, dy),
            math.degrees((bearing + math.pi) % (2 * math.pi) - math.pi))


# Mission-like moves: a turn of 10-90 degrees, then up to 25 cm.
MOVES = [(40, 25), (-60, 25), (0, 25), (75, 25), (-30, 25), (50, 25),
         (0, 25), (-80, 25), (35, 25), (0, 25), (-45, 25), (60, 25)]


async def _drive(robot, moves, mission_mapper=None):
    """Drive like mission.py does, sweeping where it sweeps."""
    for turn_deg, fwd_cm in moves:
        if abs(turn_deg) >= config.MISSION_MIN_TURN_DEG:
            await robot.observe()
            await robot.sweep()                       # before
            await robot.turn(turn_deg)
            await robot.sweep()                       # after
            scan = await robot.sweep()                # _fold
            if mission_mapper is not None:
                mission_mapper.update(scan, 0.0, math.radians(turn_deg))
        await robot.observe()
        await robot.forward(fwd_cm)
        scan = await robot.sweep()
        if mission_mapper is not None:
            mission_mapper.update(scan, fwd_cm / 100.0, 0.0)


def test_passes_everything_through() -> None:
    print("\n— the explorer's robot is untouched —")
    grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
    rng = random.Random(3)
    start = _open_spot(grid, rng)
    inner = SimRobot(grid, start)
    tracker = tt.TripTracker(grid, [start], certain=True)
    robot = tt.ListeningRobot(inner, tracker)

    async def go():
        a = await robot.turn(30)
        b = await robot.forward(20)
        s = await robot.sweep()
        return a, b, s

    a, b, s = asyncio.run(go())
    asyncio.run(tracker.drain())
    tracker.close()
    check("turn and forward return exactly what the robot said",
          a == "turned +30 degrees" and b == "moved forward about 20 cm")
    check("the sweep handed on is the robot's own sweep", len(s) > 100)
    check("stopped / reflex_trips / say go straight through",
          robot.stopped() is False and robot.reflex_trips() == 0)
    check("it commands nothing of its own",
          inner.calls == ["turn", "forward", "sweep"], str(inner.calls))
    check("and the sweep was folded", tracker.sweeps == 1)


def test_half_turns_do_not_lose_her() -> None:
    print("\n— turns that only half happen (the drivetrain on this robot) —")
    grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
    worst_mission, worst_tracker, home_errs = 0.0, 0.0, []
    for seed in range(4):
        rng = random.Random(10 + seed)
        start = _open_spot(grid, rng)
        # Every third turn slips halfway; one in six buzzes and stays put.
        inner = SimRobot(grid, start, turn_plan=(1.0, 1.0, 0.5, 1.0, 0.0, 0.5),
                         drive_plan=(1.0, 0.8, 1.0), seed=seed)
        tracker = tt.TripTracker(grid, [start], certain=True)
        robot = tt.ListeningRobot(inner, tracker)
        mission = Mapper(grid=copy.deepcopy(grid), pose=start)
        asyncio.run(_drive(robot, MOVES, mission))
        asyncio.run(tracker.drain())
        tracker.close()
        mapper, mark, certain = tracker.best()
        dm, am = _err(mission.pose, inner.truth)
        dt, at = _err(mapper.pose, inner.truth)
        worst_mission = max(worst_mission, dm)
        worst_tracker = max(worst_tracker, dt)
        # What matters for coming back: the way home as she believes it,
        # against the way home as it really is.
        believed = _way_home(mapper.pose, mark)
        real = _way_home(inner.truth, start)
        home_errs.append(abs(believed[0] - real[0]))
        print(f"    seed {seed}: explorer's own pose {dm*100:5.1f} cm / "
              f"{am:4.1f}°, tracker {dt*100:5.1f} cm / {at:4.1f}°")
    # 20 cm: inside the 28 cm arrival tolerance with room to spare. The ray
    # cast lands on the near face of a wall cell, which alone costs up to
    # ~16 cm in test_global_localise.
    check("the tracker keeps her within 20 cm through slipping turns",
          worst_tracker < 0.20, f"worst {worst_tracker*100:.1f} cm")
    check("and knows the way home to within 20 cm",
          max(home_errs) < 0.20,
          f"worst {max(home_errs)*100:.1f} cm")
    print(f"    (the explorer's own pose, same sweeps, commanded priors: "
          f"worst {worst_mission*100:.1f} cm)")


def _ambiguous_starts(grid, want=2, tries=12):
    rng = random.Random(7)
    found = []
    for _ in range(tries):
        pose = _open_spot(grid, rng)
        hyps, certain, _why = tt.start_hypotheses(grid, cast(grid, pose, rng))
        if not certain and len(hyps) == 2:
            found.append((pose, hyps))
            if len(found) >= want:
                break
    return found


def test_look_alike_starts() -> None:
    print("\n— a start that looks like somewhere else (the hostel room) —")
    grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
    starts = _ambiguous_starts(grid)
    if not starts:
        print("    (no look-alike start on this map in simulation — skipped)")
        return
    # Standing still is the same look over and over: never evidence.
    pose, hyps = starts[0]
    still = tt.TripTracker(grid, hyps, certain=False)
    rng = random.Random(5)
    for _ in range(12):
        still.note_sweep(cast(grid, pose, rng))
    asyncio.run(still.drain())
    still.close()
    check("twelve sweeps standing still do not settle a look-alike",
          not still.settled(), str(still.as_dict()["hypotheses"]))

    # Each start twice: as the search ranked it, and the other way round --
    # the explorer setting off on the WRONG look-alike is the case to fear.
    runs = [(p, h) for p, h in starts] + [(p, h[::-1]) for p, h in starts]
    for pose, hyps in runs:
        inner = SimRobot(grid, pose, turn_plan=(1.0, 0.5, 1.0),
                         seed=int(abs(pose[0]) * 100))
        tracker = tt.TripTracker(grid, hyps, certain=False)
        robot = tt.ListeningRobot(inner, tracker)
        asyncio.run(_drive(robot, MOVES))
        asyncio.run(tracker.drain())
        tracker.close()
        mapper, mark, certain = tracker.best()
        believed = _way_home(mapper.pose, mark)
        real = _way_home(inner.truth, pose)
        right = _err(hyps[0], pose)[0] < 0.3
        lead = tracker.ranked()[0]
        picked_true = _err(lead.start, pose)[0] < 0.3
        print(f"    truth ({pose[0]:.2f},{pose[1]:.2f}); candidates "
              f"{[tuple(round(v, 2) for v in h[:2]) for h in hyps]}; "
              f"first was {'right' if right else 'WRONG'}; trip picked "
              f"{'right' if picked_true else 'WRONG'}, certain={certain}; "
              f"way home off by {abs(believed[0]-real[0])*100:.0f} cm, "
              f"{abs(believed[1]-real[1]):.0f}°")
        check("a certain answer is a right one",
              (not certain) or picked_true)
        check("the way home is right whichever look-alike it followed",
              abs(believed[0] - real[0]) < 0.15
              and abs(believed[1] - real[1]) < 15.0)


def main() -> int:
    if not config.SLAM_MAP_FILE.exists():
        print("no saved map on this machine — nothing to test against")
        return 0
    before = hashlib.sha256(config.SLAM_MAP_FILE.read_bytes()).hexdigest()
    test_passes_everything_through()
    test_half_turns_do_not_lose_her()
    test_look_alike_starts()
    after = hashlib.sha256(config.SLAM_MAP_FILE.read_bytes()).hexdigest()
    check("the saved map on disk is unchanged", before == after)
    print(f"\n{_failures} failure(s)")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
