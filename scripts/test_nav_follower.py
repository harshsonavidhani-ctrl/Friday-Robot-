#!/usr/bin/env python3
"""Test nav_follower.py — file 14 steps 4-5. No hardware, no motors.

    scripts/test_nav_follower.py

WHY A SIMULATED CHASSIS. The thing worth testing is whether the loop CLOSES:
does it correct when the wheels under-deliver, does it stop when it stops
knowing where it is, does it believe it has arrived only when it has. On a
real run every one of those is confounded by the drivetrain -- a failure to
arrive could equally be a wheel that did not turn. The fake chassis here
delivers a configurable fraction of what it is asked for, so a test that fails
is the follower's fault and nothing else's.

⚠ THE SIMULATION IS DELIBERATELY UNFAIR. Its default is 70% of commanded
distance with noise on every move and a scan matcher that is worth a couple of
centimetres, because that is roughly what this robot does. A follower that
only converges on a perfect chassis has not been tested, it has been flattered.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
import nav_follower as nf                               # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


# --------------------------------------------------------------------------
# a chassis that under-delivers, and a matcher that is slightly wrong
# --------------------------------------------------------------------------

class FakeWorld:
    def __init__(self, pose=(0.0, 0.0, 0.0), cm_s=10.0, deg_s=40.0,
                 efficiency=0.7, noise_m=0.01, noise_deg=2.0,
                 match_noise_m=0.02, seed=14, dead=False) -> None:
        self.x, self.y, self.theta = pose
        self.cm_s, self.deg_s = cm_s, deg_s
        self.efficiency = efficiency
        self.noise_m, self.noise_deg = noise_m, noise_deg
        self.match_noise_m = match_noise_m
        self.dead = dead
        self.rng = np.random.default_rng(seed)
        self.commands: list[tuple[str, float]] = []

    def drive(self, direction: str, seconds: float) -> None:
        self.commands.append((direction, seconds))
        if self.dead:
            return
        if direction in ("left", "right"):
            deg = self.deg_s * seconds * self.efficiency
            deg += self.rng.normal(0.0, self.noise_deg)
            self.theta += math.radians(deg if direction == "left" else -deg)
            self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        else:
            m = self.cm_s * seconds * self.efficiency / 100.0
            m += self.rng.normal(0.0, self.noise_m)
            sign = 1.0 if direction == "forward" else -1.0
            self.x += sign * m * math.cos(self.theta)
            self.y += sign * m * math.sin(self.theta)

    def measured(self):
        n = self.rng.normal(0.0, self.match_noise_m, 2)
        return (self.x + n[0], self.y + n[1], self.theta)


class FakeMapper:
    def __init__(self, world: FakeWorld, trusted=True) -> None:
        self.world = world
        self.grid = SimpleNamespace(cell=config.SLAM_CELL_M)
        self.pose = world.measured()
        self.pose_trusted = trusted
        self.updates = 0

    def update(self, scan, moved_m=0.0, turned_rad=0.0, now=None):
        self.updates += 1
        self.pose = self.world.measured()
        return {"trusted": self.pose_trusted}


class FakeRobot:
    def __init__(self, world: FakeWorld, stop_after: int | None = None,
                 refuse=False) -> None:
        self.world = world
        self.stop_after = stop_after
        self.refuse = refuse
        self.sweeps = 0
        self.observes = 0
        self.said: list[str] = []

    async def sweep(self):
        self.sweeps += 1
        return {0: 1500, 90: 1200, 180: 900}

    async def observe(self):
        self.observes += 1

    async def drive_seconds(self, direction: str, seconds: float) -> str:
        if self.refuse:
            return "REFUSED: no fresh observation"
        self.world.drive(direction, seconds)
        return f"moved {direction} for {seconds:.1f}s"

    def say(self, text):
        self.said.append(text)

    def stopped(self) -> bool:
        return (self.stop_after is not None
                and self.world.commands.__len__() >= self.stop_after)


# --------------------------------------------------------------------------

def test_lookahead() -> None:
    print("\n— pure-pursuit geometry —")
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    pt, _ = nf.lookahead_point(path, (0.0, 0.0), 0.5)
    check("look-ahead sits on the path at the radius",
          abs(pt[0] - 0.5) < 1e-9 and abs(pt[1]) < 1e-9, f"{pt}")

    # A robot halfway along must not aim at the waypoint behind it.
    pt, _ = nf.lookahead_point(path, (1.4, 0.0), 0.4)
    check("it aims forward, never backward", pt[0] > 1.4, f"x={pt[0]:.2f}")

    # A path that doubles back past the robot must not pull the target back.
    back = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    pt, _ = nf.lookahead_point(back, (1.0, 0.02), 0.5)
    check("a doubling-back path still aims forward", pt[0] > 1.0, f"{pt}")

    pt, _ = nf.lookahead_point(path, (1.95, 0.0), 2.0)
    check("when the path is all inside the circle, aim at the end",
          abs(pt[0] - 2.0) < 1e-9, f"{pt}")
    check("a single-point path aims at that point",
          tuple(nf.lookahead_point(np.array([[3.0, 4.0]]), (0, 0), 1.0)[0])
          == (3.0, 4.0))

    check("closest_index finds the segment, not the nearest waypoint",
          nf.closest_index(path, (1.4, 0.0)) == 1,
          str(nf.closest_index(path, (1.4, 0.0))))


def test_heading_and_curvature() -> None:
    print("\n— heading error and curvature —")
    check("straight ahead is zero error",
          abs(nf.heading_error((0, 0, 0), (1, 0))) < 1e-12)
    check("left is positive",
          nf.heading_error((0, 0, 0), (0, 1)) > 0)
    check("it wraps the short way",
          abs(nf.heading_error((0, 0, math.pi - 0.1), (-1, -0.001)) ) < 0.2,
          f"{nf.heading_error((0, 0, math.pi - 0.1), (-1, -0.001)):.3f}")
    check("a target dead ahead has no curvature",
          abs(nf.curvature((0, 0, 0), (1.0, 0.0))) < 1e-12)
    k = abs(nf.curvature((0, 0, 0), (0.5, 0.5)))
    check("a target off to the side curves", k > 0.5, f"{k:.2f}")
    check("curvature is finite at the target",
          nf.curvature((0, 0, 0), (0, 0)) == 0.0)


def test_rate_model() -> None:
    print("\n— learning the rates —")
    r = nf.RateModel(forward_cm_s=10.0, turn_deg_s=40.0)
    check("seconds scale with distance",
          abs(r.seconds_for(0.5) - 5.0) < 1e-9)
    r.observe_drive(1.0, 0.20)                    # really did 20 cm in 1 s
    check("a real measurement moves the estimate", r.forward_cm_s > 10.0,
          f"{r.forward_cm_s:.1f} cm/s")
    before = r.forward_cm_s
    r.observe_drive(1.0, 0.001)                   # below the matcher's noise
    check("noise-level motion teaches nothing", r.forward_cm_s == before)

    stalled = nf.RateModel()
    for _ in range(30):
        stalled.observe_drive(1.0, 0.0)
    check("a stalled wheel cannot drag the rate to zero",
          stalled.forward_cm_s >= nf.RATE_MIN_CM_S)
    fast = nf.RateModel()
    for _ in range(30):
        fast.observe_turn(1.0, 5000.0)
    check("nor can a bad match send it to infinity",
          fast.turn_deg_s <= nf.RATE_MAX_DEG_S)

    r2 = nf.RateModel.from_calibration()
    cal = config.load_calibration()
    uncal = not (cal.get("forward_cm_per_s") and cal.get("turn_deg_per_s"))
    check("it reports honestly whether it is calibrated",
          r2.calibrated != uncal,
          f"calibration.json forward_cm_per_s={cal.get('forward_cm_per_s')}")


def test_arrival_tolerance_is_derived() -> None:
    print("\n— arrival tolerance —")
    m = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05)
    expect = config.NAV_ROBOT_RADIUS_M + 0.05 * math.sqrt(2)
    check("tolerance = robot radius + one cell diagonal",
          abs(m.tolerance - expect) < 1e-12, f"{m.tolerance:.3f} m")
    check("it is not a round number", round(m.tolerance, 2) != m.tolerance)
    coarse = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.20)
    check("a coarser map earns a looser tolerance",
          coarse.tolerance > m.tolerance,
          f"{m.tolerance:.3f} -> {coarse.tolerance:.3f}")
    check("the loose fallback is go_to's existing number",
          abs(m.loose - config.GOTO_ARRIVE_M) < 1e-12)


def test_arrival_needs_to_hold() -> None:
    print("\n— arrival is held, not sampled —")
    m = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05)
    far = m.update((5.0, 0.0), 0.0, now=100.0)
    check("far away is not arrival", not far.arrived)

    first = m.update((0.05, 0.0), 0.0, now=101.0)
    check("one good sample is not arrival", not first.arrived, first.why)
    second = m.update((0.05, 0.0), 0.0, now=102.5)
    check("two samples across a second is", second.arrived, second.why)

    # A single sample inside, then outside, must reset the dwell.
    m2 = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05)
    m2.update((0.05, 0.0), 0.0, now=200.0)
    m2.update((3.00, 0.0), 0.0, now=201.0)          # matcher jumped, or drove off
    again = m2.update((0.05, 0.0), 0.0, now=202.0)
    check("leaving the tolerance resets the window", not again.arrived,
          again.why)

    # Commanded velocity must be near zero.
    m3 = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05)
    m3.update((0.05, 0.0), 0.0, now=300.0)
    busy = m3.update((0.05, 0.0), 0.20, now=302.0)
    check("still correcting is not arrival", not busy.arrived, busy.why)
    settled = m3.update((0.05, 0.0), 0.0, now=303.0)
    check("and arrival follows once it stops correcting", settled.arrived,
          settled.why)


def test_arrival_relaxes() -> None:
    print("\n— giving up on the tight tolerance —")
    m = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05, settle_extra_s=10.0)
    just_outside = m.tolerance + 0.05
    t = 0.0
    m.update((just_outside, 0.0), 0.0, now=t)
    check("it does not relax immediately",
          not m.update((just_outside, 0.0), 0.0, now=t + 1.0).relaxed)
    late = m.update((just_outside, 0.0), 0.0, now=t + 11.0)
    check("after the extra budget the tolerance loosens", late.relaxed,
          f"{late.tolerance_m:.2f} m")
    held = m.update((just_outside, 0.0), 0.0, now=t + 13.0)
    check("and close enough is then accepted", held.arrived, held.why)
    check("the report says it was relaxed", held.relaxed)

    # Relaxing only starts once she is actually near the goal.
    far = nf.ArrivalMonitor((0.0, 0.0), cell_m=0.05, settle_extra_s=10.0)
    far.update((3.0, 0.0), 0.0, now=0.0)
    check("the clock does not run while still far away",
          not far.update((3.0, 0.0), 0.0, now=100.0).relaxed)


def _run(world, robot=None, path=None, goal=None, **kw):
    robot = robot or FakeRobot(world)
    mapper = FakeMapper(world)
    path = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]) if path is None else path
    follower = nf.Follower(robot, rates=nf.RateModel(10.0, 40.0))
    monitor = nf.ArrivalMonitor(goal if goal is not None else path[-1],
                                cell_m=0.05, dwell_s=0.0, min_samples=2)
    report = asyncio.run(follower.follow(mapper, path, goal_xy=goal,
                                         arrival=monitor, **kw))
    return report, robot, mapper


def test_follows_to_the_goal() -> None:
    print("\n— it actually gets there —")
    world = FakeWorld()
    report, robot, _ = _run(world)
    check("it arrives", report.arrived, report.why)
    dist = math.hypot(world.x - 2.0, world.y)
    check("and is really there, in the world's own frame",
          dist <= report.tolerance_m + 0.1, f"{dist:.2f} m from the goal")
    check("it used the guarded interface for every move",
          all(d in ("forward", "backward", "left", "right")
              for d, _ in world.commands))
    check("it observed before driving forward", robot.observes > 0)
    check("it re-measured the pose every tick",
          robot.sweeps >= report.ticks, f"{robot.sweeps} sweeps / "
          f"{report.ticks} ticks")


def test_corrects_a_weak_drivetrain() -> None:
    print("\n— closing the loop on a chassis that under-delivers —")
    weak = FakeWorld(efficiency=0.45, cm_s=10.0)
    report, _, _ = _run(weak)
    check("a 45%-efficient chassis still arrives", report.arrived, report.why)
    check("it learned the real rate is lower than commanded",
          report.ticks > 2, f"{report.ticks} ticks")

    off = FakeWorld(pose=(0.0, 0.8, 0.0))          # starts a metre off the path
    report2, _, _ = _run(off)
    check("a robot starting off the path is steered back onto it",
          report2.arrived, report2.why)
    check("it turned to do it", report2.turned_deg > 10.0,
          f"{report2.turned_deg:.0f}°")


def test_refuses_without_a_trusted_pose() -> None:
    print("\n— it stops when it stops knowing where it is —")
    world = FakeWorld()
    robot = FakeRobot(world)
    mapper = FakeMapper(world, trusted=False)
    follower = nf.Follower(robot, rates=nf.RateModel(10.0, 40.0))
    report = asyncio.run(follower.follow(
        mapper, np.array([[0.0, 0.0], [2.0, 0.0]])))
    check("it refuses to drive", not report.arrived)
    check("and says why", "do not know where I am" in report.why, report.why)
    check("it did not move a wheel", not world.commands,
          f"{len(world.commands)} commands issued")


def test_detects_a_stall() -> None:
    print("\n— the wheels that do not turn —")
    dead = FakeWorld(dead=True)
    report, _, _ = _run(dead, max_ticks=40)
    check("it gives up rather than driving on a guessed heading",
          not report.arrived)
    check("and names the reason", "not driving" in report.why, report.why)
    check("it stopped quickly", report.ticks <= nf.STALL_TICKS + 3,
          f"{report.ticks} ticks")


def test_honours_stop_and_budgets() -> None:
    print("\n— brakes —")
    world = FakeWorld()
    report, _, _ = _run(world, robot=FakeRobot(world, stop_after=2))
    check("stop() ends the run", not report.arrived and "stop" in report.why,
          report.why)

    report2, _, _ = _run(FakeWorld(), max_ticks=2)
    check("the tick budget is honoured",
          report2.ticks <= 2 and "tick budget" in report2.why, report2.why)

    report3, _, _ = _run(FakeWorld(), budget_m=0.05)
    check("the distance budget is honoured",
          "distance budget" in report3.why, report3.why)

    refusing = FakeWorld()
    report4, _, _ = _run(refusing, robot=FakeRobot(refusing, refuse=True),
                         max_ticks=20)
    check("a refusal from the guard does not become a silent retry loop",
          not report4.arrived, report4.why)


def test_reports_uncalibrated() -> None:
    print("\n— honesty about the numbers —")
    world = FakeWorld()
    robot = FakeRobot(world)
    mapper = FakeMapper(world)
    follower = nf.Follower(robot)                  # from_calibration()
    report = asyncio.run(follower.follow(
        mapper, np.array([[0.0, 0.0], [0.6, 0.0]]),
        arrival=nf.ArrivalMonitor((0.6, 0.0), cell_m=0.05,
                                  dwell_s=0.0, min_samples=2),
        max_ticks=30))
    cal = config.load_calibration()
    uncal = not (cal.get("forward_cm_per_s") and cal.get("turn_deg_per_s"))
    check("the report says whether it ran on measured rates",
          report.calibrated != uncal,
          f"calibrated={report.calibrated}")
    check("every move is in the event log with what it actually did",
          all("moved_m" in e and "commanded_s" in e for e in report.events),
          f"{len(report.events)} events")


def test_does_not_touch_the_motors() -> None:
    """The architectural rule, asserted against the CODE rather than the prose.

    Scanned with `ast`, not by searching the text: the module's own docstring
    names `motor.py`, `hardware.motors()` and `Motors.drive()` in order to
    forbid them, and a substring search cannot tell a prohibition from a call.
    """
    import ast

    src = (Path(__file__).resolve().parent.parent / "nav_follower.py").read_text()
    tree = ast.parse(src)

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
            if node.module == "mcp_server.tools":
                imported.update(a.name for a in node.names)

    check("it never imports motor.py", "motor" not in imported,
          str(sorted(imported)))
    check("it does import the guarded movement tool", "movement" in imported)

    calls = {ast.unparse(n.func) for n in ast.walk(tree)
             if isinstance(n, ast.Call)}
    for banned in ("hardware.motors", "motor.Motors", "Motors"):
        check(f"it never calls {banned}()", banned not in calls)
    # movement.move is handed to anyio's run_sync rather than called inline
    # (a blocking serial move on the event loop would make stop() unanswerable),
    # so it is an attribute REFERENCE, not a call node.
    refs = {ast.unparse(n) for n in ast.walk(tree)
            if isinstance(n, ast.Attribute)}
    check("the only way it moves is movement.move", "movement.move" in refs,
          str(sorted(r for r in refs if "move" in r or "motor" in r)))
    check("and it goes out on a worker thread so stop() still answers",
          any("run_sync" in c for c in calls))


def _cast(grid, pose, max_m=6.0, step_m=0.02) -> dict[int, int]:
    """The sweep the LiDAR would return standing at `pose` on this map."""
    occ = grid.occupied_mask()
    nx, ny = occ.shape
    ox, oy = grid.origin
    out: dict[int, int] = {}
    for bearing in range(0, 360):
        world = pose[2] + math.radians(bearing - config.LIDAR_FORWARD_DEG)
        dx, dy = math.cos(world), math.sin(world)
        r = 0.10
        while r < max_m:
            cx = int((pose[0] + dx * r - ox) / grid.cell)
            cy = int((pose[1] + dy * r - oy) / grid.cell)
            if not (0 <= cx < nx and 0 <= cy < ny):
                break
            if occ[cx, cy]:
                out[bearing] = int(r * 1000)
                break
            r += step_m
    return out


def _open_run(grid, prep, length_m=0.30):
    """A plannable pose with `length_m` of plannable floor straight ahead."""
    cells = np.argwhere(prep.plannable)
    for i in np.linspace(0, len(cells) - 1, 60).astype(int):
        cx, cy = cells[i]
        x = (cx + 0.5) * grid.cell + grid.origin[0]
        y = (cy + 0.5) * grid.cell + grid.origin[1]
        for heading in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            ok = True
            for k in range(1, 8):
                d = length_m * k / 7
                px, py = x + d * math.cos(heading), y + d * math.sin(heading)
                ix = int((px - grid.origin[0]) / grid.cell)
                iy = int((py - grid.origin[1]) / grid.cell)
                if not (0 <= ix < prep.plannable.shape[0]
                        and 0 <= iy < prep.plannable.shape[1]
                        and prep.plannable[ix, iy]):
                    ok = False
                    break
            if ok and len(_cast(grid, (x, y, heading))) >= 120:
                return (x, y, heading)
    return None


def test_moves_are_measured_by_the_map() -> None:
    """The bug that stopped the follower driving on the real robot.

    Uses the REAL saved map and the REAL `slam.mapper.Mapper` -- the fakes
    above ignore motion priors, which is exactly why no suite saw this. Loaded
    fresh and never saved: `Mapper.update` only changes the copy in memory.
    """
    print("\n— what a move did, asked of the real map (2026-09-23 fix) —")
    import path_planner
    from slam.grid import OccupancyGrid
    from slam.mapper import Mapper

    if not config.SLAM_MAP_FILE.exists():
        print("    (no saved map on this machine — skipped)")
        return
    grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
    start = _open_run(grid, path_planner.preprocess(grid))
    if start is None:
        print("    (no 30 cm open run on the saved map — skipped)")
        return
    step = 0.30
    moved = (start[0] + step * math.cos(start[2]),
             start[1] + step * math.sin(start[2]), start[2])
    turned = (start[0], start[1], start[2] + math.radians(30.0))

    def fresh():
        return Mapper(grid=OccupancyGrid.load(config.SLAM_MAP_FILE),
                      pose=start)

    check("a 30 cm step reads as a full step",
          nf.motion_fraction(fresh(), _cast(grid, moved), step, 0.0) == 1.0)
    check("a commanded step that did not happen reads as NO step",
          nf.motion_fraction(fresh(), _cast(grid, start), step, 0.0) == 0.0)
    half = (start[0] + step / 2 * math.cos(start[2]),
            start[1] + step / 2 * math.sin(start[2]), start[2])
    check("a step that went half as far reads as half (a slipping wheel)",
          nf.motion_fraction(fresh(), _cast(grid, half), step, 0.0) == 0.5)
    check("a 30° turn reads as a full turn",
          nf.motion_fraction(fresh(), _cast(grid, turned), 0.0,
                             math.radians(30.0)) == 1.0)
    check("a commanded turn that did not happen reads as NO turn",
          nf.motion_fraction(fresh(), _cast(grid, start), 0.0,
                             math.radians(30.0)) == 0.0)

    # What the follower used to do: fold the sweep with no motion at all.
    before = fresh()
    before.update(_cast(grid, moved))
    lag = math.hypot(before.pose[0] - moved[0], before.pose[1] - moved[1])
    check("WITHOUT a prior the tracker cannot see a 30 cm step (the old bug)",
          lag > 0.15, f"placed {lag:.2f} m from the truth")

    class Sweeper:
        def __init__(self, scan):
            self.scan = scan

        async def sweep(self):
            return self.scan

    for label, truth, want in (("after a real step", moved, 1.0),
                               ("after a stalled step", start, 0.0)):
        mapper = fresh()
        follower = nf.Follower(Sweeper(_cast(grid, truth)),
                               rates=nf.RateModel(forward_cm_s=30.0,
                                                  turn_deg_s=60.0,
                                                  calibrated=True))
        metres, rad = follower.odometry("forward", 1.0)
        asyncio.run(follower._sweep_into(mapper, metres, rad))
        err = math.hypot(mapper.pose[0] - truth[0], mapper.pose[1] - truth[1])
        check(f"the follower places her right {label}",
              err < 0.06 and follower.last_fraction == want,
              f"{err * 100:.1f} cm off, fraction {follower.last_fraction}")

    check("a test double without a map gets the old behaviour, unchanged",
          nf.motion_fraction(SimpleNamespace(pose=(0, 0, 0)), {0: 1000},
                             0.3, 0.0) == 1.0)


def main() -> int:
    print("nav_follower.py — file 14 steps 4-5")
    test_lookahead()
    test_heading_and_curvature()
    test_rate_model()
    test_arrival_tolerance_is_derived()
    test_arrival_needs_to_hold()
    test_arrival_relaxes()
    test_follows_to_the_goal()
    test_corrects_a_weak_drivetrain()
    test_refuses_without_a_trusted_pose()
    test_detects_a_stall()
    test_moves_are_measured_by_the_map()
    test_honours_stop_and_budgets()
    test_reports_uncalibrated()
    test_does_not_touch_the_motors()
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
