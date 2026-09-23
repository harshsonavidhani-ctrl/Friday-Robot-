#!/usr/bin/env python3
"""Test global_localise.py — can she find herself? No hardware, no motors.

    scripts/test_global_localise.py

HOW THIS GETS GROUND TRUTH WITHOUT DRIVING. A localiser is hard to test on a
real robot because nobody knows the true answer to compare against -- that
being the whole problem it exists to solve. So the scans here are RAY CAST
FROM THE MAP ITSELF at a pose we chose: march along each bearing until a cell
the map calls occupied, and that distance is what the LiDAR would have read
standing there. Feed that back in and the answer is checkable to the
centimetre.

⚠ A SYNTHETIC SCAN IS AN EASY SCAN, and the numbers here should be read with
that in mind. It has no glass, no chair legs moving between sweeps, no beams
returning off a person, and it comes from exactly the map being matched
against. Recovering the pose from it proves the SEARCH works; it does not
prove the robot will localise in a real room. What it does prove negatively
is worth more: if it cannot find a pose under these conditions, it certainly
cannot find one in a real room.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402
import global_localise as gl                            # noqa: E402
import path_planner as pp                               # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


def cast(grid, pose, max_m=6.0, step_m=0.02, bearings=None) -> dict[int, int]:
    """The sweep the LiDAR would return standing at `pose` on this map."""
    occ = grid.occupied_mask()
    nx, ny = occ.shape
    ox, oy = grid.origin
    out: dict[int, int] = {}
    forward = config.LIDAR_FORWARD_DEG
    for bearing in (bearings if bearings is not None else range(0, 360, 1)):
        # Bearing is measured in the LiDAR's own frame; points_from_scan undoes
        # `forward` again, so cast in the same convention it will decode.
        world = pose[2] + math.radians(bearing - forward)
        dx, dy = math.cos(world), math.sin(world)
        r = 0.10
        while r < max_m:
            cx = int((pose[0] + dx * r - ox) / grid.cell)
            cy = int((pose[1] + dy * r - oy) / grid.cell)
            if not (0 <= cx < nx and 0 <= cy < ny):
                break
            if occ[cx, cy]:
                out[int(bearing)] = int(r * 1000)
                break
            r += step_m
    return out


def _real_map():
    return OccupancyGrid.load(config.SLAM_MAP_FILE)


def _free_poses(grid, count=5):
    """A few spots she could actually be standing, spread across the map."""
    prep = pp.preprocess(grid)
    cells = np.argwhere(prep.plannable)
    if not len(cells):
        return []
    # Spread them out rather than taking neighbours.
    idx = np.linspace(0, len(cells) - 1, count).astype(int)
    out = []
    for i in idx:
        cx, cy = cells[i]
        out.append(((cx + 0.5) * grid.cell + grid.origin[0],
                    (cy + 0.5) * grid.cell + grid.origin[1]))
    return out


def test_recovers_a_known_pose() -> None:
    print("\n— can she find herself on the real map? —")
    grid = _real_map()
    prep = pp.preprocess(grid)
    errors, angles, times = [], [], []
    for (x, y) in _free_poses(grid, 5):
        for heading in (0.0, math.pi / 2, -2.2):
            truth = (x, y, heading)
            scan = cast(grid, truth)
            if len(scan) < 60:
                continue                    # standing somewhere that sees nothing
            t0 = time.perf_counter()
            fix = gl.localise(grid, scan, restrict_to=prep.plannable)
            times.append(time.perf_counter() - t0)
            if not fix.ok:
                print(f"    ({x:.2f},{y:.2f}) @{math.degrees(heading):>5.0f}° "
                      f"→ refused: {fix.why[:60]}")
                continue
            err = math.hypot(fix.pose[0] - x, fix.pose[1] - y)
            ang = abs((fix.pose[2] - heading + math.pi) % (2 * math.pi) - math.pi)
            errors.append(err)
            angles.append(math.degrees(ang))
            print(f"    ({x:.2f},{y:.2f}) @{math.degrees(heading):>5.0f}° → "
                  f"({fix.pose[0]:.2f},{fix.pose[1]:.2f}) "
                  f"@{math.degrees(fix.pose[2]):>5.0f}°  "
                  f"err {err*100:>4.0f} cm {math.degrees(ang):>4.0f}°  "
                  f"score {fix.score:.2f} margin {fix.margin:.2f}")

    check("it placed her at all", bool(errors), f"{len(errors)} fixes")
    # ⚠ THE PROPERTY THAT MATTERS IS NOT ACCURACY, IT IS NEVER BEING
    # CONFIDENTLY WRONG. A refusal costs a question; a wrong fix sends her
    # across the flat to the wrong address with no symptom anyone can see.
    wrong = [e for e in errors if e > 0.50]
    check("no fix is confidently wrong", not wrong,
          f"{len(wrong)} fixes over 50 cm out" if wrong
          else f"worst of {len(errors)} was {max(errors)*100:.0f} cm")
    if errors:
        check("median position error under 15 cm",
              float(np.median(errors)) < 0.15,
              f"median {np.median(errors)*100:.0f} cm, "
              f"worst {max(errors)*100:.0f} cm")
        check("median heading error under 10 degrees",
              float(np.median(angles)) < 10.0,
              f"median {np.median(angles):.0f}°, worst {max(angles):.0f}°")
        check("fast enough to run before a drive",
              float(np.median(times)) < 6.0,
              f"median {np.median(times):.2f} s per fix")


def test_refuses_a_scan_from_nowhere() -> None:
    print("\n— a room it has never seen —")
    grid = _real_map()
    rng = np.random.default_rng(14)
    junk = {b: int(v) for b, v in
            zip(range(0, 360, 2), rng.integers(300, 3000, 180))}
    fix = gl.localise(grid, junk)
    check("random ranges are refused, not placed", not fix.ok, fix.why[:70])

    check("an empty sweep is refused", not gl.localise(grid, {}).ok)
    check("a handful of beams is refused",
          not gl.localise(grid, {0: 1000, 90: 1200, 180: 900}).ok)


def test_refuses_when_the_room_is_ambiguous() -> None:
    print("\n— a room that looks the same in four places —")
    # A perfectly symmetric cross: four identical arms. Standing in any arm
    # looks exactly like standing in the others.
    g = OccupancyGrid(0.05)
    g.logodds = np.zeros((160, 160), np.float32)
    g.origin = np.zeros(2)
    g._empty = False
    g.logodds[:] = 3.0                              # solid
    g.logodds[70:90, 20:140] = -1.0                 # a corridor
    g.logodds[20:140, 70:90] = -1.0                 # crossed with another
    truth = (2.0, 4.0, 0.0)                         # in one arm
    scan = cast(g, truth, max_m=8.0)
    fix = gl.localise(g, scan)
    check("a symmetric room is refused rather than guessed",
          not fix.ok, fix.why[:90])
    check("and it names the two places it could be",
          fix.runner_up is not None and "two places" in fix.why)


def test_beats_what_it_replaces() -> None:
    """The point of the whole exercise, made concrete."""
    print("\n— against the 4 cm window it replaces —")
    grid = _real_map()
    prep = pp.preprocess(grid)
    cells = np.argwhere(prep.plannable)
    cx, cy = cells[len(cells) // 2]
    truth = ((cx + 0.5) * grid.cell + grid.origin[0],
             (cy + 0.5) * grid.cell + grid.origin[1], 0.9)
    scan = cast(grid, truth)

    # What _placed_mapper does today: start at the origin, fold with no
    # odometry, which gives a 4 cm window and a pinned angle.
    from slam.mapper import Mapper
    m = Mapper(grid=grid, pose=(0.0, 0.0, 0.0))
    for _ in range(config.MISSION_SETTLE_SWEEPS):
        m.update(scan)
    old_err = math.hypot(m.pose[0] - truth[0], m.pose[1] - truth[1])

    fix = gl.localise(grid, scan, restrict_to=prep.plannable)
    new_err = (math.hypot(fix.pose[0] - truth[0], fix.pose[1] - truth[1])
               if fix.ok else float("inf"))

    print(f"    truth            ({truth[0]:.2f}, {truth[1]:.2f}) "
          f"@{math.degrees(truth[2]):.0f}°")
    print(f"    _placed_mapper   ({m.pose[0]:.2f}, {m.pose[1]:.2f}) "
          f"@{math.degrees(m.pose[2]):.0f}°   err {old_err:.2f} m  "
          f"trusted={m.pose_trusted}")
    if fix.ok:
        print(f"    global localiser ({fix.pose[0]:.2f}, {fix.pose[1]:.2f}) "
              f"@{math.degrees(fix.pose[2]):.0f}°   err {new_err:.2f} m")
    else:
        print(f"    global localiser REFUSED — {fix.why[:70]}")
    check("the old path is wrong by more than a metre", old_err > 1.0,
          f"{old_err:.2f} m")
    check("and calls itself trusted anyway", m.pose_trusted,
          "this is the bug, asserted so it cannot quietly change")
    # It may legitimately refuse here: a 12.5 m2 map seen from one standing
    # position is genuinely ambiguous at some headings, and saying so is the
    # correct answer. What it must never do is what the old path does --
    # return a confident pose that is metres wrong.
    check("the global localiser is right, or says it cannot tell",
          (fix.ok and new_err < 0.20) or not fix.ok,
          f"{new_err:.2f} m" if fix.ok else f"refused: {fix.why[:60]}")
    check("it never claims a pose it cannot justify",
          not (fix.ok and new_err > 0.50),
          "a confident wrong fix is the one unacceptable outcome")


def main() -> int:
    print("global_localise.py — can she find herself?")
    test_recovers_a_known_pose()
    test_refuses_a_scan_from_nowhere()
    test_refuses_when_the_room_is_ambiguous()
    test_beats_what_it_replaces()
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
