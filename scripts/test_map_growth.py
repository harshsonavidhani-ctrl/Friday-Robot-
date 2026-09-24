#!/usr/bin/env python3
"""Test map_growth.py — the saved map grows with every explore trip.

    ./venv/bin/python scripts/test_map_growth.py

HOW THIS GETS GROUND TRUTH. Sweeps are ray cast from the REAL saved map at
poses chosen here, then folded into a brand-new trip map in a frame that
starts at one of those poses -- exactly what an explore trip builds when it
cannot place herself at the start. Fitting that trip map back onto the saved
map must recover the pose the frame started at.

⚠ IT WRITES NOTHING REAL. The real map is only read; every save goes to a
temporary directory, and the bundle follows a temporary NAV_LOG.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                           # noqa: E402
import map_growth                                       # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
import path_planner                                     # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402
from test_nav_follower import _cast                     # noqa: E402

_TMP = tempfile.TemporaryDirectory()
nsm.NAV_LOG = Path(_TMP.name) / "state" / "nav_log.jsonl"
nsm.FAILSAFE_FILE = Path(_TMP.name) / "state" / "nav_failsafe.json"
nsm.PENDING_FILE = Path(_TMP.name) / "state" / "nav_round_trip.json"
nsm.NAV_MODE_FILE = Path(_TMP.name) / "state" / "nav_mode.json"
nsm.EARLY_RETURN_FILE = Path(_TMP.name) / "state" / "nav_early_return.json"

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


def _relative(origin, pose):
    """`pose` (world) as seen from `origin` (world): the trip-frame pose."""
    dx, dy = pose[0] - origin[0], pose[1] - origin[1]
    c, s = math.cos(-origin[2]), math.sin(-origin[2])
    return (dx * c - dy * s, dx * s + dy * c, pose[2] - origin[2])


def _trip_map(saved, start, others):
    """What a trip starting at `start` builds, in its own frame."""
    trip = OccupancyGrid(saved.cell)
    for pose in [start] + others:
        scan = _cast(saved, pose)
        for _ in range(3):
            trip.integrate_scan(scan, _relative(start, pose))
    return trip


def _open_poses(saved, count=40):
    prep = path_planner.preprocess(saved)
    cells = np.argwhere(prep.plannable)
    picks = np.linspace(0, len(cells) - 1, count).astype(int)
    return [tuple(path_planner.cells_to_world([tuple(cells[i])],
                                              prep.grid)[0]) for i in picks]


def test_fit_recovers_the_start() -> None:
    print("\n— a trip's whole map finds where the trip started —")
    saved = OccupancyGrid.load(config.SLAM_MAP_FILE)
    spots = _open_poses(saved)
    tried, errors, angles, refused = 0, [], [], 0
    for k, (x, y) in enumerate(spots[::4]):
        heading = (-2.0, 0.4, 1.9)[k % 3]
        start = (x, y, heading)
        near = [p for p in spots if 0.4 < math.hypot(p[0] - x, p[1] - y) < 1.5]
        others = [(p[0], p[1], heading + 0.7) for p in near[:2]]
        trip = _trip_map(saved, start, others)
        tried += 1
        fix = map_growth.fit(saved, trip)
        if not fix.ok:
            refused += 1
            continue
        errors.append(math.hypot(fix.pose[0] - x, fix.pose[1] - y))
        angles.append(abs(math.degrees(math.atan2(
            math.sin(fix.pose[2] - heading), math.cos(fix.pose[2] - heading)))))
    check("it fitted most trips", refused <= tried // 3,
          f"{tried - refused}/{tried} fitted, {refused} refused")
    # ⚠ ~15 cm IS THIS TEST'S RAY CASTING, NOT THE FIT. A cast stops at the
    # near face of a wall; the real map's walls are several cells thick from
    # real sweeps, so the best match sits a little into them. The tracker's
    # own matcher (the polish) lands on the same spot, and
    # test_global_localise.py sees the same 11-16 cm from single casts.
    # merge() itself is exact: see test_merge_keeps_the_frame.
    check("never confidently wrong: every fit within 25 cm",
          errors and max(errors) < 0.25,
          f"median {np.median(errors) * 100:.0f} cm, worst "
          f"{max(errors) * 100:.0f} cm" if errors else "no fits")
    check("and within 6° of the true heading",
          angles and max(angles) < 6.0,
          f"worst {max(angles):.1f}°" if angles else "no fits")


def test_merge_keeps_the_frame() -> None:
    print("\n— merging keeps the saved map's frame and grows it —")
    saved = OccupancyGrid.load(config.SLAM_MAP_FILE)
    x, y = _open_poses(saved)[20]
    start = (x, y, 0.8)
    trip = _trip_map(saved, start, [])
    merged = map_growth.merge(saved, trip, start)
    check("nothing the saved map knew is lost",
          map_growth.known_m2(merged) >= map_growth.known_m2(saved) - 1e-6,
          f"{map_growth.known_m2(saved):.1f} -> "
          f"{map_growth.known_m2(merged):.1f} m²")
    # Walls the trip saw land on walls the map already had.
    tw = map_growth.trip_points(trip)
    c, s = math.cos(start[2]), math.sin(start[2])
    world = np.column_stack((tw[:, 0] * c - tw[:, 1] * s + start[0],
                             tw[:, 0] * s + tw[:, 1] * c + start[1]))
    inside = saved.inside(world)
    ix, iy = saved.cells_of(world[inside])
    near = 0
    occ = saved.occupied_mask()
    for i, j in zip(ix, iy):
        win = occ[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3]
        near += bool(win.any())
    check("the trip's walls land on the saved map's walls",
          inside.sum() and near / inside.sum() > 0.8,
          f"{near}/{int(inside.sum())} within 10 cm of a known wall")
    check("the origin (the frame) is unchanged where the map did not grow",
          np.all(merged.origin <= saved.origin + 1e-9))


def test_grow_saves_and_keeps_history() -> None:
    print("\n— growing saves the map and keeps the old one —")
    real = config.SLAM_MAP_FILE
    before_hash = hashlib.sha256(real.read_bytes()).hexdigest()
    work = Path(_TMP.name) / "map.npz"
    work.write_bytes(real.read_bytes())
    saved = OccupancyGrid.load(work)
    # A trip that saw the room from a few spots, like a real one -- and one
    # whose start this map can tell apart (a single sweep from some spots in
    # this room is genuinely ambiguous; that is what the trip is for).
    spots = _open_poses(saved)
    trip = None
    for k, (x, y) in enumerate(spots):
        near = [p for p in spots
                if 0.4 < math.hypot(p[0] - x, p[1] - y) < 1.5][:2]
        candidate = _trip_map(saved, (x, y, -1.1),
                              [(p[0], p[1], -0.4) for p in near])
        if map_growth.fit(saved, candidate).ok:
            trip = candidate
            break
    check("found a test trip this map can place", trip is not None)

    untrusted = map_growth.grow(trip, "trip", trusted=False, path=work)
    check("a trip that did not know where it was saves nothing",
          not untrusted["grew"] and work.read_bytes() == real.read_bytes(),
          untrusted["why"])

    import nav_bundle
    config_file = config.SLAM_MAP_FILE
    config.SLAM_MAP_FILE = work               # the bundle copies THIS map
    try:
        out = map_growth.grow(trip, "trip", trusted=True, path=work)
        check("a fitted trip grows the map", out["grew"] and out["fitted"],
              json.dumps({k: out.get(k) for k in ("before_m2", "after_m2",
                                                  "why")}))
        hist = nav_bundle.bundle_dir() / "history"
        kept = [h for h in hist.iterdir()] if hist.exists() else []
        check("the map as it was is kept in the bundle's history",
              any((h / "map.npz").exists()
                  and (h / "map.npz").read_bytes() == real.read_bytes()
                  for h in kept), f"{len(kept)} history entries")

        # A trip that does not belong to this map at all.
        foreign = OccupancyGrid(saved.cell)
        rng = np.random.default_rng(3)
        scan = {b: int(rng.uniform(400, 3000)) for b in range(0, 360, 2)}
        for _ in range(3):
            foreign.integrate_scan(scan, (0.0, 0.0, 0.0))
        grown = work.read_bytes()
        out = map_growth.grow(foreign, "trip", trusted=True, path=work)
        check("a trip that does not fit leaves the map exactly as it was",
              not out["grew"] and work.read_bytes() == grown, out["why"][:80])
        check("and its map is kept, not thrown away",
              out.get("kept") and Path(out["kept"], "map.npz").exists(),
              str(out.get("kept")))
    finally:
        config.SLAM_MAP_FILE = config_file
    check("the REAL map was never written",
          hashlib.sha256(real.read_bytes()).hexdigest() == before_hash)


class _SimRobot:
    """A robot that really is somewhere on the saved map, for locate().

    Sweeps are ray cast from where it truly stands; a step moves it the full
    distance when the floor ahead (or behind) is plannable, and is REFUSED
    otherwise, the way the guard refuses a move into something.
    """

    def __init__(self, grid, prep, pose):
        self.grid, self.prep = grid, prep
        self.x, self.y, self.th = pose
        self.moves = 0

    def _clear(self, dist):
        for k in range(1, 9):
            d = dist * k / 8
            px = self.x + d * math.cos(self.th)
            py = self.y + d * math.sin(self.th)
            ix, iy = self.prep.grid.cells_of(np.array([[px, py]]))
            nx, ny = self.prep.plannable.shape
            if not (0 <= ix[0] < nx and 0 <= iy[0] < ny
                    and self.prep.plannable[ix[0], iy[0]]):
                return False
        return True

    async def sweep(self):
        return _cast(self.grid, (self.x, self.y, self.th))

    async def observe(self):
        pass

    async def clearance(self):
        return (2000.0, False) if self._clear(0.5) else (300.0, True)

    async def drive_seconds(self, direction, seconds):
        import nav_state_machine as n
        d = n.DISAMBIG_STEP_M * (1 if direction == "forward" else -1)
        if not self._clear(d):
            return "REFUSED: something is in the way"
        self.x += d * math.cos(self.th)
        self.y += d * math.sin(self.th)
        self.moves += 1
        return f"moved {direction}"

    def stopped(self):
        return False


def test_locate_moves_to_break_a_tie() -> None:
    print("\n— two spots look alike: she moves a little, on the OLD map only —")
    import asyncio

    import global_localise
    saved = OccupancyGrid.load(config.SLAM_MAP_FILE)
    prep = path_planner.preprocess(saved)
    rng = np.random.default_rng(5)
    cells = np.argwhere(prep.plannable)
    placed, refused, wrong, tried = 0, 0, 0, 0
    for i in rng.choice(len(cells), 300, replace=False):
        if tried >= 5:
            break
        x, y = path_planner.cells_to_world([tuple(cells[i])], prep.grid)[0]
        th = float(rng.uniform(-math.pi, math.pi))
        if len(_cast(saved, (x, y, th))) < 100:
            continue
        first = global_localise.localise(saved, _cast(saved, (x, y, th)),
                                         restrict_to=prep.plannable)
        if first.ok or first.candidate is None:
            continue                          # only the genuinely ambiguous
        tried += 1
        robot = _SimRobot(OccupancyGrid.load(config.SLAM_MAP_FILE), prep,
                          (x, y, th))
        mapper, why = asyncio.run(nsm.locate(
            robot, grid=OccupancyGrid.load(config.SLAM_MAP_FILE)))
        if mapper is None:
            refused += 1
            continue
        err = math.hypot(mapper.pose[0] - robot.x, mapper.pose[1] - robot.y)
        if err < 0.3:
            placed += 1
        else:
            wrong += 1
            print(f"    WRONG by {err:.2f} m after {robot.moves} moves")
    check("found look-alike start spots on the real map to test",
          tried >= 3, f"{tried} tried")
    check("NEVER placed in the wrong spot", wrong == 0,
          f"{placed} placed, {refused} refused, {wrong} wrong")
    check("and moving placed her at least some of the time", placed >= 1,
          f"{placed}/{tried}")


def main() -> int:
    print("map_growth.py — the saved map grows with every explore trip")
    if not config.SLAM_MAP_FILE.exists():
        print("(no saved map on this machine — nothing to test against)")
        return 0
    test_fit_recovers_the_start()
    test_merge_keeps_the_frame()
    test_grow_saves_and_keeps_history()
    test_locate_moves_to_break_a_tie()
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
