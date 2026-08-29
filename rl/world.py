"""Persistent world model for the RL exploration stack.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
---------------------------------------------
Friday already has a good occupancy mapper -- `slam/grid.py`, a log-odds grid with
clamping, atomic save, and growth that keeps the cell lattice anchored so a wall never
smears when the map extends. Building a second one would be a mistake. This module
WRAPS it and connects it to the RL runtime, which previously had no memory at all.

It is NOT a replacement for `LocalExplorationMap`. That class must keep producing the
72 frontier rays exactly as it does today, because those rays are half of the policy's
146-dim observation and their geometry, units and scaling come from `train.py`. Change
them and the network is fed out-of-distribution input -- the single easiest way to put
a robot into a wall. `LocalExplorationMap` stays untouched and authoritative for the
observation; `WorldMap` runs beside it and owns MEMORY: what has been seen, what has
been driven through, and what is still unknown.

    LiDAR sweep ─┬─> LocalExplorationMap ──> 72 frontier rays ──> POLICY (unchanged)
                 └─> WorldMap ──> persistent occupancy + visit trace ──> PLANNER, telemetry

WHY A SEPARATE MAP FILE
-----------------------
This writes `state/rl_map.npz`, NOT `state/map.npz`. Friday's own SLAM map is built by
a scan matcher with 7.3 cm accuracy over 5.4 m; this one is currently fed a
dead-reckoned pose with no encoders and no IMU. Letting the worse estimate overwrite
the better survey would destroy the good map silently, and `.gitignore` already warns
that a stale map "would be silently loaded over a real survey". Two files, two
provenances, no confusion.

WHY INTEGRATION RUNS OFF THE CONTROL THREAD
-------------------------------------------
Folding a sweep into the grid costs ~24 ms on this Pi. The control loop's budget is
100 ms and it currently uses ~5 ms; spending a quarter of the budget on mapping every
tick would make the loop's timing hostage to map growth, which gets slower as the map
gets bigger. `MapperThread` consumes (scan, pose) snapshots at its own rate and drops
whatever it cannot keep up with. A dropped map update costs a little map freshness; a
missed control deadline costs a collision.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Friday's own modules live one directory up. Imported by path so this file works
# however the process was launched.
_FRIDAY_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_FRIDAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRIDAY_ROOT))

from slam.grid import OccupancyGrid  # noqa: E402

DEFAULT_MAP_FILE = _FRIDAY_ROOT / "state" / "rl_map.npz"

# A cell counts as VISITED when the robot's body has covered it. Seen-from-a-distance
# and driven-through are different facts: the first says the LiDAR had line of sight,
# the second says the floor is genuinely traversable at robot scale. Exploration
# planning wants the difference, so both are tracked.
VISIT_RADIUS_M = 0.22        # robot_radius; the footprint it actually sweeps


class WorldMap:
    """Persistent occupancy + visit memory, in world metres.

    Pose convention matches `LocalExplorationMap`: x, y in metres, theta in radians,
    beam 0 of the scan pointing along +theta (robot forward), beams running
    counter-clockwise. Keeping the two in the same frame is what lets the planner's
    answers be expressed to a policy that thinks in beam indices.
    """

    def __init__(self, cell_m: float | None = None,
                 map_file: Path | str | None = None,
                 max_range: float = 5.0) -> None:
        self.grid = OccupancyGrid(cell_m)
        self.map_file = Path(map_file) if map_file is not None else DEFAULT_MAP_FILE
        self.max_range = float(max_range)
        self.pose = np.zeros(3, np.float64)          # x, y, theta
        self.loaded_from_disk = False
        self.visits_loaded = False
        self.sweeps = 0
        self.dropped = 0                             # sweeps the mapper could not keep up with

        # Visit trace, in world metres. A dict of snapped cell -> visit COUNT, keyed
        # in absolute world cells rather than array indices, because the occupancy grid
        # grows and reallocates underneath while this must survive unchanged.
        #
        # Counts, not a set, because "have I been here" and "have I been here FIVE
        # TIMES" are different questions and only the second one detects a robot going
        # in circles. Phase 10's repeated-visit metric reads this.
        self._visits: dict[tuple[int, int], int] = {}
        self._visit_cell = max(self.grid.cell, 0.10)
        # Cells the robot is standing on RIGHT NOW. A count must increment when the
        # robot ENTERS a cell, not on every one of the ten pose updates per second it
        # spends there -- otherwise a parked robot invents thousands of revisits.
        self._occupying: set[tuple[int, int]] = set()
        self._trail: deque[tuple[float, float]] = deque(maxlen=4000)
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- pose ------
    def set_pose(self, x: float, y: float, theta: float) -> None:
        with self._lock:
            self.pose[:] = (x, y, theta)
            self._mark_visited(x, y)

    def _mark_visited(self, x: float, y: float) -> None:
        """Stamp the robot's footprint, not just its centre.

        A centre-only trace leaves a one-cell thread through a room and makes almost
        everything look unvisited, which would send the planner back over ground the
        robot has already physically covered.
        """
        c = self._visit_cell
        r = int(math.ceil(VISIT_RADIUS_M / c))
        cx, cy = int(math.floor(x / c)), int(math.floor(y / c))
        now = {(cx + dx, cy + dy)
               for dx in range(-r, r + 1) for dy in range(-r, r + 1)
               if dx * dx + dy * dy <= r * r}
        for cell in now - self._occupying:            # newly ENTERED cells only
            self._visits[cell] = self._visits.get(cell, 0) + 1
        self._occupying = now
        self._trail.append((float(x), float(y)))

    def has_visited(self, x: float, y: float) -> bool:
        c = self._visit_cell
        return (int(math.floor(x / c)), int(math.floor(y / c))) in self._visits

    def visit_count(self, x: float, y: float) -> int:
        """How many separate times the robot has entered this cell. 0 = never."""
        c = self._visit_cell
        return self._visits.get((int(math.floor(x / c)), int(math.floor(y / c))), 0)

    @property
    def visited_m2(self) -> float:
        return len(self._visits) * self._visit_cell ** 2

    @property
    def revisited_m2(self) -> float:
        """Area entered more than once. High relative to visited_m2 = going in circles."""
        return sum(1 for n in self._visits.values() if n > 1) * self._visit_cell ** 2

    @property
    def revisit_ratio(self) -> float:
        """0.0 = every cell entered once. Approaches 1.0 as the robot re-treads ground.

        A pure coverage number cannot see this: a robot driving the same 2 m of
        corridor forever reports a healthy constant `visited_m2` and looks fine.
        """
        if not self._visits:
            return 0.0
        return sum(1 for n in self._visits.values() if n > 1) / len(self._visits)

    # ------------------------------------------------------------ integrate ----
    def scan_to_points(self, scan_m: np.ndarray, pose: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
        """72 ranges + pose -> world points, and a mask of which may MARK occupied.

        Two beam classes must not leave a mark:

        - Max-range beams. The LD19 publishes max_range for an empty bin, which is
          indistinguishable from a genuine 5 m return. Marking those puts a phantom
          wall on a 5 m arc around the robot every sweep. They still CARVE, because
          "nothing within 5 m" is real evidence of free space.
        - Zero/invalid beams, which are dropped entirely -- they are evidence of
          nothing at all.

        Chassis self-mask beams are removed by the caller before this is reached; see
        `blind_beams` in LocalExplorationMap, and the bug where the self-mask carved
        5 m of free space straight through the robot's own body.
        """
        n = len(scan_m)
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(pose[2])
        r = np.asarray(scan_m, np.float64)
        valid = r > 1e-3
        pts = np.stack([pose[0] + r * np.cos(ang), pose[1] + r * np.sin(ang)], axis=1)
        marks = valid & (r < self.max_range - 1e-3)
        return pts[valid], marks[valid]

    def integrate(self, scan_m: np.ndarray, pose: np.ndarray,
                  blind: np.ndarray | None = None) -> int:
        """Fold one sweep in. Returns beams used."""
        scan = np.asarray(scan_m, np.float64).copy()
        if blind is not None and len(blind):
            scan[blind] = 0.0                        # dropped as invalid below
        pts, marks = self.scan_to_points(scan, pose)
        if pts.size == 0:
            return 0
        with self._lock:
            used = self.grid.integrate(pts, (float(pose[0]), float(pose[1])),
                                       occupied=pts[marks] if marks.any() else
                                       np.empty((0, 2)))
            self.sweeps += 1
        return used

    # ----------------------------------------------------------------- state ---
    def stats(self) -> dict[str, Any]:
        """Real counts, straight off the grid. Nothing derived or estimated."""
        with self._lock:
            g = self.grid
            if g.shape == (1, 1):
                # No scan has been folded in yet, so there is no occupancy to report.
                # The VISIT numbers are still real and are reported as such: a robot
                # that has driven but not yet mapped has genuinely covered ground, and
                # zeroing that here would be inventing a fact to fill a field.
                return {"cells": 0, "free": 0, "occupied": 0, "unknown": 0,
                        "free_m2": 0.0, "explored_m2": 0.0,
                        "visited_m2": self.visited_m2,
                        "revisited_m2": self.revisited_m2,
                        "revisit_ratio": self.revisit_ratio,
                        "visit_cells": len(self._visits),
                        "sweeps": self.sweeps, "sweeps_total": int(g.scans),
                        "shape": g.shape, "bounds": g.bounds}
            free = int(g.free_mask().sum())
            occ = int(g.occupied_mask().sum())
            unk = int(g.unknown_mask().sum())
            a = g.cell ** 2
            return {"cells": free + occ + unk, "free": free, "occupied": occ,
                    "unknown": unk, "free_m2": free * a, "explored_m2": (free + occ) * a,
                    "visited_m2": self.visited_m2,
                    "revisited_m2": self.revisited_m2,
                    "revisit_ratio": self.revisit_ratio,
                    "visit_cells": len(self._visits),
                    # sweeps THIS RUN vs sweeps EVER. A restored map carries a lifetime
                    # total in the grid and has nothing yet from this session; printing
                    # only the session count makes a restored map read as empty.
                    "sweeps": self.sweeps, "sweeps_total": int(g.scans),
                    "shape": g.shape, "bounds": g.bounds}

    # ------------------------------------------------------------ persistence --
    @property
    def visits_file(self) -> Path:
        """Companion to the grid file. Separate because OccupancyGrid owns its own
        npz schema and version, and bolting a foreign key into it would mean this
        module could silently invalidate Friday's map format."""
        return self.map_file.with_suffix(".visits.npz")

    def load(self) -> bool:
        """Restore the previous run's map AND visit trace.

        Missing or unreadable -> start fresh. A corrupt or version-mismatched map must
        never be fatal: the robot's job does not depend on remembering, and refusing to
        start because a cache is stale would be worse than forgetting.
        """
        try:
            self.grid = OccupancyGrid.load(self.map_file)
            self.loaded_from_disk = True
        except FileNotFoundError:
            return False
        except Exception:                             # noqa: BLE001 -- see docstring
            return False

        # The visit trace is loaded SEPARATELY and its failure is non-fatal: an
        # occupancy map without a visit trace is still a useful map. Losing it silently
        # would be bad, so it is reported through `visits_loaded`.
        self.visits_loaded = False
        try:
            with np.load(self.visits_file, allow_pickle=False) as d:
                cells, counts = d["cells"], d["counts"]
                cell_m = float(d["cell_m"])
                # A trace snapped at a different cell size describes different cells.
                # Refuse rather than reinterpret -- the same rule OccupancyGrid.load
                # applies to a map saved at another resolution.
                if abs(cell_m - self._visit_cell) < 1e-9:
                    self._visits = {(int(a), int(b)): int(n)
                                    for (a, b), n in zip(cells, counts)}
                    self.visits_loaded = True
        except FileNotFoundError:
            pass
        except Exception:                             # noqa: BLE001
            pass
        return True

    def save(self) -> Path | None:
        """Write the map and the visit trace. Returns the map path, or None."""
        try:
            with self._lock:
                if self.grid.shape == (1, 1):
                    return None                       # nothing observed; do not clobber
                target = self.grid.save(self.map_file)
                if self._visits:
                    cells = np.array(list(self._visits.keys()), np.int64)
                    counts = np.array(list(self._visits.values()), np.int32)
                    tmp = self.visits_file.with_suffix(".tmp")
                    np.savez_compressed(tmp, cells=cells, counts=counts,
                                        cell_m=self._visit_cell)
                    # np.savez_compressed appends .npz when handed a path without one.
                    src = tmp if tmp.exists() else tmp.with_suffix(".tmp.npz")
                    os.replace(src, self.visits_file)   # atomic, like the grid's save
                return target
        except Exception:                             # noqa: BLE001
            return None


class MapperThread(threading.Thread):
    """Integrates sweeps off the control thread, dropping what it cannot keep up with.

    Deliberately drop-latest rather than queue: a backlog of stale sweeps integrated
    against stale poses is worse than a gap, because each one writes confident
    geometry at the wrong place. Only the newest snapshot is ever pending.

    5 Hz against a 10 Hz control loop, so roughly every other sweep is superseded
    before it is read. That is the design working, not a fault -- MEASURED cost is
    7.4 ms mean / 14.7 ms max per 72-beam sweep on this Pi, and at 0.2 m/s the robot
    moves 4 cm between integrations, under the 5 cm cell. `world.dropped` counts
    superseded snapshots; a high count matters only if the map stops growing.
    """

    def __init__(self, world: WorldMap, hz: float = 5.0) -> None:
        super().__init__(name="mapper", daemon=True)
        self.world = world
        self.period = 1.0 / max(hz, 0.1)
        self._pending: tuple[np.ndarray, np.ndarray, np.ndarray | None] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.integrations = 0
        self.last_ms = 0.0

    def submit(self, scan: np.ndarray, pose: np.ndarray,
               blind: np.ndarray | None = None) -> None:
        """Called from the control loop. Must be O(1) and never block."""
        with self._lock:
            if self._pending is not None:
                self.world.dropped += 1
            self._pending = (np.asarray(scan).copy(), np.asarray(pose).copy(), blind)

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                job, self._pending = self._pending, None
            if job is not None:
                t0 = time.perf_counter()
                try:
                    self.world.integrate(*job)
                except Exception:                     # noqa: BLE001
                    pass                              # mapping must never kill the robot
                self.last_ms = (time.perf_counter() - t0) * 1000.0
                self.integrations += 1
            time.sleep(self.period)
