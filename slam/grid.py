"""The room itself: a persistent occupancy map in log-odds.

`core` answers "how did the robot move between these two sweeps". This answers
"what is where", and unlike the matcher it REMEMBERS -- every sweep is folded
into one array that survives a restart and a power cut.

WHY LOG-ODDS AND NOT PROBABILITIES. Two reasons, and only the second one
matters. The first is that Bayes becomes addition. The second is that a
clamped sum can CHANGE ITS MIND: a cell that has been seen as a wall two
hundred times would need two hundred contradicting looks to come free again,
so the chair Harsh moved this morning would still be a wall tonight. Clamping
at config.SLAM_LOGODDS_MIN/MAX costs the certainty nobody needed and buys a
map that clears a moved obstacle in about ten sweeps.

WHAT A MISSING BEARING MEANS, AND WHY IT IS NOT FREE SPACE. A bearing absent
from a scan is either "nothing within six metres" or "a black jacket that
returned no signal", and the LiDAR does not say which. So absent bearings
carve nothing at all. The consequence is honest and worth knowing: an open
doorway further away than LIDAR_MAX_RANGE_MM stays UNKNOWN rather than being
mapped as open floor, and the wedge Friday's own chassis blocks
(config.LIDAR_SELF_MASK, bearings 89-98) is never carved by the sweep it sits
in -- it fills in as she turns, and until it does it is unknown, not free.

UNKNOWN IS NOT DRIVEABLE. `free_mask` requires positive evidence of free
space. This is the same rule the safety interrupt already applies to a null
clearance, for the same reason: a robot that treats "nobody has looked" as
"clear" drives into the one place it has never seen.

Nothing here talks to hardware or to the MCP server. It takes point clouds in
world metres, so it is testable against a synthetic room with no robot
attached -- see scripts/test_slam.py.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np

import config
from slam.core import points_from_scan, transform

MAP_VERSION = 1


class OccupancyGrid:
    """A growable log-odds occupancy map on a fixed cell lattice.

    The lattice is anchored to multiples of `cell` in world coordinates, so
    growing the map shifts the array but never the cells: a wall keeps the
    same cell indices relative to the world however many times the map is
    extended. Without that anchoring every growth would resample the map and
    smear it a little, and it would smear a little more every time.
    """

    def __init__(self, cell_m: float | None = None) -> None:
        self.cell = float(cell_m if cell_m is not None else config.SLAM_CELL_M)
        self.logodds = np.zeros((1, 1), dtype=np.float32)
        self.origin = np.zeros(2, dtype=np.float64)   # world xy of cell (0, 0)
        self.scans = 0
        self.updated_at = 0.0
        self._empty = True

    # --- geometry --------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.logodds.shape[0]), int(self.logodds.shape[1]))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the mapped area, in metres."""
        nx, ny = self.shape
        return (float(self.origin[0]), float(self.origin[1]),
                float(self.origin[0] + nx * self.cell),
                float(self.origin[1] + ny * self.cell))

    def cells_of(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World metres -> integer cell indices. No bounds checking."""
        cells = np.floor((np.atleast_2d(points) - self.origin)
                         / self.cell).astype(np.int64)
        return cells[:, 0], cells[:, 1]

    def inside(self, points: np.ndarray) -> np.ndarray:
        """Boolean mask of which points fall within the allocated map."""
        ix, iy = self.cells_of(points)
        nx, ny = self.shape
        return (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    def _snap(self, value: float, up: bool) -> float:
        n = math.ceil(value / self.cell) if up else math.floor(value / self.cell)
        return n * self.cell

    def _grow_to(self, min_xy: np.ndarray, max_xy: np.ndarray,
                 margin_m: float = 1.0) -> None:
        """Extend the array so the given box fits, keeping the lattice fixed.

        Growth is in whole cells from a snapped origin, and a margin is added
        so that creeping one cell past the edge does not reallocate a
        480 x 480 array on the next sweep too.
        """
        low_x = self._snap(float(min_xy[0]) - margin_m, up=False)
        low_y = self._snap(float(min_xy[1]) - margin_m, up=False)
        high_x = self._snap(float(max_xy[0]) + margin_m, up=True)
        high_y = self._snap(float(max_xy[1]) + margin_m, up=True)

        if self._empty:
            new_origin = np.array([low_x, low_y], dtype=np.float64)
            nx = max(int(round((high_x - low_x) / self.cell)), 1)
            ny = max(int(round((high_y - low_y) / self.cell)), 1)
            self._cap(nx, ny)
            self.logodds = np.zeros((nx, ny), dtype=np.float32)
            self.origin = new_origin
            self._empty = False
            return

        old_x0, old_y0, old_x1, old_y1 = self.bounds
        new_x0, new_y0 = min(low_x, old_x0), min(low_y, old_y0)
        new_x1, new_y1 = max(high_x, old_x1), max(high_y, old_y1)
        if (new_x0, new_y0, new_x1, new_y1) == (old_x0, old_y0, old_x1, old_y1):
            return

        nx = int(round((new_x1 - new_x0) / self.cell))
        ny = int(round((new_y1 - new_y0) / self.cell))
        self._cap(nx, ny)
        grown = np.zeros((nx, ny), dtype=np.float32)
        off_x = int(round((old_x0 - new_x0) / self.cell))
        off_y = int(round((old_y0 - new_y0) / self.cell))
        ox, oy = self.shape
        grown[off_x:off_x + ox, off_y:off_y + oy] = self.logodds
        self.logodds = grown
        self.origin = np.array([new_x0, new_y0], dtype=np.float64)

    def _cap(self, nx: int, ny: int) -> None:
        limit = int(math.ceil(config.SLAM_MAP_MAX_M / self.cell))
        if nx > limit or ny > limit:
            raise ValueError(
                f"map would grow to {nx}x{ny} cells, over the "
                f"{config.SLAM_MAP_MAX_M} m cap -- almost certainly a bad "
                f"pose estimate rather than a very large room")

    # --- folding a sweep in ----------------------------------------------

    def integrate_scan(self, scan: dict[int, int],
                       pose: tuple[float, float, float],
                       forward_deg: float | None = None) -> int:
        """Fold one raw {bearing: mm} sweep in, from pose (x, y, theta_rad).

        theta is the robot's heading in world radians; the sweep is rotated
        into the world by it. forward_deg defaults to the measured
        config.LIDAR_FORWARD_DEG -- pass it explicitly only when working in
        the LiDAR's own uncalibrated frame.
        """
        if forward_deg is None:
            forward_deg = config.LIDAR_FORWARD_DEG
        local = points_from_scan(scan, forward_deg=forward_deg)
        world = transform(local, pose[0], pose[1], pose[2])
        return self.integrate(world, (pose[0], pose[1]))

    def integrate(self, points: np.ndarray,
                  origin_xy: tuple[float, float]) -> int:
        """Fold in a cloud already in world metres, seen from origin_xy.

        Every beam carves the cells it passed through towards free and marks
        the cell it stopped in towards occupied. Returns how many beams were
        used.

        Each cell is updated AT MOST ONCE per sweep, however many beams cross
        it. A single sweep is one observation, not four hundred: without the
        de-duplication the cell just in front of the robot is crossed by most
        of the fan and slams to the free clamp from one look, which then takes
        the full clamp width of contradicting evidence to undo.

        Where a beam's own endpoint lies on a cell another beam passed
        through, OCCUPIED WINS. A thin table leg is grazed by two beams and
        passed on both sides by twenty; letting the passes cancel the returns
        is how thin obstacles quietly vanish from a map.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if points.size == 0:
            return 0
        origin = np.asarray(origin_xy, dtype=np.float64)

        low = np.minimum(points.min(axis=0), origin)
        high = np.maximum(points.max(axis=0), origin)
        self._grow_to(low, high)

        keep = self.inside(points)
        points = points[keep]
        if len(points) == 0:
            return 0

        free_flat, occ_flat = self._trace(origin, points)
        self._apply(free_flat, config.SLAM_LOGODDS_MISS)
        self._apply(occ_flat, config.SLAM_LOGODDS_HIT)

        self.scans += 1
        self.updated_at = time.time()
        return int(len(points))

    def _apply(self, flat: np.ndarray, delta: float) -> None:
        """Add one observation's evidence to a set of flat cell indices."""
        if not flat.size:
            return
        ix, iy = np.unravel_index(flat, self.shape)
        self.logodds[ix, iy] = np.clip(self.logodds[ix, iy] + delta,
                                       config.SLAM_LOGODDS_MIN,
                                       config.SLAM_LOGODDS_MAX)

    def _trace(self, origin: np.ndarray,
               points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flat indices of the cells each beam passed through, and stopped in.

        Sampled along the ray at half a cell rather than walked with
        Bresenham: the duplicates that sampling produces are removed by the
        same np.unique that the once-per-sweep rule needs anyway, so the
        integer line algorithm would buy nothing but a Python loop over four
        hundred beams.
        """
        nx, ny = self.shape
        deltas = points - origin
        lengths = np.hypot(deltas[:, 0], deltas[:, 1])
        usable = lengths > self.cell
        occ_ix, occ_iy = self.cells_of(points)
        occ_flat = np.unique(occ_ix * ny + occ_iy)

        if not usable.any():
            return np.empty(0, dtype=np.int64), occ_flat

        deltas = deltas[usable]
        lengths = lengths[usable]
        # Stop one cell short of the return so the surface itself is not
        # carved free by the very beam that found it.
        stop = lengths - self.cell
        unit = deltas / lengths[:, None]

        steps = int(math.ceil(float(stop.max()) / (self.cell * 0.5))) + 1
        fraction = np.linspace(0.0, 1.0, max(steps, 2))[None, :]
        along = stop[:, None] * fraction
        xs = origin[0] + unit[:, 0:1] * along
        ys = origin[1] + unit[:, 1:2] * along

        ix = np.floor((xs - self.origin[0]) / self.cell).astype(np.int64)
        iy = np.floor((ys - self.origin[1]) / self.cell).astype(np.int64)
        good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        free_flat = np.unique(ix[good] * ny + iy[good])
        # Occupied wins wherever the two sets overlap.
        free_flat = free_flat[~np.isin(free_flat, occ_flat)]
        return free_flat, occ_flat

    # --- reading it back out ---------------------------------------------

    def probability(self) -> np.ndarray:
        """Log-odds -> probability of being occupied, 0..1. 0.5 is unknown."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds.astype(np.float64)))

    def occupied_mask(self) -> np.ndarray:
        return self.probability() >= config.SLAM_P_OCCUPIED

    def free_mask(self) -> np.ndarray:
        """Cells with positive evidence of being empty.

        Unknown is NOT free -- see the note at the top of this file.
        """
        return self.probability() <= config.SLAM_P_FREE

    def unknown_mask(self) -> np.ndarray:
        p = self.probability()
        return (p > config.SLAM_P_FREE) & (p < config.SLAM_P_OCCUPIED)

    def classify(self, points: np.ndarray) -> np.ndarray:
        """Per-point state at world coordinates: -1 free, 0 unknown, 1 occupied.

        Anything off the edge of the map is unknown, never free.
        """
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        out = np.zeros(len(points), dtype=np.int8)
        if self._empty or points.size == 0:
            return out
        here = self.inside(points)
        if not here.any():
            return out
        ix, iy = self.cells_of(points[here])
        p = self.probability()[ix, iy]
        state = np.zeros(len(p), dtype=np.int8)
        state[p >= config.SLAM_P_OCCUPIED] = 1
        state[p <= config.SLAM_P_FREE] = -1
        out[here] = state
        return out

    def is_free(self, x: float, y: float) -> bool:
        return bool(self.classify(np.array([[x, y]]))[0] == -1)

    def stats(self) -> dict:
        total = int(self.logodds.size)
        occupied = int(self.occupied_mask().sum())
        free = int(self.free_mask().sum())
        x0, y0, x1, y1 = self.bounds
        return {
            "cells": total,
            "shape": list(self.shape),
            "cell_m": self.cell,
            "bounds_m": [round(v, 2) for v in (x0, y0, x1, y1)],
            "occupied": occupied,
            "free": free,
            "unknown": total - occupied - free,
            "explored_m2": round((occupied + free) * self.cell ** 2, 2),
            "scans": self.scans,
        }

    def to_ascii(self, width: int = 78) -> str:
        """The map as text, because SSH is the only way anyone sees this.

        '#' occupied, '.' free, ' ' unknown. Rows are downsampled by taking
        the worst cell in each block: an obstacle must never be averaged away
        by the nine empty cells around it.
        """
        if self._empty:
            return "(empty map)"
        occ, free = self.occupied_mask(), self.free_mask()
        nx, ny = self.shape
        stride = max(1, int(math.ceil(nx / width)))
        lines = []
        # x is forward/right in world, y is left: printed with +y to the LEFT
        # and +x UP, so the picture matches standing behind the robot.
        for iy in range(ny - 1, -1, -stride):
            row = []
            for ix in range(0, nx, stride):
                block_occ = occ[ix:ix + stride, max(iy - stride + 1, 0):iy + 1]
                block_free = free[ix:ix + stride, max(iy - stride + 1, 0):iy + 1]
                row.append("#" if block_occ.any()
                           else ("." if block_free.any() else " "))
            lines.append("".join(row))
        return "\n".join(lines)

    # --- persistence ------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        """Write atomically: temp file in the same directory, then rename.

        Same idiom as face_control._write_atomic and for the same reason --
        a power cut mid-write must leave the previous map intact rather than
        a truncated one, and this map is the only thing that knows where the
        rooms are.
        """
        target = Path(path) if path is not None else Path(config.SLAM_MAP_FILE)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".map.",
                                   suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez_compressed(handle, version=MAP_VERSION,
                                    logodds=self.logodds, origin=self.origin,
                                    cell=self.cell, scans=self.scans,
                                    updated_at=self.updated_at)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "OccupancyGrid":
        """Read a saved map back. Raises FileNotFoundError if there is none.

        A map saved at a different cell size is refused rather than
        resampled: silently reinterpreting somebody's 5 cm map as 10 cm puts
        every wall in the wrong place, and a fresh map is a cheap loss.
        """
        source = Path(path) if path is not None else Path(config.SLAM_MAP_FILE)
        with np.load(source, allow_pickle=False) as data:
            version = int(data["version"])
            if version != MAP_VERSION:
                raise ValueError(f"map file is version {version}, "
                                 f"this code writes {MAP_VERSION}")
            grid = cls(cell_m=float(data["cell"]))
            grid.logodds = np.ascontiguousarray(data["logodds"],
                                                dtype=np.float32)
            grid.origin = np.asarray(data["origin"], dtype=np.float64)
            grid.scans = int(data["scans"])
            grid.updated_at = float(data["updated_at"])
            grid._empty = False
        return grid
