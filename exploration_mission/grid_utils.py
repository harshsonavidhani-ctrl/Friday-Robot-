"""One grid form for the whole exploration mission (File 21, spec 4.1).

Whatever map the mission reads -- slam's log-odds OccupancyGrid (state/map.npz,
the start tracker's sweeps folded into a fresh one) or a plain probability
array -- is normalised ONCE into a `Grid`:

    occ         int8 [NX, NY]  FREE = 0, OCC = 100, UNKNOWN = -1
    resolution  metres per cell
    origin      world (x, y) of the CORNER of cell [0, 0]

⚠ AXIS ORDER IS slam/grid.py's, NOT AN IMAGE'S: axis 0 is world x, axis 1 is
world y (`logodds[ix, iy]`). Keeping slam's order means there is no flip to get
wrong anywhere -- world <-> cell is `floor((p - origin) / resolution)` both
ways round, and a cell index from here can be handed straight back to anything
in slam/ or path_planner.py. OpenCV does not care which axis is which for the
morphology used here (symmetric kernels, 8-connectivity).

UNKNOWN IS NEVER FREE, and "traversable" is File 14's own rule, byte for byte:
`slam/plan.py::passable_mask` = FREE and not within the inflation disc of an
OCCUPIED cell (unknown is not inflated, it is simply not passable). Using the
same rule is what makes "reachable" here mean "reachable for the planner that
will actually drive there".

Pure numpy + OpenCV. No hardware, no I/O, no imports from Friday.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

FREE = 0
OCC = 100
UNKNOWN = -1


@dataclass
class Grid:
    occ: np.ndarray
    resolution: float
    origin: tuple[float, float]

    def __post_init__(self) -> None:
        self.occ = np.ascontiguousarray(self.occ, dtype=np.int8)
        self.resolution = float(self.resolution)
        self.origin = (float(self.origin[0]), float(self.origin[1]))

    # --- geometry -----------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int]:
        return int(self.occ.shape[0]), int(self.occ.shape[1])

    def world_to_cell(self, x, y):
        """World metres -> integer (ix, iy). Scalars in, ints out; arrays in,
        int64 arrays out. No bounds check (see `inside`)."""
        ix = np.floor((np.asarray(x, dtype=np.float64) - self.origin[0])
                      / self.resolution).astype(np.int64)
        iy = np.floor((np.asarray(y, dtype=np.float64) - self.origin[1])
                      / self.resolution).astype(np.int64)
        if ix.ndim == 0:
            return int(ix), int(iy)
        return ix, iy

    def cell_to_world(self, ix, iy):
        """Cell index -> world metres of that cell's CENTRE."""
        x = self.origin[0] + (np.asarray(ix, dtype=np.float64) + 0.5) * self.resolution
        y = self.origin[1] + (np.asarray(iy, dtype=np.float64) + 0.5) * self.resolution
        if np.ndim(x) == 0:
            return float(x), float(y)
        return x, y

    def inside(self, ix, iy):
        nx, ny = self.shape
        return (np.asarray(ix) >= 0) & (np.asarray(ix) < nx) \
            & (np.asarray(iy) >= 0) & (np.asarray(iy) < ny)

    # --- masks --------------------------------------------------------------
    def free(self) -> np.ndarray:
        return self.occ == FREE

    def occupied(self) -> np.ndarray:
        return self.occ == OCC

    def unknown(self) -> np.ndarray:
        return self.occ == UNKNOWN

    def explored_m2(self) -> float:
        """FREE cells x resolution^2 (spec 4.1: used for stagnation, status,
        summary). Walls are deliberately not counted: it is floor she knows."""
        return float(np.count_nonzero(self.occ == FREE)) * self.resolution ** 2


# --- normalisation ------------------------------------------------------------

def from_probability(prob: np.ndarray, origin, resolution: float,
                     free_below: float, occupied_above: float) -> Grid:
    """Probability of occupied (0..1, 0.5 = unknown) -> Grid.

    FREE if p <= free_below, OCC if p >= occupied_above, else UNKNOWN. Both
    ends INCLUSIVE, because that is what slam/grid.py's free_mask/occupied_mask
    do (`<= SLAM_P_FREE`, `>= SLAM_P_OCCUPIED`) and a cell must not be free for
    the planner and unknown here.
    """
    p = np.asarray(prob, dtype=np.float64)
    occ = np.full(p.shape, UNKNOWN, dtype=np.int8)
    occ[p <= free_below] = FREE
    occ[p >= occupied_above] = OCC
    return Grid(occ, resolution, (float(origin[0]), float(origin[1])))


def from_logodds(logodds: np.ndarray, origin, resolution: float,
                 free_below: float, occupied_above: float) -> Grid:
    """slam's log-odds array -> Grid, with slam's own probability formula."""
    lo = np.asarray(logodds, dtype=np.float64)
    prob = 1.0 - 1.0 / (1.0 + np.exp(lo))
    return from_probability(prob, origin, resolution, free_below, occupied_above)


def from_occupancy_grid(grid, free_below: float | None = None,
                        occupied_above: float | None = None) -> Grid:
    """A slam.grid.OccupancyGrid -> Grid.

    Thresholds default to the grid's OWN masks (config.SLAM_P_FREE /
    SLAM_P_OCCUPIED via free_mask/occupied_mask), so FREE here is exactly
    FREE for slam/plan.py. Pass thresholds only to override that on purpose.
    """
    if free_below is None and occupied_above is None:
        occ = np.full(grid.shape, UNKNOWN, dtype=np.int8)
        occ[grid.free_mask()] = FREE
        occ[grid.occupied_mask()] = OCC
        return Grid(occ, grid.cell, (float(grid.origin[0]), float(grid.origin[1])))
    return from_probability(grid.probability(), grid.origin, grid.cell,
                            free_below if free_below is not None else 0.25,
                            occupied_above if occupied_above is not None else 0.65)


# --- coarse grid (spec 4.1) -----------------------------------------------------

def downsample(grid: Grid, resolution_out: float) -> Grid:
    """Coarsen to `resolution_out` (the 10 cm segmentation grid).

    A coarse cell is OCC if ANY fine cell is OCC (a wall must never be averaged
    away), FREE if >= 50% of its fine cells are FREE, else UNKNOWN. The lattice
    stays anchored at the fine origin; the edge is padded with UNKNOWN.
    """
    k = int(round(float(resolution_out) / grid.resolution))
    if k <= 1:
        return Grid(grid.occ.copy(), grid.resolution, grid.origin)
    nx, ny = grid.shape
    px, py = (-nx) % k, (-ny) % k
    occ = np.pad(grid.occ, ((0, px), (0, py)), constant_values=UNKNOWN)
    bx, by = occ.shape[0] // k, occ.shape[1] // k
    blocks = occ.reshape(bx, k, by, k)
    any_occ = (blocks == OCC).any(axis=(1, 3))
    free_frac = (blocks == FREE).sum(axis=(1, 3)) / float(k * k)
    out = np.full((bx, by), UNKNOWN, dtype=np.int8)
    out[free_frac >= 0.5] = FREE
    out[any_occ] = OCC
    return Grid(out, grid.resolution * k, grid.origin)


# --- traversable (File 14's rule) -------------------------------------------------

def disc(radius_cells: int) -> np.ndarray:
    """The same disc slam/plan.py inflates with: gx^2 + gy^2 <= r^2."""
    r = max(int(radius_cells), 0)
    span = np.arange(-r, r + 1)
    gx, gy = np.meshgrid(span, span, indexing="ij")
    return ((gx ** 2 + gy ** 2) <= r ** 2).astype(np.uint8)


def inflate_cells(inflate_m: float, resolution: float) -> int:
    """ceil(inflate / cell), exactly as passable_mask computes it."""
    return int(math.ceil(float(inflate_m) / float(resolution)))


def traversable(grid: Grid, inflate_m: float) -> np.ndarray:
    """Cells the robot's CENTRE may occupy: FREE and not within `inflate_m` of
    an OCC cell. Unknown is not inflated and is never traversable -- the same
    as slam/plan.py::passable_mask."""
    solid = grid.occupied().astype(np.uint8)
    cells = inflate_cells(inflate_m, grid.resolution)
    if cells > 0 and solid.any():
        solid = cv2.dilate(solid, disc(cells), borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
    return grid.free() & (solid == 0)
