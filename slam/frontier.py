"""The edge of what Friday knows, and which bit of it is worth driving to.

A FRONTIER is a cell the map knows is free that touches a cell it knows
nothing about. Driving to one is the only move guaranteed to reveal something
new, and that is the whole argument for exploring this way: picking headings
re-covers floor that has already been covered and never finishes, because
nothing in a heading knows what has already been seen. Frontier exploration
terminates -- when there are no frontiers left, the space really is mapped.

⚠ CHOOSING BY DISTANCE ALONE OR BY SIZE ALONE BOTH FAIL, in opposite and
equally recognisable ways. Nearest-first leaves the robot nibbling round the
edge of the room it is standing in, taking the cheapest scrap every time.
Biggest-first sends it across the flat and back for every scrap that happens
to be wide. The quantity that matters is new area PER METRE TRAVELLED:

    utility = expected new cells / (driving distance + FRONTIER_NEAR_M)

DISTANCE IS BY ROAD, NOT BY CROW. A frontier two metres away through a wall is
not two metres away, and ranking by straight lines sends the robot at walls
all day. slam/plan.DistanceField gives true path distance to every cell in a
single Dijkstra, so ranking a dozen candidates costs a dozen array lookups.

EXPECTED NEW AREA IS NOT THE FRONTIER'S WIDTH. A two-cell gap into an unmapped
room is worth far more than a wide frontier lying against the sensor's range
limit in floor that is already half seen, and width cannot tell them apart.
So the gain is the count of UNKNOWN cells near the frontier, which can.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as _field

import numpy as np
from scipy.ndimage import binary_dilation, label

import config
from slam.grid import OccupancyGrid
from slam.plan import DistanceField

_EIGHT = np.ones((3, 3), dtype=bool)


@dataclass
class Frontier:
    """One contiguous stretch of the known/unknown boundary."""

    centroid: tuple[float, float]
    approach: tuple[float, float]     # where to actually drive, world metres
    size: int                         # frontier cells
    gain: int                         # unknown cells it is expected to reveal
    distance: float                   # by road, metres
    utility: float
    cells: np.ndarray = _field(repr=False, default=None)

    def key(self, cell_m: float = 0.25) -> tuple[int, int]:
        """A coarse identity, so the same frontier is recognised between
        cycles even though its cells shift as the map fills in."""
        return (int(round(self.centroid[0] / cell_m)),
                int(round(self.centroid[1] / cell_m)))


def frontier_mask(grid: OccupancyGrid) -> np.ndarray:
    """Free cells that touch an unknown one."""
    unknown = grid.unknown_mask()
    if not unknown.any():
        return np.zeros(grid.shape, dtype=bool)
    return grid.free_mask() & binary_dilation(unknown, structure=_EIGHT)


def _split(labelled: np.ndarray, groups: int, cell_m: float):
    """Connected frontier arcs, chopped into destination-sized pieces.

    Connected components alone are not enough. Early in a run every free cell
    at the edge of what has been seen touches unknown, and those cells form
    ONE arc wrapping most of the way round the robot -- 559 cells of it on the
    test flat. Averaging that gives a centroid in the middle of the room the
    robot is already standing in, so it drives nowhere and the run ends in the
    first corridor claiming the flat is mapped.

    Chopping is by position, into bins about a doorway wide, which is the
    scale at which "over there" is a real instruction. Small arcs are left
    alone -- there is nothing to gain from cutting up a gap that is already
    the size of a destination.
    """
    span = max(int(round(config.FRONTIER_SPLIT_M / cell_m)), 1)
    for tag in range(1, groups + 1):
        cells = np.argwhere(labelled == tag)
        if len(cells) <= config.FRONTIER_MAX_CELLS:
            yield cells
            continue
        bins = cells // span
        keys = bins[:, 0] * 100000 + bins[:, 1]
        for key in np.unique(keys):
            yield cells[keys == key]


def find(grid: OccupancyGrid, distances: DistanceField,
         min_cells: int | None = None,
         ignore: set | None = None) -> list[Frontier]:
    """Every frontier worth considering, best first.

    A frontier whose own cells are all unreachable -- sealed behind the
    inflation of a doorway, or across a threshold too narrow for the body --
    is dropped rather than kept as an eternally-pending goal.
    """
    min_cells = config.FRONTIER_MIN_CELLS if min_cells is None else min_cells
    ignore = ignore or set()
    mask = frontier_mask(grid)
    if not mask.any() or not distances.ok:
        return []

    labelled, groups = label(mask, structure=_EIGHT)
    unknown = grid.unknown_mask()
    reach = distances.distance_map()
    radius = int(round(config.FRONTIER_GAIN_RADIUS_M / grid.cell))

    out: list[Frontier] = []
    for group in _split(labelled, groups, grid.cell):
        cells = group
        if len(cells) < min_cells:
            continue

        here = reach[cells[:, 0], cells[:, 1]]
        usable = np.isfinite(here)
        if not usable.any():
            continue

        # ⚠ DRIVE TO THE MIDDLE OF A FRONTIER, NOT TO ITS NEAREST CORNER.
        # The nearest reachable cell of a frontier is almost always the one
        # directly under the robot -- the edge of what this very sweep could
        # see. Targeting it produces a path zero or one cells long, so the
        # robot "arrives" without moving, learns nothing, and the run stalls
        # in the first room while reporting that it is exploring. The
        # centroid is an actual destination.
        centre = cells.mean(axis=0)
        to_centre = np.hypot(cells[:, 0] - centre[0], cells[:, 1] - centre[1])
        pick = int(np.argmin(np.where(usable, to_centre, np.inf)))
        approach_cell = cells[pick]
        distance = float(here[pick])

        cx, cy = int(round(centre[0])), int(round(centre[1]))
        window = unknown[max(cx - radius, 0):cx + radius + 1,
                         max(cy - radius, 0):cy + radius + 1]
        gain = int(window.sum())

        centroid = tuple((centre + 0.5) * grid.cell + grid.origin)
        item = Frontier(
            centroid=(float(centroid[0]), float(centroid[1])),
            approach=tuple(float(v) for v in
                           (approach_cell + 0.5) * grid.cell + grid.origin),
            size=int(len(cells)), gain=gain, distance=distance,
            utility=gain / (distance + config.FRONTIER_NEAR_M),
            cells=cells)
        if item.key() in ignore:
            continue
        out.append(item)

    out.sort(key=lambda f: f.utility, reverse=True)
    return out


def choose(grid: OccupancyGrid, distances: DistanceField,
           ignore: set | None = None) -> Frontier | None:
    """The single best frontier, or None when the space is fully mapped.

    None is a real answer and the caller should treat it as one: it is how
    exploration finishes. It also means "nothing reachable is left", which is
    not quite the same thing -- a sealed room stays unmapped and Friday should
    say she is done rather than claim the flat is fully explored.
    """
    options = find(grid, distances, ignore=ignore)
    return options[0] if options else None


def describe(item: Frontier | None, pose: tuple[float, float, float]) -> str:
    """One short line about where the next frontier is, in robot terms.

    Bearings are relative to the nose and in plain words, because this ends up
    spoken aloud -- 'about two metres ahead and to my left' is what a person
    can picture. See personality.md for why the phrasing stays this short.
    """
    if item is None:
        return "nothing left to explore that I can reach"
    dx = item.approach[0] - pose[0]
    dy = item.approach[1] - pose[1]
    bearing = math.degrees(math.atan2(dy, dx) - pose[2])
    bearing = (bearing + 180.0) % 360.0 - 180.0
    if abs(bearing) < 25:
        where = "straight ahead"
    elif abs(bearing) > 155:
        where = "behind me"
    elif bearing > 0:
        where = "to my left" if bearing < 115 else "back to my left"
    else:
        where = "to my right" if bearing > -115 else "back to my right"
    return f"unexplored space {where}, about {item.distance:.1f} metres away"
