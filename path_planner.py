"""path_planner.py — file 14 steps 1-3: from occupancy grid to a drivable route.

    grid ──> despeckle ──> inflate ──> A* ──> smooth ──> waypoints
     (step 1, read-only on a copy)   (step 2)  (step 3)

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT REBUILD
---------------------------------------------------------
Friday has had a planner since 2026-08-26. `slam/plan.py::passable_mask` already
inflates obstacles by `config.NAV_INFLATE_M` (robot radius 0.21 m + clearance
0.09 m) and already refuses to plan across unknown cells, and `DistanceField`
already answers "how far is every reachable cell" in one Dijkstra pass. Two of
step 1's three requirements were therefore already met, by code that drives the
real robot today. Writing a second inflater would mean two numbers to keep in
step, and the day they disagreed the wrong one would be the one steering.

So this module OWNS exactly one new thing -- speck filtering -- and calls
`slam/plan.py` for the rest. Nothing in `slam/` or `rl/` is modified by this
file; it reads through the accessors those modules already expose
(`OccupancyGrid.load()` for the saved survey, `WorldMap.grid_snapshot()` for a
live run) and works on a private copy.

WHY SPECKS MATTER MORE THAN THEIR SIZE SUGGESTS
-----------------------------------------------
A speck is one or two isolated cells the map believes are occupied, with no
wall structure attached: a LiDAR return off a chair leg at a glancing angle, a
dust mote, a scan-match hiccup. Ignore them and they are 25 cm² of nothing.
Inflate them and each one becomes a 60 cm disc of no-go -- 2,800 cm², a hundred
times its own size -- dropped in the middle of open floor. A handful across a
doorway seals it, and the planner then reports, correctly and uselessly, that
there is no route to the kitchen.

MEASURED on the two maps as they stand on 2026-09-21:

    state/map.npz      326 occupied cells ->  10 specks (14 cells, 4.3%)
                       plannable cells 3,053 -> 3,293  (+7.9%)
    state/rl_map.npz   249 occupied cells -> 189 specks (203 cells, 81.5%)
                       plannable cells 92,381 -> 103,882  (+12.4%)

Read the second line twice. FOUR FIFTHS of the RL map's walls are specks, and
what survives filtering is 46 cells of wall in a 480x434 grid -- a room with
essentially no structure in it. Filtering does not fix that map; it measures
how little is in it. See [[friday-map-has-saturated]], and note that this is
why `from_saved_map` defaults to the survey map and not that one.

⚠ THE HONEST RISK, AND WHY IT IS ACCEPTED. At a 5 cm cell, a genuinely thin
obstacle -- a table leg, a mic stand, a chair's castor -- can legitimately BE
one or two cells. Filtering is not free: it can clear a real object and let a
route be planned straight over it. Three things make that survivable, and all
three must stay true:

  * this only changes what is PLANNED, never what is driven. Every leg still
    goes out through `safety.guarded_move`, which re-checks clearance every
    100 ms and stops the robot for anything physically in front of it;
  * `rl/near_memory.py` keeps obstacles alive near the body for a few seconds
    after the sensors lose them, and the transient layer blocks the person who
    is standing there right now, neither of which reads this mask;
  * a real object is seen again on the next sweep, and a cell seen twice is no
    longer a speck.

A filtered speck therefore costs a stop, not a collision. An unfiltered one
costs a room Friday will not enter.

⚠ ORDER IS NOT NEGOTIABLE: despeckle FIRST, inflate SECOND. Inflating first
turns each speck into a disc that is far larger than two cells, and nothing
downstream can tell that disc from a wall.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label

import config
from slam.grid import OccupancyGrid
from slam.plan import passable_mask

# A component of this many cells or fewer, touching nothing larger, is noise.
# Two, not three: at 5 cm a 3-cell component is 15 cm of something, which is a
# stool leg or a cable coil -- small, but large enough that driving through it
# is a collision rather than a scare.
SPECK_MAX_CELLS = 2

# 8-connectivity, so a diagonal pair counts as ONE two-cell speck rather than
# two singles. A wall seen at 45 degrees is a diagonal chain of cells, and
# under 4-connectivity every cell of it would look isolated.
_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)


def _free_logodds() -> float:
    """A log-odds value that `free_mask` will accept, and no more than that.

    Cleared specks are set to just past the free threshold rather than to
    SLAM_LOGODDS_MIN. The map has not proved this cell is empty -- it has
    failed to prove it is a wall -- and writing in certainty nobody measured is
    how a copy of a map stops matching the map. One more sweep of real evidence
    overrules this; a clamped extreme would take ten.
    """
    p = max(min(config.SLAM_P_FREE - 0.01, 0.499), 0.001)
    return math.log(p / (1.0 - p))


@dataclass
class Preprocessed:
    """The plannable mask, plus what had to be done to get it.

    The counts are here because a mask alone cannot tell you why a route was
    refused. `specks` climbing run over run means the map is getting noisier,
    not that the room is getting fuller.
    """

    grid: OccupancyGrid                 # the despeckled COPY, never the caller's
    plannable: np.ndarray               # bool, True where the robot's centre may sit
    cell_m: float
    inflate_m: float
    specks: int = 0                     # components cleared
    speck_cells: int = 0                # cells in them
    occupied_cells: int = 0             # after despeckling
    free_cells: int = 0
    unknown_cells: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def plannable_cells(self) -> int:
        return int(self.plannable.sum())

    @property
    def plannable_m2(self) -> float:
        return self.plannable_cells * self.cell_m ** 2

    def stats(self) -> dict:
        """Plain counts, for telemetry and for the dashboard."""
        return {
            "cell_m": self.cell_m,
            "inflate_m": round(self.inflate_m, 3),
            "specks_cleared": self.specks,
            "speck_cells": self.speck_cells,
            "occupied_cells": self.occupied_cells,
            "free_cells": self.free_cells,
            "unknown_cells": self.unknown_cells,
            "plannable_cells": self.plannable_cells,
            "plannable_m2": round(self.plannable_m2, 2),
            "notes": list(self.notes),
        }

    def __str__(self) -> str:
        return (f"{self.plannable_cells} plannable cells "
                f"({self.plannable_m2:.1f} m²), {self.specks} specks cleared, "
                f"inflated by {self.inflate_m:.2f} m")


def copy_grid(grid: OccupancyGrid) -> OccupancyGrid:
    """A detached copy, so nothing here can write through to a live map.

    Built the same way `WorldMap.grid_snapshot` builds one. The caller may be
    handing in the grid a run is actively folding sweeps into, and a mask
    computed from an array that reallocates underneath you is the kind of bug
    that only shows up on the robot.
    """
    clone = OccupancyGrid(grid.cell)
    clone.logodds = np.asarray(grid.logodds).copy()
    clone.origin = np.asarray(grid.origin).copy()
    clone.scans = int(grid.scans)
    clone.updated_at = float(grid.updated_at)
    clone._empty = bool(grid._empty)
    return clone


def speck_mask(occupied: np.ndarray,
               max_cells: int = SPECK_MAX_CELLS) -> tuple[np.ndarray, int]:
    """Which occupied cells are isolated noise. Returns (mask, component count).

    Connected-component labelling does the whole job: under 8-connectivity a
    component of one or two cells IS, by definition, one that touches no larger
    structure. There is nothing further to test.
    """
    occupied = np.asarray(occupied, dtype=bool)
    if not occupied.any() or max_cells <= 0:
        return np.zeros_like(occupied), 0

    labelled, count = label(occupied, structure=_CONNECTIVITY_8)
    if count == 0:
        return np.zeros_like(occupied), 0

    # sizes[0] is the background; component i occupies sizes[i].
    sizes = np.bincount(labelled.ravel(), minlength=count + 1)
    small = np.flatnonzero(sizes[1:] <= max_cells) + 1
    if not len(small):
        return np.zeros_like(occupied), 0

    mask = np.isin(labelled, small)
    return mask, int(len(small))


def despeckle(grid: OccupancyGrid,
              max_cells: int = SPECK_MAX_CELLS
              ) -> tuple[OccupancyGrid, int, int]:
    """A copy of `grid` with isolated specks demoted to free.

    Returns (copy, components cleared, cells cleared). The input is never
    touched -- this is the whole reason the copy is taken first.
    """
    clone = copy_grid(grid)
    if clone._empty:
        return clone, 0, 0

    mask, count = speck_mask(clone.occupied_mask(), max_cells)
    if count:
        clone.logodds[mask] = _free_logodds()
    return clone, count, int(mask.sum())


def preprocess(grid: OccupancyGrid,
               extra_blocked: np.ndarray | None = None,
               inflate_m: float | None = None,
               max_speck_cells: int = SPECK_MAX_CELLS) -> Preprocessed:
    """Everything step 1 asks for, in the order it has to happen.

    1. copy the grid, so the exploration system's own map is untouched
    2. clear isolated specks (they would inflate into discs of phantom wall)
    3. inflate what is left by robot radius + clearance, via `slam/plan.py`
    4. keep only cells confirmed FREE -- unknown is not plannable, which is
       `free_mask`'s rule already and is not relaxed here

    `extra_blocked` is world-frame points to block on top of the map, passed
    straight through to `passable_mask`: the transient layer's returns, so a
    person who is not on the map is still planned around. They are added AFTER
    despeckling and are never filtered -- a live return is evidence about now,
    not a stale mark on a map, and two cells of person is still a person.
    """
    inflate_m = config.NAV_INFLATE_M if inflate_m is None else float(inflate_m)
    clean, specks, speck_cells = despeckle(grid, max_speck_cells)

    notes: list[str] = []
    if clean._empty:
        empty = np.zeros(clean.shape, dtype=bool)
        notes.append("the map is empty — nothing has been folded into it yet")
        return Preprocessed(grid=clean, plannable=empty, cell_m=clean.cell,
                            inflate_m=inflate_m, notes=notes)

    plannable = passable_mask(clean, extra_blocked=extra_blocked,
                              inflate_m=inflate_m)

    occupied = int(clean.occupied_mask().sum())
    free = int(clean.free_mask().sum())
    result = Preprocessed(
        grid=clean, plannable=plannable, cell_m=clean.cell,
        inflate_m=inflate_m, specks=specks, speck_cells=speck_cells,
        occupied_cells=occupied, free_cells=free,
        unknown_cells=int(clean.unknown_mask().sum()), notes=notes)

    # Two failures worth naming at the point they happen, rather than leaving
    # the caller to work them out from a mask of zeros.
    if not plannable.any():
        notes.append("no cell is plannable — everything free is within "
                     f"{inflate_m:.2f} m of something solid")
    elif free and result.plannable_cells < free * 0.10:
        notes.append(f"inflation removed {100 * (1 - result.plannable_cells / free):.0f}% "
                     "of the free space — suspect the map before the planner")
    if occupied and speck_cells > occupied:
        notes.append("more cells were specks than remain as wall — this map is "
                     "noise with a room in it")
    return result


def from_saved_map(path: Path | str | None = None,
                   **kwargs) -> Preprocessed:
    """Preprocess the saved survey map (`state/map.npz` by default).

    This is the frame `go_to` and `slam/places.py` work in: scan-matched, and
    the only one with a relocaliser that refuses rather than guesses. The RL
    stack's `state/rl_map.npz` is dead-reckoned and currently carries hundreds
    of square metres of free space against under one of wall, so a route
    planned across it would be a confident path through floor nobody has
    stood on. Pass that grid in explicitly if you want it; it is not the
    default and should not become one without the drift being fixed first.
    """
    return preprocess(OccupancyGrid.load(path), **kwargs)


def from_world(world, **kwargs) -> Preprocessed | None:
    """Preprocess a live run's map, via `WorldMap.grid_snapshot()`.

    Returns None when the run has not mapped anything yet. Read-only: the
    snapshot is already a private copy, and this takes another.
    """
    snapshot = world.grid_snapshot()
    if snapshot is None:
        return None
    return preprocess(snapshot, **kwargs)


# =====================================================================
# STEP 2 — A*, and the reason it exists next to a Dijkstra that works
# =====================================================================
# `slam/plan.py::DistanceField` runs ONE Dijkstra from the robot to every
# reachable cell, and that is the right tool for the job it was built for:
# frontier exploration compares a dozen candidate destinations, so solving all
# of them at once is cheaper than a dozen searches. File 14's job is the
# opposite shape -- one known destination, "take me home" -- and there a
# heuristic search stops as soon as it touches the goal instead of flooding the
# whole map first. On the RL map's 103,882 plannable cells that is the
# difference between expanding a hundred thousand cells and expanding a few
# thousand. Neither replaces the other; they answer different questions.
#
# WHY THE COST GRADIENT, AND WHY IT IS NOT MORE INFLATION. A path that is
# technically legal can still run with its shoulder against the wall for six
# metres, because every cell on it is passable and the planner has no reason to
# prefer one legal cell over another. On a robot whose pose is dead-reckoned
# and worth a few centimetres, that path is one pose error away from a scrape.
# The fix is NOT to inflate further -- inflation is binary and sealing another
# 10 cm would close doorways that are genuinely passable (see the warning in
# slam/plan.py). A soft gradient keeps every legal cell legal and merely makes
# the ones near walls expensive, so open floor is preferred WHERE THERE IS
# OPEN FLOOR and a narrow doorway is still taken when it is the only way.

# How far from an obstacle the gradient still costs anything. Beyond this,
# open floor is flat and the planner is free to take the geometrically
# shortest route.
NAV_GRADIENT_M = 0.35

# Cost of hugging the inflation boundary, as a multiple of a straight step.
# At 1.0 a wall-grazing step costs double, so the search will detour up to
# about its own length to find the middle of a corridor -- and no further,
# which is what keeps a doorway usable when it is the only route.
#
# MEASURED, rounding a block's corner in a 5 m room (scripts/test_path_planner.py):
#
#     weight 0.0   5.99 m   minimum clearance 0.050 m   <- shoulder on the wall
#     weight 1.0   6.65 m   minimum clearance 0.320 m   <- +11% length
#     weight 3.0   6.71 m   minimum clearance 0.350 m   <- no further gain
#
# Clearance saturates at NAV_GRADIENT_M because past that the cost is flat, so
# turning the weight up buys nothing: the REACH is the tuning knob, not the
# weight. 1.0 already sits at the knee.
NAV_GRADIENT_WEIGHT = 1.0

# A ceiling on work, not on distance. A search that has expanded this many
# cells on this Pi has been running for seconds; returning "no route" is more
# useful than blocking the caller. 0 disables the cap.
NAV_MAX_EXPANSIONS = 250_000

# How far the search may look for a legal cell when the robot (or the goal) is
# standing somewhere inflation calls solid. Same rule and same metre as
# `DistanceField._nearest_passable`, deliberately: two planners that disagree
# about where the robot is would be worse than either.
NAV_SNAP_M = 1.0

_SQRT2 = math.sqrt(2.0)


def clearance_m(plannable: np.ndarray, cell_m: float) -> np.ndarray:
    """Metres from every cell to the nearest cell the robot may not enter.

    Measured on the PLANNABLE mask, so the distance is already from the
    inflated boundary: 0.0 means "one step from where the body would touch",
    not "touching". The map edge counts as solid -- an unmapped border is not
    open floor, and a path that hugs the edge of the survey is hugging the
    part of the room nobody has looked at.
    """
    padded = np.zeros((plannable.shape[0] + 2, plannable.shape[1] + 2), bool)
    padded[1:-1, 1:-1] = plannable
    return (distance_transform_edt(padded)[1:-1, 1:-1] * cell_m).astype(np.float32)


def gradient_cost(plannable: np.ndarray, cell_m: float,
                  gradient_m: float = NAV_GRADIENT_M,
                  weight: float = NAV_GRADIENT_WEIGHT) -> np.ndarray:
    """Extra metres of cost charged for entering each cell, near walls only.

    Linear from `weight * cell_m` at the inflation boundary down to zero at
    `gradient_m` away. Linear rather than exponential because it has to be
    read and reasoned about on a robot, and because the shape of the falloff
    matters far less than the fact that there is one.

    ⚠ EVERY VALUE HERE IS >= 0, and that is load-bearing: it is what keeps the
    octile heuristic admissible. A negative cost anywhere and A* would start
    returning paths that are not shortest, silently.
    """
    if gradient_m <= 0.0 or weight <= 0.0:
        return np.zeros(plannable.shape, np.float32)
    d = clearance_m(plannable, cell_m)
    near = np.clip(1.0 - d / gradient_m, 0.0, 1.0)
    return (near * weight * cell_m).astype(np.float32)


def _snap(plannable: np.ndarray, cell: tuple[int, int],
          snap_m: float, cell_m: float) -> tuple[int, int] | None:
    """The nearest plannable cell, or None if there is none close enough."""
    nx, ny = plannable.shape
    ix, iy = int(cell[0]), int(cell[1])
    if 0 <= ix < nx and 0 <= iy < ny and plannable[ix, iy]:
        return (ix, iy)
    if not plannable.any() or snap_m <= 0:
        return None
    legal = np.argwhere(plannable)
    away = np.hypot(legal[:, 0] - ix, legal[:, 1] - iy)
    best = int(np.argmin(away))
    if away[best] * cell_m > snap_m:
        return None
    return (int(legal[best, 0]), int(legal[best, 1]))


def _octile(ax: int, ay: int, bx: int, by: int, cell_m: float) -> float:
    """Exact cost of the cheapest obstacle-free 8-connected route, in metres.

    This is the right heuristic for an 8-connected grid and it is admissible
    BECAUSE obstacles and the gradient can only ever make the real route more
    expensive, never less. Manhattan would overestimate here and would start
    returning non-shortest paths; Euclidean underestimates and just searches
    more cells than it needs to.
    """
    dx = abs(ax - bx)
    dy = abs(ay - by)
    lo, hi = (dx, dy) if dx < dy else (dy, dx)
    return (hi - lo + _SQRT2 * lo) * cell_m


def astar(grid, start_cell: tuple[int, int], goal_cell: tuple[int, int],
          inflation_radius: float | None = None, *,
          gradient_m: float = NAV_GRADIENT_M,
          gradient_weight: float = NAV_GRADIENT_WEIGHT,
          allow_diagonal: bool | None = None,
          snap_m: float = NAV_SNAP_M,
          max_expansions: int = NAV_MAX_EXPANSIONS,
          ) -> list[tuple[int, int]]:
    """A route from `start_cell` to `goal_cell` as a list of grid cells.

    Returns [] when there is no route -- never an approximation of one, and
    never the direction of one. An unreachable destination is a fact to report;
    driving in its general direction is how a robot ends up somewhere nobody
    asked for.

    Args:
        grid: an `OccupancyGrid`, or a `Preprocessed` from step 1. Pass the
            latter when planning repeatedly against one map -- preprocessing
            costs a despeckle and a dilation, and redoing it per call on the
            480x434 map is most of the planning budget.
        start_cell, goal_cell: (ix, iy) indices into the grid.
        inflation_radius: metres, defaults to `config.NAV_INFLATE_M`. Ignored
            when `grid` is already a `Preprocessed`, which carries its own.
        gradient_m, gradient_weight: the soft cost near walls. Set the weight
            to 0 for a plain shortest path.
        allow_diagonal: defaults to `config.NAV_ALLOW_DIAGONAL`.
        snap_m: how far to look for a legal cell if start or goal is inside
            the inflated region -- which the start very often is, because
            parking next to a wall puts the robot there.
        max_expansions: work ceiling; 0 for none.
    """
    prep = (grid if isinstance(grid, Preprocessed)
            else preprocess(grid, inflate_m=inflation_radius))
    plannable = prep.plannable
    cell_m = prep.cell_m
    if not plannable.any():
        return []

    diagonal = (config.NAV_ALLOW_DIAGONAL if allow_diagonal is None
                else bool(allow_diagonal))

    start = _snap(plannable, start_cell, snap_m, cell_m)
    goal = _snap(plannable, goal_cell, snap_m, cell_m)
    if start is None or goal is None:
        return []
    if start == goal:
        return [start]

    nx, ny = plannable.shape
    extra = gradient_cost(plannable, cell_m, gradient_m, gradient_weight)

    flat_ok = plannable.ravel()
    flat_extra = extra.ravel()
    n = flat_ok.size
    g = np.full(n, np.inf, np.float64)
    came = np.full(n, -1, np.int64)
    closed = np.zeros(n, bool)

    def idx(ix: int, iy: int) -> int:
        return ix * ny + iy

    start_i, goal_i = idx(*start), idx(*goal)
    gx, gy = goal
    g[start_i] = 0.0
    heap = [(_octile(start[0], start[1], gx, gy, cell_m), start_i)]
    expansions = 0

    # Straights first, diagonals last, so the gate below can be applied to
    # exactly the steps that need it.
    steps = [(1, 0, cell_m), (-1, 0, cell_m), (0, 1, cell_m), (0, -1, cell_m)]
    if diagonal:
        steps += [(1, 1, _SQRT2 * cell_m), (1, -1, _SQRT2 * cell_m),
                  (-1, 1, _SQRT2 * cell_m), (-1, -1, _SQRT2 * cell_m)]

    while heap:
        _, current = heapq.heappop(heap)
        if closed[current]:
            continue
        if current == goal_i:
            break
        closed[current] = True
        expansions += 1
        if max_expansions and expansions > max_expansions:
            return []

        cx, cy = divmod(current, ny)
        here = g[current]
        for dx, dy, step in steps:
            ax, ay = cx + dx, cy + dy
            if not (0 <= ax < nx and 0 <= ay < ny):
                continue
            nxt = idx(ax, ay)
            if closed[nxt] or not flat_ok[nxt]:
                continue
            # A diagonal step needs both of its orthogonal neighbours, so a
            # path cannot slip through the corner where two obstacles touch.
            # That gap is zero centimetres wide and the robot wedges in it.
            # `slam/plan.py::_graph` gates it the same way, for the same
            # reason found on the same chassis.
            if dx and dy and not (flat_ok[idx(cx + dx, cy)]
                                  and flat_ok[idx(cx, cy + dy)]):
                continue
            tentative = here + step + flat_extra[nxt]
            if tentative < g[nxt]:
                g[nxt] = tentative
                came[nxt] = current
                heapq.heappush(
                    heap, (tentative + _octile(ax, ay, gx, gy, cell_m), nxt))

    if not np.isfinite(g[goal_i]):
        return []

    chain = [goal_i]
    while came[chain[-1]] >= 0:
        chain.append(int(came[chain[-1]]))
    chain.reverse()
    return [(int(i // ny), int(i % ny)) for i in chain]


# =====================================================================
# STEP 3 — smoothing the staircase out
# =====================================================================
# An 8-connected search returns a staircase whenever the true line lies
# between two of its eight headings: forty waypoints where three would do.
# That matters more on this chassis than on most, because it cannot arc --
# `path_to_moves` turns every waypoint into a STOP, a TURN and a DRIVE, and
# this is the chassis whose turns fail outright 22% of the time
# ([[friday-drivetrain-execution-floors]]). Every waypoint removed is a turn
# that cannot fail to happen.
#
# ⚠ DELIBERATE DEVIATION FROM THE SPEC, WHICH ASKED FOR A BRESENHAM CHECK.
# Raw Bresenham steps diagonally between two cells that touch at a corner, so
# it reports line of sight through exactly the zero-width gap the diagonal
# gate above exists to forbid -- it would smooth a safe path into an
# unsafe one. Sampling the segment at half-cell intervals cannot skip a cell
# at any angle, and is what `DistanceField.clear_between` already does on the
# same maps. Same answer as a supercover Bresenham, four lines instead of
# twenty, and consistent with the planner that drives today.


def line_of_sight(plannable: np.ndarray, a: tuple[int, int],
                  b: tuple[int, int]) -> bool:
    """Is every cell the segment a->b passes through plannable?

    ⚠ THE SAMPLING ALONE IS NOT ENOUGH, and finer sampling does not fix it.
    A segment at exactly 45 degrees runs along the line x = y, so every point
    sampled on it rounds to a cell ON that diagonal -- (5,5), (6,6) -- and
    never to the two cells it squeezes between, (5,6) and (6,5). Sample it a
    thousand times and those two are still never tested. So a pair of
    obstacles touching at a corner would be reported as clear sight, and
    smoothing would turn a safe path into one that drives through a
    zero-width gap: exactly the wedge the A* diagonal gate exists to prevent,
    reintroduced by the step that was supposed to tidy the path up.

    The second half of this function is therefore not an optimisation. Every
    diagonal move between consecutive sampled cells must also have both of its
    orthogonal companions free, which is the same gate `_graph` and `astar`
    apply, enforced here on the line instead of on the lattice.
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    span = math.hypot(bx - ax, by - ay)
    steps = max(int(span / 0.5) + 1, 2)
    t = np.linspace(0.0, 1.0, steps)
    ix = np.rint(ax + (bx - ax) * t).astype(np.int64)
    iy = np.rint(ay + (by - ay) * t).astype(np.int64)
    nx, ny = plannable.shape
    if (ix < 0).any() or (iy < 0).any() or (ix >= nx).any() or (iy >= ny).any():
        return False
    if not plannable[ix, iy].all():
        return False

    # Half-cell sampling moves at most one cell per step, so a diagonal step
    # is exactly the corner case above.
    dx, dy = np.diff(ix), np.diff(iy)
    diag = (dx != 0) & (dy != 0)
    if not diag.any():
        return True
    px, py = ix[:-1][diag], iy[:-1][diag]
    return bool(plannable[px + dx[diag], py].all()
                and plannable[px, py + dy[diag]].all())


def smooth(path: list[tuple[int, int]], plannable: np.ndarray,
           max_passes: int = 8) -> list[tuple[int, int]]:
    """Shortcut a raw grid path down to the waypoints that earn their place.

    Two stages, because one of them alone is either too slow or not good
    enough:

    1. DROP-THE-MIDDLE, repeatedly: a waypoint whose neighbours can see each
       other is doing nothing. This is what the spec describes, and it
       collapses a staircase fast -- each pass roughly halves it, so a
       40-waypoint stair is down to single figures in three or four passes,
       at one cheap line check per waypoint.
    2. FARTHEST-VISIBLE, once, on what is left. This is the stronger
       string-pull and it is O(n^2) line checks, which is why it runs second:
       after stage 1 there are a handful of waypoints, so the square of that
       is nothing. Run on the raw path it would be tens of thousands of
       checks on a Pi that is already CPU-bound ([[friday-pi-is-cpu-bound]]).

    The endpoints are never moved. The start is where the robot actually is
    and the goal is what was asked for; shortcutting either would mean
    arriving somewhere adjacent to the answer.
    """
    if len(path) <= 2:
        return list(path)

    kept = list(path)
    for _ in range(max_passes):
        out = [kept[0]]
        i = 1
        while i < len(kept) - 1:
            if line_of_sight(plannable, out[-1], kept[i + 1]):
                i += 1                       # kept[i] adds nothing: drop it
            else:
                out.append(kept[i])
                i += 1
        out.append(kept[-1])
        if len(out) == len(kept):
            break
        kept = out

    pulled = [kept[0]]
    i = 0
    while i < len(kept) - 1:
        far = i + 1
        for j in range(len(kept) - 1, i, -1):
            if line_of_sight(plannable, kept[i], kept[j]):
                far = j
                break
        pulled.append(kept[far])
        i = far
    return pulled


def cells_to_world(cells: list[tuple[int, int]],
                   grid: OccupancyGrid) -> np.ndarray:
    """Cell indices -> world-metre waypoints at cell centres.

    This is the hand-off to the existing follower: the result goes straight
    into `slam/plan.py::path_to_moves(path, heading_rad)`, which is what turns
    waypoints into the (turn degrees, drive metres) pairs this chassis
    actually executes. Same (N, 2) float array `DistanceField.path_to`
    returns, so anything that consumes one consumes the other.
    """
    if not cells:
        return np.empty((0, 2))
    arr = np.asarray(cells, dtype=np.float64)
    return (arr + 0.5) * grid.cell + grid.origin


def plan(grid, start_cell: tuple[int, int], goal_cell: tuple[int, int],
         **kwargs) -> tuple[list[tuple[int, int]], Preprocessed]:
    """Steps 1-3 in one call: preprocess, search, smooth.

    Returns (smoothed cells, the Preprocessed it planned on). The second is
    worth keeping: it carries the speck counts and the notes explaining a
    refusal, and it can be handed back in as `grid` to plan again without
    paying for preprocessing twice.
    """
    prep = (grid if isinstance(grid, Preprocessed)
            else preprocess(grid, inflate_m=kwargs.pop("inflation_radius", None)))
    raw = astar(prep, start_cell, goal_cell, **kwargs)
    return smooth(raw, prep.plannable), prep
