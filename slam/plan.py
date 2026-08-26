"""Getting from here to there across the map, without touching anything.

Three pieces, in the order they are used:

    passable_mask   which cells the robot may occupy, obstacles inflated
    distance_field  how far every reachable cell is, by road not by crow
    path_to         the actual sequence of waypoints to one of them

WHY ONE DISTANCE FIELD RATHER THAN A* PER GOAL. Frontier exploration has to
compare a dozen candidate destinations before it picks one, and A* would
re-plan from scratch for each. A single Dijkstra from the robot gives the true
path distance to EVERY cell in one pass, so comparing candidates is then an
array lookup. Planning the chosen route afterwards is free: the predecessors
are already there.

WHY THE ROBOT IS PLANNED AS A POINT. Obstacles are inflated by the robot's
radius plus a margin (config.NAV_INFLATE_M) before anything is searched, so a
path that exists is a path the body fits through. The alternative -- checking
a footprint at every cell of every candidate -- costs the same thing many
times over.

⚠ INFLATION IS ALSO THE FIRST THING TO SUSPECT when Friday refuses to plan
through a door that is plainly wide enough. 30 cm of inflation on both jambs
leaves 10 cm of a 70 cm doorway, and one stray cell closes it. The planner is
usually right and the map is usually the problem.

UNKNOWN IS NOT PASSABLE. Same rule as everywhere else in slam/: a path is
only a path across floor somebody has actually looked at. This is what makes
frontier exploration work rather than being a nuisance -- the robot drives to
the EDGE of what it knows, looks, and the edge moves.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

import config
from slam.grid import OccupancyGrid

# (dx, dy) for the eight neighbours, diagonals last so they can be gated.
_STRAIGHT = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def passable_mask(grid: OccupancyGrid,
                  extra_blocked: np.ndarray | None = None,
                  inflate_m: float | None = None) -> np.ndarray:
    """Cells the robot's centre may legally sit in.

    `extra_blocked` is world points to block on top of the map -- the
    transient layer's returns, so a person who is not on the map is still
    planned around. They inflate exactly like walls do.
    """
    inflate_m = config.NAV_INFLATE_M if inflate_m is None else inflate_m
    solid = grid.occupied_mask()
    if extra_blocked is not None and len(extra_blocked):
        here = grid.inside(extra_blocked)
        if here.any():
            ix, iy = grid.cells_of(extra_blocked[here])
            solid = solid.copy()
            solid[ix, iy] = True

    cells = int(math.ceil(inflate_m / grid.cell))
    if cells > 0:
        span = np.arange(-cells, cells + 1)
        gx, gy = np.meshgrid(span, span, indexing="ij")
        disc = (gx ** 2 + gy ** 2) <= cells ** 2
        solid = binary_dilation(solid, structure=disc)
    return grid.free_mask() & ~solid


def _graph(passable: np.ndarray, cell_m: float):
    """The passable cells as a sparse graph, ready for Dijkstra.

    Diagonal steps require BOTH orthogonal neighbours to be free, so a path
    cannot slip through the corner where two obstacles touch. That gap is
    zero centimetres wide in reality and the robot wedges in it.
    """
    index = np.full(passable.shape, -1, dtype=np.int64)
    count = int(passable.sum())
    index[passable] = np.arange(count)
    rows, cols, costs = [], [], []

    def link(dx: int, dy: int, cost: float, gate: bool) -> None:
        src = index[max(0, -dx):index.shape[0] - max(0, dx),
                    max(0, -dy):index.shape[1] - max(0, dy)]
        dst = index[max(0, dx):index.shape[0] - max(0, -dx),
                    max(0, dy):index.shape[1] - max(0, -dy)]
        good = (src >= 0) & (dst >= 0)
        if gate:
            side_a = index[max(0, dx):index.shape[0] - max(0, -dx),
                           max(0, -dy):index.shape[1] - max(0, dy)]
            side_b = index[max(0, -dx):index.shape[0] - max(0, dx),
                           max(0, dy):index.shape[1] - max(0, -dy)]
            good &= (side_a >= 0) & (side_b >= 0)
        rows.append(src[good])
        cols.append(dst[good])
        costs.append(np.full(int(good.sum()), cost))

    for dx, dy in _STRAIGHT:
        link(dx, dy, cell_m, gate=False)
    if config.NAV_ALLOW_DIAGONAL:
        for dx, dy in _DIAGONAL:
            link(dx, dy, cell_m * math.sqrt(2.0), gate=True)

    if not rows or not sum(len(r) for r in rows):
        return index, count, None
    graph = coo_matrix((np.concatenate(costs),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(count, count)).tocsr()
    return index, count, graph


class DistanceField:
    """Path distance from one pose to every reachable cell, plus the routes.

    Built once per planning cycle. `distance_to` and `path_to` are then
    lookups, which is what lets frontier selection compare a dozen candidate
    destinations by true driving distance instead of by straight lines
    through walls.
    """

    def __init__(self, grid: OccupancyGrid, start_xy: tuple[float, float],
                 extra_blocked: np.ndarray | None = None) -> None:
        self.grid = grid
        self.passable = passable_mask(grid, extra_blocked)
        self.start_xy = tuple(float(v) for v in start_xy)
        self._index, count, graph = _graph(self.passable, grid.cell)
        self.reachable = 0
        self._dist = None
        self._prev = None
        self._start_cell = self._nearest_passable(start_xy)
        if graph is None or self._start_cell is None:
            return
        source = int(self._index[self._start_cell])
        self._dist, self._prev = dijkstra(graph, directed=False,
                                          indices=source,
                                          return_predecessors=True)
        self.reachable = int(np.isfinite(self._dist).sum())

    def _nearest_passable(self, xy) -> tuple[int, int] | None:
        """The robot is not always standing somewhere the map calls passable.

        Inflation puts the robot 'inside' a wall whenever it is parked close
        to one, and the pose is worth a few centimetres anyway. Refusing to
        plan in that situation would strand Friday against every wall she ever
        approached, so the search starts from the nearest legal cell instead
        -- within a metre, or not at all.
        """
        if not self.passable.any():
            return None
        ix, iy = self.grid.cells_of(np.array([xy]))
        cell = (int(ix[0]), int(iy[0]))
        if (0 <= cell[0] < self.passable.shape[0]
                and 0 <= cell[1] < self.passable.shape[1]
                and self.passable[cell]):
            return cell
        legal = np.argwhere(self.passable)
        away = np.hypot(legal[:, 0] - ix[0], legal[:, 1] - iy[0])
        best = int(np.argmin(away))
        if away[best] * self.grid.cell > 1.0:
            return None
        return (int(legal[best, 0]), int(legal[best, 1]))

    @property
    def ok(self) -> bool:
        return self._dist is not None

    def distance_map(self) -> np.ndarray:
        """Per-cell driving distance in metres; inf where unreachable."""
        out = np.full(self.grid.shape, np.inf)
        if self._dist is None:
            return out
        out[self.passable] = self._dist
        return out

    def distance_to(self, xy) -> float:
        if self._dist is None:
            return float("inf")
        ix, iy = self.grid.cells_of(np.atleast_2d(xy))
        if not (0 <= ix[0] < self.passable.shape[0]
                and 0 <= iy[0] < self.passable.shape[1]):
            return float("inf")
        node = self._index[ix[0], iy[0]]
        return float(self._dist[node]) if node >= 0 else float("inf")

    def path_to(self, xy, simplify: bool = True) -> np.ndarray:
        """World waypoints from the robot to xy, or an empty array.

        Simplified by line of sight rather than returned cell by cell: the
        raw route is a staircase of 5 cm steps, and a differential drive
        asked to follow that turns four hundred times to cross a room.
        """
        if self._dist is None:
            return np.empty((0, 2))
        ix, iy = self.grid.cells_of(np.atleast_2d(xy))
        if not (0 <= ix[0] < self.passable.shape[0]
                and 0 <= iy[0] < self.passable.shape[1]):
            return np.empty((0, 2))
        node = self._index[ix[0], iy[0]]
        if node < 0 or not np.isfinite(self._dist[node]):
            return np.empty((0, 2))

        chain = [int(node)]
        while self._prev[chain[-1]] >= 0:
            chain.append(int(self._prev[chain[-1]]))
        chain.reverse()

        cells = np.argwhere(self.passable)
        route = cells[chain]
        world = (route + 0.5) * self.grid.cell + self.grid.origin
        return self._simplify(world) if simplify else world

    def _simplify(self, world: np.ndarray) -> np.ndarray:
        """Drop every waypoint that can be skipped without clipping anything."""
        if len(world) <= 2:
            return world
        kept = [0]
        i = 0
        while i < len(world) - 1:
            far = i + 1
            for j in range(len(world) - 1, i, -1):
                if self.clear_between(world[i], world[j]):
                    far = j
                    break
            kept.append(far)
            i = far
        return world[kept]

    def clear_between(self, a, b) -> bool:
        """Is the straight line from a to b entirely on passable cells?

        Sampled at half a cell, which cannot skip a cell diagonally at this
        step size and costs one vectorised index.
        """
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(int(span / (self.grid.cell * 0.5)) + 1, 2)
        t = np.linspace(0.0, 1.0, steps)
        xs = a[0] + (b[0] - a[0]) * t
        ys = a[1] + (b[1] - a[1]) * t
        ix, iy = self.grid.cells_of(np.column_stack([xs, ys]))
        if (ix < 0).any() or (iy < 0).any():
            return False
        if (ix >= self.passable.shape[0]).any() or (iy >= self.passable.shape[1]).any():
            return False
        return bool(self.passable[ix, iy].all())


def path_to_moves(path: np.ndarray, heading_rad: float
                  ) -> list[tuple[float, float]]:
    """Waypoints -> [(turn degrees, drive metres), ...] for a differential drive.

    Turn then drive, one pair per leg, because that is the only motion this
    chassis has: it cannot arc. Turns are wrapped to +/-180 so it never takes
    the long way round.
    """
    moves = []
    theta = heading_rad
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        want = math.atan2(dy, dx)
        turn = (want - theta + math.pi) % (2 * math.pi) - math.pi
        moves.append((math.degrees(turn), math.hypot(dx, dy)))
        theta = want
    return moves
