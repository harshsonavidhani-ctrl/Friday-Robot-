"""Synthetic occupancy maps for the File 21 tests (spec step 4).

    m = closed_room()          # m.grid (5 cm Grid), m.pose, m.note
    og = to_occupancy_grid(m.grid)     # a real slam OccupancyGrid (log-odds)

Five shapes the spec names, plus a big house for timing:

    closed_room         nothing left to explore
    three_rooms         a row of three rooms joined by 0.8 m doors; each room
                        can be explored or still unknown
    open_door           a room whose door opens on an unknown corridor
    gap_frontier        unknown floor seen through a 0.1 m slit: a frontier
                        she cannot reach
    speckle             a closed room peppered the way a LiDAR map is: single
                        unknown cells, single wall specks, small unknown pockets
    big_house           20 x 20 m, many rooms, for the Pi timing budget

Everything is [ix, iy] = [x, y] at 5 cm, origin (0, 0), like slam/grid.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from exploration_mission import grid_utils as gu

RES = 0.05
WALL = 0.10          # wall thickness
DOOR = 0.80


@dataclass
class Map:
    grid: gu.Grid
    pose: tuple[float, float, float]
    note: str = ""


def _c(m: float) -> int:
    return int(round(m / RES))


def blank(w_m: float, h_m: float, fill: int = gu.UNKNOWN) -> gu.Grid:
    return gu.Grid(np.full((_c(w_m), _c(h_m)), fill, np.int8), RES, (0.0, 0.0))


def rect(g: gu.Grid, x0: float, y0: float, x1: float, y1: float, value: int) -> None:
    g.occ[_c(x0):_c(x1), _c(y0):_c(y1)] = value


def room(g: gu.Grid, x0: float, y0: float, x1: float, y1: float,
         explored: bool = True) -> None:
    """Walls on the rectangle's border (inside it), floor within. An
    unexplored room is left as it was (unknown): nobody has looked."""
    if not explored:
        return
    rect(g, x0, y0, x1, y1, gu.OCC)
    rect(g, x0 + WALL, y0 + WALL, x1 - WALL, y1 - WALL, gu.FREE)


def door_v(g: gu.Grid, x: float, y_mid: float, width: float = DOOR) -> None:
    """A doorway through a vertical wall centred on x."""
    rect(g, x - WALL, y_mid - width / 2, x + WALL, y_mid + width / 2, gu.FREE)


def closed_room() -> Map:
    g = blank(6, 5)
    room(g, 1, 1, 5, 4)
    return Map(g, (3.0, 2.5, 0.0), "closed: no frontier")


def three_rooms(explored=(True, True, True)) -> Map:
    """A | B | C, each 3.5 x 4 m, doors (0.8 m) at y = 3 in the shared walls.
    The door INTO an unexplored room is open floor as far as the wall line."""
    g = blank(12.5, 6)
    xs = [(1.0, 4.5), (4.5, 8.0), (8.0, 11.5)]
    for (x0, x1), seen in zip(xs, explored):
        room(g, x0, 1, x1, 5, seen)
    for i, x in enumerate((4.5, 8.0)):
        if explored[i] and explored[i + 1]:
            door_v(g, x, 3.0)
        elif explored[i]:                       # seen from the left room only
            rect(g, x - WALL, 3.0 - DOOR / 2, x, 3.0 + DOOR / 2, gu.FREE)
        elif explored[i + 1]:
            rect(g, x, 3.0 - DOOR / 2, x + WALL, 3.0 + DOOR / 2, gu.FREE)
    return Map(g, (2.5, 3.0, 0.0), f"three rooms, explored={explored}")


def open_door() -> Map:
    g = blank(8, 5)
    room(g, 1, 1, 5, 4)
    rect(g, 5 - WALL, 2.5 - DOOR / 2, 5.0, 2.5 + DOOR / 2, gu.FREE)   # the doorway
    return Map(g, (2.0, 2.5, 0.0), "door on the east wall opens on unknown")


def gap_frontier() -> Map:
    g = blank(8, 5)
    room(g, 1, 1, 5, 4)
    rect(g, 5 - WALL, 2.45, 5.0, 2.55, gu.FREE)       # a 0.1 m slit in the wall
    rect(g, 5.0, 2.0, 6.0, 3.0, gu.FREE)              # floor seen through it
    return Map(g, (2.0, 2.5, 0.0), "frontier beyond a 0.1 m gap: unreachable")


def speckle(seed: int = 5) -> Map:
    m = closed_room()
    g = m.grid
    rng = np.random.default_rng(seed)
    free = np.argwhere(g.free())
    # single unknown cells (beam gaps): 4% of the floor
    for ix, iy in free[rng.choice(len(free), size=len(free) // 25, replace=False)]:
        g.occ[ix, iy] = gu.UNKNOWN
    # isolated single wall specks: 12 of them
    for ix, iy in free[rng.choice(len(free), size=12, replace=False)]:
        g.occ[ix, iy] = gu.OCC
    # small unknown pockets, 0.2 x 0.2 m and 0.3 x 0.2 m
    rect(g, 2.0, 2.0, 2.2, 2.2, gu.UNKNOWN)
    rect(g, 3.6, 1.6, 3.9, 1.8, gu.UNKNOWN)
    return Map(g, (3.0, 2.5, 0.0), "speckle: nothing real to explore")


def big_house(size_m: float = 20.0, rooms: int = 4) -> Map:
    """size x size metres, rooms x rooms grid of rooms, every shared wall with
    a door; the top row unexplored, so there are frontiers at several doors."""
    g = blank(size_m + 2, size_m + 2)
    step = size_m / rooms
    for i in range(rooms):
        for j in range(rooms):
            seen = j < rooms - 1
            room(g, 1 + i * step, 1 + j * step, 1 + (i + 1) * step,
                 1 + (j + 1) * step, seen)
    for i in range(rooms):
        for j in range(rooms - 1):
            if i < rooms - 1:                                  # door to the right
                door_v(g, 1 + (i + 1) * step, 1 + (j + 0.5) * step)
            # door upward (through a horizontal wall)
            xm, y = 1 + (i + 0.5) * step, 1 + (j + 1) * step
            if j < rooms - 2:
                rect(g, xm - DOOR / 2, y - WALL, xm + DOOR / 2, y + WALL, gu.FREE)
            else:                                              # into the unknown row
                rect(g, xm - DOOR / 2, y - WALL, xm + DOOR / 2, y, gu.FREE)
    return Map(g, (1 + step / 2, 1 + step / 2, 0.0), f"{size_m:.0f} m house")


def to_occupancy_grid(g: gu.Grid):
    """The same map as a slam OccupancyGrid (log-odds +-3), so File 14's own
    despeckle and passable_mask can be run on it."""
    from slam.grid import OccupancyGrid
    og = OccupancyGrid(cell_m=g.resolution)
    lo = np.zeros(g.shape, np.float32)
    lo[g.occ == gu.FREE] = -3.0
    lo[g.occ == gu.OCC] = 3.0
    og.logodds = lo
    og.origin = np.array(g.origin, dtype=np.float64)
    og.scans = 1
    og._empty = False
    return og
