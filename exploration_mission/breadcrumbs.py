"""The trail she drove, and the way back along it (File 21 step 3, spec 4.10).

While a mission runs and SLAM is tracking, `Trail.update(x, y, t)` drops a
crumb every `spacing_m` of straight-line distance from the last one. Driving
round in a loop and coming back near an older crumb CUTS the loop -- the
crumbs in between go, so the way back never retraces a detour.

`backtrack_path` is rung 5 of the return ladder (spec 4.11): the planner found
no way home, but the trail is ground she has already driven. From the crumb
nearest her it walks the trail back to the start, drops crumbs that are now
inside an obstacle (the map's walls and File 14's runtime obstacles, inflated
by File 14's own inflation), and skips crumbs she can see past in a straight
line on the traversable grid (Bresenham). The result is a short list of
waypoints for File 14's follower.

Pure numpy; the grid comes in as a grid_utils.Grid.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from exploration_mission import grid_utils as gu

# A loop is only a loop if the older crumb is this many crumbs back: the crumb
# just laid is always within ~spacing of the previous one, and 0.25 m spacing
# sits inside the 0.30 m loop-cut radius.
LOOP_MIN_GAP = 3


class Trail:
    def __init__(self, spacing_m: float = 0.25, loop_cut_m: float = 0.30,
                 crumbs: Iterable | None = None):
        self.spacing_m = float(spacing_m)
        self.loop_cut_m = float(loop_cut_m)
        self.crumbs: list[tuple[float, float, float]] = [
            (float(c[0]), float(c[1]), float(c[2])) for c in (crumbs or [])]
        self.cuts = 0

    def __len__(self) -> int:
        return len(self.crumbs)

    def update(self, x: float, y: float, t: float) -> bool:
        """Offer the current tracked pose. True when a crumb was laid."""
        if not self.crumbs:
            self.crumbs.append((float(x), float(y), float(t)))
            return True
        lx, ly, _ = self.crumbs[-1]
        if math.hypot(x - lx, y - ly) < self.spacing_m:
            return False
        older = self.crumbs[:-LOOP_MIN_GAP] if len(self.crumbs) > LOOP_MIN_GAP else []
        if older:
            pts = np.asarray([(c[0], c[1]) for c in older])
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            near = np.nonzero(d <= self.loop_cut_m)[0]
            if near.size:
                keep = int(near[0])          # the OLDEST crumb it came back to
                del self.crumbs[keep + 1:]
                self.cuts += 1
        self.crumbs.append((float(x), float(y), float(t)))
        return True

    def tail(self, keep: int) -> list[list[float]]:
        """The last `keep` crumbs, for the checkpoint."""
        return [list(c) for c in self.crumbs[-int(keep):]]

    def to_list(self) -> list[list[float]]:
        return [list(c) for c in self.crumbs]


# --- the way back along it ------------------------------------------------------

def bresenham(ix0: int, iy0: int, ix1: int, iy1: int) -> list[tuple[int, int]]:
    """Every cell on the line from (ix0, iy0) to (ix1, iy1), both included."""
    cells = []
    dx, dy = abs(ix1 - ix0), -abs(iy1 - iy0)
    sx, sy = (1 if ix0 < ix1 else -1), (1 if iy0 < iy1 else -1)
    err = dx + dy
    x, y = ix0, iy0
    while True:
        cells.append((x, y))
        if x == ix1 and y == iy1:
            return cells
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def blocked_mask(grid: gu.Grid, inflate_m: float,
                 extra_points: Iterable | None = None) -> np.ndarray:
    """Where the robot's centre may NOT be: OCC (map + runtime obstacles)
    dilated by File 14's inflation. Unknown is not "blocked" here -- a crumb
    in unmapped floor was still driven."""
    occ = grid.occupied().copy()
    pts = None if extra_points is None else np.asarray(list(extra_points), float)
    if pts is not None and pts.size:
        ix, iy = grid.world_to_cell(pts[:, 0], pts[:, 1])
        inside = grid.inside(ix, iy)
        occ[ix[inside], iy[inside]] = True
    solid = gu.Grid(np.where(occ, gu.OCC, gu.FREE).astype(np.int8),
                    grid.resolution, grid.origin)
    return ~gu.traversable(solid, inflate_m)


def line_of_sight(grid: gu.Grid, traversable: np.ndarray, a, b) -> bool:
    """Every cell on the straight line a -> b is traversable (FREE and clear
    of inflated obstacles). Leaving the map is not line of sight."""
    ax, ay = grid.world_to_cell(a[0], a[1])
    bx, by = grid.world_to_cell(b[0], b[1])
    for cx, cy in bresenham(ax, ay, bx, by):
        if not grid.inside(cx, cy) or not traversable[cx, cy]:
            return False
    return True


def backtrack_path(crumbs: list, pose, grid: gu.Grid, inflate_m: float,
                   runtime_obstacles: Iterable | None = None,
                   body_radius_m: float | None = None
                   ) -> list[tuple[float, float]]:
    """Waypoints from the crumb nearest `pose` back to the first crumb (the
    start), blocked crumbs dropped, straight-line-visible ones skipped.

    A crumb is DROPPED only when she could not stand on it now: within her
    body radius (`body_radius_m`, File 14's NAV_ROBOT_RADIUS_M) of an obstacle
    -- "inside an obstacle", spec 4.10. Not for being inside the clearance
    margin: a crumb she drove through a tight doorway is the most valuable one
    on the trail. SKIPPING a crumb needs a line of sight on the fully inflated
    traversable grid (`inflate_m`), so a shortcut is only ever one the planner
    would also take. Consecutive crumbs are always allowed: she drove that leg,
    and the mission's map may simply not cover it. The start is always kept
    (rung 6 relaxes the goal if the start itself is blocked).
    """
    if not crumbs:
        return []
    pts = [(float(c[0]), float(c[1])) for c in crumbs]
    arr = np.asarray(pts)
    near = int(np.argmin(np.hypot(arr[:, 0] - pose[0], arr[:, 1] - pose[1])))
    back = pts[near::-1]                              # nearest -> ... -> start
    body = inflate_m if body_radius_m is None else body_radius_m
    occupied_by_body = blocked_mask(grid, body, runtime_obstacles)

    def is_blocked(p) -> bool:
        ix, iy = grid.world_to_cell(p[0], p[1])
        return bool(grid.inside(ix, iy) and occupied_by_body[ix, iy])

    start = back[-1]
    kept = [p for p in back[:-1] if not is_blocked(p)] + [start]
    trav = ~blocked_mask(grid, inflate_m, runtime_obstacles) & grid.free()
    out = [kept[0]]
    i = 0
    while i < len(kept) - 1:
        j = i + 1
        while j + 1 < len(kept) and line_of_sight(grid, trav, kept[i], kept[j + 1]):
            j += 1
        out.append(kept[j])
        i = j
    return out
