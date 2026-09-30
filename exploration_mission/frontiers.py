"""Frontiers: where known floor meets the unknown (File 21 step 4, spec 4.3/4.4).

    result = find(fine_grid, pose, inflate_m, params, blacklist, crumbs)
    result.remaining      # reachable, not blacklisted, best score first
    done = completion.update(len(result.remaining), localized=True)

WHAT THE MISSION USES THEM FOR. "Explored" means no reachable frontier is left
(Yamauchi). Here that answers three questions -- is the mission finished, is
this area complete (step 5), and what to say for status. They do NOT steer:
the RL run's own frontier planner does the driving (rl/explorer.py), and a
second one would be the second planner hard rule 3 forbids.

HOW (all on the 10 cm coarse grid, OpenCV only, no per-cell Python loops):
  1. traversable = File 14's own rule on the FINE grid (FREE, not within
     NAV_INFLATE_M of a wall), then a coarse cell is traversable if ANY of its
     fine cells is -- so a doorway the planner can use never closes just
     because it was coarsened.
  2. frontier cell = FREE cell with an UNKNOWN 8-neighbour: free & dilate(unknown).
  3. clusters = 8-connected components; shorter than min_frontier_size_m go.
  4. ONE geodesic distance field from the robot by masked wavefront
     (new = dilate(reached) & traversable & ~reached, the pass number is the
     distance). It gives reachability and path distance for every cluster.
  5. score = gain*length - potential*path_dist + heading*cos(dtheta)
             - revisit*revisits            (explore_lite: near space first)

⚠ ONE RULE THE SPEC DOES NOT HAVE: an unknown pocket smaller than
`min_unknown_m2` makes no frontier. At 6 m the LiDAR's beams are 10 cm apart,
so far floor is left peppered with cells nobody's beam landed in, and the
shadow behind a chair leg is a pocket she can never drive into. Each of those
would be a frontier for ever, and "no frontiers left" would never come. A
pocket she could not fit inside is not somewhere to explore. Unknown that runs
off the edge of the map is never a pocket.

Distances are in 8-connected steps x the cell size (the spec's wavefront), so
a diagonal run reads ~29% short. It is a ranking number, not odometry.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from exploration_mission import grid_utils as gu

_K3 = np.ones((3, 3), np.uint8)
# How far from a non-traversable robot cell a traversable one may be and still
# be "where she is" (she is often inside the inflation margin of a wall).
START_SNAP_M = 0.5
REVISIT_RADIUS_M = 0.5


@dataclass
class Params:
    seg_resolution_m: float = 0.10
    min_frontier_size_m: float = 0.30
    min_unknown_m2: float = 0.15
    gain_scale: float = 1.0
    potential_scale: float = 3.0
    heading_scale: float = 0.5
    revisit_scale: float = 1.0

    @classmethod
    def from_config(cls, cfg) -> "Params":
        e = cfg.explore
        return cls(cfg.grid.seg_resolution_m, e.min_frontier_size_m,
                   e.min_unknown_m2, e.gain_scale, e.potential_scale,
                   e.heading_scale, e.revisit_scale)


@dataclass
class Frontier:
    id: int
    cells: np.ndarray                  # (n, 2) coarse [ix, iy]
    length_m: float
    centroid: tuple[float, float]      # world metres
    reachable: bool = False
    goal_cell: tuple[int, int] | None = None
    goal: tuple[float, float] | None = None
    path_dist_m: float = math.inf
    revisits: int = 0
    score: float = -math.inf
    blacklisted: bool = False

    def as_dict(self) -> dict:
        return {"id": self.id, "length_m": round(self.length_m, 2),
                "centroid": [round(v, 2) for v in self.centroid],
                "reachable": self.reachable,
                "goal": None if self.goal is None else [round(v, 2) for v in self.goal],
                "path_dist_m": (None if not math.isfinite(self.path_dist_m)
                                else round(self.path_dist_m, 2)),
                "score": None if not math.isfinite(self.score) else round(self.score, 2),
                "blacklisted": self.blacklisted}


@dataclass
class Result:
    frontiers: list[Frontier]          # every cluster, best score first
    remaining: list[Frontier]          # reachable and not blacklisted
    coarse: gu.Grid
    traversable: np.ndarray            # coarse bool
    dist_cells: np.ndarray             # coarse float32, inf = unreachable
    start_cell: tuple[int, int] | None
    ms: dict = field(default_factory=dict)

    @property
    def best(self) -> Frontier | None:
        return self.remaining[0] if self.remaining else None

    def reachable_length_m(self) -> float:
        """For the curiosity score (spec 4.15): metres of reachable frontier."""
        return float(sum(f.length_m for f in self.remaining))


# --- pieces ---------------------------------------------------------------------

def pool_any(mask: np.ndarray, k: int) -> np.ndarray:
    """Fine bool mask -> coarse: True if ANY of the k x k fine cells is."""
    if k <= 1:
        return mask.copy()
    nx, ny = mask.shape
    padded = np.pad(mask, ((0, (-nx) % k), (0, (-ny) % k)), constant_values=False)
    bx, by = padded.shape[0] // k, padded.shape[1] // k
    return padded.reshape(bx, k, by, k).any(axis=(1, 3))


def coarse_view(fine: gu.Grid, inflate_m: float, seg_resolution_m: float
                ) -> tuple[gu.Grid, np.ndarray]:
    """(coarse grid, coarse traversable). Traversability is decided on the
    FINE grid by File 14's rule, then pooled."""
    k = max(int(round(seg_resolution_m / fine.resolution)), 1)
    return gu.downsample(fine, seg_resolution_m), pool_any(
        gu.traversable(fine, inflate_m), k)


def real_unknown(coarse: gu.Grid, min_unknown_m2: float) -> np.ndarray:
    """UNKNOWN cells that are somewhere to explore: pockets smaller than
    min_unknown_m2 are dropped, unless they touch the map's edge."""
    unknown = coarse.unknown().astype(np.uint8)
    if min_unknown_m2 <= 0 or not unknown.any():
        return unknown.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=8)
    if n <= 1:
        return unknown.astype(bool)
    min_cells = min_unknown_m2 / (coarse.resolution ** 2)
    nx, ny = unknown.shape
    left, top = stats[:, cv2.CC_STAT_LEFT], stats[:, cv2.CC_STAT_TOP]
    w, h = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]
    # OpenCV's x is our axis 1 and its y our axis 0; "touches an edge" is
    # symmetric, so only the pairing with the right bound matters.
    at_edge = (left == 0) | (top == 0) | (left + w == ny) | (top + h == nx)
    keep = (stats[:, cv2.CC_STAT_AREA] >= min_cells) | at_edge
    keep[0] = False                                   # label 0 = not unknown
    return keep[labels]


def frontier_mask(coarse: gu.Grid, min_unknown_m2: float = 0.0) -> np.ndarray:
    unknown = real_unknown(coarse, min_unknown_m2).astype(np.uint8)
    near_unknown = cv2.dilate(unknown, _K3, borderType=cv2.BORDER_CONSTANT,
                              borderValue=0)
    return coarse.free() & (near_unknown > 0)


def snap_start(traversable: np.ndarray, cell: tuple[int, int], resolution: float
               ) -> tuple[int, int] | None:
    """The robot's cell, or the nearest traversable one within START_SNAP_M."""
    nx, ny = traversable.shape
    ix, iy = cell
    if 0 <= ix < nx and 0 <= iy < ny and traversable[ix, iy]:
        return ix, iy
    r = int(math.ceil(START_SNAP_M / resolution))
    x0, x1 = max(ix - r, 0), min(ix + r + 1, nx)
    y0, y1 = max(iy - r, 0), min(iy + r + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return None
    window = np.argwhere(traversable[x0:x1, y0:y1])
    if not len(window):
        return None
    d = np.hypot(window[:, 0] + x0 - ix, window[:, 1] + y0 - iy)
    best = int(np.argmin(d))
    if d[best] * resolution > START_SNAP_M:
        return None
    return int(window[best, 0] + x0), int(window[best, 1] + y0)


def distance_field(traversable: np.ndarray, start: tuple[int, int] | None
                   ) -> np.ndarray:
    """8-connected geodesic distance in CELLS from `start` over `traversable`
    (float32, inf where unreachable). Masked wavefront, spec 4.3:

        new = dilate(reached, 3x3) & traversable & ~reached     pass p
        distance[new] = p

    ONLY THE FRONT IS DILATED, inside its own bounding box. New cells can only
    appear next to the cells added on the previous pass, so growing the whole
    `reached` image every pass (a 3x3 dilate of the full map: ~95 us on this
    Pi at 220 x 220, times one pass per cell of path) is mostly work on cells
    that finished long ago. In a house the front is a small patch; the window
    is that patch plus one cell.
    """
    dist = np.full(traversable.shape, np.inf, dtype=np.float32)
    if start is None or not traversable[start]:
        return dist
    avail = np.array(traversable, dtype=np.uint8)      # traversable, not yet reached
    nx, ny = avail.shape
    dist[start] = 0.0
    avail[start] = 0
    front = np.ones((1, 1), np.uint8)
    r0, c0 = int(start[0]), int(start[1])               # the front's corner
    p = 0
    while True:
        fh, fw = front.shape
        R0, C0 = max(r0 - 1, 0), max(c0 - 1, 0)
        R1, C1 = min(r0 + fh + 1, nx), min(c0 + fw + 1, ny)
        window = np.zeros((R1 - R0, C1 - C0), np.uint8)
        window[r0 - R0:r0 - R0 + fh, c0 - C0:c0 - C0 + fw] = front
        grown = cv2.dilate(window, _K3, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
        new = cv2.bitwise_and(grown, np.ascontiguousarray(avail[R0:R1, C0:C1]))
        if cv2.countNonZero(new) == 0:
            return dist
        p += 1
        bx, by, bw, bh = cv2.boundingRect(new)          # x = axis 1, y = axis 0
        front = np.ascontiguousarray(new[by:by + bh, bx:bx + bw])
        r0, c0 = R0 + by, C0 + bx
        sel = front.astype(bool)
        dist[r0:r0 + bh, c0:c0 + bw][sel] = p
        avail[r0:r0 + bh, c0:c0 + bw][sel] = 0


def count_revisits(goal: tuple[float, float], crumbs: Iterable | None,
                   radius_m: float = REVISIT_RADIUS_M) -> int:
    """How many times the trail PASSED within radius of the goal: entries into
    the disc, not crumbs inside it."""
    if not crumbs:
        return 0
    pts = np.asarray([(c[0], c[1]) for c in crumbs], dtype=np.float64)
    if pts.size == 0:
        return 0
    inside = np.hypot(pts[:, 0] - goal[0], pts[:, 1] - goal[1]) <= radius_m
    return int(inside[0]) + int(np.count_nonzero(inside[1:] & ~inside[:-1]))


# --- blacklist + completion (spec 4.4) -----------------------------------------

class Blacklist:
    """A frontier whose goal failed `after_fails` times is out for this
    mission: a disc of radius_m round its centroid. Lives in the checkpoint."""

    def __init__(self, after_fails: int = 2, radius_m: float = 0.5,
                 entries: list | None = None):
        self.after_fails = int(after_fails)
        self.radius_m = float(radius_m)
        self.entries: list[dict] = [dict(e) for e in (entries or [])]

    def _near(self, xy) -> dict | None:
        for e in self.entries:
            if math.hypot(e["x"] - xy[0], e["y"] - xy[1]) <= self.radius_m:
                return e
        return None

    def fail(self, centroid) -> bool:
        """Count one failure. True once it is blacklisted."""
        e = self._near(centroid)
        if e is None:
            e = {"x": float(centroid[0]), "y": float(centroid[1]), "fails": 0}
            self.entries.append(e)
        e["fails"] += 1
        return e["fails"] >= self.after_fails

    def blocks(self, centroid) -> bool:
        e = self._near(centroid)
        return e is not None and e["fails"] >= self.after_fails

    def to_list(self) -> list[dict]:
        return [dict(e) for e in self.entries]


class Completion:
    """`done_after` CONSECUTIVE frontier ticks with nothing reachable and not
    blacklisted, while localized. A tick while lost proves nothing and starts
    the count again."""

    def __init__(self, done_after: int = 3):
        self.done_after = int(done_after)
        self.empty = 0

    def update(self, remaining: int, localized: bool) -> bool:
        if not localized or remaining > 0:
            self.empty = 0
            return False
        self.empty += 1
        return self.empty >= self.done_after


# --- the whole thing --------------------------------------------------------------

def find(fine: gu.Grid, pose, inflate_m: float, params: Params | None = None,
         blacklist: Blacklist | None = None, crumbs: Iterable | None = None
         ) -> Result:
    """Every frontier cluster, scored; `remaining` is what is left to explore."""
    p = params or Params()
    t0 = time.perf_counter()
    coarse, trav = coarse_view(fine, inflate_m, p.seg_resolution_m)
    t1 = time.perf_counter()
    fmask = frontier_mask(coarse, p.min_unknown_m2).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(fmask, connectivity=8)
    t2 = time.perf_counter()
    start = snap_start(trav, coarse.world_to_cell(pose[0], pose[1]),
                       coarse.resolution)
    dist = distance_field(trav, start)
    t3 = time.perf_counter()

    res = coarse.resolution
    min_cells = p.min_frontier_size_m / res
    reached = np.isfinite(dist)
    out: list[Frontier] = []
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < min_cells - 1e-9:
            continue
        cells = np.argwhere(labels == lab)
        # OpenCV centroids are (axis1, axis0) = (iy, ix) in our order.
        cix, ciy = float(cents[lab][1]), float(cents[lab][0])
        cx, cy = coarse.cell_to_world(cix, ciy)
        f = Frontier(id=lab, cells=cells, length_m=area * res, centroid=(cx, cy))
        # Reachable if any cell, grown by one coarse cell, was reached.
        x0, y0 = cells.min(axis=0)
        x1, y1 = cells.max(axis=0)
        x0, y0 = max(x0 - 1, 0), max(y0 - 1, 0)
        x1, y1 = min(x1 + 2, labels.shape[0]), min(y1 + 2, labels.shape[1])
        near = cv2.dilate((labels[x0:x1, y0:y1] == lab).astype(np.uint8), _K3)
        cand = np.argwhere((near > 0) & reached[x0:x1, y0:y1])
        if len(cand):
            cand = cand + np.array([x0, y0])
            d = np.hypot(cand[:, 0] - cix, cand[:, 1] - ciy)
            gix, giy = (int(v) for v in cand[int(np.argmin(d))])
            f.reachable = True
            f.goal_cell = (gix, giy)
            f.goal = coarse.cell_to_world(gix, giy)
            f.path_dist_m = float(dist[gix, giy]) * res
            f.revisits = count_revisits(f.goal, crumbs)
            bearing = math.atan2(f.goal[1] - pose[1], f.goal[0] - pose[0])
            f.score = (p.gain_scale * f.length_m
                       - p.potential_scale * f.path_dist_m
                       + p.heading_scale * math.cos(bearing - pose[2])
                       - p.revisit_scale * f.revisits)
        f.blacklisted = bool(blacklist is not None and blacklist.blocks(f.centroid))
        out.append(f)
    # Highest score first; a tie goes to the nearer one. Unreachable last.
    out.sort(key=lambda f: (-f.score if math.isfinite(f.score) else math.inf,
                            f.path_dist_m))
    remaining = [f for f in out if f.reachable and not f.blacklisted]
    t4 = time.perf_counter()
    return Result(out, remaining, coarse, trav, dist, start,
                  {"coarse": (t1 - t0) * 1e3, "detect": (t2 - t1) * 1e3,
                   "distance": (t3 - t2) * 1e3, "score": (t4 - t3) * 1e3,
                   "total": (t4 - t0) * 1e3})
