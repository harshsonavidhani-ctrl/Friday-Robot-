"""Rooms from the map, and when to ask what one is called (File 21 step 5).

    seg = segment(coarse, params)               # spec 4.5, morphological
    mark_complete(seg, frontier_result)         # no reachable frontier inside
    tracker.assign(seg)                         # the same room keeps its key
    match_areas(seg, saved_areas, params)       # is it a place she has saved
    choice = choose_label_target(...)           # spec 4.6: ask about which one

HOW A ROOM IS FOUND (all on the 10 cm grid, unknown counts as wall):
  1. erode the free floor by door_half_width_m: doorways up to 0.8 m close,
     and each room is left as its own blob (a seed);
  2. seeds smaller than min_seed_m2 are dropped;
  3. the seeds grow back into the free floor one ring at a time -- a
     multi-label wavefront where a contested cell goes to the HIGHER label,
     so the same map always gives the same rooms;
  4. a room smaller than min_room_m2 is merged into the neighbour it shares
     the longest border with;
  5. each room gets its area, an outline (<= 40 points, in world metres), a
     centre that is always INSIDE it (the point furthest from its walls, not
     the centroid, which falls outside an L-shaped room) and a radius.

KNOWN LIMIT (spec 4.5): an opening wider than 0.8 m does not close, so an
open-plan hall and dining area is one area. Harsh splits it by saying "save
this as dining room" where he stands.

IDENTITY IS BY OVERLAP, IN WORLD METRES. The map's array grows and its origin
moves, so a room is never compared by array index: a saved area's polygon (or
last tick's) is drawn onto the CURRENT grid and compared by IoU. >= 0.5 is
the same place.

The trigger (spec 4.6) is a pure function of what is handed to it -- no
clock, no speech, no stores -- so every condition is tested on its own.
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
GROW_PASS_CAP = 200                       # spec 4.5 step 4


@dataclass
class Params:
    door_half_width_m: float = 0.40
    min_seed_m2: float = 0.5
    min_room_m2: float = 1.5
    polygon_epsilon_m: float = 0.10
    max_polygon_vertices: int = 40
    same_area_iou: float = 0.5
    grow_update_ratio: float = 1.2
    complete_margin_m: float = 0.2

    @classmethod
    def from_config(cls, cfg) -> "Params":
        s = cfg.segmentation
        return cls(s.door_half_width_m, s.min_seed_m2, s.min_room_m2,
                   s.polygon_epsilon_m, s.max_polygon_vertices, s.same_area_iou,
                   s.grow_update_ratio)


@dataclass
class Segment:
    label: int                              # this tick's label in `labels`
    mask: np.ndarray                        # coarse bool
    area_m2: float
    polygon: list[list[float]]              # world metres, <= max vertices
    center: tuple[float, float]             # world metres, inside the room
    radius_m: float
    complete: bool = False
    key: int | None = None                  # stable across ticks (Tracker)
    area_id: str | None = None              # the saved area it IS, if any
    area_iou: float = 0.0

    def as_dict(self) -> dict:
        return {"key": self.key, "area_m2": round(self.area_m2, 2),
                "center": [round(v, 2) for v in self.center],
                "radius_m": round(self.radius_m, 2), "complete": self.complete,
                "area_id": self.area_id,
                "polygon": [[round(x, 2), round(y, 2)] for x, y in self.polygon]}


@dataclass
class Segmentation:
    coarse: gu.Grid
    labels: np.ndarray                      # uint16, 0 = no room
    segments: list[Segment]
    ms: dict = field(default_factory=dict)

    def at(self, x: float, y: float) -> Segment | None:
        """The room a world point is in, or None."""
        ix, iy = self.coarse.world_to_cell(x, y)
        if not self.coarse.inside(ix, iy):
            return None
        lab = int(self.labels[ix, iy])
        for s in self.segments:
            if s.label == lab:
                return s
        return None


# =============================================================================
# 4.5 room segmentation
# =============================================================================

def _grow(labels: np.ndarray, free: np.ndarray) -> np.ndarray:
    """Seeds grow into unlabelled FREE cells, one 3x3 ring a pass. cv2.dilate
    on the label image is a max filter, so a cell two seeds reach on the same
    pass takes the higher label: deterministic."""
    labels = labels.copy()
    open_cells = free & (labels == 0)
    for _ in range(GROW_PASS_CAP):
        if not open_cells.any():
            break
        grown = cv2.dilate(labels, _K3, borderType=cv2.BORDER_CONSTANT,
                           borderValue=0)
        take = open_cells & (grown > 0)
        if not take.any():
            break
        labels[take] = grown[take]
        open_cells &= ~take
    return labels


def _borders(labels: np.ndarray) -> dict[tuple[int, int], int]:
    """How many 4-neighbour cell pairs each pair of rooms shares."""
    out: dict[tuple[int, int], int] = {}
    for a, b in ((labels[:-1, :], labels[1:, :]), (labels[:, :-1], labels[:, 1:])):
        touching = (a != b) & (a > 0) & (b > 0)
        if not touching.any():
            continue
        lo = np.minimum(a[touching], b[touching]).astype(np.int64)
        hi = np.maximum(a[touching], b[touching]).astype(np.int64)
        pairs, counts = np.unique(lo * 65536 + hi, return_counts=True)
        for p, c in zip(pairs, counts):
            key = (int(p // 65536), int(p % 65536))
            out[key] = out.get(key, 0) + int(c)
    return out


def _merge_small(labels: np.ndarray, min_cells: float) -> np.ndarray:
    """Rooms under min_room_m2 join the neighbour with the longest shared
    border, smallest first. One with no neighbour stays as it is."""
    labels = labels.copy()
    while True:
        ids, sizes = np.unique(labels[labels > 0], return_counts=True)
        small = sorted((int(s), int(i)) for i, s in zip(ids, sizes) if s < min_cells)
        if not small:
            return labels
        borders = _borders(labels)
        merged = False
        for _, lab in small:
            best, best_len = None, 0
            for (a, b), n in borders.items():
                other = b if a == lab else (a if b == lab else None)
                if other is not None and (n > best_len
                                          or (n == best_len and best is not None
                                              and other > best)):
                    best, best_len = other, n
            if best is not None:
                labels[labels == lab] = best
                merged = True
                break                        # borders changed: recompute
        if not merged:
            return labels


def outline(mask: np.ndarray, coarse: gu.Grid, epsilon_m: float,
            max_vertices: int) -> list[list[float]]:
    """The largest outer contour, simplified until it has <= max_vertices,
    as world metres (cell centres)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    eps = max(epsilon_m / coarse.resolution, 0.1)
    poly = cv2.approxPolyDP(contour, eps, True)
    while len(poly) > max_vertices:
        eps *= 1.5
        poly = cv2.approxPolyDP(contour, eps, True)
    out = []
    for pt in poly.reshape(-1, 2):
        # OpenCV points are (axis 1, axis 0) = (iy, ix) in slam's order.
        x, y = coarse.cell_to_world(int(pt[1]), int(pt[0]))
        out.append([float(x), float(y)])
    return out


def interior_point(mask: np.ndarray, coarse: gu.Grid) -> tuple[float, float]:
    """The cell furthest from the room's own edge: always inside it."""
    padded = np.pad(mask.astype(np.uint8), 1)            # the map edge is a wall
    dt = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    ix, iy = np.unravel_index(int(np.argmax(dt)), dt.shape)
    x, y = coarse.cell_to_world(int(ix), int(iy))
    return float(x), float(y)


def segment(coarse: gu.Grid, params: Params | None = None) -> Segmentation:
    p = params or Params()
    t0 = time.perf_counter()
    res = coarse.resolution
    free = coarse.free()
    r = int(round(p.door_half_width_m / res))
    eroded = cv2.erode(free.astype(np.uint8), gu.disc(r),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    n, seeds, stats, _ = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    labels = np.zeros(free.shape, np.uint16)
    min_seed_cells = p.min_seed_m2 / (res * res)
    nxt = 0
    for lab in range(1, n):
        if stats[lab, cv2.CC_STAT_AREA] >= min_seed_cells - 1e-9:
            nxt += 1
            labels[seeds == lab] = nxt
    t1 = time.perf_counter()
    if nxt:
        labels = _grow(labels, free)
        labels = _merge_small(labels, p.min_room_m2 / (res * res))
    t2 = time.perf_counter()
    segments: list[Segment] = []
    for lab in (int(v) for v in np.unique(labels) if v > 0):
        mask = labels == lab
        area = float(mask.sum()) * res * res
        segments.append(Segment(
            label=lab, mask=mask, area_m2=area,
            polygon=outline(mask, coarse, p.polygon_epsilon_m,
                            p.max_polygon_vertices),
            center=interior_point(mask, coarse),
            radius_m=max(0.5, min(2.0, math.sqrt(area / math.pi)))))
    t3 = time.perf_counter()
    return Segmentation(coarse, labels, segments,
                        {"seeds": (t1 - t0) * 1e3, "grow": (t2 - t1) * 1e3,
                         "describe": (t3 - t2) * 1e3, "total": (t3 - t0) * 1e3})


def mark_complete(seg: Segmentation, frontier_result,
                  margin_m: float = 0.2) -> None:
    """spec 4.5 step 8: a room is complete when no REACHABLE frontier cell
    (blacklisted ones no longer count) lies inside it grown by margin_m."""
    fmask = np.zeros(seg.labels.shape, np.uint8)
    for f in getattr(frontier_result, "remaining", []):
        fmask[f.cells[:, 0], f.cells[:, 1]] = 1
    any_frontier = bool(fmask.any())
    k = gu.disc(max(int(round(margin_m / seg.coarse.resolution)), 0))
    for s in seg.segments:
        if not any_frontier:
            s.complete = True
            continue
        near = cv2.dilate(s.mask.astype(np.uint8), k,
                          borderType=cv2.BORDER_CONSTANT, borderValue=0)
        s.complete = not bool((near & fmask).any())


# =============================================================================
# identity: the same place across ticks, runs and a growing map
# =============================================================================

def rasterize(polygon: Iterable, coarse: gu.Grid) -> np.ndarray:
    """A world-metre polygon drawn onto the current coarse grid (bool)."""
    mask = np.zeros(coarse.shape, np.uint8)
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    if len(pts) < 3:
        return mask.astype(bool)
    arr = np.asarray(pts)
    ix = np.floor((arr[:, 0] - coarse.origin[0]) / coarse.resolution)
    iy = np.floor((arr[:, 1] - coarse.origin[1]) / coarse.resolution)
    cv_pts = np.stack([iy, ix], axis=1).astype(np.int32)     # (col, row)
    cv2.fillPoly(mask, [cv_pts], 1)
    return mask.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.count_nonzero(a | b))
    return 0.0 if union == 0 else float(np.count_nonzero(a & b)) / union


# A room seen only in part overlaps its saved area by less than 0.5 (half a
# room is IoU ~0.5 at best), yet it is plainly that room: most of it lies
# INSIDE the saved outline. Not in the spec, which has IoU only; without it
# she asked "what should I call it?" about half of a kitchen she had already
# been told was the kitchen (simulated second mission, 2026-09-30).
INSIDE_FRACTION = 0.70


def inside_fraction(seg_mask: np.ndarray, ref: np.ndarray) -> float:
    n = int(np.count_nonzero(seg_mask))
    return 0.0 if n == 0 else float(np.count_nonzero(seg_mask & ref)) / n


def _best_pairs(segments: list[Segment], refs: list[np.ndarray],
                min_iou: float, inside: float | None = None
                ) -> list[tuple[int, int, float]]:
    """One-to-one (segment index, ref index, iou), best overlap first. With
    `inside`, a segment lying at least that fraction within a ref also pairs
    with it (several segments may then share one ref: parts of one room)."""
    scored = []
    for i, s in enumerate(segments):
        for j, ref in enumerate(refs):
            v = iou(s.mask, ref)
            if v >= min_iou:
                scored.append((v, i, j))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    used_s, used_r, out = set(), set(), []
    for v, i, j in scored:
        if i in used_s or j in used_r:
            continue
        used_s.add(i)
        used_r.add(j)
        out.append((i, j, v))
    if inside is not None:
        for i, s in enumerate(segments):
            if i in used_s:
                continue
            best = max(((inside_fraction(s.mask, ref), j) for j, ref in enumerate(refs)),
                       default=(0.0, -1))
            if best[0] >= inside:
                used_s.add(i)
                out.append((i, best[1], iou(s.mask, refs[best[1]])))
    return out


class Tracker:
    """Gives each room a key that survives the next tick and a grown map."""

    def __init__(self, same_area_iou: float = 0.5):
        self.same_area_iou = same_area_iou
        self._prev: list[tuple[int, list]] = []          # (key, polygon)
        self._next = 1

    def assign(self, seg: Segmentation) -> None:
        refs = [rasterize(poly, seg.coarse) for _, poly in self._prev]
        for s in seg.segments:
            s.key = None
        for i, j, _ in _best_pairs(seg.segments, refs, self.same_area_iou):
            seg.segments[i].key = self._prev[j][0]
        for s in seg.segments:
            if s.key is None:
                s.key = self._next
                self._next += 1
        self._prev = [(s.key, s.polygon) for s in seg.segments]


def match_areas(seg: Segmentation, areas: Iterable[dict],
                same_area_iou: float = 0.5) -> None:
    """Sets segment.area_id for each room that IS a saved area (IoU >= 0.5)."""
    areas = [a for a in areas if a.get("polygon")]
    refs = [rasterize(a["polygon"], seg.coarse) for a in areas]
    for s in seg.segments:
        s.area_id, s.area_iou = None, 0.0
    for i, j, v in _best_pairs(seg.segments, refs, same_area_iou, INSIDE_FRACTION):
        seg.segments[i].area_id = areas[j]["id"]
        seg.segments[i].area_iou = v


def growth_updates(seg: Segmentation, areas: Iterable[dict],
                   grow_update_ratio: float = 1.2, same_area_iou: float = 0.5
                   ) -> list[tuple[str, Segment]]:
    """spec 3.5 growing map: a saved area whose room (same place, IoU >= 0.5,
    already set by match_areas) is now >= 20% bigger -> update it silently."""
    by_id = {a["id"]: a for a in areas}
    out = []
    for s in seg.segments:
        a = by_id.get(s.area_id) if s.area_id else None
        # ONLY A COMPLETE ROOM, and only the same place by IoU. Mid-mission two
        # rooms whose shared wall is not mapped yet are one transient blob; it
        # overlapped a saved kitchen and "grew" it across the next room
        # (simulated second mission, 2026-09-30). A room with no frontier left
        # in it has its walls, and its shape is worth keeping.
        if (a is not None and s.complete and s.area_iou >= same_area_iou
                and s.area_m2 >= grow_update_ratio * float(a["area_m2"])):
            out.append((a["id"], s))
    return out


# =============================================================================
# 4.6 when to ask for a label
# =============================================================================

class Dwell:
    """How long she has been inside each room this mission, and when she left.
    Fed the key of the room she is in (or None) with the time, at loop rate."""

    def __init__(self):
        self.inside_s: dict[int, float] = {}
        self.exited_at: dict[int, float] = {}
        self.current: int | None = None
        self._t: float | None = None

    def update(self, key: int | None, now: float) -> None:
        if self._t is not None and self.current is not None:
            self.inside_s[self.current] = (self.inside_s.get(self.current, 0.0)
                                           + max(now - self._t, 0.0))
        if key != self.current:
            if self.current is not None:
                self.exited_at[self.current] = now
            self.current = key
            if key is not None:
                self.inside_s.setdefault(key, 0.0)
        self._t = now

    def entered(self, key: int) -> bool:
        return key in self.inside_s

    def to_dict(self) -> dict:
        return {"inside_s": {str(k): v for k, v in self.inside_s.items()},
                "exited_at": {str(k): v for k, v in self.exited_at.items()},
                "current": self.current}


@dataclass
class LabelParams:
    trigger: str = "segmentation"           # segmentation | distance | both
    min_area_m2: float = 2.5
    min_dwell_s: float = 15.0
    recent_exit_s: float = 20.0
    cooldown_s: float = 30.0
    distance_prompt_m: float = 4.0

    @classmethod
    def from_config(cls, cfg) -> "LabelParams":
        lb = cfg.label
        return cls(lb.trigger, lb.min_area_m2, lb.min_dwell_s, lb.recent_exit_s,
                   lb.cooldown_s, lb.distance_prompt_m)


@dataclass
class LabelChoice:
    segment: Segment
    rule: str                                # "segmentation" | "distance"
    complete: bool


def why_not(s: Segment, dwell: Dwell, now: float, p: LabelParams,
            final: bool = False) -> list[str]:
    """The per-room conditions of the segmentation trigger that FAIL (empty =
    this room qualifies). Cooldown and state are global, checked by the caller.
    `final` is the one last ask at completion: the room she is standing in
    needs no dwell time then ("so the room Friday is in gets asked", 4.6) --
    a small room seen whole from the doorway is finished in seconds -- and it
    need not be COMPLETE either. The first real mission (2026-10-02) ended on
    its 2-minute cap with frontiers left in the very room she was in, so
    nothing was asked and the room was saved unnamed; a room cut short by the
    clock is asked about (and saved as incomplete), never skipped."""
    out = []
    here_at_the_end = final and dwell.current == s.key
    if s.area_m2 < p.min_area_m2:
        out.append("too small")
    if not s.complete and not here_at_the_end:
        out.append("not complete")
    if s.area_id is not None:
        out.append("already saved")
    inside = dwell.inside_s.get(s.key, 0.0)
    exited = dwell.exited_at.get(s.key)
    recent = (dwell.current != s.key and exited is not None
              and now - exited < p.recent_exit_s)
    if not (inside >= p.min_dwell_s or recent or here_at_the_end):
        out.append("not dwelt in, not just left")
    return out


def choose_label_target(seg: Segmentation, dwell: Dwell, now: float,
                        last_question_t: float | None, exploring: bool,
                        params: LabelParams | None = None,
                        distance_from_saved_m: float | None = None,
                        ignore_cooldown: bool = False) -> LabelChoice | None:
    """spec 4.6. Which room to ask about now, or None.

    `exploring` is "the mission state is EXPLORE" (never while paused, on a
    go-to, or returning). `distance_from_saved_m` is how far she is from the
    nearest saved area (or from the start when nothing is saved yet); only the
    distance trigger reads it. `ignore_cooldown` is the one last ask at
    completion (spec 4.6, last paragraph).
    """
    p = params or LabelParams()
    if not exploring:
        return None
    if (not ignore_cooldown and last_question_t is not None
            and now - last_question_t < p.cooldown_s):
        return None

    if p.trigger in ("segmentation", "both"):
        ok = [s for s in seg.segments
              if not why_not(s, dwell, now, p, final=ignore_cooldown)]
        if ok:
            here = [s for s in ok if s.key == dwell.current]
            if here:
                return LabelChoice(here[0], "segmentation", here[0].complete)
            ok.sort(key=lambda s: -dwell.exited_at.get(s.key, -math.inf))
            return LabelChoice(ok[0], "segmentation", ok[0].complete)

    if p.trigger in ("distance", "both"):
        here = next((s for s in seg.segments if s.key == dwell.current), None)
        if (here is not None and here.area_id is None
                and here.area_m2 >= p.min_area_m2
                and distance_from_saved_m is not None
                and distance_from_saved_m >= p.distance_prompt_m):
            return LabelChoice(here, "distance", here.complete)
    return None


def never_entered(seg: Segmentation, dwell: Dwell,
                  min_area_m2: float = 2.5) -> list[Segment]:
    """spec 4.6, at completion: complete, unsaved rooms >= min_area she never
    drove into. They are saved unnamed and queued on the dashboard."""
    return [s for s in seg.segments
            if s.complete and s.area_id is None and s.area_m2 >= min_area_m2
            and not dwell.entered(s.key)]
