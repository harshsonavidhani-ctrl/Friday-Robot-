"""map_growth.py — the saved navigation map grows with every explore trip.

    result = map_growth.grow(trip_grid, frame="saved" | "trip", trusted=True)

Harsh, 2026-09-24: "it uses map like map growes while exploring and it should
use the map". The saved navigation map (state/map.npz) was 13 m² from 30
sweeps; from much of the hostel room two spots on it looked alike, so she
could not place herself and every trip had to start a throw-away map. A map
that grows with each trip stops being ambiguous: the next trip starts on it.

TWO WAYS A TRIP'S MAP COMES BACK, matching the two ways a trip starts
(`nav_tools._explore_trip`):

  * frame "saved": she could be placed on the saved map, so the trip
    explored ON A COPY of it -- same frame, new evidence folded in. That copy
    simply replaces the file.
  * frame "trip": she could not, so the trip began a fresh map whose origin
    is the spot she started from. Its occupied cells are then points
    relative to that spot, and `global_localise.localise_points` finds where
    that spot is on the saved map -- a room's worth of walls is far more
    distinctive than one sweep, which is why this can succeed where the
    start-of-trip fix refused. The trip's evidence is then added to the saved
    map cell by cell. If it does not fit, the saved map is left exactly as it
    was and the trip's map is kept in the bundle's history, not thrown away.

⚠ THE OLD MAP IS NEVER LOST. Before anything is written, `nav_bundle` is
refreshed so the bundle holds the map as it was; writing the new one then
makes the bundle copy that pair into history/ (history is never deleted).
Places saved on the old map stay valid: both paths keep the saved map's
frame.

⚠ A MAP IS ONLY SAVED FROM A TRIP THAT KNEW WHERE IT WAS. A trip whose final
pose could not be matched may have smeared walls into the wrong place, and
folding that into the map every later journey plans on would make one bad
trip permanent. `trusted=False` saves nothing.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path

import numpy as np

import config

log = logging.getLogger("friday.map_growth")

# A trip map is fitted with fewer of its points judgeable than one sweep
# needs (global_localise.MIN_COVERAGE, 0.45): a trip that explored past the
# saved map's edge legitimately has walls the saved map has never seen.
FIT_MIN_COVERAGE = 0.30

# A save that would leave the map knowing less than this fraction of what it
# knew is refused: growth must not be a way to lose the map.
MIN_KEEP_FRACTION = 0.90


def known_m2(grid) -> float:
    if grid is None:
        return 0.0
    return float((~grid.unknown_mask()).sum()) * grid.cell ** 2


def trip_points(trip) -> np.ndarray:
    """The trip map's walls, as points relative to where the trip started."""
    cells = np.argwhere(trip.occupied_mask())
    if not len(cells):
        return np.empty((0, 2))
    return trip.origin + (cells + 0.5) * trip.cell


def fit(saved, trip):
    """Where the trip's start spot is on `saved`. A global_localise.Fix.

    Two stages, like finding her from one sweep: the whole-map search says
    WHICH place (and refuses when two look alike), then the tracker's own
    matcher polishes it. The search works on the map's lattice and was
    measured 16 cm out on ray-cast trips -- close enough to know the place,
    not to lay one map's walls on another's without doubling them.
    """
    import global_localise
    from slam.core import LikelihoodField, match_pose

    points = trip_points(trip)
    fix = global_localise.localise_points(saved, points,
                                          min_coverage=FIT_MIN_COVERAGE)
    if not fix.ok:
        return fix
    try:
        field = LikelihoodField.from_grid(saved)
        pose, score, _sharp = match_pose(field, points, fix.pose,
                                         prior_m=POLISH_M,
                                         prior_deg=POLISH_DEG)
        if score >= config.SLAM_MIN_SCORE:
            fix.pose = tuple(float(v) for v in pose)
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not polish the fit: %s", exc)
    return fix


# How far the polish may move a fit: a little over the search's measured
# error, and no further -- a different place is the search's job, not this.
POLISH_M = 0.25
POLISH_DEG = 6.0


def merge(saved, trip, pose):
    """A new grid: `saved` plus the trip's evidence, placed at `pose`.

    Resampled from the TARGET side -- every cell of the saved lattice under
    the trip's footprint looks up the trip cell beneath its centre -- so a
    rotated trip map leaves no unknown pinholes in open floor, which the
    planner would otherwise read as specks it may not cross.
    """
    out = copy.deepcopy(saved)
    nonzero = np.argwhere(trip.logodds != 0.0)
    if not len(nonzero):
        return out
    x, y, theta = (float(v) for v in pose)
    c, s = math.cos(theta), math.sin(theta)

    local = trip.origin + (nonzero + 0.5) * trip.cell
    world = np.column_stack((local[:, 0] * c - local[:, 1] * s + x,
                             local[:, 0] * s + local[:, 1] * c + y))
    out._grow_to(world.min(axis=0), world.max(axis=0), margin_m=0.5)

    ix, iy = out.cells_of(np.array([world.min(axis=0), world.max(axis=0)]))
    i0, i1 = int(ix[0]), int(ix[1])
    j0, j1 = int(iy[0]), int(iy[1])
    nx, ny = out.shape
    i0, j0 = max(i0 - 1, 0), max(j0 - 1, 0)
    i1, j1 = min(i1 + 1, nx - 1), min(j1 + 1, ny - 1)
    gi, gj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1),
                         indexing="ij")
    cx = out.origin[0] + (gi + 0.5) * out.cell - x
    cy = out.origin[1] + (gj + 0.5) * out.cell - y
    # The inverse of the trip->saved transform: back into the trip's frame.
    tx, ty = c * cx + s * cy, -s * cx + c * cy
    ti = np.floor((tx - trip.origin[0]) / trip.cell).astype(np.int64)
    tj = np.floor((ty - trip.origin[1]) / trip.cell).astype(np.int64)
    tnx, tny = trip.shape
    valid = (ti >= 0) & (ti < tnx) & (tj >= 0) & (tj < tny)
    add = np.zeros(gi.shape, dtype=np.float32)
    add[valid] = trip.logodds[ti[valid], tj[valid]]
    out.logodds[gi, gj] = np.clip(out.logodds[gi, gj] + add,
                                  config.SLAM_LOGODDS_MIN,
                                  config.SLAM_LOGODDS_MAX)
    out.scans = int(out.scans) + int(trip.scans)
    out.updated_at = time.time()
    return out


def _keep_unfitted(trip, why: str) -> str | None:
    """A trip map that did not fit goes into the bundle's history, not away."""
    try:
        import nav_bundle
        dest = (nav_bundle.bundle_dir() / "history"
                / f"{time.strftime('%Y%m%d-%H%M%S')}-trip-not-fitted")
        dest.mkdir(parents=True, exist_ok=True)
        trip.save(dest / "map.npz")
        (dest / "why.txt").write_text(why + "\n")
        return str(dest)
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not keep the unfitted trip map: %s", exc)
        return None


def grow(trip, frame: str, trusted: bool = True,
         path: Path | str | None = None) -> dict:
    """Fold an explore trip's map into the saved map. Never raises."""
    from slam.grid import OccupancyGrid

    target = Path(path) if path is not None else Path(config.SLAM_MAP_FILE)
    if not trusted:
        return {"grew": False, "why": "the trip did not end knowing where it "
                                      "was, so its map is not saved"}
    try:
        saved = OccupancyGrid.load(target)
    except FileNotFoundError:
        saved = None
    except Exception as exc:                     # noqa: BLE001
        return {"grew": False, "why": f"could not read the saved map: {exc}"}
    before = known_m2(saved)

    fitted = None
    try:
        if saved is None or frame == "saved":
            # No map yet: the trip's map IS the map, its origin where she
            # started. Or the trip explored on a copy of this very map.
            new = trip
        else:
            fix = fit(saved, trip)
            if not fix.ok:
                kept = _keep_unfitted(trip, fix.why)
                log.info("trip map did not fit the saved map: %s", fix.why)
                return {"grew": False, "fitted": False, "why": fix.why,
                        "kept": kept}
            fitted = fix.as_dict()
            new = merge(saved, trip, fix.pose)
    except Exception as exc:                     # noqa: BLE001
        # _grow_to refuses a map past SLAM_MAP_MAX_M: a bad pose, not a big
        # room. The saved map stays as it is.
        log.warning("the map was not grown: %s", exc)
        return {"grew": False, "why": f"the map was not grown: {exc}"}

    after = known_m2(new)
    if saved is not None and after < MIN_KEEP_FRACTION * before:
        return {"grew": False, "why": (f"refused: the grown map would know "
                                       f"{after:.1f} m² against {before:.1f}")}
    try:
        import nav_bundle
        nav_bundle.save_quietly("before the map grew")   # the old pair, kept
        new.save(target)
        nav_bundle.save_quietly(
            f"the map grew by exploring: {before:.1f} -> {after:.1f} m²")
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not save the grown map: %s", exc)
        return {"grew": False, "why": f"could not save: {exc}"}
    log.info("map grew %.1f -> %.1f m² (%s)", before, after,
             "fitted" if fitted else frame)
    return {"grew": True, "before_m2": round(before, 1),
            "after_m2": round(after, 1), "added_m2": round(after - before, 1),
            "frame": frame, "fitted": fitted}
