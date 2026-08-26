"""Mapping tools: the LiDAR's raw sweep, and building a map from where I stand.

⚠ NOTHING IN HERE MOVES THE ROBOT. Every tool is read-only: it takes sweeps
from the sensor that is already running and folds them into slam/. That is
deliberate, and it is the whole point of this file existing before any
exploration tool does. The mapping code in slam/ has never seen a real room
-- every test it has passed was against a ray-cast simulation -- so the first
thing to find out is whether a real LiDAR in a real room produces a map that
looks like the room. Finding that out must not require the robot to drive.

WHY THESE LIVE HERE AND NOT IN slam/. The MCP server owns the LiDAR port and
is the only thing allowed to open it; slam/ deliberately knows nothing about
hardware and takes {bearing: mm} dicts so it can be tested without any. This
file is the join between them, and it is the ONLY place the two meet.

The order to use them in, on a robot that has never done this before:

    lidar_sweep()      does the sensor answer, and do the numbers look sane
    map_here()         fold N sweeps from a standing start into a map
    describe_map()     what does the map think it has, in words

Then look at the picture: `./venv/bin/python scripts/map_here.py --show`.
"""

from __future__ import annotations

import json
import math
import time

import config
from mcp_server import hardware
from mcp_server.registry import tool
from slam.grid import OccupancyGrid
from slam.mapper import Mapper


def _sweep(max_age_s: float | None = None) -> dict[int, int]:
    """One sweep from the running LiDAR, self-mask and stale bearings gone.

    Defaults to config.LIDAR_SWEEP_MAX_AGE_S, NOT to the 1.0 s that
    Lidar.scan() uses for clearance. Mapping and clearance want opposite
    things from that window: a clearance check wants anything seen recently,
    and mapping wants one coherent instant. See the config note.
    """
    if max_age_s is None:
        max_age_s = config.LIDAR_SWEEP_MAX_AGE_S
    return hardware.ranges().lidar.scan(max_age_s=max_age_s)


@tool
def lidar_sweep(full: bool = False) -> str:
    """The LiDAR's raw 360° sweep, for checking the sensor before trusting it.

    ⚠ THE BRAIN SHOULD NEVER PASS full=True. It adds the complete per-degree
    table — 350-odd numbers — which is a wall of tokens that says less than
    the sector summary below it. It exists for scripts that need to feed real
    sweeps into slam/ (scripts/pivot_test.py), because the MCP server owns the
    LiDAR port and is the only thing that may read it.

    Args:
        full: include the complete {bearing: mm} table. Scripts only.

    Returns JSON: how many bearings answered, the nearest and furthest return,
    the widest gap in coverage, and a coarse 24-sector summary in millimetres
    (null where nothing answered). The full per-degree table is NOT returned —
    360 numbers is a wall of tokens that says less than the summary does.

    WHAT GOOD LOOKS LIKE, in a normal room: 250-350 bearings answering, a
    nearest return well over 200 mm, and no gap wider than about 20°. A gap of
    90° or more means a whole quadrant is returning nothing, which is a real
    fault — a loose cable, or something resting against the dome — not a quiet
    room. Bearings 89-98 are absent BY DESIGN: that is Friday's own chassis
    (config.LIDAR_SELF_MASK) and it is masked in sensors.py.

    Read-only. Does not move the robot.
    """
    scan = _sweep()
    if not scan:
        return json.dumps({"beams": 0,
                           "error": "no bearings answered — the LiDAR is not "
                                    "spinning, not connected, or every return "
                                    "is stale"})

    bearings = sorted(scan)
    gaps = [(b - a) for a, b in zip(bearings, bearings[1:])]
    gaps.append(bearings[0] + 360 - bearings[-1])
    sectors = {}
    for start in range(0, 360, 15):
        inside = [scan[b] for b in bearings if start <= b < start + 15]
        sectors[f"{start:03d}"] = min(inside) if inside else None

    # Ages too, because a "scan" is NOT a sweep -- Lidar.scan() hands back a
    # rolling window in which every bearing carries its own capture time, and
    # how far those times spread is exactly how much a moving robot smears the
    # cloud it folds into the map. Measuring it needs the port, so it has to
    # come from here.
    body = {}
    if full:
        ages = hardware.ranges().lidar.scan_ages(
            max_age_s=config.LIDAR_SWEEP_MAX_AGE_S)
        body = {"bearings_mm": {str(b): scan[b] for b in bearings},
                "ages_ms": {str(b): round(ages[b] * 1000.0, 1)
                            for b in bearings if b in ages}}
    return json.dumps({
        **body,
        "beams": len(scan),
        "nearest_mm": min(scan.values()),
        "furthest_mm": max(scan.values()),
        "widest_gap_deg": max(gaps),
        "sectors_mm": sectors,
        "forward_bearing_deg": config.LIDAR_FORWARD_DEG,
        "note": "bearings 89-98 are Friday's own chassis and are masked",
    }, separators=(",", ":"))


@tool
def map_here(sweeps: int = 30) -> str:
    """Build a map of this room from where I am standing. Does NOT move.

    Folds `sweeps` real LiDAR sweeps into the persistent map at
    config.SLAM_MAP_FILE and reports what came out. The robot stays exactly
    where it is, so this is safe to run at any time, including while somebody
    is holding it.

    This is the first honest test of the mapping code against a real room. A
    single standing position cannot map a flat — it can only map what is in
    line of sight — so a good result is a recognisable outline of THIS room
    with a shadow behind every solid thing, not a complete floor plan.

    Returns JSON: how much floor was established, how much is occupied, how
    much is still unknown, and `looks_like_a_room` — a sanity check, not a
    guarantee. If that is false, do not trust anything built on top of it and
    say so plainly rather than carrying on.
    """
    sweeps = max(1, min(int(sweeps), 200))
    mapper = Mapper(pose=(0.0, 0.0, 0.0))

    used, empty = 0, 0
    started = time.monotonic()
    for _ in range(sweeps):
        scan = _sweep()
        if not scan:
            empty += 1
        else:
            mapper.update(scan)
            used += 1
        # A sweep is 100 ms of sensor. Asking faster returns the same one
        # twice, which the transient gate would then treat as confirmation.
        time.sleep(0.12)

    if not used:
        return json.dumps({"error": "no sweeps answered — see lidar_sweep()",
                           "sweeps_tried": sweeps})

    stats = mapper.grid.stats()
    floor = mapper.grid.free_mask().sum() * mapper.grid.cell ** 2
    walls = stats["occupied"]
    # A room has floor, has walls around it, and is not mostly nothing. These
    # are deliberately loose: the question is "did this work at all", not
    # "is this a good map".
    sane = floor >= 1.0 and walls >= 50 and stats["cells"] > 0
    saved = mapper.grid.save()

    return json.dumps({
        "sweeps_used": used,
        "sweeps_empty": empty,
        "seconds": round(time.monotonic() - started, 1),
        "floor_m2": round(float(floor), 2),
        "occupied_cells": walls,
        "unknown_cells": stats["unknown"],
        "extent_m": stats["bounds_m"],
        "looks_like_a_room": bool(sane),
        "saved_to": str(saved),
        "how_to_look": "./venv/bin/python scripts/map_here.py --show",
    }, separators=(",", ":"))


@tool
def describe_map() -> str:
    """What the saved map knows, in words rather than numbers.

    Reads config.SLAM_MAP_FILE. Says how big the mapped area is, how much of
    it is floor, and how far the furthest wall is. Returns an error if no map
    has been built yet rather than inventing an empty one.

    Read-only. Does not move the robot.
    """
    try:
        grid = OccupancyGrid.load()
    except FileNotFoundError:
        return json.dumps({"error": "no map yet — run map_here() first"})

    stats = grid.stats()
    x0, y0, x1, y1 = grid.bounds
    floor = grid.free_mask().sum() * grid.cell ** 2
    occupied = grid.occupied_mask()
    if occupied.any():
        cells = __import__("numpy").argwhere(occupied)
        centres = (cells + 0.5) * grid.cell + grid.origin
        furthest = float(max(math.hypot(x, y) for x, y in centres))
    else:
        furthest = 0.0

    return json.dumps({
        "floor_m2": round(float(floor), 2),
        "walls_cells": stats["occupied"],
        "unknown_cells": stats["unknown"],
        "size_m": [round(x1 - x0, 2), round(y1 - y0, 2)],
        "furthest_wall_m": round(furthest, 2),
        "scans": stats["scans"],
        "summary": f"about {floor:.1f} square metres of floor, "
                   f"walls out to {furthest:.1f} metres",
    }, separators=(",", ":"))
