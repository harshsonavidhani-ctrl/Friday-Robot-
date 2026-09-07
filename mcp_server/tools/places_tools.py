"""places_tools.py — step 9 §5 + §9: name a place, then be told to go there.

    remember_place("the door")     tag where I am standing
    list_places()                  what I know, and how far each one is
    forget_place("the door")
    go_to("the door")              LOOK IT UP, PLAN A ROUTE, DRIVE THERE

`slam/places.py` has existed since 2026-08-27 and nothing ever exposed it, so
Friday could remember a place and had no way to be told about one. This is the
wiring, plus the driving half in `slam/navigate.py`.

⚠ THE FRAME IS THE HARD PART, NOT THE DRIVING. A place is stored as x/y in the
saved map's coordinates, so "go to the door" is only meaningful if Friday knows
where SHE is in that same map. `state/map.npz` carries no pose -- `map_here`
starts every run at (0, 0, 0) -- so the anchor has to be re-established at the
start of every one of these calls by matching what she can see against what the
map remembers.

**When that match fails, these tools REFUSE.** They do not fall back to
odometry, or to the last known pose, or to driving in roughly the right
direction. A wrong frame does not produce a wandering robot; it produces a
confident one going to the wrong address, which is worse and much harder to
notice. See [[friday-map-relocalisation-ratchet]] for the same lesson learned
expensively on the RL stack's map.
"""

from __future__ import annotations

import json
import logging
import math

import anyio.to_thread

import config
from mcp_server import hardware
from mcp_server.registry import tool
from mcp_server.tools.explore import _Robot
from slam import places
from slam.grid import OccupancyGrid
from slam.mapper import Mapper
from slam.navigate import Navigator

log = logging.getLogger("friday.tools.places")


async def _placed_mapper() -> tuple[Mapper | None, str]:
    """Load the saved map and work out where she is standing in it.

    Returns (mapper, "") on success, or (None, why-not). The caller must treat
    a None as a refusal to act, not as a reason to guess.
    """
    try:
        grid = OccupancyGrid.load()
    except (OSError, ValueError) as exc:
        return None, (f"I have no map saved yet ({exc}). Run map_here() or "
                      f"explore_room() first, then name places in it.")

    known = float(grid.free_mask().sum() + grid.occupied_mask().sum())
    if known * grid.cell ** 2 < config.GOTO_MIN_MAP_M2:
        return None, (f"the saved map only covers "
                      f"{known * grid.cell ** 2:.1f} square metres, which is "
                      f"too little to place myself in. Map more first.")

    # Fold sweeps in from a standstill. Mapper._locate matches each one against
    # the map it was given; no odometry is claimed because nothing moved.
    mapper = Mapper(grid=grid, pose=(0.0, 0.0, 0.0))
    robot = _Robot()
    for _ in range(config.MISSION_SETTLE_SWEEPS):
        scan = await anyio.to_thread.run_sync(_sweep_once, robot)
        if scan:
            mapper.update(scan)

    if not mapper.pose_trusted:
        return None, ("I can see the room but I cannot match it to my map, so "
                      "I do not know where I am standing. I will not drive to "
                      "a place I would only be guessing the direction of.")
    return mapper, ""


def _sweep_once(robot) -> dict[int, int]:
    from mcp_server.tools import mapping
    return mapping._sweep()


@tool
async def remember_place(name: str, note: str = "") -> str:
    """Tag where I am standing right now with a name, like "the door".

    I work out where I am on my saved map first. If I cannot place myself I
    say so rather than saving a position I am not sure of -- a place saved at
    the wrong coordinates is worse than no place at all, because I will drive
    to it confidently later.

    Args:
        name: what to call this spot: "the door", "the sofa", "my charger".
        note: anything else worth remembering about it.
    """
    name = (name or "").strip()
    if not name:
        return "I need a name for this place."

    mapper, why = await _placed_mapper()
    if mapper is None:
        return f"I did not save that: {why}"

    row = places.remember_here(name, mapper.pose, note=note or None)
    return (f"Saved “{row['name']}” at x={row['x']:.2f} y={row['y']:.2f} "
            f"on my map. Say “go to {row['name']}” and I will drive there.")


@tool
def list_places() -> str:
    """Every place I know by name, with where it is on my map."""
    rows = places.all()
    if not rows:
        return ("I do not know any places yet. Stand somewhere and say "
                "“remember this as the door”.")
    lines = [f"I know {len(rows)} place(s):"]
    for r in rows:
        seen = f", visited {r['visits']}x" if r.get("visits") else ""
        lines.append(f"  • {r['name']} — x={r['x']:.2f} y={r['y']:.2f}{seen}")
    return "\n".join(lines)


@tool
def forget_place(name: str) -> str:
    """Forget a place I have remembered."""
    return (f"Forgotten “{name}”." if places.forget(name)
            else f"I do not know anywhere called “{name}”.")


@tool
async def go_to(place: str, budget_m: float | None = None,
                budget_seconds: float | None = None) -> str:
    """Drive to a place I know by name, planning a route around what I know.

    I DRIVE MYSELF when you call this. I look the place up, work out where I
    am, plan a route across the map, and drive it in short legs — re-planning
    after every leg so anything I discover on the way gets routed around
    rather than driven into. I scan before every move and the safety
    interrupt is active throughout. Call stop() and I stop between moves.

    I refuse rather than guess if I cannot find the place, cannot work out
    where I am, or cannot find a route to it.

    Args:
        place: the name, or part of it — "door" finds "the door".
        budget_m: how far I may drive trying.
        budget_seconds: how long I may take.
    """
    target = places.find(place)
    if target is None:
        known = ", ".join(r["name"] for r in places.all()) or "nowhere yet"
        return (f"I do not know anywhere called “{place}”. I know: {known}.")

    budget_m = config.GOTO_BUDGET_M if budget_m is None else float(budget_m)
    budget_seconds = (config.GOTO_BUDGET_S if budget_seconds is None
                      else float(budget_seconds))
    if budget_m <= 0 or budget_seconds <= 0:
        return "a budget of zero means there is nothing to do. Refusing."

    mapper, why = await _placed_mapper()
    if mapper is None:
        return f"I am not going to set off: {why}"

    hardware.ABORT.clear()
    robot = _Robot()
    robot._trips = robot.reflex_trips()      # baseline, not a stop
    report = await Navigator(robot).go(
        mapper, (target["x"], target["y"]), name=target["name"],
        arrive_m=target.get("radius_m") or config.GOTO_ARRIVE_M,
        budget_m=budget_m, budget_s=budget_seconds)

    if report["arrived"]:
        places.visited(target["name"])
    try:
        mapper.grid.save()
        report["map_saved"] = str(config.SLAM_MAP_FILE)
    except Exception as exc:                 # noqa: BLE001
        report["map_saved"] = f"could not save the map: {exc}"
    log.info("go_to %s: %s", target["name"], report["why"])
    return json.dumps(report, indent=2)
