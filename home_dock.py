"""home_dock.py — Home is saved at boot, and exploring always ends there.

    home_dock.save_boot_home(result)      boot: no home yet? this spot is home
    home_dock.home_heading()              the heading saved with home (radians)
    home_dock.wants_home_after(status)    did this run end on its own?
    home_dock.request_home_after_explore()  ask the MCP server to drive home

Harsh, 2026-09-27: "Saving the Home Position (Initial Pose) ... save its initial
starting coordinates (x, y, theta) as the permanent Home or Dock waypoint" and
"its path planner will automatically calculate the safest route back to the
starting coordinates while avoiding any new obstacles along the way."

THE HOME IS SAVED ONCE, AND KEPT. At boot the environment check (file 19)
places her on the active map from one LiDAR sweep, without moving. If that
environment has NO home yet and the placement is unambiguous, the boot pose
(x, y and theta) becomes `home` -- theta kept in the place's note, since the
places table has no heading column. A brand-new environment has no map yet:
its map will be built from where she stands, so home is that map's origin,
(0, 0, 0). An existing home is NEVER moved automatically ("permanent"); he
moves it by standing there and saying "remember this as home". ⚠ An
AMBIGUOUS fix (two look-alike spots) never saves home: nav_state_machine.
set_home's own warning -- a wrong home poisons every later return.

EXPLORING ENDS AT HOME. The exploration run is untouched (rl/, slam/,
exploration_mode.py, explore.py -- the hard rule): it explores, and its own
way-back drives to where the run started. When the run has ended BY ITSELF
(its timer, or nowhere left to explore -- never a person's "stop"),
environment_watch calls request_home_after_explore(), which asks the MCP
server's return_home_after_explore tool to finish the trip with file 14's
navigation engine: relocalise on the saved map, A* with the clearance
gradient, one guarded move per tick, RuntimeObstacles + re-planning around
anything new in the way, FAILSAFE if it cannot. Already within
HOME_ARRIVE_M of home: she just says so. Arrived: she turns to face the
saved heading, one guarded pivot.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

import config

log = logging.getLogger("friday.home_dock")


def _note(row: dict | None) -> dict:
    try:
        data = json.loads((row or {}).get("note") or "{}")
        return data if isinstance(data, dict) else {}
    except (TypeError, ValueError):
        return {}


def home_row() -> dict | None:
    try:
        import nav_state_machine as nsm
        return nsm.home_place()
    except Exception as exc:                        # noqa: BLE001
        log.debug("no home lookup: %s", exc)
        return None


def home_heading() -> float | None:
    """The heading saved with home, radians, or None if none was saved."""
    theta = _note(home_row()).get("theta")
    try:
        return None if theta is None else float(theta)
    except (TypeError, ValueError):
        return None


def save_boot_home(result: dict) -> str:
    """Boot: make this spot home if there is none. -> what happened (logged)."""
    if not getattr(config, "HOME_SAVE_AT_BOOT", True):
        return "off"
    if home_row() is not None:
        row = home_row()
        why = (f"home kept at ({row['x']:.2f}, {row['y']:.2f}) — it is "
               f"permanent; say 'remember this as home' to move it")
        log.info(why)
        return why
    import nav_state_machine as nsm
    state = result.get("state")
    if state == "no_map":
        pose, how = (0.0, 0.0, 0.0), "the origin of the new map"
    elif state == "matched" and result.get("pose") and result.get("sure"):
        pose, how = tuple(float(v) for v in result["pose"]), "the boot pose"
    else:
        why = ("not saving home: the boot fix was not certain "
               f"({result.get('why') or state})")
        log.info(why)
        return why
    note = json.dumps({"theta": round(pose[2], 4), "set_by": "boot",
                       "at": time.time()})
    nsm.set_home(pose, note=note)
    why = (f"home saved at boot: ({pose[0]:.2f}, {pose[1]:.2f}, "
           f"{math.degrees(pose[2]):.0f}°) — {how}")
    log.info(why)
    return why


def wants_home_after(status: dict, log_tail: str = "") -> tuple[bool, str]:
    """Did this exploration run end ON ITS OWN? (go home?, why)."""
    if not getattr(config, "EXPLORE_ENDS_AT_HOME", True):
        return False, "switched off (EXPLORE_ENDS_AT_HOME)"
    state = str(status.get("state") or "").lower()
    why = str(status.get("stopped_because") or status.get("why") or "").lower()
    tail = (log_tail or "").lower()
    # "interrupted" = clear_stale_run found a run cut off (power loss): never
    # a reason to set off on her own.
    if state in ("stopped", "failed", "error", "interrupted", "killed") \
            or "hand" in why \
            or "stop" in why or "sigint" in tail[-600:] \
            or "stopped by hand" in tail:
        return False, f"the run was stopped ({state} {why})".strip()
    if status.get("exit_code") not in (None, 0):
        return False, f"the run failed (exit {status.get('exit_code')})"
    if home_row() is None:
        return False, "no home saved"
    return True, "the run ended on its own"


def request_home_after_explore() -> str:
    """From the environment watch's thread: ask the MCP server to go home.

    Through an MCP client session like every other caller (brain.mcp_session),
    so the journey runs on the server's own event loop and keeps every guard.
    """
    import asyncio

    async def ask() -> str:
        import brain
        async with brain.mcp_session() as session:
            result = await session.call_tool("return_home_after_explore", {})
            return " ".join(getattr(p, "text", "") or ""
                            for p in (result.content or []))
    try:
        said = asyncio.run(ask())
    except Exception as exc:                        # noqa: BLE001
        said = f"could not ask for the way home: {type(exc).__name__}: {exc}"
    log.info("after exploring: %s", said[:200])
    return said


def heading_turn(current: float, target: float) -> tuple[str, float] | None:
    """(direction, seconds) to face `target`, or None if close enough."""
    diff = math.atan2(math.sin(target - current), math.cos(target - current))
    if abs(math.degrees(diff)) < float(config.HOME_FACE_TOLERANCE_DEG):
        return None
    seconds = abs(math.degrees(diff)) / float(config.AGENCY_TURN_DEG_PER_S)
    return ("left" if diff > 0 else "right",
            min(seconds, float(config.HOME_FACE_MAX_S)))
