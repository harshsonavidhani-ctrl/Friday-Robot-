"""File 19's tools: which environment this is, and whether the map is busy.

READ-ONLY, on purpose. Creating or switching an environment, and naming a new
area, are his words (offline.py's env_* intents, handled in main.py) or the
dashboard's one naming box -- never something the brain can decide to do.
"""

from __future__ import annotations

import environments
import run_summary
from mcp_server.registry import tool


@tool
def map_in_use() -> str:
    """Empty when the navigation map is free; otherwise why it is not (a run
    or a journey is using it). main.py asks before swapping environments."""
    try:
        from mcp_server.tools import autonomy, nav_tools
        if autonomy._running():
            return "an exploration run is going"
        journey = nav_tools._journey_running()
        if journey:
            return f"I'm on a journey ({journey})"
    except Exception as exc:                        # noqa: BLE001
        return f"I can't tell if the map is free ({type(exc).__name__})"
    return ""


@tool
def environment_status() -> str:
    """Which place (environment) she is in, whether the boot check recognised
    it, how many new areas wait for a name, and the last run's summary."""
    env = environments.active() or {}
    status = environments.status()
    waiting = environments.pending()
    parts = [f"I'm in {env.get('name', 'an unnamed place')}."]
    state = status.get("state")
    if state == "no_match":
        parts.append("At start-up my scan did not match its map — Harsh "
                     "should say whether we're somewhere new.")
    elif state == "no_map":
        parts.append("This place has no map yet.")
    if waiting:
        parts.append(f"{len(waiting)} new area"
                     f"{'s wait' if len(waiting) != 1 else ' waits'} for a "
                     f"name.")
    runs = run_summary.panel().get("runs") or []
    if runs:
        parts.append(f"Last run: {runs[0].get('line', '')}")
    return " ".join(parts)
