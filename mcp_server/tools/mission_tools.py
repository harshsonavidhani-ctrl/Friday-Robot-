"""File 21's doorway: the exploration mission, for main.py and the dashboard.

PLUMBING, NOT THE BRAIN'S. All five are in config.BRAIN_HIDDEN_TOOLS, so the
model never sees them: the mission is driven by his words through the offline
router (main.handle -> mission_voice), by her own questions (mission_answer)
and by the dashboard (mission_status / mission_start / mission_name_area).
Each returns JSON text and never raises. With config.EXPLORATION_MISSION_SERVICE
off there is no mission, and every one of them says so and does nothing.
"""

from __future__ import annotations

import json

from exploration_mission import service
from mcp_server.registry import tool


def _j(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"))


@tool
def mission_voice(text: str, age_s: float = 0.0) -> str:
    """Internal (main.py): one transcript, before the brain. {"handled": bool}."""
    return _j(service.voice(text, age_s))


@tool
def mission_answer(ask_id: str, text: str = "") -> str:
    """Internal (main.py): the words that answered the mission's question."""
    return _j(service.answer(ask_id, text))


@tool
def mission_status() -> str:
    """Internal (dashboard): the exploration mission's panel, read-only."""
    return _j(service.status())


@tool
def mission_start(minutes: float = 0.0) -> str:
    """Internal (dashboard): the Explore button starts a mission."""
    return _j(service.start_mission(minutes or None))


@tool
def mission_name_area(area_id: str, name: str) -> str:
    """Internal (dashboard): name or rename one saved area. Naming only."""
    return _j(service.name_area(area_id, name))
