"""Navigation, assembled for the dashboard. READ-ONLY.

File 14 step 11. THIS MODULE still writes nothing: places are read through
`slam/places.py`, the attempt log through `nav_state_machine.recent_attempts()`,
and the map is opened only to learn its bounds.

The PANEL is no longer read-only, though, and the change was deliberate: Harsh
asked for target navigation "via voice triggered and the dashboard triggered"
(2026-09-21), so `server.py` grew a `POST /api/nav`. The concern that kept it
read-only — an autonomous drive being one stray tap away — is answered where
the Explore button answers it: every action goes out through the MCP server
and its safety interrupt, never the motors, and the destination must be a
place she already knows.

⚠ THE MARKERS ARE NOT DRAWN OVER THE MAP PICTURE, AND THAT IS A CORRECTNESS
DECISION RATHER THAN A SHORTCUT. The Map panel shows `state/rl_map.npz` -- the
RL exploration stack's dead-reckoned map. Landmarks are stored in
`state/map.npz`'s frame, the scan-matched survey, because that is the frame
`go_to`, `navigate_to` and `remember_place` all work in. The two maps have
different origins, different extents and different provenance: on 2026-09-21
the survey covered 7.5 x 8.1 m and the RL map claimed 24 x 24 m for the same
flat. Drawing survey coordinates onto the RL picture would put "the kitchen"
somewhere it has never been, and it would look authoritative while doing it.

So this returns marker positions normalised to the SURVEY map's own bounds,
and the panel draws its own small plan from them. It says which frame it is in
underneath, because a map picture that does not say what it is is how the two
got confused in the first place.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("friday.dashboard.nav")


def _bounds() -> dict:
    """The survey map's extent, or why it has none. Opens, never writes."""
    try:
        import config
        from slam.grid import OccupancyGrid
        grid = OccupancyGrid.load(config.SLAM_MAP_FILE)
        x0, y0, x1, y1 = grid.bounds
        known = int(grid.free_mask().sum() + grid.occupied_mask().sum())
        return {"ok": True, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "size_m": [round(x1 - x0, 1), round(y1 - y0, 1)],
                "known_m2": round(known * grid.cell ** 2, 1),
                "sweeps": int(grid.scans), "cell_m": grid.cell,
                "file": "state/map.npz"}
    except Exception as exc:                     # noqa: BLE001
        return {"ok": False, "why": f"no survey map yet ({exc})",
                "file": "state/map.npz"}


def _bundle_status() -> dict:
    try:
        import nav_bundle
        return nav_bundle.status()
    except Exception as exc:                     # noqa: BLE001
        return {"saved": False, "why": str(exc)}


def _places() -> list[dict]:
    try:
        from slam import places
        return places.all()
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not read places: %s", exc)
        return []


def current() -> dict:
    """Everything the two new panels need, in one read."""
    import nav_state_machine as nsm

    bounds = _bounds()
    rows = _places()
    home = next((r for r in rows if r["name"] == nsm.HOME_PLACE_NAME), None)

    landmarks = []
    for row in rows:
        mark = {
            "name": row["name"],
            "x": round(row["x"], 2),
            "y": round(row["y"], 2),
            "radius_m": round(float(row.get("radius_m") or 0.0), 2),
            "visits": int(row.get("visits") or 0),
            "is_home": row["name"] == nsm.HOME_PLACE_NAME,
            "from_home": (None if home is None else
                          round(math.hypot(row["x"] - home["x"],
                                           row["y"] - home["y"]), 2)),
        }
        if bounds.get("ok"):
            span_x = max(bounds["x1"] - bounds["x0"], 1e-6)
            span_y = max(bounds["y1"] - bounds["y0"], 1e-6)
            # 0..1 across the survey map. y is flipped so the drawing reads
            # the same way up as the Map panel's picture does.
            mark["u"] = round((row["x"] - bounds["x0"]) / span_x, 4)
            mark["v"] = round(1.0 - (row["y"] - bounds["y0"]) / span_y, 4)
            mark["on_map"] = 0.0 <= mark["u"] <= 1.0 and 0.0 <= mark["v"] <= 1.0
        landmarks.append(mark)
    landmarks.sort(key=lambda m: (not m["is_home"], m["name"]))

    attempts = nsm.recent_attempts(limit=12)
    home_runs = [a for a in attempts if a.get("destination") == "home"]

    return {
        "ok": True,
        "frame": bounds,
        "landmarks": landmarks,
        "home": (None if home is None else
                 {"x": round(home["x"], 2), "y": round(home["y"], 2),
                  "visits": int(home.get("visits") or 0)}),
        "attempts": attempts,
        "last_return": home_runs[0] if home_runs else None,
        "states": [s.value for s in nsm.NavState],
        "round_trip": nsm.round_trip_enabled(),
        "waiting_to_come_back": nsm.pending_return(),
        # Step 17: stopped and waiting for a person to say "try again".
        "failsafe": nsm.failsafe_record(),
        # Step 16: the last time exploration ran out of frontiers and she
        # came back early rather than waiting for the timer.
        "early_return": nsm.last_early_return(),
        # Everything above, kept permanently together with the map it belongs
        # to (nav_bundle.py, 2026-09-23).
        "bundle": _bundle_status(),
        "note": ("landmarks are in the survey map's frame (state/map.npz), "
                 "which is NOT the frame of the picture in the Map panel"),
    }
