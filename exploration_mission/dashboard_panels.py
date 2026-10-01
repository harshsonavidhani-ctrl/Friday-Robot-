"""The exploration mission's dashboard panel (File 21 step 11, spec 4.16).

    GET  /api/mission  -> panel(mgr)            (through the MCP tool mission_status)
    POST /api/mission  -> service.name_area()   naming ONLY: {area_id, name}

Read-only here. Everything is drawn in ONE frame, so nothing from one frame is
ever painted onto another (CLAUDE.md PART 5: "drawing one's coordinates on the
other's picture is wrong and looks right"):

  * a mission is running: its own saved frame (the home anchor's when it
    started at home, else "mission:<id>"); the trail, pose, start and
    frontiers are turned into it with the mission's start pose;
  * otherwise: the home anchor's frame.

Areas saved in any other frame are counted, not drawn.
"""

from __future__ import annotations

import math
import time

from exploration_mission import area_store, mission_map

MISSIONS_SHOWN = 8


def _r(v, nd: int = 2):
    return None if v is None else round(float(v), nd)


def _pts(points, nd: int = 2) -> list[list[float]]:
    return [[round(float(p[0]), nd), round(float(p[1]), nd)] for p in points]


def _wall(iso) -> float | None:
    try:
        return round(area_store.iso_to_wall(iso), 1) if iso else None
    except (TypeError, ValueError):
        return None


def _mission_row(m: dict) -> dict:
    return {"id": m["id"], "status": m.get("status"),
            "started_wall": _wall(m.get("started_at")),
            "stop_reason": m.get("stop_reason"),
            "started_at": m.get("started_at"), "ended_at": m.get("ended_at"),
            "distance_m": _r(m.get("distance_m"), 1),
            "explored_m2": _r(m.get("explored_m2_end"), 1),
            "home_error_m": _r(m.get("home_error_m")),
            "rth_attempts": m.get("rth_attempts"),
            "rungs_used": m.get("rungs_used") or [],
            "by": next((e.get("by") for e in (m.get("events") or [])
                        if e.get("kind") == "start"), None)}


def panel(mgr, areas: bool = True) -> dict:
    """The whole panel as plain JSON-able data. Holds the manager's lock for
    the read (one tick at most), so the picture is one instant."""
    with mgr._lock:
        snap = mgr.snapshot()
        env = mgr.env_id if mgr.mission is not None else mgr.a.env.env_id()
        home = mgr.store.home_anchor(env)
        live = mgr.mission is not None
        frame = (mgr._frame() if live
                 else (home or {}).get("frame") or area_store.FRAME_MAP)
        out: dict = {"enabled": True, "running": True, "env_id": env,
                     "state": mgr.state, "frame": frame, "snapshot": snap,
                     "mapped_m2": snap.get("explored_m2"),
                     "home": None, "start": None, "pose": None, "trail": [],
                     "frontiers": []}
        if home is not None and home.get("frame") == frame:
            out["home"] = {"x": _r(home["x"]), "y": _r(home["y"]),
                           "theta": _r(home["theta"], 3)}
        if live:
            m = mgr.mission
            try:
                out["elapsed_s"] = round(time.time()
                                         - area_store.iso_to_wall(m["started_at"]))
            except (TypeError, ValueError):
                out["elapsed_s"] = None
            start = mgr.T if mgr.T is not None else (0.0, 0.0, 0.0)
            out["start"] = {"x": _r(start[0]), "y": _r(start[1]),
                            "theta": _r(start[2], 3)}
            pose = mission_map.compose(mgr.T, mgr.pose) if mgr.T is not None \
                else mgr.pose
            out["pose"] = {"x": _r(pose[0]), "y": _r(pose[1]), "theta": _r(pose[2], 3)}
            out["trail"] = _pts(mgr._to_store([(c[0], c[1]) for c in mgr.trail.crumbs]))
            res = mgr.analysis.latest
            if res:
                out["frontiers"] = [
                    {"x": p[0], "y": p[1], "length_m": _r(f.length_m)}
                    for f, p in zip(res["frontiers"].remaining,
                                    _pts(mgr._to_store([f.centroid for f in
                                                        res["frontiers"].remaining])))]
        if areas:
            every = mgr.store.list_areas(env)
            drawn = [a for a in every if a["frame"] == frame]
            out["areas"] = [{"id": a["id"], "name": a["name"],
                             "display_name": a["display_name"], "status": a["status"],
                             "area_m2": _r(a["area_m2"], 1),
                             "center": [_r(a.get("center_x")), _r(a.get("center_y"))],
                             "polygon": _pts(a["polygon"] or [])} for a in drawn]
            out["areas_elsewhere"] = len(every) - len(drawn)
            out["naming_queue"] = [{"id": a["id"], "display_name": a["display_name"],
                                    "status": a["status"],
                                    "area_m2": _r(a["area_m2"], 1)}
                                   for a in every if a["status"] in ("unnamed", "stale")]
            out["missions"] = [_mission_row(m) for m in
                               mgr.store.missions(env, limit=MISSIONS_SHOWN)]
        ladder = snap.get("return")
        out["ladder"] = ladder
        return out


def bounds(data: dict, pad_m: float = 0.5) -> tuple[float, float, float, float] | None:
    """(xmin, ymin, xmax, ymax) round everything drawn, for a picture."""
    xs, ys = [], []
    for a in data.get("areas") or []:
        for x, y in a["polygon"]:
            xs.append(x)
            ys.append(y)
    for key in ("home", "start", "pose"):
        p = data.get(key)
        if p:
            xs.append(p["x"])
            ys.append(p["y"])
    for x, y in data.get("trail") or []:
        xs.append(x)
        ys.append(y)
    if not xs:
        return None
    return (min(xs) - pad_m, min(ys) - pad_m, max(xs) + pad_m, max(ys) + pad_m)


def finite(data) -> bool:
    """True when nothing in the panel is NaN / inf (JSON would refuse it)."""
    if isinstance(data, float):
        return math.isfinite(data)
    if isinstance(data, dict):
        return all(finite(v) for v in data.values())
    if isinstance(data, (list, tuple)):
        return all(finite(v) for v in data)
    return True
