"""environments.py — File 19: the places Friday lives in, and what she finds there.

    environments.ensure()            first run: today's map + places become "Home"
    environments.active()            the environment she is in
    environments.boot_check(scan)    one sweep, no movement: same place or not?
    environments.switch_to(name)     his words only -- never on her own
    environments.create_new(name)    his words only -- never on her own
    environments.pending()           areas waiting for a name
    environments.name_pending(id, name)   THE dashboard's one write for file 19

⚠ EXPLORATION IS NOT TOUCHED. Nothing here decides where she drives, when, or
how. rl/, slam/, exploration_mode.py and mcp_server/tools/explore.py are read
at most; this file only watches the files they already write.

ONE LIVE ENVIRONMENT. The navigation map (config.SLAM_MAP_FILE) and the
`places` table always belong to the ACTIVE environment -- so file 14's
target-navigation lookup (slam/places.find) resolves against the active one
without a line of slam/ changing, and the same landmark name can exist in two
environments (the places table's UNIQUE name only ever holds one
environment's names). An inactive environment keeps its map and places in
state/environments/<env_id>/. Switching saves the live pair there first and
then loads the other: a map is copied before it is ever overwritten, and
never deleted.

NEVER AUTO-CREATED. When the boot sweep does not fit the active map anywhere,
that is FLAGGED (state/environment_status.json, the dashboard, and a line she
says) and he chooses: "Friday, we're at <name>" or "Friday, this is a new
place". A sweep that fits even ambiguously is the same environment.

The records are memory, type "environment" (memory_governance.TYPES).
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import config

log = logging.getLogger("friday.environments")

KIND = "environment"

SCHEMA = """
CREATE TABLE IF NOT EXISTS environments (
    env_id       TEXT PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL DEFAULT 'environment',
    map_file     TEXT NOT NULL,
    landmarks    TEXT NOT NULL DEFAULT '[]',
    obstacle_log TEXT NOT NULL DEFAULT '{}',
    active       INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    last_active  REAL
);
CREATE TABLE IF NOT EXISTS pending_landmarks (
    id         INTEGER PRIMARY KEY,
    env_id     TEXT NOT NULL,
    x          REAL NOT NULL,
    y          REAL NOT NULL,
    area_m2    REAL NOT NULL,
    radius_m   REAL NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    name       TEXT,
    created_at REAL NOT NULL,
    asked_at   REAL,
    named_at   REAL
);
"""

_ready = False


def _db():
    global _ready
    import memory
    connection = memory.db()
    if not _ready:
        with memory._lock:
            connection.executescript(SCHEMA)
            connection.commit()
        _ready = True
    return connection


def _lock():
    import memory
    return memory._lock


def _dir(env_id: str) -> Path:
    return Path(config.ENV_DIR) / env_id


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:30] or "place"


# --- the places table (file 14's landmarks), read and swapped, never altered ---

def live_places() -> list[dict]:
    rows = _db().execute("SELECT * FROM places ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def _load_places(rows: list[dict]) -> None:
    """Replace the live places with an environment's. Same columns back."""
    with _lock():
        _db().execute("DELETE FROM places")
        for r in rows:
            columns = [c for c in r if c != "id"]
            _db().execute(
                f"INSERT INTO places ({', '.join(columns)}) VALUES "
                f"({', '.join('?' for _ in columns)})",
                [r[c] for c in columns])
        _db().commit()


# --- records --------------------------------------------------------------------

def _row(r) -> dict[str, Any]:
    return {"env_id": r["env_id"], "name": r["name"], "kind": r["kind"],
            "map_file": r["map_file"], "active": bool(r["active"]),
            "landmarks": json.loads(r["landmarks"] or "[]"),
            "obstacle_log": json.loads(r["obstacle_log"] or "{}"),
            "created_at": r["created_at"], "last_active": r["last_active"]}


def all_envs() -> list[dict]:
    ensure()
    return [_row(r) for r in _db().execute(
        "SELECT * FROM environments ORDER BY active DESC, last_active DESC")]


def active() -> dict | None:
    ensure()
    r = _db().execute("SELECT * FROM environments WHERE active=1").fetchone()
    return _row(r) if r else None


def by_name(name: str) -> dict | None:
    ensure()
    want = " ".join((name or "").lower().split())
    for env in all_envs():
        if env["name"].lower() == want:
            return env
    return None


def ensure() -> None:
    """Step 1's migration, once: whatever map and places exist become the
    first environment. Nothing is reset, moved or lost."""
    if _db().execute("SELECT COUNT(*) FROM environments").fetchone()[0]:
        return
    now = time.time()
    env_id = f"{_slug(config.ENV_FIRST_NAME)}-{int(now)}"
    with _lock():
        _db().execute(
            "INSERT INTO environments (env_id, name, kind, map_file, landmarks,"
            " active, created_at, last_active) VALUES (?,?,?,?,?,1,?,?)",
            (env_id, config.ENV_FIRST_NAME, KIND, str(config.SLAM_MAP_FILE),
             json.dumps([p["name"] for p in live_places()]), now, now))
        _db().commit()
    log.info("environment %r created from the existing map and %d places",
             config.ENV_FIRST_NAME, len(live_places()))


def _snapshot_live(env: dict) -> None:
    """Copy the live map and places into this environment's folder."""
    folder = _dir(env["env_id"])
    folder.mkdir(parents=True, exist_ok=True)
    live_map = Path(config.SLAM_MAP_FILE)
    if live_map.exists():
        shutil.copy2(live_map, folder / "map.npz")
    places = live_places()
    (folder / "places.json").write_text(json.dumps(places))
    with _lock():
        _db().execute("UPDATE environments SET landmarks=? WHERE env_id=?",
                      (json.dumps([p["name"] for p in places]), env["env_id"]))
        _db().commit()


def _activate(env_id: str) -> None:
    now = time.time()
    with _lock():
        _db().execute("UPDATE environments SET active=0")
        _db().execute("UPDATE environments SET active=1, last_active=? "
                      "WHERE env_id=?", (now, env_id))
        _db().commit()
    _write_status({"state": "matched", "env_id": env_id,
                   "why": "switched by Harsh"})


def switch_to(name: str) -> str:
    """His words: "we're at <name>". The live pair is saved first."""
    target = by_name(name)
    current = active()
    if target is None:
        known = ", ".join(e["name"] for e in all_envs())
        return f"I don't know a place called {name}. I know {known}."
    if current and target["env_id"] == current["env_id"]:
        _write_status({"state": "matched", "env_id": target["env_id"],
                       "why": "he said so"})
        return f"We're already in {target['name']}."
    if current:
        _snapshot_live(current)
    folder = _dir(target["env_id"])
    live_map = Path(config.SLAM_MAP_FILE)
    if (folder / "map.npz").exists():
        shutil.copy2(folder / "map.npz", live_map)
    elif live_map.exists():
        live_map.unlink()                    # it is safe in the old folder
    try:
        rows = json.loads((folder / "places.json").read_text())
    except (OSError, ValueError):
        rows = []
    _load_places(rows)
    _activate(target["env_id"])
    log.info("environment -> %s (%d places)", target["name"], len(rows))
    return f"Okay, we're in {target['name']}. I've loaded its map."


def create_new(name: str | None = None) -> str:
    """His words: "this is a new place". Starts an EMPTY map and no places;
    the old environment keeps everything in its folder."""
    current = active()
    base = name.strip() if name else f"Place {len(all_envs()) + 1}"
    if by_name(base):
        return f"There's already a place called {base}."
    if current:
        _snapshot_live(current)
    live_map = Path(config.SLAM_MAP_FILE)
    if live_map.exists():
        live_map.unlink()                    # copied into the old folder above
    _load_places([])
    now = time.time()
    env_id = f"{_slug(base)}-{int(now)}"
    with _lock():
        _db().execute(
            "INSERT INTO environments (env_id, name, kind, map_file, created_at,"
            " last_active) VALUES (?,?,?,?,?,?)",
            (env_id, base, KIND, str(config.SLAM_MAP_FILE), now, now))
        _db().commit()
    _activate(env_id)
    log.info("new environment %r (confirmed by Harsh)", base)
    # Its map will be built from where she stands: home is its origin.
    try:
        import home_dock
        home_dock.save_boot_home({"state": "no_map"})
    except Exception as exc:                        # noqa: BLE001
        log.warning("new environment without a home: %s", exc)
    return (f"Okay — {base} is a new place. I'll build its map from here; "
            f"{current['name'] if current else 'the old one'} is kept.")


# --- step 2: which environment is this? (at boot, no movement) -----------------

def status() -> dict:
    try:
        return json.loads(Path(config.ENV_STATUS_FILE).read_text())
    except (OSError, ValueError):
        return {}


def _write_status(data: dict) -> None:
    path = Path(config.ENV_STATUS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({**data, "at": time.time()}))
    tmp.replace(path)


def judge(fix) -> tuple[str, str]:
    """A global_localise.Fix -> ("matched" | "no_match", why).

    Any real fit is the same environment -- including an AMBIGUOUS one (two
    spots fit; that is still this map) and a map that only covers a little
    around here. Only no fit anywhere suggests somewhere else.
    """
    if fix.ok:
        return "matched", f"placed (score {fix.score:.2f})"
    if fix.candidate is not None and fix.score >= float(config.ENV_MATCH_MIN_SCORE):
        return "matched", f"fits, two look-alike spots (score {fix.score:.2f})"
    return "no_match", fix.why or f"no fit (score {fix.score:.2f})"


def boot_check(scan: dict[int, int] | None) -> dict:
    """Step 2. One sweep against the active map. Never moves, never creates."""
    env = active()
    result = {"state": "unknown", "env_id": env["env_id"] if env else None,
              "env_name": env["name"] if env else None}
    live_map = Path(config.SLAM_MAP_FILE)
    if not live_map.exists():
        result.update(state="no_map", why="this environment has no map yet")
    elif not scan:
        result.update(state="unknown", why="no LiDAR sweep")
    else:
        try:
            import global_localise
            from slam.grid import OccupancyGrid
            fix = global_localise.localise(OccupancyGrid.load(live_map), scan)
            state, why = judge(fix)
            result.update(state=state, why=why, score=round(fix.score, 3))
            # For home_dock: the pose, and whether it was UNAMBIGUOUS.
            if fix.pose is not None or fix.candidate is not None:
                result["pose"] = [round(float(v), 4)
                                  for v in (fix.pose or fix.candidate)]
            result["sure"] = bool(fix.ok)
        except Exception as exc:                    # noqa: BLE001
            result.update(state="unknown", why=f"{type(exc).__name__}: {exc}")
    if result["state"] == "no_match":
        others = [e["name"] for e in all_envs() if not e["active"]]
        result["ask"] = (
            f"I don't recognise this place from my map of {result['env_name']}."
            + (f" Are we at {', or '.join(others)}?" if others else "")
            + " Say: Friday, we're at, and the name, or: Friday, this is a new"
              " place.")
    _write_status(result)
    log.info("boot check: %s — %s", result["state"], result.get("why"))
    return result


# --- step 4: new areas and their names ------------------------------------------

def add_pending(x: float, y: float, area_m2: float, radius_m: float) -> dict:
    env = active()
    now = time.time()
    with _lock():
        cursor = _db().execute(
            "INSERT INTO pending_landmarks (env_id, x, y, area_m2, radius_m, "
            "created_at) VALUES (?,?,?,?,?,?)",
            (env["env_id"], x, y, area_m2, radius_m, now))
        _db().commit()
    log.info("new area in %s: %.1f m² at (%.2f, %.2f) — waiting for a name",
             env["name"], area_m2, x, y)
    return {"id": cursor.lastrowid, "x": x, "y": y, "area_m2": area_m2}


def pending(env_id: str | None = None) -> list[dict]:
    env_id = env_id or (active() or {}).get("env_id")
    rows = _db().execute("SELECT * FROM pending_landmarks WHERE env_id=? AND "
                         "status='pending' ORDER BY id", (env_id,)).fetchall()
    return [dict(r) for r in rows]


def mark_asked(pending_id: int) -> None:
    with _lock():
        _db().execute("UPDATE pending_landmarks SET asked_at=? WHERE id=?",
                      (time.time(), int(pending_id)))
        _db().commit()


_NAME_OK = re.compile(r"^[\w' -]+$", re.UNICODE)


def validate_name(name: str) -> tuple[str | None, str]:
    """(clean name, "") or (None, why). Non-empty, reasonable, a real name."""
    clean = " ".join((name or "").strip().strip(".!?").split())
    clean = re.sub(r"^(?:the\s+)?(?:area\s+)?(?:called\s+)?", "", clean,
                   flags=re.I) or clean
    if not clean:
        return None, "the name is empty"
    if len(clean) > int(config.ENV_NAME_MAX):
        return None, f"keep it under {config.ENV_NAME_MAX} characters"
    if not _NAME_OK.match(clean):
        return None, "letters, numbers, spaces, hyphens and apostrophes only"
    try:
        import landmarks
        if landmarks._is_no_name(clean):
            return None, "that isn't a name"
    except Exception:                               # noqa: BLE001
        pass
    return clean, ""


def name_pending(pending_id: int, name: str) -> str:
    """Name one pending area -- a landmark in the ACTIVE environment only."""
    clean, why = validate_name(name)
    if clean is None:
        return f"Not saved: {why}."
    row = _db().execute("SELECT * FROM pending_landmarks WHERE id=?",
                        (int(pending_id),)).fetchone()
    env = active()
    if row is None or row["status"] != "pending":
        return "That area isn't waiting for a name any more."
    if env is None or row["env_id"] != env["env_id"]:
        return "That area belongs to a different place; switch to it first."
    if any(p["name"] == clean.lower() for p in live_places()):
        return f"There's already a place called {clean} here."
    from slam import places                        # called, never changed
    places.remember_here(clean, (row["x"], row["y"], 0.0),
                         note="named from a discovered area",
                         radius_m=row["radius_m"])
    with _lock():
        _db().execute("UPDATE pending_landmarks SET status='named', name=?, "
                      "named_at=? WHERE id=?",
                      (clean.lower(), time.time(), int(pending_id)))
        _db().execute("UPDATE environments SET landmarks=? WHERE env_id=?",
                      (json.dumps([p["name"] for p in live_places()]),
                       env["env_id"]))
        _db().commit()
    try:
        import nav_state_machine as nsm
        nsm._bundle("a discovered area was named")  # file 14 keeps its bundle
    except Exception as exc:                        # noqa: BLE001
        log.debug("nav bundle not refreshed: %s", exc)
    log.info("named %r in %s", clean, env["name"])
    return f"Saved. That area is now {clean}."


# --- step 6: obstacles near named places ----------------------------------------

def _world_cells(grid, mask) -> set[tuple[int, int]]:
    import landmarks
    return landmarks.AreaWatch._world_cells(grid, mask)


def obstacles_near_places(before, after) -> list[dict]:
    """New real obstacles (despeckled, file 14's own filter) inside a named
    place's radius, ONE entry per place per map change, not per cell."""
    import path_planner
    new_cells = (_world_cells(after, path_planner.preprocess(after).grid
                              .occupied_mask())
                 - (_world_cells(before, path_planner.preprocess(before).grid
                                 .occupied_mask()) if before is not None
                    else set()))
    if not new_cells:
        return []
    cell = after.cell
    pts = np.array([((x + 0.5) * cell, (y + 0.5) * cell) for x, y in new_cells])
    rows = live_places()
    if not rows:
        return []
    # Each new cell belongs to the NEAREST place whose radius holds it, so an
    # obstacle where two places overlap is counted once, not twice.
    centres = np.array([(p["x"], p["y"]) for p in rows])
    radii = np.array([float(p.get("radius_m") or config.PLACE_RADIUS_M)
                      for p in rows])
    dist = np.hypot(pts[:, None, 0] - centres[None, :, 0],
                    pts[:, None, 1] - centres[None, :, 1])
    dist[dist > radii[None, :]] = np.inf
    owner = np.argmin(dist, axis=1)
    held = np.isfinite(dist[np.arange(len(pts)), owner])
    label = _classification()
    found = []
    for index, place in enumerate(rows):
        inside = pts[held & (owner == index)]
        if len(inside) >= int(config.ENV_OBSTACLE_MIN_CELLS):
            found.append({"place": place["name"], "at": time.time(),
                          "x": round(float(inside[:, 0].mean()), 2),
                          "y": round(float(inside[:, 1].mean()), 2),
                          "cells": int(len(inside)), "class": label})
    return found


def _classification() -> str:
    """The scene watcher's biggest named thing, if fresh -- else unclassified."""
    try:
        scene = json.loads(Path(config.SCENE_JSON).read_text())
        if time.time() - float(scene.get("taken_at") or 0) <= 20:
            labels = [o["label"] for o in scene.get("objects") or []
                      if o.get("label") != "person"
                      and float(o.get("score") or 0) >= 0.6]
            if labels:
                return labels[0]
    except (OSError, ValueError, KeyError):
        pass
    return "unclassified"


def log_obstacles(entries: list[dict]) -> None:
    """Bounded: the last ENV_OBSTACLE_KEEP per place (file 09's approach)."""
    env = active()
    if not entries or env is None:
        return
    book = env["obstacle_log"]
    keep = int(config.ENV_OBSTACLE_KEEP)
    for e in entries:
        book.setdefault(e["place"], []).append(e)
        book[e["place"]] = book[e["place"]][-keep:]
    with _lock():
        _db().execute("UPDATE environments SET obstacle_log=? WHERE env_id=?",
                      (json.dumps(book), env["env_id"]))
        _db().commit()


def known_area_m2(path: Path) -> float | None:
    """Mapped (known free or occupied) area of a saved grid, in m²."""
    try:
        from slam.grid import OccupancyGrid
        grid = OccupancyGrid.load(path)
    except Exception:                               # noqa: BLE001
        return None
    known = grid.free_mask() | grid.occupied_mask()
    return float(known.sum()) * grid.cell ** 2


def panel() -> dict:
    """Everything the Environments panel shows. Never raises."""
    try:
        env = active()
        return {
            "environments": [{"env_id": e["env_id"], "name": e["name"],
                              "active": e["active"],
                              "landmarks": len(e["landmarks"]),
                              "last_active": e["last_active"]}
                             for e in all_envs()],
            "active": env["name"] if env else None,
            "status": status(),
            "pending": [{"id": p["id"], "area_m2": round(p["area_m2"], 1),
                         "x": round(p["x"], 2), "y": round(p["y"], 2),
                         "created_at": p["created_at"]} for p in pending()],
            "places": [p["name"] for p in live_places()],
            "obstacles": {place: {"count": len(items),
                                  "last": items[-1] if items else None}
                          for place, items in (env or {}).get(
                              "obstacle_log", {}).items()},
        }
    except Exception as exc:                        # noqa: BLE001
        return {"error": str(exc)}
