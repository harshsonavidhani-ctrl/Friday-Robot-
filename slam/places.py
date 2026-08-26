"""places.py — step 9 section 5: the map, but in words.

    places.remember_here("the kitchen", pose)      name where you are
    places.nearest(pose)                           what am I standing in
    places.find("kitchen")                         where is it, to drive to
    places.all()                                   everything known

WHY A SEPARATE TABLE AND NOT A FACT. memory.remember() stores sentences and
recalls them by text similarity, which is right for "Harsh likes his tea
strong" and useless for "where is the kitchen". A place needs coordinates in
the map's frame so the planner can be pointed at it, a radius so "am I there
yet" has an answer, and a name that is unique so naming somewhere twice
corrects it instead of creating a second one. That is a table, not a sentence.

⚠ THE MAP'S FRAME IS ARBITRARY AND IT MOVES. Coordinates here mean nothing
except relative to the current map -- the origin is wherever Friday happened
to start, and loop closure rewrites poses when it fires. So:

  * A place is stored with the map generation it was named in. If the map is
    rebuilt from scratch, old places are stale and say so rather than sending
    the robot confidently to a spot that no longer means anything.
  * Places are NOT a substitute for looking. "Go to the kitchen" plans a route
    to a remembered point; it does not assume the kitchen is still empty.

Nothing here drives, looks, or speaks. It is a store with coordinates.
"""

from __future__ import annotations

import math
import time

import config
import memory


def _now() -> float:
    return time.time()


def remember_here(name: str, pose, kind: str = "place",
                  note: str | None = None,
                  radius_m: float | None = None) -> dict:
    """Name the spot Friday is standing on. Re-naming MOVES it, not duplicates.

    Somewhere named twice is almost always a correction -- the first attempt
    was taken a metre short of the doorway, or before the map settled. Adding
    a second "the kitchen" a metre from the first would make `find` a coin
    toss, so the name is unique and the newer position wins.
    """
    name = " ".join(name.strip().lower().split())
    if not name:
        raise ValueError("a place needs a name")
    radius = config.PLACE_RADIUS_M if radius_m is None else radius_m
    now = _now()
    with memory._lock:                      # same lock every writer uses
        cur = memory.db().execute(
            "SELECT id, visits FROM places WHERE name = ?", (name,)).fetchone()
        if cur:
            memory.db().execute(
                "UPDATE places SET x=?, y=?, radius_m=?, kind=?, note=?, "
                "last_seen=? WHERE id=?",
                (float(pose[0]), float(pose[1]), radius, kind, note, now,
                 cur["id"]))
            place_id = cur["id"]
        else:
            cur = memory.db().execute(
                "INSERT INTO places (name, x, y, radius_m, kind, note, "
                "created_at, last_seen, visits) VALUES (?,?,?,?,?,?,?,?,0)",
                (name, float(pose[0]), float(pose[1]), radius, kind, note,
                 now, now))
            place_id = cur.lastrowid
        memory.db().commit()
    return {"id": place_id, "name": name, "x": float(pose[0]),
            "y": float(pose[1]), "radius_m": radius, "kind": kind}


def _row(row) -> dict:
    return {"id": row["id"], "name": row["name"], "x": row["x"],
            "y": row["y"], "radius_m": row["radius_m"], "kind": row["kind"],
            "note": row["note"], "visits": row["visits"],
            "last_seen": row["last_seen"]}


def all() -> list[dict]:                    # noqa: A001 - reads well in use
    rows = memory.db().execute(
        "SELECT * FROM places ORDER BY last_seen DESC").fetchall()
    return [_row(r) for r in rows]


def find(query: str) -> dict | None:
    """A place by name. Exact first, then 'the' dropped, then substring.

    Deliberately not fuzzy-scored: sending a robot several metres across a
    flat on a near-miss string match is worse than saying it does not know.
    """
    q = " ".join(query.strip().lower().split())
    if not q:
        return None
    rows = all()
    for row in rows:
        if row["name"] == q:
            return row
    bare = q[4:] if q.startswith("the ") else q
    for row in rows:
        if row["name"] in (bare, f"the {bare}"):
            return row
    hits = [r for r in rows if bare in r["name"]]
    return hits[0] if len(hits) == 1 else None


def nearest(pose, within_m: float | None = None) -> dict | None:
    """Which known place is Friday in or beside, if any."""
    best, best_gap = None, math.inf
    for row in all():
        gap = math.hypot(row["x"] - pose[0], row["y"] - pose[1])
        limit = row["radius_m"] if within_m is None else within_m
        if gap <= limit and gap < best_gap:
            best, best_gap = row, gap
    if best:
        best = dict(best, distance_m=round(best_gap, 2))
    return best


def visited(name: str) -> None:
    """Count a visit, so a patrol can prefer somewhere not seen lately."""
    with memory._lock:
        memory.db().execute(
            "UPDATE places SET visits = visits + 1, last_seen = ? "
            "WHERE name = ?", (_now(), name))
        memory.db().commit()


def forget(name: str) -> bool:
    with memory._lock:
        cur = memory.db().execute("DELETE FROM places WHERE name = ?",
                                  (name.strip().lower(),))
        memory.db().commit()
    return cur.rowcount > 0
