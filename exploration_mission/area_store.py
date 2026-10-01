"""Everything the exploration mission keeps (File 21 step 2, spec 3.5 / 3.6).

    store = AreaStore(places=adapters.places)      # memory.db + state/exploration_mission
    store.ensure_schema()                          # backs memory.db up first, once
    area = store.add_area("home-1790517173", map_id, polygon=[...], center=(x, y),
                          area_m2=9.4, name="kitchen", source="voice",
                          mission_id=m)            # saved THE MOMENT it is named

WHERE IT LIVES. The same backend as Friday's memory: three new tables in
memory.db (areas, home_anchor, missions, plus a one-row meta table for the
schema version), written through this module's OWN connection with
`PRAGMA synchronous=FULL` so a committed area survives a power cut. The DB's
journal mode (WAL, set by memory.py) is never touched. Before the tables are
first created in a DB that already holds Friday's memory, the DB is copied
with sqlite's backup() -- the way File 17 did before its one schema change.
The mission checkpoint, the full breadcrumb trails and forgotten.jsonl are
files under state/exploration_mission/envs/<env_id>/ (the spec's
data/exploration/<env_id>/; Friday keeps runtime data in state/), written by
durable.py: atomic, with a .bak of the previous good copy.

NOTHING IS DELETED AUTOMATICALLY. An area goes away only through `forget()`
(Harsh's voice, after a yes) and even then its row is appended to
forgotten.jsonl FIRST. A map reset makes areas `stale`, never gone.

NAMES. Lowercase, trimmed, a leading the/my/our/a/an dropped, spaces
collapsed; unique per environment. `display_name` keeps Harsh's words
("Harsh's room"). An area nobody named is saved as "area N" (status
unnamed) and waits in the naming queue.

FILE 14's LANDMARKS. A NAMED area in the saved map's frame is also a row in
File 14's places table (kind "area", note "[map:<id>] area:<uuid>"), so
"Friday, go to the kitchen" works the moment it is saved. Unnamed areas and
areas in a mission's own frame (not placed on the saved map) are never
landmarks: a landmark in the wrong frame sends her somewhere else. A place
Harsh saved himself with the same name is the same place when it lies inside
the area's polygon -- linked, never moved.

Only adapters.py imports Friday's code; this module gets File 14's places
through the `places` adapter it is handed.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
import re
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from exploration_mission import durable

log = logging.getLogger("friday.exploration_mission.area_store")

SCHEMA_VERSION = 1
META_TABLE = "exploration_mission_meta"
FRAME_MAP = "map"                       # the saved SLAM map, File 14's frame
STATUSES = ("named", "unnamed", "stale")
MISSION_STATUSES = ("running", "returned", "help_wait", "failsafe",
                    "interrupted", "aborted")
STOP_REASONS = ("no_frontiers", "time_cap", "voice_stop", "power_low", "failsafe")

# spec 3.5 / 4.7: a leading article or possessive pronoun is not the name.
LEADING = ("the", "my", "our", "a", "an")
# spec 4.7: the WHOLE name may not be one of these ("home office" is fine).
RESERVED = frozenset({
    "stop", "cancel", "yes", "no", "haan", "nahi", "skip", "save", "friday",
    "explore", "home", "help", "continue",
    "what", "why", "who", "how", "huh", "sorry", "kya", "kaun"})
# LATIN SCRIPT ONLY. The transcript comes from faster-whisper with language "en"
# (spec 4.7: "so Hinglish comes out in Latin script"), so a name in any other
# script is not something Harsh said -- it is one of whisper's hallucinations
# on noise (14 of them in his own turn log, 2026-09-30).
_ALLOWED = re.compile(r"^[a-z0-9]+(?:[ '\-][a-z0-9]+|')*$")
_UNNAMED = re.compile(r"^area (\d+)$")

SCHEMA_V1 = f"""
CREATE TABLE IF NOT EXISTS areas (
    id               TEXT PRIMARY KEY,
    env_id           TEXT NOT NULL,
    map_id           TEXT,
    frame            TEXT NOT NULL DEFAULT '{FRAME_MAP}',
    name             TEXT,
    display_name     TEXT,
    aliases          TEXT NOT NULL DEFAULT '[]',
    status           TEXT NOT NULL,
    complete         INTEGER NOT NULL DEFAULT 1,
    center_x         REAL NOT NULL,
    center_y         REAL NOT NULL,
    radius_m         REAL NOT NULL,
    polygon          TEXT NOT NULL,
    area_m2          REAL NOT NULL,
    hints            TEXT NOT NULL DEFAULT '{{}}',
    source           TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    first_mission_id TEXT,
    visit_count      INTEGER NOT NULL DEFAULT 0,
    obstacle_events  INTEGER NOT NULL DEFAULT 0,
    schema_version   INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS areas_env_name ON areas(env_id, name)
    WHERE name IS NOT NULL;
CREATE INDEX IF NOT EXISTS areas_env ON areas(env_id);
CREATE TABLE IF NOT EXISTS home_anchor (
    env_id         TEXT PRIMARY KEY,
    x              REAL NOT NULL,
    y              REAL NOT NULL,
    theta          REAL NOT NULL,
    signature      TEXT,
    map_id         TEXT,
    frame          TEXT NOT NULL DEFAULT '{FRAME_MAP}',
    set_at         TEXT NOT NULL,
    set_by         TEXT NOT NULL,
    schema_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS missions (
    id                TEXT PRIMARY KEY,
    env_id            TEXT NOT NULL,
    map_id            TEXT,
    started_at        TEXT NOT NULL,
    ended_at          TEXT,
    start_x           REAL,
    start_y           REAL,
    start_theta       REAL,
    start_signature   TEXT,
    end_x             REAL,
    end_y             REAL,
    home_error_m      REAL,
    status            TEXT NOT NULL,
    stop_reason       TEXT,
    explored_m2_start REAL,
    explored_m2_end   REAL,
    distance_m        REAL,
    areas_new         TEXT NOT NULL DEFAULT '[]',
    areas_updated     TEXT NOT NULL DEFAULT '[]',
    rth_attempts      INTEGER NOT NULL DEFAULT 0,
    rungs_used        TEXT NOT NULL DEFAULT '[]',
    events            TEXT NOT NULL DEFAULT '[]',
    schema_version    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS missions_env ON missions(env_id, started_at);
CREATE TABLE IF NOT EXISTS {META_TABLE} (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Each step takes the DB from version k to k+1. v0 -> v1 creates everything.
SQL_MIGRATIONS: dict[int, str] = {0: SCHEMA_V1}

_AREA_JSON = ("aliases", "polygon", "hints")
_MISSION_JSON = ("start_signature", "areas_new", "areas_updated", "rungs_used",
                 "events")
_MISSION_FIELDS = frozenset({
    "map_id", "ended_at", "start_x", "start_y", "start_theta", "start_signature",
    "end_x", "end_y", "home_error_m", "status", "stop_reason",
    "explored_m2_start", "explored_m2_end", "distance_m", "areas_new",
    "areas_updated", "rth_attempts", "rungs_used", "events"})


class StoreError(ValueError):
    pass


class InvalidName(StoreError):
    pass


class DuplicateName(StoreError):
    def __init__(self, name: str, other: dict | None, suggestion: str):
        super().__init__(f"{name!r} is already somewhere else")
        self.name, self.other, self.suggestion = name, other, suggestion


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# =============================================================================
# names
# =============================================================================

def _words(text: str) -> list[str]:
    text = str(text).replace("’", "'").strip()
    text = text.strip(" \t.,!?;:\"()[]")
    return text.split()


def clean_display(text: str) -> str:
    """Harsh's words, tidied: surrounding punctuation off, spaces collapsed, a
    leading the/my/our/a/an dropped (never the last word). Case kept."""
    words = _words(text)
    while len(words) > 1 and words[0].lower() in LEADING:
        words.pop(0)
    return " ".join(words)


def normalize_name(text: str) -> str:
    """The stored, compared form: clean_display, lowercased."""
    return clean_display(text).lower()


def validate_name(text: str, max_words: int = 4,
                  max_chars: int = 30) -> tuple[str, str]:
    """(name, display_name), or InvalidName saying why (spec 4.7)."""
    display = clean_display(text)
    name = display.lower()
    if not name:
        raise InvalidName("no name")
    if len(name.split()) > max_words:
        raise InvalidName(f"more than {max_words} words")
    if len(name) > max_chars:
        raise InvalidName(f"longer than {max_chars} characters")
    if not any(ch.isalpha() for ch in name):
        raise InvalidName("no letters")
    if not _ALLOWED.match(name):
        raise InvalidName("only a-z, digits, spaces, ' and -")
    if name in RESERVED or name in LEADING:
        raise InvalidName(f"{name!r} is a reserved word")
    return name, display


def unnamed_number(name: str | None) -> int | None:
    m = _UNNAMED.match(name or "")
    return int(m.group(1)) if m else None


def point_in_polygon(x: float, y: float, polygon: Iterable) -> bool:
    """Even-odd ray cast. Points exactly on an edge may land either side."""
    pts = [(float(p[0]), float(p[1])) for p in polygon]
    inside = False
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[i - 1]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


@dataclass
class NameCheck:
    """What a proposed name would mean for this area (spec 4.7 point 8)."""
    kind: str                      # ok | same_place | duplicate_area |
    name: str = ""                 #   duplicate_landmark | invalid
    display: str = ""
    other: dict | None = None      # the area or landmark it collides with
    suggestion: str = ""           # "kitchen 2"
    why: str = ""


# =============================================================================
# the store
# =============================================================================

@dataclass
class _Config:
    events_cap: int = 500
    max_name_words: int = 4
    max_name_chars: int = 30
    checkpoint_keep: int = 500


class AreaStore:
    def __init__(self, db_path: Path | str | None = None,
                 data_dir: Path | str | None = None, places=None,
                 events_cap: int = 500, max_name_words: int = 4,
                 max_name_chars: int = 30, checkpoint_keep: int = 500,
                 backup_before_create: bool = True):
        if db_path is None or data_dir is None:
            import config                              # Friday's paths only
            db_path = db_path or config.MEMORY_DB
            data_dir = data_dir or (Path(config.ROOT) / "state"
                                    / "exploration_mission")
        self.db_path = Path(db_path)
        self.data_dir = Path(data_dir)
        self.places = places
        self.cfg = _Config(events_cap, max_name_words, max_name_chars,
                           checkpoint_keep)
        self.backup_before_create = backup_before_create
        self._lock = threading.RLock()
        self._con: sqlite3.Connection | None = None
        self.backup_file: Path | None = None

    # --- connection -----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        if self._con is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(str(self.db_path), timeout=15.0,
                                  isolation_level=None, check_same_thread=False)
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA synchronous=FULL")
            con.execute("PRAGMA busy_timeout=15000")
            self._con = con
        return self._con

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None

    @contextmanager
    def _tx(self):
        """One transaction: BEGIN IMMEDIATE ... COMMIT, rolled back on error."""
        with self._lock:
            con = self._connect()
            con.execute("BEGIN IMMEDIATE")
            try:
                yield con
            except BaseException:
                con.execute("ROLLBACK")
                raise
            con.execute("COMMIT")

    # --- schema -----------------------------------------------------------------
    def schema_version(self) -> int:
        con = self._connect()
        has_meta = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (META_TABLE,)).fetchone()
        if not has_meta:
            return 0
        row = con.execute(f"SELECT value FROM {META_TABLE} WHERE key="
                          "'schema_version'").fetchone()
        return int(row[0]) if row else 0

    def ensure_schema(self) -> int:
        """Create or migrate the tables. Returns the version now in the DB."""
        with self._lock:
            current = self.schema_version()
            if current > SCHEMA_VERSION:
                raise StoreError(f"memory.db has exploration schema v{current}; "
                                 f"this code knows v{SCHEMA_VERSION}")
            if current == SCHEMA_VERSION:
                return current
            if current == 0 and self.backup_before_create:
                self._backup()
            for step in range(current, SCHEMA_VERSION):
                with self._tx() as con:
                    for statement in SQL_MIGRATIONS[step].split(";"):
                        if statement.strip():
                            con.execute(statement)
                    con.execute(f"INSERT OR REPLACE INTO {META_TABLE} (key, value)"
                                " VALUES ('schema_version', ?)", (str(step + 1),))
                log.info("exploration tables migrated v%d -> v%d", step, step + 1)
            return SCHEMA_VERSION

    def _backup(self) -> None:
        """Copy the whole DB (sqlite backup(), WAL-safe) before first use --
        only when it already holds other tables, i.e. Friday's memory."""
        con = self._connect()
        others = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                             "type='table'").fetchone()[0]
        if not others:
            return
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.db_path.with_name(f"{self.db_path.name}.bak-{stamp}-file21")
        dest = sqlite3.connect(str(target))
        try:
            con.backup(dest)
        finally:
            dest.close()
        self.backup_file = target
        log.warning("memory.db backed up to %s before the exploration tables "
                    "were added", target.name)

    # --- rows -------------------------------------------------------------------
    @staticmethod
    def _area(row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for k in _AREA_JSON:
            d[k] = json.loads(d[k]) if d[k] is not None else None
        d["complete"] = bool(d["complete"])
        return d

    @staticmethod
    def _mission(row) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        for k in _MISSION_JSON:
            d[k] = json.loads(d[k]) if d[k] is not None else None
        return d

    def get(self, area_id: str) -> dict | None:
        return self._area(self._connect().execute(
            "SELECT * FROM areas WHERE id=?", (area_id,)).fetchone())

    def list_areas(self, env_id: str, statuses: Iterable[str] | None = None
                   ) -> list[dict]:
        rows = self._connect().execute(
            "SELECT * FROM areas WHERE env_id=? ORDER BY rowid",
            (env_id,)).fetchall()
        out = [self._area(r) for r in rows]
        if statuses is not None:
            keep = set(statuses)
            out = [a for a in out if a["status"] in keep]
        return out

    def by_name(self, env_id: str, name: str) -> dict | None:
        """Exact name, or an exact alias. Never a substring (File 14's rule:
        a near miss sends her metres to the wrong room)."""
        want = normalize_name(name)
        if not want:
            return None
        row = self._connect().execute(
            "SELECT * FROM areas WHERE env_id=? AND name=?", (env_id, want)
        ).fetchone()
        if row is not None:
            return self._area(row)
        for area in self.list_areas(env_id):
            if want in (area["aliases"] or []):
                return area
        return None

    def naming_queue(self, env_id: str) -> list[dict]:
        """Unnamed and stale areas: what the dashboard offers to (re)name."""
        return self.list_areas(env_id, statuses=("unnamed", "stale"))

    # --- names ------------------------------------------------------------------
    def next_unnamed_name(self, env_id: str) -> str:
        used = {unnamed_number(a["name"]) for a in self.list_areas(env_id)}
        n = 1
        while n in used:
            n += 1
        return f"area {n}"

    def next_free_name(self, env_id: str, name: str) -> str:
        """"kitchen" -> "kitchen 2" (or 3, ...), free in areas AND landmarks."""
        n = 2
        while (self.by_name(env_id, f"{name} {n}") is not None
               or self._landmark(f"{name} {n}") is not None):
            n += 1
        return f"{name} {n}"

    def _landmark(self, name: str) -> dict | None:
        if self.places is None:
            return None
        try:
            return self.places.find_exact(name)
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not read File 14's places: %s", exc)
            return None

    def _owns(self, place: dict | None, area_id: str) -> bool:
        return bool(place) and f"area:{area_id}" in (place.get("note") or "")

    def check_name(self, env_id: str, text: str, polygon=None,
                   exclude_id: str | None = None) -> NameCheck:
        try:
            name, display = validate_name(text, self.cfg.max_name_words,
                                          self.cfg.max_name_chars)
        except InvalidName as exc:
            return NameCheck("invalid", why=str(exc))
        other = self.by_name(env_id, name)
        if other is not None and other["id"] != exclude_id:
            return NameCheck("duplicate_area", name, display, other,
                             self.next_free_name(env_id, name))
        place = self._landmark(name)
        if place is not None and not (exclude_id and self._owns(place, exclude_id)):
            if polygon and point_in_polygon(place["x"], place["y"], polygon):
                return NameCheck("same_place", name, display, place)
            return NameCheck("duplicate_landmark", name, display, place,
                             self.next_free_name(env_id, name))
        return NameCheck("ok", name, display)

    # --- areas: write -----------------------------------------------------------
    def add_area(self, env_id: str, map_id: str | None, *, polygon, center,
                 area_m2: float, radius_m: float | None = None,
                 name: str | None = None, source: str = "voice",
                 mission_id: str | None = None, complete: bool = True,
                 hints: dict | None = None, frame: str = FRAME_MAP) -> dict:
        """Save an area NOW (spec 3.6: never batched). name=None -> "area N",
        unnamed. A name taken by another area or by a File 14 landmark
        elsewhere raises DuplicateName; one Harsh saved INSIDE this polygon is
        linked as the same place."""
        area_id = uuid.uuid4().hex
        polygon = [[float(p[0]), float(p[1])] for p in polygon]
        radius = (float(radius_m) if radius_m is not None
                  else clamp(math.sqrt(max(area_m2, 0.0) / math.pi), 0.5, 2.0))
        hints = dict(hints or {})
        with self._lock:
            if name is None:
                status, nm = "unnamed", self.next_unnamed_name(env_id)
                display = nm
            else:
                chk = self.check_name(env_id, name, polygon)
                if chk.kind == "invalid":
                    raise InvalidName(chk.why)
                if chk.kind in ("duplicate_area", "duplicate_landmark"):
                    raise DuplicateName(chk.name, chk.other, chk.suggestion)
                if chk.kind == "same_place":
                    hints["linked_place"] = chk.other["name"]
                status, nm, display = "named", chk.name, chk.display
            stamp = now_iso()
            with self._tx() as con:
                con.execute(
                    "INSERT INTO areas (id, env_id, map_id, frame, name, "
                    "display_name, aliases, status, complete, center_x, center_y,"
                    " radius_m, polygon, area_m2, hints, source, created_at, "
                    "updated_at, first_mission_id, schema_version) VALUES "
                    "(?,?,?,?,?,?,'[]',?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (area_id, env_id, map_id, frame, nm, display, status,
                     int(bool(complete)), float(center[0]), float(center[1]),
                     radius, json.dumps(polygon), float(area_m2),
                     json.dumps(hints), source, stamp, stamp, mission_id,
                     SCHEMA_VERSION))
                if mission_id is not None:
                    self._append_json(con, "missions", mission_id, "areas_new",
                                      area_id)
            area = self.get(area_id)
        self._sync_landmark(area)
        log.info("area saved: %s (%s, %.1f m2, %s)", display, status, area_m2,
                 source)
        return area

    def rename(self, area_id: str, text: str, source: str = "voice") -> dict:
        with self._lock:
            area = self.get(area_id)
            if area is None:
                raise StoreError("no such area")
            chk = self.check_name(area["env_id"], text, area["polygon"],
                                  exclude_id=area_id)
            if chk.kind == "invalid":
                raise InvalidName(chk.why)
            if chk.kind in ("duplicate_area", "duplicate_landmark"):
                raise DuplicateName(chk.name, chk.other, chk.suggestion)
            hints = dict(area["hints"] or {})
            if chk.kind == "same_place":
                hints["linked_place"] = chk.other["name"]
            old_name = area["name"]
            new_status = "stale" if area["status"] == "stale" else "named"
            with self._tx() as con:
                con.execute(
                    "UPDATE areas SET name=?, display_name=?, status=?, hints=?, "
                    "source=?, updated_at=? WHERE id=?",
                    (chk.name, chk.display, new_status, json.dumps(hints), source,
                     now_iso(), area_id))
            area = self.get(area_id)
        self._drop_own_landmark(old_name, area_id)
        self._sync_landmark(area)
        log.info("area renamed: %s -> %s", old_name, area["name"])
        return area

    def forget(self, area_id: str, why: str = "voice") -> dict | None:
        """Harsh's forget, after his yes: forgotten.jsonl FIRST, then the row,
        then its landmark (ours, or the one linked as the same place)."""
        with self._lock:
            area = self.get(area_id)
            if area is None:
                return None
            place = None
            for nm in {area["name"], (area["hints"] or {}).get("linked_place")}:
                row = self._landmark(nm) if nm else None
                if row and (self._owns(row, area_id)
                            or nm == (area["hints"] or {}).get("linked_place")):
                    place = row
            durable.append_jsonl(self.env_dir(area["env_id"]) / "forgotten.jsonl",
                                 {"forgotten_at": now_iso(), "why": why,
                                  "area": area, "landmark": place})
            with self._tx() as con:
                con.execute("DELETE FROM areas WHERE id=?", (area_id,))
        if place is not None and self.places is not None:
            try:
                self.places.remove(place["name"])
            except Exception as exc:                    # noqa: BLE001
                log.warning("area forgotten but landmark %r not removed: %s",
                            place["name"], exc)
        log.info("area forgotten: %s", area["name"])
        return area

    def update_geometry(self, area_id: str, *, polygon, center, area_m2: float,
                        radius_m: float | None = None,
                        map_id: str | None = None,
                        mission_id: str | None = None) -> dict:
        """The growing-map rule's write (spec 3.5): same name, new shape.
        Our own landmark follows the centre; a linked one of Harsh's never
        moves without his confirmation."""
        radius = (float(radius_m) if radius_m is not None
                  else clamp(math.sqrt(max(area_m2, 0.0) / math.pi), 0.5, 2.0))
        with self._lock:
            area = self.get(area_id)
            if area is None:
                raise StoreError("no such area")
            with self._tx() as con:
                con.execute(
                    "UPDATE areas SET polygon=?, center_x=?, center_y=?, area_m2=?,"
                    " radius_m=?, map_id=COALESCE(?, map_id), updated_at=? "
                    "WHERE id=?",
                    (json.dumps([[float(p[0]), float(p[1])] for p in polygon]),
                     float(center[0]), float(center[1]), float(area_m2), radius,
                     map_id, now_iso(), area_id))
                if mission_id is not None:
                    self._append_json(con, "missions", mission_id,
                                      "areas_updated", area_id)
            area = self.get(area_id)
        self._sync_landmark(area)
        return area

    def reframe(self, area_id: str, *, polygon, center, frame: str,
                map_id: str | None) -> dict:
        """Move an area into another frame (a mission's own -> the saved map's),
        once the mission has been fitted onto the saved map. Same area, same
        name; a named one becomes a File 14 landmark from here on."""
        with self._lock:
            if self.get(area_id) is None:
                raise StoreError("no such area")
            with self._tx() as con:
                con.execute(
                    "UPDATE areas SET polygon=?, center_x=?, center_y=?, frame=?, "
                    "map_id=?, updated_at=? WHERE id=?",
                    (json.dumps([[float(p[0]), float(p[1])] for p in polygon]),
                     float(center[0]), float(center[1]), frame, map_id, now_iso(),
                     area_id))
            area = self.get(area_id)
        self._sync_landmark(area)
        return area

    def add_alias(self, area_id: str, alias: str) -> dict:
        alias = normalize_name(alias)
        with self._tx() as con:
            row = con.execute("SELECT aliases FROM areas WHERE id=?",
                              (area_id,)).fetchone()
            if row is None:
                raise StoreError("no such area")
            aliases = json.loads(row[0])
            if alias and alias not in aliases:
                aliases.append(alias)
            con.execute("UPDATE areas SET aliases=?, updated_at=? WHERE id=?",
                        (json.dumps(aliases), now_iso(), area_id))
        return self.get(area_id)

    def count_visit(self, area_id: str) -> None:
        with self._tx() as con:
            con.execute("UPDATE areas SET visit_count=visit_count+1 WHERE id=?",
                        (area_id,))

    def count_obstacle(self, area_id: str) -> None:
        with self._tx() as con:
            con.execute("UPDATE areas SET obstacle_events=obstacle_events+1 "
                        "WHERE id=?", (area_id,))

    def mark_stale(self, env_id: str, current_map_id: str | None) -> list[str]:
        """spec 3.5 map_id guard: areas saved on a different map become stale.
        Kept, name and all -- never deleted. Returns the ids newly stale, so
        the caller says "My map was reset..." once and only when there are."""
        if not current_map_id:
            return []
        with self._tx() as con:
            rows = con.execute(
                "SELECT id FROM areas WHERE env_id=? AND frame=? AND status!='stale'"
                " AND map_id IS NOT NULL AND map_id!=?",
                (env_id, FRAME_MAP, current_map_id)).fetchall()
            ids = [r[0] for r in rows]
            for area_id in ids:
                con.execute("UPDATE areas SET status='stale', updated_at=? "
                            "WHERE id=?", (now_iso(), area_id))
        if ids:
            log.warning("%d area(s) are stale: map %s is not the map they were "
                        "saved on", len(ids), current_map_id)
        return ids

    # --- File 14 landmark sync --------------------------------------------------
    def _syncable(self, area: dict | None) -> bool:
        return bool(area) and area["status"] == "named" \
            and area["frame"] == FRAME_MAP and self.places is not None

    def _sync_landmark(self, area: dict | None) -> None:
        """Named + saved-map frame -> an upsert of our own places row. Never
        raises: the area is already saved; resync_landmarks() repairs later."""
        if not self._syncable(area):
            return
        linked = (area["hints"] or {}).get("linked_place")
        if linked:
            return                         # Harsh's own place IS this place
        try:
            existing = self.places.find_exact(area["name"])
            if existing is not None and not self._owns(existing, area["id"]):
                log.warning("landmark %r belongs to someone else; not moved",
                            area["name"])
                return
            self.places.upsert(area["name"], area["center_x"], area["center_y"],
                               area["radius_m"],
                               self.places.stamp(area["map_id"],
                                                 f"area:{area['id']}"))
        except Exception as exc:                        # noqa: BLE001
            log.warning("landmark for %r not synced: %s", area["name"], exc)

    def _drop_own_landmark(self, name: str | None, area_id: str) -> None:
        if not name or self.places is None:
            return
        try:
            row = self.places.find_exact(name)
            if self._owns(row, area_id):
                self.places.remove(name)
        except Exception as exc:                        # noqa: BLE001
            log.warning("old landmark %r not removed: %s", name, exc)

    def resync_landmarks(self, env_id: str) -> int:
        """Boot-time repair: every named, saved-frame area has its landmark
        (a power cut between the commit and the upsert leaves one missing)."""
        n = 0
        for area in self.list_areas(env_id, statuses=("named",)):
            if not self._syncable(area) or (area["hints"] or {}).get("linked_place"):
                continue
            row = self._landmark(area["name"])
            if row is None or (self._owns(row, area["id"]) and (
                    abs(row["x"] - area["center_x"]) > 1e-6
                    or abs(row["y"] - area["center_y"]) > 1e-6)):
                self._sync_landmark(area)
                n += 1
        return n

    # --- home anchor ------------------------------------------------------------
    def set_home_anchor(self, env_id: str, x: float, y: float, theta: float,
                        signature: Iterable | None, map_id: str | None,
                        set_by: str = "auto", frame: str = FRAME_MAP) -> dict:
        """The mission start anchor (spec 4.9). NOT File 10's permanent `home`
        place, which this never touches."""
        if set_by not in ("auto", "voice"):
            raise StoreError("set_by is auto or voice")
        sig = None if signature is None else _encode_signature(signature)
        with self._tx() as con:
            con.execute(
                "INSERT OR REPLACE INTO home_anchor (env_id, x, y, theta, "
                "signature, map_id, frame, set_at, set_by, schema_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (env_id, float(x), float(y), float(theta), sig, map_id, frame,
                 now_iso(), set_by, SCHEMA_VERSION))
        return self.home_anchor(env_id)

    def home_anchor(self, env_id: str) -> dict | None:
        row = self._connect().execute("SELECT * FROM home_anchor WHERE env_id=?",
                                      (env_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["signature"] = (None if d["signature"] is None
                          else _decode_signature(d["signature"]))
        return d

    # --- missions ---------------------------------------------------------------
    def start_mission(self, env_id: str, map_id: str | None, start_pose,
                      start_signature: Iterable | None = None,
                      explored_m2_start: float | None = None) -> dict:
        mission_id = uuid.uuid4().hex
        sig = None if start_signature is None else _encode_signature(start_signature)
        with self._tx() as con:
            con.execute(
                "INSERT INTO missions (id, env_id, map_id, started_at, start_x, "
                "start_y, start_theta, start_signature, status, explored_m2_start,"
                " schema_version) VALUES (?,?,?,?,?,?,?,?,'running',?,?)",
                (mission_id, env_id, map_id, now_iso(), float(start_pose[0]),
                 float(start_pose[1]), float(start_pose[2]), sig,
                 explored_m2_start, SCHEMA_VERSION))
        return self.mission(mission_id)

    def update_mission(self, mission_id: str, **fields) -> dict:
        bad = set(fields) - _MISSION_FIELDS
        if bad:
            raise StoreError(f"not mission fields: {sorted(bad)}")
        if "status" in fields and fields["status"] not in MISSION_STATUSES:
            raise StoreError(f"mission status {fields['status']!r}")
        if fields.get("stop_reason") not in (None, *STOP_REASONS):
            raise StoreError(f"stop reason {fields['stop_reason']!r}")
        if not fields:
            return self.mission(mission_id)
        values = []
        for key, value in fields.items():
            if key == "start_signature" and value is not None:
                value = _encode_signature(value)
            elif key in _MISSION_JSON and value is not None:
                value = json.dumps(value)
            values.append(value)
        with self._tx() as con:
            cur = con.execute(
                f"UPDATE missions SET {', '.join(f'{k}=?' for k in fields)} "
                "WHERE id=?", (*values, mission_id))
            if cur.rowcount != 1:
                raise StoreError("no such mission")
        return self.mission(mission_id)

    def add_event(self, mission_id: str, event: dict) -> None:
        """Append to the mission's compact event log, keeping the newest
        events_cap."""
        with self._tx() as con:
            row = con.execute("SELECT events FROM missions WHERE id=?",
                              (mission_id,)).fetchone()
            if row is None:
                raise StoreError("no such mission")
            events = json.loads(row[0])
            events.append(dict(event, t=event.get("t", now_iso())))
            events = events[-self.cfg.events_cap:]
            con.execute("UPDATE missions SET events=? WHERE id=?",
                        (json.dumps(events), mission_id))

    def mission(self, mission_id: str) -> dict | None:
        m = self._mission(self._connect().execute(
            "SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone())
        if m is not None and m["start_signature"] is not None:
            m["start_signature"] = _decode_signature(m["start_signature"])
        return m

    def missions(self, env_id: str, limit: int = 20,
                 status: str | None = None) -> list[dict]:
        sql = "SELECT id FROM missions WHERE env_id=?"
        args: list[Any] = [env_id]
        if status is not None:
            sql += " AND status=?"
            args.append(status)
        sql += " ORDER BY started_at DESC, rowid DESC LIMIT ?"
        args.append(int(limit))
        return [self.mission(r[0]) for r in self._connect().execute(sql, args)]

    @staticmethod
    def _append_json(con, table: str, row_id: str, column: str, value) -> None:
        row = con.execute(f"SELECT {column} FROM {table} WHERE id=?",
                          (row_id,)).fetchone()
        if row is None:
            raise StoreError(f"no such {table[:-1]}: {row_id}")
        items = json.loads(row[0])
        if value not in items:
            items.append(value)
        con.execute(f"UPDATE {table} SET {column}=? WHERE id=?",
                    (json.dumps(items), row_id))

    # --- files: checkpoint, trails ----------------------------------------------
    def env_dir(self, env_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(env_id)) or "home"
        return self.data_dir / "envs" / safe

    def checkpoint_file(self, env_id: str) -> Path:
        return self.env_dir(env_id) / "mission_checkpoint.json"

    def save_checkpoint(self, env_id: str, payload: dict) -> None:
        """spec 3.6: every checkpoint_s and on every state change. Atomic,
        previous good copy kept as .bak; breadcrumbs trimmed to the last
        checkpoint_keep."""
        data = dict(payload)
        crumbs = data.get("breadcrumbs")
        if isinstance(crumbs, list):
            data["breadcrumbs"] = crumbs[-self.cfg.checkpoint_keep:]
        data["schema_version"] = SCHEMA_VERSION
        data["saved_at"] = now_iso()
        durable.save_json(self.checkpoint_file(env_id), data)

    def load_checkpoint(self, env_id: str) -> tuple[dict | None, str | None]:
        data, source = durable.load_json(self.checkpoint_file(env_id))
        if data is None:
            return None, None
        return migrate_record("checkpoint", data), source

    def map_file(self, env_id: str) -> Path:
        """The mission's OWN map (start frame), saved with each checkpoint so a
        power cut keeps it -- and so she can be found on it again."""
        return self.env_dir(env_id) / "mission_map.npz"

    def pose_file(self, env_id: str) -> Path:
        return self.env_dir(env_id) / "mission_pose.json"

    def save_pose(self, env_id: str, mission_id: str, pose, t: float) -> None:
        """Her pose in the mission's frame, once a second, fsynced: after a
        power cut it says where to LOOK for her (within a metre or so), which
        is what tells two look-alike rooms apart."""
        durable.atomic_write_json(self.pose_file(env_id), {
            "mission_id": mission_id, "pose": [float(v) for v in pose], "t": float(t)})

    def load_pose(self, env_id: str, mission_id: str) -> dict | None:
        try:
            data = json.loads(self.pose_file(env_id).read_text())
        except (OSError, ValueError):
            return None
        if data.get("mission_id") != mission_id or len(data.get("pose") or []) != 3:
            return None
        return data

    def save_trail(self, env_id: str, mission_id: str, crumbs: list) -> Path:
        path = self.env_dir(env_id) / "trails" / f"{mission_id}.json"
        durable.save_json(path, {"schema_version": SCHEMA_VERSION,
                                 "mission_id": mission_id, "crumbs": crumbs})
        return path


# =============================================================================
# helpers
# =============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _encode_signature(sig: Iterable) -> str:
    """360 ranges -> JSON, NaN (no return) as null."""
    out = []
    for v in sig:
        f = float(v) if v is not None else float("nan")
        out.append(None if not math.isfinite(f) else round(f, 4))
    return json.dumps(out)


def _decode_signature(value) -> list[float]:
    """JSON text or an already-decoded list -> floats, null back to NaN."""
    items = json.loads(value) if isinstance(value, (str, bytes)) else value
    return [float("nan") if v is None else float(v) for v in items]


# JSON record migrations: (kind, from_version) -> function returning the record
# at from_version + 1. v0 is "written before schema_version existed".
def _checkpoint_v0_to_v1(rec: dict) -> dict:
    rec = dict(rec)
    rec.setdefault("breadcrumbs", [])
    rec.setdefault("blacklist", [])
    rec.setdefault("pending_label", None)
    rec.setdefault("counters", {})
    return rec


JSON_MIGRATIONS = {("checkpoint", 0): _checkpoint_v0_to_v1}


def migrate_record(kind: str, record: dict) -> dict:
    version = int(record.get("schema_version", 0) or 0)
    if version > SCHEMA_VERSION:
        raise StoreError(f"{kind} is schema v{version}; this code knows "
                         f"v{SCHEMA_VERSION}")
    while version < SCHEMA_VERSION:
        step = JSON_MIGRATIONS.get((kind, version))
        record = step(record) if step else dict(record)
        version += 1
        record["schema_version"] = version
    return record
