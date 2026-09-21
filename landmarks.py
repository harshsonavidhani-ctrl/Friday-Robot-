"""landmarks.py — file 14 step 7: asking what to call a room she just found.

    new contiguous free space ──> not inside a known place? ──> ASK ──> save

⚠ THIS MODULE CANNOT SPEAK OR LISTEN, AND THAT IS DELIBERATE. The microphone
belongs to ONE process: `main.py` / `wake_word.py` hold the `stt.AudioStream`,
and nothing under `mcp_server/` ever opens it -- checked, not assumed. But
exploration runs INSIDE the MCP server. So a landmark prompt that called
`stt.capture()` from here would be a second process opening the same USB mic
while the wake word is listening on it, which is the failure that already cost
a debugging session once ([[friday-wake-word-starts-first]]).

So `ask` is INJECTED. Whoever calls this supplies a way to ask a question and
get an answer, and supplies the one that is right for the process they are in:

    * the brain (main.py) owns the mic -- it can pass a real speak-and-listen;
    * the MCP server does not -- it should pass a queue-it-for-the-brain, or
      nothing at all, in which case nothing is asked and nothing is saved.

`ask(question: str) -> str | None` may be sync or async. Returning None or an
empty string means "no answer", and no answer means NOTHING IS SAVED. A
landmark named from a misheard reply is worse than no landmark, because
`go_to` will drive to it later with complete confidence.

WHAT COUNTS AS A DISCOVERY. Not "the free area went up" -- it goes up on
almost every sweep, by a few cells, as the LiDAR trims the edges of what it
already knows. What counts is a CONTIGUOUS new region of at least
`MIN_AREA_M2`, because that is the shape of walking into a room, and the
shape that noise does not have.

WHERE IT IS SAVED. `slam/places.py::remember_here` -- the same store as
`remember_place`, `go_to` and `list_places`, so a named room is a place like
any other and every existing tool can already see it. The `places` table has
no `map_session_id` column and `slam/` is read-only to this file, so the
session id is written into the `note` field behind a `[map:...]` marker.
`session_of()` reads it back. That is a compromise and it is the cheapest one
available: the alternative is a schema change to a WAL database that three
processes have open.

WHY THE SESSION ID MATTERS AT ALL. A landmark is a coordinate in a particular
map's frame. Rebuild the map and every stored coordinate silently becomes
meaningless -- the name still resolves, the drive still happens, the
destination is somewhere else entirely. Stamping the session lets a later step
say "these places were named against a map that no longer exists" instead of
driving to them.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import label

import config

log = logging.getLogger("friday.landmarks")

# A contiguous new region this big is a room, an alcove or a landing. Below it
# is the map growing at its edges, which happens constantly and means nothing.
# Four square metres is about a 2 m x 2 m space.
MIN_AREA_M2 = 4.0

# How far from a stored place still counts as "inside" it, on top of that
# place's own radius. A landmark's radius says how big the thing is; this says
# how close you have to be before naming it again is redundant.
INSIDE_MARGIN_M = 0.5

# The radius a newly named area gets. `config.PLACE_RADIUS_M` is what
# `remember_place` gives a spot; a room deserves more, and this is scaled from
# the area actually discovered instead of being a constant.
MIN_RADIUS_M = config.PLACE_RADIUS_M
MAX_RADIUS_M = 3.0

_CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
_SESSION_FILE = config.ROOT / "state" / "map_session.json"
_SESSION_MARK = re.compile(r"\[map:([^\]]+)\]")


# --------------------------------------------------------------------------
# which map are we in
# --------------------------------------------------------------------------

def map_session_id(grid=None) -> str:
    """A stable id for the CURRENT life of the map, minted once and reused.

    Rotated when the map is detectably not the same map any more -- the sweep
    count going DOWN means it was rebuilt or replaced, and every coordinate
    stored against the old one is now a coordinate in a dead frame. Growing
    sweep counts are the normal case and keep the same id, because a map that
    grew is still the same map.
    """
    scans = int(getattr(grid, "scans", 0) or 0)
    try:
        saved = json.loads(_SESSION_FILE.read_text())
        if scans >= int(saved.get("scans", 0)):
            if scans > int(saved.get("scans", 0)):
                saved["scans"] = scans
                _write_session(saved)
            return str(saved["id"])
        log.warning("map sweeps went %d -> %d: the map was rebuilt, new session",
                    int(saved.get("scans", 0)), scans)
    except (OSError, ValueError, KeyError):
        pass

    fresh = {"id": time.strftime("map-%Y%m%d-%H%M%S"), "scans": scans,
             "created_at": time.time()}
    _write_session(fresh)
    return fresh["id"]


def _write_session(data: dict) -> None:
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(_SESSION_FILE)
    except OSError as exc:                       # noqa: BLE001
        log.warning("could not write %s: %s", _SESSION_FILE, exc)


def stamp_note(session_id: str, note: str | None) -> str:
    return f"[map:{session_id}] {note or ''}".strip()


def session_of(row: dict | None) -> str | None:
    """Which map session a stored place was named in, if it says."""
    if not row:
        return None
    found = _SESSION_MARK.search(row.get("note") or "")
    return found.group(1) if found else None


# --------------------------------------------------------------------------
# noticing that somewhere new has appeared
# --------------------------------------------------------------------------

@dataclass
class Discovery:
    """A contiguous region of free space that was not known last time."""

    area_m2: float
    centroid: tuple[float, float]
    cells: int
    radius_m: float

    def as_dict(self) -> dict:
        return {"area_m2": round(self.area_m2, 2), "cells": self.cells,
                "centroid": [round(v, 2) for v in self.centroid],
                "radius_m": round(self.radius_m, 2)}


class AreaWatch:
    """Remembers what was free last time, so "new" means something.

    Holds the previous free mask and its origin, and re-registers the two on
    every call: the grid GROWS and its array is reallocated as it does, so
    comparing this sweep's array to the last one index-for-index would report
    the entire map as new every time the room got bigger. Comparison is done
    in world cell coordinates, which do not move -- `OccupancyGrid` anchors its
    lattice for exactly this reason.
    """

    def __init__(self, min_area_m2: float = MIN_AREA_M2) -> None:
        self.min_area_m2 = float(min_area_m2)
        self._free: set[tuple[int, int]] = set()
        self.checks = 0

    @staticmethod
    def _world_cells(grid, mask: np.ndarray) -> set[tuple[int, int]]:
        """Free cells as lattice coordinates, independent of array indexing."""
        ox = int(round(grid.origin[0] / grid.cell))
        oy = int(round(grid.origin[1] / grid.cell))
        return {(int(x) + ox, int(y) + oy) for x, y in np.argwhere(mask)}

    def update(self, grid) -> Discovery | None:
        """Fold in the current grid; return a big new region if there is one."""
        self.checks += 1
        mask = grid.free_mask()
        now = self._world_cells(grid, mask)

        if not self._free:
            # First look: everything already on the map is "known", not
            # discovered. Otherwise the first call would offer to name the
            # entire flat, which is not a landmark, it is the map.
            self._free = now
            return None

        fresh = now - self._free
        self._free = now
        if not fresh:
            return None

        # Label the new cells so a region is a region, not a scatter of edge
        # trimming from all over the map that happens to add up.
        xs = [c[0] for c in fresh]
        ys = [c[1] for c in fresh]
        x0, y0 = min(xs), min(ys)
        shape = (max(xs) - x0 + 1, max(ys) - y0 + 1)
        patch = np.zeros(shape, bool)
        for x, y in fresh:
            patch[x - x0, y - y0] = True

        labelled, count = label(patch, structure=_CONNECTIVITY_8)
        if not count:
            return None
        sizes = np.bincount(labelled.ravel())[1:]
        biggest = int(np.argmax(sizes)) + 1
        cells = int(sizes[biggest - 1])
        area = cells * grid.cell ** 2
        if area < self.min_area_m2:
            return None

        where = np.argwhere(labelled == biggest)
        cx = (where[:, 0].mean() + x0 + 0.5) * grid.cell
        cy = (where[:, 1].mean() + y0 + 0.5) * grid.cell
        # A circle of the same area, clamped: a name should cover roughly what
        # was found, and never so much that it swallows the next room.
        radius = float(np.clip(math.sqrt(area / math.pi),
                               MIN_RADIUS_M, MAX_RADIUS_M))
        return Discovery(area, (float(cx), float(cy)), cells, radius)


def inside_known_place(pose, places_rows=None) -> dict | None:
    """The stored place she is already standing in, if any."""
    from slam import places
    rows = places.all() if places_rows is None else places_rows
    for row in rows:
        reach = float(row.get("radius_m") or config.PLACE_RADIUS_M)
        if math.hypot(pose[0] - row["x"], pose[1] - row["y"]) <= reach + INSIDE_MARGIN_M:
            return row
    return None


# --------------------------------------------------------------------------
# asking, and saving only if the answer was real
# --------------------------------------------------------------------------

async def _call(fn, *args):
    """Call `fn`, awaiting it if it is a coroutine function."""
    out = fn(*args)
    if inspect.isawaitable(out):
        out = await out
    return out


@dataclass
class NamingResult:
    saved: bool = False
    name: str = ""
    why: str = ""
    asked: bool = False
    discovery: dict = field(default_factory=dict)
    place: dict = field(default_factory=dict)
    overwrote: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class LandmarkNamer:
    """Notices a new area, asks what it is, and stores the answer.

    `ask(question) -> str | None` and `confirm(question) -> bool | str | None`
    are supplied by the caller (see the module docstring for why). `confirm`
    defaults to `ask`, with a yes/no reading of whatever comes back.
    """

    def __init__(self, ask=None, confirm=None,
                 min_area_m2: float = MIN_AREA_M2,
                 watch: AreaWatch | None = None) -> None:
        self.ask = ask
        self.confirm = confirm
        self.watch = watch or AreaWatch(min_area_m2)

    async def consider(self, grid, pose) -> NamingResult:
        """One check. Cheap and safe to call every few sweeps.

        Returns a result explaining itself either way, because "why did she
        not ask about the kitchen" is otherwise unanswerable from a log.
        """
        found = self.watch.update(grid)
        if found is None:
            return NamingResult(why="nothing new big enough")
        result = NamingResult(discovery=found.as_dict())

        known = inside_known_place(pose)
        if known is not None:
            result.why = f"already inside “{known['name']}”"
            return result

        if self.ask is None:
            result.why = ("found somewhere new, but nothing here can ask what "
                          "to call it — no `ask` was supplied")
            log.info("unnamed discovery: %s", found.as_dict())
            return result

        result.asked = True
        answer = await _call(
            self.ask,
            f"I have found somewhere new, about {found.area_m2:.0f} square "
            f"metres. What should I call it?")
        name = _clean(answer)
        if not name:
            result.why = "no answer, so nothing was saved"
            return result

        return await self.save(name, pose, found, result)

    async def save(self, name: str, pose, found: Discovery,
                   result: NamingResult | None = None) -> NamingResult:
        """Store the name, confirming first if it would overwrite one."""
        from slam import places

        result = result or NamingResult(discovery=found.as_dict())
        result.name = name

        existing = places.find(name)
        if existing is not None:
            gap = math.hypot(existing["x"] - pose[0], existing["y"] - pose[1])
            # `remember_here` MOVES a place that is named again -- by design,
            # because renaming is usually a correction. But it does it
            # silently, and silently moving "the kitchen" three rooms away
            # because two places look alike is not a correction, it is data
            # loss. So: ask first, always, and say which case it is.
            where = (f"about {gap:.1f} m from here"
                     if gap > (existing.get("radius_m") or config.PLACE_RADIUS_M)
                     else "right about here")
            agreed = await self._agree(
                f"I already know somewhere called “{name}”, {where}. "
                f"Should I move it to where I am now?")
            if not agreed:
                result.why = (f"“{name}” already exists {where} and I was not "
                              "told to move it, so I left it alone")
                return result
            result.overwrote = {"name": existing["name"],
                                "x": round(existing["x"], 2),
                                "y": round(existing["y"], 2),
                                "moved_m": round(gap, 2)}

        session = map_session_id()
        row = places.remember_here(name, pose, radius_m=found.radius_m,
                                   note=stamp_note(session, None))
        result.saved = True
        result.place = {"name": row["name"], "x": round(row["x"], 2),
                        "y": round(row["y"], 2),
                        "radius_m": round(float(row.get("radius_m") or 0.0), 2),
                        "map_session_id": session}
        result.why = f"saved “{row['name']}”"
        log.info("named a landmark: %s", result.place)
        return result

    async def _agree(self, question: str) -> bool:
        fn = self.confirm or self.ask
        if fn is None:
            return False
        return _is_yes(await _call(fn, question))


_YES = ("yes", "yeah", "yep", "yup", "sure", "ok", "okay", "go ahead",
        "do it", "please do", "move it", "correct", "right")
_NO = ("no", "nope", "don't", "dont", "do not", "leave it", "cancel",
       "stop", "never mind", "nevermind")


def _is_yes(answer) -> bool:
    """Yes must be said. Anything unclear is a no.

    Asymmetric on purpose: the question being answered is "may I overwrite
    something you named earlier", and a mishearing that reads as yes destroys
    a landmark, while one that reads as no costs another question.
    """
    if answer is True:
        return True
    if not isinstance(answer, str):
        return False
    text = answer.strip().lower()
    if any(word in text for word in _NO):
        return False
    return any(text == word or text.startswith(word + " ") or word in text.split()
               for word in _YES)


def _clean(answer) -> str:
    """A usable place name, or "" if the answer was not one.

    Strips the polite framing people actually say -- "call it the kitchen",
    "it's the hallway" -- because the name is what goes in the database and
    what `go_to` has to match later.
    """
    if not isinstance(answer, str):
        return ""
    text = " ".join(answer.strip().split())
    if not text:
        return ""
    text = re.sub(r"^(you can |let'?s |i'?d |we )?(call|name) (it|this|that) ",
                  "", text, flags=re.I)
    text = re.sub(r"^(it'?s|this is|that'?s|its) ", "", text, flags=re.I)
    text = text.strip(" .!,").strip()
    # A refusal is not a name, and neither is a sentence.
    if not text or _is_no_name(text) or len(text.split()) > 5:
        return ""
    return text.lower()


def _is_no_name(text: str) -> bool:
    low = text.strip().lower()
    return low in ("no", "nothing", "never mind", "nevermind", "skip",
                   "don't", "dont", "cancel", "i don't know", "dunno",
                   "not sure", "no idea")
