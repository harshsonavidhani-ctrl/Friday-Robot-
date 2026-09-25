#!/usr/bin/env python3
"""Tests for trip_objects.py — what she saw on an explore trip.

    ./venv/bin/python scripts/test_trip_objects.py

No camera, no hardware, no model. Every file is a temporary one: the real
state/scene.json and memory.db are never read or written.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                        # noqa: E402
import trip_objects as to                            # noqa: E402

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0
_TMP = tempfile.TemporaryDirectory()
SCENE = Path(_TMP.name) / "scene.json"


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


def frame(at: float, *objects: tuple[str, float]) -> None:
    """Write scene.json the way scene.py does: one frame, its time, its objects."""
    SCENE.write_text(json.dumps({
        "taken_at": at, "camera": "up",
        "objects": [{"label": label, "score": score, "side": "ahead",
                     "bearing_deg": 0, "area": 0.1}
                    for label, score in objects]}))


def sightings(start: float, **kw):
    kw.setdefault("remember", [].append)
    return to.TripSightings(scene_json=SCENE, now=start, **kw)


def test_counting() -> None:
    print("\n— what counts as seen —")
    t0 = 1000.0
    frame(t0 - 5, ("sofa", 0.95))                 # before the trip
    seen = sightings(t0)
    seen.poll()
    check("a picture from before the trip is not counted", seen.frames == 0)

    frame(t0 + 8, ("chair", 0.7), ("stop sign", 0.62))
    check("a new picture is counted", seen.poll() and seen.frames == 1)
    check("the same picture read twice is counted once",
          not seen.poll() and seen.frames == 1)
    frame(t0 + 16, ("chair", 0.72), ("bottle", 0.66), ("bottle", 0.64))
    seen.poll()
    frame(t0 + 24, ("television", 0.88))
    seen.poll()
    got = dict(seen.confirmed())
    check("seen in two pictures: a chair", got.get("chair") == 1, str(got))
    check("named once, weakly: the stop sign is left out",
          "stop sign" not in got, str(got))
    check("named once but strongly: the television counts",
          got.get("television") == 1, str(got))
    check("two bottles in one picture, seen once weakly: not yet",
          "bottle" not in got, str(got))
    frame(t0 + 32, ("bottle", 0.7))
    seen.poll()
    got = dict(seen.confirmed())
    check("bottles once confirmed: the most at once is two, not the sum",
          got.get("bottle") == 2, str(got))
    check("the glimpses are kept for the dashboard, not said",
          "stop sign" in seen.as_dict()["glimpsed"])
    check("a broken scene file is not a crash",
          (SCENE.write_text("{not json"), seen.poll())[1] is False)


def test_sentences() -> None:
    print("\n— what she says —")
    check("a / an", to.spoken("chair", 1) == "a chair"
          and to.spoken("apple", 1) == "an apple")
    check("plurals, and people", to.spoken("bottle", 2) == "two bottles"
          and to.spoken("person", 3) == "three people"
          and to.spoken("knife", 2) == "two knives")
    check("detector names made spoken",
          to.spoken("dining table", 1) == "a table"
          and to.spoken("mobile phone", 2) == "two phones")
    check("a list reads like a sentence",
          to.listing(["a chair", "two bottles", "a television"])
          == "a chair, two bottles and a television")

    t0 = 2000.0
    seen = sightings(t0)
    for k in range(3):
        frame(t0 + 8 * (k + 1), ("chair", 0.7), ("bottle", 0.7),
              ("bottle", 0.66))
        seen.poll()
    said = seen.sentence()
    check("the report", said == "On this trip I saw a chair and two bottles."
          or said == "On this trip I saw two bottles and a chair.", said)

    empty = sightings(t0)
    frame(t0 + 8)
    empty.poll()
    check("pictures but nothing she knows: she says exactly that",
          empty.sentence() == "I did not spot anything I know the name of "
                              "on this trip.", empty.sentence())

    blind = sightings(3000.0, camera_reason=lambda: (
        "the dashboard is streaming the up camera"))
    frame(2990.0, ("chair", 0.9))                   # stale: before the trip
    said = blind.finish()
    check("no pictures because the dashboard had the camera: says why, and "
          "does not claim she saw nothing",
          "dashboard was showing it" in said, said)
    other = sightings(3000.0, camera_reason=lambda: None)
    check("no pictures for another reason: says so plainly",
          other.finish().startswith("My camera did not give me any"))


def test_remembering() -> None:
    print("\n— what she remembers —")
    notes: list[str] = []
    t0 = time.time()
    report = Path(_TMP.name) / "trip_objects.json"
    seen = sightings(t0, remember=notes.append, report_file=report)
    frame(t0 + 8, ("chair", 0.7))
    seen.poll()
    frame(t0 + 16, ("chair", 0.75))
    said = seen.finish()
    check("one note per trip, naming what she saw",
          len(notes) == 1 and "I saw a chair" in notes[0]
          and notes[0].startswith("When I explored on"), str(notes))
    data = json.loads(report.read_text())
    check("and the trip's sightings are written for the dashboard",
          data["seen"][0]["label"] == "chair" and data["said"] == said,
          str(data))

    nothing: list[str] = []
    quiet = sightings(t0 + 100, remember=nothing.append)
    quiet.finish()
    check("nothing seen, nothing remembered", nothing == [])

    def broken(_text):
        raise RuntimeError("database is locked")
    fragile = sightings(t0 + 200, remember=broken)
    frame(t0 + 208, ("tv", 0.9))
    fragile.poll()
    check("a memory that fails does not end the trip",
          isinstance(fragile.finish(), str))
    check("the real memory kind is its own, so it can be found again",
          to.MEMORY_KIND == "exploration_sighting")


def test_the_trip_collects_as_it_goes() -> None:
    """The real `_watch_objects` loop from nav_tools, on a temp file."""
    print("\n— collecting during a trip (nav_tools._watch_objects) —")
    from mcp_server.tools import nav_tools

    saved = config.SCENE_BRIDGE_POLL_S
    config.SCENE_BRIDGE_POLL_S = 0.05
    try:
        start = time.time()
        seen = sightings(start)

        async def trip():
            watcher = asyncio.create_task(nav_tools._watch_objects(seen))
            for k in range(3):
                frame(start + 1 + k, ("chair", 0.7))
                await asyncio.sleep(0.2)
            watcher.cancel()
            return seen.finish()

        said = asyncio.run(trip())
        check("every new frame during the trip is collected",
              seen.frames == 3, f"{seen.frames} frames")
        check("and reported at the end", said == "On this trip I saw a chair.",
              said)
    finally:
        config.SCENE_BRIDGE_POLL_S = saved


def main() -> int:
    test_counting()
    test_sentences()
    test_remembering()
    test_the_trip_collects_as_it_goes()
    print(f"\n{_failures} failure(s)")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
