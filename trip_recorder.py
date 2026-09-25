"""trip_recorder.py — a flight recorder for explore trips.

    trip_recorder.start()      # at the start of a trip: a new file
    trip_recorder.note("sweep", src="follower", scan={...})
    trip_recorder.stop()

Every sweep, every move and every match result of one trip, one JSON line
each, in state/trip_recordings/. Harsh's trip at 2026-09-25 22:35 failed on
the first sweep after a turn, and the log could only say THAT it failed ("the
sweep did not match"), not what the sweep was or what the turn really did.
With the sweeps on disk a failed trip can be replayed against the real Mapper
offline, as many times as a fix needs, without driving her again.

It records only; nothing reads it back while she drives. A write that fails
is dropped silently rather than allowed near a drive.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("friday.trip_recorder")

DIR = config.ROOT / "state" / "trip_recordings"
# Oldest recordings beyond this many are deleted when a new trip starts. A
# trip is a few hundred kilobytes.
KEEP = 30

_lock = threading.Lock()
_fh = None
_path: Path | None = None


def start(directory: Path | None = None) -> Path | None:
    """Open a new recording. Returns its path, or None if it could not."""
    global _fh, _path
    stop()
    folder = Path(directory or DIR)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        old = sorted(folder.glob("trip-*.jsonl"))
        for stale in old[:max(0, len(old) - KEEP + 1)]:
            stale.unlink()
        path = folder / time.strftime("trip-%Y%m%d-%H%M%S.jsonl")
        with _lock:
            _fh = open(path, "a", encoding="utf-8")
            _path = path
    except OSError as exc:
        log.warning("could not start a trip recording: %s", exc)
        return None
    log.info("recording this trip to %s", path)
    return path


def note(kind: str, **data) -> None:
    """One line: what happened, and when. A no-op when nothing is recording."""
    if _fh is None:
        return
    row = {"t": round(time.time(), 3), "kind": kind}
    row.update(data)
    try:
        line = json.dumps(row, default=_plain)
        with _lock:
            if _fh is not None:
                _fh.write(line + "\n")
                _fh.flush()
    except (OSError, TypeError, ValueError):
        pass


def stop() -> Path | None:
    global _fh, _path
    with _lock:
        fh, path = _fh, _path
        _fh, _path = None, None
    if fh is not None:
        try:
            fh.close()
        except OSError:
            pass
    return path


def active() -> Path | None:
    return _path


def _plain(value):
    try:
        import numpy as np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:
        pass
    return str(value)


def load(path) -> list[dict]:
    """Read a recording back, sweeps as {bearing: mm} with int keys."""
    rows = []
    for text in Path(path).read_text().splitlines():
        row = json.loads(text)
        if isinstance(row.get("scan"), dict):
            row["scan"] = {int(k): int(v) for k, v in row["scan"].items()}
        rows.append(row)
    return rows
