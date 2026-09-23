#!/usr/bin/env python3
"""nav_bundle.py — navigation's data, kept permanently WITH the map it belongs to.

    python3 nav_bundle.py status
    python3 nav_bundle.py save [--reason "before the hostel test"]
    python3 nav_bundle.py restore [--force]

Harsh, 2026-09-23: "it should use the old map saved one it should also save
all the data there only like the new return to home or navigation". Asked what
he meant, he chose: bundle it with the map.

    state/map-permanent/navigation/
        map.npz               the survey map navigation reads -- a COPY
        places.json           home, the kitchen, ... in THAT map's coordinates
        nav_log.jsonl         every trip
        nav_round_trip.json   the come-back spot, while one is armed
        nav_failsafe.json     what she was doing when she stopped for help
        nav_mode.json         round trip on/off
        nav_early_return.json the last time exploration ran out and she came back
        map_session.json      which build of the map the places belong to
        bundle.json           manifest: when, why, the map's sha256, counts
        history/<stamp>/      the bundle as it was, every time the MAP or the
                              PLACES changed

WHY TOGETHER. A saved place is a coordinate, and a coordinate means nothing
without the map it was measured on. Kept apart -- places in memory.db, the map
in state/, the rest scattered beside it -- one can change while the other does
not, and nothing records which place belonged to which map. Here they are
saved as one unit, and whenever either changes, the pair as it was goes into
history first.

⚠ NEVER DELETES HISTORY, NEVER WRITES THE LIVE MAP. `save()` only reads
state/map.npz and memory.db; everything it writes is inside the bundle
folder. The current bundle MIRRORS the live state -- a FAILSAFE record that
has been cleared is not kept as if it were still waiting -- but every map and
every set of places it ever held is in history/ for good. `restore` is the
one command that writes outside the folder, and it refuses to replace a live
map unless told to with --force, after putting that live map into history.

⚠ IT FOLLOWS THE STATE DIRECTORY NAVIGATION IS USING. The folder is resolved
from `nav_state_machine.NAV_LOG` at call time, so the test suites -- which
point NAV_LOG at a temporary directory -- bundle into that temporary
directory and never touch the real one. A save can never break navigation:
every caller goes through `save_quietly()`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402

log = logging.getLogger("friday.nav_bundle")

_LOCK = threading.Lock()

# The live files mirrored into the bundle, by the nav_state_machine attribute
# that names them (so a test's redirect is honoured) or a fixed path.
_NSM_FILES = ("NAV_LOG", "PENDING_FILE", "FAILSAFE_FILE", "NAV_MODE_FILE",
              "EARLY_RETURN_FILE")


def bundle_dir() -> Path:
    import nav_state_machine as nsm
    return Path(nsm.NAV_LOG).parent / "map-permanent" / "navigation"


def _live_files() -> list[Path]:
    import nav_state_machine as nsm
    files = [Path(getattr(nsm, name)) for name in _NSM_FILES]
    files.append(Path(nsm.NAV_LOG).parent / "map_session.json")
    return files


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _places() -> list[dict]:
    """Every saved place, as plain data, in a stable order."""
    from slam import places
    rows = []
    for row in places.all():
        rows.append({k: row.get(k) for k in
                     ("name", "x", "y", "radius_m", "kind", "note",
                      "created_at", "last_seen", "visits")})
    return sorted(rows, key=lambda r: r["name"])


def _places_key(rows: list[dict]) -> str:
    """What counts as the places CHANGING: names and positions, not visits."""
    core = [(r["name"], round(float(r["x"]), 3), round(float(r["y"]), 3),
             round(float(r.get("radius_m") or 0.0), 3)) for r in rows]
    return hashlib.sha256(json.dumps(core).encode()).hexdigest()


def manifest(folder: Path | None = None) -> dict | None:
    try:
        return json.loads(((folder or bundle_dir()) / "bundle.json")
                          .read_text())
    except (OSError, ValueError):
        return None


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _snapshot(folder: Path, why: str) -> Path | None:
    """Copy the current bundle into history/. Copies, never moves."""
    if manifest(folder) is None:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = folder / "history" / stamp
    n = 1
    while dest.exists():                      # two saves in one second
        n += 1
        dest = folder / "history" / f"{stamp}-{n}"
    dest.mkdir(parents=True)
    for item in folder.iterdir():
        if item.is_file():
            shutil.copy2(item, dest / item.name)
    _atomic_write(dest / "why.txt", why + "\n")
    return dest


def save(reason: str = "") -> dict:
    """Refresh the bundle from the live state. Returns its manifest."""
    with _LOCK:
        folder = bundle_dir()
        folder.mkdir(parents=True, exist_ok=True)
        live_map = Path(config.SLAM_MAP_FILE)
        map_sha = _sha256(live_map)
        rows = _places()
        places_key = _places_key(rows)

        before = manifest(folder)
        archived = None
        if before is not None:
            changed = []
            if before.get("map_sha256") != map_sha:
                changed.append("the map")
            if before.get("places_key") != places_key:
                changed.append("the places")
            if changed:
                archived = _snapshot(folder, f"{' and '.join(changed)} "
                                             f"changed: {reason}")

        if map_sha is not None:
            if _sha256(folder / "map.npz") != map_sha:
                shutil.copy2(live_map, folder / "map.npz")
        _atomic_write(folder / "places.json",
                      json.dumps({"map_sha256": map_sha, "places": rows},
                                 indent=2))

        mirrored = []
        for src in _live_files():
            dest = folder / src.name
            if src.exists():
                shutil.copy2(src, dest)
                mirrored.append(src.name)
            elif dest.exists():
                # Mirrors the live state: a cleared record is not waiting any
                # more. The old copy is in history if the pair ever changed.
                dest.unlink()

        trips = 0
        try:
            trips = sum(1 for _ in (folder / "nav_log.jsonl").open())
        except OSError:
            pass
        out = {
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "at": time.time(),
            "reason": reason,
            "map_file": str(live_map),
            "map_sha256": map_sha,
            "places_key": places_key,
            "places": [r["name"] for r in rows],
            "trips_logged": trips,
            "files": sorted((["map.npz"] if map_sha else [])
                            + ["places.json"] + mirrored),
            "history": len(list((folder / "history").glob("*")))
                       if (folder / "history").exists() else 0,
            "archived_previous": str(archived) if archived else None,
        }
        _atomic_write(folder / "bundle.json", json.dumps(out, indent=2))
        return out


def save_quietly(reason: str = "") -> None:
    """save(), for callers in the middle of navigating: never raises."""
    try:
        save(reason)
    except Exception as exc:                  # noqa: BLE001
        log.warning("could not refresh the navigation bundle (%s): %s",
                    reason, exc)


def status() -> dict:
    """For the dashboard and the CLI: what is saved with the map, and is it
    still the map navigation is using."""
    folder = bundle_dir()
    m = manifest(folder)
    if m is None:
        return {"saved": False, "folder": str(folder)}
    live = _sha256(Path(config.SLAM_MAP_FILE))
    return {
        "saved": True,
        "folder": str(folder),
        "at": m.get("at"),
        "reason": m.get("reason"),
        "places": m.get("places", []),
        "trips_logged": m.get("trips_logged", 0),
        "history": m.get("history", 0),
        "same_map_as_live": live is not None and live == m.get("map_sha256"),
    }


def restore(force: bool = False) -> str:
    """Put the bundle back: its map (only with force) and its places.

    Places are re-saved through `slam/places.py` exactly as `remember_place`
    saves them. The live map is only replaced with --force, and never without
    first archiving the live one, because a map is the one thing here that
    cannot be rebuilt by standing somewhere and saying a name.
    """
    from slam import places

    folder = bundle_dir()
    m = manifest(folder)
    if m is None:
        return f"nothing to restore: no bundle in {folder}"
    said = []
    live_map = Path(config.SLAM_MAP_FILE)
    if (folder / "map.npz").exists() and _sha256(live_map) != m.get("map_sha256"):
        if not force:
            return ("the live map is not the bundle's map. Its places belong "
                    "to the bundle's map, so restoring them onto this one "
                    "would put every place in the wrong spot. Run with --force "
                    "to put the bundle's map back as well (the live map is "
                    "archived first).")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        keep = folder / "history" / f"{stamp}-live-before-restore"
        keep.mkdir(parents=True, exist_ok=True)
        if live_map.exists():
            shutil.copy2(live_map, keep / "map.npz")
        shutil.copy2(folder / "map.npz", live_map)
        said.append(f"map restored (the live one kept in {keep})")
    rows = json.loads((folder / "places.json").read_text())["places"]
    for row in rows:
        places.remember_here(row["name"], (row["x"], row["y"], 0.0),
                             kind=row.get("kind") or "place",
                             note=row.get("note"))
    said.append(f"{len(rows)} place(s) restored: "
                f"{', '.join(r['name'] for r in rows) or 'none'}")
    return "; ".join(said)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("status", "save", "restore"))
    ap.add_argument("--reason", default="by hand")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=2))
    elif args.command == "save":
        print(json.dumps(save(args.reason), indent=2))
    else:
        print(restore(force=args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
