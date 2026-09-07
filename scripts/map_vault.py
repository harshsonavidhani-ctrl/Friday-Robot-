#!/usr/bin/env python3
"""Friday's permanent map vault. NOTHING IN HERE EVER DELETES A MAP.

    python3 scripts/map_vault.py backup [--reason "before the 02:00 run"]
    python3 scripts/map_vault.py restore [--force]
    python3 scripts/map_vault.py status

Harsh, 2026-09-08: *"pls save this permanent and never ever delet its map and
yes it will be there unefected"*.

WHY THIS IS A SEPARATE SCRIPT AND NOT A PATCH TO pi_deploy.py
-------------------------------------------------------------
The same message said the exploration mode must not change -- "no changes in
exploration mode pls it will be untoched only it will be wired up". So the
permanence lives OUTSIDE the run: the systemd units call this before and after
every exploration, and the run itself behaves exactly as it did in the
`run-20260904-2118-best` configuration. `ControlLoop._archive_map` still moves a
rejected map aside and still rebuilds, because writing into a map the matcher
has positively placed somewhere else is what produced 335 disconnected occupied
blobs and one wall at three different offsets. That behaviour is CORRECT and is
left alone; this makes it harmless, because the good map is already copied out.

THE RATCHET, AND WHY IT IS NOT `>=`
-----------------------------------
A rebuilt map is small: the failure this vault exists to survive is a 321 m^2
map being replaced by a 4 m^2 one. But a healthy map also SHRINKS slightly --
measured 2026-09-08, a clean 10-minute run took the map 321.8 -> 321.6 m^2,
because the log-odds clamp lets cells change their mind (see
[[friday-slam]]'s "clamp the log-odds"). A strict `>=` would refuse that run's
map and quietly throw away ten minutes of new visit data.

So the gate is `live >= vault * KEEP_FRACTION` (0.95): it refuses a collapse and
accepts ordinary breathing. A refused map is still written to history -- being
refused for the vault slot is not a reason to lose the file.

`explored_m2` is read through `world.WorldMap`, the same class the run uses, so
this script and the run's log can never disagree about how big a map is.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl"))

import world                                            # noqa: E402

STATE = ROOT / "state"
VAULT = STATE / "map-permanent"
HISTORY = VAULT / "history"
META = VAULT / "vault.json"

# The three files that ARE the map. A map without its pose cannot be
# relocalised at all (see [[friday-map-relocalisation-ratchet]]), so the pose
# travels with the grid or the backup is worthless.
PARTS = ("rl_map.npz", "rl_map.pose.json", "rl_map.visits.npz")

KEEP_FRACTION = 0.95


def _measure(map_file: Path) -> dict:
    """Size a map on disk using the run's own definition of explored area."""
    if not map_file.exists():
        return {"exists": False, "explored_m2": 0.0, "sweeps": 0, "occupied": 0}
    w = world.WorldMap()
    w.map_file = map_file
    try:
        if not w.load():
            return {"exists": True, "loadable": False, "explored_m2": 0.0,
                    "sweeps": 0, "occupied": 0}
    except Exception as exc:                            # noqa: BLE001
        # A corrupt file must not take the vault down -- it is exactly when a
        # map is unreadable that the backup matters most.
        return {"exists": True, "loadable": False, "error": str(exc),
                "explored_m2": 0.0, "sweeps": 0, "occupied": 0}
    s = w.stats()
    return {"exists": True, "loadable": True,
            "explored_m2": round(float(s["explored_m2"]), 1),
            "occupied": int(s["occupied"]),
            "sweeps": int(s["sweeps_total"])}


def _snapshot_dir(stamp: str) -> Path:
    """A history directory that CANNOT collide with an existing one.

    The stamp has one-second resolution, and a test caught two backups in the
    same second writing into the same directory -- the second silently
    overwrote the first. In a script whose entire promise is "nothing is ever
    deleted", a same-second overwrite is the one bug that matters.
    """
    base = HISTORY / stamp
    if not base.exists():
        return base
    n = 2
    while (cand := HISTORY / f"{stamp}-{n}").exists():
        n += 1
    return cand


def _copy_set(src_dir: Path, dst_dir: Path) -> list[str]:
    """Copy the map triplet. copy2, never move: the source must stay put."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for name in PARTS:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
            done.append(name)
    return done


def cmd_backup(reason: str) -> int:
    live = _measure(STATE / "rl_map.npz")
    held = _measure(VAULT / "rl_map.npz")

    if not live.get("loadable"):
        print(f"backup SKIPPED: no loadable map at {STATE/'rl_map.npz'} "
              f"({live}). The vault is untouched.")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    # HISTORY FIRST, ALWAYS, WHATEVER THE VERDICT. A map that loses the ratchet
    # is still ten minutes of driving and is still kept.
    snap = _snapshot_dir(stamp)
    _copy_set(STATE, snap)

    floor = held["explored_m2"] * KEEP_FRACTION
    if held.get("loadable") and live["explored_m2"] < floor:
        print(f"vault NOT replaced: live {live['explored_m2']} m^2 is below "
              f"{floor:.1f} m^2 ({KEEP_FRACTION:.0%} of the vault's "
              f"{held['explored_m2']} m^2) -- this looks like a rebuild, not a run.")
        print(f"  the live map was still saved to {snap}")
        return 0

    copied = _copy_set(STATE, VAULT)
    META.write_text(json.dumps({
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": reason,
        "explored_m2": live["explored_m2"],
        "occupied_cells": live["occupied"],
        "sweeps_total": live["sweeps"],
        "files": copied,
        "previous_explored_m2": held["explored_m2"],
        "history_snapshot": str(snap.relative_to(STATE)),
    }, indent=2) + "\n")
    grew = live["explored_m2"] - held["explored_m2"]
    print(f"vault updated: {live['explored_m2']} m^2 "
          f"({grew:+.1f} vs the {held['explored_m2']} m^2 held), "
          f"{live['occupied']} occupied cells, {live['sweeps']} sweeps.")
    print(f"  vault:   {VAULT}")
    print(f"  history: {snap}")
    return 0


def cmd_restore(force: bool) -> int:
    held = _measure(VAULT / "rl_map.npz")
    if not held.get("loadable"):
        print(f"nothing to restore: the vault at {VAULT} has no loadable map.")
        return 1
    live = _measure(STATE / "rl_map.npz")
    if live.get("loadable") and not force:
        print(f"live map present ({live['explored_m2']} m^2) -- not overwriting. "
              f"Use --force to replace it with the vault's {held['explored_m2']} m^2.")
        return 0
    # Even a forced restore keeps what it replaces.
    if live.get("exists"):
        snap = _snapshot_dir(f"{time.strftime('%Y%m%d-%H%M%S')}-replaced")
        _copy_set(STATE, snap)
        print(f"  the map being replaced was saved to {snap}")
    copied = _copy_set(VAULT, STATE)
    print(f"restored {held['explored_m2']} m^2 from the vault: {', '.join(copied)}")
    return 0


def cmd_status() -> int:
    live, held = _measure(STATE / "rl_map.npz"), _measure(VAULT / "rl_map.npz")
    print(f"live  {STATE/'rl_map.npz'}")
    print(f"      {live}")
    print(f"vault {VAULT/'rl_map.npz'}")
    print(f"      {held}")
    if META.exists():
        print(f"meta  {json.loads(META.read_text())}")
    snaps = sorted(p.name for p in HISTORY.glob("*")) if HISTORY.exists() else []
    print(f"history: {len(snaps)} snapshot(s)"
          + (f", newest {snaps[-1]}" if snaps else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backup", help="copy the live map into the vault (ratcheted)")
    b.add_argument("--reason", default="manual")
    r = sub.add_parser("restore", help="copy the vault back over the live map")
    r.add_argument("--force", action="store_true")
    sub.add_parser("status", help="show both maps")
    a = ap.parse_args()
    if a.cmd == "backup":
        return cmd_backup(a.reason)
    if a.cmd == "restore":
        return cmd_restore(a.force)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main())
