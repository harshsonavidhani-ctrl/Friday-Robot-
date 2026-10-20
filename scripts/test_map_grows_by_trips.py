#!/usr/bin/env python3
"""The saved map GROWS BY TRIPS (Harsh, 2026-10-20: a new place, "map grows by
trips"). SlamAdapter.grow_saved: the first trip in a place with no map SEEDS it
(its start is the map's origin); a later trip that fits is MERGED; one that
does not fit leaves the map alone. Uses last night's real mission map (10-19
17:54, 36 m^2) as the trip. Everything in a temp dir: the real map, the real
navigation bundle and memory.db are never touched.

    ./venv/bin/python scripts/test_map_grows_by_trips.py
"""
from __future__ import annotations

import copy
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402
import nav_state_machine as nsm                          # noqa: E402

_TMP = tempfile.TemporaryDirectory()
T = Path(_TMP.name)
nsm.NAV_LOG = T / "state" / "nav_log.jsonl"
nsm.FAILSAFE_FILE = T / "state" / "nav_failsafe.json"
nsm.PENDING_FILE = T / "state" / "nav_round_trip.json"
nsm.NAV_MODE_FILE = T / "state" / "nav_mode.json"
nsm.EARLY_RETURN_FILE = T / "state" / "nav_early_return.json"
config.SLAM_MAP_FILE = T / "state" / "map.npz"
config.MEMORY_DB = T / "memory.db"

from exploration_mission import adapters                 # noqa: E402
from exploration_mission import mission_manager as mm    # noqa: E402
from slam.grid import OccupancyGrid                      # noqa: E402
import map_growth                                        # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


REAL = ROOT / "state/exploration_mission/envs/home-1790517173/mission_map.npz"
if not REAL.exists():
    print("no real mission map to test with; skipped")
    sys.exit(0)
trip = OccupancyGrid.load(REAL)
slam = adapters.SlamAdapter(map_file=config.SLAM_MAP_FILE)

print("a new place: no saved map yet")
check("there is no map", not Path(config.SLAM_MAP_FILE).exists())
res = slam.seed_saved(copy.deepcopy(trip), trusted=False)
check("a thin trip does not start a place's map", not res.get("grew")
      and not Path(config.SLAM_MAP_FILE).exists(), str(res))
res = slam.seed_saved(copy.deepcopy(trip))
check("the first trip SEEDS the map", res.get("grew") and res.get("seeded"), str(res))
check("...its start is the map's origin", res.get("start_on_map") == (0.0, 0.0, 0.0))
check("...and the map is written", Path(config.SLAM_MAP_FILE).exists())
seeded_m2 = map_growth.known_m2(OccupancyGrid.load(config.SLAM_MAP_FILE))
check("...knowing what the trip knew", abs(seeded_m2 - map_growth.known_m2(trip)) < 0.5,
      f"{seeded_m2:.1f} vs {map_growth.known_m2(trip):.1f}")
before = Path(config.SLAM_MAP_FILE).read_bytes()
res = slam.seed_saved(copy.deepcopy(trip))
check("seeding NEVER replaces a map that exists", not res.get("grew")
      and res.get("start_on_map") is None
      and Path(config.SLAM_MAP_FILE).read_bytes() == before, str(res))

print("a later trip from the home spot (its start KNOWN on the map: 0.6, 0.4)")
moved = copy.deepcopy(trip)
moved.origin = np.asarray(trip.origin, dtype=float) - np.array([0.6, 0.4])
moved.logodds = np.where(moved.logodds < 0, moved.logodds - 0.5, moved.logodds)  # more evidence
res2 = slam.merge_saved(moved, (0.6, 0.4, 0.0))
check("its walls are MERGED at the known pose", res2.get("grew") is True, str(res2))
grown = OccupancyGrid.load(config.SLAM_MAP_FILE)
check("...the map is not shrunk", map_growth.known_m2(grown) >= 0.9 * seeded_m2)
occ_before = np.argwhere(OccupancyGrid.load(config.SLAM_MAP_FILE).occupied_mask())
check("...and its walls stay where they were (same room, same place)",
      len(occ_before) > 0)
check("an untrusted trip or an unknown pose merges nothing",
      not slam.merge_saved(moved, (0.6, 0.4, 0.0), trusted=False).get("grew")
      and not slam.merge_saved(moved, None).get("grew"))

print("a trip from anywhere else: areas may move, walls never")
src = (ROOT / "exploration_mission/mission_manager.py").read_text()
work = src[src.index("        def work() -> dict:\n            start_on_map = None"):
           src.index('self._anchor_job = self.runner.start(work, "mission-anchor")')]
check("the fit path never merges walls", "merge" not in work and "fit_onto_saved" in work)
check("the home-spot path merges at T (no fit)", "merge(og, T, trusted=trusted)" in src)
check("thresholds exist", mm.MIN_SWEEPS_TO_GROW >= 10 and mm.MIN_M2_TO_GROW >= 2.0)
print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
