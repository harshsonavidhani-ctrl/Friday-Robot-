"""The map while she is making it, for the dashboard.

Harsh, 2026-10-30: the judges said it should update in the app while she moves.
The run saves state/rl_map.npz only when it ends, so the page showed the last run
until this one was over.

Its own thread, never the control loop: every period_s it takes the same snapshot
the planner takes and writes one small file next to the map, atomically. zlib lets
go of the GIL for the part that costs anything, and this thread sits below the
loop. A failure costs a stale picture and nothing else. The file is removed when
the run ends, which tells the page to go back to the saved map.

READ BY dashboard/map_view.py. It is never loaded by a run and never becomes a map.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("live_map")
PERIOD_S = 1.0      # "like gps": her position moves, the map grows behind her
NICE = 10


class LiveMap(threading.Thread):
    def __init__(self, world, path: Path, period_s: float = PERIOD_S) -> None:
        super().__init__(name="live-map", daemon=True)
        self.world, self.path, self.period = world, Path(path), period_s
        self._stop_event = threading.Event()
        self.writes = 0
        self.failed = False

    def write_once(self) -> bool:
        w = self.world
        snap = w.grid_snapshot()
        if snap is None:
            return False
        with w._lock:
            cells = np.array(list(w._visits.keys()), np.int64).reshape(-1, 2)
            cell_m = float(w._visit_cell)
            pose = np.array(w.pose, np.float64)
        tmp = self.path.with_name(self.path.name + ".tmp.npz")
        np.savez_compressed(tmp, logodds=snap.logodds, origin=snap.origin,
                            cell=snap.cell, scans=snap.scans, updated_at=time.time(),
                            cells=cells, cell_m=cell_m, pose=pose)
        os.replace(tmp, self.path)
        self.writes += 1
        return True

    def run(self) -> None:
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), NICE)
        except OSError:
            pass
        while not self._stop_event.wait(self.period):
            try:
                self.write_once()
            except Exception as exc:                  # noqa: BLE001
                if not self.failed:
                    LOGGER.warning("live map not written (%s: %s)", type(exc).__name__, exc)
                self.failed = True

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self.path.unlink()
        except OSError:
            pass
