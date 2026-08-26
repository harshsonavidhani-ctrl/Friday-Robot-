"""Things that are there right now, which are not necessarily part of the room.

A person walking past returns a beam exactly as convincingly as a wall does.
The difference is not in the reading, it is in the PERSISTENCE: a wall is
still there on the next sweep, and on the one after that, and from a different
angle. So a return does not go onto the permanent map on sight -- it comes
here first, and is promoted only once the same cell has been hit in
config.SLAM_CONFIRM_SCANS separate sweeps.

WHAT THIS IS NOT. This is not a filter that throws readings away. An
unconfirmed return is a real object and Friday must not drive into it; it is
simply not believed to be FURNITURE yet. So `blocking_points` returns
everything seen recently, confirmed or not, and the navigation layer avoids
all of it. Only the map is choosy.

THE TWO TIMESCALES, AND WHY THEY DIFFER.

  promotion   3 sweeps   -- slow to write to the map
  avoidance   4 seconds  -- an obstacle blocks the path until it stops being
                            seen, which only has to outlast the gap between
                            sweeps, because a thing still there is seen again
  forgetting  ~12 sweeps -- once on the map, the beams that later pass
                            through carve it free (slam/grid.py's clamp)

Slow to add and slow to forget is the useful asymmetry. The opposite -- quick
to add, quick to forget -- gives a map that flickers with every passer-by and
a robot that confidently drives at a doorway a suitcase is still standing in.

CELLS ARE ABSOLUTE, NOT GRID INDICES. Keys here are floor(world / cell) with
no origin subtracted, because OccupancyGrid's array origin moves every time
the map grows and these counts must survive that.
"""

from __future__ import annotations

import time

import numpy as np

import config


class TransientLayer:
    """Per-cell evidence about returns that are not yet part of the map."""

    def __init__(self, cell_m: float | None = None) -> None:
        self.cell = float(cell_m if cell_m is not None else config.SLAM_CELL_M)
        # (ix, iy) -> [sweeps seen in, first seen, last seen]
        self._seen: dict[tuple[int, int], list[float]] = {}
        self.sweeps = 0

    def _cells(self, points: np.ndarray) -> np.ndarray:
        """World metres -> absolute lattice cells, de-duplicated.

        De-duplicated because one sweep is ONE observation of a cell however
        many of its beams land there -- otherwise a nearby object, which soaks
        up thirty beams, is 'confirmed' by a single sweep and the whole point
        of the gate is lost.
        """
        if points.size == 0:
            return np.empty((0, 2), dtype=np.int64)
        cells = np.floor(np.atleast_2d(points) / self.cell).astype(np.int64)
        return np.unique(cells, axis=0)

    def observe(self, points: np.ndarray,
                now: float | None = None) -> np.ndarray:
        """Record one sweep's returns. Returns the subset that is CONFIRMED.

        The returned array is what may be written to the permanent map: the
        points whose cell has now been hit in enough separate sweeps.
        """
        now = time.time() if now is None else now
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        self.sweeps += 1
        if points.size == 0:
            self.expire(now)
            return np.empty((0, 2), dtype=np.float64)

        for ix, iy in self._cells(points):
            row = self._seen.get((int(ix), int(iy)))
            if row is None:
                self._seen[(int(ix), int(iy))] = [1.0, now, now]
            else:
                row[0] += 1.0
                row[2] = now

        self.expire(now)
        keep = self.confirmed_mask(points)
        return points[keep]

    def count_at(self, points: np.ndarray) -> np.ndarray:
        """How many separate sweeps each point's cell has been seen in."""
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if points.size == 0:
            return np.empty(0, dtype=np.int64)
        cells = np.floor(points / self.cell).astype(np.int64)
        return np.array([int(self._seen.get((int(a), int(b)), (0,))[0])
                         for a, b in cells], dtype=np.int64)

    def confirmed_mask(self, points: np.ndarray) -> np.ndarray:
        return self.count_at(points) >= config.SLAM_CONFIRM_SCANS

    def expire(self, now: float | None = None) -> int:
        """Drop cells not seen within the avoidance lifetime. Returns how many.

        Expiring resets the promotion count too, and that is intended: three
        sweeps spread over a minute, as a person wanders in and out of the
        same doorway, is not evidence of a wall. The confirmations have to be
        CONSECUTIVE enough to overlap.
        """
        now = time.time() if now is None else now
        cutoff = now - config.SLAM_TRANSIENT_TTL_S
        stale = [key for key, row in self._seen.items() if row[2] < cutoff]
        for key in stale:
            del self._seen[key]
        return len(stale)

    def blocking_points(self, now: float | None = None) -> np.ndarray:
        """Every cell seen recently, as world centres. For AVOIDANCE.

        Confirmed or not, on the map or not -- if a beam came back from there
        in the last few seconds, nothing should drive through it.
        """
        self.expire(now)
        if not self._seen:
            return np.empty((0, 2), dtype=np.float64)
        cells = np.array(list(self._seen.keys()), dtype=np.float64)
        return (cells + 0.5) * self.cell

    def unconfirmed_points(self, now: float | None = None) -> np.ndarray:
        """Recent returns that have NOT earned a place on the map.

        This is the closest thing Friday has to "someone is standing there":
        a real return, seen too few times to be the room. Useful to narrate
        and useful to avoid; never useful to map.
        """
        self.expire(now)
        rows = [cell for cell, row in self._seen.items()
                if row[0] < config.SLAM_CONFIRM_SCANS]
        if not rows:
            return np.empty((0, 2), dtype=np.float64)
        return (np.array(rows, dtype=np.float64) + 0.5) * self.cell

    def stats(self) -> dict:
        confirmed = sum(1 for row in self._seen.values()
                        if row[0] >= config.SLAM_CONFIRM_SCANS)
        return {"tracked": len(self._seen), "confirmed": confirmed,
                "unconfirmed": len(self._seen) - confirmed,
                "sweeps": self.sweeps}
