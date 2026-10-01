"""The mission's own map, built from what the RL run publishes (File 21 step 7).

WHY THE MISSION HAS TO BUILD ONE. The spec assumed a SLAM map growing beside
the RL explorer. On this robot the run holds the LiDAR, its own map is dead
reckoned (never plan on it), and the saved map (state/map.npz) is not written
during a run. What the run does publish, once a second, is the start
tracker's pose and the sweep it was computed from. So the mission folds those
sweeps into a slam OccupancyGrid of its own, in the START FRAME: the mission
start is (0, 0, 0) and every area, crumb and frontier of a mission lives in
that frame until the mission is anchored onto the saved map.

⚠ WHICH SWEEPS GO IN (`policy`). The start tracker matches each sweep against
the START's own sweep, so its pose is scan-matched while the start room is in
view and dead reckoning beyond it (1-4 m out over a run, per rl/way_back.py).
A sweep folded at a dead-reckoned pose draws walls where they are not.

    "matched"  only fixes the tracker matched (and the first).  [default]
    "all"      every fix, matched or not.

"matched" is the conservative choice, not a measured one: it can never smear
the map, and it cannot map a room the start cannot see. Which is right for a
real house has to be decided on a recording of a real run
(exploration_mission.recorder) -- `replay()` below is for exactly that.
"""

from __future__ import annotations

import math
from typing import Iterable

POLICIES = ("matched", "all")


def compose(a, b) -> tuple[float, float, float]:
    """Pose a, then b expressed in a's frame."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + b[0] * c - b[1] * s, a[1] + b[0] * s + b[1] * c,
            (a[2] + b[2] + math.pi) % (2 * math.pi) - math.pi)


def inverse(p) -> tuple[float, float, float]:
    c, s = math.cos(p[2]), math.sin(p[2])
    return (-(p[0] * c + p[1] * s), p[0] * s - p[1] * c,
            (-p[2] + math.pi) % (2 * math.pi) - math.pi)


def transform_points(points: Iterable, t) -> list[list[float]]:
    """Points in frame B -> frame A, where `t` is B's pose in A."""
    c, s = math.cos(t[2]), math.sin(t[2])
    return [[t[0] + float(p[0]) * c - float(p[1]) * s,
             t[1] + float(p[0]) * s + float(p[1]) * c] for p in points]


class MissionMap:
    def __init__(self, slam, policy: str = "matched"):
        if policy not in POLICIES:
            raise ValueError(f"policy is one of {POLICIES}")
        self.slam = slam
        self.policy = policy
        self.og = slam.new_grid()
        self.seen = 0            # fixes offered
        self.used = 0            # sweeps folded in
        self.matched = 0
        self._last_t = None

    def add(self, fix: dict) -> bool:
        """Fold one published fix in. True if its sweep went into the map."""
        if fix.get("t") == self._last_t or not fix.get("scan"):
            return False
        self._last_t = fix.get("t")
        self.seen += 1
        matched = str(fix.get("reason", "")).startswith("matched")
        self.matched += int(matched)
        if self.policy == "matched" and not matched and self.used > 0:
            return False
        pose = tuple(float(v) for v in fix["pose"])
        # The published sweep is already robot frame (0 = ahead): no offset.
        self.og.integrate_scan(fix["scan"], pose, forward_deg=0.0)
        self.used += 1
        return True

    def restore(self, og) -> None:
        """Carry on with a map saved before a power cut."""
        self.og = og
        self.used = max(int(getattr(og, "scans", 0) or 0), 1)

    def save(self, path) -> bool:
        """slam's own atomic save (temp file, fsync, rename)."""
        if self.used == 0:
            return False
        self.og.save(path)
        return True

    def planner_grid(self):
        """The map as File 14's planner would see it (despeckled, normalised),
        or None before the first sweep."""
        if self.used == 0:
            return None
        return self.slam.planner_grid(self.og)

    def stats(self) -> dict:
        return {"policy": self.policy, "seen": self.seen, "used": self.used,
                "matched": self.matched}


def replay(rows: Iterable[dict], slam, policy: str = "matched") -> MissionMap:
    """Build the map a mission would have built from a recording."""
    m = MissionMap(slam, policy)
    for row in rows:
        m.add(row)
    return m
