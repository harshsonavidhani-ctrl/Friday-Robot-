"""Short-term obstacle memory: what she saw near her a few seconds ago.

Harsh, 2026-09-13: "the lidar obstacles misses near it" and "the obstacle memory
should be used not just kept".

WHY THIS EXISTS. Every stage of the chain decides on what the sensors report THIS
tick, and near her body the sensors report very little:

  * the LiDAR DISCARDS every return below `lidar_floor` (0.22 m, her own radius),
    so an obstacle she drives up to vanishes from the scan exactly when it matters;
  * the ToF is an 8x8 patch looking FORWARD only, so the moment she pivots, the
    thing it just saw is outside every sensor she has.

MEASURED on the 2026-09-12 14:41 run: 1357 ticks (over two minutes) emergency-
stopped with the ToF reading an object 0.088 m off her nose while the LiDAR's
forward corridor read 0.00 -- nothing valid. The planner, which steers on the
LiDAR, had nothing to steer around; she pivoted, forgot, and drove back into it.

A professional rover keeps a local costmap for exactly this: obstacles persist
for a short time after the sensor loses them. This is that, kept deliberately
small:

  * points are stored in the ODOMETRY frame (the dead-reckoned pose the loop
    already keeps), so a pivot moves them around her instead of erasing them;
  * they expire after `ttl_s` -- dead reckoning has no encoders and drifts, and a
    few seconds is where it is still trustworthy;
  * a remembered LiDAR point is CLEARED the moment a live beam at that bearing
    returns from beyond it (the thing moved, or was never there);
  * each point carries its SOURCE. ToF barrier points are a synthetic wall across
    her corridor (see Navigator._tof_constraint), so they steer the planner and the
    escape direction but are never allowed to veto a reverse: if she was commanded
    to pivot and did not physically turn, dead reckoning would swing that wall
    behind her and refuse the one escape that works.

It never touches the policy observation or the persistent map. It only ADDS
obstacles, so everything it feeds can only become more careful, never less.
"""
from __future__ import annotations

import math
import time

import numpy as np

TTL_S = 4.0            # how long a point is trusted after it was last seen
RANGE_M = 1.0          # only LiDAR returns this close are worth remembering
REPORT_RADIUS_M = 1.2  # points further than this from her are not reported
CELL_M = 0.05          # one point per cell, newest wins
MAX_POINTS = 800       # hard bound on memory and per-tick cost
CLEAR_MARGIN_M = 0.10  # a live return must be this much beyond a point to clear it

SRC_LIDAR = 0
SRC_TOF = 1


class NearObstacleMemory:
    """Obstacle points remembered for a few seconds in the odometry frame."""

    def __init__(self, cfg, ttl_s: float | None = None,
                 range_m: float | None = None) -> None:
        self.cfg = cfg
        self.ttl_s = float(ttl_s if ttl_s is not None
                           else getattr(cfg, "near_memory_ttl_s", TTL_S))
        self.range_m = float(range_m if range_m is not None
                             else getattr(cfg, "near_memory_range_m", RANGE_M))
        self._xy = np.zeros((0, 2), np.float64)   # odometry frame
        self._t = np.zeros(0, np.float64)         # when last seen
        self._src = np.zeros(0, np.int8)
        # Telemetry: how much work it did, so a run can show it was USED.
        self.updates = 0
        self.cleared = 0
        self.expired = 0

    def __len__(self) -> int:
        return int(self._xy.shape[0])

    # ------------------------------------------------------------- frames ----
    @staticmethod
    def _to_odom(pose, pts_robot: np.ndarray) -> np.ndarray:
        x, y, th = (float(p) for p in pose)
        c, s = math.cos(th), math.sin(th)
        px, py = pts_robot[:, 0], pts_robot[:, 1]
        return np.stack([x + px * c - py * s, y + px * s + py * c], axis=1)

    @staticmethod
    def _to_robot(pose, pts_odom: np.ndarray) -> np.ndarray:
        x, y, th = (float(p) for p in pose)
        c, s = math.cos(th), math.sin(th)
        dx, dy = pts_odom[:, 0] - x, pts_odom[:, 1] - y
        return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)

    # ------------------------------------------------------------- update ----
    def update(self, pose, scan_m: np.ndarray | None,
               blind_beams: np.ndarray | None = None,
               tof_points: np.ndarray | None = None,
               now: float | None = None) -> None:
        """Forget what is stale or contradicted, then remember what is near now."""
        now = time.monotonic() if now is None else float(now)
        self.updates += 1
        cfg = self.cfg
        floor = float(getattr(cfg, "lidar_floor", cfg.robot_radius))
        max_range = float(cfg.lidar_max_range)

        # 1. EXPIRE. Dead reckoning drifts; an old point is at the wrong address.
        if self._t.size:
            keep = (now - self._t) <= self.ttl_s
            self.expired += int(np.count_nonzero(~keep))
            self._xy, self._t, self._src = self._xy[keep], self._t[keep], self._src[keep]

        blind = np.zeros(0, np.int32) if blind_beams is None else np.asarray(
            blind_beams, np.int32)
        scan = None if scan_m is None else np.asarray(scan_m, np.float64)

        # 2. CLEAR what a live beam now sees THROUGH. Only a VALID return beyond the
        # point counts: a beam with no return (max_range) may be a dark, absorbing
        # surface, and treating that as "clear" would erase exactly the thing the ToF
        # saw. Points below the LiDAR floor cannot be contradicted by it at all -- that
        # blindness is the reason they are remembered.
        if scan is not None and self._xy.shape[0]:
            n = scan.size
            rel = self._to_robot(pose, self._xy)
            r = np.hypot(rel[:, 0], rel[:, 1])
            beam = (np.round(np.arctan2(rel[:, 1], rel[:, 0]) % (2 * math.pi)
                             / (2 * math.pi / n)).astype(np.int64)) % n
            live = scan[beam]
            seen_past = ((r >= floor) & (live >= floor) & (live < max_range)
                         & (live > r + CLEAR_MARGIN_M))
            if blind.size:
                seen_past &= ~np.isin(beam, blind)
            if seen_past.any():
                self.cleared += int(np.count_nonzero(seen_past))
                keep = ~seen_past
                self._xy, self._t, self._src = (self._xy[keep], self._t[keep],
                                                self._src[keep])

        # 3. REMEMBER what is near now.
        new_xy, new_src = [], []
        if scan is not None:
            n = scan.size
            bearing = np.arange(n) * (2 * math.pi / n)
            ok = (np.isfinite(scan) & (scan >= floor) & (scan < max_range)
                  & (scan <= self.range_m))
            if blind.size:
                ok[blind[(blind >= 0) & (blind < n)]] = False
            if ok.any():
                pts = np.stack([scan[ok] * np.cos(bearing[ok]),
                                scan[ok] * np.sin(bearing[ok])], axis=1)
                new_xy.append(self._to_odom(pose, pts))
                new_src.append(np.full(pts.shape[0], SRC_LIDAR, np.int8))
        if tof_points is not None and len(tof_points):
            pts = np.asarray(tof_points, np.float64).reshape(-1, 2)
            new_xy.append(self._to_odom(pose, pts))
            new_src.append(np.full(pts.shape[0], SRC_TOF, np.int8))
        if new_xy:
            add_xy = np.vstack(new_xy)
            self._xy = np.vstack([self._xy, add_xy])
            self._t = np.concatenate([self._t, np.full(add_xy.shape[0], now)])
            self._src = np.concatenate([self._src] + new_src)
            self._compact()

    def _compact(self) -> None:
        """One point per cell (newest wins), then the newest MAX_POINTS."""
        if not self._xy.shape[0]:
            return
        cells = np.floor(self._xy / CELL_M).astype(np.int64)
        key = cells[:, 0] * 1_000_003 + cells[:, 1] + self._src.astype(np.int64) * 7
        order = np.argsort(-self._t, kind="stable")          # newest first
        _, first = np.unique(key[order], return_index=True)
        idx = order[first]
        idx = idx[np.argsort(-self._t[idx], kind="stable")][:MAX_POINTS]
        self._xy, self._t, self._src = self._xy[idx], self._t[idx], self._src[idx]

    # ------------------------------------------------------------- queries ---
    def points(self, pose, sources: tuple[int, ...] = (SRC_LIDAR, SRC_TOF),
               radius_m: float = REPORT_RADIUS_M) -> np.ndarray:
        """Remembered points near her, in the ROBOT frame. (N, 2) float32."""
        if not self._xy.shape[0]:
            return np.zeros((0, 2), np.float32)
        mask = np.isin(self._src, np.asarray(sources, np.int8))
        if not mask.any():
            return np.zeros((0, 2), np.float32)
        rel = self._to_robot(pose, self._xy[mask])
        near = np.hypot(rel[:, 0], rel[:, 1]) <= radius_m
        return rel[near].astype(np.float32)

    def virtual_scan(self, pose, n_beams: int,
                     sources: tuple[int, ...] = (SRC_LIDAR, SRC_TOF),
                     radius_m: float = REPORT_RADIUS_M) -> np.ndarray:
        """Per-beam nearest remembered range, lidar_max_range where there is none.

        In the scan's own convention (beam 0 down the nose), so it can be fused with
        the live scan by `np.minimum` for any stage that reads a scan. Ranges here MAY
        be below lidar_floor: that is the point of remembering them.
        """
        out = np.full(n_beams, float(self.cfg.lidar_max_range), np.float32)
        pts = self.points(pose, sources, radius_m)
        if not pts.shape[0]:
            return out
        r = np.hypot(pts[:, 0], pts[:, 1])
        beam = (np.round(np.arctan2(pts[:, 1], pts[:, 0]) % (2 * math.pi)
                         / (2 * math.pi / n_beams)).astype(np.int64)) % n_beams
        np.minimum.at(out, beam, r.astype(np.float32))
        return out
