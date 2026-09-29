"""Where the mission started, and whether she is back (File 21 step 3, spec 4.9).

A SIGNATURE is what the start spot looks like to the LiDAR: 360 numbers, the
median range at each whole degree over a few sweeps taken standing still,
clipped at max_range_m, NaN where nothing came back. It is the cheap, coarse
check (SMAD-style: "does this look like the start, and turned by how much?");
the precise pose always comes from SLAM.

FRAME. Every sweep here is in the ROBOT frame: bearing 0 is straight ahead and
bearings grow counter-clockwise (x forward, y left -- slam/core.py's
convention). The RL run publishes its sweeps that way already; a raw sweep
from friday-mcp's LiDAR is turned into it by `raw_to_robot` (raw - 51.8 deg,
config.LIDAR_FORWARD_DEG).

SHIFT. `similarity(cur, ref)` tries all 360 rotations at once and reports
`shift_deg` = how far the robot has TURNED since the reference, in the same
sense as a heading (counter-clockwise positive, -180..180]. A robot turned
+37 deg sees the world 37 deg further clockwise, so cur[b] = ref[b + 37].

Pure numpy. No hardware; the live capture is the mission manager's job
(adapters.LidarAdapter gives it robot-frame sweeps).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

log = logging.getLogger("friday.exploration_mission.home_anchor")

N_BINS = 360
# idx[k, j] = (j - k) % 360, so ref[idx[k]] == np.roll(ref, k): every rotation
# of the reference as one (360, 360) gather. Built once.
_IDX = (np.arange(N_BINS)[None, :] - np.arange(N_BINS)[:, None]) % N_BINS


def raw_to_robot(scan: dict, forward_deg: float) -> dict[float, int]:
    """friday-mcp's raw {bearing: mm} -> robot frame (0 = ahead, CCW)."""
    return {(float(b) - float(forward_deg)) % 360.0: int(mm)
            for b, mm in scan.items() if mm is not None}


def scan_to_bins(scan: dict | None, max_range_m: float | None = None) -> np.ndarray:
    """Robot-frame {bearing_deg: mm} -> float32[360] metres, NaN = no return.

    Bearings round to the nearest whole degree; two in one bin keep the
    nearer (a return is evidence, and the nearer one is what is there).
    """
    out = np.full(N_BINS, np.nan, dtype=np.float32)
    if not scan:
        return out
    for deg, mm in scan.items():
        try:
            d, r = int(round(float(deg))) % N_BINS, float(mm) / 1000.0
        except (TypeError, ValueError):
            continue
        if not (r > 0 and math.isfinite(r)):
            continue
        if max_range_m is not None:
            r = min(r, float(max_range_m))
        if math.isnan(out[d]) or r < out[d]:
            out[d] = r
    return out


def signature(scans: Iterable[dict], max_range_m: float = 8.0) -> np.ndarray:
    """Per-degree median over the sweeps (NaNs ignored), clipped, float32[360].
    A bin that no sweep saw stays NaN: "no return", never "far"."""
    scans = list(scans)
    if not scans:
        return np.full(N_BINS, np.nan, dtype=np.float32)
    stack = np.stack([scan_to_bins(s, max_range_m) for s in scans])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN bin -> NaN
        med = np.nanmedian(stack, axis=0)
    return np.minimum(med, max_range_m).astype(np.float32)


def mean_pose(poses: Iterable) -> tuple[float, float, float]:
    """Mean (x, y) and the CIRCULAR mean of theta, so +179 and -179 average to
    180, not 0."""
    p = np.asarray(list(poses), dtype=np.float64)
    if p.size == 0:
        raise ValueError("no poses")
    th = math.atan2(float(np.sin(p[:, 2]).mean()), float(np.cos(p[:, 2]).mean()))
    return float(p[:, 0].mean()), float(p[:, 1].mean()), th


@dataclass
class Similarity:
    mad_m: float           # mean |cur - ref| over bins valid in both, best shift
    overlap: float         # fraction of the 360 bins valid in both, at that shift
    shift_deg: float       # how far she has turned since ref (CCW +), -180..180
    match: bool

    def as_dict(self) -> dict:
        return {"mad_m": round(self.mad_m, 3), "overlap": round(self.overlap, 3),
                "shift_deg": round(self.shift_deg, 1), "match": self.match}


def similarity(cur: np.ndarray, ref: np.ndarray, min_overlap: float = 0.6,
               match_mad_m: float = 0.20) -> Similarity:
    """Best rotation of `ref` against `cur`, all 360 at once (spec 4.9).

    Among the shifts with overlap >= min_overlap, the one with the lowest MAD
    wins; match = that MAD <= match_mad_m. No shift with enough overlap ->
    no match, MAD inf.
    """
    cur = np.asarray(cur, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    rolled = ref[_IDX]                                  # (360 shifts, 360 bins)
    valid = np.isfinite(rolled) & np.isfinite(cur)[None, :]
    count = valid.sum(axis=1)
    diff = np.where(valid, np.abs(rolled - cur[None, :]), 0.0).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mad = np.where(count > 0, diff / np.maximum(count, 1), np.inf)
    overlap = count / float(N_BINS)
    mad = np.where(overlap >= min_overlap, mad, np.inf)
    k = int(np.argmin(mad))
    best = float(mad[k])
    # roll(ref, k) matching cur means cur[b] = ref[b - k]: she turned by -k.
    shift = (-k + 180) % 360 - 180
    if shift == -180:
        shift = 180
    return Similarity(mad_m=best, overlap=float(overlap[k]),
                      shift_deg=float(shift),
                      match=bool(math.isfinite(best) and best <= match_mad_m))


def heading_agrees(shift_deg: float, heading_change_rad: float,
                   tolerance_deg: float = 20.0) -> bool:
    """spec 4.9 sanity: the signature's turn and SLAM's heading change agree.
    A disagreement is logged by the caller, never acted on here."""
    d = (shift_deg - math.degrees(heading_change_rad) + 180.0) % 360.0 - 180.0
    return abs(d) <= tolerance_deg


@dataclass
class Anchor:
    """A captured start: pose (mean over the capture) + signature."""
    x: float
    y: float
    theta: float
    signature: np.ndarray
    scans: int

    def as_record(self) -> dict:
        return {"x": self.x, "y": self.y, "theta": self.theta,
                "signature": [None if not math.isfinite(v) else round(float(v), 4)
                              for v in self.signature],
                "scans": self.scans}


def capture(scans: list[dict], poses: list, max_range_m: float = 8.0) -> Anchor:
    """spec 4.9 capture(): the sweeps and poses already taken standing still
    (the caller guarantees motors stopped and SLAM tracking)."""
    if not scans:
        raise ValueError("no sweeps to capture from")
    x, y, th = mean_pose(poses)
    return Anchor(x, y, th, signature(scans, max_range_m), len(scans))
