"""Synthetic LiDAR for the File 21 tests: ray-cast rooms, robot-frame sweeps.

    walls = room_walls()                           # an irregular room + a box
    scan = ray_cast(walls, (1.0, 1.0, 0.3))        # {deg: mm}, 0 = ahead, CCW
    scan = grid_ray_cast(occupied, origin, cell, pose)   # on a real map grid

Robot frame exactly as home_anchor expects: bearing 0 straight ahead, bearings
counter-clockwise, x forward / y left (slam/core.py).
"""

from __future__ import annotations

import math

import numpy as np


def polygon_walls(points) -> np.ndarray:
    pts = np.asarray(points, float)
    return np.stack([pts, np.roll(pts, -1, axis=0)], axis=1)      # (N, 2, 2)


def room_walls() -> np.ndarray:
    """An irregular 5 x 4 m room with an alcove and a box: no symmetry, so a
    rotation has exactly one right answer."""
    room = polygon_walls([[0, 0], [5, 0], [5, 2.2], [4.2, 2.2], [4.2, 4.0],
                          [1.3, 4.0], [1.3, 3.1], [0, 3.1]])
    box = polygon_walls([[3.0, 0.6], [3.7, 0.6], [3.7, 1.1], [3.0, 1.1]])
    shelf = polygon_walls([[0.2, 2.2], [0.7, 2.2], [0.7, 2.5], [0.2, 2.5]])
    return np.concatenate([room, box, shelf])


def ray_cast(walls: np.ndarray, pose, max_range: float = 8.0, noise_m: float = 0.0,
             drop: float = 0.0, rng=None) -> dict[int, int]:
    rng = rng or np.random.default_rng(0)
    x, y, th = pose
    ang = th + np.radians(np.arange(360))
    d = np.stack([np.cos(ang), np.sin(ang)], axis=1)               # (360, 2)
    p, q = walls[:, 0], walls[:, 1]
    e = q - p                                                      # (N, 2)
    o = np.array([x, y])
    po = p - o
    denom = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (po[None, :, 0] * e[None, :, 1] - po[None, :, 1] * e[None, :, 0]) / denom
        u = (po[None, :, 0] * d[:, None, 1] - po[None, :, 1] * d[:, None, 0]) / denom
    ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (u >= 0) & (u <= 1)
    t = np.where(ok, t, np.inf).min(axis=1)
    out = {}
    for b in range(360):
        r = t[b]
        if not np.isfinite(r) or r > max_range or rng.random() < drop:
            continue
        r = r + (rng.normal(0, noise_m) if noise_m else 0.0)
        out[b] = int(round(r * 1000))
    return out


def grid_ray_cast(occupied: np.ndarray, origin, cell: float, pose,
                  max_range: float = 6.0, step: float = 0.02) -> dict[int, int]:
    """March each of 360 rays over an occupancy mask ([ix, iy] = [x, y])."""
    x, y, th = pose
    out = {}
    rs = np.arange(step, max_range, step)
    for b in range(360):
        a = th + math.radians(b)
        px = x + rs * math.cos(a)
        py = y + rs * math.sin(a)
        ix = np.floor((px - origin[0]) / cell).astype(int)
        iy = np.floor((py - origin[1]) / cell).astype(int)
        inside = (ix >= 0) & (ix < occupied.shape[0]) & (iy >= 0) & (iy < occupied.shape[1])
        hit = np.zeros_like(inside)
        hit[inside] = occupied[ix[inside], iy[inside]]
        k = np.argmax(hit) if hit.any() else -1
        if k >= 0:
            out[b] = int(round(rs[k] * 1000))
    return out
