"""Five test rooms for an A/B of two versions of Friday's exploration code.

Each world subclasses the version's own pi_deploy.SimulatedWorld, so the control
loop is exactly that version's; only the room changes. The world also keeps the
score: bumps (the 0.22 m body touching anything, with where on the body), time
standing still, standing still with a clear way ahead (hesitating), how fast she
reacts to a person stepping in front, and how much floor the LiDAR saw.

Times are in simulated seconds (the dt each step is called with), so the same
scoring works for the plain sim (one step per 0.1 s control tick) and the BODY
sim (the drivetrain model stepping at 50 Hz).
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

R_BODY = 0.22          # cfg.robot_radius in both versions


def rect(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def poly(pts):
    return [(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]


def chair(cx, cy, s=0.42, r=0.02):
    h = s / 2
    return [(cx - h, cy - h, r), (cx + h, cy - h, r), (cx - h, cy + h, r), (cx + h, cy + h, r)]


def build(name):
    """(walls, circles, low_walls, low_circles, popup). Start: (0,0), facing +x."""
    walls, circles, low_w, low_c = [], [], [], []
    popup = False
    if name == "corridor":          # 1.4 m wide, 11 m, an L turn, a door recess
        walls += poly([(-1, -0.7), (4, -0.7), (4, -1.0), (5, -1.0), (5, -0.7),
                       (10, -0.7), (10, 6), (8.6, 6), (8.6, 0.7), (-1, 0.7)])
    elif name == "chairs":          # dining room: table, 6 chairs, sofa, cupboard
        walls += rect(-1.2, -2.0, 4.2, 2.0)
        walls += rect(-0.8, 1.3, 1.4, 2.0)            # sofa against the wall
        walls += rect(3.6, -2.0, 4.2, -0.6)           # cupboard
        tx, ty = 2.1, 0.1
        circles += [(tx - 0.6, ty - 0.4, 0.025), (tx + 0.6, ty - 0.4, 0.025),
                    (tx - 0.6, ty + 0.4, 0.025), (tx + 0.6, ty + 0.4, 0.025)]
        for cx, cy in ((tx - 0.3, ty - 0.75), (tx + 0.3, ty - 0.75), (tx - 0.3, ty + 0.75),
                       (tx + 0.3, ty + 0.75), (0.6, -1.4), (3.4, 1.4)):
            circles += chair(cx, cy)
    elif name == "two_rooms":       # 4x4 + 4x4, joined by a 0.8 m door off centre
        walls += poly([(-2, -2), (6, -2), (6, 2), (-2, 2)])
        walls += [((2.0, -2), (2.0, 0.6)), ((2.0, 1.4), (2.0, 2))]
        circles += [(0.5, -1.2, 0.3), (-1.0, 1.0, 0.35), (4.2, -0.8, 0.4), (3.2, 1.2, 0.25)]
    elif name == "low_things":      # things under the LiDAR's plane: ToF only
        walls += rect(-1.5, -2.5, 4.5, 2.5)
        low_c += [(1.4, 0.15, 0.09), (2.6, -1.1, 0.09), (0.8, 1.5, 0.12), (3.3, 0.9, 0.1)]
        low_w += rect(1.9, 1.6, 3.6, 2.5)             # low sofa base / bed frame
        circles += [(3.0, -2.0, 0.3)]
    elif name == "popup":           # open room; a person steps in front, 3 times
        walls += rect(-4, -4, 4, 4)
        circles += [(2.0, 2.0, 0.3), (-2.0, -1.5, 0.3)]
        popup = True
    else:
        raise ValueError(name)
    return walls, circles, low_w, low_c, popup


def _ray_segments(ox, oy, dx, dy, segs):
    """Distance along each ray (dx,dy arrays) to the nearest segment; inf if none."""
    t = np.full(dx.shape, np.inf)
    for (ax, ay), (bx, by) in segs:
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        ok = np.abs(den) > 1e-12
        wx, wy = ax - ox, ay - oy
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (wx * ey - wy * ex) / den
            uu = (wx * dy - wy * dx) / den
        hit = ok & (tt > 1e-6) & (uu >= 0) & (uu <= 1)
        t = np.where(hit & (tt < t), tt, t)
    return t


def _ray_circles(ox, oy, dx, dy, circles):
    t = np.full(dx.shape, np.inf)
    for cx, cy, r in circles:
        px, py = cx - ox, cy - oy
        proj = px * dx + py * dy
        perp2 = px * px + py * py - proj * proj
        inside = (proj > 0) & (perp2 < r * r)
        tt = proj - np.sqrt(np.clip(r * r - perp2, 0, None))
        t = np.where(inside & (tt > 0) & (tt < t), tt, t)
    return t


def _nearest(x, y, segs, circles, kind):
    """(distance, bearing_world_rad, kind) of the closest surface."""
    best = (math.inf, 0.0, kind)
    for (ax, ay), (bx, by) in segs:
        ex, ey = bx - ax, by - ay
        L2 = ex * ex + ey * ey
        u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * ex + (y - ay) * ey) / L2))
        px, py = ax + u * ex, ay + u * ey
        d = math.hypot(x - px, y - py)
        if d < best[0]:
            best = (d, math.atan2(py - y, px - x), kind)
    for cx, cy, r in circles:
        d = math.hypot(x - cx, y - cy) - r
        if d < best[0]:
            best = (d, math.atan2(cy - y, cx - x), kind)
    return best


def _clearance(x, y, segs, circles):
    return min(_nearest(x, y, segs, circles, "")[0], math.inf)


LAST = [None]


def make_world_class(base, world, out_path):
    walls, circles, low_w, low_c, popup = build(world)

    class ABWorld(base):
        def __init__(self, cfg, seed=0):
            super().__init__(cfg, seed=seed)
            LAST[0] = self
            self.ticks = 0
            self.t = 0.0
            self.person = None          # (x, y, r, born_t)
            self.popups = []
            self.contacts = []
            self.score = dict(world=world, ticks=0, bumps=0, contact_s=0.0,
                              min_clearance=9.0, still_s=0.0, hesitate_s=0.0,
                              longest_still_s=0.0, driven_m=0.0, turned_rad=0.0,
                              reverse_s=0.0, first_move_s=None, v_max_seen=0.0)
            self._still_run = 0.0
            self._in_contact = False
            self._seen = set()
            self._visited = set()
            self._scan_n = 0
            self._dumped_t = -9.0

        # ---- geometry -------------------------------------------------------
        def _circles(self, low):
            c = list(circles)
            if self.person is not None:
                c.append(self.person[:3])
            if low:
                c += low_c
            return c

        def _cast(self, ang, low):
            x, y = float(self.pose[0]), float(self.pose[1])
            dx, dy = np.cos(ang), np.sin(ang)
            segs = walls + (low_w if low else [])
            return np.minimum(_ray_segments(x, y, dx, dy, segs),
                              _ray_circles(x, y, dx, dy, self._circles(low)))

        def lidar(self):
            c = self.cfg
            th = float(self.pose[2])
            sub = 10                                   # 0.5 deg rays, min per 5 deg bin
            n = c.n_beams * sub
            ang = th + (np.arange(n) - sub // 2) * (2 * np.pi / n)
            t = self._cast(ang, low=False)
            rng = np.minimum(t, c.lidar_max_range).reshape(c.n_beams, sub).min(axis=1)
            self._scan_n += 1
            if self._scan_n % 5 == 0:                  # floor the LiDAR has seen
                x, y = float(self.pose[0]), float(self.pose[1])
                tt = np.minimum(t, c.lidar_max_range)[::2]
                aa = ang[::2]
                for d in np.arange(0.1, c.lidar_max_range, 0.1):
                    m = tt > d
                    if not m.any():
                        break
                    gx = np.floor((x + d * np.cos(aa[m])) / 0.1).astype(int)
                    gy = np.floor((y + d * np.sin(aa[m])) / 0.1).astype(int)
                    self._seen.update(zip(gx.tolist(), gy.tolist()))
            noise = self.rng.normal(0, 0.015, c.n_beams)
            out = np.clip(rng + noise, 0.0, c.lidar_max_range).astype(np.float32)
            # THE CHASSIS BEAMS (2026-10-27): the real LD19 thread masks the
            # self-mask bearings, so those bins go out as lidar_max_range. This sim
            # used to hand the policy TRUE ranges there -- inputs it never gets on
            # the robot or in training (round 3 reversed 30%+ here, 0.1% in
            # training). SIM_MASK=0 = the old behaviour, to reproduce old results.
            if os.environ.get("SIM_MASK", "1") != "0":
                if getattr(self, "_blind", None) is None:
                    mod = sys.modules[base.__module__]
                    mask = getattr(c, "self_mask_deg", ()) or ()
                    self._blind = (np.asarray(mod.masked_beam_indices(
                        mask, getattr(c, "lidar_offset_deg", 0.0), c.n_beams), int)
                        if mask and hasattr(mod, "masked_beam_indices") else np.zeros(0, int))
                out[self._blind] = c.lidar_max_range
            return out

        def tof(self):
            """8x8 OBSTACLE ranges in metres, as the hardware thread publishes after
            its floor fit: top half sees tall things, bottom half low ones too."""
            c = self.cfg
            th = float(self.pose[2])
            cols = th + np.radians(np.linspace(22.5, -22.5, 8))   # 45 deg across
            hi = np.minimum(self._cast(cols, low=False), c.tof_max_range)
            lo = np.minimum(self._cast(cols, low=True), c.tof_max_range)
            depth = np.empty((8, 8), np.float32)
            depth[0:4, :] = hi[None, :]
            depth[4:8, :] = lo[None, :]
            return depth + self.rng.normal(0, 0.005, (8, 8)).astype(np.float32)

        def frame(self):
            h, w = self.cfg.cam_height, self.cfg.cam_width
            g = np.linspace(150.0, 95.0, h, dtype=np.float32)[:, None]
            img = g + self.rng.normal(0.0, 2.5, (h, w)).astype(np.float32)
            return img.clip(0, 255).astype(np.uint8)

        # ---- motion + score -------------------------------------------------
        def step(self, v, omega, dt):
            s = self.score
            self.ticks += 1
            self.t += dt
            s["ticks"] = self.ticks
            x0, y0, th0 = (float(p) for p in self.pose)
            super().step(v, omega, dt)
            x1, y1, th1 = (float(p) for p in self.pose)
            # where she went, once a simulated second (2026-10-27: why no model
            # gets through two_rooms' door)
            if int(self.t) != int(self.t - dt):
                s.setdefault("path", []).append((round(x1, 2), round(y1, 2)))
            segs = walls + low_w
            circ = self._circles(low=True)
            clear1 = _clearance(x1, y1, segs, circ)
            moved = math.hypot(x1 - x0, y1 - y0)
            if clear1 < R_BODY and moved > 1e-9 and clear1 < _clearance(x0, y0, segs, circ) + 1e-9:
                self.pose[0], self.pose[1] = x0, y0      # blocked: the body is touching
                s["contact_s"] += dt
                if not self._in_contact:
                    s["bumps"] += 1
                    pc = list(circles) + ([self.person[:3]] if self.person else [])
                    d, brg, kind = min(_nearest(x0, y0, walls, pc, "tall"),
                                       _nearest(x0, y0, low_w, low_c, "low"))
                    rel = math.degrees(math.atan2(math.sin(brg - th0), math.cos(brg - th0)))
                    self.contacts.append(dict(t=round(self.t, 1), kind=kind,
                                              bearing=round(rel), v=round(v, 2),
                                              w=round(omega, 2)))
                self._in_contact = True
                moved = 0.0
            else:
                self._in_contact = False
            x, y = float(self.pose[0]), float(self.pose[1])
            s["min_clearance"] = round(min(s["min_clearance"], _clearance(x, y, segs, circ)), 3)
            s["driven_m"] += moved
            s["v_max_seen"] = max(s["v_max_seen"], round(v, 3))
            s["turned_rad"] += abs(math.atan2(math.sin(th1 - th0), math.cos(th1 - th0)))
            if v < -0.02:
                s["reverse_s"] += dt
            still = abs(v) < 0.02 and abs(omega) < 0.05
            if not still and s["first_move_s"] is None:
                s["first_move_s"] = round(self.t, 1)
            if still:
                s["still_s"] += dt
                self._still_run += dt
                ahead = self._cast(np.array([th1 - 0.35, th1, th1 + 0.35]), low=True).min()
                if ahead > 0.8:
                    s["hesitate_s"] += dt
                s["longest_still_s"] = max(s["longest_still_s"], round(self._still_run, 1))
            else:
                self._still_run = 0.0
            self._visited.add((math.floor(x / 0.25), math.floor(y / 0.25)))
            if popup:
                self._popup_step(v)
            if self.t - self._dumped_t >= 5.0:
                self._dumped_t = self.t
                self.dump()

        def _popup_step(self, v):
            if self.person is not None:
                px, py, pr, born = self.person
                p = self.popups[-1]
                d = math.hypot(px - float(self.pose[0]), py - float(self.pose[1])) - pr - R_BODY
                p["min_gap_m"] = round(min(p["min_gap_m"], d), 3)
                if p["react_s"] is None and v <= 0.05:
                    p["react_s"] = round(self.t - born, 2)
                if self.t - born >= 4.0:
                    self.person = None
                return
            due = len(self.popups) < 3 and self.t >= 20.0 + 25.0 * len(self.popups)
            if due and v > 0.1:
                x, y, th = (float(p) for p in self.pose)
                px, py = x + 0.75 * math.cos(th), y + 0.75 * math.sin(th)
                if _clearance(px, py, walls, circles) > 0.5:
                    self.person = (px, py, 0.18, self.t)
                    self.popups.append(dict(at_s=round(self.t, 1), v_at=round(v, 2),
                                            react_s=None, min_gap_m=9.0))

        def dump(self):
            s = dict(self.score)
            T = max(1e-6, self.t)
            s["sim_s"] = round(self.t, 1)
            s["seen_m2"] = round(len(self._seen) * 0.01, 1)
            s["visited_m2"] = round(len(self._visited) * 0.0625, 2)
            s["driven_m"] = round(s["driven_m"], 2)
            s["turned_deg"] = round(math.degrees(s.pop("turned_rad")), 0)
            s["contact_s"] = round(s["contact_s"], 1)
            s["still_pct"] = round(100 * s["still_s"] / T, 1)
            s["hesitate_pct"] = round(100 * s["hesitate_s"] / T, 1)
            s["reverse_pct"] = round(100 * s["reverse_s"] / T, 1)
            s["popups"] = self.popups
            s["contacts"] = self.contacts[:200]
            tmp = out_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(s, f)
            os.replace(tmp, out_path)

    return ABWorld
