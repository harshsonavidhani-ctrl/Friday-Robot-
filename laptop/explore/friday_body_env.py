"""Friday's REAL body inside Harsh's PPO exploration env (his train.py), for the
2026-10-25 fine-tune. Harsh: "train my pre-trained model in my laptop"; asked
how, he chose: fine-tune his model in a simulator matched to the real Friday,
and replace today's model only if the new one wins in the simulator first.

`FridayExplorationEnv` subclasses `train.DiffDriveExplorationEnv`. The
observation (146) and action (2) are unchanged, so a fine-tuned policy drops
into pi_deploy exactly where exploration_model.onnx goes. What changes, each
taken from the robot rather than invented:

1. THE FRONTIER RAYS ARE LIVE. pi_deploy's update_and_frontier (fixed
   2026-08-28) computes them against the map as it stood BEFORE this scan;
   train.py computes them after the scan has already marked every cell before
   the hit as known, so in training all 72 were the constant 5 m on every step.
   The policy never learned to use 72 of its 146 inputs.
2. THE MAP COMES FROM WHERE SHE THINKS SHE IS. pi_deploy integrates each scan
   at the pose dead-reckoned from executed() -- the duty-linear twist, not the
   motion that happened. The frontier rays come from that belief map; the
   reward still counts only what she TRULY saw.
3. 14 BLIND BEAMS (the chassis self-mask 71:79,88:100,221:237,319:334 at the
   -51.8 deg mounting offset, masked_beam_indices on the Pi): they read 5 m,
   never mark either map and are never a frontier -- as on the robot.
4. THE DRIVETRAIN. pi_deploy's live chain after the policy: forward capped at
   0.125 m/s (MODE_MAX_FORWARD_SPEED), any reverse lifted to 0.20 m/s
   (MODE_REVERSE_FLOOR), to_wheel_speeds at wheel_speed_max 0.403, duty =
   linear to a 0.85 ceiling with the stiction floor 80 (pivots 112) and the
   creep cutoff 40, a 2-tick x1.6 kick from rest; then the wheels of
   scripts/sim_ab/body.py (measured: break-away and keep-going duties per
   motion, a pivot break-away that varies 110-130 per attempt, the right side
   ~7% weaker, 0.12 s motor lag).
5. ROOMS LIKE HIS: an 8 m square, 1-3 walls with 0.8-1.1 m doorways, more and
   smaller furniture, and 10 cm posts (chair and table legs) that 5-degree
   beams can miss at a distance.

ROUND 2 (2026-10-27, after the ft1 A/B on the Pi: ft1 reversed 17-26% of the
time where ACT reverses ~0%, and in "two_rooms" it seldom went through the
door -- 18.4 m2 seen against ACT's 31.4):
6. STRICT REVERSE, as the robot does it (MODE_STRICT_REVERSE): backing up only
   into space the LiDAR SEES clear behind her (a blind beam is not clear), at
   most 1.2 s at a time; otherwise the reverse executes as a stop. Plus a
   small cost per reversing step (R_REVERSE).
7. HOUSES: HOUSE_SHARE of the rooms are 2-6 rooms in a grid joined by
   0.8-1.1 m doorways, with furniture kept out of every door.

Not modelled (they only ever REDUCE authority on the robot): the governor,
navigator, ToF reflex and rate limiter. The A/B on the Pi
(scripts/sim_ab, BODY=1) runs the policy through ALL of them and is the judge.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home()))          # Harsh's train.py lives in his home folder
import train as T                             # noqa: E402

BLIND = np.array([3, 4, 5, 7, 8, 9, 33, 34, 35, 36, 53, 54, 55, 56])

# pi_deploy, live (modes/confident_explore.env + RobotConfig + FridayMotorDriver)
MAX_FORWARD = 0.125
REVERSE_FLOOR = 0.20
WHEEL_SPEED_MAX = 0.403
WHEEL_BASE = 0.35
CEILING = int(255 * 0.85)
MIN_DUTY, TURN_MIN_DUTY, CUTOFF = 80, 112, 40
KICK_TICKS, KICK_SCALE = 2, 1.6

# round 2
R_REVERSE = -0.03                 # per step that backs up
REVERSE_MAX_STEPS = 12            # strict reverse: at most 1.2 s at a time
REAR_CLEAR_M = 0.50               # ... and only with this much SEEN clear behind
REAR_BEAMS = np.arange(27, 46)    # 135..225 deg
HOUSE_SHARE = 0.6

# scripts/sim_ab/body.py (measured)
SIDE_VMAX, B_EFF, TAU = 0.55, 0.42, 0.12


def friday_config(cfg: T.EnvConfig) -> T.EnvConfig:
    return replace(cfg, world_size=8.0, n_obstacles=(8, 18), n_walls=(1, 3))


def mps_to_duty(mps: float, floor: int) -> int:
    """FridayMotorDriver._mps_to_duty with the live floors."""
    duty = int(round(CEILING * (mps / WHEEL_SPEED_MAX)))
    if duty != 0 and abs(duty) < floor:
        if abs(duty) < CUTOFF:
            return 0
        duty = int(math.copysign(min(floor, CEILING), duty))
    return max(-CEILING, min(CEILING, duty))


def side_speed(d: float, moving: bool, mode: str, pivot_start: float, gain: float) -> float:
    """body._side: one side's ground speed (m/s) at duty d."""
    a = abs(d)
    if mode == "pivot":
        start, keep = pivot_start, 105
    elif mode == "reverse":
        start, keep = 90, 60
    else:
        start, keep = 65, 50
    if a < (keep if moving else start):
        return 0.0
    if mode == "pivot":
        s = (0.26 * (a - 105) / (153 - 105) if a <= 153
             else 0.26 + 0.26 * (1.68 / 1.24 - 1.0) * (a - 153) / (189 - 153))
    else:
        s = SIDE_VMAX * (a - 50) / (255 - 50)
    return math.copysign(s * gain, d)


class FridayExplorationEnv(T.DiffDriveExplorationEnv):
    HOUSE_DOOR_W = (0.8, 1.1)          # m
    def __init__(self, cfg: T.EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__(friday_config(cfg or T.EnvConfig()), render_mode)
        self.blind = np.zeros(self.cfg.n_beams, bool)
        self.blind[BLIND[BLIND < self.cfg.n_beams]] = True
        self._fresh_body()

    def _fresh_body(self) -> None:
        self.belief = self.pose.copy()
        self.belief_known = np.zeros_like(self.occ)
        self._duty_prev = [0, 0]
        self._kick = [0, 0]
        self._moving = [False, False]
        self._pivot_start: float | None = None
        self._vw = [0.0, 0.0]
        self._rev_steps = 0
        self._reversed = False
        self.reverse_blocked = 0
        rng = getattr(self, "np_random", None)
        weak = float(rng.uniform(0.88, 0.98)) if rng is not None else 0.93
        right_weaker = bool(rng.random() < 0.8) if rng is not None else True
        self._side_gain = (1.0, weak) if right_weaker else (weak, 1.0)

    # --------------------------------------------------------------- rooms ----
    def _generate_map(self) -> None:
        self.house = bool(self.np_random.random() < HOUSE_SHARE)
        if self.house:
            return self._generate_house()
        c, n = self.cfg, self.n_cells
        rng = self.np_random
        self.occ = np.zeros((n, n), np.uint8)
        self.occ[:2, :] = self.occ[-2:, :] = self.occ[:, :2] = self.occ[:, -2:] = self.OCCUPIED
        for _ in range(rng.integers(c.n_walls[0], c.n_walls[1] + 1)):
            horizontal = bool(rng.integers(0, 2))
            pos = int(rng.integers(int(0.2 * n), int(0.8 * n)))
            thick = int(rng.integers(1, 3))
            door = int(rng.integers(int(0.15 * n), int(0.85 * n)))
            door_w = int(round(float(rng.uniform(0.8, 1.1)) / c.resolution))
            lo, hi = max(door - door_w // 2, 0), min(door + (door_w + 1) // 2, n)
            if horizontal:
                self.occ[pos:pos + thick, :] = self.OCCUPIED
                self.occ[pos:pos + thick, lo:hi] = self.FREE
            else:
                self.occ[:, pos:pos + thick] = self.OCCUPIED
                self.occ[lo:hi, pos:pos + thick] = self.FREE
        for _ in range(rng.integers(c.n_obstacles[0], c.n_obstacles[1] + 1)):
            h = int(rng.integers(int(0.2 / c.resolution), int(1.2 / c.resolution) + 1))
            w = int(rng.integers(int(0.2 / c.resolution), int(1.2 / c.resolution) + 1))
            r0 = int(rng.integers(3, max(n - h - 3, 4)))
            c0 = int(rng.integers(3, max(n - w - 3, 4)))
            self.occ[r0:r0 + h, c0:c0 + w] = self.OCCUPIED
        for _ in range(rng.integers(0, 7)):                    # chair / table legs
            r0, c0 = int(rng.integers(3, n - 3)), int(rng.integers(3, n - 3))
            self.occ[r0, c0] = self.OCCUPIED

    def _generate_house(self) -> None:
        """2-3 x 1-2 rooms, a 0.8-1.1 m door in every wall between two rooms,
        furniture and chair legs, and every door kept clear 0.4 m each side."""
        c, n, rng = self.cfg, self.n_cells, self.np_random
        res = c.resolution
        self.occ = np.zeros((n, n), np.uint8)
        self.occ[:2, :] = self.occ[-2:, :] = self.occ[:, :2] = self.occ[:, -2:] = self.OCCUPIED

        def cuts(k: int) -> list[int]:
            return [int(b + rng.integers(-5, 6)) for b in np.linspace(2, n - 2, k + 1)[1:-1]]
        xc, yc = cuts(int(rng.integers(2, 4))), cuts(int(rng.integers(1, 3)))
        xs, ys = [2] + xc + [n - 2], [2] + yc + [n - 2]
        doors = []
        for x in xc:                                           # walls along y, a door per stretch
            self.occ[:, x:x + 2] = self.OCCUPIED
            for y0, y1 in zip(ys[:-1], ys[1:]):
                w = int(round(rng.uniform(*self.HOUSE_DOOR_W) / res))
                d = int(rng.integers(y0 + 3, max(y0 + 4, y1 - w - 2)))
                doors.append((d, d + w, x - 4, x + 6))
        for y in yc:                                           # walls along x
            self.occ[y:y + 2, :] = self.OCCUPIED
            for x0, x1 in zip(xs[:-1], xs[1:]):
                w = int(round(rng.uniform(*self.HOUSE_DOOR_W) / res))
                d = int(rng.integers(x0 + 3, max(x0 + 4, x1 - w - 2)))
                doors.append((y - 4, y + 6, d, d + w))
        for _ in range(rng.integers(4, 11)):
            h = int(rng.integers(int(0.2 / res), int(0.8 / res) + 1))
            w = int(rng.integers(int(0.2 / res), int(0.8 / res) + 1))
            r0, c0 = int(rng.integers(3, n - h - 3)), int(rng.integers(3, n - w - 3))
            self.occ[r0:r0 + h, c0:c0 + w] = self.OCCUPIED
        for _ in range(rng.integers(0, 7)):
            self.occ[int(rng.integers(3, n - 3)), int(rng.integers(3, n - 3))] = self.OCCUPIED
        for r0, r1, c0, c1 in doors:                           # the door and 0.4 m each side
            self.occ[max(2, r0):min(n - 2, r1), max(2, c0):min(n - 2, c1)] = self.FREE

    def _sample_free_pose(self) -> np.ndarray:
        pose = super()._sample_free_pose()
        self.pose = pose
        self._fresh_body()                                     # reset() calls this per try
        return pose

    # ------------------------------------------------------------- sensing ----
    def _beam_cells(self, pose) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        x, y, th = (float(v) for v in pose)
        ang = self._beam_angles + th
        px = x + np.cos(ang)[:, None] * self._ray_depths[None, :]
        py = y + np.sin(ang)[:, None] * self._ray_depths[None, :]
        cc = np.clip((px / c.resolution).astype(np.int32), 0, self.n_cells - 1)
        rr = np.clip((py / c.resolution).astype(np.int32), 0, self.n_cells - 1)
        return rr, cc

    def _raycast(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg
        rr, cc = self._beam_cells(self.pose)
        occ_hits = self.occ[rr, cc] == self.OCCUPIED
        hit_any = occ_hits.any(axis=1)
        hit_idx = np.where(hit_any, occ_hits.argmax(axis=1), self._n_ray_samples - 1)
        ranges = np.where(hit_any, self._ray_depths[hit_idx], c.lidar_max_range).astype(np.float32)
        # what she TRULY saw (the reward): blind beams see nothing
        see = ~self.blind
        sample_idx = np.arange(self._n_ray_samples)[None, :]
        free_mask = (sample_idx < hit_idx[:, None]) & see[:, None]
        flat_free = np.unique(rr[free_mask] * self.n_cells + cc[free_mask])
        if flat_free.size:
            known_flat = self.known.reshape(-1)
            reach_flat = self._reachable_free.reshape(-1)
            newly = (known_flat[flat_free] == 0) & reach_flat[flat_free]
            self._known_free_count += int(newly.sum())
            known_flat[flat_free] = 1
        wall = hit_any & see
        self.known[rr[wall, hit_idx[wall]], cc[wall, hit_idx[wall]]] = 1
        return ranges, self._belief_update(ranges)

    def _belief_update(self, scan: np.ndarray) -> np.ndarray:
        """pi_deploy LocalExplorationMap.update_and_frontier, at the BELIEVED pose:
        the frontier first, against the map before this scan; then integrate."""
        c = self.cfg
        rr, cc = self._beam_cells(self.belief)
        n_s = self._n_ray_samples
        hit_idx = np.clip((scan / self._ray_step).astype(np.int32), 0, n_s - 1)
        hit_any = (scan < c.lidar_max_range - 1e-3) & ~self.blind
        frontier = np.full(c.n_beams, c.lidar_max_range, np.float32)
        if c.use_frontier_rays:
            unknown = self.belief_known[rr, cc] == 0
            unk_any = unknown.any(axis=1)
            unk_idx = np.where(unk_any, unknown.argmax(axis=1), n_s - 1)
            visible = unk_any & (unk_idx <= hit_idx) & ~self.blind
            frontier = np.where(visible, self._ray_depths[unk_idx], c.lidar_max_range).astype(np.float32)
        free_mask = (np.arange(n_s)[None, :] < hit_idx[:, None]) & ~self.blind[:, None]
        flat_free = np.unique(rr[free_mask] * self.n_cells + cc[free_mask])
        if flat_free.size:
            self.belief_known.reshape(-1)[flat_free] = 1
        if hit_any.any():
            self.belief_known[rr[hit_any, hit_idx[hit_any]], cc[hit_any, hit_idx[hit_any]]] = 1
        return frontier

    def _observe(self, ranges: np.ndarray, frontier: np.ndarray) -> np.ndarray:
        obs = super()._observe(ranges, frontier)
        obs[:self.cfg.n_beams][self.blind] = 1.0                # a blind beam reads 5 m
        return obs

    # ------------------------------------------------------------ dynamics ----
    def _integrate(self, v: float, omega: float) -> bool:
        """The command through pi_deploy's live chain and the measured wheels.
        Moves the TRUE pose (collisions) and the BELIEVED one (the map)."""
        c = self.cfg
        v = min(v, MAX_FORWARD)
        if v < 0.0 and abs(v) < REVERSE_FLOOR:
            v = -REVERSE_FLOOR
        if v < 0.0:
            self._rev_steps += 1
            if self._rev_steps > REVERSE_MAX_STEPS or not self._rear_seen_clear():
                v = 0.0                                        # strict reverse: refused
                self.reverse_blocked += 1
        else:
            self._rev_steps = 0
        self._reversed = v < 0.0
        vr, vl = v + omega * WHEEL_BASE / 2.0, v - omega * WHEEL_BASE / 2.0
        peak = max(abs(vr), abs(vl))
        if peak > WHEEL_SPEED_MAX:
            s = WHEEL_SPEED_MAX / peak
            vr, vl = vr * s, vl * s
        pivot = (vl > 0 > vr) or (vr > 0 > vl)
        floor = TURN_MIN_DUTY if pivot else MIN_DUTY
        duties = [mps_to_duty(vl, floor), mps_to_duty(vr, floor)]
        sent = list(duties)
        for i in (0, 1):
            if duties[i] != 0 and (self._duty_prev[i] == 0 or (duties[i] > 0) != (self._duty_prev[i] > 0)):
                self._kick[i] = KICK_TICKS
            if duties[i] == 0:
                self._kick[i] = 0
            if self._kick[i] > 0:
                sent[i] = int(max(-CEILING, min(CEILING, round(duties[i] * KICK_SCALE))))
                self._kick[i] -= 1
        self._duty_prev = duties
        # what pi_deploy believes it did (executed(): duty-linear, the unkicked duties)
        scale = WHEEL_SPEED_MAX / CEILING
        b_v = (duties[0] + duties[1]) * scale / 2.0
        b_w = (duties[1] - duties[0]) * scale / WHEEL_BASE
        # the wheels
        l, r = sent
        is_pivot = (l > 0 > r) or (r > 0 > l)
        if is_pivot and self._pivot_start is None:
            self._pivot_start = float(self.np_random.uniform(110, 130))
        if not is_pivot:
            self._pivot_start = None
        mode = "pivot" if is_pivot else ("reverse" if l < 0 and r < 0 else "drive")
        h = c.dt / c.n_substeps
        x, y, th = float(self.pose[0]), float(self.pose[1]), float(self.pose[2])
        bx, by, bth = float(self.belief[0]), float(self.belief[1]), float(self.belief[2])
        for _ in range(c.n_substeps):
            sl = side_speed(l, self._moving[0], mode, self._pivot_start or 120.0, self._side_gain[0])
            sr = side_speed(r, self._moving[1], mode, self._pivot_start or 120.0, self._side_gain[1])
            self._moving = [sl != 0.0, sr != 0.0]
            k = min(1.0, h / TAU)
            self._vw[0] += ((sl + sr) / 2.0 - self._vw[0]) * k
            self._vw[1] += ((sr - sl) / B_EFF - self._vw[1]) * k
            tv, tw = self._vw
            th_mid = th + tw * h / 2.0
            x += tv * math.cos(th_mid) * h
            y += tv * math.sin(th_mid) * h
            th = math.atan2(math.sin(th + tw * h), math.cos(th + tw * h))
            b_mid = bth + b_w * h / 2.0
            bx += b_v * math.cos(b_mid) * h
            by += b_v * math.sin(b_mid) * h
            bth = math.atan2(math.sin(bth + b_w * h), math.cos(bth + b_w * h))
            self.pose[:] = (x, y, th)
            self.belief[:] = (bx, by, bth)
            if self._stencil_hits(x, y, self._collision_stencil):
                return True
        return False


    def _rear_seen_clear(self) -> bool:
        """The robot's strict reverse: the rear beams it can SEE all show more
        than REAR_CLEAR_M; a blind beam is unknown, not clear."""
        seen = REAR_BEAMS[~self.blind[REAR_BEAMS]]
        return bool(seen.size) and float(self._last_scan[seen].min()) > REAR_CLEAR_M

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self._reversed:
            reward += R_REVERSE
        return obs, reward, terminated, truncated, info


def make_env_fn(cfg: T.EnvConfig, seed: int, rank: int):
    """train.make_env_fn with Friday's body (top level, so SubprocVecEnv can pickle it)."""
    def _init():
        env = FridayExplorationEnv(cfg)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init
