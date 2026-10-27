"""Round 3 of the exploration fine-tune (2026-10-27). Harsh: "train the model
like with gyro also and also train it like to avoid the difficulties the
robot faces while exploration you know it by running 2 min tests before".
He chose the gyro as an INPUT (pi_deploy feeds it only to a model exactly one
input wider, rl/tests/test_gyro_input.py).

Builds on friday_body_env.FridayExplorationEnv (rounds 1-2: live frontier
rays, the dead-reckoned map, the 14 blind beams, the measured drivetrain,
strict reverse) and adds, each from what the robot actually ran into:

1. THE GYRO. Observation 147 = 72 LiDAR + 72 frontier + the turn rate she
   MADE (the gyro: true turn rate + noise, the executed turn on the ~2% of
   ticks a reply is missing) / omega_max + the last action -- the order
   pi_deploy builds it in. The believed heading turns with the gyro, as
   pi_deploy's dead reckoning does (ex_v along the gyro heading), not with the
   commanded turn: a stalled pivot no longer turns her map.
2. THE SAFETY LAYERS THAT HELD HER. The 2026-10-19 runs: governor on 42-52%
   of ticks, reflex on 23-40%. pi_deploy's governor (no forward motion with a
   LiDAR return under 0.36 m in the forward +-45 deg, proportional from
   0.55 m; blind beams see nothing) and the ToF reflex (+-22 deg off the
   bumper, stop under 0.30 m, low things included) run here too, and every
   tick they cut her forward speed costs R_INTERVENE -- so she learns to keep
   out of their way instead of leaning on them.
3. THINGS UNDER THE LIDAR (the 2-min "low_things" room, and the 19 Oct run
   that ended pressed into something at ToF 0.12 m): low obstacles the
   LiDAR never sees; only the reflex stops for them, and touching one is a
   crash.
4. A PERSON STEPPING IN (the "popup" room): a 0.18 m disk 0.75 m ahead
   while she drives, for 4 s, up to 3 times.
5. THE 2-MIN ROOMS, RANDOMISED: an L corridor 1.2-1.6 m wide with door
   recesses, a dining room (table + chair legs, sofa, cupboard), two rooms
   joined by a 0.8-1.0 m door, a room of low things, an open room with a
   person -- never the exact test layouts, so the Pi A/B stays a test on rooms
   she has not seen. Plus round 2's houses and the random rooms.
   World 12 m (the corridor needs it). Each map is mirrored/rotated at random.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from gymnasium import spaces

import friday_body_env as fb
from friday_body_env import T

R_INTERVENE = -0.02           # per tick the governor or reflex cut her forward speed
GOV_STOP, GOV_SLOW = 0.36, 0.55   # pi_deploy: hard_stop / safe_distance (range from centre)
GOV_BEAMS = np.r_[np.arange(0, 10), np.arange(63, 72)]    # forward +-45 deg
TOF_BEAMS = np.r_[np.arange(0, 5), np.arange(68, 72)]     # forward +-22 deg
TOF_OFFSET, TOF_STOP = 0.18, 0.30     # the bumper sensor sits ~0.18 m ahead of centre
GYRO_NOISE, GYRO_DROP = 0.03, 0.02    # rad/s per reading; share of ticks with no reply
OMEGA_MAX = 1.60
GYRO_AT = 144                          # its place in the observation
SCENES = (("corridor", 0.12), ("dining", 0.12), ("two_rooms", 0.12), ("low_things", 0.12),
          ("open", 0.10), ("house", 0.27), ("random", 0.15))
POPUP_SHARE = 0.3                      # of the non-"open" rooms that also get a person
# HIS env class, taken from the class itself: finetune.point_train_at swaps
# T.DiffDriveExplorationEnv for this file's class, and calling the base through
# that name recursed until Python gave up (round 3's first start, 22:38).
_BASE = fb.FridayExplorationEnv.__bases__[0]


def friday3_config(cfg: T.EnvConfig) -> T.EnvConfig:
    return replace(cfg, world_size=12.0, n_obstacles=(10, 22), n_walls=(1, 4))


class Friday3Env(fb.FridayExplorationEnv):
    SCENES = SCENES
    DOOR_W = (0.8, 1.0)                 # two_rooms
    CORRIDOR_W, CORRIDOR_L1 = (1.2, 1.6), (6.0, 9.5)
    MIN_ROOM_M2 = 6.0      # train.py's reset keeps a map only above 25% of the world (36 m2 here)

    def __init__(self, cfg: T.EnvConfig | None = None, render_mode: str | None = None) -> None:
        _BASE.__init__(self, friday3_config(cfg or T.EnvConfig()), render_mode)
        c = self.cfg
        self.blind = np.zeros(c.n_beams, bool)
        self.blind[fb.BLIND[fb.BLIND < c.n_beams]] = True
        low = np.concatenate([self.observation_space.low[:GYRO_AT], [-1.0], self.observation_space.low[GYRO_AT:]])
        high = np.concatenate([self.observation_space.high[:GYRO_AT], [1.0], self.observation_space.high[GYRO_AT:]])
        self.observation_space = spaces.Box(low=low.astype(np.float32), high=high.astype(np.float32),
                                            dtype=np.float32)
        self.occ_low = np.zeros((self.n_cells, self.n_cells), np.uint8)
        self._fresh_body()

    def _fresh_body(self) -> None:
        super()._fresh_body()
        rng = getattr(self, "np_random", None)
        self._gyro_bias = float(rng.normal(0.0, 0.002)) if rng is not None else 0.0
        self._gyro_obs = 0.0
        self._intervened = False
        self.interventions = 0
        self._true_ranges = np.full(72, 5.0, np.float32)
        self._tof = 5.0
        self._person: tuple[list, int] | None = None
        self._popups = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """train.py's reset, with a smaller minimum room: the dining room and the
        room of low things are 16-35 m2, and its 25%-of-the-world rule would have
        thrown nearly all of them away."""
        T.gym.Env.reset(self, seed=seed)
        c = self.cfg
        if c.domain_randomization:
            self._gain = self.np_random.uniform(0.90, 1.10, size=2).astype(np.float32)
            self._noise_std = float(c.lidar_noise_std * self.np_random.uniform(0.5, 1.8))
            self._dropout_p = float(c.lidar_dropout_p * self.np_random.uniform(0.0, 2.0))
        for _ in range(20):
            self._generate_map()
            self.pose = self._sample_free_pose()
            self._reachable_free = self._flood_fill_reachable(
                self._world_to_cell(float(self.pose[0]), float(self.pose[1])))
            self._reachable_count = max(int(self._reachable_free.sum()), 1)
            if self._reachable_count * self.cell_area > self.MIN_ROOM_M2:
                break
        self.known = np.zeros_like(self.occ)
        self._known_free_count = 0
        self._last_action[:] = 0.0
        self._steps = 0
        self._trajectory = [(float(self.pose[0]), float(self.pose[1]))]
        ranges, frontier = self._raycast()
        return self._observe(ranges, frontier), {"coverage": self.coverage}

    # ------------------------------------------------------------------ rooms ---
    def _generate_map(self) -> None:
        rng = self.np_random
        names, weights = zip(*self.SCENES)
        self.scene = str(rng.choice(names, p=np.array(weights) / sum(weights)))
        n = self.n_cells
        self.occ_low = np.zeros((n, n), np.uint8)
        self.house = self.scene == "house"
        if self.scene == "house":
            self._generate_house()
        elif self.scene == "random":
            fb.FridayExplorationEnv._generate_map(self)          # may itself pick a house
            self.house = False
        else:
            getattr(self, "_room_" + self.scene)()
        self.popup = self.scene == "open" or bool(rng.random() < POPUP_SHARE)
        k = int(rng.integers(0, 4))                               # rotate / mirror
        self.occ, self.occ_low = np.rot90(self.occ, k).copy(), np.rot90(self.occ_low, k).copy()
        if rng.random() < 0.5:
            self.occ, self.occ_low = self.occ[:, ::-1].copy(), self.occ_low[:, ::-1].copy()

    def _cells(self, m: float) -> int:
        return int(round(m / self.cfg.resolution))

    def _blank(self) -> None:
        self.occ = np.ones((self.n_cells, self.n_cells), np.uint8)   # solid; rooms are carved

    def _carve(self, x0: float, y0: float, x1: float, y1: float, grid=None) -> None:
        g = self.occ if grid is None else grid
        n = self.n_cells
        r0, r1 = sorted((self._cells(y0), self._cells(y1)))
        c0, c1 = sorted((self._cells(x0), self._cells(x1)))
        g[max(2, r0):min(n - 2, r1), max(2, c0):min(n - 2, c1)] = 0

    def _box(self, x0: float, y0: float, x1: float, y1: float, low: bool = False) -> None:
        g = self.occ_low if low else self.occ
        r0, r1 = sorted((self._cells(y0), self._cells(y1)))
        c0, c1 = sorted((self._cells(x0), self._cells(x1)))
        g[r0:max(r0 + 1, r1), c0:max(c0 + 1, c1)] = 1

    def _disk(self, x: float, y: float, r: float, low: bool = False) -> None:
        g = self.occ_low if low else self.occ
        n = self.n_cells
        rr, cc = np.ogrid[:n, :n]
        res = self.cfg.resolution
        g[((cc + 0.5) * res - x) ** 2 + ((rr + 0.5) * res - y) ** 2 <= max(r, res * 0.6) ** 2] = 1

    def _room_corridor(self) -> None:
        rng = self.np_random
        self._blank()
        w = float(rng.uniform(*self.CORRIDOR_W))
        x0, y0 = 1.0, float(rng.uniform(1.5, 4.0))
        l1, l2 = float(rng.uniform(*self.CORRIDOR_L1)), float(rng.uniform(3.0, 5.5))
        self._carve(x0, y0, x0 + l1, y0 + w)                       # the long leg
        self._carve(x0 + l1 - w, y0, x0 + l1, y0 + w + l2)         # the turn
        for _ in range(int(rng.integers(0, 3))):                    # door recesses
            dx = float(rng.uniform(x0 + 1.0, x0 + l1 - 2.0))
            side = y0 - 0.4 if rng.random() < 0.5 else y0 + w
            self._carve(dx, side, dx + 1.0, side + 0.4)

    def _room_dining(self) -> None:
        rng = self.np_random
        self._blank()
        W, H = float(rng.uniform(4.5, 6.0)), float(rng.uniform(3.5, 4.6))
        x0, y0 = float(rng.uniform(1.0, 11.0 - W)), float(rng.uniform(1.0, 11.0 - H))
        self._carve(x0, y0, x0 + W, y0 + H)
        tx, ty = x0 + float(rng.uniform(1.6, W - 1.6)), y0 + float(rng.uniform(1.4, H - 1.4))
        for sx, sy in ((-0.6, -0.4), (0.6, -0.4), (-0.6, 0.4), (0.6, 0.4)):     # table legs
            self._disk(tx + sx, ty + sy, 0.03)
        seats = [(tx - 0.3, ty - 0.75), (tx + 0.3, ty - 0.75), (tx - 0.3, ty + 0.75), (tx + 0.3, ty + 0.75)]
        seats += [(float(rng.uniform(x0 + 0.5, x0 + W - 0.5)), float(rng.uniform(y0 + 0.5, y0 + H - 0.5)))
                  for _ in range(int(rng.integers(0, 3)))]
        for cx, cy in seats:                                         # chairs: four legs each
            for sx in (-0.21, 0.21):
                for sy in (-0.21, 0.21):
                    self._disk(cx + sx, cy + sy, 0.02)
        self._box(x0, y0 + H - 0.7, x0 + 2.2, y0 + H)                # sofa on a wall
        self._box(x0 + W - 0.6, y0, x0 + W, y0 + 1.4)                # cupboard

    def _room_two_rooms(self) -> None:
        rng = self.np_random
        self._blank()
        W, H = float(rng.uniform(7.0, 9.0)), float(rng.uniform(3.5, 4.5))
        x0, y0 = float(rng.uniform(1.0, 11.0 - W)), float(rng.uniform(1.0, 11.0 - H))
        self._carve(x0, y0, x0 + W, y0 + H)
        wx = x0 + W * float(rng.uniform(0.4, 0.6))
        self._box(wx, y0, wx + 0.1, y0 + H)                          # the dividing wall
        d = float(rng.uniform(*self.DOOR_W))
        dy = y0 + float(rng.uniform(0.3, H - d - 0.3))
        self._carve(wx - 0.05, dy, wx + 0.15, dy + d)                # its door
        for _ in range(int(rng.integers(3, 6))):
            self._disk(float(rng.uniform(x0 + 0.6, x0 + W - 0.6)), float(rng.uniform(y0 + 0.6, y0 + H - 0.6)),
                       float(rng.uniform(0.25, 0.4)))
        self._carve(wx - 0.5, dy, wx + 0.6, dy + d)                  # keep the doorway clear

    def _room_low_things(self) -> None:
        rng = self.np_random
        self._blank()
        W, H = float(rng.uniform(5.0, 7.0)), float(rng.uniform(4.0, 5.5))
        x0, y0 = float(rng.uniform(1.0, 11.0 - W)), float(rng.uniform(1.0, 11.0 - H))
        self._carve(x0, y0, x0 + W, y0 + H)
        for _ in range(int(rng.integers(3, 7))):
            self._disk(float(rng.uniform(x0 + 0.5, x0 + W - 0.5)), float(rng.uniform(y0 + 0.5, y0 + H - 0.5)),
                       float(rng.uniform(0.08, 0.13)), low=True)
        bx, by = float(rng.uniform(x0, x0 + W - 1.7)), float(rng.uniform(y0, y0 + H - 0.9))
        self._box(bx, by, bx + 1.7, by + 0.9, low=True)               # a low sofa base / bed frame
        for _ in range(int(rng.integers(1, 3))):
            self._disk(float(rng.uniform(x0 + 0.6, x0 + W - 0.6)), float(rng.uniform(y0 + 0.6, y0 + H - 0.6)), 0.3)

    def _room_open(self) -> None:
        rng = self.np_random
        self._blank()
        W, H = float(rng.uniform(6.5, 9.0)), float(rng.uniform(6.5, 9.0))
        x0, y0 = float(rng.uniform(1.0, 11.0 - W)), float(rng.uniform(1.0, 11.0 - H))
        self._carve(x0, y0, x0 + W, y0 + H)
        for _ in range(2):
            self._disk(float(rng.uniform(x0 + 1, x0 + W - 1)), float(rng.uniform(y0 + 1, y0 + H - 1)), 0.3)

    # ------------------------------------------------------------ collisions ---
    def _stencil_hits(self, x: float, y: float, stencil: np.ndarray) -> bool:
        if super()._stencil_hits(x, y, stencil):
            return True
        r0, c0 = self._world_to_cell(x, y)
        rr, cc = stencil[:, 0] + r0, stencil[:, 1] + c0
        ok = (rr >= 0) & (rr < self.n_cells) & (cc >= 0) & (cc < self.n_cells)
        return bool(self.occ_low[rr[ok], cc[ok]].any())

    # --------------------------------------------------------------- sensing ---
    def _raycast(self):
        ranges, frontier = super()._raycast()
        self._true_ranges = ranges
        # the bumper ToF also sees what is under the LiDAR
        rr, cc = self._beam_cells(self.pose)
        rr, cc = rr[TOF_BEAMS], cc[TOF_BEAMS]
        low = self.occ_low[rr, cc] == 1
        low_r = np.where(low.any(axis=1), self._ray_depths[low.argmax(axis=1)], self.cfg.lidar_max_range)
        self._tof = float(min(ranges[TOF_BEAMS].min(), low_r.min())) - TOF_OFFSET
        return ranges, frontier

    def _observe(self, ranges, frontier):
        obs = super()._observe(ranges, frontier)
        return np.concatenate([obs[:GYRO_AT], [self._gyro_obs], obs[GYRO_AT:]]).astype(np.float32)

    # -------------------------------------------------------------- dynamics ---
    def _integrate(self, v: float, omega: float) -> bool:
        """The governor and the reflex first (as on the robot, they only ever cut
        forward speed), then rounds 1-2's drive chain and wheels -- with the
        believed heading from the gyro."""
        asked = v
        seen = GOV_BEAMS[~self.blind[GOV_BEAMS]]
        d = float(self._true_ranges[seen].min()) if seen.size else self.cfg.lidar_max_range
        if v > 0.0:
            if d < GOV_STOP:
                v = 0.0
            elif d < GOV_SLOW:
                v *= (d - GOV_STOP) / (GOV_SLOW - GOV_STOP)
            if self._tof < TOF_STOP:
                v = 0.0
        self._intervened = asked > 0.02 and v < asked - 0.02
        self.interventions += self._intervened
        b_th0 = float(self.belief[2])
        th0 = float(self.pose[2])
        hit = super()._integrate(v, omega)
        # the gyro: what she really turned, measured
        made = math.atan2(math.sin(float(self.pose[2]) - th0), math.cos(float(self.pose[2]) - th0)) / self.cfg.dt
        believed = math.atan2(math.sin(float(self.belief[2]) - b_th0), math.cos(float(self.belief[2]) - b_th0)) / self.cfg.dt
        if self.np_random.random() < GYRO_DROP:
            rate = believed                                          # no reply: the executed turn
        else:
            rate = made + self._gyro_bias + float(self.np_random.normal(0.0, GYRO_NOISE))
            th = b_th0 + rate * self.cfg.dt                          # dead reckoning turns with it
            self.belief[2] = math.atan2(math.sin(th), math.cos(th))
        self._gyro_obs = float(np.clip(rate / OMEGA_MAX, -1.0, 1.0))
        return hit

    def step(self, action):
        self._popup_tick()
        obs, reward, terminated, truncated, info = super().step(action)
        if self._intervened:
            reward += R_INTERVENE
        return obs, reward, terminated, truncated, info

    def _popup_tick(self) -> None:
        """A person steps in 0.75 m ahead while she drives, for 4 s, up to 3 times."""
        if not getattr(self, "popup", False):
            return
        if self._person is not None:
            cells, until = self._person
            if self._steps >= until:
                for r, c in cells:
                    self.occ[r, c] = 0
                self._person = None
            return
        due = self._popups < 3 and self._steps >= 200 + 250 * self._popups
        if not due or abs(self._vw[0]) < 0.05:
            return
        x, y, th = (float(v) for v in self.pose)
        px, py = x + 0.75 * math.cos(th), y + 0.75 * math.sin(th)
        n, res = self.n_cells, self.cfg.resolution
        rr, cc = np.ogrid[:n, :n]
        disk = ((cc + 0.5) * res - px) ** 2 + ((rr + 0.5) * res - py) ** 2 <= 0.18 ** 2
        if self.occ[disk].any():
            return
        cells = list(zip(*np.nonzero(disk)))
        for r, c in cells:
            self.occ[r, c] = 1
        self._person = (cells, self._steps + 40)
        self._popups += 1


class Friday4Env(Friday3Env):
    """Round 4 (2026-10-27): round 3 lost the Pi A/B on two_rooms' 0.8 m door
    (18.4 m2 vs ACT's 31.4: centred, the jambs are 0.40 m from her centre and
    the governor stops at 0.36) and on the corridor (13.8 vs 22.9). Same body,
    more and narrower doorways, longer and narrower corridors."""
    SCENES = (("corridor", 0.20), ("dining", 0.08), ("two_rooms", 0.22), ("low_things", 0.10),
              ("open", 0.08), ("house", 0.22), ("random", 0.10))
    DOOR_W = (0.78, 0.95)
    HOUSE_DOOR_W = (0.78, 0.95)
    CORRIDOR_W, CORRIDOR_L1 = (1.0, 1.5), (7.5, 10.5)


def make_env_fn(cfg: T.EnvConfig, seed: int, rank: int):
    """train.make_env_fn with round 3's env (top level, so SubprocVecEnv can pickle it)."""
    def _init():
        env = Friday3Env(cfg)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init


def make_env_fn4(cfg: T.EnvConfig, seed: int, rank: int):
    """The same for round 4."""
    def _init():
        env = Friday4Env(cfg)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        return env
    return _init
