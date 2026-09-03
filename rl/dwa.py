"""Pick a drivable arc, instead of proposing one and having it vetoed.

WHY THIS EXISTS
===============

Harsh, 2026-09-04, after four autonomous runs: *"i want like the real robots
navigate with confidence"*, and earlier the same night, *"it hessitates a lot"* and
*"traversing in a zig-zag way in an open straight hallway"*.

The chain this module joins is `policy -> planner bias -> governor -> visual gate ->
ToF reflex -> rate limiter -> motors`, and every stage after the policy can only
REDUCE authority. That is the right safety design and it is not being changed here.
But it means one network, which sees 72 beams and no further ahead than now, is the
only thing that ever DECIDES anything, and everything downstream can only say no.

The failure mode that falls out of that is not stopping. It is SPINNING. The
governor's clamp is `v_out = min(v_out, 0.0)` -- forward is zeroed and `omega` passes
through untouched -- so a robot that is refused permission to drive still has full
permission to turn, and turns. MEASURED over the four runs of 2026-09-04:

    gate had the last word:  SAFETY_OVERRIDE 68.7% of ticks (best run; 88.9% worst)
    rule replaced network:   66.5% of ticks
    turning per metre:       515 deg/m best, 1799 deg/m worst

515 deg/m means she turned about one and a half full revolutions for every metre
travelled, in a room she could see 1.6 m down. A corridor should be near zero.

WHAT REAL ROBOTS DO INSTEAD
===========================

A local planner (DWA, DWB, TEB -- the family behind every ROS nav stack) does not
propose-then-veto. It SAMPLES candidate commands, forward-simulates each one against
the obstacle field, throws away the ones that collide, and drives the best survivor.
The command that reaches the wheels is already collision-free, so the emergency layer
becomes the exception rather than the steering mechanism.

That is the whole idea here, and it is why this improves "confidence": the robot stops
discovering obstacles at the moment it is already too close to do anything but brake.

DELIBERATE DEPARTURES FROM TEXTBOOK DWA, EACH FOR A MEASURED REASON
==================================================================

1. THE WINDOW IS NOT THE ONE-TICK REACHABLE SET. Textbook DWA samples only what the
   accelerations can reach within dt. Here dt = 0.1 s and max_lin_accel = 0.8, so that
   window is +-0.08 m/s -- from a standstill DWA could only ever propose a crawl, and
   would never evaluate the arc it is actually committing to. This samples the full
   permitted velocity range as CANDIDATE STEADY STATES and leaves the approach rate to
   `RateLimiter`, which is a separate, already-tested stage. Double-applying slew here
   would fight it.

2. ONLY FORWARD ARCS ARE SAMPLED (`v > 0`). If nothing forward survives, `choose()`
   returns None and the caller keeps today's behaviour exactly -- escape turn, reverse
   lift, recovery controller, all untouched and all still covered by the 44 safety
   self-test assertions. Reverse is an ESCAPE, owned by machinery that already exists
   and already has a rear gate; it is not a navigation choice this module should make.
   The useful side effect: DWA can never CHOOSE to pivot in place, which is the exact
   behaviour it exists to remove.

3. IT NEVER RAISES A CAP. `v_allow` comes in from the caller already reduced by
   whatever the policy and the earlier stages permit, and no candidate exceeds it. This
   module can only ever return something as slow as, or slower than, what was already
   allowed -- so the chain's "every stage may only reduce authority" invariant holds
   with this stage inserted.

WHAT IT DOES NOT DO
===================

It does not touch the 146-dim observation. The policy still chooses WHERE to go -- its
intent is what `desired_bearing` scores against when there is no exploration target --
and this module only decides whether that intent is DRIVABLE, and picks the nearest
drivable thing to it. A goal is never smuggled into the network's input.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

# Sampling resolution. 5 x 21 = 105 candidates, each simulated over SIM_STEPS points
# against up to n_beams obstacles -- ~91k squared distances, which numpy does in about
# a millisecond. Measured on this Pi in test_dwa.py, which FAILS if it regresses:
# the control budget is 100 ms and the loop already runs at 62 ms mean.
V_SAMPLES = 5
OMEGA_SAMPLES = 21          # odd, so straight ahead (omega = 0) is always a candidate

# How far ahead to simulate. Long enough to see a wall before committing, short enough
# that the arc is still a fair description of what she will do (the policy re-decides
# every tick anyway).
SIM_TIME_S = 1.2
SIM_STEPS = 12

# THE HORIZON IS A DISTANCE, NOT JUST A TIME -- and this is not a refinement, it is a
# bug fix that test_dwa caught. With a fixed 1.2 s horizon a candidate at the 0.15 m/s
# stiction floor looks only 0.18 m ahead, so "creep straight at a wall 0.5 m away"
# simulates as perfectly safe and WINS on heading score. Slow commands were being
# handed a safety advantage they had not earned, and the resulting behaviour -- inching
# toward an obstacle instead of deciding to go round it -- is precisely the hesitation
# this module exists to remove.
#
# So every candidate is simulated until it has covered LOOKAHEAD_M of its own path,
# with two caps that keep the prediction meaningful:
#   * MAX_SIM_TIME_S, because predicting several seconds ahead for a robot that
#     re-decides every 100 ms is fiction, and
#   * MAX_SWEEP_RAD, because past a quarter turn an arc curls back on itself and the
#     "path" stops describing anything the robot will actually do.
LOOKAHEAD_M = 0.60          # ~safe_distance: see at least as far as the governor cares
MAX_SIM_TIME_S = 4.0
MAX_SWEEP_RAD = math.pi / 2

# NO TUNNELLING: the step is v*T/K, and in the regime where T is set by LOOKAHEAD_M
# that is LOOKAHEAD_M/K = 0.05 m regardless of speed -- far below the ~0.26 m body
# clearance, so a thin obstacle cannot fall between two samples. test_dwa asserts it.

# Score weights. Heading keeps her pointed where the planner/policy wants; clearance
# keeps her off the walls; VELOCITY IS WEIGHTED HIGH ON PURPOSE, because a command
# that makes no progress is the failure this module exists to remove -- with velocity
# cheap, the highest-scoring "safe" answer is always to sit still and turn.
W_HEADING = 1.0
W_CLEARANCE = 0.4
W_VELOCITY = 0.7

# Clearance beyond this is not worth more score. Without a cap, open space dominates
# every other term and she drives down the middle of the room ignoring the target.
CLEARANCE_CAP_M = 1.20


@dataclass
class DWAResult:
    v: float
    omega: float
    clearance: float          # [m] closest the body comes to anything along the arc
    score: float
    considered: int           # candidates sampled
    safe: int                 # candidates that survived collision checking
    ms: float                 # wall time spent choosing
    why: str


class DynamicWindowPlanner:
    """Chooses (v, omega) by simulating arcs against the live scan.

    Stateless between calls apart from counters and a cached beam trig table; the scan
    is the only world model it uses, so nothing here can drift the way dead reckoning
    does. That is deliberate -- this is the layer that has to be right when the map is
    wrong.
    """

    def __init__(self, cfg,
                 v_samples: int = V_SAMPLES,
                 omega_samples: int = OMEGA_SAMPLES,
                 sim_time_s: float = SIM_TIME_S,
                 sim_steps: int = SIM_STEPS,
                 w_heading: float = W_HEADING,
                 w_clearance: float = W_CLEARANCE,
                 w_velocity: float = W_VELOCITY) -> None:
        self.cfg = cfg
        self.v_samples = int(v_samples)
        self.omega_samples = int(omega_samples)
        self.sim_time_s = float(sim_time_s)
        self.sim_steps = int(sim_steps)
        self.w_heading = float(w_heading)
        self.w_clearance = float(w_clearance)
        self.w_velocity = float(w_velocity)

        self._trig: tuple[np.ndarray, np.ndarray] | None = None
        self._times: np.ndarray | None = None

        # Telemetry. `blocked` is the interesting one: it counts ticks where NOTHING
        # forward was drivable, which is the honest definition of "she had to fall back
        # on the old behaviour" and should be rare in an open room.
        self.calls = 0
        self.chosen = 0
        self.blocked = 0
        self.last_ms = 0.0

    # ---------------------------------------------------------------- geometry --
    def _beam_trig(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(cos, sin) per beam. Beam 0 down the nose, matching LidarGovernor."""
        if self._trig is None or self._trig[0].size != n:
            bearing = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
            self._trig = (np.cos(bearing), np.sin(bearing))
        return self._trig

    def _horizons(self, vs: np.ndarray, omegas: np.ndarray) -> np.ndarray:
        """Per-candidate simulation time [s]. See LOOKAHEAD_M for why it is not fixed."""
        reach = LOOKAHEAD_M / np.maximum(np.abs(vs), 1e-6)
        t = np.maximum(self.sim_time_s, reach)
        t = np.minimum(t, MAX_SIM_TIME_S)
        return np.minimum(t, MAX_SWEEP_RAD / np.maximum(np.abs(omegas), 1e-6))

    def _time_steps(self, horizons: np.ndarray) -> np.ndarray:
        """(M, K) sample times. Starts at the FIRST step, never t=0.

        Including t=0 would call every command a collision the moment anything sits
        inside the clearance radius -- including her own settled position against the
        wall she is trying to drive away from.
        """
        frac = np.linspace(1.0 / self.sim_steps, 1.0, self.sim_steps)
        return horizons[:, None] * frac[None, :]

    def _obstacle_points(self, scan_m: np.ndarray,
                         blind_beams: np.ndarray | None) -> np.ndarray:
        """Scan -> (N, 2) obstacle points in the robot frame. Invalid beams dropped.

        THREE KINDS OF BEAM ARE NOT OBSTACLES, and conflating any of them with free
        space has already cost this project a debugging session:

          * `>= lidar_max_range` means the beam saw NOTHING, not that the space is
            clear to 5 m. `LocalExplorationMap` once carved free space through the
            chassis exactly this way.
          * `< min_valid` is inside her own footprint -- geometrically impossible, so
            it is a self-return or speckle, and it is already rejected at the source.
          * a masked beam is looking at her own body.

        Dropping them is safe HERE in a way it would not be in a free-space map,
        because this module only ever asks "does anything I can see block this arc" --
        a beam that saw nothing cannot block anything, and the ToF reflex and the
        governor's scrape guard still cover the near field underneath.
        """
        n = scan_m.size
        cos_b, sin_b = self._beam_trig(n)
        r = np.asarray(scan_m, dtype=np.float64)

        min_valid = float(getattr(self.cfg, "lidar_floor", self.cfg.robot_radius))
        valid = np.isfinite(r) & (r >= min_valid) & (r < self.cfg.lidar_max_range)
        if blind_beams is not None and np.size(blind_beams):
            valid[np.asarray(blind_beams, dtype=np.int32)] = False
        if not valid.any():
            return np.zeros((0, 2), dtype=np.float64)
        return np.stack([r[valid] * cos_b[valid], r[valid] * sin_b[valid]], axis=1)

    def _arcs(self, vs: np.ndarray, omegas: np.ndarray) -> tuple[np.ndarray, np.ndarray,
                                                                np.ndarray]:
        """Forward-simulate every (v, omega) pair. -> (x, y, theta_end), each (M, K)/(M,).

        Exact arc integration, not Euler stepping: for constant (v, omega) the path is
        a circle of radius v/omega, and the closed form has no accumulating error. The
        straight-line case is separated out because v/omega diverges as omega -> 0.
        """
        t = self._time_steps(self._horizons(vs, omegas))    # (M, K)
        v = vs[:, None]                                     # (M, 1)
        w = omegas[:, None]                                 # (M, 1)
        theta = w * t                                       # (M, K)

        straight = np.abs(w) < 1e-6
        radius = np.divide(v, w, out=np.zeros_like(v), where=~straight)

        x = np.where(straight, v * t, radius * np.sin(theta))
        y = np.where(straight, np.zeros_like(theta), radius * (1.0 - np.cos(theta)))
        return x, y, theta[:, -1]

    # ------------------------------------------------------------------ choose --
    def choose(self, scan_m: np.ndarray, v_allow: float, omega_want: float,
               desired_bearing: float | None = None,
               blind_beams: np.ndarray | None = None,
               v_floor: float = 0.15) -> DWAResult | None:
        """Best drivable arc, or None if nothing forward survives.

        `v_allow`  -- the fastest this stage may return. Comes in already reduced by
                      the policy and anything upstream; never exceeded.
        `omega_want` -- the turn the policy asked for. Used as the heading intent when
                      no exploration target is supplied, so the NETWORK still steers.
        `desired_bearing` -- relative bearing to the exploration target [rad], if any.
        `v_floor`  -- do not sample below the drivetrain's executable speed. Commands
                      under it become creep-pulses or stops downstream, so proposing
                      them is proposing nothing.

        None means "no forward arc is safe" and the caller MUST fall through to the
        existing behaviour. It is never a command.
        """
        t0 = time.perf_counter()
        self.calls += 1

        v_allow = float(min(v_allow, self.cfg.v_max))
        if v_allow <= 0.0:
            self.blocked += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            return None

        lo = min(float(v_floor), v_allow)
        vs = (np.array([v_allow], dtype=np.float64) if lo >= v_allow
              else np.linspace(lo, v_allow, self.v_samples))
        omegas = np.linspace(-self.cfg.omega_max, self.cfg.omega_max, self.omega_samples)

        vv, ww = np.meshgrid(vs, omegas, indexing="ij")
        vv, ww = vv.ravel(), ww.ravel()                    # (M,)
        x, y, theta_end = self._arcs(vv, ww)               # (M, K), (M, K), (M,)

        obstacles = self._obstacle_points(scan_m, blind_beams)
        clear_r = float(self.cfg.robot_radius + self.cfg.collision_margin)

        if obstacles.shape[0] == 0:
            # Nothing visible blocks anything. Every candidate is safe and clearance is
            # unbounded -- which is a real state (an open room, or a dead LiDAR), and
            # the staleness check upstream is what distinguishes those two, not this.
            clearance = np.full(vv.shape, np.inf)
            safe = np.ones(vv.shape, dtype=bool)
        else:
            dx = x[:, :, None] - obstacles[None, None, :, 0]
            dy = y[:, :, None] - obstacles[None, None, :, 1]
            d = np.sqrt(dx * dx + dy * dy)                 # (M, K, N)
            clearance = d.min(axis=(1, 2))                 # (M,)
            safe = clearance > clear_r

        if not safe.any():
            self.blocked += 1
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            return None

        # ---- score the survivors ------------------------------------------------
        # Heading intent: the exploration target if there is one, otherwise where the
        # policy was trying to turn over this horizon. Either way the arc that ends up
        # pointing closest to it wins that term.
        want = (float(desired_bearing) if desired_bearing is not None
                else float(omega_want) * self.sim_time_s)
        want = math.atan2(math.sin(want), math.cos(want))
        err = np.abs(np.arctan2(np.sin(want - theta_end), np.cos(want - theta_end)))
        heading_score = 1.0 - err / math.pi

        clear_score = np.minimum(clearance, CLEARANCE_CAP_M) / CLEARANCE_CAP_M
        vel_score = vv / max(v_allow, 1e-6)

        score = (self.w_heading * heading_score
                 + self.w_clearance * clear_score
                 + self.w_velocity * vel_score)
        score = np.where(safe, score, -np.inf)

        best = int(np.argmax(score))
        self.chosen += 1
        self.last_ms = (time.perf_counter() - t0) * 1000.0
        return DWAResult(
            v=float(vv[best]),
            omega=float(ww[best]),
            clearance=float(clearance[best]),
            score=float(score[best]),
            considered=int(vv.size),
            safe=int(safe.sum()),
            ms=self.last_ms,
            why=(f"arc v={vv[best]:.2f} w={ww[best]:+.2f} clear={clearance[best]:.2f}m "
                 f"({int(safe.sum())}/{int(vv.size)} safe)"),
        )
