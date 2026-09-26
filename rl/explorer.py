"""Frontier-based exploration planning for the RL stack.

THE PROBLEM THIS SOLVES
-----------------------
The policy is a good LOCAL navigator and a blind one. Its 146 inputs are 72 LiDAR
beams, 72 frontier rays and its own last action -- there is no goal channel, nothing
that says "the unexplored room is that way". So it explores the way water finds a
drain: locally downhill, tick by tick, with no notion of having chosen anywhere. That
produced 579 degrees of turning per metre travelled on the baseline run.

WHY THE TARGET IS NOT PUT INTO THE OBSERVATION
----------------------------------------------
The obvious move -- append a goal vector, or bend the frontier rays toward the chosen
target -- is the one thing that must not happen. Those 72 rays have the geometry,
units and scaling of `train.py`; anything else is out-of-distribution input to a
network that cannot report its own confusion. Writing a synthetic goal into a sensor
channel is also just lying to the policy, and it would be indistinguishable at
runtime from the sensor being broken.

So the planner does not talk to the policy at all. It arbitrates AFTER it:

    policy (real observations) ─> proposed omega
                                       │
    planner (persistent map) ──> desired bearing
                                       │
                                       v
              agree?  ──> pass through, ACTION_SOURCE = RL_POLICY
              differ? ──> BOUNDED bias toward target, ACTION_SOURCE = EXPLORATION_PLANNER

Three properties make this safe and honest:

  1. The planner only ever influences HEADING. Forward speed stays the policy's, and
     then the safety chain's. A planner cannot make the robot go faster at anything.
  2. The bias is CAPPED and additive, never a replacement. The policy keeps authority
     over local geometry, which it can see at 10 Hz and the coarse map cannot.
  3. Every biased tick is labelled EXPLORATION_PLANNER in telemetry. A rule that
     steers the robot while wearing the policy's name is the exact defect the
     provenance work exists to prevent.

The governor, visual gate, ToF reflex and rate limiter all run downstream of the
bias, unchanged, so nothing here can reduce a safety margin.

COST
----
MEASURED on this Pi over a 53 m^2 map: `DistanceField` 86.6 ms, `frontier.find`
178.5 ms -- 265 ms for one cycle, against a 100 ms control budget. It runs on its own
thread at 0.5 Hz and plans against a SNAPSHOT of the grid, because the mapper thread
reallocates that array as the map grows.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

_FRIDAY_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(_FRIDAY_ROOT) not in sys.path:
    sys.path.insert(0, str(_FRIDAY_ROOT))

from slam import frontier as _frontier  # noqa: E402
from slam.plan import DistanceField  # noqa: E402

# How close the robot must get before a target counts as reached. Generous, because
# the point of a frontier is to SEE past it, not to stand on it -- the unknown space
# behind it is revealed from several metres away, and insisting on arrival wastes the
# trip. `slam/frontier.py` already targets the centroid rather than the near corner
# for the same reason.
REACHED_M = 0.60

# A target that has survived this long without being reached is abandoned. Not because
# it is necessarily bad, but because the map underneath it has changed enough that the
# choice deserves remaking, and a robot pursuing a stale goal looks identical to one
# that is stuck.
TARGET_MAX_AGE_S = 45.0

# Bearing error below which the policy is considered to already agree. The policy is
# frontier-seeking on its own, so this is hit often and the planner stays quiet.
ALIGN_TOL_RAD = math.radians(35.0)

# Hard cap on the planner's steering contribution, as a fraction of omega_max. The
# policy must keep enough authority to dodge what the map cannot see.
MAX_BIAS_FRACTION = 0.55


class ExplorationPlanner(threading.Thread):
    """Chooses where to explore next, on its own thread, from the persistent map.

    Owns nothing about actuation. It publishes a target and a desired bearing; the
    control loop decides what to do with them and records that decision.
    """

    def __init__(self, world: Any, period_s: float = 2.0,
                 reached_m: float = REACHED_M) -> None:
        super().__init__(name="planner", daemon=True)
        self.world = world
        self.period = float(period_s)
        self.reached_m = float(reached_m)
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.target: Any = None                 # slam.frontier.Frontier | None
        self.target_since = 0.0
        self.targets_chosen = 0
        # ARRIVED and REVEALED are different outcomes and must not share a counter.
        # A parked robot with sensors running reports 11 "reached" targets if they do,
        # because frontiers near it stop being frontiers as the map fills in -- the
        # map completed, the robot went nowhere. Only `targets_arrived` is evidence
        # that driving happened.
        self.targets_arrived = 0      # robot physically got within reached_m
        self.targets_revealed = 0     # frontier ceased to exist without arrival
        self.targets_abandoned = 0
        self.cycles = 0
        self.last_ms = 0.0
        self.last_error: str = ""
        self.frontier_count = 0
        self.exhausted = False                  # no reachable frontier remains
        # Targets that could not be reached. Keyed coarsely by `Frontier.key()` so the
        # same place is recognised between cycles even as its cells shift.
        self.blacklist: set = set()
        # THE WAY BACK (rl/way_back.py). Once set, the target is the start and
        # nothing here replaces it: no frontier search, no abandoning, no
        # retiring on arrival -- the control loop decides when she is back.
        self.home: Any = None

    # ------------------------------------------------------------------ state ---
    def snapshot(self) -> dict[str, Any]:
        """Thread-safe view for telemetry. Plain data only -- no live objects."""
        with self._lock:
            t = self.target
            return {
                "has_target": t is not None,
                "target_xy": (float(t.approach[0]), float(t.approach[1])) if t else None,
                "target_gain": int(t.gain) if t else 0,
                "target_distance": float(t.distance) if t else None,
                "target_utility": float(t.utility) if t else 0.0,
                "target_age_s": (time.monotonic() - self.target_since) if t else 0.0,
                "frontiers": self.frontier_count,
                "exhausted": self.exhausted,
                "chosen": self.targets_chosen,
                "arrived": self.targets_arrived,
                "revealed": self.targets_revealed,
                "abandoned": self.targets_abandoned,
                "cycles": self.cycles,
                "plan_ms": self.last_ms,
                "blacklisted": len(self.blacklist),
                "error": self.last_error,
            }

    def desired_bearing(self, pose: np.ndarray) -> float | None:
        """World-frame bearing from the robot to the current target, or None."""
        with self._lock:
            t = self.target
        if t is None:
            return None
        return math.atan2(t.approach[1] - float(pose[1]),
                          t.approach[0] - float(pose[0]))

    def bearing_error(self, pose: np.ndarray) -> float | None:
        """Signed angle the robot must turn to face the target, wrapped to +-pi.

        Positive = turn left (counter-clockwise), matching the sign of `omega`.
        """
        b = self.desired_bearing(pose)
        if b is None:
            return None
        return (b - float(pose[2]) + math.pi) % (2.0 * math.pi) - math.pi

    def distance_to_target(self, pose: np.ndarray) -> float | None:
        with self._lock:
            t = self.target
        if t is None:
            return None
        return math.hypot(t.approach[0] - float(pose[0]),
                          t.approach[1] - float(pose[1]))

    def go_home(self, target: Any) -> None:
        """Steer for `target` (the start) from now on, instead of a frontier.

        Called again whenever the start's estimate moves; the same object may be
        passed each time. Everything downstream -- the steering bias, the local
        navigator -- reads the target exactly as it read a frontier.
        """
        with self._lock:
            if self.home is not target:
                self.target_since = time.monotonic()
            self.home = target
            self.target = target

    # --------------------------------------------------------------- lifecycle --
    def stop(self) -> None:
        self._stop.set()

    def note_progress(self, pose: np.ndarray) -> None:
        """Called from the control loop: retire a target once it has been reached.

        Cheap on purpose -- a distance comparison, no planning. Reaching a target must
        be noticed at control rate, not on the next 2 s planning cycle, or the robot
        keeps steering toward somewhere it is already standing.
        """
        if self.home is not None:
            return
        d = self.distance_to_target(pose)
        if d is not None and d <= self.reached_m:
            with self._lock:
                if self.target is not None:
                    self.targets_arrived += 1
                    self.target = None

    def _plan_once(self) -> None:
        if self.home is not None:
            return
        grid = self.world.grid_snapshot()
        if grid is None or grid.shape == (1, 1):
            return
        pose = np.array(self.world.pose, dtype=float)

        t0 = time.perf_counter()
        try:
            field = DistanceField(grid, (float(pose[0]), float(pose[1])))
            if not field.ok:
                # Not an error: it happens when the robot's own cell is not yet
                # marked passable, which is normal in the first seconds of a run.
                self.last_error = "no reachable space yet"
                return
            options = _frontier.find(grid, field, ignore=self.blacklist)
        except Exception as exc:                          # noqa: BLE001
            # Planning must never take the robot down. Without a target the loop
            # simply runs as pure RL, which is exactly the pre-Phase-3 behaviour.
            self.last_error = f"{type(exc).__name__}: {exc}"
            return
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            self.cycles += 1

        self.last_error = ""
        self.frontier_count = len(options)

        with self._lock:
            if self.home is not None:
                return                  # go_home() landed while this cycle planned
            current = self.target
            self.exhausted = not options

            if not options:
                self.target = None
                return

            best = options[0]
            if current is None:
                self.target = best
                self.target_since = time.monotonic()
                self.targets_chosen += 1
                return

            age = time.monotonic() - self.target_since

            # Is the current target still on the board? A frontier that has been
            # consumed -- the unknown behind it now mapped -- vanishes from `options`,
            # and continuing to drive at it would be chasing a place that no longer
            # has anything to reveal.
            still_there = any(o.key() == current.key() for o in options)

            if not still_there:
                # The unknown behind it is now mapped. That is a good outcome and it
                # is NOT an arrival -- it happens to a stationary robot too.
                self.targets_revealed += 1
                self.target = best
                self.target_since = time.monotonic()
                self.targets_chosen += 1
                return

            if age > TARGET_MAX_AGE_S:
                # Abandon AND blacklist: 45 s of trying without arriving is evidence
                # about this destination, not just about this moment.
                self.blacklist.add(current.key())
                self.targets_abandoned += 1
                self.target = best
                self.target_since = time.monotonic()
                self.targets_chosen += 1
                return

            # Switch only for a decisively better option. Hysteresis matters more than
            # it looks: frontier utilities shuffle every cycle as the map fills in, and
            # a planner that always takes the current maximum oscillates between two
            # near-equal goals and drives nowhere.
            if best.key() != current.key() and best.utility > current.utility * 1.5:
                self.target = best
                self.target_since = time.monotonic()
                self.targets_chosen += 1

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                self._plan_once()
            except Exception as exc:                      # noqa: BLE001
                self.last_error = f"{type(exc).__name__}: {exc}"
            self._stop.wait(self.period)


def steering_bias(error_rad: float | None, omega_policy: float, omega_max: float,
                  align_tol: float = ALIGN_TOL_RAD,
                  max_fraction: float = MAX_BIAS_FRACTION) -> tuple[float, str]:
    """How much to nudge omega toward the target. Returns (delta, reason).

    Returns (0.0, reason) whenever the planner should stay out of the way, which is
    the common case -- the policy is frontier-seeking on its own and usually already
    heading somewhere sensible.

    The bias is proportional to the part of the error BEYOND the tolerance, so it
    fades to zero as the robot comes into alignment instead of switching off at a
    threshold and causing a visible twitch.
    """
    if error_rad is None:
        return 0.0, "no target"
    if abs(error_rad) <= align_tol:
        return 0.0, "policy already aligned with target"

    # Already turning the right way, and hard? Leave it alone. Adding to a turn the
    # policy is already committed to is how a robot overshoots and starts weaving.
    excess = abs(error_rad) - align_tol
    want = math.copysign(1.0, error_rad)
    if math.copysign(1.0, omega_policy) == want and abs(omega_policy) >= 0.5 * omega_max:
        return 0.0, "policy already turning toward target"

    cap = max_fraction * omega_max
    delta = want * min(cap, (excess / math.pi) * 2.0 * cap)
    return float(delta), (f"target is {math.degrees(abs(error_rad)):.0f} deg "
                          f"{'left' if error_rad > 0 else 'right'}")
