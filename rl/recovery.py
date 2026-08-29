"""Stuck detection and recovery, from real motion history rather than from a timer.

WHAT WAS ALREADY THERE
----------------------
`ControlLoop` has had ONE trap detector since before this module: forward blocked by
the governor, not moving, for 5 consecutive ticks -> pivot toward open space. It was
written for a specific measured failure (parked at 0.33 m with the governor zeroing
forward on 268 of 300 ticks) and it works. It is kept, and it becomes one trigger
among several rather than the only one.

It cannot see the other ways this robot fails to make progress:

  OSCILLATION   left-right-left-right, each turn undoing the last. Every individual
                command is sane; the sequence goes nowhere. Invisible per-tick.
  SPINNING      commanding rotation with no translation for a long stretch. Different
                from a deliberate pivot only in DURATION, so only a window can tell.
  NO PROGRESS   commanding motion and not moving. On this drivetrain this is normal
                for a few ticks (the creep cutoff turns sub-floor requests into stops)
                and pathological for fifty.
  RE-TREADING   driving, genuinely, in circles -- over ground already covered. Only
                the persistent visit trace can see this one.

WHY DETECTION USES EXECUTED MOTION, NOT COMMANDED
-------------------------------------------------
The creep cutoff, stiction floor and duty cap make executed != commanded structurally
on this drivetrain: a 0.5 m/s request executes at 0.184, and a 0.06 m/s request
executes as a full stop. A detector fed commanded twists would conclude the robot was
travelling nicely while it sat buzzing against a wall -- which is exactly the failure
that made `--motor-min-duty 72` a default rather than an option.

RECOVERY IS A STATE MACHINE, AND IT VERIFIES
--------------------------------------------
    MONITOR -> (trigger) -> RECOVERING -> VERIFY -> MONITOR
                                 ^                     |
                                 +---- (still stuck) --+  escalate

Escalation matters. A robot that responds to every trap with the same 0.5 s pivot,
forever, is not recovering -- it is stuck in a louder way. Each successive failure
turns further and holds longer, and after the last level the planner's current target
is blacklisted, because at that point the destination is the problem.

Recovery only ever commands a PIVOT IN PLACE. A pivot cannot approach anything -- the
footprint does not translate -- so it reduces no safety margin, and the governor,
reflex and limiter all still run on the result. Reverse is not used because this
drivetrain cannot execute it (see the capability adapter in pi_deploy).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- detection windows -----------------------------------------------------------
# All in control ticks at 10 Hz.
OSC_WINDOW = 24                 # 2.4 s of turn history
OSC_MIN_FLIPS = 5               # direction reversals within the window
OSC_MIN_RATE = 0.25             # rad/s below which a "turn" is not really a turn

SPIN_WINDOW = 40                # 4 s
SPIN_MIN_ROTATION = math.radians(300.0)   # nearly a full turn...
SPIN_MAX_TRAVEL = 0.10                    # ...while going essentially nowhere

PROGRESS_WINDOW = 50            # 5 s
PROGRESS_MIN_TRAVEL = 0.08      # m of executed travel expected in that window
PROGRESS_MIN_COMMAND = 0.05     # only counts if it was ASKING to move

# Recovery escalation. Each level pivots further and holds longer.
LEVELS = (
    {"turn_frac": 0.70, "ticks": 8},      # ~0.8 s
    {"turn_frac": 0.90, "ticks": 14},     # ~1.4 s
    {"turn_frac": 1.00, "ticks": 22},     # ~2.2 s, then blacklist the target
)
VERIFY_TICKS = 30               # 3 s of normal driving before calling it recovered


@dataclass
class RecoveryEvent:
    """One recovery, logged. Every field is measured, none inferred."""
    t: float
    tick: int
    trigger: str
    level: int
    detail: str
    pose: tuple[float, float, float]
    resolved: bool = False
    resolved_after_ticks: int | None = None


@dataclass
class MotionHistory:
    """Rolling record of what the WHEELS did, not what was asked of them."""
    omega: deque = field(default_factory=lambda: deque(maxlen=max(OSC_WINDOW, SPIN_WINDOW)))
    speed: deque = field(default_factory=lambda: deque(maxlen=max(PROGRESS_WINDOW, SPIN_WINDOW)))
    cmd_v: deque = field(default_factory=lambda: deque(maxlen=PROGRESS_WINDOW))
    dt: float = 0.1

    def push(self, exec_v: float, exec_omega: float, cmd_v: float) -> None:
        self.omega.append(float(exec_omega))
        self.speed.append(float(exec_v))
        self.cmd_v.append(float(cmd_v))

    def travel(self, n: int) -> float:
        """Path length actually executed over the last n ticks."""
        s = list(self.speed)[-n:]
        return sum(abs(v) for v in s) * self.dt

    def rotation(self, n: int) -> float:
        w = list(self.omega)[-n:]
        return sum(abs(x) for x in w) * self.dt

    def flips(self, n: int, min_rate: float) -> int:
        """Direction reversals among turns big enough to count as turns."""
        w = [x for x in list(self.omega)[-n:] if abs(x) >= min_rate]
        return sum(1 for a, b in zip(w, w[1:]) if (a > 0) != (b > 0))


class RecoveryController:
    """Detects the ways this robot fails to progress, and drives a bounded escape."""

    def __init__(self, cfg: Any, world: Any = None) -> None:
        self.cfg = cfg
        self.world = world
        self.hist = MotionHistory(dt=1.0 / cfg.control_hz)
        self.state = "MONITOR"
        self.level = -1
        self.remaining = 0
        self.turn_sign = 1.0
        self.verify_left = 0
        self.events: list[RecoveryEvent] = []
        self.active_event: RecoveryEvent | None = None
        self.triggers = {"oscillation": 0, "spinning": 0, "no_progress": 0,
                         "boxed_in": 0, "retreading": 0}
        self.recoveries = 0
        self.successes = 0
        self.escalations = 0
        self._cooldown = 0

    # ------------------------------------------------------------------ detect ---
    def _detect(self, scan: np.ndarray, boxed_in: bool) -> tuple[str, str] | None:
        """Returns (trigger, detail) or None. Order is by how specific the evidence is."""
        h = self.hist

        if boxed_in:
            return ("boxed_in",
                    f"forward blocked with {float(np.min(scan)):.2f} m clearance "
                    f"and no reverse available")

        if len(h.omega) >= SPIN_WINDOW:
            rot, tr = h.rotation(SPIN_WINDOW), h.travel(SPIN_WINDOW)
            if rot >= SPIN_MIN_ROTATION and tr <= SPIN_MAX_TRAVEL:
                return ("spinning",
                        f"{math.degrees(rot):.0f} deg of rotation with {tr:.3f} m "
                        f"travelled over {SPIN_WINDOW} ticks")

        if len(h.omega) >= OSC_WINDOW:
            f = h.flips(OSC_WINDOW, OSC_MIN_RATE)
            if f >= OSC_MIN_FLIPS and h.travel(OSC_WINDOW) <= SPIN_MAX_TRAVEL * 2:
                return ("oscillation",
                        f"{f} turn reversals in {OSC_WINDOW} ticks while travelling "
                        f"{h.travel(OSC_WINDOW):.3f} m")

        if len(h.speed) >= PROGRESS_WINDOW:
            asked = sum(abs(v) for v in list(h.cmd_v)[-PROGRESS_WINDOW:]) / PROGRESS_WINDOW
            if asked >= PROGRESS_MIN_COMMAND and h.travel(PROGRESS_WINDOW) < PROGRESS_MIN_TRAVEL:
                return ("no_progress",
                        f"commanded {asked:.3f} m/s on average but executed "
                        f"{h.travel(PROGRESS_WINDOW):.3f} m in {PROGRESS_WINDOW} ticks")
        return None

    # ----------------------------------------------------------------- respond ---
    def update(self, scan: np.ndarray, exec_v: float, exec_omega: float,
               cmd_v: float, boxed_in: bool, pose: np.ndarray, tick: int,
               open_side: float) -> tuple[float, float, str] | None:
        """Called every tick. Returns (v, omega, reason) to FORCE, or None to pass.

        `open_side` is +1/-1 from the caller's live-scan direction chooser -- the same
        primitive the capability adapter uses. Recovery does not re-derive it: two
        implementations of "which way is clearer" that could disagree is a bug waiting
        to happen.
        """
        self.hist.push(exec_v, exec_omega, cmd_v)

        if self._cooldown > 0:
            self._cooldown -= 1

        # --- executing a recovery -------------------------------------------------
        if self.state == "RECOVERING":
            self.remaining -= 1
            if self.remaining <= 0:
                self.state = "VERIFY"
                self.verify_left = VERIFY_TICKS
                return None
            return (0.0, self.turn_sign * self.cfg.omega_max,
                    f"recovery level {self.level + 1}: pivoting toward open space")

        # --- did it work? ---------------------------------------------------------
        if self.state == "VERIFY":
            self.verify_left -= 1
            still = self._detect(scan, boxed_in)
            if still is not None and self.verify_left <= 0:
                self._escalate(still, pose, tick)
                return (0.0, self.turn_sign * self.cfg.omega_max,
                        f"recovery level {self.level + 1}: still stuck, turning further")
            if self.verify_left <= 0:
                # Cleared. Record it and stand down.
                if self.active_event is not None:
                    self.active_event.resolved = True
                    self.active_event.resolved_after_ticks = tick - self.active_event.tick
                    self.successes += 1
                    self.active_event = None
                self.state = "MONITOR"
                self.level = -1
                self._cooldown = 20     # do not immediately re-trigger on the same window
            return None

        # --- monitoring -----------------------------------------------------------
        if self._cooldown > 0:
            return None
        found = self._detect(scan, boxed_in)
        if found is None:
            return None
        self._begin(found, pose, tick, open_side)
        return (0.0, self.turn_sign * self.cfg.omega_max,
                f"recovery level 1: {found[0]}")

    def _begin(self, found: tuple[str, str], pose: np.ndarray, tick: int,
               open_side: float) -> None:
        trigger, detail = found
        self.triggers[trigger] = self.triggers.get(trigger, 0) + 1
        self.recoveries += 1
        self.level = 0
        self.turn_sign = 1.0 if open_side >= 0 else -1.0
        self.remaining = LEVELS[0]["ticks"]
        self.state = "RECOVERING"
        self.active_event = RecoveryEvent(
            t=time.time(), tick=tick, trigger=trigger, level=1, detail=detail,
            pose=tuple(float(x) for x in pose))
        self.events.append(self.active_event)

    def _escalate(self, found: tuple[str, str], pose: np.ndarray, tick: int) -> None:
        self.level = min(self.level + 1, len(LEVELS) - 1)
        self.escalations += 1
        self.remaining = LEVELS[self.level]["ticks"]
        self.state = "RECOVERING"
        if self.active_event is not None:
            self.active_event.level = self.level + 1
        # Last resort: the destination itself is suspect, so stop pursuing it.
        if self.level == len(LEVELS) - 1:
            self.blacklist_requested = True

    # -------------------------------------------------------------------- state --
    @property
    def at_last_level(self) -> bool:
        return self.level >= len(LEVELS) - 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "level": self.level + 1 if self.level >= 0 else 0,
            "recoveries": self.recoveries,
            "successes": self.successes,
            "escalations": self.escalations,
            "triggers": dict(self.triggers),
            "active": None if self.active_event is None else {
                "trigger": self.active_event.trigger,
                "detail": self.active_event.detail,
                "level": self.active_event.level,
            },
        }
