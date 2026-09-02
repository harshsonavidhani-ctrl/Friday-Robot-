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

RECOVERY PIVOTS, AND WHEN THAT CANNOT WORK IT BACKS OFF
-------------------------------------------------------
This module used to command a pivot in place and NOTHING else, on the stated grounds
that "this drivetrain cannot execute reverse". That premise was measured and is false:
reverse was being thrown away by the capability adapter before it ever reached the
driver, and with the adapter disabled it executes (the driver lifts a -0.150 m/s
request to the stiction floor). See the pi_deploy commit of 2026-09-02.

The old behaviour failed exactly where it mattered most. `boxed_in` means "forward
blocked", and nose-against-an-obstacle is the one trap a pivot cannot escape: inside
the pivot sweep the ToF reflex freezes rotation as well -- correctly, or she would
grind along whatever she is touching. Measured: 86 recovery ticks produced 84 pivots
and ZERO translation while she sat 5.5 cm from a wall.

So a pivot is still the first move for the traps a turn CAN fix (oscillation,
spinning, no-progress). For `boxed_in`, and for any escalation -- where "pivoting did
not work" is itself evidence that pivoting is not the manoeuvre -- recovery reverses
while turning toward open space.

None of this bypasses the safety chain. Recovery issues a REQUEST; the governor, the
reflex and the limiter all still run on it, and forward motion stays vetoed by the
reflex regardless of what is asked for here.
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

# A recovery maneuver used to run its FULL tick budget with no check that it had
# worked. Measured over a 90 s exploration run: the policy asked to go forward on
# 222 of 316 ticks and moved on 9, while RECOVERY_CONTROLLER held authority for 112
# -- and `_maneuver` only ever returns a pivot or a reverse, so every one of those
# ticks made forward motion impossible BY CONSTRUCTION, whether or not the way had
# already opened. A level-1 pivot that clears the obstruction on tick 2 still burned
# the remaining 6.
#
# So a maneuver is now released as soon as its own trigger stops firing. This only
# ever hands authority BACK to the policy sooner; the governor, visual gate, ToF
# reflex and rate limiter all sit downstream and are untouched, so it cannot widen
# any safety margin. Escalation is unchanged -- VERIFY still has to pass.
MIN_MANEUVER_TICKS = 3          # let a maneuver actually do something before judging it
CLEARED_TRAVEL_M = 0.05         # movement that says a motion-history trigger has broken
CLEARED_AHEAD_MARGIN_M = 0.10   # clearance past hard_stop before calling a path open


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
        self.trigger: str | None = None
        self.triggers = {"oscillation": 0, "spinning": 0, "no_progress": 0,
                         "boxed_in": 0, "retreading": 0}
        self.recoveries = 0
        self.successes = 0
        self.escalations = 0
        self.early_exits = 0
        self._maneuver_ticks = 0
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

    def _cleared(self, scan: np.ndarray, boxed_in: bool) -> bool:
        """Has the condition that triggered THIS maneuver stopped holding?

        Deliberately trigger-specific rather than "is _detect quiet". `_detect` is
        ordered by evidence and would answer for whichever condition happens to fire
        now, which is not the question -- a spin recovery must not be released because
        the robot is no longer boxed in.
        """
        if self.trigger == "boxed_in":
            # NOT `not boxed_in`. That flag is a stuck COUNTER
            # (`_stuck_ticks >= escape_after_ticks`), so it goes false the instant the
            # maneuver starts moving her -- which says the pivot began, not that it
            # achieved anything. Releasing on it measured worse than not releasing at
            # all: recovery authority fell 112 -> 12 ticks over a 90 s run while
            # forward ticks went 10 -> 0, because she was handed back into a path that
            # was still blocked and then sat through VERIFY doing nothing.
            #
            # The maneuver exists to open a driveable heading, so that is what has to
            # be true to end it: real clearance in the forward arc, past the distance
            # at which the governor forbids forward motion anyway.
            arc = getattr(self.cfg, "governor_forward_arc_deg", 90.0)
            n = scan.size
            bearing = (np.arange(n) * (360.0 / n) + 180.0) % 360.0 - 180.0
            ahead = scan[np.abs(bearing) <= arc]
            if ahead.size == 0:
                return False
            return float(np.min(ahead)) > self.cfg.hard_stop + CLEARED_AHEAD_MARGIN_M
        # spinning / oscillation / no_progress are all "the robot is not getting
        # anywhere" measured over a window. The evidence they have broken is that it
        # has started covering ground again. A pivot recovery cannot satisfy this,
        # which is correct: pivoting is not progress.
        return self.hist.travel(MIN_MANEUVER_TICKS) >= CLEARED_TRAVEL_M

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
            self._maneuver_ticks += 1
            if self.remaining <= 0:
                self.state = "VERIFY"
                self.verify_left = VERIFY_TICKS
                return None
            if (self._maneuver_ticks >= MIN_MANEUVER_TICKS
                    and self._cleared(scan, boxed_in)):
                # Worked early. Hand the robot back now instead of pivoting at a gap
                # it can already drive through. VERIFY still runs, so a premature
                # release is caught and escalated exactly as a failed one would be.
                self.early_exits += 1
                self.state = "VERIFY"
                self.verify_left = VERIFY_TICKS
                return None
            return self._maneuver(f"recovery level {self.level + 1}")

        # --- did it work? ---------------------------------------------------------
        if self.state == "VERIFY":
            self.verify_left -= 1
            still = self._detect(scan, boxed_in)
            if still is not None and self.verify_left <= 0:
                self._escalate(still, pose, tick)
                return self._maneuver(f"recovery level {self.level + 1}: still stuck")
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
        return self._maneuver(f"recovery level 1: {found[0]}")

    def _begin(self, found: tuple[str, str], pose: np.ndarray, tick: int,
               open_side: float) -> None:
        trigger, detail = found
        self.trigger = trigger
        self.triggers[trigger] = self.triggers.get(trigger, 0) + 1
        self.recoveries += 1
        self.level = 0
        self.turn_sign = 1.0 if open_side >= 0 else -1.0
        self.remaining = LEVELS[0]["ticks"]
        self._maneuver_ticks = 0
        self.state = "RECOVERING"
        self.active_event = RecoveryEvent(
            t=time.time(), tick=tick, trigger=trigger, level=1, detail=detail,
            pose=tuple(float(x) for x in pose))
        self.events.append(self.active_event)

    def _maneuver(self, note: str) -> tuple[float, float, str]:
        """The command a recovery actually issues.

        This used to be a pivot, unconditionally and at every one of the three exit
        points -- `(0.0, turn_sign * omega_max)`, forward speed hardcoded to zero. So
        recovery could ROTATE and nothing else: measured over one run, 86 recovery
        ticks produced 84 pivots and ZERO translation.

        That is the wrong move for the case it most needs to handle. `boxed_in` means
        "forward blocked and no reverse available" -- its own detector says so -- and
        nose-against-an-obstacle is the one situation where turning cannot help:
        inside the pivot sweep the reflex freezes rotation anyway (correctly, or she
        would grind along whatever she is touching). Backing off is the only escape
        that exists, so recovery now takes it, and turns toward open space while it
        does. Escalation does the same, because "pivoting did not work" is evidence
        that pivoting is not the manoeuvre.

        Nothing here bypasses the safety chain: the reverse is a REQUEST, clamped by
        the governor and the reflex exactly like any other. It just stops asking for
        the one thing that provably cannot work.
        """
        if self.trigger == "boxed_in" or self.level >= 1:
            return (self.cfg.v_min, self.turn_sign * self.cfg.omega_max * 0.5,
                    f"{note}: backing off toward open space")
        return (0.0, self.turn_sign * self.cfg.omega_max, f"{note}: pivoting toward open space")

    def _escalate(self, found: tuple[str, str], pose: np.ndarray, tick: int) -> None:
        self.trigger = found[0]
        self.level = min(self.level + 1, len(LEVELS) - 1)
        self.escalations += 1
        self.remaining = LEVELS[self.level]["ticks"]
        self._maneuver_ticks = 0
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
            # Maneuvers released as soon as their own trigger stopped firing, rather
            # than running the full tick budget. High relative to `recoveries` means
            # the escapes are working quickly and the robot is getting its authority
            # back; zero means every recovery ran to its budget, which is what this
            # counter exists to make visible.
            "early_exits": self.early_exits,
            "triggers": dict(self.triggers),
            "active": None if self.active_event is None else {
                "trigger": self.active_event.trigger,
                "detail": self.active_event.detail,
                "level": self.active_event.level,
            },
        }
