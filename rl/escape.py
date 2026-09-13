"""Decisive escape: one committed manoeuvre instead of a new answer every tick.

Harsh, 2026-09-13: "change its behaviour of exploring ... make it like behave and
explore like professional rovers do", and he chose this fix second, after the
short-term obstacle memory.

WHY THIS EXISTS. When she is blocked, FOUR stages each make their own decision
every tick from their own counter -- the escape pivot (2b), the reverse escape
(2b-bis), recovery (2c) and the rear gate (2d) -- with the navigator and the
policy re-asking to drive forward in between. The run logs show the result as a
gate column that changes almost every line:

    09-13 09:50  reverse_escape, planner, reverse_escape, governor, planner,
                 reverse_escape, reverse_escape, governor, navigator, ...

and MEASURED on the 2026-09-12 runs: forward on 2-22% of ticks, pivoting in
place on 26-77%, stopped on 9-41%. A tick of backing up is undone by a tick of
creeping forward; a pivot left starts, then flips. Nothing is ever finished.

A professional rover does the opposite: it decides ONCE and carries the
manoeuvre through --

    BACK  straight back, away from what stopped her, while the rear is clear,
          for a bounded distance;
    TURN  pivot toward the more open side, the side chosen ONCE at the start
          and never flipped, until the way ahead is clear or a bound is hit;
    then hand control back to the navigator, with a cooldown so the escape
    cannot immediately re-trigger on the same obstacle.

WHAT IT CANNOT DO. It only ever asks for a straight reverse or a pivot, and
every safety stage still runs on the result: the rear gate (2d) refuses a
blocked reverse, the ToF reflex freezes a pivot that would scrape, and the rate
limiter shapes the command. When the rear gate refuses BACK, the next tick sees
the rear is not clear and moves on to TURN instead of arguing with the gate.
Every phase is bounded in ticks, so it can never hold the robot forever.
"""
from __future__ import annotations

import math

IDLE = "idle"
BACK = "back"
TURN = "turn"

TRIGGER_TICKS = 5            # blocked this long = stuck, not a passing brake
BACK_MAX_TICKS = 15          # ~1.5 s at 10 Hz
BACK_DIST_M = 0.15           # executed distance that is enough to turn in
TURN_MAX_TICKS = 25          # ~2.5 s
TURN_MAX_RAD = math.radians(150.0)
CLEAR_TICKS = 2              # the way ahead must stay clear this long to exit
COOLDOWN_TICKS = 15          # give the navigator 1.5 s to drive before re-arming
REAR_MARGIN_M = 0.05         # rear must be this much beyond rear_stop to back up


class EscapeManeuver:
    """BACK -> TURN -> hand back. Decides once per episode; bounded everywhere."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.phase = IDLE
        self._blocked = 0
        self._ticks = 0
        self._cooldown = 0
        self._side = 1.0
        self._backed = 0.0
        self._turned = 0.0
        self._clear = 0
        # Telemetry, so a run can show the escape did its job.
        self.episodes = 0
        self.backs = 0
        self.turns = 0
        self.exits_clear = 0
        self.exits_bounded = 0
        self.ticks_active = 0

    @property
    def active(self) -> bool:
        return self.phase != IDLE

    @property
    def side(self) -> float:
        return self._side

    def _enter(self, phase: str) -> None:
        self.phase = phase
        self._ticks = 0
        self._backed = 0.0
        self._turned = 0.0
        self._clear = 0
        if phase == BACK:
            self.backs += 1
        elif phase == TURN:
            self.turns += 1

    def _exit(self, clear: bool) -> None:
        if clear:
            self.exits_clear += 1
        else:
            self.exits_bounded += 1
        self.phase = IDLE
        self._blocked = 0
        self._cooldown = COOLDOWN_TICKS

    def step(self, *, blocked: bool, ahead_m: float, rear_min_m: float,
             can_reverse: bool, open_side: float, exec_v: float,
             exec_omega: float, dt: float) -> tuple[float, float, str] | None:
        """(v, omega, why) to command this tick, or None to leave the tick alone.

        `blocked`   -- forward refused and she is not moving forward this tick.
        `ahead_m`   -- the nearest thing ahead from every sensor (LiDAR corridor
                       and ToF), which decides when TURN has found a way out.
        `rear_min_m`-- the rear gate's own clearance, so BACK and the gate agree.
        `open_side` -- +1 left / -1 right from the shared chooser; read ONCE.
        `exec_v`, `exec_omega` -- what the wheels actually did, for the bounds.
        """
        c = self.cfg

        if self.phase == IDLE:
            if self._cooldown > 0:
                self._cooldown -= 1
                self._blocked = 0
                return None
            self._blocked = self._blocked + 1 if blocked else 0
            if self._blocked < TRIGGER_TICKS:
                return None
            self.episodes += 1
            self._side = 1.0 if open_side >= 0 else -1.0     # decided ONCE
            rear_clear = rear_min_m > c.rear_stop + REAR_MARGIN_M
            self._enter(BACK if (can_reverse and rear_clear) else TURN)

        self.ticks_active += 1

        if self.phase == BACK:
            self._ticks += 1
            self._backed += abs(float(exec_v)) * dt
            rear_clear = rear_min_m > c.rear_stop + REAR_MARGIN_M
            if (rear_clear and self._backed < BACK_DIST_M
                    and self._ticks <= BACK_MAX_TICKS):
                return (c.v_min, 0.0,
                        f"escape BACK {self._ticks}/{BACK_MAX_TICKS}: backed "
                        f"{self._backed:.2f} m, {rear_min_m:.2f} m clear behind")
            self._enter(TURN)

        # TURN
        self._ticks += 1
        self._turned += abs(float(exec_omega)) * dt
        self._clear = self._clear + 1 if ahead_m >= c.safe_distance else 0
        if self._clear >= CLEAR_TICKS:
            self._exit(clear=True)
            return None
        if self._turned >= TURN_MAX_RAD or self._ticks > TURN_MAX_TICKS:
            self._exit(clear=False)
            return None
        return (0.0, self._side * c.omega_max,
                f"escape TURN {'left' if self._side > 0 else 'right'} "
                f"{self._ticks}/{TURN_MAX_TICKS}: {ahead_m:.2f} m ahead, "
                f"needs {c.safe_distance:.2f} m")
