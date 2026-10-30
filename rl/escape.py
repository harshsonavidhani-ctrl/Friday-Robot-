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
BACK_MAX_TICKS = 40          # ~4 s at 10 Hz: she backs until there is room to turn
# 2026-10-31, the 5-minute run. 507 of the ticks with the bumper within 20 cm were this
# turn, at full power, flipping sides. Turns begun with the bumper ...
#   under 10 cm reached < 8 cm 95% of the time, 10-20 cm 50%, 20-30 cm 11%, 30-40 cm 4%.
# So she does not turn until it reads this much, unless she truly cannot back up. The
# ToF measures it: "backed 0.15 m" from the commanded speed was really 2-5 cm.
TURN_ROOM_M = 0.30
# -0.15 m/s only hums on the floor (M-81: 1.1 cm/s measured, 29 of 37 back-outs); the
# stage that lifts reverses to reverse_floor_mps runs before this one. 2026-10-10 floor
# test: -0.20 moved 9.8 cm in 1 s, -0.25 moved 13.5 cm.
BACK_SPEED_MPS = 0.25
# ...and she always gains this much first, however much she has: she stops at 0.38 m,
# so "room already" meant no back-out at all and the turn swung her corners into low
# things off to the side, out of the ToF's view (sim, low_things: 0 -> 11 bumps).
BACK_GAIN_M = 0.12
# Under this the other stages may not turn her at all (cfg.tof_pivot_min_clearance, was
# 0.12): stopped, she is blocked, and this escape backs her out. It alone may turn
# closer, and only when she cannot back up.
NO_TURN_M = 0.20
TURN_MAX_TICKS = 25          # ~2.5 s
TURN_MAX_RAD = math.radians(150.0)
CLEAR_TICKS = 2              # the way ahead must stay clear this long to exit
COOLDOWN_TICKS = 15          # give the navigator 1.5 s to drive before re-arming
REAR_MARGIN_M = 0.05         # rear must be this much beyond rear_stop to back up
# 2026-10-30, the real wall: boxed in, she turned the same way 13 times in 39 s with
# the bumper closing 0.07 -> 0.02 m each time (and the fast reflex stands aside for
# this turn, so nothing else was going to stop it). A turn that brings the bumper
# this much closer, once it is this near, is the wrong way: leave it at once.
TURN_WATCH_M = TURN_ROOM_M
TURN_CLOSE_M = 0.04
# Not freed: the other side next, and soon. COOLDOWN_TICKS is for a way out found.
RETRY_TICKS = 3


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
        self._back_goal = TURN_ROOM_M
        self._turned = 0.0
        self._clear = 0
        self._turn_tof = None        # the widest the bumper has been in this turn
        self._flipped = False
        self._bad_side = 0.0         # the way that did not free her last time
        # Telemetry, so a run can show the escape did its job.
        self.episodes = 0
        self.backs = 0
        self.turns = 0
        self.flips = 0
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
        self._turn_tof = None
        if phase == BACK:
            self.backs += 1
        elif phase == TURN:
            self.turns += 1

    def _exit(self, clear: bool) -> None:
        if clear:
            self.exits_clear += 1
            self._bad_side = 0.0
        else:
            self.exits_bounded += 1
            self._bad_side = self._side
        self.phase = IDLE
        self._blocked = 0
        self._cooldown = COOLDOWN_TICKS if clear else RETRY_TICKS

    def _begin_back(self, room_m: float) -> None:
        self._enter(BACK)
        self._back_goal = max(TURN_ROOM_M, room_m + BACK_GAIN_M)

    def _back(self, room_m: float, rear_min_m: float) -> tuple[float, float, str]:
        c = self.cfg
        v = max(abs(c.v_min), float(getattr(c, "reverse_floor_mps", 0.0)), BACK_SPEED_MPS)
        return (-v, 0.0, f"escape BACK {self._ticks}/{BACK_MAX_TICKS}: {room_m:.2f} m "
                f"ahead, needs {self._back_goal:.2f}, {rear_min_m:.2f} m clear behind")

    def step(self, *, blocked: bool, ahead_m: float, rear_min_m: float,
             can_reverse: bool, open_side: float, exec_v: float,
             exec_omega: float, dt: float, bumper_m: float = math.inf,
             tof_side: float = 0.0) -> tuple[float, float, str] | None:
        """(v, omega, why) to command this tick, or None to leave the tick alone.

        `blocked`   -- forward refused and she is not moving forward this tick.
        `ahead_m`   -- the nearest thing ahead from every sensor (LiDAR corridor
                       and ToF), which decides when TURN has found a way out.
        `rear_min_m`-- the rear gate's own clearance, so BACK and the gate agree.
        `open_side` -- +1 left / -1 right from the shared chooser; read ONCE.
        `exec_v`, `exec_omega` -- what the wheels actually did, for the bounds.
        `bumper_m`  -- the ToF alone: room to turn, and a turn it says is closing
                       is left at once.
        `tof_side`  -- which way is AWAY from what the bumper sees to one side, 0 if
                       it is dead ahead. It beats open_side: the LiDAR is over the
                       top of what stopped her.
        """
        c = self.cfg
        room = bumper_m if math.isfinite(bumper_m) else ahead_m
        can_back = can_reverse and rear_min_m > c.rear_stop + REAR_MARGIN_M

        if self.phase == IDLE:
            if self._cooldown > 0:
                self._cooldown -= 1
                self._blocked = 0
                return None
            self._blocked = self._blocked + 1 if blocked else 0
            if self._blocked < TRIGGER_TICKS:
                return None
            self.episodes += 1
            if tof_side:
                self._side = 1.0 if tof_side > 0 else -1.0   # decided ONCE
            else:
                self._side = 1.0 if open_side >= 0 else -1.0
                if self._side == self._bad_side:
                    self._side = -self._side                 # not the way that failed
            self._flipped = False
            if can_back:
                self._begin_back(room)
            else:
                self._enter(TURN)

        self.ticks_active += 1

        if self.phase == BACK:
            self._ticks += 1
            self._backed += abs(float(exec_v)) * dt
            if can_back and room < self._back_goal and self._ticks <= BACK_MAX_TICKS:
                return self._back(room, rear_min_m)
            self._enter(TURN)

        # TURN
        widest = min(bumper_m, TURN_WATCH_M + TURN_CLOSE_M)    # 4 m is "nothing there"
        if self._turn_tof is None or widest > self._turn_tof:
            self._turn_tof = widest
        elif (bumper_m < TURN_WATCH_M
              and self._turn_tof - bumper_m >= TURN_CLOSE_M):
            if self._flipped:                     # closing both ways: stop pushing
                self._exit(clear=False)
                return None
            self._flipped = True
            self.flips += 1
            self._side = -self._side
            if can_back:                          # room first, then the other way
                self._begin_back(room)
                self._ticks = 1
                return self._back(room, rear_min_m)
            self._ticks, self._turned, self._turn_tof = 0, 0.0, bumper_m
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
