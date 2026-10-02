"""Undo the last move: the way out when every safety rule says "don't".

Harsh, 2026-09-28: "it gets stucked at one place sometimes rather than trying to
avoid it and like continuing the explore ... dont bump okay". He chose "undo the
last move" over a plain back-off once the numbers were on the table.

THE LOCK THIS EXISTS FOR (run of 2026-09-27 22:00:44, 2 min 40 s, until the run
ended):
    forward   vetoed by the ToF reflex        (bumper zone 0.09 m, stop 0.24)
    turning   frozen by the same reflex       (0.09 m <= tof_pivot_min_clearance 0.12)
    reverse   refused by the rear gate        (rear 0.33 m <= rear_stop 0.36)
Three correct rules, and together they said (0, 0) on every tick while the
policy asked to pivot 238 times and the narrator said "Recovery successful".
She had pivoted INTO it: the tick before, she was turning at -1.6 rad/s with the
LiDAR at 0.29-0.41 m, and something low came round into the bumper's view at
9 cm.

WHY UNDOING IS SAFE WHEN BACKING OFF IS NOT
-------------------------------------------
Reverse on this drivetrain is binary at 0.15 m/s, and the rear gate's own budget
puts the worst-case stop at 11.5 cm. She had 11 cm. A blind back-off is not
safe there. But the space she just moved through was under her own body a
moment ago. Retracing that move sweeps only space the robot has just occupied:
    - a TURN is undone by turning back the other way, for as long as she turned;
    - a DRIVE is undone by reversing along it, never further than what is known
      free behind her (the distance she drove, or the rear LiDAR's clearance past
      her body, whichever is more) minus the pessimistic coast, capped at 8 cm.
The rear LiDAR still guards every reverse: nothing inside REAR_GUARD_M. Forward
is never issued. The undo stops as soon as the bumper has room to turn
(the normal chain takes over) and aborts if the bumper reading gets closer.

After MAX_TRIES undos at the same spot she stops and asks for help, and the spot
goes on the avoid list so the planner stops steering her back into it.

Pure logic, no I/O: the control loop feeds it and applies what it returns.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Frozen this many consecutive ticks (10 Hz) before anything is undone. A brake
# that clears by itself in under a second is the chain working, not a lock.
LOCK_TICKS = 8
# How far back a move can be found to retrace.
HISTORY_S = 4.0
# Harsh's cap on a reversing undo.
UNDO_MAX_M = 0.08
# A reversing undo shorter than this is not worth issuing (it executes as a stop).
UNDO_MIN_M = 0.02
# No single undo runs longer than this, turn or drive.
UNDO_MAX_S = 1.5
# A turn shorter than this is too short to break static friction; undo it for at
# least this long. The swept space is still the one she just turned through, as
# long as the original turn was at least this long -- shorter turns are not
# retraced as turns at all (see _plan).
TURN_MIN_S = 0.3
# The pessimistic coast at v_min, from RobotConfig's rear_stop budget
# ((0.15 / 0.403) * 0.22 m, scaled by v). Reverse travel plus this must fit in the
# known-free space behind her.
COAST_M = 0.082
# The rear LiDAR must show at least this much past her body for any reverse
# (Harsh's "4 cm behind the body").
REAR_GUARD_EXTRA_M = 0.04
# The undo has done its job once the bumper reads this much (the reflex's own
# freeze line + a margin, so the chain's escape turn is allowed at once).
ROOM_MARGIN_M = 0.04
# The bumper reading dropping this much during an undo means it is making things
# worse: stop.
WORSE_M = 0.02
# Locks within this distance of the last one are the same place.
SAME_SPOT_M = 0.30
# Undos per place before she asks for help.
MAX_TRIES = 2
# Free this long and the place's tries are forgiven.
FREE_RESET_S = 20.0
# KEEP TRYING (Harsh, 2026-10-02: "actively try ... if it gets blocked", and he
# had rejected "please move me"). With `keep_trying`, MAX_TRIES at one place no
# longer ends in a request and a wait for a person: she says once that she is
# trying another way, the place goes on the avoid list as before, and after
# REST_S the lock is undone again from whatever she has moved since. Bounded
# by the run itself (its time, and the way back's own budget).
REST_S = 3.0
# While asking, free this many consecutive ticks = someone moved her (or it moved).
FREED_TICKS = 10
# Say the request again this often while nobody helps.
ASK_REPEAT_S = 30.0
# The planner steers away from frontiers within this of a place she got stuck.
AVOID_RADIUS_M = 0.5

TURN_W = 0.1        # rad/s: executed rotation above this is turning
DRIVE_V = 0.03      # m/s: executed speed above this is driving


@dataclass
class _Step:
    v: float
    w: float
    dt: float


@dataclass
class _Undo:
    kind: str                     # "turn" | "drive"
    v: float
    w: float
    duration_s: float
    why: str
    started: float = 0.0
    tof_at_start: float = math.inf


@dataclass
class Unstick:
    robot_radius: float
    v_min: float                  # negative: the one executable reverse
    omega_max: float
    tof_freeze_m: float           # the reflex's tof_pivot_min_clearance
    keep_trying: bool = False     # no "please move me": rest REST_S, try again

    state: str = "idle"           # idle | undoing | asking | resting
    history: deque = field(default_factory=lambda: deque(maxlen=200))
    frozen_ticks: int = 0
    free_ticks: int = 0
    tries: int = 0
    spot: tuple[float, float] | None = None
    free_since: float | None = None
    undo: _Undo | None = None
    asked_at: float = -math.inf
    avoid: list = field(default_factory=list)       # [(x, y)] -- the loop reads it
    events: list = field(default_factory=list)      # (key, text) to say -- the loop drains it
    stats: dict = field(default_factory=lambda: {
        "locks": 0, "undos": 0, "undo_turns": 0, "undo_drives": 0,
        "undo_freed": 0, "undo_worse": 0, "undo_rear_stop": 0,
        "no_move_to_undo": 0, "asked": 0, "helped": 0})

    # ------------------------------------------------------------------ feed ---
    def record(self, v_exec: float, w_exec: float, dt: float) -> None:
        """What the wheels actually did this tick. Undo ticks are recorded too;
        the history is cleared when an undo ends, so an undo is never retraced."""
        self.history.append(_Step(float(v_exec), float(w_exec), float(dt)))
        total = 0.0
        for i in range(len(self.history) - 1, -1, -1):
            total += self.history[i].dt
            if total > HISTORY_S:
                for _ in range(i):
                    self.history.popleft()
                break

    # ------------------------------------------------------------------ plan ---
    def _plan(self, rear_min: float) -> _Undo | None:
        """The newest real move, retraced. None when there is nothing safe to undo."""
        steps = list(self.history)
        # Skip the frozen tail: the lock itself moved nothing.
        i = len(steps) - 1
        while i >= 0 and abs(steps[i].v) < DRIVE_V and abs(steps[i].w) < TURN_W:
            i -= 1
        if i < 0:
            return None
        newest = steps[i]
        turning = abs(newest.v) < DRIVE_V
        if turning:
            angle, secs = 0.0, 0.0
            while (i >= 0 and abs(steps[i].v) < DRIVE_V
                   and abs(steps[i].w) >= TURN_W
                   and math.copysign(1, steps[i].w) == math.copysign(1, newest.w)):
                angle += steps[i].w * steps[i].dt
                secs += steps[i].dt
                i -= 1
            if secs < TURN_MIN_S:
                return None
            rate = abs(angle) / secs
            return _Undo("turn", 0.0, -math.copysign(min(rate, self.omega_max), angle),
                         min(secs, UNDO_MAX_S),
                         f"undoing a {math.degrees(abs(angle)):.0f} deg turn "
                         f"({secs:.1f} s the other way)")
        if newest.v < 0:
            # She was already reversing into this. Undoing that means FORWARD, which
            # the ToF reflex forbids and this never issues.
            return None
        dist, secs, wsum = 0.0, 0.0, 0.0
        while i >= 0 and steps[i].v >= DRIVE_V:
            dist += steps[i].v * steps[i].dt
            wsum += steps[i].w * steps[i].dt
            secs += steps[i].dt
            i -= 1
        behind = max(dist, rear_min - self.robot_radius)
        allowed = min(UNDO_MAX_M, behind - COAST_M)
        if allowed < UNDO_MIN_M:
            return None
        speed = abs(self.v_min)
        # Reversing along an arc retraces it: negate both halves of the twist.
        w = -(wsum / secs) if secs > 0 else 0.0
        w = max(-self.omega_max, min(self.omega_max, w))
        return _Undo("drive", -speed, w, min(allowed / speed, UNDO_MAX_S),
                     f"backing {allowed * 100:.0f} cm along the path I came in on")

    # ---------------------------------------------------------------- update ---
    def update(self, *, frozen: bool, tof_min: float | None, rear_min: float,
               pose: tuple[float, float, float] | Any,
               now: float | None = None) -> tuple[float, float, str] | None:
        """Called every tick after the ToF reflex. Returns (v, omega, why) to
        issue instead of the chain's output, or None to leave it alone."""
        now = time.monotonic() if now is None else now
        x, y = float(pose[0]), float(pose[1])
        tof = math.inf if tof_min is None else float(tof_min)

        if self.state == "undoing":
            u = self.undo
            guard = self.robot_radius + REAR_GUARD_EXTRA_M
            if tof >= self.tof_freeze_m + ROOM_MARGIN_M:
                self.stats["undo_freed"] += 1
                return self._end_undo(now, freed=True)
            if tof < u.tof_at_start - WORSE_M:
                self.stats["undo_worse"] += 1
                return self._end_undo(now, freed=False)
            if u.v < 0 and rear_min <= guard:
                self.stats["undo_rear_stop"] += 1
                return self._end_undo(now, freed=False)
            if now - u.started >= u.duration_s:
                return self._end_undo(now, freed=False)
            return (min(u.v, 0.0), u.w, u.why)

        if not frozen:
            self.frozen_ticks = 0
            if self.state == "asking":
                self.free_ticks += 1
                if self.free_ticks >= FREED_TICKS:
                    self.state = "idle"
                    self.tries = 0
                    self.stats["helped"] += 1
                    self.events.append(("unstuck_thanks",
                                        "Thank you. Carrying on exploring."))
                    self.free_since = now
            elif self.state == "idle":
                if self.free_since is None:
                    self.free_since = now
                if now - self.free_since >= FREE_RESET_S:
                    self.tries = 0
            return None

        self.free_ticks = 0
        self.free_since = None
        self.frozen_ticks += 1

        if self.state == "asking":
            if now - self.asked_at >= ASK_REPEAT_S:
                self._ask(now, again=True)
            return None
        if self.state == "resting":
            if now - self.asked_at < REST_S:
                return None
            self.state, self.tries = "idle", 0     # and try again, below

        if self.frozen_ticks < LOCK_TICKS:
            return None

        # A lock. Same place as the last one?
        if self.frozen_ticks == LOCK_TICKS:
            self.stats["locks"] += 1
        if self.spot is None or math.hypot(x - self.spot[0],
                                           y - self.spot[1]) > SAME_SPOT_M:
            self.spot = (x, y)
            self.tries = 0
        if self.tries >= MAX_TRIES:
            self._ask(now)
            return None
        plan = self._plan(rear_min)
        if plan is None:
            self.stats["no_move_to_undo"] += 1
            self._ask(now)
            return None
        # A reverse must also clear the LiDAR guard before it starts.
        if plan.v < 0 and rear_min <= self.robot_radius + REAR_GUARD_EXTRA_M:
            self.stats["undo_rear_stop"] += 1
            self._ask(now)
            return None
        self.tries += 1
        self.stats["undos"] += 1
        self.stats["undo_turns" if plan.kind == "turn" else "undo_drives"] += 1
        plan.started = now
        plan.tof_at_start = tof
        self.undo = plan
        self.state = "undoing"
        if self.tries == 1:
            self.events.append(("unstuck_undo",
                                "I'm boxed in. Undoing my last move."))
        return (min(plan.v, 0.0), plan.w, plan.why)

    # --------------------------------------------------------------- helpers ---
    def _end_undo(self, now: float, *, freed: bool) -> None:
        self.state = "idle"
        self.undo = None
        self.frozen_ticks = 0
        # Never retrace an undo: the next lock plans from moves made after it.
        self.history.clear()
        if freed:
            self.free_since = now
        return None

    def _ask(self, now: float, again: bool = False) -> None:
        if self.keep_trying:
            self.stats["rests"] = self.stats.get("rests", 0) + 1
            if self.spot is not None and self.spot not in self.avoid:
                self.avoid.append(self.spot)
            if self.stats["rests"] == 1 or self.stats["rests"] % 5 == 0:
                self.events.append(("unstuck_retry",
                                    "I'm boxed in. Trying another way."))
            self.state, self.asked_at, self.free_ticks = "resting", now, 0
            return
        if not again:
            self.stats["asked"] += 1
            if self.spot is not None:
                self.avoid.append(self.spot)
        self.state = "asking"
        self.asked_at = now
        self.free_ticks = 0
        self.events.append(("unstuck_help", "I'm stuck. Please move me a little."))

    def avoid_keys(self, cell_m: float = 0.25) -> set:
        """Frontier keys (slam.frontier.Frontier.key) near every stuck spot."""
        keys = set()
        r = int(math.ceil(AVOID_RADIUS_M / cell_m))
        for sx, sy in self.avoid:
            cx, cy = int(round(sx / cell_m)), int(round(sy / cell_m))
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if math.hypot(dx, dy) * cell_m <= AVOID_RADIUS_M:
                        keys.add((cx + dx, cy + dy))
        return keys

    def snapshot(self) -> dict[str, Any]:
        return {"state": self.state, "tries": self.tries,
                "avoid_spots": len(self.avoid), **self.stats}
