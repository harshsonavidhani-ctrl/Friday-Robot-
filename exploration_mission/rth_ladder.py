"""Getting back to where the mission started, and knowing she has (File 21
step 8, spec 4.11 / 4.12).

THE LADDER. An attempt is one drive at the start. When an attempt fails,
refuses or times out, the next rung is applied and she tries again:

    0  the first attempt        on this robot: the RL run's OWN way back
    1  wait 5 s                 a person or pet in the way
    2  clear File 14's runtime obstacle layer (never the map)   stale obstacles
    3  slow spin + relocalise   poor localisation     "Checking where I am."
    4  back up 0.25 m, only if the rear +-30 deg is clear by 0.35 m   wedged
    5  retrace the breadcrumbs  no route, but the trail was drivable
                                                       "Retracing my path."
    6  goal = nearest free, reachable cell within tolerance of the start
                                the start spot itself is blocked
                                                       "Getting as close as I can."
    7  HELP                     needs a human

Cheap and harmless first, then localisation, then physical unsticking, then
the known-good path, then a relaxed goal, then a person. BOUNDED ON PURPOSE:
a stalled TT motor drains the battery and heats up. She never gives up
silently -- HELP says how far she is, retries rungs 1-6 every 30 s for up to
20 cycles, and only then parks in HELP_WAIT with one last line.

WHO DRIVES. Rung 0 is the run's way back (it ended 0.31-0.47 m out on the
real robot). Rungs 1-6 go through `engine` -- File 14, which can only have
the wheels once the run has ended and returned the hardware. The engine is an
adapter: with none available there is no ladder to climb, and she goes
straight to HELP rather than count down ten minutes of attempts that cannot
happen.

This module commands nothing itself and never blocks: `tick(now)` is called
at loop rate and starts or polls one engine action at a time. Every motion
the engine makes is File 14's guarded motion; the reflex layer can veto it.

ARRIVAL (4.12) is position only: within arrive_tolerance_m of the start,
continuously, for arrive_dwell_s -- and only moving beyond tolerance +
hysteresis resets that dwell, so pose jitter cannot. Then one quick look: the
LiDAR signature against the start's. A mismatch is NOT a failure (furniture
and people move): it is `verified = false`, logged, shown, never looped on.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from exploration_mission import breadcrumbs, frontiers, home_anchor
from exploration_mission import grid_utils as gu

log = logging.getLogger("friday.exploration_mission.rth")

ATTEMPT_TIMEOUT_S = 180.0       # an engine journey that never ends is a failure
# The look-again rung turns up to a full circle a quarter at a time, with a
# look and a search after each: ~25-40 s on the Pi. At 30 s the ladder used to
# give up on it mid-turn and run its next rungs into "busy" (2026-10-02 22:45).
MANEUVER_TIMEOUT_S = 75.0
ENGINE_BUSY_WAIT_S = 30.0        # a rung waits this long for the engine to free
SPOKEN = {3: "rung3", 5: "rung5", 6: "rung6"}


# --- 4.12 arrival -----------------------------------------------------------------

class ArrivalCheck:
    def __init__(self, tolerance_m: float = 0.35, hysteresis_m: float = 0.10,
                 dwell_s: float = 2.0):
        self.tolerance_m, self.hysteresis_m, self.dwell_s = (
            float(tolerance_m), float(hysteresis_m), float(dwell_s))
        self.since: float | None = None

    def update(self, distance_m: float | None, now: float) -> bool:
        """Feed the distance to the start. True once she has been within
        tolerance for the dwell time. An unknown distance changes nothing."""
        if distance_m is not None:
            if distance_m <= self.tolerance_m:
                if self.since is None:
                    self.since = now
            elif distance_m > self.tolerance_m + self.hysteresis_m:
                self.since = None
            # between the two: jitter -- a dwell already running keeps running
        return self.since is not None and now - self.since >= self.dwell_s


def verify(sweeps: list[dict], start_signature, max_range_m: float = 8.0,
           match_mad_m: float = 0.20, min_overlap: float = 0.60) -> dict:
    """The quick signature at arrival against the start's (spec 4.12)."""
    if not sweeps or start_signature is None:
        return {"verified": False, "why": "no sweep to compare"
                if not sweeps else "no start signature"}
    cur = home_anchor.signature(sweeps, max_range_m)
    ref = np.asarray([float("nan") if v is None else float(v)
                      for v in start_signature], dtype=np.float32)
    s = home_anchor.similarity(cur, ref, min_overlap, match_mad_m)
    return {"verified": bool(s.match), **s.as_dict()}


# --- rung 6: the relaxed goal ------------------------------------------------------

def relaxed_goal(grid: gu.Grid, start_xy, robot_xy, inflate_m: float,
                 tolerance_m: float = 0.35, ring_step_m: float = 0.05
                 ) -> tuple[float, float] | None:
    """The nearest free cell to the start that she can actually reach, within
    tolerance of it -- searched ring by ring outward (0, 5, 10 ... cm), so the
    first hit is the closest. None if every such cell is blocked or cut off."""
    trav = gu.traversable(grid, inflate_m)
    start_cell = frontiers.snap_start(trav, grid.world_to_cell(*robot_xy),
                                      grid.resolution)
    reach = np.isfinite(frontiers.distance_field(trav, start_cell))
    if not reach.any():
        return None
    ix, iy = np.nonzero(reach)
    cx, cy = grid.cell_to_world(ix, iy)
    d = np.hypot(cx - start_xy[0], cy - start_xy[1])
    step = max(ring_step_m, 1e-6)
    ring = 0.0
    while ring <= tolerance_m + 1e-9:
        hit = np.nonzero(d <= ring + step / 2.0)[0]
        if len(hit):
            best = hit[int(np.argmin(d[hit]))]
            return float(cx[best]), float(cy[best])
        ring += step
    return None


# --- 4.11 the ladder ----------------------------------------------------------------

@dataclass
class Attempt:
    rung: int
    result: str                 # arrived | failed | refused | timeout | skipped
    distance_m: float | None
    duration_s: float
    cycle: int = 0

    def as_dict(self) -> dict:
        return {"rung": self.rung, "result": self.result,
                "distance_m": None if self.distance_m is None
                else round(self.distance_m, 2),
                "duration_s": round(self.duration_s, 1), "cycle": self.cycle}


@dataclass
class LadderConfig:
    wait_s: float = 5.0
    backup_m: float = 0.25
    backup_rear_clear_m: float = 0.35
    arrive_tolerance_m: float = 0.35
    ring_step_m: float = 0.05
    help_retry_s: float = 30.0
    help_max_cycles: int = 20
    help_speak_every: int = 3

    @classmethod
    def from_config(cls, cfg) -> "LadderConfig":
        r = cfg["return"]
        return cls(r.wait_s, r.backup_m, r.backup_rear_clear_m, r.arrive_tolerance_m,
                   r.ring_step_m, r.help_retry_s, r.help_max_cycles,
                   r.help_speak_every)


class Ladder:
    """One return to the start. `tick(now)` -> None while working, "arrived",
    or "help_wait" (parked; only a person's "come back" restarts it).

    `engine` (adapters.NavAdapter / FakeNav):
        available() -> bool
        attempt(goal_xy, waypoints=None) -> bool     start a journey
        busy() -> bool      outcome() -> "ARRIVED" | anything else = failed
        clear_runtime_obstacles()      spin_relocalise() -> bool
        back_up(m) -> bool             rear_clear_m() -> float | None
        distance_to(goal_xy) -> float | None         pose() -> (x, y, th) | None
        stop()
    `plan`: {"goal": (x, y), "trail": [...], "grid": gu.Grid | None,
             "inflate_m": ..., "body_radius_m": ..., "obstacles": [...]}
    """

    def __init__(self, engine, cfg: LadderConfig, say: Callable[..., object],
                 plan: dict):
        self.e, self.cfg, self.say, self.plan = engine, cfg, say, plan
        self.goal = tuple(plan["goal"])
        self.attempts: list[Attempt] = []
        self.rung = 0                    # the rung the current attempt follows
        self.cycle = 0                   # HELP cycles done
        self.in_help = False
        self.phase = "external"          # external | wait | maneuver | attempt |
        self._t = 0.0                    #   help_pause | done
        self._maneuver = ""
        self.last_distance: float | None = None
        self.result: str | None = None

    # -- bookkeeping --
    def _distance(self) -> float | None:
        """How far from the start: the engine's view if it has one, else the
        last distance any attempt reported (the run's own, at the least) --
        never a made-up zero in "I'm stuck about N meters"."""
        d = None
        if self.e.available():
            try:
                d = self.e.distance_to(self.goal)
            except Exception:                           # noqa: BLE001
                d = None
        return self.last_distance if d is None else d

    def _record(self, result: str, now: float, distance=None) -> None:
        d = self._distance() if distance is None else distance
        if d is not None:
            self.last_distance = d
        a = Attempt(self.rung, result, d, now - self._t, self.cycle)
        self.attempts.append(a)
        log.info("return attempt: %s", a.as_dict())

    def rungs_used(self) -> list[dict]:
        return [a.as_dict() for a in self.attempts]

    # -- the first attempt is someone else's drive (the run's own way back) --
    def begin(self, now: float) -> None:
        self.phase, self.rung, self._t = "external", 0, now

    def external_result(self, arrived: bool, distance: float | None, now: float
                        ) -> str | None:
        """The run's way back ended. Arrived -> done; else climb."""
        self._record("arrived" if arrived else "failed", now, distance)
        if arrived:
            return self._done("arrived")
        return self._next_rung(now)

    def restart(self, now: float) -> str | None:
        """"Friday, come back" from HELP_WAIT: the whole ladder again, from a
        plain attempt."""
        self.result, self.in_help, self.cycle = None, False, 0
        self.rung, self._t = 0, now
        if not self.e.available():
            self.say("help", d=self._distance() or 0.0)
            return self._done("help_wait")
        self._attempt(now)
        return None

    def _done(self, result: str) -> str:
        self.phase, self.result = "done", result
        return result

    # -- climbing --
    def _next_rung(self, now: float) -> str | None:
        if not self.e.available():
            # No engine to climb with: say so at once, do not count down.
            self.e_stop_quietly()
            self.say("help", d=self._distance() or 0.0)
            return self._done("help_wait")
        self.rung += 1
        self._t = now
        if self.rung >= 7:
            return self._help(now)
        if self.rung in SPOKEN and not self.in_help:
            self.say(SPOKEN[self.rung])
        if self.rung == 1:
            self.phase = "wait"
        elif self.rung == 2:
            self.e.clear_runtime_obstacles()
            self._attempt(now)
        elif self.rung == 3:
            self._start_maneuver("spin", self.e.spin_relocalise(), now)
        elif self.rung == 4:
            rear = self.e.rear_clear_m()
            if rear is None or rear < self.cfg.backup_rear_clear_m:
                self._record("skipped", now)             # blind or too close behind
                return self._next_rung(now)
            self._start_maneuver("backup", self.e.back_up(self.cfg.backup_m), now)
        elif self.rung == 5:
            self._attempt(now, waypoints=self._backtrack())
        elif self.rung == 6:
            goal = self._relaxed()
            if goal is None:
                self._record("skipped", now)
                return self._next_rung(now)
            self._attempt(now, goal=goal)
        return None

    def e_stop_quietly(self) -> None:
        try:
            self.e.stop()
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not stop the engine: %s", exc)

    def _start_maneuver(self, name: str, started: bool, now: float) -> None:
        self._maneuver = name
        if started:
            self.phase = "maneuver"
        else:
            self._attempt(now)               # the manoeuvre refused: just try again

    def _backtrack(self) -> list | None:
        grid, pose = self.plan.get("grid"), self.e.pose()
        trail = self.plan.get("trail") or []
        if grid is None or pose is None or not trail:
            return None
        wp = breadcrumbs.backtrack_path(
            trail, pose, grid, self.plan["inflate_m"], self.plan.get("obstacles"),
            self.plan.get("body_radius_m"))
        return wp or None

    def _relaxed(self) -> tuple[float, float] | None:
        grid, pose = self.plan.get("grid"), self.e.pose()
        if grid is None or pose is None:
            return None
        return relaxed_goal(grid, self.goal, pose[:2], self.plan["inflate_m"],
                            self.cfg.arrive_tolerance_m, self.cfg.ring_step_m)

    def _attempt(self, now: float, goal=None, waypoints=None) -> None:
        self._t = now
        if self.e.busy():
            # The last manoeuvre is still finishing (stopping takes a moment):
            # wait for it rather than call this rung refused.
            self.phase, self._pending = "engine_busy", (goal, waypoints)
            return
        started = self.e.attempt(goal or self.goal, waypoints)
        self.phase = "attempt" if started else "refused"

    def _help(self, now: float) -> str | None:
        self.e_stop_quietly()
        if not self.in_help:
            self.in_help, self.cycle = True, 0
            self.say("help", d=self._distance() or 0.0)
        elif self.cycle >= self.cfg.help_max_cycles:
            return self._done("help_wait")
        elif self.cycle % self.cfg.help_speak_every == 0:
            self.say("help", d=self._distance() or 0.0)
        self.phase, self._t = "help_pause", now
        return None

    # -- the tick --
    def tick(self, now: float) -> str | None:
        if self.phase in ("external", "done"):
            return self.result
        if self.phase == "wait":
            if now - self._t >= self.cfg.wait_s:
                self._attempt(now)
        elif self.phase == "maneuver":
            if not self.e.busy() or now - self._t > MANEUVER_TIMEOUT_S:
                if self.e.busy():
                    self.e_stop_quietly()
                self._attempt(now)
        elif self.phase == "engine_busy":
            if not self.e.busy():
                goal, waypoints = getattr(self, "_pending", (None, None))
                self._attempt(now, goal, waypoints)
            elif now - self._t > ENGINE_BUSY_WAIT_S:
                self._record("refused", now)
                return self._next_rung(now)
        elif self.phase == "refused":
            self._record("refused", now)
            return self._next_rung(now)
        elif self.phase == "attempt":
            if self.e.busy():
                if now - self._t > ATTEMPT_TIMEOUT_S:
                    self.e_stop_quietly()
                    self._record("timeout", now)
                    return self._next_rung(now)
                return None
            if self.e.outcome() == "ARRIVED":
                self._record("arrived", now)
                return self._done("arrived")
            self._record("failed", now)
            return self._next_rung(now)
        elif self.phase == "help_pause":
            if now - self._t >= self.cfg.help_retry_s:
                self.cycle += 1
                if self.cycle > self.cfg.help_max_cycles:
                    return self._done("help_wait")
                self.rung = 0                # rungs 1-6 again
                return self._next_rung(now)
        return None
