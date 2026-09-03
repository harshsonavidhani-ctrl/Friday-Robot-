"""Decide what KIND of situation she is in, then navigate according to it.

WHY THIS EXISTS
===============

Harsh, 2026-09-04: *"make it like it can take decisions and work intelligently
according to it"*, and *"i want like the real robots navigate with confidence"*.

`dwa.py` answers "which arc is drivable right now". Asked fresh every 100 ms, that is
still not confidence -- it is a robot re-deciding its entire life ten times a second.
When two ways round an obstacle score nearly the same, a per-tick argmax will flip
between them on sensor noise alone, and flipping between two good answers is exactly
what a zig-zag IS. MEASURED on 2026-09-04: 515 deg of turning per metre travelled in
the best run, 1799 deg/m in the worst, in rooms she could see over a metre down.

So this layer adds the two things that separate a local planner from navigation:

  1. AN ASSESSMENT. Open room, corridor, cluttered, or blocked are different problems
     and want different behaviour. Driving a corridor like an open room is how you
     scrape a doorway; driving an open room like a corridor is how you crawl.

  2. A COMMITMENT. Once she has chosen a way round something, that choice gets a bonus
     for as long as it stays valid. A rival has to be clearly better -- not
     marginally, not within noise -- to take the decision away. This is the same
     lesson `explorer.py` already learned with `target_hysteresis`, applied to
     steering instead of to goals.

Everything here is ADVISORY over dwa.py, which is itself advisory over the safety
chain. Nothing in this file can make the robot go faster than the caller already
allowed, and nothing in it can override the governor, the reflex or the rear gate.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

# --- assessment thresholds -----------------------------------------------------------
# Front clearance above which the way ahead counts as genuinely open. Chosen against
# `safe_distance` (0.55): below that the governor is already bleeding speed off, so
# calling it "open" would be claiming something the chain below disagrees with.
OPEN_AHEAD_M = 1.10
# Side room below which she is between two things -- a doorway, a hallway, furniture.
CORRIDOR_SIDE_M = 0.90
# Front clearance below which nothing forward is worth planning; hand back to the
# escape/recovery machinery, which owns reverse and pivots and is already tested.
BLOCKED_AHEAD_M = 0.40
# A direction counts as a GAP only if there is this much room to run down it. Set at
# OPEN_AHEAD_M deliberately: "somewhere she fits" and "somewhere she can go" are
# different questions, and answering the first when you meant the second is how a robot
# steers into a pillar. See _widest_gap.
GAP_MIN_RANGE_M = OPEN_AHEAD_M

# --- commitment ----------------------------------------------------------------------
# How much better a rival direction must score before it may steal a live commitment.
# Same shape of rule as explorer.py's target_hysteresis = 0.35 and for the same reason:
# without it, the cheapest option moves as she turns and the decision chases it.
COMMIT_MARGIN = 0.18
# A commitment older than this is re-opened regardless. Commitment is for crossing a
# doorway, not for driving into a wall with conviction.
COMMIT_MAX_S = 3.0
# ...and it is dropped immediately if the committed side stops being drivable at all.

# --- per-situation speed policy ------------------------------------------------------
# Fractions of the caller's allowance, never of v_max: this file cannot raise a cap.
SPEED_OPEN = 1.00
SPEED_CORRIDOR = 0.75      # a doorway is not the place to find out about slew
SPEED_CLUTTERED = 0.55

OPEN = "open"
CORRIDOR = "corridor"
CLUTTERED = "cluttered"
BLOCKED = "blocked"


@dataclass
class Assessment:
    situation: str
    ahead_m: float
    left_m: float
    right_m: float
    widest_gap_bearing: float | None    # [rad] centre of the largest drivable gap
    widest_gap_width: float             # [rad] angular width of it
    why: str


@dataclass
class Decision:
    v: float
    omega: float
    situation: str
    committed: bool
    clearance: float
    why: str
    assessment: Assessment | None = None
    dwa_ms: float = 0.0
    telemetry: dict = field(default_factory=dict)


class Navigator:
    """Situation assessment + commitment, wrapped around a DynamicWindowPlanner."""

    def __init__(self, cfg, planner) -> None:
        self.cfg = cfg
        self.planner = planner            # dwa.DynamicWindowPlanner

        # live commitment
        self._commit_sign = 0             # -1 right, +1 left, 0 none
        self._commit_since = 0.0
        self._commit_situation = ""
        self._last_situation = ""

        # telemetry
        self.assessments: dict[str, int] = {OPEN: 0, CORRIDOR: 0,
                                            CLUTTERED: 0, BLOCKED: 0}
        self.commitments = 0
        self.commit_breaks = 0
        self.handoffs = 0                 # ticks handed back to the old behaviour

    # ------------------------------------------------------------- assessment --
    def assess(self, scan_m: np.ndarray,
               blind_beams: np.ndarray | None = None) -> Assessment:
        """What kind of place is this? Answered from the scan alone.

        Deliberately NOT from the map: the map dead-reckons and has no relocalisation,
        so it can be confidently wrong about where she is. The scan cannot -- it is
        only ever a statement about now.
        """
        n = scan_m.size
        r = np.asarray(scan_m, dtype=np.float64).copy()
        bearing = np.arange(n) * (2.0 * np.pi / n)
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))   # wrap to +-pi

        # A beam that saw nothing is not an obstacle at 5 m; for "how much room is
        # there" it is the most room there can be, so max_range is the right value to
        # keep here (unlike in dwa.py, where it must not become a collision).
        valid = np.isfinite(r)
        if blind_beams is not None and np.size(blind_beams):
            valid[np.asarray(blind_beams, dtype=np.int32)] = False
        min_valid = float(getattr(self.cfg, "lidar_floor", self.cfg.robot_radius))
        valid &= r >= min_valid
        r = np.where(valid, r, self.cfg.lidar_max_range)

        def sector(lo_deg: float, hi_deg: float) -> float:
            lo, hi = math.radians(lo_deg), math.radians(hi_deg)
            m = (bearing >= lo) & (bearing <= hi) & valid
            return float(np.min(r[m])) if m.any() else float(self.cfg.lidar_max_range)

        ahead = sector(-30.0, 30.0)
        left = sector(50.0, 110.0)
        right = sector(-110.0, -50.0)

        gap_b, gap_w = self._widest_gap(r, bearing, valid)

        if ahead <= BLOCKED_AHEAD_M:
            sit, why = BLOCKED, f"ahead {ahead:.2f} m -- nothing forward to plan"
        elif ahead >= OPEN_AHEAD_M and min(left, right) >= CORRIDOR_SIDE_M:
            sit, why = OPEN, f"ahead {ahead:.2f} m, sides {left:.2f}/{right:.2f} m"
        elif min(left, right) < CORRIDOR_SIDE_M and ahead >= OPEN_AHEAD_M:
            sit, why = CORRIDOR, (f"between {left:.2f} m and {right:.2f} m, "
                                  f"{ahead:.2f} m to run")
        else:
            sit, why = CLUTTERED, f"ahead {ahead:.2f} m, sides {left:.2f}/{right:.2f} m"

        self.assessments[sit] = self.assessments.get(sit, 0) + 1
        return Assessment(sit, ahead, left, right, gap_b, gap_w, why)

    def _widest_gap(self, r: np.ndarray, bearing: np.ndarray,
                    valid: np.ndarray) -> tuple[float | None, float]:
        """Centre and width of the widest forward gap she actually fits through.

        THE WIDEST GAP, NOT THE CENTRE OF MASS -- the same rule `friday_freespace.py`
        already asserts for the camera. Two doorways either side of a pillar have their
        centre of mass ON the pillar, and a robot that steers by centroid drives into
        it.
        """
        fwd = (np.abs(bearing) <= math.radians(100.0))
        # A GAP IS SOMEWHERE SHE CAN TRAVEL, NOT MERELY SOMEWHERE SHE FITS. This was
        # `r > 2*(robot_radius + margin)` = 0.52 m, and test_dwa caught it: a pillar at
        # 0.70 m cleared that bar, so the pillar itself was reported as the widest gap
        # and the robot would have steered at it. A direction only counts as open if
        # there is real room to run down it.
        passable = fwd & (r > GAP_MIN_RANGE_M)

        idx = np.where(fwd)[0]
        if idx.size == 0 or not passable[idx].any():
            return None, 0.0
        order = idx[np.argsort(bearing[idx])]
        best_len, best_run, run = 0, [], []
        for i in order:
            if passable[i]:
                run.append(i)
                if len(run) > best_len:
                    best_len, best_run = len(run), list(run)
            else:
                run = []
        if not best_run:
            return None, 0.0
        b = bearing[np.array(best_run)]
        return float(np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b)))), \
            float(b.max() - b.min())

    # -------------------------------------------------------------- commitment --
    def _apply_commitment(self, candidates_omega: float, score_now: float,
                          situation: str) -> tuple[float, bool, str]:
        """Bias toward the direction already chosen, unless a rival clearly beats it."""
        now = time.monotonic()
        sign = 0 if abs(candidates_omega) < 1e-3 else (1 if candidates_omega > 0 else -1)

        if self._commit_sign == 0 or situation != self._commit_situation:
            self._commit_sign, self._commit_since = sign, now
            self._commit_situation = situation
            if sign:
                self.commitments += 1
            return candidates_omega, False, "new commitment"

        if now - self._commit_since > COMMIT_MAX_S:
            self._commit_sign, self._commit_since = sign, now
            self.commit_breaks += 1
            return candidates_omega, False, "commitment expired"

        if sign == self._commit_sign or sign == 0:
            return candidates_omega, True, "holding line"

        # A genuine reversal. It is allowed, but it has to have earned it -- the caller
        # only gets here when the winning arc points the other way, and `score_now` is
        # how much better it was. Under the margin, keep the old direction's arc.
        if score_now >= COMMIT_MARGIN:
            self._commit_sign, self._commit_since = sign, now
            self.commit_breaks += 1
            return candidates_omega, False, f"switched (margin {score_now:.2f})"
        return candidates_omega, True, f"held against rival ({score_now:.2f})"

    def release(self) -> None:
        """Forget the commitment. Called when something else takes the wheel."""
        self._commit_sign = 0
        self._commit_situation = ""

    # ------------------------------------------------------------------ decide --
    def decide(self, scan_m: np.ndarray, v_allow: float, omega_want: float,
               desired_bearing: float | None = None,
               blind_beams: np.ndarray | None = None,
               v_floor: float = 0.15) -> Decision | None:
        """Assess, plan, commit. None means "not mine -- use the existing behaviour".

        The None path is load-bearing: BLOCKED and "no safe arc" both return it, and
        the caller then runs exactly the code it ran before this module existed. This
        module can therefore only ever improve the cases it handles, and cannot break
        the ones it declines.
        """
        a = self.assess(scan_m, blind_beams)

        if a.situation == BLOCKED:
            self.handoffs += 1
            self.release()
            return None

        speed = {OPEN: SPEED_OPEN, CORRIDOR: SPEED_CORRIDOR,
                 CLUTTERED: SPEED_CLUTTERED}[a.situation]

        # In a corridor or clutter, steer for the gap she actually fits through rather
        # than at a target behind a wall. The target still wins in the open, where the
        # gap is meaningless because everything is a gap.
        want = desired_bearing
        if a.situation in (CORRIDOR, CLUTTERED) and a.widest_gap_bearing is not None:
            want = (a.widest_gap_bearing if desired_bearing is None else
                    0.5 * (a.widest_gap_bearing + desired_bearing))

        res = self.planner.choose(scan_m, v_allow * speed, omega_want,
                                  desired_bearing=want, blind_beams=blind_beams,
                                  v_floor=v_floor)
        if res is None:
            self.handoffs += 1
            self.release()
            return None

        omega, committed, commit_why = self._apply_commitment(
            res.omega, res.score, a.situation)

        self._last_situation = a.situation
        return Decision(
            v=res.v,
            omega=omega,
            situation=a.situation,
            committed=committed,
            clearance=res.clearance,
            why=f"{a.situation}: {a.why}; {res.why}; {commit_why}",
            assessment=a,
            dwa_ms=res.ms,
            telemetry={"safe": res.safe, "considered": res.considered,
                       "gap_w_deg": math.degrees(a.widest_gap_width)},
        )
