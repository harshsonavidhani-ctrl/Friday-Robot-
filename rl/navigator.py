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
#
# MEASURED AND RETUNED, 2026-09-04. At 1.10 m / 0.90 m this robot read OPEN on 6 ticks
# out of 1634 in a real six-minute run, and CLUTTERED on 1324 -- so she sat permanently
# in the most conservative speed band, and the speed authority granted for open running
# was being multiplied away (0.30 x 0.55 = 0.165 m/s, barely over the stiction floor).
# These are tuning numbers taken from that distribution, not derived constants, and the
# room they were measured in is small.
OPEN_AHEAD_M = 0.90
# Side room below which she is between two things -- a doorway, a hallway, furniture.
CORRIDOR_SIDE_M = 0.70
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
# How many CONSECUTIVE ticks the committed direction must have no surviving arc at all
# before that counts as the direction being gone rather than a moment of the turn.
#
# THIS IS THE L-SHAPED CORNER FIX. Harsh, 2026-09-04: "it was still stuttering in a
# L-SHAPED corner/turns it hesitates a lot there". Turning INTO a corner means forward
# is blocked on the inside of the turn -- that is what a corner IS -- so the committed
# direction routinely loses every collision-free arc for a tick or two mid-manoeuvre.
# That was being read as "the other way is infinitely better" (see the margin of inf in
# decide()), so she switched, the mirror image happened on the next tick, and she
# switched back. MEASURED over the 20-minute run of 2026-09-04 21:18: omega changed
# sign 103 times per 1000 driving ticks in the 0.30-0.60 m band, against 22 in open
# space -- the flipping is concentrated exactly where both walls are in range.
#
# 3 ticks is 300 ms at 10 Hz: long enough to outlast scan jitter and the geometry of a
# turn, short enough that a genuine dead end is recognised in a third of a second. She
# never drives a rejected arc while this holds -- see _apply_commitment.
COMMIT_STARVE_TICKS = 3
# ...and it is dropped immediately if the committed side stops being drivable at all.

# --- per-situation speed policy ------------------------------------------------------
# Fractions of the caller's allowance, never of v_max: this file cannot raise a cap.
SPEED_OPEN = 1.00
SPEED_CORRIDOR = 0.75      # a doorway is not the place to find out about slew
SPEED_CLUTTERED = 0.65     # was 0.55: see OPEN_AHEAD_M for why that band was too slow

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
        self._commit_starved = 0          # consecutive ticks with no arc on that side
        self._commit_since = 0.0
        self._commit_situation = ""
        self._last_situation = ""

        # telemetry
        self.assessments: dict[str, int] = {OPEN: 0, CORRIDOR: 0,
                                            CLUTTERED: 0, BLOCKED: 0}
        self.commitments = 0
        # EXPIRY AND REVERSAL ARE NOT THE SAME EVENT and must not share a counter.
        # They were both `commit_breaks`, and the resulting "94 broken vs 28 held" read
        # as constant flip-flopping when COMMIT_MAX_S = 3 s over a 360 s run accounts
        # for ~120 expiries on its own. Only `commit_switches` is evidence of the
        # behaviour this mechanism exists to suppress.
        self.commit_expiries = 0
        self.commit_switches = 0
        # The subset of switches caused by the committed side running out of arcs
        # rather than by a rival scoring better. In a corner these were ALL of them.
        self.commit_starve_switches = 0
        self.handoffs = 0                 # ticks handed back to the old behaviour
        # ...and WHY, because "declined" covers two different situations and lumping
        # them is the same mistake `commit_breaks` made. `blocked` is the assessment
        # refusing to plan at all; `no_arc` is every candidate colliding.
        self.handoff_blocked = 0
        self.handoff_no_arc = 0
        # Ticks where the ToF contributed an obstacle the LiDAR could not see. If this
        # is large while the reflex still fires constantly, the two are STILL
        # disagreeing and the near-field model needs looking at again.
        self.tof_informed = 0
        self.tof_slowed = 0
        self.tof_braked = 0

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

    # ------------------------------------------------------ the near field ------
    def _tof_obstacle_points(self, depth) -> np.ndarray:
        """The near field the LiDAR cannot see, as obstacle points. (N, 2), robot frame.

        WHY THIS EXISTS. The navigator planned against the LiDAR alone, and the LiDAR
        DISCARDS every return below `lidar_floor` (0.22 m, her own body radius) and is
        masked across the chassis wedge entirely. The ToF reflex stops at 0.30 m and
        brakes from 0.60 m. So there was a band right in front of the bumper that one
        component could see and the other planned through: she would commit to an arc
        into something only the ToF knew about, drive at it, and be slammed to a halt.
        MEASURED 2026-09-04: navigator engaged on 94% of ticks while the reflex fired on
        59% of them -- two components steering the same robot from different worlds.
        Ending every run wedged is what that looks like from across the room.

        NO FIELD OF VIEW IS ASSUMED, because this project has never measured one and
        `config.py` has no constant for it. Deriving it from the floor profile was tried
        and rejected: the equal-angle flat-floor fit put the sensor 12 mm above the
        ground with 13 mm residuals, which is not a geometry, it is a coincidence.

        Instead the return is modelled as what the reflex already treats it as -- SOMETHING
        ACROSS HER PATH AT THIS RANGE -- laid out over the width of the corridor her own
        body sweeps. `corridor_half` is measured and already used by the governor, so the
        model introduces no new unknown. It is deliberately conservative: it can make her
        turn away from something slightly narrower than her corridor, and it cannot fail
        to notice something wider.

        The k-th smallest is `tof_min_valid_zones`, the SAME quorum the reflex uses, so
        the two agree about what an obstacle is. Sharing that rule is the entire point:
        a planner and a veto that disagree produce exactly the fight seen above.
        """
        pts, _ = self._tof_constraint(depth)
        return pts

    def _tof_constraint(self, depth) -> tuple[np.ndarray, float]:
        """(barrier points, speed scale) from the near field.

        A BARRIER AT EVERY RANGE THE ToF REPORTS, plus the reflex's own proportional
        speed law on top. Both halves, always.

        THIS WAS TRIED THE OTHER WAY AND IT WAS THE WORST REGRESSION OF THE PROJECT.
        Restricting the barrier to inside `tof_stop_distance` -- on the reasoning that
        "a planner must not be stricter than the emergency layer it feeds" -- was
        measured back to back on hardware:

            |                     | barrier at every range | only inside stop |
            | ground driven       | 16.1 m^2               | 5.6 m^2          |
            | ToF reflex firing   | 30%                    | 64%              |
            | navigator handbacks | 49%                    | 75%              |

        THE REASONING WAS WRONG BECAUSE THE REFLEX ONLY BRAKES BECAUSE IT CANNOT
        STEER. It gets a single k-th-smallest range with no bearing attached, so the
        only safe thing it can do with "something at 0.45 m" is bleed speed off. The
        planner has the whole arc set, so the same fact means "go round it" -- and the
        barrier is the ONLY way that fact reaches the arc search. Delete it and she
        drives straight at near obstacles, slower and slower, until she is inside
        0.30 m and hands the tick to recovery. Matching the reflex's thresholds threw
        away the one advantage the planner has over it.

        Nor is a barrier here "stricter" than the reflex: it deletes the arcs that run
        INTO the thing while leaving every arc that goes AROUND it, which is a
        different operation from capping speed and is not comparable on a scale of
        conservatism. A handback only happens when EVERY arc is gone.

        The barrier stays FOV-free: a full-corridor wall at that range, because no ToF
        field of view has ever been measured on this robot (see the module note).
        """
        empty = np.zeros((0, 2), dtype=np.float32)
        if depth is None:
            return empty, 1.0
        d = np.asarray(depth, dtype=np.float64).reshape(-1)
        valid = d[np.isfinite(d) & (d > 0.0) & (d < self.cfg.tof_max_range)]
        k = int(getattr(self.cfg, "tof_min_valid_zones", 3))
        if valid.size < k:
            return empty, 1.0                 # too few zones agree: speckle, not a thing
        rng = float(np.sort(valid)[k - 1])

        # Speed scale: the reflex's own proportional law, so the two never disagree
        # about how fast is acceptable at this range.
        slow = float(self.cfg.tof_slow_distance)
        stop = float(self.cfg.tof_stop_distance)
        scale = 1.0
        if rng < slow:
            scale = max(float((rng - stop) / max(slow - stop, 1e-6)), 0.0)

        half = float(self.cfg.corridor_half)
        ys = np.linspace(-half, half, 5)
        pts = np.stack([np.full_like(ys, rng), ys], axis=1).astype(np.float32)
        return pts, scale

    # -------------------------------------------------------------- commitment --
    def _apply_commitment(self, candidates_omega: float, margin: float,
                          situation: str,
                          starved: bool = False) -> tuple[float, bool, str]:
        """Bias toward the direction already chosen, unless a rival clearly beats it.

        `margin` is how much better the winning arc is THAN THE BEST ARC IN THE
        COMMITTED DIRECTION -- a difference, not a score.

        MEASURED, 2026-09-04: this was being passed the winner's ABSOLUTE score, which
        in production runs about 1.5 against a COMMIT_MARGIN of 0.18, so "has the rival
        earned it" was true on every single reversal and the commitment could never
        hold. Two hardware runs reported 121/56 and 94/28 broken-to-held because of it.
        A threshold is only a threshold if the quantity compared against it is on the
        same scale.
        """
        now = time.monotonic()
        sign = 0 if abs(candidates_omega) < 1e-3 else (1 if candidates_omega > 0 else -1)

        # A COMMITMENT DOES NOT BELONG TO A SITUATION LABEL. It used to be released
        # whenever `situation` changed, and MEASURED on hardware that destroyed it: 121
        # commitments broken against 56 held over a 6-minute run, because the label
        # flickers between CORRIDOR and CLUTTERED as she moves and the boundary between
        # them is a threshold on a noisy range. Releasing on a flickering label is the
        # same failure as explorer.py's target chasing the cheapest frontier -- the
        # thing meant to provide steadiness was itself following the noise.
        #
        # What may end a commitment is now only: it has run long enough (COMMIT_MAX_S),
        # the situation became BLOCKED (release() is called for that on the way out), or
        # a rival genuinely beat it by COMMIT_MARGIN. A label change is not evidence.
        if self._commit_sign == 0:
            self._commit_sign, self._commit_since = sign, now
            self._commit_situation = situation
            if sign:
                self.commitments += 1
            return candidates_omega, False, "new commitment"
        self._commit_situation = situation

        if now - self._commit_since > COMMIT_MAX_S:
            self._commit_sign, self._commit_since = sign, now
            self.commit_expiries += 1
            return candidates_omega, False, "commitment expired"

        if sign == self._commit_sign or sign == 0:
            self._commit_starved = 0
            return candidates_omega, True, "holding line"

        # THE COMMITTED DIRECTION HAS NO SURVIVING ARC AT ALL. That is not evidence
        # that the other way is better; going INTO a corner blocks the inside of the
        # turn, which is what a corner is. Keep the commitment for a few ticks and let
        # it come back.
        #
        # SHE STILL DRIVES THE SAFE ARC THIS TICK -- `committed=False` is returned, so
        # decide() uses the planner's winner and never the rejected side. All that is
        # held is which way she MEANS to go, so that when the inside of the turn opens
        # up again she resumes it instead of having flipped away.
        if starved:
            self._commit_starved += 1
            if self._commit_starved < COMMIT_STARVE_TICKS:
                return (candidates_omega, False,
                        f"committed side starved {self._commit_starved}/"
                        f"{COMMIT_STARVE_TICKS}, holding the intent")
            self._commit_sign, self._commit_since = sign, now
            self._commit_starved = 0
            self.commit_switches += 1
            self.commit_starve_switches += 1
            return candidates_omega, False, "switched (committed side genuinely gone)"

        # A genuine reversal. It is allowed, but it has to have earned it by
        # COMMIT_MARGIN over the best arc still available in the committed direction.
        self._commit_starved = 0
        if margin >= COMMIT_MARGIN:
            self._commit_sign, self._commit_since = sign, now
            self.commit_switches += 1
            return candidates_omega, False, f"switched (margin {margin:.2f})"
        return candidates_omega, True, f"held against rival ({margin:.2f})"

    def release(self) -> None:
        """Forget the commitment. Called when something else takes the wheel."""
        self._commit_sign = 0
        self._commit_starved = 0
        self._commit_situation = ""

    # ------------------------------------------------------------------ decide --
    def decide(self, scan_m: np.ndarray, v_allow: float, omega_want: float,
               desired_bearing: float | None = None,
               blind_beams: np.ndarray | None = None,
               v_floor: float = 0.15,
               tof_depth=None) -> Decision | None:
        """Assess, plan, commit. None means "not mine -- use the existing behaviour".

        The None path is load-bearing: BLOCKED and "no safe arc" both return it, and
        the caller then runs exactly the code it ran before this module existed. This
        module can therefore only ever improve the cases it handles, and cannot break
        the ones it declines.
        """
        a = self.assess(scan_m, blind_beams)

        if a.situation == BLOCKED:
            self.handoffs += 1
            self.handoff_blocked += 1
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

        near, tof_scale = self._tof_constraint(tof_depth)
        # NOT mutually exclusive any more: a barrier and a speed cut are two separate
        # consequences of the same reading and both apply from tof_slow_distance in.
        # `tof_braked` is the subset already inside the reflex's stop distance, which
        # is where a barrier and an emergency stop agree; the gap between the two
        # counts is the band where the barrier is doing work the reflex cannot.
        if near.shape[0]:
            self.tof_informed += 1
            if tof_scale <= 0.0:
                self.tof_braked += 1
        if tof_scale < 1.0:
            self.tof_slowed += 1
        # The ToF's proportional band SLOWS her rather than deleting her options -- the
        # same thing the reflex does with it, one stage later. Floored at the drivetrain
        # floor: a cap under the stiction floor is a stop wearing a speed's clothing,
        # and it would hand back a tick she could have driven slowly.
        allow = v_allow * speed
        if tof_scale < 1.0:
            allow = max(allow * tof_scale, min(v_floor, allow))
        res = self.planner.choose(scan_m, allow, omega_want,
                                  desired_bearing=want, blind_beams=blind_beams,
                                  v_floor=v_floor, extra_obstacles=near)
        if res is None:
            self.handoffs += 1
            self.handoff_no_arc += 1
            self.release()
            return None

        # The margin is measured against the best arc STILL AVAILABLE in the committed
        # direction. If that direction has no surviving arc at all there is nothing to
        # hold on to, and the commitment must yield rather than steer her into the one
        # place the collision check just rejected.
        held_sign = self._commit_sign
        rival = res.alternatives.get(held_sign) if held_sign else None
        # STARVED is "the committed direction has no collision-free arc this tick",
        # which is a different fact from "the rival scores better" and used to be
        # collapsed into the same infinite margin. See COMMIT_STARVE_TICKS.
        starved = held_sign != 0 and rival is None
        margin = (res.score - rival[0]) if rival else float("inf")

        omega, committed, commit_why = self._apply_commitment(
            res.omega, margin, a.situation, starved=starved)

        v = res.v
        if committed and rival is not None:
            # HOLDING MEANS DRIVING THE COMMITTED DIRECTION'S ARC. This used to return
            # the rival's omega even while reporting "held", so holding and switching
            # produced identical commands and the whole mechanism was inert.
            _, v, omega = rival

        self._last_situation = a.situation
        return Decision(
            v=v,
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
