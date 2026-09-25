"""nav_follower.py — file 14 steps 4-5: driving the path, and knowing you arrived.

    planned waypoints ──> look-ahead ──> heading error ──> ONE GUARDED MOVE
              ^                                                   |
              └────────── pose re-measured from the LiDAR ─────────┘

⚠ TWO PLACES WHERE THIS DELIBERATELY DOES NOT DO WHAT THE SPEC ASKED, both
flagged when this was built rather than discovered later.

1. THE SPEC SAID "correct via differential drive through motor.py's existing
   interface". THIS DOES NOT TOUCH motor.py. Every command goes out through
   `movement.move()` -> `safety.guarded_move()`, the same door
   `slam/navigate.py` and every MCP tool uses. Driving the motors directly
   would skip the look-before-you-move gate, the forward clearance check, the
   100 ms re-check while moving and the duration cap -- the four things that
   keep an autonomous run from being a collision. `slam/navigate.py` says it
   plainly: "There is no fast path here and there must never be one." A
   follower that owns the wheels is exactly that fast path.

2. TRUE PURE PURSUIT COMMANDS A CURVATURE, AND THIS CHASSIS CANNOT BE HANDED
   ONE. The guarded interface offers four words -- forward, backward, left,
   right -- and left/right pivot in place; there is no arc primitive behind
   it. So the pure-pursuit GEOMETRY is real and unmodified (look-ahead point,
   heading error, curvature), and what changes is the actuation: a heading
   error larger than the dead band is spent as a pivot, and anything smaller
   is spent as a short forward step. The curvature is not thrown away -- it
   shortens the step, so a tight arc is driven as several short straight
   chords rather than one long one that leaves the path.

WHY EVERY TICK COSTS A LIDAR SWEEP. There are no wheel encoders. The only
thing on this robot that knows where it is is the scan matcher, so a "control
tick" here is: sweep, match, decide, one short guarded move. That is roughly
one tick per second, not the 20 Hz a pure-pursuit controller normally runs at,
and it is why the steps are short and the look-ahead is close. Nothing is
integrated between ticks: `mapper.pose` after the match IS the pose, and if
the match fails (`pose_trusted` False) this STOPS rather than carrying on from
odometry. A robot navigating on dead reckoning with no encoders is a robot
driving somewhere nobody asked for -- see [[friday-map-relocalisation-ratchet]].

⚠ THE ROBOT IS NOT CALIBRATED. `state/calibration.json` reads
forward_cm_per_s = 0 and turn_deg_per_s = 0 today, which is why
`movement.move_distance()` and `turn_degrees()` both refuse outright, and why
`go_to` cannot currently drive a single leg. This follower therefore commands
in SECONDS through `movement.move()`, which is guarded but not
calibration-gated, and learns the two rates from the pose it measures anyway
(`RateModel`). That is not a way around the refusal: the refusal exists
because an OPEN-LOOP distance would be a guess, and this loop is closed. The
report says `calibrated: false` when it ran on learned rates so nobody reads
its distances as measured ones.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

log = logging.getLogger("friday.nav_follower")

# --- pure pursuit ----------------------------------------------------------

# How far ahead on the path to aim. Short, because the tick rate is set by the
# LiDAR and not by a controller: aiming two metres ahead at one decision per
# second means driving a metre with the steering already stale. A shade over
# two body radii is the distance at which the pivot-then-step actuation still
# tracks the path rather than cutting its corners.
LOOKAHEAD_M = 0.45
LOOKAHEAD_MIN_M = 0.25

# Below this the heading error is not worth a pivot: this chassis fails to
# execute 22% of its turns ([[friday-drivetrain-execution-floors]]), so a
# needless small turn is not free, it is a coin flip that can end the run.
TURN_DEADBAND_DEG = config.MISSION_MIN_TURN_DEG          # 10.0

# One forward step, capped where the mission code caps it. Short steps are
# what make a one-hertz loop safe.
STEP_MAX_M = config.MISSION_MAX_LEG_M                    # 0.25
STEP_MIN_M = 0.04

# --- arrival (step 5) ------------------------------------------------------

# NOT a round number, on purpose. The goal is a cell, so its position is only
# known to within that cell's diagonal; the body has a radius; and the scan
# matcher is worth a few centimetres. Arriving means the footprint covers the
# goal cell, which is the sum of the first two:
ARRIVE_TOL_M = config.NAV_ROBOT_RADIUS_M + config.SLAM_CELL_M * math.sqrt(2.0)

# What "close enough" degrades to when it cannot settle. This is
# config.GOTO_ARRIVE_M, the number `go_to` already accepts as arrival, so the
# loose case here is no looser than what ships today.
ARRIVE_LOOSE_M = config.GOTO_ARRIVE_M                    # 0.45

# One sample is a coincidence: the matcher can put her inside the tolerance
# for one sweep and 12 cm outside it on the next. Roughly the last second of
# pose updates, and at ~1 Hz that is two of them.
ARRIVE_DWELL_S = 1.0
ARRIVE_MIN_SAMPLES = 2

# Inside this radius the look-ahead point is behind the goal and pure pursuit
# has nothing left to chase, so it hands over to a proportional controller
# aimed straight at the goal.
FINAL_APPROACH_M = 0.60

# How long to keep trying for the tight tolerance before accepting the loose
# one. Hunting a 28 cm target with a ±20% drivetrain forever is worse than
# stopping 40 cm away and saying so.
SETTLE_EXTRA_S = 25.0

# --- rates -----------------------------------------------------------------

# Used ONLY while calibration reads zero, and corrected from measured pose
# within the first few ticks. Deliberately slow: an underestimate makes the
# first steps too short, which costs a tick. An overestimate drives too far.
DEFAULT_FORWARD_CM_S = 12.0
DEFAULT_TURN_DEG_S = 45.0

# Sanity band for a learned rate. A stalled wheel measures as "zero cm per
# second", and without this the follower would respond by commanding
# ever-longer moves into whatever it is stuck against.
RATE_MIN_CM_S, RATE_MAX_CM_S = 3.0, 60.0
RATE_MIN_DEG_S, RATE_MAX_DEG_S = 10.0, 180.0

# A measurement only teaches the model something if it is clearly above the
# matcher's own noise.
LEARN_MIN_M = 0.05
LEARN_MIN_DEG = 6.0
LEARN_ALPHA = 0.3

# ⚠ A STALL CANNOT BE JUDGED ONE MOVE AT A TIME, and the first version of this
# tried to. The test caught it: a chassis with its wheels off the ground
# measured 2-3 cm of "motion" per move, because that is the scan matcher's own
# noise, and the per-sample floor kept resetting. The matcher is worth ~7 cm
# over 5.4 m (slam/grid.py) and a legitimate short step is only 4 cm, so there
# is NO per-sample threshold that separates a stalled robot from a slow one.
#
# What does separate them is the RATIO over several moves: noise does not
# accumulate with the distance commanded, and real motion does. Three moves
# that between them delivered less than a quarter of what they asked for is a
# drivetrain that is not driving. Near the goal the commanded steps get small
# enough that even this blurs, and there the budgets are the backstop.
STALL_TICKS = 3
STALL_RATIO = 0.25

# ⚠ WHAT A MOVE ACTUALLY DID IS A QUESTION FOR THE MAP, NOT FOR THE PRIOR.
# `Mapper.update` searches a window sized to the odometry it is told about:
# 20% of the claimed motion, never less than 4 cm / 3 degrees. That has two
# failure modes, and this module had the first one:
#
#   * Tell it NOTHING after a 30 cm step and it searches 4 cm around where she
#     WAS. The true pose is outside the window, so she is either placed where
#     she started (a false stall) or not placed at all ("I do not know where I
#     am"). The follower passed no prior after any move until 2026-09-23; the
#     test double ignores priors, so no suite could see it.
#   * Tell it the COMMANDED motion and it searches 6 cm around where she would
#     be IF the wheels did their job. A wedged robot is then "measured" as
#     having moved, and a stall -- step 4's whole subject -- becomes invisible.
#
# So both are asked. The sweep is matched at "moved as commanded", "moved
# half" and "did not move", each within the window the tracker itself would
# use, then each answer is judged on the WHOLE scan (see motion_fraction for
# why not the tracker's own score), and the best explanation is the prior the
# mapper gets. Ties go to the commanded motion, which is what the tracker would
# have assumed anyway. About 0.3 s per move on the Pi; a tick already costs a
# whole LiDAR sweep.
MOTION_HYPOTHESES = (1.0, 0.5, 0.0)

# How long a guarded move really ran, from its own words: safety.MoveResult
# says "right STOPPED after 0.2s of 1.2s — stop was called".
_RAN_FOR = re.compile(r"STOPPED after ([0-9.]+)s of")


def _wrap(radians: float) -> float:
    return (radians + math.pi) % (2.0 * math.pi) - math.pi


def motion_fraction(mapper, scan, moved_m: float, turned_rad: float) -> float:
    """How much of a commanded move the map says happened: 1.0, 0.5 or 0.0.

    Scores `scan` against the map at each of MOTION_HYPOTHESES, using the
    tracker's own matcher and the window the tracker would have used for that
    much motion, and returns the best. READ ONLY: it never calls
    `mapper.update`, so asking costs the map nothing.

    Anything that is not a real `slam.mapper.Mapper` -- a test double without
    a likelihood field, a map too thin to match against -- gets 1.0, which is
    exactly what the mapper would have assumed without this.
    """
    field_of = getattr(mapper, "_map_field", None)
    if field_of is None:
        return 1.0
    try:
        import copy

        from slam.core import match_pose, points_from_scan, transform

        field = field_of()
        if field is None:
            return 1.0
        local = points_from_scan(scan, forward_deg=mapper.forward_deg)
        if len(local) < config.SLAM_MATCH_MIN_BEAMS:
            return 1.0
        # ⚠ COMPARE WITH EVERY BEAM IN THE DENOMINATOR. The tracker's score
        # divides by the beams that land on mapped cells, which is right for
        # refining one pose and wrong for choosing between several: the
        # hypothesis that points more of the scan at unmapped floor gets a
        # smaller denominator and a flattering score (seen above 1.0).
        # Measured on the real map, 60 poses: tracker score picked right 23/60
        # for a full step and 0/60 for a half step; this picks 60/60 for
        # full, half, stalled, turned and not-turned. A shallow copy, so the
        # mapper's cached field keeps its mask.
        fair = copy.copy(field)
        fair.known = None
        x, y, theta = (float(v) for v in mapper.pose)
        best, best_score = 1.0, -1.0
        for fraction in MOTION_HYPOTHESES:
            m, r = moved_m * fraction, turned_rad * fraction
            mid = theta + r / 2.0              # Mapper._predict's convention
            prior = (x + m * math.cos(mid), y + m * math.sin(mid),
                     _wrap(theta + r))
            # Refined inside the window the tracker would give that much
            # motion, then judged on the whole scan.
            pose, _score, _sharp = match_pose(
                field, local, prior,
                prior_m=max(config.SLAM_ODOM_ERROR * abs(m),
                            config.SLAM_POSE_PRIOR_FLOOR_M),
                prior_deg=max(config.SLAM_ODOM_ERROR * abs(math.degrees(r)),
                              config.SLAM_POSE_PRIOR_FLOOR_DEG))
            score = fair.score(transform(local, pose[0], pose[1], pose[2]))
            if score > best_score + 1e-6:
                best, best_score = fraction, float(score)
        return best
    except Exception as exc:                     # noqa: BLE001
        log.warning("could not weigh what the move did: %s", exc)
        return 1.0


def closest_index(path: np.ndarray, xy) -> int:
    """Index of the path segment the robot is nearest to.

    Searched over segments rather than waypoints: after smoothing the
    waypoints can be metres apart, and the nearest WAYPOINT to a robot halfway
    along a leg is regularly the one behind it.
    """
    if len(path) < 2:
        return 0
    a, b = path[:-1], path[1:]
    seg = b - a
    length2 = np.einsum("ij,ij->i", seg, seg)
    length2[length2 == 0.0] = 1e-12
    t = np.clip(np.einsum("ij,ij->i", np.asarray(xy) - a, seg) / length2, 0.0, 1.0)
    nearest = a + t[:, None] * seg
    return int(np.argmin(np.hypot(nearest[:, 0] - xy[0], nearest[:, 1] - xy[1])))


def lookahead_point(path: np.ndarray, xy, radius: float
                    ) -> tuple[np.ndarray, int]:
    """The point `radius` ahead along the path. Returns (point, segment index).

    Walks FORWARD from the nearest segment and takes the furthest intersection
    of the look-ahead circle with the path, so a route that doubles back past
    the robot cannot pull the target backwards. When the whole remaining path
    is inside the circle -- which is the normal state near the end -- the
    target is the final waypoint, and the final waypoint is the goal.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 0:
        return np.asarray(xy, dtype=np.float64), 0
    if len(path) == 1:
        return path[0], 0

    start = closest_index(path, xy)
    cx, cy = float(xy[0]), float(xy[1])
    best: tuple[np.ndarray, int] | None = None

    for i in range(start, len(path) - 1):
        ax, ay = path[i]
        dx, dy = path[i + 1] - path[i]
        fx, fy = ax - cx, ay - cy
        a = dx * dx + dy * dy
        if a <= 1e-12:
            continue
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            continue
        root = math.sqrt(disc)
        # BOTH roots are on the path and inside the circle's rim; the nearer
        # one is BEHIND the robot on this segment. Taking it is how a route
        # that doubles back drags the target backwards and the robot turns
        # round to chase where it came from. Always the furthest forward.
        forward = [t for t in ((-b + root) / (2.0 * a), (-b - root) / (2.0 * a))
                   if 0.0 <= t <= 1.0]
        if forward:
            t = max(forward)
            best = (np.array([ax + t * dx, ay + t * dy]), i)
    if best is not None:
        return best
    return path[-1], len(path) - 2


def heading_error(pose, target) -> float:
    """Radians to turn to face `target` from `pose`. Positive is left."""
    return _wrap(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])


def curvature(pose, target) -> float:
    """Pure pursuit's 1/R for the arc from pose through target.

    Kept even though this chassis cannot be handed a curvature, because it is
    the honest measure of how sharply the path is turning here, and the step
    length is scaled by it: a tight arc driven as one long straight chord
    leaves the path, and the same arc driven as three short ones does not.
    """
    dist = math.hypot(target[0] - pose[0], target[1] - pose[1])
    if dist < 1e-6:
        return 0.0
    return 2.0 * math.sin(heading_error(pose, target)) / dist


@dataclass
class RateModel:
    """How far a second of driving actually goes, learned while driving.

    Seeded from `state/calibration.json` when somebody has measured it, and
    from a deliberately slow default when nobody has. Either way each
    commanded move is checked against the pose that comes back, so the model
    converges on what the robot is doing today -- on this carpet, at this
    battery voltage, which is not what it was doing when it was calibrated.
    """

    forward_cm_s: float = DEFAULT_FORWARD_CM_S
    turn_deg_s: float = DEFAULT_TURN_DEG_S
    calibrated: bool = False
    samples: int = 0

    @classmethod
    def from_calibration(cls) -> "RateModel":
        cal = config.load_calibration()
        fwd = float(cal.get("forward_cm_per_s") or 0.0)
        turn = float(cal.get("turn_deg_per_s") or 0.0)
        if fwd > 0.0 and turn > 0.0:
            return cls(fwd, turn, calibrated=True)
        log.warning("uncalibrated (forward_cm_per_s=%.1f turn_deg_per_s=%.1f); "
                    "commanding in seconds and learning from pose", fwd, turn)
        return cls()

    def seconds_for(self, metres: float) -> float:
        return abs(metres) * 100.0 / max(self.forward_cm_s, 1e-3)

    def seconds_for_turn(self, degrees: float) -> float:
        return abs(degrees) / max(self.turn_deg_s, 1e-3)

    def observe_drive(self, seconds: float, metres: float) -> None:
        if seconds <= 0.0 or metres < LEARN_MIN_M:
            return
        rate = metres * 100.0 / seconds
        self.forward_cm_s = float(np.clip(
            (1 - LEARN_ALPHA) * self.forward_cm_s + LEARN_ALPHA * rate,
            RATE_MIN_CM_S, RATE_MAX_CM_S))
        self.samples += 1

    def observe_turn(self, seconds: float, degrees: float) -> None:
        if seconds <= 0.0 or abs(degrees) < LEARN_MIN_DEG:
            return
        rate = abs(degrees) / seconds
        self.turn_deg_s = float(np.clip(
            (1 - LEARN_ALPHA) * self.turn_deg_s + LEARN_ALPHA * rate,
            RATE_MIN_DEG_S, RATE_MAX_DEG_S))
        self.samples += 1


@dataclass
class Arrival:
    """Why the follower does or does not believe it has arrived."""

    arrived: bool = False
    distance_m: float = float("inf")
    tolerance_m: float = ARRIVE_TOL_M
    dwell_s: float = 0.0
    samples: int = 0
    relaxed: bool = False
    why: str = "not yet"


class ArrivalMonitor:
    """Step 5: arrival is a state that has held, not a distance that ticked.

    Three conditions, all of which must be true at once:

      * the pose is inside a tolerance sized from the map and the body, not
        picked for being a round number;
      * it has STAYED there for about a second of pose updates -- one sample
        is the scan matcher having a good sweep, not the robot being there;
      * the controller is not currently asking for motion. With no encoders
        there is no measured velocity to test, so this is COMMANDED velocity,
        which is the honest version of the check: it says the controller has
        run out of corrections to make, not that the wheels have stopped.

    And one release valve: after `settle_extra_s` inside the final approach
    the tolerance relaxes to `loose_m`, because hunting a 28 cm target with a
    drivetrain worth +/-20% does not converge, it just uses the budget up.
    """

    def __init__(self, goal_xy, cell_m: float | None = None,
                 tolerance_m: float | None = None,
                 loose_m: float = ARRIVE_LOOSE_M,
                 dwell_s: float = ARRIVE_DWELL_S,
                 min_samples: int = ARRIVE_MIN_SAMPLES,
                 settle_extra_s: float = SETTLE_EXTRA_S) -> None:
        self.goal = (float(goal_xy[0]), float(goal_xy[1]))
        cell = config.SLAM_CELL_M if cell_m is None else float(cell_m)
        self.tolerance = (config.NAV_ROBOT_RADIUS_M + cell * math.sqrt(2.0)
                          if tolerance_m is None else float(tolerance_m))
        self.loose = max(float(loose_m), self.tolerance)
        self.dwell_s = float(dwell_s)
        self.min_samples = int(min_samples)
        self.settle_extra_s = float(settle_extra_s)
        self._inside_since: float | None = None
        self._samples = 0
        self._approach_since: float | None = None

    def active_tolerance(self, now: float) -> tuple[float, bool]:
        if (self._approach_since is not None
                and now - self._approach_since >= self.settle_extra_s):
            return self.loose, True
        return self.tolerance, False

    def update(self, pose, commanded_speed: float, now: float | None = None
               ) -> Arrival:
        now = time.monotonic() if now is None else now
        dist = math.hypot(pose[0] - self.goal[0], pose[1] - self.goal[1])

        if dist <= FINAL_APPROACH_M:
            if self._approach_since is None:
                self._approach_since = now
        else:
            self._approach_since = None

        tolerance, relaxed = self.active_tolerance(now)
        out = Arrival(distance_m=dist, tolerance_m=tolerance, relaxed=relaxed)

        if dist > tolerance:
            self._inside_since, self._samples = None, 0
            out.why = f"{dist:.2f} m away, tolerance {tolerance:.2f} m"
            return out

        if self._inside_since is None:
            self._inside_since = now
        self._samples += 1
        out.dwell_s = now - self._inside_since
        out.samples = self._samples

        if commanded_speed > STEP_MIN_M:
            out.why = "in tolerance, but still correcting"
            return out
        if out.samples < self.min_samples or out.dwell_s < self.dwell_s:
            out.why = (f"in tolerance for {out.dwell_s:.1f} s / "
                       f"{out.samples} samples — holding")
            return out

        out.arrived = True
        out.why = (f"within {tolerance:.2f} m for {out.dwell_s:.1f} s "
                   f"across {out.samples} pose updates, not correcting"
                   + (" (relaxed tolerance)" if relaxed else ""))
        return out


@dataclass
class Executed:
    """What one guarded move actually did, as opposed to what it was told."""

    moved_m: float = 0.0
    turned_deg: float = 0.0
    refused: bool = False
    said: str = ""


@dataclass
class FollowReport:
    arrived: bool = False
    why: str = "budget"
    ticks: int = 0
    driven_m: float = 0.0
    turned_deg: float = 0.0
    seconds: float = 0.0
    distance_m: float = float("inf")
    tolerance_m: float = ARRIVE_TOL_M
    relaxed: bool = False
    calibrated: bool = False
    final_approach_ticks: int = 0
    pose: list[float] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["driven_m"] = round(self.driven_m, 2)
        d["turned_deg"] = round(self.turned_deg, 1)
        d["seconds"] = round(self.seconds, 1)
        d["distance_m"] = round(self.distance_m, 2)
        d["tolerance_m"] = round(self.tolerance_m, 2)
        return d


class Follower:
    """One "drive this path" run. Never raises; always reports why it stopped.

    `robot` is the same six-call object `slam/navigate.py` and
    `mcp_server/tools/explore.py` already use (`_Robot`), plus one addition:
    `drive_seconds(direction, seconds)`, which must route through
    `movement.move()` so the guard applies. `mapper` is a `slam.mapper.Mapper`
    whose pose is already established in the map's frame -- exactly what
    `places_tools._placed_mapper()` returns, and this refuses to start without
    it for the same reason that function refuses: a follower in the wrong
    frame drives confidently to the wrong address.
    """

    def __init__(self, robot, rates: RateModel | None = None) -> None:
        self.robot = robot
        self.rates = rates or RateModel.from_calibration()
        self.events: list[dict] = []
        self.last_fraction = 1.0
        self.last_match = ""
        # Whether the pose was verified by the last sweep (see _sweep_into).
        self._held_trust = False

    async def _sweep_into(self, mapper, moved_m: float = 0.0,
                          turned_rad: float = 0.0) -> bool:
        """Fold one sweep and return whether the pose can be trusted.

        The odometry hint is passed as a PRIOR for the matcher, which is what
        it is for -- after the map has been asked how much of it happened (see
        MOTION_HYPOTHESES). The pose that comes out is the matched one;
        nothing here ever uses the integrated guess as a position.
        """
        scan = await self.robot.sweep()
        if not scan:
            return False
        self.last_fraction = 1.0
        if moved_m or turned_rad:
            self.last_fraction = motion_fraction(mapper, scan, moved_m,
                                                 turned_rad)
        info = mapper.update(scan, moved_m * self.last_fraction,
                             turned_rad * self.last_fraction)
        if isinstance(info, dict):
            import trip_recorder
            trip_recorder.note(
                "match", src="follower", moved=moved_m, turned=turned_rad,
                fraction=self.last_fraction, reason=info.get("reason"),
                score=info.get("score"), sharpness=info.get("sharpness"),
                pose=[round(float(v), 4) for v in mapper.pose])
        # WHY a sweep did or did not match ("score too low", "geometry
        # ambiguous", ...). Tonight's lost trip (2026-09-24 23:55) logged only
        # that it was lost; the reason is what says which fix it needs.
        self.last_match = (info.get("reason", "") if isinstance(info, dict)
                           else "")
        trusted = bool(mapper.pose_trusted)
        # ⚠ STANDING STILL, "GEOMETRY AMBIGUOUS" IS NOT "LOST". Parked, the
        # matcher searches 4 cm / 3 degrees, and sharpness is the peak over the
        # MEDIAN of that window -- where every candidate is nearly right, so a
        # correct match can fall under the gate. 2026-09-25 23:14: every tour
        # leg gave up on its first sweep, score 0.58-0.61 (floor 0.15),
        # sharpness 1.33 (gate 1.4), nothing having moved. With no motion the
        # mapper keeps the prior, which IS the last verified pose, so that
        # trust carries over. Only with no motion, only on this one reason,
        # only after a verified pose: after any move the rule is unchanged.
        if (not trusted and not moved_m and not turned_rad
                and self.last_match == "geometry ambiguous"
                and self._held_trust):
            mapper.pose_trusted = trusted = True
            self.last_match = "held: nothing moved since the last match"
        self._held_trust = trusted
        return trusted

    def odometry(self, direction: str, seconds: float) -> tuple[float, float]:
        """What a move of `seconds` should have done: (metres, radians).

        From the rate model, so it is only as good as the rates -- which is
        why it is used as a prior to be checked, never as a position. Left is
        counter-clockwise, the same sign `heading_error` steers by.
        """
        if direction in ("left", "right"):
            rad = math.radians(self.rates.turn_deg_s * seconds)
            return 0.0, (rad if direction == "left" else -rad)
        metres = self.rates.forward_cm_s * seconds / 100.0
        return (metres if direction == "forward" else -metres), 0.0

    async def step(self, mapper, direction: str,
                   seconds: float) -> "Executed":
        """One guarded move and its measured result. See `_execute`."""
        return await self._execute(mapper, direction, seconds)

    async def follow(self, mapper, path, goal_xy=None, *,
                     budget_m: float | None = None,
                     budget_s: float | None = None,
                     max_ticks: int | None = None,
                     arrival: ArrivalMonitor | None = None) -> FollowReport:
        """Drive `path` (world-metre waypoints) to its end, closing the loop."""
        path = np.asarray(path, dtype=np.float64)
        report = FollowReport(calibrated=self.rates.calibrated)
        if len(path) == 0:
            report.why = "empty path"
            return report

        goal = tuple(float(v) for v in (goal_xy if goal_xy is not None
                                        else path[-1]))
        budget_m = config.GOTO_BUDGET_M if budget_m is None else float(budget_m)
        budget_s = config.GOTO_BUDGET_S if budget_s is None else float(budget_s)
        max_ticks = config.GOTO_MAX_LEGS if max_ticks is None else int(max_ticks)
        monitor = arrival or ArrivalMonitor(goal, cell_m=mapper.grid.cell)
        report.tolerance_m = monitor.tolerance
        # A mapper handed over already verified (the trip tracker's) starts
        # trusted: its first standstill sweep need not re-earn it.
        self._held_trust = bool(getattr(mapper, "pose_trusted", False))

        started = time.monotonic()
        commanded = 0.0
        asked_m: deque[float] = deque(maxlen=STALL_TICKS)
        got_m: deque[float] = deque(maxlen=STALL_TICKS)
        asked_deg: deque[float] = deque(maxlen=STALL_TICKS)
        got_deg: deque[float] = deque(maxlen=STALL_TICKS)

        def stalled() -> bool:
            for asked, got in ((asked_m, got_m), (asked_deg, got_deg)):
                if (len(asked) == STALL_TICKS
                        and sum(got) < STALL_RATIO * sum(asked)):
                    return True
            return False

        while True:
            if self.robot.stopped():
                report.why = "stop was called"
                break
            if report.ticks >= max_ticks:
                report.why = "tick budget"
                break
            if report.driven_m >= budget_m:
                report.why = "distance budget"
                break
            now = time.monotonic()
            report.seconds = now - started
            if report.seconds >= budget_s:
                report.why = "time budget"
                break

            report.ticks += 1

            # ---- 1. WHERE AM I, REALLY. Measured, never integrated. --------
            if not await self._sweep_into(mapper):
                report.why = ("the sweep did not match the map, so I do not "
                              "know where I am. I will not drive on dead "
                              "reckoning"
                              + (f" ({self.last_match})" if self.last_match
                                 else ""))
                break
            pose = tuple(float(v) for v in mapper.pose)
            report.pose = [round(v, 3) for v in pose]

            # ---- 2. HAVE I ARRIVED. Held, not sampled. ---------------------
            state = monitor.update(pose, commanded, now)
            report.distance_m = state.distance_m
            report.tolerance_m = state.tolerance_m
            report.relaxed = state.relaxed
            if state.arrived:
                report.arrived, report.why = True, state.why
                break

            # ---- 3. WHAT TO AIM AT -----------------------------------------
            final = state.distance_m <= FINAL_APPROACH_M
            if final:
                report.final_approach_ticks += 1
                target = np.asarray(goal, dtype=np.float64)
            else:
                radius = max(LOOKAHEAD_MIN_M,
                             min(LOOKAHEAD_M, state.distance_m))
                target, _ = lookahead_point(path, pose[:2], radius)

            alpha = heading_error(pose, target)
            reach = math.hypot(target[0] - pose[0], target[1] - pose[1])

            # ---- 4. ONE MOVE, THROUGH THE GUARD ----------------------------
            if abs(math.degrees(alpha)) >= TURN_DEADBAND_DEG:
                degrees = math.degrees(alpha)
                # Proportional on the final approach: full correction out on
                # the path, gentler than that when lining up on the goal, so
                # it settles instead of oscillating across the tolerance.
                if final:
                    degrees *= 0.6
                degrees = float(np.clip(degrees, -config.MAX_INCREMENT_DEGREES,
                                        config.MAX_INCREMENT_DEGREES))
                seconds = self.rates.seconds_for_turn(degrees)
                commanded = STEP_MAX_M      # "still correcting", for arrival
                done = await self._execute(
                    mapper, "left" if degrees > 0 else "right", seconds)
                if done.refused:
                    report.why = f"refused: {done.said}"
                    break
                self.rates.observe_turn(seconds, done.turned_deg)
                report.turned_deg += abs(done.turned_deg)
                asked_deg.append(abs(degrees))
                got_deg.append(abs(done.turned_deg))
            else:
                # Pure pursuit's curvature does not steer this chassis, but it
                # does shorten the step: the sharper the arc, the less of it
                # may be driven as a straight chord.
                kappa = abs(curvature(pose, target))
                step = min(STEP_MAX_M, reach, state.distance_m)
                if kappa > 1e-6:
                    step = min(step, 0.25 / kappa)
                if final:
                    step = min(step, state.distance_m * 0.6)
                step = max(step, 0.0)
                if step < STEP_MIN_M:
                    commanded = 0.0
                    # Nothing worth driving and nothing worth turning: the
                    # controller has run out of corrections. Let the monitor
                    # see a quiet tick rather than inventing a move.
                    continue
                commanded = step
                seconds = self.rates.seconds_for(step)
                done = await self._execute(mapper, "forward", seconds)
                if done.refused:
                    report.why = f"refused: {done.said}"
                    break
                self.rates.observe_drive(seconds, done.moved_m)
                report.driven_m += done.moved_m
                asked_m.append(step)
                got_m.append(done.moved_m)

            if stalled():
                report.why = (f"the last {STALL_TICKS} moves delivered under "
                              f"{STALL_RATIO:.0%} of what they asked for — the "
                              "wheels are not driving, and everything after "
                              "this would come off a heading that is a guess")
                break

        report.seconds = time.monotonic() - started
        report.events = self.events
        log.info("follow: %s", report.as_dict())
        return report

    async def _execute(self, mapper, direction: str,
                       seconds: float) -> "Executed":
        """One guarded move, then measure what it actually did.

        Everything it reports is AS MEASURED by the scan matcher, never as
        commanded: the rate model, the stall window and the distance budget
        are all fed these numbers, which is the whole point of closing the
        loop.

        A REFUSAL ENDS THE RUN rather than being retried, which is the rule
        `slam/navigate.py` already follows. The guard refuses for reasons that
        another identical attempt will not change -- nobody has looked yet,
        something is inside the stop distance -- so a retry loop is just the
        same refusal at a higher rate, with the robot still sitting there.
        """
        seconds = float(np.clip(seconds, 0.05, config.MAX_MOVE_SECONDS))
        before = tuple(float(v) for v in mapper.pose)
        # ⚠ LOOK BEFORE EVERY MOVE, TURNS INCLUDED. `embodiment.check` refuses
        # ANY move with no observation, or one gone stale since she last
        # moved -- not only forward ones. This used to observe before forward
        # steps only, and a journey's first move is nearly always a turn to
        # face the route, so on the robot every journey was refused before a
        # wheel turned ("Friday has not looked at the room yet", Harsh's
        # dashboard trip, 2026-09-23 22:43). The fakes never enforced it.
        # Range sensors only, no camera: cheap next to the sweep that follows.
        await self.robot.observe()
        said = await self.robot.drive_seconds(direction, seconds)
        refused = "REFUSED" in said.upper()

        # A refused move did nothing, and saying otherwise would hand the
        # matcher a motion that never happened. A move CUT SHORT did only the
        # part it ran for: 2026-09-25 22:43 a 1.2 s turn stopped after 0.2 s
        # was handed to the matcher as the full 90 degrees, the thin map
        # accepted it, and her heading was 75 degrees out from then on.
        ran = seconds if not refused else 0.0
        cut = _RAN_FOR.search(said or "")
        if cut and not refused:
            ran = min(seconds, float(cut.group(1)))
        moved_m, turned_rad = ((0.0, 0.0) if ran <= 0.0
                               else self.odometry(direction, ran))
        await self._sweep_into(mapper, moved_m, turned_rad)
        after = tuple(float(v) for v in mapper.pose)
        moved = math.hypot(after[0] - before[0], after[1] - before[1])
        turned = math.degrees(_wrap(after[2] - before[2]))

        self.events.append({
            "direction": direction, "commanded_s": round(seconds, 2),
            "moved_m": round(moved, 3), "turned_deg": round(turned, 1),
            "fraction": self.last_fraction,
            "refused": refused, "said": said,
            "rate_cm_s": round(self.rates.forward_cm_s, 1),
            "rate_deg_s": round(self.rates.turn_deg_s, 1),
        })
        return Executed(moved, turned, refused, said)


class GuardedRobot:
    """The real robot, with the one call `Follower` needs that `_Robot` lacks.

    `mcp_server/tools/explore.py::_Robot` already wraps sweep / observe / say /
    stopped / reflex_trips, and every one of those runs the blocking tool in a
    worker thread so `stop()` can still be answered while a move is in flight.
    That class is reused rather than reimplemented; this adds `drive_seconds`,
    which is `movement.move()` -- the guarded, calibration-free primitive --
    on the same worker thread for the same reason.

    ⚠ `movement.move()` AND NOTHING ELSE. Not `hardware.motors()`, not
    `Motors.drive()`. Everything this class exists to do is to be the one
    narrow place where the follower touches the robot, so that the guard
    cannot be gone round by accident later.

    Imported lazily so the geometry above stays testable on a machine with no
    LiDAR, no motor board and no MCP server running.
    """

    def __init__(self) -> None:
        from mcp_server.tools.explore import _Robot
        self._inner = _Robot()

    def __getattr__(self, name: str):
        # sweep / observe / say / stopped / reflex_trips, unchanged.
        return getattr(self._inner, name)

    def arm(self) -> int:
        """Baseline the reflex trip counter before a run starts.

        `_Robot.stopped()` tells a HUMAN stop from a SAFETY trip by watching
        whether the trip count moved, so it needs a baseline taken at the
        start rather than a zero. Without this the first reflex trip of the
        session reads as "a person called stop" and ends the run, which is
        the bug `go_to` fixed with the same one line.
        """
        self._inner._trips = self._inner.reflex_trips()
        return self._inner._trips

    async def drive_seconds(self, direction: str, seconds: float) -> str:
        import anyio.to_thread

        import trip_recorder
        from mcp_server.tools import movement
        said = await anyio.to_thread.run_sync(movement.move, direction,
                                              float(seconds))
        trip_recorder.note("move", src="follower", direction=direction,
                           seconds=float(seconds), said=said)
        return said

    async def sweep(self) -> dict[int, int]:
        import trip_recorder
        scan = await self._inner.sweep()
        trip_recorder.note("sweep", src="follower", scan=scan)
        return scan

    async def clearance(self) -> tuple[float | None, bool]:
        """(millimetres ahead, is it blocking). READ ONLY — moves nothing.

        `safety.check_forward_clearance()` is the same reading the guard
        itself uses to refuse a move, so asking it here cannot disagree with
        the thing that stopped her. A None clearance comes back as BLOCKED,
        not clear: nobody can see ahead is not permission to drive.
        """
        import anyio.to_thread

        from mcp_server import safety
        mm, _source, blocked = await anyio.to_thread.run_sync(
            safety.check_forward_clearance)
        return mm, bool(blocked)

    async def halt(self) -> str:
        """Stop the wheels and stay stopped. Used when entering FAILSAFE."""
        import anyio.to_thread

        from mcp_server.tools import movement
        return await anyio.to_thread.run_sync(movement.stop)
