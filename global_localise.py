"""global_localise.py — finding her on her own map, from nothing.

    saved map + one LiDAR sweep  ──>  (x, y, theta) anywhere in the map
                                 └──>  or a refusal, when the room is ambiguous

⚠ WHY THIS EXISTS. `places_tools._placed_mapper()` is the "work out where I
am" step behind `go_to`, `remember_place`, `navigate_to` and `return_home`.
It builds `Mapper(grid=saved, pose=(0,0,0))` and folds sweeps with NO
odometry, which in `Mapper._locate` gives:

    prior_m   = max(SLAM_ODOM_ERROR * 0.0, SLAM_POSE_PRIOR_FLOOR_M) = 0.04 m
    prior_deg = max(SLAM_ODOM_ERROR * 0.0, SLAM_POSE_PRIOR_FLOOR_DEG) = 3 deg
    hold_angle = SLAM_HOLD_WHEN_STILL and parked and no odometry     = True

So the matcher may move her FOUR CENTIMETRES, and the angle is pinned
outright. That is a refinement, not a search: stand her anywhere in the room
and she is placed at the map origin -- and `pose_trusted` comes back True, so
nothing downstream doubts it. Observed 2026-09-22: after an evening of being
driven metres around the room, two consecutive `navigate_to` calls both
reported pose (-0.0, -0.0, 0.0). See [[friday-cannot-find-herself]].

⚠ THE FIX IS NOT TO WIDEN THAT FLOOR. It protects the TRACKING case, where a
large window lets the matcher jump the robot onto a different wall that
happens to look similar. Global localisation is a different question --
"where in this whole map am I?" rather than "how wrong is my estimate?" -- and
it gets its own entry point rather than loosening one that is load-bearing
somewhere else.

HOW IT WORKS. For a FIXED heading, the score of every possible translation is
a cross-correlation: the score at offset t is the sum of the likelihood field
read at each beam's cell plus t, and that is exactly what correlating the
field with the scan's own raster computes. So one FFT per candidate heading
gives every position at once, instead of scoring poses one at a time. The
whole map at 10-degree steps costs well under a second on this Pi, where
looping over positions would cost minutes.

Scoring is `slam.core.LikelihoodField`'s own, including its `known` mask in
the denominator -- beams that land on floor nobody has mapped are removed
from the divisor rather than counted as misses. Using the same scorer as the
tracker is deliberate: two different notions of "how good is this pose" on
one robot is how they drift apart.

⚠ IT REFUSES WHEN THE ROOM IS AMBIGUOUS, and that is the most important thing
in the file. A corridor looks like itself every two metres; four bare walls
look like themselves every 90 degrees. A localiser that always returns its
best guess will, in those rooms, confidently return the WRONG one -- and a
robot that believes a wrong pose drives to the wrong address with no symptom
anyone can see. So the runner-up is checked: if a hypothesis somewhere else
scores nearly as well, this says so and returns nothing.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field as _field

import numpy as np
from scipy.signal import fftconvolve

import config
from slam.core import LikelihoodField, points_from_scan

log = logging.getLogger("friday.global_localise")

# Heading steps for the coarse pass. 10 degrees is fine enough that the true
# pose is within half a step of a candidate, and the polish pass closes that.
COARSE_STEP_DEG = 10.0
FINE_STEP_DEG = 2.0
FINE_SPAN_DEG = 12.0

# Beams kept for the coarse pass. The LiDAR gives ~340; scoring is linear in
# beam count and the answer does not improve past this, because neighbouring
# beams in a 340-beam sweep see nearly the same wall.
COARSE_BEAMS = 120

# ⚠ KEEP MORE THAN ONE HYPOTHESIS OUT OF THE COARSE PASS. Taking only the
# best cell per heading and refining that one loses every room where the true
# pose is the runner-up at coarse resolution -- and with 120 beams and 10
# degree steps it often is. Measured: a pose that every other heading found to
# 11 cm came back 2.26 m out and 180 degrees flipped, because the flipped
# hypothesis happened to win the coarse pass and the fine pass could only
# search +/-12 degrees around it. Refining the top few with the FULL scan lets
# the right one overtake.
COARSE_KEEP = 6

# A pose must beat the best hypothesis THIS FAR AWAY by this margin, or the
# room is too self-similar to answer honestly.
AMBIGUITY_RADIUS_M = 0.75
AMBIGUITY_MARGIN = 0.12

# ⚠ A POSE MUST EXPLAIN MOST OF THE SCAN, NOT A CORNER OF IT. Scoring is
# (likelihood over judgeable beams) / (judgeable beams), which is right for
# tracking but is wide open during a global search: a pose facing mostly into
# unmapped space has almost no beams in the denominator, so the handful that
# do land on a wall can score it far above the truth.
#
# MEASURED on the real map, the pose this rule was written for:
#     the true pose   score 0.94, judged on 256/256 beams (100% coverage)
#     what won        score 2.26, judged on  34/256 beams ( 13% coverage)
#
# `SLAM_MATCH_MIN_BEAMS` (40) is an absolute floor and 34 beams nearly cleared
# it. The question that matters is not "how many beams" but "what FRACTION of
# what she can see does this pose account for" -- 13% is not a fix, it is a
# coincidence with a good view of one wall.
MIN_COVERAGE = 0.45

# Below this the scan simply does not match the map anywhere -- she is in a
# different room, or the map is too thin to place anything in.
MIN_SCORE = 0.35


@dataclass
class Fix:
    """Where she is, how sure, and what came second."""

    pose: tuple[float, float, float] | None = None
    score: float = 0.0
    runner_up: tuple[float, float, float] | None = None
    runner_up_score: float = 0.0
    margin: float = 0.0
    why: str = ""
    candidates: list = _field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.pose is not None

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "pose": ([round(v, 3) for v in self.pose] if self.pose else None),
            "score": round(self.score, 3),
            "runner_up": ([round(v, 3) for v in self.runner_up]
                          if self.runner_up else None),
            "runner_up_score": round(self.runner_up_score, 3),
            "margin": round(self.margin, 3),
            "why": self.why,
        }


def _subsample(points: np.ndarray, keep: int) -> np.ndarray:
    if len(points) <= keep:
        return points
    step = len(points) / float(keep)
    idx = (np.arange(keep) * step).astype(np.int64)
    return points[idx]


def _rotate(points: np.ndarray, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.column_stack((points[:, 0] * c - points[:, 1] * s,
                            points[:, 0] * s + points[:, 1] * c))


def _cell_to_world(field: LikelihoodField, ix: int, iy: int, theta: float
                   ) -> tuple[float, float, float]:
    """A winning correlation cell, as a pose in the map's frame.

    ⚠ THE OUTPUT INDEX IS THE ROBOT'S CELL, not an offset from the canvas
    centre, and getting that wrong is invisible in the score -- every fix
    comes back confident and wrong by exactly (cx, cy) cells. Worked through:
    `fftconvolve(A, B[::-1], mode="same")[i]` sums A at `k + i - N//2` over
    B's nonzero k. The raster puts beam k at `offset + cx`, so the field is
    read at `offset + i`, which is the robot standing at cell i.
    """
    return (float(field.min_xy[0] + (ix + 0.5) * field.cell),
            float(field.min_xy[1] + (iy + 0.5) * field.cell),
            float((theta + math.pi) % (2 * math.pi) - math.pi))


def _score_all_translations(field: LikelihoodField, rotated: np.ndarray
                            ) -> np.ndarray:
    """Score EVERY position on the map for one heading, in one correlation.

    The returned array is indexed by the robot's cell on the map's own
    lattice: out[ix, iy] is the score of standing at that cell with the scan
    rotated as given.

    The beam raster is built on a canvas the size of the map and centred, so
    correlating it against the field slides the whole scan over every cell.
    `fftconvolve` with the raster flipped is a cross-correlation.
    """
    nx, ny = field.shape
    cx, cy = nx // 2, ny // 2

    # Beam endpoints as offsets in cells from the robot, dropped onto a canvas
    # centred at (cx, cy).
    ix = np.rint(rotated[:, 0] / field.cell).astype(np.int64) + cx
    iy = np.rint(rotated[:, 1] / field.cell).astype(np.int64) + cy
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    raster = np.zeros((nx, ny), dtype=np.float32)
    if not inside.any():
        return raster
    np.add.at(raster, (ix[inside], iy[inside]), 1.0)

    flipped = raster[::-1, ::-1]
    hits = fftconvolve(field.field, flipped, mode="same")
    seen = fftconvolve(field.known.astype(np.float32), flipped, mode="same")

    # ⚠ THE DENOMINATOR IS THE BEAMS THE MAP CAN JUDGE, exactly as
    # LikelihoodField._normalise does it. Dividing by all beams instead would
    # penalise every pose that looks at unmapped floor, which on a 12 m2 map
    # is most of them -- and the bias points AWAY from the open parts of the
    # room, which is precisely where she usually is.
    need = max(float(config.SLAM_MATCH_MIN_BEAMS),
               MIN_COVERAGE * float(len(rotated)))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(seen >= need, hits / np.maximum(seen, 1e-9), 0.0)
    return (out / field._perfect).astype(np.float32)


def localise(grid, scan: dict[int, int], forward_deg: float | None = None,
             coarse_step_deg: float = COARSE_STEP_DEG,
             beams: int = COARSE_BEAMS,
             restrict_to: np.ndarray | None = None) -> Fix:
    """Find her anywhere in `grid` from one sweep. Never guesses.

    Args:
        grid: the saved OccupancyGrid to localise against.
        scan: one raw sweep, {bearing_deg: mm}.
        forward_deg: the LiDAR's forward bearing; defaults to the configured one.
        restrict_to: optional bool mask of cells she could possibly be standing
            on -- pass `path_planner.preprocess(grid).plannable` and the search
            skips poses inside walls, which is both faster and stops a
            reflection scoring well from inside the sofa.
    """
    # ⚠ DEFAULT TO THE ROBOT'S FRAME, NOT THE LIDAR'S. points_from_scan with
    # forward_deg=None returns points in the LiDAR's own frame, which is fine
    # for matching two sweeps to each other but is rotated by LIDAR_FORWARD_DEG
    # (51.8 deg) from the frame the MAP is in. Leaving it None put every fix
    # out by exactly that angle -- a constant error, which is the signature of
    # a convention mismatch rather than a bad search.
    if forward_deg is None:
        forward_deg = config.LIDAR_FORWARD_DEG
    local = points_from_scan(scan, forward_deg=forward_deg)
    if len(local) < config.SLAM_MATCH_MIN_BEAMS:
        return Fix(why=f"only {len(local)} usable beams in that sweep")
    try:
        field = LikelihoodField.from_grid(grid)
    except (ValueError, AttributeError) as exc:
        return Fix(why=f"cannot build a field from this map ({exc})")

    coarse = _subsample(local, beams)
    nx, ny = field.shape

    allowed = None
    if restrict_to is not None and restrict_to.shape == (nx, ny):
        allowed = restrict_to

    # ---- coarse: every position, every 10 degrees, one FFT per heading ---
    stack = []                       # (score, ix, iy, theta), best per heading
    for step in np.arange(0.0, 360.0, coarse_step_deg):
        theta = math.radians(step)
        scores = _score_all_translations(field, _rotate(coarse, theta))
        if allowed is not None:
            scores = np.where(allowed, scores, 0.0)
        flat = int(np.argmax(scores))
        ix, iy = divmod(flat, ny)
        stack.append((float(scores[ix, iy]), ix, iy, theta))

    stack.sort(reverse=True)
    if not stack or stack[0][0] < MIN_SCORE:
        got = stack[0][0] if stack else 0.0
        return Fix(score=got,
                   why=("nothing on my map looks like what I can see "
                        f"(best score {got:.2f}). I may be in a different "
                        "room, or the map is too thin to place myself in."))

    # ---- refine the best few, with every beam this time ------------------
    refined = []
    for _, bix, biy, btheta in stack[:COARSE_KEEP]:
        best_here = None
        for delta in np.arange(-FINE_SPAN_DEG, FINE_SPAN_DEG + 1e-9,
                               FINE_STEP_DEG):
            theta = btheta + math.radians(delta)
            scores = _score_all_translations(field, _rotate(local, theta))
            if allowed is not None:
                scores = np.where(allowed, scores, 0.0)
            # Only near the candidate: a different peak elsewhere is a
            # different hypothesis and gets its own refinement.
            window = np.zeros_like(scores, dtype=bool)
            half = int(round(AMBIGUITY_RADIUS_M / field.cell))
            window[max(bix - half, 0):bix + half + 1,
                   max(biy - half, 0):biy + half + 1] = True
            local_scores = np.where(window, scores, 0.0)
            flat = int(np.argmax(local_scores))
            ix, iy = divmod(flat, ny)
            got = float(local_scores[ix, iy])
            if best_here is None or got > best_here[0]:
                best_here = (got, ix, iy, theta)
        if best_here:
            refined.append(best_here)

    if not refined:
        return Fix(why="the refinement pass found nothing to settle on")
    refined.sort(reverse=True)
    score, ix, iy, theta = refined[0]
    pose = _cell_to_world(field, ix, iy, theta)

    # ---- is anywhere ELSE nearly as good? --------------------------------
    rival, rival_score = None, 0.0
    for other_score, oix, oiy, otheta in refined[1:]:
        ox, oy, _ = _cell_to_world(field, oix, oiy, otheta)
        if math.hypot(ox - pose[0], oy - pose[1]) >= AMBIGUITY_RADIUS_M:
            rival = _cell_to_world(field, oix, oiy, otheta)
            rival_score = other_score
            break

    margin = score - rival_score if rival else score
    fix = Fix(score=score, runner_up=rival, runner_up_score=rival_score,
              margin=margin)
    if rival is not None and margin < AMBIGUITY_MARGIN:
        fix.why = (f"I can see two places I could be — ({pose[0]:.1f}, "
                   f"{pose[1]:.1f}) and ({rival[0]:.1f}, {rival[1]:.1f}) — and "
                   f"they look almost the same from here ({score:.2f} against "
                   f"{rival_score:.2f}). I will not guess between them.")
        return fix

    fix.pose = pose
    fix.why = (f"placed at ({pose[0]:.2f}, {pose[1]:.2f}) facing "
               f"{math.degrees(pose[2]):.0f}°, score {score:.2f}"
               + (f", next best elsewhere {rival_score:.2f}" if rival else ""))
    log.info("global fix: %s", fix.as_dict())
    return fix
