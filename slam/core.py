"""Points, the likelihood field, and the scan matcher.

This is the piece everything else in slam/ is built on, and it is deliberately
the piece with no robot in it: it takes {bearing: mm} dicts and numpy arrays
and gives back poses, so it can be tested against recorded sweeps.

WHY A LIKELIHOOD FIELD RATHER THAN NEAREST-NEIGHBOUR ICP. ICP needs a nearest
neighbour per point per iteration, which on a Pi means a KD-tree rebuild every
scan and a search that is hard to bound. A likelihood field is built once per
reference scan as a blurred image, and scoring a candidate pose is then one
array index and one sum -- a few hundred microseconds for 350 points. Scoring
a thousand candidate poses is still cheap, so the search can be exhaustive
over a small window instead of iterative, which is also what makes it robust:
it cannot fall into a local minimum it never visited.

WHY THE SCAN MATCHER AND NOT A COSINE. Translating the robot by t does not
shift a range reading by t.u along the same bearing -- after moving, the beam
at that bearing lands somewhere else on the surface entirely, and the range
changes by (t.n)/(u.n) for a surface with normal n. That is only t.u where the
surface faces the beam squarely, which almost nothing in a room does. Fitting
a cosine to the range deltas of a real room explained 12% of the variance when
it was tried. Matching the point clouds makes no assumption about the
surfaces at all.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter

import config


def points_from_scan(scan: dict[int, int],
                     forward_deg: float | None = None,
                     max_mm: float | None = None) -> np.ndarray:
    """{raw bearing: mm} -> (N, 2) array of metres in the robot's frame.

    x is forward and y is left, the usual robot convention, IF forward_deg is
    given. With forward_deg None the points are in the LiDAR's own frame --
    which is still perfectly good for matching one sweep against another,
    because both are in the same frame. That is what lets the forward bearing
    itself be measured by matching.

    Bearings looking at Friday's own chassis are already gone: sensors.Lidar
    drops them inside scan(). Zero and out-of-range returns are dropped here.
    """
    limit = max_mm if max_mm is not None else config.LIDAR_MAX_RANGE_MM
    bearings = []
    ranges = []
    for bearing, mm in scan.items():
        if mm is None or mm <= 0 or mm > limit:
            continue
        bearings.append(bearing if forward_deg is None else bearing - forward_deg)
        ranges.append(mm / 1000.0)
    if not ranges:
        return np.empty((0, 2), dtype=np.float64)
    theta = np.radians(np.asarray(bearings, dtype=np.float64))
    r = np.asarray(ranges, dtype=np.float64)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


def transform(points: np.ndarray, dx: float, dy: float,
              dtheta: float) -> np.ndarray:
    """Rigid transform of a point cloud. dtheta in radians, rotation first."""
    if points.size == 0:
        return points
    cos, sin = math.cos(dtheta), math.sin(dtheta)
    rotated = np.empty_like(points)
    rotated[:, 0] = points[:, 0] * cos - points[:, 1] * sin + dx
    rotated[:, 1] = points[:, 0] * sin + points[:, 1] * cos + dy
    return rotated


class LikelihoodField:
    """A blurred occupancy image of one scan, for scoring candidate poses.

    Built once from a reference cloud; after that, scoring a transformed
    cloud is an array lookup and a sum. The blur is what makes the score a
    smooth function of pose instead of a field of spikes -- without it, a
    candidate that is one cell out scores zero rather than nearly-right, and
    nothing can be optimised.
    """

    def __init__(self, points: np.ndarray, cell_m: float | None = None,
                 blur_m: float | None = None, pad_m: float = 0.5) -> None:
        self.cell = cell_m if cell_m is not None else config.SLAM_CELL_M
        blur = blur_m if blur_m is not None else config.SLAM_BLUR_M
        if points.size == 0:
            raise ValueError("cannot build a likelihood field from no points")

        self.min_xy = points.min(axis=0) - pad_m
        span = (points.max(axis=0) + pad_m) - self.min_xy
        shape = np.maximum(np.ceil(span / self.cell).astype(int) + 1, 2)
        self.shape = (int(shape[0]), int(shape[1]))

        grid = np.zeros(self.shape, dtype=np.float32)
        ix, iy = self._indices(points)
        np.add.at(grid, (ix, iy), 1.0)
        grid = gaussian_filter(grid, sigma=blur / self.cell, mode="constant")
        self.field = grid

        # ⚠ NORMALISE AGAINST A PERFECT MATCH, NOT AGAINST THE PEAK CELL.
        # Dividing by grid.max() looks like it gives a 0..1 score and does
        # not: after blurring, a point sitting exactly where it was seen
        # lands on a cell well below the peak, so a FLAWLESS match scored
        # 0.36 and there was no way to read a threshold off that number.
        # Scoring the reference against itself gives the value a perfect
        # match actually produces, and dividing by it makes 1.0 mean perfect
        # -- so SLAM_MIN_SCORE can be reasoned about instead of tuned.
        ix, iy = self._indices(points)
        self._perfect = float(self.field[ix, iy].sum() / len(points)) or 1.0

    def _indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = np.floor((points - self.min_xy) / self.cell).astype(np.int64)
        ix = np.clip(cells[:, 0], 0, self.shape[0] - 1)
        iy = np.clip(cells[:, 1], 0, self.shape[1] - 1)
        return ix, iy

    def score(self, points: np.ndarray) -> float:
        """Mean likelihood of these points, 0..1. Higher is better aligned.

        1.0 is as good as the reference scan matching itself; see the note
        in __init__ about why that normalisation is not optional.

        Points that fall outside the reference's extent score zero rather
        than being dropped, so a candidate pose that throws half the scan off
        the edge of what was seen is correctly penalised instead of being
        rewarded for having fewer points to disagree about.
        """
        if points.size == 0:
            return 0.0
        inside = np.all((points >= self.min_xy) &
                        (points < self.min_xy + np.array(self.shape) * self.cell),
                        axis=1)
        if not inside.any():
            return 0.0
        ix, iy = self._indices(points[inside])
        return float(self.field[ix, iy].sum() / len(points) / self._perfect)


def match_translation(reference: np.ndarray, query: np.ndarray,
                      distance_m: float, step_deg: float = 1.0,
                      refine: bool = True) -> tuple[float, float]:
    """Which way did the robot go, given that it went `distance_m` straight?

    The robot translated by a known distance along an unknown bearing and did
    not rotate. So there is exactly one unknown, and it can be searched
    exhaustively rather than descended into: for each candidate bearing, undo
    that translation on the query cloud and score it against the reference.

    Both clouds are in the LiDAR's own frame, which is the point -- this needs
    no prior knowledge of which way forward is, and is therefore how forward
    gets measured in the first place.

    Returns (bearing_deg, score), where 1.0 is as good as a scan matching
    itself. Below config.SLAM_MIN_SCORE the answer should not be believed.
    """
    # COARSE TO FINE, AND THE FINE PASS IS NOT OPTIONAL. A blur wide enough
    # to make the global search safe is also wide enough that a 25 cm move in
    # almost ANY direction still lands most points near some surface -- on the
    # real robot, 127 of 360 bearings scored within 75% of the peak and the
    # answer looked untrustworthy. Re-scored at a tenth of the blur, the same
    # data gave a 13.6x peak-to-median ratio and the same bearing. So the wide
    # blur finds the basin and the narrow one reads the number out of it.
    coarse_field = LikelihoodField(reference)

    def best_over(field: LikelihoodField,
                  candidates: np.ndarray) -> tuple[float, float]:
        best_deg, best_score = 0.0, -1.0
        for deg in candidates:
            rad = math.radians(deg)
            # The robot moved +distance along `deg`, so in its own frame the
            # world moved -distance. Undo it and the clouds overlap.
            moved = transform(query,
                              distance_m * math.cos(rad),
                              distance_m * math.sin(rad), 0.0)
            value = field.score(moved)
            if value > best_score:
                best_deg, best_score = float(deg), value
        return best_deg, best_score

    coarse, coarse_score = best_over(coarse_field,
                                     np.arange(0.0, 360.0, step_deg))
    if not refine:
        return coarse % 360.0, coarse_score

    fine_field = LikelihoodField(reference,
                                 cell_m=config.SLAM_FINE_CELL_M,
                                 blur_m=config.SLAM_FINE_BLUR_M)
    window = np.arange(coarse - 4 * step_deg, coarse + 4 * step_deg + 1e-9,
                       step_deg / 10)
    deg, score = best_over(fine_field, window)
    return deg % 360.0, score


def match_confidence(reference: np.ndarray, query: np.ndarray,
                     distance_m: float, step_deg: float = 2.0) -> float:
    """How sharply the best bearing beats the alternatives.

    The score alone does not say whether an answer is trustworthy: a low peak
    over a much lower field is a good answer, and a high peak over an almost
    equally high field is not. This is peak divided by median across all
    bearings, measured at the fine resolution -- above about 5 the answer is
    a real peak, near 1 the geometry simply does not constrain it.
    """
    field = LikelihoodField(reference, cell_m=config.SLAM_FINE_CELL_M,
                            blur_m=config.SLAM_FINE_BLUR_M)
    scores = []
    for deg in np.arange(0.0, 360.0, step_deg):
        rad = math.radians(deg)
        scores.append(field.score(transform(query,
                                            distance_m * math.cos(rad),
                                            distance_m * math.sin(rad), 0.0)))
    scores = np.asarray(scores)
    median = float(np.median(scores))
    return float(scores.max() / median) if median > 1e-9 else float("inf")
