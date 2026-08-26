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

    def _raw_indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices without clipping -- may fall outside the field."""
        cells = np.floor((points - self.min_xy) / self.cell).astype(np.int64)
        return cells[:, 0], cells[:, 1]

    def _indices(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ix, iy = self._raw_indices(points)
        return (np.clip(ix, 0, self.shape[0] - 1),
                np.clip(iy, 0, self.shape[1] - 1))

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


    def score_xy(self, x: np.ndarray, y: np.ndarray) -> float:
        """score() for coordinates already split into two arrays.

        Same number, without building an (N, 2) array to pass it in. That
        allocation is most of the cost when scoring a thousand candidate
        poses, which is exactly what match_pose does.
        """
        ix = np.floor((x - self.min_xy[0]) / self.cell).astype(np.int64)
        iy = np.floor((y - self.min_xy[1]) / self.cell).astype(np.int64)
        good = ((ix >= 0) & (ix < self.shape[0]) &
                (iy >= 0) & (iy < self.shape[1]))
        if not good.any():
            return 0.0
        return float(self.field[ix[good], iy[good]].sum()
                     / len(x) / self._perfect)

    def score_shifts(self, points: np.ndarray, half: int) -> np.ndarray:
        """Score every whole-cell translation in a (2*half+1) square, at once.

        THE POINT OF THIS. Scoring a candidate pose is a table lookup per
        point, and shifting the pose by exactly one cell shifts every one of
        those lookups by exactly one index. So a whole square of candidate
        translations is the same index array read at different offsets -- the
        cell coordinates are computed once for the sweep instead of once per
        candidate, which is what took a third of a second on the Pi and now
        takes a few milliseconds.

        The field is padded rather than bounds-checked per shift: a mask over
        350 points, eighty-one times per rotation, costs more than the copy.
        """
        n = len(points)
        if n == 0:
            return np.zeros((2 * half + 1, 2 * half + 1))
        ix, iy = self._raw_indices(points)
        keep = ((ix >= -half) & (ix < self.shape[0] + half) &
                (iy >= -half) & (iy < self.shape[1] + half))
        out = np.zeros((2 * half + 1, 2 * half + 1))
        if not keep.any():
            return out
        pad = 2 * half
        padded = np.pad(self.field, pad, mode="constant")
        stride = padded.shape[1]
        flat = padded.reshape(-1)
        base = (ix[keep] + pad) * stride + (iy[keep] + pad)
        for i in range(-half, half + 1):
            row = base + i * stride
            for j in range(-half, half + 1):
                out[i + half, j + half] = flat[row + j].sum()
        return out / n / self._perfect


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


def match_pose(field: LikelihoodField, query: np.ndarray,
               prior: tuple[float, float, float] = (0.0, 0.0, 0.0),
               window_m: float | None = None,
               window_deg: float | None = None,
               ) -> tuple[tuple[float, float, float], float, float]:
    """Correct a pose prior by matching a sweep against a reference field.

    `prior` is where odometry says the robot is, in world metres and radians.
    The search is an exhaustive sweep of a small box around it -- there are no
    wheel encoders, so the prior is a TIMED guess worth about +/-20%, and the
    window has to be wide enough to contain that error over one sweep.

    Exhaustive rather than gradient descent for the same reason match_translation
    is: scoring is an array lookup, so a thousand candidates cost tens of
    milliseconds, and a search that visits every candidate cannot settle into a
    local maximum. A corridor still has a ridge rather than a peak along its
    length -- that is real ambiguity in the geometry, not a failure of the
    search, and it is what the sharpness return is for.

    Returns ((x, y, theta), score, sharpness). Score is 0..1 against a perfect
    match; sharpness is peak over median across the whole window, and near 1
    it means the room did not constrain the answer however good the score
    looked. Below config.SLAM_MIN_SCORE / SLAM_MIN_CONFIDENCE the caller
    should keep its odometry and say that it did.
    """
    window_m = config.SLAM_POSE_WINDOW_M if window_m is None else window_m
    window_deg = config.SLAM_POSE_WINDOW_DEG if window_deg is None else window_deg
    if query.size == 0:
        return prior, 0.0, 1.0

    px, py, pth = prior
    half = max(int(round(window_m / field.cell)), 1)

    # COARSE: whole-cell translations, scored a square at a time.
    best = (px, py, pth)
    best_score = -1.0
    spread = []
    for dth in np.radians(np.arange(-window_deg, window_deg + 1e-9, 3.0)):
        rotated = transform(query, px, py, pth + dth)
        scores = field.score_shifts(rotated, half)
        spread.append(scores)
        flat_best = int(np.argmax(scores))
        i, j = divmod(flat_best, scores.shape[1])
        if scores[i, j] > best_score:
            best_score = float(scores[i, j])
            best = (px + (i - half) * field.cell,
                    py + (j - half) * field.cell, pth + dth)

    # FINE: sub-cell, in a box one cell wide around the coarse answer. Going
    # finer than the blur only reads the smoothing, so a third of a cell and
    # one degree is where it stops paying.
    cx, cy, cth = best
    step = field.cell / 3.0
    offsets = np.arange(-field.cell, field.cell + 1e-9, step)
    for dth in np.radians(np.arange(-3.0, 3.0 + 1e-9, 1.0)):
        rotated = transform(query, cx, cy, cth + dth)
        for dx in offsets:
            shifted_x = rotated[:, 0] + dx
            for dy in offsets:
                value = field.score_xy(shifted_x, rotated[:, 1] + dy)
                if value > best_score:
                    best_score = value
                    best = (cx + dx, cy + dy, cth + dth)

    everything = np.concatenate([block.ravel() for block in spread])
    median = float(np.median(everything))
    sharpness = (float(everything.max() / median) if median > 1e-9
                 else float("inf"))
    return (float(best[0]), float(best[1]), float(best[2])), best_score, sharpness
