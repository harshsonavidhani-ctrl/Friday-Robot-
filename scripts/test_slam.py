"""Does the scan matcher actually recover a translation it was not told?

    ./venv/bin/python scripts/test_slam.py

No robot needed. A synthetic room is ray-cast from two known poses and the
matcher is asked to work out which way the robot went; the answer is checked
against the truth. This exists because the first attempt at measuring
LIDAR_FORWARD_DEG fitted a cosine to the range deltas, got a confident-looking
number, and was wrong -- the model was wrong, and nothing in the pipeline was
able to say so. A test with a known answer would have caught it in seconds.

The room is deliberately awkward: oblique walls and a pillar, because a cosine
fit works on a circular room and fails on a real one.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from slam.core import match_translation, points_from_scan  # noqa: E402

# (x0, y0, x1, y1) segments in metres. A skewed room with a pillar in it.
ROOM = [
    (-3.0, -2.5, 3.5, -2.2),      # bottom wall, slightly off square
    (3.5, -2.2, 3.1, 2.6),        # right wall, oblique
    (3.1, 2.6, -2.6, 2.9),        # top wall
    (-2.6, 2.9, -3.0, -2.5),      # left wall
    (0.6, 0.4, 1.0, 0.4),         # a pillar, four short sides
    (1.0, 0.4, 1.0, 0.9),
    (1.0, 0.9, 0.6, 0.9),
    (0.6, 0.9, 0.6, 0.4),
]


def cast(pose: tuple[float, float], bearing_deg: float) -> float | None:
    """Distance in metres from pose along bearing to the nearest wall."""
    ox, oy = pose
    rad = math.radians(bearing_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    best = None
    for x0, y0, x1, y1 in ROOM:
        sx, sy = x1 - x0, y1 - y0
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-12:
            continue
        t = ((x0 - ox) * sy - (y0 - oy) * sx) / denom       # along the ray
        u = ((x0 - ox) * dy - (y0 - oy) * dx) / denom       # along the segment
        if t > 0.05 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


def sweep(pose: tuple[float, float], noise_mm: float = 8.0,
          rng: np.random.Generator | None = None) -> dict[int, int]:
    """A full 360-degree scan from a pose, as sensors.Lidar.scan() returns."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for bearing in range(360):
        if any(low <= bearing <= high for low, high in config.LIDAR_SELF_MASK):
            continue                                # Friday's own chassis
        hit = cast(pose, bearing)
        if hit is None:
            continue
        mm = hit * 1000.0 + rng.normal(0.0, noise_mm)
        if 0 < mm <= config.LIDAR_MAX_RANGE_MM:
            out[bearing] = int(mm)
    return out


def check(name: str, got: float, want: float, tol: float) -> bool:
    delta = abs((got - want + 180.0) % 360.0 - 180.0)
    ok = delta <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.1f}째, "
          f"want {want:.1f}째, off by {delta:.1f}째 (tolerance {tol}째)")
    return ok


def main() -> int:
    rng = np.random.default_rng(7)
    start = (0.0, 0.0)
    failures = 0

    print("recovering a known translation from two synthetic sweeps\n")
    for truth, distance in ((0.0, 0.25), (90.0, 0.25), (203.0, 0.25),
                            (317.0, 0.25), (45.0, 0.40), (270.0, 0.15)):
        rad = math.radians(truth)
        end = (start[0] + distance * math.cos(rad),
               start[1] + distance * math.sin(rad))
        before = points_from_scan(sweep(start, rng=rng))
        after = points_from_scan(sweep(end, rng=rng))
        got, score = match_translation(before, after, distance)
        # 25 cm at 5 cm cells with a 12 cm blur: a couple of degrees is the
        # resolution the geometry supports, not a slack tolerance.
        ok = check(f"moved {distance * 100:.0f} cm along {truth:.0f}째",
                   got, truth, 6.0)
        print(f"        score {score:.3f}")
        failures += not ok

    print("\na sweep matched against itself scores 1.0 by construction")
    here = points_from_scan(sweep(start, rng=rng))
    _, score = match_translation(here, here, 0.0, refine=False)
    ok = 0.95 <= score <= 1.05
    print(f"  {'PASS' if ok else 'FAIL'}  self-match score {score:.3f} "
          f"(want ~1.0)")
    failures += not ok

    print("\na WRONG answer must score clearly worse, or the threshold is "
          "meaningless")
    rad = math.radians(30.0)
    end = (start[0] + 0.25 * math.cos(rad), start[1] + 0.25 * math.sin(rad))
    before = points_from_scan(sweep(start, rng=rng))
    after = points_from_scan(sweep(end, rng=rng))
    right = match_translation(before, after, 0.25)[1]
    from slam.core import LikelihoodField, transform
    field = LikelihoodField(before)
    wrong_rad = math.radians(210.0)                  # 180 degrees out
    wrong = field.score(transform(after, 0.25 * math.cos(wrong_rad),
                                  0.25 * math.sin(wrong_rad), 0.0))
    print(f"        correct bearing scores {right:.3f}")
    print(f"        180째 out scores        {wrong:.3f}")
    ok = right > wrong * 1.3 and wrong < config.SLAM_MIN_SCORE < right
    print(f"  {'PASS' if ok else 'FAIL'}  SLAM_MIN_SCORE "
          f"({config.SLAM_MIN_SCORE}) separates them")
    failures += not ok

    print(f"\n{'all good' if not failures else str(failures) + ' failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
