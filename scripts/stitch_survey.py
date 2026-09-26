#!/usr/bin/env python3
"""Stitch a survey's standing-still spot maps into one map. Moves nothing.

    ./venv/bin/python scripts/stitch_survey.py state/map-permanent/surveys/<time>

Spot 0 is the frame. Each next spot is placed by the measured drive
(moves.json), refined by fitting its walls onto the map so far NEAR that
guess (never a whole-map search: this room has look-alikes), and merged with
map_growth.merge only if at least MIN_COVERAGE of its wall points land within
NEAR_M of a wall already there. Writes stitched.npz, PNGs and
stitch_report.txt into the folder. It does NOT replace state/map.npz.

2026-09-26, first survey: spots 0, 2, 3, 4 merged (58-74% on, fits moved
0-12 cm from the drive), spot 1 left out (32%); walls 1.61 m2 against the
09-19 map's 0.82; 5/8 recorded start sweeps placed surely against 4/8.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
from PIL import Image                                # noqa: E402
from scipy.ndimage import distance_transform_edt     # noqa: E402

import map_growth                                    # noqa: E402
from slam.core import LikelihoodField, match_pose    # noqa: E402
from slam.grid import OccupancyGrid                  # noqa: E402

MIN_COVERAGE = 0.45
NEAR_M = 0.08


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def compose(pose, fwd: float, turn: float):
    x, y, th = pose
    mid = th + turn / 2.0
    return (x + fwd * math.cos(mid), y + fwd * math.sin(mid), wrap(th + turn))


def to_world(points: np.ndarray, pose) -> np.ndarray:
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return np.column_stack((points[:, 0] * c - points[:, 1] * s + x,
                            points[:, 0] * s + points[:, 1] * c + y))


def coverage(acc, points: np.ndarray, pose) -> float:
    """Fraction of `points` (placed at `pose`) within NEAR_M of a wall of acc."""
    occ = acc.occupied_mask()
    dist = distance_transform_edt(~occ) * acc.cell
    w = to_world(points, pose)
    ix, iy = acc.cells_of(w)
    inside = (ix >= 0) & (iy >= 0) & (ix < occ.shape[0]) & (iy < occ.shape[1])
    near = np.zeros(len(w), bool)
    near[inside] = dist[ix[inside], iy[inside]] <= NEAR_M
    return float(near.mean()) if len(w) else 0.0


def png(grid, path: Path) -> None:
    img = np.full(grid.occupied_mask().shape, 128, np.uint8)
    img[grid.free_mask()] = 255
    img[grid.occupied_mask()] = 0
    Image.fromarray(np.flipud(img.T)).resize(
        (img.shape[0] * 3, img.shape[1] * 3), Image.NEAREST).save(path)


def main(folder: Path) -> int:
    moves = json.loads((folder / "moves.json").read_text())
    spots = []
    while (folder / f"spot_{len(spots)}.npz").exists():
        spots.append(OccupancyGrid.load(folder / f"spot_{len(spots)}.npz"))
    if not spots:
        print(f"no spot maps in {folder}")
        return 1

    acc = spots[0]
    pose = (0.0, 0.0, 0.0)
    report = [f"spot 0: the frame, {int(acc.occupied_mask().sum())} wall cells"]
    png(spots[0], folder / "spot_0.png")
    for k in range(1, len(spots)):
        for rec in moves[k - 1] if k - 1 < len(moves) else []:
            est = rec.get("est")
            fwd, turn = ((est[0], est[1]) if est and est[2] >= 0.2
                         else rec["asked"])
            pose = compose(pose, fwd, turn)
        guess = pose
        pts = map_growth.trip_points(spots[k])
        field = LikelihoodField.from_grid(acc)
        best = (coverage(acc, pts, guess), guess)
        for dth in (0, 8, -8, 16, -16):
            seed = (guess[0], guess[1], wrap(guess[2] + math.radians(dth)))
            p, _s, _sh = match_pose(field, pts, seed, prior_m=0.25,
                                    prior_deg=6.0)
            cov = coverage(acc, pts, p)
            if cov > best[0]:
                best = (cov, tuple(float(v) for v in p))
        cov, fitted = best
        moved = math.hypot(fitted[0] - guess[0], fitted[1] - guess[1])
        turned = math.degrees(abs(wrap(fitted[2] - guess[2])))
        if cov >= MIN_COVERAGE:
            acc = map_growth.merge(acc, spots[k], fitted)
            pose = fitted
            report.append(f"spot {k}: merged, {cov:.0%} of its walls on the "
                          f"map's (fit moved it {moved * 100:.0f} cm, "
                          f"{turned:.0f} deg from the drive)")
        else:
            report.append(f"spot {k}: LEFT OUT, only {cov:.0%} of its walls fit")
        png(spots[k], folder / f"spot_{k}.png")
    acc.save(folder / "stitched.npz")
    png(acc, folder / "stitched.png")
    c2 = acc.cell ** 2
    report.append(f"stitched: free {acc.free_mask().sum() * c2:.1f} m2, walls "
                  f"{acc.occupied_mask().sum() * c2:.2f} m2")
    (folder / "stitch_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
