#!/usr/bin/env python3
"""EXPERIMENT, not in run_all: key views for the way back's tracker (2026-10-10).

⚠ RESULT: way_back.KEY_VIEWS ships OFF. With the start beside a box, key views
raise "matched" 58% -> 100% but end 0.74 m out = dead reckoning's 0.73, where
the start's view alone ends 0.08-0.12 m out. Re-run this against REAL
recordings (FRIDAY_TRACK_RECORD) before switching them on.

The way back's tracker following her where the START cannot be seen.

Harsh: "do the proper fix". The 14:39 run among chairs matched 58 of 512 sweeps
against the start's own view (11%; 65-89% in emptier rooms): lost from ~1.4 m,
the mission's map 16 scans thin, no route home. Key views (way_back.KEY_VIEWS):
a sweep the start's view cannot place is matched against the nearest views she
took on the way, each one a confident match of its own.

Two rooms joined by a 0.8 m doorway. She starts in room A and drives through
into room B, which the start's view never saw. Checks:
  * start-only loses her in room B (the failure);
  * with key views she stays matched there, and ends nearer the truth than
    dead reckoning;
  * where the start's view CAN place her, key views change nothing.

    ./venv/bin/python rl/tests/test_key_views.py        (~1 min)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

HERE_T = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE_T.parent))
sys.path.insert(0, str(HERE_T.parent.parent))

import way_back as wb                                    # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    if not ok:
        fails += 1


CELL = 0.05
# Room A: x 0..3, y 0..3. Room B: x 3..7, y 0..3 -- one wall at x=3 with a
# doorway at y 1.1..1.9. Furniture in B so it is not featureless.
W, H = int(7.4 / CELL), int(3.4 / CELL)
occ = np.zeros((W, H), bool)


def wall(x0, y0, x1, y1):
    n = int(max(abs(x1 - x0), abs(y1 - y0)) / (CELL / 2)) + 1
    for k in range(n + 1):
        x, y = x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n
        occ[int(x / CELL), int(y / CELL)] = True


wall(0.1, 0.1, 7.1, 0.1); wall(0.1, 3.1, 7.1, 3.1)
wall(0.1, 0.1, 0.1, 3.1); wall(7.1, 0.1, 7.1, 3.1)
wall(3.1, 0.1, 3.1, 1.1); wall(3.1, 1.9, 3.1, 3.1)        # the doorway
for bx, by in ((4.2, 0.6), (5.6, 2.3), (6.3, 0.9), (4.8, 2.6)):   # boxes in B
    wall(bx, by, bx + 0.4, by); wall(bx + 0.4, by, bx + 0.4, by + 0.4)
    wall(bx + 0.4, by + 0.4, bx, by + 0.4); wall(bx, by + 0.4, bx, by)
wall(1.0, 2.2, 1.6, 2.2)                                  # a shelf in A
# THE START NEXT TO A BOX (14:39: she started ~25 cm from the box put down for
# the reverse floor test): most of the start's view is the box.
wall(1.45, 1.05, 1.45, 1.95); wall(1.45, 1.05, 1.85, 1.05)
wall(1.45, 1.95, 1.85, 1.95); wall(1.85, 1.05, 1.85, 1.95)


def sweep(pose, rng, max_r=6.0):
    out = {}
    for deg in range(360):
        a = pose[2] + math.radians(deg)
        dx, dy = math.cos(a), math.sin(a)
        r = 0.1
        while r < max_r:
            i, j = int((pose[0] + dx * r) / CELL), int((pose[1] + dy * r) / CELL)
            if not (0 <= i < W and 0 <= j < H):
                break
            if occ[i, j]:
                out[deg] = int(1000 * (r + rng.gauss(0, 0.01)))
                break
            r += 0.02
    return out


# A path: room A, through the door, round room B.
WAYPOINTS = [(1.2, 1.5), (1.2, 0.7), (2.4, 0.7), (3.1, 1.5), (3.9, 1.5),
             (4.6, 1.6), (5.2, 1.3), (5.8, 1.6), (6.4, 1.8)]


def drive(key_views, seed):
    wb.KEY_VIEWS = key_views
    rng = random.Random(seed)
    truth = [WAYPOINTS[0][0], WAYPOINTS[0][1], 0.0]
    start = tuple(truth)
    dead = [0.0, 0.0, 0.0]
    tr = wb.StartTracker(8.0)
    k = 0
    for wx, wy in WAYPOINTS[1:]:
        while math.hypot(wx - truth[0], wy - truth[1]) > 0.08:
            dt = 0.1
            want = math.atan2(wy - truth[1], wx - truth[0])
            err = wb._wrap(want - truth[2])
            om = max(-1.0, min(1.0, 2.0 * err))
            v = 0.25 if abs(err) < 0.4 else 0.0
            slip = rng.uniform(0.75, 1.0)                 # distance: timed, it slips
            truth[0] += v * slip * math.cos(truth[2]) * dt
            truth[1] += v * slip * math.sin(truth[2]) * dt
            truth[2] = wb._wrap(truth[2] + om * dt)
            dead[0] += v * math.cos(dead[2]) * dt
            dead[1] += v * math.sin(dead[2]) * dt
            dead[2] = wb._wrap(dead[2] + om * dt)         # the gyro: the turn she made
            if k % 3 == 0:
                tr.submit(sweep(truth, rng), tuple(dead), gyro_gaps=0)
                tr._fold(*tr._next_job())
            k += 1
    rel_true = wb._compose(wb._inverse(tuple(truth)), start)
    rel_est = wb._compose(wb._inverse(tuple(dead)), tr.home_rl())
    rel_dead = wb._compose(wb._inverse(tuple(dead)), (0.0, 0.0, 0.0))
    return (math.hypot(rel_est[0] - rel_true[0], rel_est[1] - rel_true[1]),
            math.hypot(rel_dead[0] - rel_true[0], rel_dead[1] - rel_true[1]),
            tr.matched, tr.updates, tr.mapper.kf_matches, len(tr.mapper.keyframes))


print("--- through a doorway into a room the start never saw ---")
results = {}
for seed in (1, 2):
    old = drive(False, seed)
    new = drive(True, seed)
    results[seed] = (old, new)
    print(f"  seed {seed}: start only {old[0]:.2f} m, matched {old[2]}/{old[3]} | "
          f"key views {new[0]:.2f} m, matched {new[2]}/{new[3]} "
          f"({new[4]} by key views, {new[5]} views) | dead reckoning {new[1]:.2f} m")
for seed, (old, new) in results.items():
    check(f"seed {seed}: with key views she stays matched in room B "
          f"(>= 80% of sweeps, and more than the start alone)",
          new[2] >= 0.8 * new[3] and new[2] > old[2], f"{new[2]}/{new[3]} vs {old[2]}/{old[3]}")
    check(f"seed {seed}: ...and the start is found nearer than dead reckoning puts it",
          new[0] < new[1], f"{new[0]:.2f} vs dead reckoning {new[1]:.2f}")
    check(f"seed {seed}: ...and no worse than the start's view alone (+0.10 m)",
          new[0] <= old[0] + 0.10, f"{new[0]:.2f} vs {old[0]:.2f}")

print("--- where the start's view CAN place her, nothing changes ---")
wb.KEY_VIEWS = True
m = wb._StartMatcher(sweep((1.2, 1.5, 0.0), random.Random(3)))
rng = random.Random(3)
for k in range(8):
    m.update(sweep((1.2, 1.5 - 0.04 * (k + 1), 0.0), rng), moved_m=0.0, turned_rad=0.0,
             turn_known=True)
check("in room A every sweep matches the START's view, never a key view",
      m.kf_matches == 0, f"{m.kf_matches} key-view matches")

wb.KEY_VIEWS = True
print("--- PASSED ---" if not fails else f"--- {fails} FAILED ---")
sys.exit(1 if fails else 0)
