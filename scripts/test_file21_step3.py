#!/usr/bin/env python3
"""File 21 step 3: the start signature (home_anchor) and the trail (breadcrumbs).

    ./venv/bin/python scripts/test_file21_step3.py

Network blocked, no hardware. Sweeps are ray-cast: an irregular synthetic room,
and the REAL saved map (state/map.npz, read only) for the look-alike case.
Timings are measured on this Pi.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import breadcrumbs as bc       # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402
from exploration_mission import home_anchor as ha       # noqa: E402
import synth                                            # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


WALLS = synth.room_walls()
HERE = (1.2, 1.4)
rng = np.random.default_rng(3)


def sig_at(x, y, th_deg, n=5, noise=0.01, drop=0.03, seed=0):
    r = np.random.default_rng(seed)
    scans = [synth.ray_cast(WALLS, (x, y, math.radians(th_deg)), noise_m=noise,
                            drop=drop, rng=r) for _ in range(n)]
    return ha.signature(scans, 8.0)


# =============================================================================
print("--- frames ---")

rb = ha.raw_to_robot({51.8: 1000, 141.8: 2000, 10: 500}, 51.8)
check("raw 51.8 deg (the measured forward bearing) is robot 0 = ahead",
      rb.get(0.0) == 1000 or rb.get(360.0) == 1000, str(rb))
check("raw 141.8 is robot 90 = left (counter-clockwise)",
      abs([k for k, v in rb.items() if v == 2000][0] - 90.0) < 1e-9)
check("raw 10 wraps to robot 318.2", abs([k for k, v in rb.items() if v == 500][0]
                                          - 318.2) < 1e-9)
b = ha.scan_to_bins({0.4: 1500, 359.6: 1400, 90: 2500, 91: 0, 180: 9000}, 8.0)
check("scan_to_bins: rounding, the nearer return wins a shared bin",
      abs(b[0] - 1.4) < 1e-6 and abs(b[90] - 2.5) < 1e-6)
check("scan_to_bins: no return is NaN, far is clipped to max range",
      math.isnan(b[91]) and abs(b[180] - 8.0) < 1e-6)
check("the adapters use the same scan_to_bins", ad.scan_to_bins is ha.scan_to_bins)


# =============================================================================
print("--- signature + similarity (spec 4.9) ---")

ref = sig_at(*HERE, 17.0, seed=1)
check("a signature is 360 float32 bins", ref.shape == (360,) and ref.dtype == np.float32)
check("most bins see a wall in a closed room",
      int(np.isfinite(ref).sum()) >= 340, str(int(np.isfinite(ref).sum())))
for turn in (37, -37, 90, 170, -170, 180):
    cur = sig_at(*HERE, 17.0 + turn, seed=2)
    s = ha.similarity(cur, ref)
    err = (s.shift_deg - turn + 180) % 360 - 180
    check(f"turned {turn:+d} deg in place -> match, shift {turn:+d} +/- 1",
          s.match and abs(err) <= 1.0, str(s.as_dict()))
same = ha.similarity(ref, ref)
check("identical -> MAD 0, shift 0, full overlap",
      same.mad_m == 0 and same.shift_deg == 0 and same.overlap > 0.9)
moved = ha.similarity(sig_at(HERE[0] + 0.1, HERE[1], 17.0, seed=3), ref)
check("moved 0.1 m -> still a match", moved.match, str(moved.as_dict()))
moved = ha.similarity(sig_at(HERE[0] + 0.07, HERE[1] - 0.07, 60.0, seed=4), ref)
check("moved 0.1 m AND turned 43 deg -> match, shift ~43",
      moved.match and abs(moved.shift_deg - 43) <= 2, str(moved.as_dict()))
far = ha.similarity(sig_at(HERE[0] + 1.0, HERE[1], 17.0, seed=5), ref)
check("moved 1 m -> no match", not far.match, str(far.as_dict()))
far = ha.similarity(sig_at(3.0, 3.2, 17.0, seed=6), ref)
check("another spot in the room -> no match", not far.match, str(far.as_dict()))
half = sig_at(*HERE, 17.0, seed=7).copy()
half[:180] = np.nan
s = ha.similarity(half, ref)
check("half the bins missing -> overlap under 0.6 -> no match (MAD inf)",
      not s.match and s.overlap < 0.6 and math.isinf(s.mad_m), str(s.as_dict()))
most = sig_at(*HERE, 17.0, seed=8).copy()
most[:100] = np.nan
s = ha.similarity(most, ref)
check("100 bins missing (72% overlap) -> still a match", s.match and s.overlap >= 0.6,
      str(s.as_dict()))
check("an empty signature never matches",
      not ha.similarity(np.full(360, np.nan, np.float32), ref).match)
check("min_overlap and match_mad_m are honoured",
      not ha.similarity(sig_at(HERE[0] + 0.1, HERE[1], 17.0, seed=9), ref,
                        match_mad_m=0.001).match)

scans = [synth.ray_cast(WALLS, (*HERE, 0.3)) for _ in range(4)]
outlier = {k: 300 for k in range(360)}                      # someone at the dome
med = ha.signature(scans + [outlier], 8.0)
check("the median ignores one bad sweep out of five",
      float(np.nanmax(np.abs(med - ha.signature(scans, 8.0)))) < 1e-6)
check("a bin no sweep saw stays NaN (no return, never 'far')",
      math.isnan(ha.signature([{0: 1000}], 8.0)[5]))
check("the heading sanity check (spec: within 20 deg)",
      ha.heading_agrees(37.0, math.radians(30)) and
      not ha.heading_agrees(37.0, math.radians(0))
      and ha.heading_agrees(179.0, math.radians(-175)))


# =============================================================================
print("--- capture: pose mean, circular theta ---")

x, y, th = ha.mean_pose([(1, 2, math.radians(179)), (1.2, 2.2, math.radians(-179))])
check("circular mean of +179 and -179 is 180, not 0",
      abs(abs(math.degrees(th)) - 180.0) < 1e-6 and abs(x - 1.1) < 1e-9
      and abs(y - 2.1) < 1e-9, str(math.degrees(th)))
x, y, th = ha.mean_pose([(0, 0, math.radians(a)) for a in (-10, 0, 10, 20)])
check("circular mean of -10..20 is 5", abs(math.degrees(th) - 5.0) < 1e-6)
anchor = ha.capture(scans, [(1.2, 1.4, 0.3)] * 4)
rec = anchor.as_record()
check("capture -> an Anchor with the signature, storable (NaN as null)",
      anchor.scans == 4 and len(rec["signature"]) == 360
      and all(v is None or isinstance(v, float) for v in rec["signature"]))
try:
    ha.capture([], [])
    refused = False
except ValueError:
    refused = True
check("capture refuses zero sweeps", refused)


# =============================================================================
print("--- the real saved map (look-alike room, read only) ---")

try:
    from slam.grid import OccupancyGrid
    og = OccupancyGrid.load(config.SLAM_MAP_FILE)
    real = gu.from_occupancy_grid(og)
    occ = real.occupied()
    trav = gu.traversable(real, config.NAV_INFLATE_M)
    cells = np.argwhere(trav)
    spots = cells[rng.choice(len(cells), size=min(12, len(cells)), replace=False)]
    results, overlaps = [], []
    for ix, iy in spots:
        px, py = real.cell_to_world(ix, iy)
        r0 = ha.signature([synth.grid_ray_cast(occ, real.origin, real.resolution,
                                               (px, py, 0.4))], 8.0)
        overlaps.append(float(np.isfinite(r0).sum()) / 360.0)
        r90 = ha.signature([synth.grid_ray_cast(occ, real.origin, real.resolution,
                                                (px, py, 0.4 + math.pi / 2))], 8.0)
        s90 = ha.similarity(r90, r0)
        # 1 m away, still on free floor, if there is such a spot
        d = np.hypot(cells[:, 0] - ix, cells[:, 1] - iy) * real.resolution
        cand = cells[(d > 0.95) & (d < 1.05)]
        s1 = None
        if len(cand):
            qx, qy = real.cell_to_world(*cand[0])
            r1 = ha.signature([synth.grid_ray_cast(occ, real.origin, real.resolution,
                                                   (qx, qy, 0.4))], 8.0)
            s1 = ha.similarity(r1, r0)
        results.append((s90, s1))
    seeing = [(s90, s1, r) for (s90, s1), r in zip(results, overlaps)]
    enough = [s90 for s90, _, r in seeing if r >= 0.6]
    too_few = [s90 for s90, _, r in seeing if r < 0.6]
    ok90 = sum(1 for s in enough if s.match and abs(s.shift_deg - 90) <= 1)
    far_pairs = [s1 for _, s1 in results if s1 is not None]
    wrong = sum(1 for s1 in far_pairs if s1.match)
    print(f"        {len(results)} real spots ({len(too_few)} see a mapped wall in "
          f"< 60% of bearings): turned 90 -> {ok90}/{len(enough)} matched at 90; "
          f"1 m away -> {wrong}/{len(far_pairs)} false matches")
    check("real map: every spot seeing walls in >= 60% of bearings matches a "
          "90 deg turn at 90", ok90 == len(enough) and len(enough) >= 8)
    check("real map: a spot seeing walls in < 60% is refused (spec overlap rule), "
          "never guessed", all(not s.match for s in too_few))
    check("real map: 1 m away -> no false match at any spot", wrong == 0)
except FileNotFoundError:
    check("real map present", False, "state/map.npz missing")


# =============================================================================
print("--- breadcrumbs (spec 4.10) ---")

t = bc.Trail(0.25, 0.30)
for i in range(301):                                   # 3 m straight, 1 cm steps
    t.update(i * 0.01, 0.0, float(i))
xs = [c[0] for c in t.crumbs]
check("a crumb every 0.25 m: 13 crumbs over 3 m",
      len(t) == 13 and all(abs(b - a - 0.25) < 0.011 for a, b in zip(xs, xs[1:])),
      str([round(v, 2) for v in xs]))
check("pose jitter under the spacing lays nothing",
      not t.update(3.0 + 0.05, 0.02, 999.0) and len(t) == 13)
loop = bc.Trail(0.25, 0.30)
path = ([(i * 0.05, 0.0) for i in range(41)]            # east 2 m
        + [(2.0, i * 0.05) for i in range(1, 21)]       # north 1 m
        + [(2.0 - i * 0.05, 1.0) for i in range(1, 21)] # west 1 m
        + [(1.0, 1.0 - i * 0.05) for i in range(1, 21)] # south 1 m, back on the line
        + [(1.0 - i * 0.05, 0.0) for i in range(1, 21)])  # west 1 m, back to start
for k, (px, py) in enumerate(path):
    loop.update(px, py, float(k))
check("driving a loop back onto the trail cuts it", loop.cuts >= 1, str(loop.cuts))
check("...so the trail back is short: no crumb left out on the loop",
      all(c[1] < 0.3 for c in loop.crumbs), str([(round(c[0], 2), round(c[1], 2))
                                                  for c in loop.crumbs]))
check("the cut keeps the OLDEST crumb it came back to (the start stays first)",
      loop.crumbs[0][:2] == (0.0, 0.0))
check("tail() gives the checkpoint's last N", len(t.tail(5)) == 5
      and t.tail(5)[-1][0] == xs[-1])
t2 = bc.Trail(crumbs=t.to_list())
check("a trail restores from its checkpoint list", t2.crumbs == t.crumbs)
check("bresenham: endpoints included, 8-connected",
      bc.bresenham(0, 0, 3, 1) == [(0, 0), (1, 0), (2, 1), (3, 1)])

# An L-shaped corridor: 0.8 m wide, east then north. Walls are OCC.
BODY = 0.21                                            # File 14's NAV_ROBOT_RADIUS_M
g = gu.Grid(np.full((120, 120), gu.OCC, np.int8), 0.05, (0.0, 0.0))
g.occ[4:110, 10:26] = gu.FREE                          # x 0.2..5.5, y 0.5..1.3
g.occ[94:110, 10:119] = gu.FREE                        # x 4.7..5.5, y 0.5..5.95
g.occ[40:44, 60:70] = gu.OCC                           # clutter (irrelevant)
trail = []
for i in range(0, 21):                                 # east along y=0.9
    trail.append((0.7 + i * 0.22, 0.9, float(i)))
for i in range(1, 21):                                 # north along x=5.1
    trail.append((5.1, 0.9 + i * 0.22, float(20 + i)))
wp = bc.backtrack_path(trail, (5.1, 5.2), g, 0.30, body_radius_m=BODY)
check("backtrack starts at the crumb nearest her and ends at the start",
      wp[0] == (5.1, trail[-1][1]) and wp[-1] == (0.7, 0.9), str(wp[:2] + wp[-1:]))
check("straight legs collapse to their corner: 3 waypoints",
      len(wp) == 3 and abs(wp[1][0] - 5.1) < 1e-9, str(wp))
box = [(5.1 + dx, 3.1 + dy) for dx in (-0.05, 0, 0.05) for dy in (-0.05, 0, 0.05)]
wp2 = bc.backtrack_path(trail, (5.1, 5.2), g, 0.30, runtime_obstacles=box,
                        body_radius_m=BODY)
blocked_crumbs = [c for c in trail if abs(c[0] - 5.1) < 1e-9
                  and abs(c[1] - 3.1) < BODY + 0.05]
check("a crumb now inside an obstacle (File 14's runtime layer) is dropped",
      blocked_crumbs and all((c[0], c[1]) not in wp2 for c in blocked_crumbs),
      str(blocked_crumbs))
check("...and the line of sight past it is broken, so the path keeps "
      "waypoints on both sides", len(wp2) > len(wp), str(wp2))
wp3 = bc.backtrack_path(trail[:10], (2.0, 0.9), g, 0.30, body_radius_m=BODY)
check("from the middle of the trail: the nearest crumb, back to the start",
      wp3[0] == (trail[6][0], 0.9) and wp3[-1] == (0.7, 0.9) and len(wp3) == 2,
      str(wp3))
check("no trail -> no path", bc.backtrack_path([], (0, 0), g, 0.3) == [])
door = gu.Grid(np.full((60, 60), gu.FREE, np.int8), 0.05, (0.0, 0.0))
door.occ[30, 36] = gu.OCC            # a door jamb 6 cells (0.30 m) from the crumb
dtrail = [(0.5, 1.525, 0.0), (1.525, 1.525, 1.0), (2.5, 1.525, 2.0)]
kept_body = bc.backtrack_path(dtrail, (2.5, 1.525), door, 0.30, body_radius_m=BODY)
kept_infl = bc.backtrack_path(dtrail, (2.5, 1.525), door, 0.30)
check("a crumb beside a jamb (inside the 0.30 clearance disc, outside her "
      "0.21 body disc) is KEPT -- it is the doorway",
      (1.525, 1.525) in kept_body and (1.525, 1.525) not in kept_infl,
      f"{kept_body} vs {kept_infl}")


# =============================================================================
print("--- the LiDAR adapter (never a second reader) ---")

la = ad.LidarAdapter(sweep_fn=lambda: {51.8: 1234, 141.8: 2000}, forward_deg=51.8)
sw = la.sweep()
check("a raw sweep comes back in the robot frame",
      any(abs(k) < 1e-9 or abs(k - 360) < 1e-9 for k in sw) and 2000 in sw.values())
outside = ad.LidarAdapter()
check("outside friday-mcp, with no run publishing, it refuses (None) rather "
      "than open the LiDAR", outside.sweep() is None
      and "mcp_server.hardware" not in sys.modules)


class _Run:
    def live(self, max_age_s=None):
        return {"scan": {0: 1500, 90: 2500}}


check("with a run publishing, the run's robot-frame sweep is used",
      ad.LidarAdapter(run=_Run()).sweep() == {0.0: 1500, 90.0: 2500})


# =============================================================================
print("--- timing on this Pi (spec: similarity < 20 ms) ---")

cur = sig_at(*HERE, 50.0, seed=11)
for _ in range(5):
    ha.similarity(cur, ref)
times = []
for _ in range(50):
    t0 = time.perf_counter()
    ha.similarity(cur, ref)
    times.append((time.perf_counter() - t0) * 1000)
med_ms, worst = statistics.median(times), max(times)
print(f"        similarity: median {med_ms:.2f} ms, worst {worst:.2f} ms (50 runs)")
check("similarity median under 20 ms", med_ms < 20.0)
t0 = time.perf_counter()
ha.signature([synth.ray_cast(WALLS, (*HERE, 0.3)) for _ in range(20)], 8.0)
sig_ms = (time.perf_counter() - t0) * 1000
print(f"        20-sweep signature incl. ray casting: {sig_ms:.1f} ms")
bt = bc.Trail()
for i in range(2000):
    bt.update(math.cos(i / 50) * i * 0.002, math.sin(i / 50) * i * 0.002, float(i))
t0 = time.perf_counter()
for i in range(100):
    bt.update(5 + i * 0.3, 5.0, float(3000 + i))
upd_ms = (time.perf_counter() - t0) * 1000 / 100
print(f"        trail update with {len(bt)} crumbs: {upd_ms:.3f} ms")
check("a trail update stays well inside a 10 Hz tick (< 5 ms)", upd_ms < 5.0)
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
