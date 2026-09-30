#!/usr/bin/env python3
"""File 21 step 4: frontiers -- detect, cluster, reach, score, blacklist, done.

    ./venv/bin/python scripts/test_file21_step4.py

Network blocked, no hardware. Maps are tests/exploration_mission/make_maps.py,
turned into real slam OccupancyGrids and passed through File 14's own
despeckle (adapters.SlamAdapter.planner_grid), so "reachable" is tested
against the planner's rule. The wavefront is checked cell for cell against a
plain BFS. Timings are measured on this Pi.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import frontiers as fr         # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402
import make_maps as mm                                  # noqa: E402

INFLATE = config.NAV_INFLATE_M
SLAM = ad.SlamAdapter()
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def planner(m: mm.Map) -> gu.Grid:
    """The synthetic map as File 14's planner would see it."""
    return SLAM.planner_grid(mm.to_occupancy_grid(m.grid))


def find(m: mm.Map, **kw) -> fr.Result:
    params = kw.pop("params", None) or fr.Params(**{
        k: kw.pop(k) for k in list(kw) if k in fr.Params.__dataclass_fields__})
    return fr.find(planner(m), kw.pop("pose", m.pose), INFLATE, params, **kw)


def bfs(trav: np.ndarray, start) -> np.ndarray:
    d = np.full(trav.shape, np.inf, np.float32)
    d[start] = 0
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                u, v = x + dx, y + dy
                if (dx or dy) and 0 <= u < trav.shape[0] and 0 <= v < trav.shape[1] \
                        and trav[u, v] and not np.isfinite(d[u, v]):
                    d[u, v] = d[x, y] + 1
                    q.append((u, v))
    return d


def same_field(a: np.ndarray, b: np.ndarray) -> bool:
    fa, fb = np.isfinite(a), np.isfinite(b)
    return bool(np.array_equal(fa, fb) and np.array_equal(a[fa], b[fb]))


# =============================================================================
print("--- detection on the spec's maps ---")

r = find(mm.closed_room())
check("closed room: no frontier at all", not r.frontiers and not r.remaining)
r = find(mm.three_rooms())
check("three explored rooms: no frontier", not r.frontiers)
r = find(mm.three_rooms((True, True, False)))
f = r.best
check("room C unexplored: exactly one frontier, at the B->C door",
      len(r.frontiers) == 1 and f is not None
      and abs(f.centroid[0] - 7.95) < 0.15 and abs(f.centroid[1] - 3.0) < 0.15,
      str([x.as_dict() for x in r.frontiers]))
check("...a 0.8 m door is a 0.8 m frontier", f and abs(f.length_m - 0.8) <= 0.2,
      str(f and f.length_m))
check("...reachable from room A through the A->B door, ~5.5 m of path",
      f and f.reachable and 5.0 <= f.path_dist_m <= 6.5, str(f and f.path_dist_m))
check("...its goal is a reached cell within a coarse cell of the frontier",
      f and np.isfinite(r.dist_cells[f.goal_cell])
      and math.hypot(f.goal[0] - f.centroid[0], f.goal[1] - f.centroid[1]) <= 0.45)
r = find(mm.three_rooms((True, False, False)))
check("rooms B and C unexplored: one frontier, at the A->B door (C is not "
      "known to exist yet)", len(r.frontiers) == 1
      and abs(r.best.centroid[0] - 4.45) < 0.15)
r = find(mm.open_door())
check("open door on an unknown corridor: one reachable frontier, ~0.8 m",
      len(r.remaining) == 1 and abs(r.best.length_m - 0.8) <= 0.2
      and abs(r.best.centroid[0] - 4.95) < 0.15)
r = find(mm.gap_frontier())
check("unknown floor seen through a 0.1 m slit: a frontier exists...",
      len(r.frontiers) >= 1)
check("...but it is unreachable, so nothing REMAINS to explore",
      not r.remaining and all(not x.reachable for x in r.frontiers))
check("...and it is ignored, not deleted (still listed, score -inf, no goal)",
      r.frontiers[0].goal is None and r.frontiers[0].score == -math.inf)


# =============================================================================
print("--- speckle (beam gaps, wall specks, pockets) ---")

sp = mm.speckle()
r = find(sp)
check("a speckled closed room has nothing left to explore",
      not r.frontiers and not r.remaining, str([x.as_dict() for x in r.frontiers]))
clean_room = find(mm.closed_room())
check("File 14's despeckle kept the floor drivable (within 10% of the clean room)",
      int(r.traversable.sum()) >= 0.9 * int(clean_room.traversable.sum()),
      f"{int(r.traversable.sum())} vs {int(clean_room.traversable.sum())}")
raw = fr.find(sp.grid, sp.pose, INFLATE)                 # NOT despeckled
check("...without it, 12 one-cell specks inflate into phantom walls",
      int(raw.traversable.sum()) < 0.8 * int(clean_room.traversable.sum()),
      f"{int(raw.traversable.sum())} vs {int(clean_room.traversable.sum())}")
no_rule = find(sp, min_unknown_m2=0.0)
check("without the pocket rule the 0.2 m pockets are frontiers for ever",
      len(no_rule.remaining) == 2, str(len(no_rule.remaining)))
tiny = find(sp, min_unknown_m2=0.0, min_frontier_size_m=5.0)
check("min_frontier_size_m drops clusters shorter than it", not tiny.frontiers)
edge = gu.Grid(np.full((40, 40), gu.FREE, np.int8), 0.10, (0.0, 0.0))
edge.occ[0:2, 18:21] = gu.UNKNOWN                        # 0.06 m2, but at the edge
check("unknown that runs off the map's edge is never 'a pocket'",
      bool(fr.real_unknown(edge, 0.15).any()))
edge.occ[20:22, 20:23] = gu.UNKNOWN                      # same size, mid-floor
ru = fr.real_unknown(edge, 0.15)
check("...the same patch in the middle of the floor is",
      not ru[20:22, 20:23].any() and ru[0:2, 18:21].all())


# =============================================================================
print("--- the pieces ---")

t = gu.Grid(np.array([[0, 0, 0, 0], [0, 0, -1, -1], [0, 100, -1, -1]], np.int8),
            0.10, (0, 0))
fm = fr.frontier_mask(t)
want = (t.occ == 0) & np.array([[0, 1, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0]], bool)
check("frontier cell = FREE with an UNKNOWN 8-neighbour (walls are not)",
      np.array_equal(fm, want), str(fm.astype(int)))
fine = np.zeros((5, 4), bool)
fine[0, 0] = fine[4, 3] = True
check("pool_any: a coarse cell is traversable if any fine cell is",
      fr.pool_any(fine, 2).tolist() == [[True, False], [False, False],
                                        [False, True]])
house = find(mm.three_rooms())
bx, by = house.coarse.world_to_cell(6.0, 3.0)            # middle of room B
check("a 0.8 m door (0.2 m of planner-passable gap) stays open at 10 cm",
      np.isfinite(house.dist_cells[bx, by]))
for name, m in (("three rooms", mm.three_rooms()), ("gap", mm.gap_frontier()),
                ("big house", mm.big_house())):
    rr = find(m)
    check(f"wavefront == BFS, cell for cell ({name}, "
          f"{int(np.isfinite(rr.dist_cells).sum())} cells)",
          same_field(rr.dist_cells, bfs(rr.traversable, rr.start_cell)))
maze = np.random.default_rng(0).random((150, 150)) > 0.35
st = tuple(int(v) for v in np.argwhere(maze)[0])
check("wavefront == BFS on a random maze, unreachable cells inf",
      same_field(fr.distance_field(maze, st), bfs(maze, st))
      and not np.isfinite(fr.distance_field(maze, st)).all())
check("a start that is not traversable reaches nothing",
      not np.isfinite(fr.distance_field(maze, tuple(
          int(v) for v in np.argwhere(~maze)[0]))).any())
room = find(mm.closed_room(), pose=(1.25, 2.5, 0.0))     # 15 cm from the west wall
check("standing inside the wall's clearance margin: snapped to the nearest "
      "traversable cell, the room is still reachable",
      room.start_cell is not None and int(np.isfinite(room.dist_cells).sum()) > 500)
lost = find(mm.closed_room(), pose=(5.7, 4.7, 0.0))      # out in the unknown
check("standing in unknown space far from any floor: nothing reachable",
      lost.start_cell is None and not np.isfinite(lost.dist_cells).any())
off = find(mm.closed_room(), pose=(-30.0, 50.0, 0.0))
check("a pose off the map does not crash", off.start_cell is None and not off.remaining)


# =============================================================================
print("--- scoring (spec 4.4) ---")


def two_doors(east_width=0.8) -> mm.Map:
    g = mm.blank(9, 6)
    mm.room(g, 1, 1, 7, 5)
    mm.rect(g, 1.0, 3.0 - 0.4, 1.0 + mm.WALL, 3.0 + 0.4, gu.FREE)           # west door
    mm.rect(g, 7 - mm.WALL, 3.0 - east_width / 2, 7.0, 3.0 + east_width / 2,
            gu.FREE)                                                      # east door
    return mm.Map(g, (4.0, 3.0, 0.0))


def side(f: fr.Frontier) -> str:
    return "east" if f.centroid[0] > 4 else "west"


r = find(two_doors(), pose=(2.0, 3.0, 0.0))
check("two frontiers found", len(r.remaining) == 2)
check("default weights: distance dominates, the NEAR door wins",
      side(r.best) == "west", str([x.as_dict() for x in r.remaining]))
r = find(two_doors(east_width=1.6), pose=(2.0, 3.0, 0.0), potential_scale=0.0,
         heading_scale=0.0)
check("gain only: the WIDER frontier wins", side(r.best) == "east"
      and r.best.length_m > 1.3, str([x.as_dict() for x in r.remaining]))
r = find(two_doors(), pose=(4.0, 3.0, 0.0))
check("equidistant, facing east: heading picks the east door", side(r.best) == "east")
r = find(two_doors(), pose=(4.0, 3.0, math.pi))
check("...facing west: the west door", side(r.best) == "west")
east_goal = [x for x in r.frontiers if side(x) == "east"][0].goal
west_goal = [x for x in r.frontiers if side(x) == "west"][0].goal
crumbs = ([(west_goal[0] + 0.1, 3.0, 0.0)] + [(4.0, 3.0, 1.0)]
          + [(west_goal[0] + 0.2, 3.0, 2.0)] + [(4.0, 3.0, 3.0)])
r = find(two_doors(), pose=(4.0, 3.0, math.pi), crumbs=crumbs)
wf = [x for x in r.frontiers if side(x) == "west"][0]
check("the trail passed the west door twice: revisits = 2", wf.revisits == 2)
check("...so, although she faces it, the east door wins", side(r.best) == "east")
check("revisits count PASSES, not crumbs inside the disc",
      fr.count_revisits((0, 0), [(0, 0, 0), (0.1, 0, 1), (0.2, 0, 2)]) == 1
      and fr.count_revisits((0, 0), [(0, 0, 0), (3, 0, 1), (0.1, 0, 2)]) == 2
      and fr.count_revisits((0, 0), []) == 0)
r = find(two_doors(), pose=(2.0, 3.0, 0.0), gain_scale=0.0, potential_scale=0.0,
         heading_scale=0.0)
check("a tie on score goes to the nearer one",
      r.remaining[0].score == r.remaining[1].score and side(r.best) == "west")
f0 = r.best
manual = (1.0 * f0.length_m - 3.0 * f0.path_dist_m + 0.5 * math.cos(
    math.atan2(f0.goal[1] - 3.0, f0.goal[0] - 2.0) - 0.0) - 1.0 * 0)
r2 = find(two_doors(), pose=(2.0, 3.0, 0.0))
check("score = gain*length - potential*path + heading*cos(dtheta) - revisit*n",
      abs(r2.best.score - manual) < 1e-6, f"{r2.best.score} vs {manual}")
check("frontiers serialise for the dashboard",
      isinstance(json.dumps([x.as_dict() for x in r2.frontiers]), str))
check("reachable frontier metres (curiosity's input)",
      abs(r2.reachable_length_m() - sum(x.length_m for x in r2.remaining)) < 1e-9
      and r2.reachable_length_m() > 1.0)


# =============================================================================
print("--- blacklist + completion (spec 4.4) ---")

bl = fr.Blacklist(after_fails=2, radius_m=0.5)
r = find(two_doors(), pose=(2.0, 3.0, 0.0))
west = r.best
check("one failure does not blacklist", bl.fail(west.centroid) is False
      and not bl.blocks(west.centroid))
check("the second does", bl.fail(west.centroid) is True and bl.blocks(west.centroid))
check("the disc is 0.5 m: a centroid 0.4 m away is the same frontier, 0.6 m is not",
      bl.blocks((west.centroid[0] + 0.4, west.centroid[1]))
      and not bl.blocks((west.centroid[0] + 0.6, west.centroid[1])))
r = find(two_doors(), pose=(2.0, 3.0, 0.0), blacklist=bl)
check("a blacklisted frontier never counts as remaining",
      len(r.remaining) == 1 and side(r.best) == "east"
      and [x for x in r.frontiers if side(x) == "west"][0].blacklisted)
bl2 = fr.Blacklist(2, 0.5, entries=json.loads(json.dumps(bl.to_list())))
check("the blacklist survives the checkpoint (JSON round trip)",
      bl2.blocks(west.centroid) and len(bl2.entries) == 1)
east = r.best
bl2.fail(east.centroid)
bl2.fail(east.centroid)
r = find(two_doors(), pose=(2.0, 3.0, 0.0), blacklist=bl2)
check("with every frontier blacklisted nothing remains", not r.remaining
      and len(r.frontiers) == 2)

done = fr.Completion(done_after=3)
check("two empty checks are not enough",
      [done.update(0, True), done.update(0, True)] == [False, False])
check("the third consecutive one is", done.update(0, True) is True)
done = fr.Completion(3)
seq = [done.update(0, True), done.update(0, True), done.update(1, True),
       done.update(0, True), done.update(0, True)]
check("a frontier appearing starts the count again", seq == [False] * 5)
done = fr.Completion(3)
seq = [done.update(0, True), done.update(0, True), done.update(0, False),
       done.update(0, True), done.update(0, True), done.update(0, True)]
check("a check while NOT localized proves nothing and restarts the count",
      seq == [False, False, False, False, False, True])


# =============================================================================
print("--- config + the real saved map ---")

p = fr.Params.from_config(mcfg.load())
check("Params come from config.EXPLORATION_MISSION",
      (p.seg_resolution_m, p.min_frontier_size_m, p.potential_scale,
       p.min_unknown_m2) == (0.10, 0.30, 3.0, 0.15))
real = SLAM.planner_grid()
if real is None:
    check("real map present", False, "state/map.npz missing")
else:
    trav = gu.traversable(real, INFLATE)
    ix, iy = np.argwhere(trav)[len(np.argwhere(trav)) // 2]
    px, py = real.cell_to_world(ix, iy)
    rr = fr.find(real, (px, py, 0.0), INFLATE, p)
    print(f"        real map {real.shape} @ {real.resolution} m: "
          f"{len(rr.frontiers)} clusters, {len(rr.remaining)} reachable "
          f"({rr.reachable_length_m():.1f} m), {rr.ms['total']:.1f} ms")
    check("runs on the real saved map (read only)", rr.start_cell is not None)


# =============================================================================
print("--- timing on this Pi (spec: < 150 ms for a 20 x 20 m map) ---")

open20 = mm.Map(mm.blank(22, 22), (11.0, 11.0, 0.0))
mm.room(open20.grid, 1, 1, 21, 21)
mm.rect(open20.grid, 21 - mm.WALL, 10.6, 21.0, 11.4, gu.FREE)
for name, m in (("20 m house, 16 rooms", mm.big_house()),
                ("one open 20 m room (worst case)", open20)):
    g = planner(m)
    fr.find(g, m.pose, INFLATE)
    times = []
    for _ in range(11):
        times.append(fr.find(g, m.pose, INFLATE).ms["total"])
    med, worst = statistics.median(times), max(times)
    last = fr.find(g, m.pose, INFLATE).ms
    print(f"        {name}: median {med:.1f} ms, worst {worst:.1f} ms "
          f"(coarse {last['coarse']:.1f}, detect {last['detect']:.1f}, "
          f"distance {last['distance']:.1f}, score {last['score']:.1f}); "
          f"load {os.getloadavg()[0]:.1f}")
    check(f"{name}: median under 150 ms", med < 150.0)
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
