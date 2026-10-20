#!/usr/bin/env python3
"""Test path_planner.py — file 14 steps 1-3. No hardware, no motors.

    scripts/test_path_planner.py

Two kinds of check, and both are needed.

SYNTHETIC, because only a map you built yourself can tell you that the wall
survived and the speck did not. A real map has no ground truth: you cannot
look at state/map.npz and say which of its 326 occupied cells are chair legs.

AGAINST THE REAL MAPS, because the synthetic grid cannot tell you whether the
filter matters. It turns out it matters enormously on one of them and barely
on the other, and that difference is the most useful thing this file prints.

⚠ THE READ-ONLY CHECK IS THE IMPORTANT ONE. File 14 may not change the
exploration system's map by so much as a cell, and the easiest way to break
that rule is a numpy view handed out where a copy was meant. `test_read_only`
asserts the caller's grid is byte-identical afterwards.

The other check worth knowing about is `test_astar_matches_dijkstra`, which
holds the new A* against `slam/plan.py`'s Dijkstra on the same map. They are
different algorithms answering the same question, so they must agree to the
millimetre -- and on the real maps they must agree about UNREACHABLE too,
which is the stricter half: both refuse routes that exist only through a
zero-width diagonal gap.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                           # noqa: E402
import path_planner as pp                               # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402
from slam.plan import (DistanceField, passable_mask,    # noqa: E402
                       path_to_moves)

PASS, FAIL = "  ok  ", " FAIL "
_failures = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _failures
    if not ok:
        _failures += 1
    print(f"[{PASS if ok else FAIL}] {label}{'  — ' + detail if detail else ''}")


def _grid(shape=(40, 40), cell=0.05) -> OccupancyGrid:
    """An all-free grid with nothing in it yet."""
    g = OccupancyGrid(cell)
    g.logodds = np.full(shape, -1.0, np.float32)
    g.origin = np.zeros(2, np.float64)
    g._empty = False
    return g


def test_specks() -> None:
    """A wall survives; singles, pairs and diagonal pairs do not."""
    print("\n— speck filtering —")
    g = _grid()
    occ = np.zeros(g.shape, bool)
    occ[10, 5:25] = True                              # a wall, 20 cells
    occ[20, 20] = occ[21, 20] = occ[21, 21] = True    # 3-cell L: NOT a speck
    occ[30, 30] = occ[38, 38] = True                  # two singles
    occ[35, 5] = occ[36, 6] = True                    # a diagonal pair
    occ[5, 35] = occ[5, 36] = True                    # an orthogonal pair
    g.logodds[occ] = 3.0

    mask, n = pp.speck_mask(g.occupied_mask())
    check("four components cleared", n == 4, f"got {n}")
    check("six cells cleared", int(mask.sum()) == 6, f"got {int(mask.sum())}")

    survived = g.occupied_mask() & ~mask
    check("the wall is intact", bool(survived[10, 5:25].all()))
    check("a 3-cell component survives", int(survived[20:22, 20:22].sum()) == 3)
    check("singles are gone", not survived[30, 30] and not survived[38, 38])
    check("a diagonal pair counts as one speck, and goes",
          not survived[35, 5] and not survived[36, 6])
    check("orthogonal pair goes", not survived[5, 35] and not survived[5, 36])

    _, n3 = pp.speck_mask(g.occupied_mask(), max_cells=3)
    check("max_cells=3 also takes the L", n3 == 5, f"got {n3}")


def test_cleared_speck_is_free() -> None:
    """Cleared means FREE — not unknown, which would still block planning."""
    print("\n— what a cleared speck becomes —")
    g = _grid((20, 20))
    g.logodds[10, 10] = 3.0
    clean, n, cells = pp.despeckle(g)
    check("one speck, one cell", n == 1 and cells == 1, f"got {n}/{cells}")
    check("reads as free afterwards", bool(clean.free_mask()[10, 10]))
    check("not unknown", not clean.unknown_mask()[10, 10])
    check("not occupied", not clean.occupied_mask()[10, 10])
    check("one more sweep of wall evidence would overrule it",
          clean.logodds[10, 10] > config.SLAM_LOGODDS_MIN,
          f"logodds {clean.logodds[10, 10]:.2f} vs floor "
          f"{config.SLAM_LOGODDS_MIN}")


def test_read_only() -> None:
    """The caller's grid must come back byte-for-byte unchanged."""
    print("\n— read-only —")
    g = _grid((30, 30))
    g.logodds[15, 15] = 3.0
    before = g.logodds.copy()

    clean, _, _ = pp.despeckle(g)
    check("despeckle does not touch the input",
          np.array_equal(g.logodds, before))
    check("it returns a different array object",
          clean.logodds is not g.logodds)

    result = pp.preprocess(g)
    check("preprocess does not touch the input",
          np.array_equal(g.logodds, before))
    check("the result's grid is not the input", result.grid is not g)

    result.grid.logodds[0, 0] = 99.0
    check("writing to the result cannot reach the input",
          np.array_equal(g.logodds, before))


def test_unknown_is_not_plannable() -> None:
    """Step 1's third rule: unknown is as unplannable as wall."""
    print("\n— unknown is not traversable —")
    g = _grid((40, 40))
    g.logodds[:, 20:] = 0.0                # right half: never looked at
    r = pp.preprocess(g)
    check("no unknown cell is plannable",
          not (r.plannable & r.grid.unknown_mask()).any())
    check("no occupied cell is plannable",
          not (r.plannable & r.grid.occupied_mask()).any())
    check("the known half is still plannable", r.plannable[:, :20].any())
    check("the unknown half is not", not r.plannable[:, 20:].any())


def test_inflation() -> None:
    """Inflation is slam/plan.py's, unchanged, at the configured radius."""
    print("\n— inflation —")
    g = _grid((60, 60))
    g.logodds[30, 30] = 3.0
    g.logodds[31, 30] = 3.0
    g.logodds[30, 31] = 3.0                # a 3-cell blob: survives filtering
    r = pp.preprocess(g)
    check("the obstacle survived the filter", r.occupied_cells == 3,
          f"got {r.occupied_cells}")
    check("inflate_m is config.NAV_INFLATE_M",
          abs(r.inflate_m - config.NAV_INFLATE_M) < 1e-9)

    cells = int(round(config.NAV_INFLATE_M / g.cell))
    check("a cell one radius away is blocked",
          not r.plannable[30 + cells - 1, 30])
    check("a cell well beyond the radius is free",
          bool(r.plannable[30 + cells + 3, 30]))


def test_order_matters() -> None:
    """Despeckle-then-inflate must beat inflate-alone, or step 1 is pointless."""
    print("\n— order: despeckle before inflate —")
    g = _grid((80, 80))
    for x, y in ((20, 20), (30, 45), (55, 25), (60, 60), (44, 12)):
        g.logodds[x, y] = 3.0              # five scattered single-cell specks

    naive = int(passable_mask(g).sum())
    r = pp.preprocess(g)
    check("five specks found", r.specks == 5, f"got {r.specks}")
    check("filtering recovers plannable space",
          r.plannable_cells > naive,
          f"{naive} -> {r.plannable_cells} "
          f"(+{100 * (r.plannable_cells - naive) / naive:.1f}%)")

    per_speck = (r.plannable_cells - naive) / 5
    check("each speck was costing far more than its own size",
          per_speck > 20, f"{per_speck:.0f} cells each")


def test_empty_map() -> None:
    print("\n— degenerate inputs —")
    r = pp.preprocess(OccupancyGrid())
    check("an empty map plans nothing and says why",
          r.plannable_cells == 0 and r.notes, str(r.notes))
    mask, n = pp.speck_mask(np.zeros((5, 5), bool))
    check("an empty mask is not a crash", n == 0 and not mask.any())
    g = _grid((10, 10))
    _, n0, _ = pp.despeckle(g, max_cells=0)
    check("max_cells=0 disables filtering", n0 == 0)


def test_extra_blocked() -> None:
    """A live return is evidence about now, and is never filtered away."""
    print("\n— extra_blocked —")
    g = _grid((60, 60))
    here = np.array([[1.5, 1.5]])          # one point: would be a speck on a map
    a = pp.preprocess(g)
    b = pp.preprocess(g, extra_blocked=here)
    check("a single live point still blocks",
          b.plannable_cells < a.plannable_cells,
          f"{a.plannable_cells} -> {b.plannable_cells}")


def test_real_maps() -> None:
    """Against the maps as they actually are on disk."""
    print("\n— the real maps —")
    for name in ("map.npz", "rl_map.npz"):
        path = config.ROOT / "state" / name
        if not path.exists():
            print(f"[ skip ] {name} — not on disk")
            continue
        g = OccupancyGrid.load(path)
        occ = int(g.occupied_mask().sum())
        naive = int(passable_mask(g).sum())
        r = pp.preprocess(g)
        share = 100 * r.speck_cells / max(occ, 1)
        print(f"  {name}: {g.shape[0]}x{g.shape[1]} cells, {occ} occupied")
        print(f"    {r.specks} specks ({r.speck_cells} cells, {share:.1f}% "
              f"of occupied), {r.occupied_cells} wall cells left")
        print(f"    plannable {naive} -> {r.plannable_cells} "
              f"({100 * (r.plannable_cells - naive) / max(naive, 1):+.1f}%)")
        for note in r.notes:
            print(f"    ⚠ {note}")
        check(f"{name}: the saved map is unchanged by planning",
              int(g.occupied_mask().sum()) == occ)
        if r.plannable.any():
            free = np.argwhere(r.plannable)[0]
            xy = (float(r.grid.origin[0] + (free[0] + 0.5) * r.grid.cell),
                  float(r.grid.origin[1] + (free[1] + 0.5) * r.grid.cell))
            DistanceField(r.grid, xy)
            check(f"{name}: a DistanceField builds on the result", True)


def _room() -> OccupancyGrid:
    """An open room with a block rising from one wall, 5 m x 5 m.

    The shortest route from one side to the other rounds the block's corner,
    and rounds it TIGHTLY unless something makes tight expensive. That is the
    whole point of this room.
    """
    g = _grid((100, 100))
    g.logodds[40:60, 0:50] = 3.0
    return g


def test_astar_matches_dijkstra() -> None:
    """With the gradient off, A* must return the true shortest path.

    Checked against `slam/plan.py`'s Dijkstra, which is the planner that
    drives the robot today. If these two ever disagree about a distance, one
    of them is wrong and the robot is being steered by it.
    """
    print("\n— A* optimality —")
    prep = pp.preprocess(_room())
    a, b = (10, 20), (90, 20)
    path = pp.astar(prep, a, b, gradient_weight=0.0)
    check("a route exists", bool(path))

    world = pp.cells_to_world(path, prep.grid)
    length = float(np.sum(np.hypot(np.diff(world[:, 0]), np.diff(world[:, 1]))))
    field = DistanceField(prep.grid, pp.cells_to_world([a], prep.grid)[0])
    optimum = field.distance_to(pp.cells_to_world([b], prep.grid)[0])
    check("A* length == Dijkstra optimum", abs(length - optimum) < 1e-6,
          f"A* {length:.3f} m vs Dijkstra {optimum:.3f} m")
    check("the path starts and ends where asked",
          path[0] == a and path[-1] == b)


def test_octile_is_admissible() -> None:
    """The heuristic must never exceed the real cost, or A* stops being exact."""
    print("\n— the octile heuristic —")
    cell = 0.05
    check("straight run is exact",
          abs(pp._octile(0, 0, 10, 0, cell) - 10 * cell) < 1e-12)
    check("pure diagonal is exact",
          abs(pp._octile(0, 0, 10, 10, cell) - 10 * math.sqrt(2) * cell) < 1e-12)
    check("zero at the goal", pp._octile(7, 7, 7, 7, cell) == 0.0)

    rng = np.random.default_rng(14)
    worst = 0.0
    for _ in range(400):
        ax, ay, bx, by = rng.integers(0, 60, 4)
        h = pp._octile(int(ax), int(ay), int(bx), int(by), cell)
        # Straight-line distance is a hard lower bound on any real route.
        straight = math.hypot(int(ax) - int(bx), int(ay) - int(by)) * cell
        worst = max(worst, straight - h)
    check("never below the straight-line bound", worst <= 1e-12,
          f"worst shortfall {worst:.2e}")


def test_gradient_prefers_clearance() -> None:
    """The soft cost must buy real clearance, at a length it is worth."""
    print("\n— the cost gradient —")
    prep = pp.preprocess(_room())
    clear = pp.clearance_m(prep.plannable, prep.cell_m)

    def run(weight):
        path = pp.astar(prep, (10, 20), (90, 20), gradient_weight=weight)
        world = pp.cells_to_world(path, prep.grid)
        length = float(np.sum(np.hypot(np.diff(world[:, 0]),
                                       np.diff(world[:, 1]))))
        return path, length, min(clear[c] for c in path)

    _, flat_len, flat_min = run(0.0)
    _, soft_len, soft_min = run(pp.NAV_GRADIENT_WEIGHT)
    check("the plain shortest path grazes the corner", flat_min <= 0.10,
          f"{flat_min:.3f} m clearance")
    check("the gradient pulls it off the wall", soft_min > flat_min * 3,
          f"{flat_min:.3f} m -> {soft_min:.3f} m")
    check("and does not pay much for it", soft_len < flat_len * 1.25,
          f"{flat_len:.2f} m -> {soft_len:.2f} m "
          f"(+{100 * (soft_len / flat_len - 1):.0f}%)")

    _, _, hard_min = run(pp.NAV_GRADIENT_WEIGHT * 3)
    check("clearance saturates at the gradient's reach, not the weight",
          abs(hard_min - soft_min) <= 2 * prep.cell_m,
          f"w=1 {soft_min:.3f} m vs w=3 {hard_min:.3f} m")


def test_gradient_still_takes_the_only_way() -> None:
    """A preference, never a refusal: a tight doorway is still used."""
    print("\n— the gradient is a preference, not a veto —")
    g = _grid((120, 80))
    g.logodds[60, :] = 3.0
    g.logodds[60, 31:49] = -1.0            # 0.9 m doorway
    prep = pp.preprocess(g)
    path = pp.astar(prep, (10, 40), (110, 40))
    check("it still goes through", bool(path) and any(c[0] == 60 for c in path))
    check("a 0.4 m doorway is correctly refused after inflation",
          not pp.astar(pp.preprocess(_narrow_door()), (10, 40), (110, 40)))


def _narrow_door() -> OccupancyGrid:
    g = _grid((120, 80))
    g.logodds[60, :] = 3.0
    g.logodds[60, 36:44] = -1.0            # 0.4 m: 0.30 m inflation seals it
    return g


def test_no_corner_cutting() -> None:
    """A diagonal step needs both orthogonal neighbours — in search AND in LOS."""
    print("\n— corner cutting —")
    plannable = np.ones((12, 12), bool)
    plannable[5, 5] = plannable[6, 6] = False
    check("LOS refuses the zero-width diagonal gap",
          not pp.line_of_sight(plannable, (5, 6), (6, 5)))
    other = np.ones((12, 12), bool)
    other[5, 6] = other[6, 5] = False
    check("LOS refuses it on the other diagonal too",
          not pp.line_of_sight(other, (5, 5), (6, 6)))
    check("an open diagonal is still visible",
          pp.line_of_sight(np.ones((12, 12), bool), (2, 2), (9, 9)))
    check("LOS refuses a straight wall",
          not pp.line_of_sight(plannable, (5, 4), (5, 8)))

    # And the search itself must not slip through.
    g = _grid((20, 20))
    g.logodds[9, 9] = g.logodds[10, 10] = 3.0
    prep = pp.preprocess(g, max_speck_cells=0)   # keep the two cells as wall
    path = pp.astar(prep, (10, 9), (9, 10))
    stepped = any(abs(path[i][0] - path[i + 1][0]) == 1
                  and abs(path[i][1] - path[i + 1][1]) == 1
                  and not prep.plannable[path[i][0], path[i + 1][1]]
                  for i in range(len(path) - 1))
    check("the search does not cut the corner either", not stepped)


def test_no_route_is_not_a_guess() -> None:
    print("\n— refusing —")
    g = _grid((60, 60))
    g.logodds[30, :] = 3.0                 # a wall with no door at all
    prep = pp.preprocess(g)
    check("a sealed room returns no route", pp.astar(prep, (10, 30), (50, 30)) == [])
    check("an off-map goal returns no route",
          pp.astar(prep, (10, 30), (500, 500)) == [])
    check("a work ceiling returns no route rather than blocking",
          pp.astar(prep, (10, 10), (10, 50), max_expansions=5) == [])
    check("start == goal is one cell", len(pp.astar(prep, (10, 10), (10, 10))) == 1)


def test_open_corridor() -> None:
    """2026-10-20: the way home's retrace hands File 14 her trail, and the
    margin can close a gap she has just driven through. open_corridor opens
    the strip under the trail -- never where she could not stand."""
    print("\n— a trail she drove opens a gap the margin closed —")
    g = _grid((80, 80))
    g.logodds[40, :] = 3.0                  # a wall across the room...
    g.logodds[40, 35:45] = -1.0             # ...with a 0.50 m door in it
    prep = pp.preprocess(g)
    a, b = (20, 40), (60, 40)
    check("the margin closes the 0.50 m door (no route without the trail)",
          pp.astar(prep, a, b) == [], f"inflate {prep.inflate_m:.2f} m")
    trail = pp.cells_to_world([a, (40, 40), b], prep.grid)
    before = prep.plannable.copy()
    opened = pp.open_corridor(prep, trail)
    check("with her trail through the door there is a route",
          len(pp.astar(opened, a, b)) >= 2)
    check("the caller's mask is untouched", np.array_equal(prep.plannable, before))
    check("...and it says what it opened", any("trail opened" in n for n in opened.notes))

    sealed = _grid((80, 80))
    sealed.logodds[40, :] = 3.0             # no door: the 'trail' is a bad pose
    sp = pp.preprocess(sealed)
    check("a trail THROUGH a solid wall opens nothing there",
          pp.astar(pp.open_corridor(sp, trail), a, b) == [])

    tight = _grid((80, 80))
    tight.logodds[40, :] = 3.0
    tight.logodds[40, 38:42] = -1.0         # 0.20 m: narrower than her body
    tp = pp.preprocess(tight)
    check("a gap narrower than her body stays closed, trail or not",
          pp.astar(pp.open_corridor(tp, trail), a, b) == [])
    check("one point is no trail: unchanged", pp.open_corridor(prep, trail[:1]) is prep)


def test_snapping() -> None:
    """Parked against a wall, the robot is 'inside' an obstacle. Plan anyway."""
    print("\n— snapping a start that inflation calls solid —")
    g = _grid((60, 60))
    g.logodds[30, :] = 3.0
    g.logodds[30, 20:40] = -1.0
    prep = pp.preprocess(g)
    against = (28, 10)                      # beside the wall, inside inflation
    check("that start really is unplannable", not prep.plannable[against])
    path = pp.astar(prep, against, (50, 30))
    check("it plans from the nearest legal cell", bool(path))
    check("but not from an absurd distance",
          pp.astar(prep, against, (50, 30), snap_m=0.0) == [])


def test_smoothing() -> None:
    """Fewer waypoints, same safety, same endpoints."""
    print("\n— smoothing —")
    prep = pp.preprocess(_room())
    raw = pp.astar(prep, (10, 20), (90, 20))
    sm = pp.smooth(raw, prep.plannable)
    check("the staircase is gone", len(sm) < len(raw) / 4,
          f"{len(raw)} cells -> {len(sm)} waypoints")
    check("endpoints are untouched", sm[0] == raw[0] and sm[-1] == raw[-1])
    check("every leg is still collision-free",
          all(pp.line_of_sight(prep.plannable, sm[i], sm[i + 1])
              for i in range(len(sm) - 1)))

    world = pp.cells_to_world(sm, prep.grid)
    raw_world = pp.cells_to_world(raw, prep.grid)
    slen = float(np.sum(np.hypot(np.diff(world[:, 0]), np.diff(world[:, 1]))))
    rlen = float(np.sum(np.hypot(np.diff(raw_world[:, 0]),
                                 np.diff(raw_world[:, 1]))))
    check("and it is no longer than the staircase was", slen <= rlen + 1e-9,
          f"{rlen:.2f} m -> {slen:.2f} m")
    check("smoothing twice changes nothing",
          pp.smooth(sm, prep.plannable) == sm)
    check("short paths pass straight through",
          pp.smooth([(1, 1), (2, 2)], prep.plannable) == [(1, 1), (2, 2)])


def test_hands_off_to_the_follower() -> None:
    """The output must feed `path_to_moves` with no adapter in between."""
    print("\n— hand-off to slam/plan.py —")
    prep = pp.preprocess(_room())
    cells, _ = pp.plan(prep, (10, 20), (90, 20))
    world = pp.cells_to_world(cells, prep.grid)
    check("cells_to_world gives an (N, 2) float array",
          world.ndim == 2 and world.shape[1] == 2, str(world.shape))
    moves = path_to_moves(world, 0.0)
    check("path_to_moves accepts it", len(moves) == len(cells) - 1,
          f"{len(moves)} legs from {len(cells)} waypoints")
    check("every leg is a real drive",
          all(dist > 0 for _, dist in moves))
    check("plan() returns the Preprocessed it used",
          isinstance(pp.plan(prep, (10, 20), (90, 20))[1], pp.Preprocessed))


def test_astar_on_real_maps() -> None:
    """Against the saved maps, cross-checked with the deployed planner."""
    print("\n— A* on the real maps —")
    from scipy.ndimage import label as _label
    for name in ("map.npz", "rl_map.npz"):
        path_ = config.ROOT / "state" / name
        if not path_.exists():
            print(f"[ skip ] {name} — not on disk")
            continue
        prep = pp.preprocess(OccupancyGrid.load(path_))
        lab, _ = _label(prep.plannable, structure=np.ones((3, 3), bool))
        counts = np.bincount(lab.ravel())[1:]
        big = np.argwhere(lab == int(np.argmax(counts)) + 1)
        a = tuple(int(v) for v in big[np.argmin(big[:, 0] + big[:, 1])])
        b = tuple(int(v) for v in big[np.argmax(big[:, 0] + big[:, 1])])

        t0 = time.perf_counter()
        raw = pp.astar(prep, a, b, gradient_weight=0.0)
        ms = (time.perf_counter() - t0) * 1000
        field = DistanceField(prep.grid, pp.cells_to_world([a], prep.grid)[0])
        optimum = field.distance_to(pp.cells_to_world([b], prep.grid)[0])

        if raw:
            sm = pp.smooth(raw, prep.plannable)
            world = pp.cells_to_world(raw, prep.grid)
            length = float(np.sum(np.hypot(np.diff(world[:, 0]),
                                           np.diff(world[:, 1]))))
            print(f"  {name}: {a} -> {b}, {len(raw)} cells -> {len(sm)} "
                  f"waypoints, {length:.2f} m in {ms:.0f} ms")
            check(f"{name}: A* agrees with Dijkstra",
                  abs(length - optimum) < 1e-6,
                  f"{length:.3f} vs {optimum:.3f}")
            check(f"{name}: smoothed legs are safe",
                  all(pp.line_of_sight(prep.plannable, sm[i], sm[i + 1])
                      for i in range(len(sm) - 1)))
        else:
            print(f"  {name}: no route {a} -> {b} ({ms:.0f} ms)")
            check(f"{name}: Dijkstra agrees it is unreachable",
                  not np.isfinite(optimum),
                  "unreachable for both planners"
                  if not np.isfinite(optimum)
                  else "A* refused but Dijkstra found a route")


def main() -> int:
    print("path_planner.py — file 14 steps 1-3")
    test_specks()
    test_cleared_speck_is_free()
    test_read_only()
    test_unknown_is_not_plannable()
    test_inflation()
    test_order_matters()
    test_empty_map()
    test_extra_blocked()
    test_real_maps()
    test_astar_matches_dijkstra()
    test_octile_is_admissible()
    test_gradient_prefers_clearance()
    test_gradient_still_takes_the_only_way()
    test_no_corner_cutting()
    test_no_route_is_not_a_guess()
    test_open_corridor()
    test_snapping()
    test_smoothing()
    test_hands_off_to_the_follower()
    test_astar_on_real_maps()
    print(f"\n{'ALL PASSED' if not _failures else f'{_failures} FAILED'}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
