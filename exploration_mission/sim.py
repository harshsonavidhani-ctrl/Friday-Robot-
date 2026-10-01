"""A whole exploration mission as a dry run: no motors, no speaker, no network
(File 21 step 12).

    ./venv/bin/python -m exploration_mission.sim                  # three rooms
    ./venv/bin/python -m exploration_mission.sim --map real       # a COPY of state/map.npz
    ./venv/bin/python -m exploration_mission.sim --map uneven --replies "hall,kitchen,"

Everything the mission really runs, runs: the state machine, its own map
(slam's OccupancyGrid), File 14's despeckle, frontiers, rooms, the label
dialog's grammar, the area store, the return ladder, the arrival check, the
late fit onto the saved map. Faked: the clock (a mission takes seconds), the
RL run (simworld.SimRun: a 2D ray-cast LiDAR on the true map and a simple
driver that follows a tour in place of the policy), the voice (her lines are
PRINTED; his answers are typed), the face (printed).

Her questions wait for a line on stdin (empty = silence). `--replies` gives
the answers up front instead (comma-separated; an empty item is silence), for
a run nobody watches.

Nothing here touches the robot or her files: the database and every file go
to a temp folder (`--keep DIR` to look at them afterwards), `--map real`
copies state/map.npz before reading it, and the network is blocked for the
whole run (tests/exploration_mission/netguard.py) -- a mission that tried to
reach out would fail loudly here.
"""

from __future__ import annotations

import argparse
import datetime
import math
import shutil
import statistics
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests" / "exploration_mission"

TICK_S = 0.1
TOUR_POINTS = 6
TOUR_CLEARANCE_M = 0.30          # the fake driver keeps this far from walls
TOUR_STEP_M = 0.30               # waypoints along the planned tour


# --- what she says, what he types -------------------------------------------------

class PrintVoice:
    """FakeVoice's interface. say() prints; listen_once() reads a line."""

    def __init__(self, clock, replies: list[str | None] | None, out=print):
        self.clock, self.out = clock, out
        self.replies = None if replies is None else list(replies)
        self.said: list[str] = []
        self.listened: list[float] = []

    def say(self, text):
        self.said.append(text)
        self.out(f"  FRIDAY: {text}")
        return text

    def can_listen(self):
        return True

    def listen_once(self, timeout_s):
        self.listened.append(timeout_s)
        if self.replies is not None:
            answer = self.replies.pop(0) if self.replies else None
            self.out(f"  HARSH : {answer}" if answer else "  HARSH : (silence)")
            return answer or None
        try:
            answer = input(f"  (listening {timeout_s:g} s -- type the answer, "
                           f"empty = silence) > ").strip()
        except EOFError:
            answer = ""
        return answer or None


class PrintFace:
    def __init__(self, out=print):
        self.out, self.shown = out, []

    def names(self):
        from exploration_mission import adapters as ad
        return ad.FakeFace().names()

    def set(self, name):
        self.shown.append(name)
        self.out(f"  [face: {name}]")
        return name


# --- the worlds -------------------------------------------------------------------

def _clearance_tour(grid, start_xy=None) -> tuple[list[tuple[float, float]], tuple]:
    """A tour of a real map for the fake driver: up to TOUR_POINTS spots spread
    over the reachable floor (farthest-point sampling), visited nearest-first
    along 8-connected shortest paths that keep TOUR_CLEARANCE_M from walls and
    never enter unknown cells."""
    from scipy import ndimage
    res = grid.resolution
    free = grid.free()
    clear = ndimage.distance_transform_edt(free) * res
    ok = free & (clear >= TOUR_CLEARANCE_M)
    if not ok.any():
        raise SystemExit("the map has no floor with room to drive")
    if start_xy is None:
        start = np.unravel_index(np.argmax(np.where(ok, clear, 0.0)), ok.shape)
    else:
        start = grid.world_to_cell(*start_xy)
    labels, _ = ndimage.label(ok, structure=np.ones((3, 3)))
    ok = labels == labels[start]

    def bfs(src):
        dist = np.full(ok.shape, -1, np.int32)
        prev = {}
        dist[src] = 0
        q = deque([src])
        while q:
            c = q.popleft()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if (0 <= n[0] < ok.shape[0] and 0 <= n[1] < ok.shape[1]
                            and ok[n] and dist[n] < 0):
                        dist[n] = dist[c] + 1
                        prev[n] = c
                        q.append(n)
        return dist, prev

    cells = np.argwhere(ok)
    picks = [tuple(start)]
    d_min = np.hypot(*(cells - np.array(start)).T)
    for _ in range(TOUR_POINTS - 1):
        i = int(np.argmax(d_min))
        if d_min[i] * res < 0.6:
            break
        picks.append(tuple(cells[i]))
        d_min = np.minimum(d_min, np.hypot(*(cells - cells[i]).T))
    order, here, todo, path_cells = [], picks[0], picks[1:], [picks[0]]
    while todo:
        dist, prev = bfs(here)
        todo.sort(key=lambda c: dist[c] if dist[c] >= 0 else 10 ** 9)
        nxt = todo.pop(0)
        if dist[nxt] < 0:
            continue
        leg, c = [], nxt
        while c != here:
            leg.append(c)
            c = prev[c]
        path_cells += list(reversed(leg))
        order.append(nxt)
        here = nxt
    every = max(1, int(round(TOUR_STEP_M / res)))
    keep = path_cells[::every] + [path_cells[-1]]
    xy = [grid.cell_to_world(int(c[0]), int(c[1])) for c in keep]
    return [(round(x, 3), round(y, 3)) for x, y in xy], tuple(start)


def world(name: str, tmp: Path):
    """(truth Map, tour, saved map file or None, a one-line description)."""
    import make_maps as mm
    import simworld as sw
    if name == "three":
        return mm.three_rooms(), sw.THREE_ROOM_TOUR, None, \
            "three identical 3.5 x 4 m rooms in a row (make_maps.three_rooms)"
    if name == "uneven":
        m = mm.uneven_rooms()
        saved = tmp / "saved_map.npz"
        mm.to_occupancy_grid(m.grid).save(saved)
        return m, sw.UNEVEN_TOUR, saved, \
            "three rooms of different sizes, with their SLAM map saved (late fit)"
    if name == "closed":
        return mm.closed_room(), [(3.0, 2.5), (2.0, 2.0), (4.0, 2.0), (4.0, 3.0),
                                  (2.0, 3.0), (3.0, 2.5)], None, "one closed 4 x 3 m room"
    if name == "real":
        import config
        from exploration_mission import grid_utils as gu
        from slam.grid import OccupancyGrid
        src = Path(config.SLAM_MAP_FILE) if hasattr(config, "SLAM_MAP_FILE") \
            else ROOT / "state" / "map.npz"
        saved = tmp / "saved_map.npz"
        shutil.copy2(src, saved)                        # a COPY: never the real one
        og = OccupancyGrid.load(saved)
        grid = gu.from_occupancy_grid(og)
        # The world the LiDAR sees: what the map calls a wall, and -- so the
        # beams stop at the edge of what was ever seen -- everything unknown.
        truth = gu.Grid(np.where(grid.free(), gu.FREE, gu.OCC).astype(np.int8),
                        grid.resolution, grid.origin)
        tour, start = _clearance_tour(grid)
        known = float(grid.free().sum()) * grid.resolution ** 2
        return mm.Map(truth, (tour[0][0], tour[0][1], 0.0), "real"), tour, saved, \
            (f"a copy of {src.name}: {grid.shape[0] * grid.resolution:.2f} x "
             f"{grid.shape[1] * grid.resolution:.2f} m, {known:.1f} m2 of known floor, "
             f"{len(tour)} tour points")
    raise SystemExit(f"unknown map {name!r}")


# --- the run -----------------------------------------------------------------------

def run(args, out=print) -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))
    import netguard
    netguard.block()

    from exploration_mission import adapters as ad
    from exploration_mission import area_store as st
    from exploration_mission import config as mcfg
    from exploration_mission import intents
    from exploration_mission import mission_manager as M
    import simworld as sw

    tmp = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="friday-sim-"))
    tmp.mkdir(parents=True, exist_ok=True)
    truth, tour, saved, about = world(args.map, tmp)
    out(f"world: {about}")

    clock = ad.FakeClock(start=time.time())
    _now_iso = st.now_iso
    st.now_iso = lambda: datetime.datetime.fromtimestamp(   # the store on the sim's
        clock.wall()).astimezone().isoformat(timespec="seconds")   # clock
    try:
        sim = sw.SimRun(clock, truth, tour, matched_within_m=args.matched_within,
                        fail_return_at_m=args.fail_return)
        replies = None if args.replies is None else [
            (r.strip() or None) for r in args.replies.split(",")]
        voice = PrintVoice(clock, replies, out)
        slam = ad.SlamAdapter(map_file=saved or tmp / "no_saved_map.npz",
                              session_file=tmp / "session.json")
        slam.map_id = lambda read_only=False: f"sim-{args.map}"
        slam.localise = lambda scan: None                   # one sweep never places a start
        a = ad.fake_adapters(clock=clock, run=sim, voice=voice, face=PrintFace(out),
                             slam=slam, lidar=sim.lidar(),
                             nav=ad.FakeNav(during_run=False, clock=clock, engine=False),
                             talk=ad.FakeTalk(quiet=3600.0))
        store = st.AreaStore(db_path=tmp / "memory.db", data_dir=tmp / "data",
                             places=a.places)
        store.ensure_schema()
        over = {}
        if args.minutes:
            over["explore"] = {"mission_cap_min": float(args.minutes)}
        cfg = mcfg.load(over)
        mgr = M.MissionManager(a, store, cfg, runner=M.InlineRunner(),
                               inline_analysis=True, map_policy=args.policy)
        router = intents.Router(
            mgr, resolve_area=lambda n: store.by_name(a.env.env_id(), n),
            resolve_place=a.places.find_exact, monotonic=clock.monotonic)

        states, t = [], 0.0
        frontier_ms, segment_ms = [], []

        def tick() -> str:
            nonlocal t
            clock.advance(TICK_S)
            sim.step(TICK_S)
            s = mgr.tick()
            t += TICK_S
            res = mgr.analysis.latest
            if res is not None and (not frontier_ms or res["t"] != frontier_ms[-1][0]):
                frontier_ms.append((res["t"], res["frontier_ms"]))
                if res["segment_ms"] is not None:
                    segment_ms.append(res["segment_ms"])
            if not states or states[-1][1] != s:
                states.append((round(t, 1), s))
                out(f"[{t:6.1f} s] state {s}")
            return s

        out("boot: restore -> relocalise -> idle (no motion)")
        mgr.boot()
        states.append((0.0, mgr.state))         # set before the first tick
        out(f"[{t:6.1f} s] state {mgr.state}")
        while not (mgr.booted and mgr.state == "IDLE") and t < 120:
            tick()
        out(f'HARSH : "{args.say}"')
        if not router.handle(args.say, mgr.context()):
            out("  (not a mission command -- it would go to her brain)")
            if not args.keep:
                shutil.rmtree(tmp, ignore_errors=True)
            return {"tmp": tmp, "said": voice.said, "states": [], "mission": None}
        started = t
        while t - started < args.max_s:
            s = tick()
            if s == "IDLE" and t - started > 3.0 and mgr.mission is None:
                break
            if s in ("HELP_WAIT", "FAILSAFE") and t - started > 3.0:
                for _ in range(int(5.0 / TICK_S)):
                    tick()
                break
    finally:
        st.now_iso = _now_iso

    # --- what was saved ---
    env = a.env.env_id()
    out("")
    out("=== saved ===")
    for ar in store.list_areas(env):
        out(f"  area {ar['display_name']!r:16} {ar['status']:8} {ar['area_m2']:5.1f} m2 "
            f"frame {ar['frame']}")
    m = store.missions(env, limit=1)
    if m:
        m = m[0]
        out(f"  mission {m['status']} ({m.get('stop_reason')}), driven "
            f"{m.get('distance_m') or 0:.1f} m, mapped {m.get('explored_m2_end') or 0:.1f} m2, "
            f"ended {m.get('home_error_m') if m.get('home_error_m') is not None else '?'} "
            f"m from the start, return tries {m.get('rth_attempts')}")
    frames = {ar["frame"] for ar in store.list_areas(env)}
    if saved is None:
        out("  no saved SLAM map in this world: the areas stay in the mission's frame")
    elif frames == {st.FRAME_MAP}:
        out("  the whole mission map FITTED the saved map: areas moved into its frame "
            "(named ones are File 14 landmarks)")
    elif frames:
        out("  the whole-map fit did not take (look-alikes or too little overlap): "
            "the areas stay in the mission's frame, nothing guessed")
    truth_d = math.hypot(sim.truth[0] - tour[0][0], sim.truth[1] - tour[0][1])
    out(f"  truly {truth_d:.2f} m from where she started; home anchor "
        f"{'saved' if store.home_anchor(env) else 'none'}; landmarks synced: "
        f"{sorted(a.places.rows) or 'none'}")
    ticks = list(mgr.tick_ms)
    fms = [v for _, v in frontier_ms]
    out("=== timings on this machine ===")
    if ticks:
        p95 = sorted(ticks)[int(0.95 * (len(ticks) - 1))]
        out(f"  tick median {statistics.median(ticks):.2f} ms, p95 {p95:.2f} ms "
            f"(budget 100 ms at 10 Hz; spec p95 < 20 ms)")
    if fms:
        out(f"  frontiers median {statistics.median(fms):.0f} ms, max {max(fms):.0f} ms "
            f"(spec < 150 ms)")
    if segment_ms:
        out(f"  segmentation median {statistics.median(segment_ms):.0f} ms, max "
            f"{max(segment_ms):.0f} ms (spec < 300 ms)")
    out(f"  she said {len(voice.said)} lines; simulated {t:.0f} s")
    if args.keep:
        out(f"data kept in {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"tmp": tmp, "said": voice.said, "states": [s for _, s in states],
            "areas": store.list_areas(env) if args.keep else None,
            "truth_m": truth_d, "mission": m}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default="three", choices=("three", "uneven", "closed", "real"))
    ap.add_argument("--say", default="explore the house",
                    help='what he says to start it (default "explore the house")')
    ap.add_argument("--replies", default=None,
                    help="her questions' answers, comma-separated (empty = silence); "
                         "without it, they are typed")
    ap.add_argument("--policy", default="matched", help="the mission map's policy")
    ap.add_argument("--minutes", type=float, default=None, help="mission cap")
    ap.add_argument("--matched-within", type=float, default=None,
                    help="the run's tracker only matches this close to the start")
    ap.add_argument("--fail-return", type=float, default=None,
                    help="the run's own way back gives up this far out (shows HELP)")
    ap.add_argument("--max-s", type=float, default=1800.0, help="simulated seconds cap")
    ap.add_argument("--keep", default=None, help="keep the run's data in this folder")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the mission's own log (transitions, timings, fits)")
    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="    log: %(message)s")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
