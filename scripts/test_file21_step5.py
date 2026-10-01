#!/usr/bin/env python3
"""File 21 step 5: room segmentation, room identity, and when to ask a name.

    ./venv/bin/python scripts/test_file21_step5.py

Network blocked, no hardware, temp files only. Maps are the synthetic ones
(tests/exploration_mission/make_maps.py) through File 14's own despeckle, and
the real saved map read-only. Timings are measured on this Pi.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import area_store as st        # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import frontiers as fr         # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402
from exploration_mission import recorder                # noqa: E402
from exploration_mission import segmentation as sg      # noqa: E402
import make_maps as mm                                  # noqa: E402

INFLATE = config.NAV_INFLATE_M
SLAM = ad.SlamAdapter()
TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s5-"))
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def run(m: mm.Map, blacklist=None, **kw):
    """(frontier result, segmentation with complete flags) for a map."""
    g = SLAM.planner_grid(mm.to_occupancy_grid(m.grid))
    r = fr.find(g, m.pose, INFLATE, blacklist=blacklist)
    s = sg.segment(r.coarse, sg.Params(**kw))
    sg.mark_complete(s, r)
    return r, s


def room_at(s: sg.Segmentation, x: float, y: float):
    return s.at(x, y)


# =============================================================================
print("--- rooms (spec 4.5) ---")

r, s = run(mm.three_rooms())
check("the 3-room house is exactly 3 areas", len(s.segments) == 3,
      str([x.as_dict() for x in s.segments]))
a, b, c = room_at(s, 2.5, 3), room_at(s, 6.2, 3), room_at(s, 9.8, 3)
check("...one per room (A, B, C are three different areas)",
      a and b and c and len({a.label, b.label, c.label}) == 3)
check("each is about its real floor area (3.3 x 3.8 m = 12.5 m2)",
      all(11.5 <= x.area_m2 <= 13.5 for x in s.segments),
      str([x.area_m2 for x in s.segments]))
check("all three complete: nothing reachable is left unexplored",
      all(x.complete for x in s.segments))
check("radius = clamp(sqrt(area/pi), 0.5, 2.0)",
      all(abs(x.radius_m - 2.0) < 1e-9 for x in s.segments))
check("outlines have at most 40 points and cover their room (IoU >= 0.95)",
      all(3 <= len(x.polygon) <= 40
          and sg.iou(x.mask, sg.rasterize(x.polygon, s.coarse)) >= 0.95
          for x in s.segments))
check("every centre is inside its own room",
      all(s.at(*x.center) is x and st.point_in_polygon(*x.center, x.polygon)
          for x in s.segments))
r2, s2 = run(mm.three_rooms())
check("the same map gives the same rooms (deterministic)",
      np.array_equal(s.labels, s2.labels))
check("the floor in a doorway belongs to a room (no cell left unassigned)",
      int((s.labels > 0).sum()) == int(s.coarse.free().sum()))


def two_rooms(opening: float) -> mm.Map:
    g = mm.blank(9.5, 6)
    mm.room(g, 1, 1, 4.5, 5)
    mm.room(g, 4.5, 1, 8.0, 5)
    mm.door_v(g, 4.5, 3.0, width=opening)
    return mm.Map(g, (2.5, 3.0, 0.0))


check("a 0.8 m door splits two rooms", len(run(two_rooms(0.8))[1].segments) == 2)
check("a 0.6 m door splits them too", len(run(two_rooms(0.6))[1].segments) == 2)
check("a 1.2 m opening does NOT (the documented limit: one open-plan area)",
      len(run(two_rooms(1.2))[1].segments) == 1)

lmap = mm.blank(7, 7)
mm.rect(lmap, 1, 1, 6, 6, gu.OCC)                       # a block of wall...
mm.rect(lmap, 1.1, 1.1, 5.9, 2.4, gu.FREE)              # ...with an L cut out:
mm.rect(lmap, 1.1, 1.1, 2.4, 5.9, gu.FREE)              #   two 1.3 m wide arms
_, sl = run(mm.Map(lmap, (1.7, 1.7, 0.0)))
L = sl.segments[0] if sl.segments else None
cent = None
if L is not None:
    cells = np.argwhere(L.mask)
    cent = sl.coarse.cell_to_world(*(int(round(v)) for v in cells.mean(axis=0)))
check("an L-shaped room is one area", len(sl.segments) == 1)
check("...its CENTROID falls outside the room", L is not None
      and sl.at(*cent) is None, str(cent))
check("...its centre (furthest from the walls) is inside the room and its outline",
      L is not None and sl.at(*L.center) is L
      and st.point_in_polygon(*L.center, L.polygon), str(L and L.center))

alc = mm.blank(8, 6)
mm.room(alc, 1, 1, 5, 5)
mm.rect(alc, 5 - mm.WALL, 2.7, 5.0, 3.3, gu.FREE)       # a 0.6 m opening
mm.rect(alc, 5.0, 2.5, 6.0, 3.5, gu.OCC)                # the alcove's walls
mm.rect(alc, 5.0, 2.6, 5.9, 3.4, gu.FREE)               # 0.9 x 0.8 m of floor
_, sa = run(mm.Map(alc, (3.0, 3.0, 0.0)))
check("a small alcove is part of its room, not an area of its own",
      len(sa.segments) == 1 and sa.at(5.5, 3.0) is sa.segments[0])

lab = np.zeros((12, 12), np.uint16)
lab[0:6, 0:6] = 1                                       # big
lab[0:6, 6:12] = 2                                      # big
lab[6:8, 1:6] = 3                                       # small: 5 cells on room 1,
lab[6:8, 6:8] = 3                                       #        2 cells on room 2
merged = sg._merge_small(lab, min_cells=30)
check("a room under min_room_m2 merges into the neighbour with the LONGEST border",
      set(np.unique(merged)) == {0, 1, 2} and merged[7, 3] == 1 and merged[7, 7] == 1)
alone = np.zeros((10, 10), np.uint16)
alone[1:3, 1:3] = 1
check("a small room with no neighbour is left as it is",
      np.array_equal(sg._merge_small(alone, 30), alone))
tie = np.zeros((1, 7), np.uint16)
tie[0, 0], tie[0, 6] = 1, 2
grown = sg._grow(tie, np.ones((1, 7), bool))
check("a cell two seeds reach together goes to the higher label",
      grown.tolist() == [[1, 1, 1, 2, 2, 2, 2]], str(grown.tolist()))
_, stiny = run(mm.closed_room(), min_seed_m2=50.0)
check("a seed under min_seed_m2 is dropped (nothing becomes a room)",
      len(stiny.segments) == 0)
many = np.zeros((60, 60), bool)
many[5:55, 5:55] = True
many[5:55:2, 5] = False                                 # a saw-toothed wall
poly = sg.outline(many, gu.Grid(np.zeros((60, 60), np.int8), 0.1, (0, 0)), 0.01, 12)
check("the outline is simplified until it fits max_polygon_vertices",
      3 <= len(poly) <= 12, str(len(poly)))


# =============================================================================
print("--- complete = no reachable frontier inside (spec 4.5 step 8) ---")

r, s = run(mm.three_rooms((True, True, False)))
a, b = room_at(s, 2.5, 3), room_at(s, 6.2, 3)
check("room C unexplored: A is complete", len(s.segments) == 2 and a.complete)
check("...B is not: the frontier at its door to C is inside it", not b.complete)
bl = fr.Blacklist(2, 0.5)
bl.fail(r.best.centroid)
bl.fail(r.best.centroid)
_, s_bl = run(mm.three_rooms((True, True, False)), blacklist=bl)
check("...once that frontier is blacklisted it no longer counts: B is complete",
      room_at(s_bl, 6.2, 3).complete)
rg, sgap = run(mm.gap_frontier())
check("an UNREACHABLE frontier (behind the 0.1 m slit) does not hold a room open",
      len(rg.frontiers) >= 1 and sgap.at(2.0, 2.5).complete)


# =============================================================================
print("--- identity across ticks and a growing map ---")

tracker = sg.Tracker(0.5)
_, snap1 = run(mm.three_rooms((True, True, False)))
tracker.assign(snap1)
ka, kb = room_at(snap1, 2.5, 3).key, room_at(snap1, 6.2, 3).key
full = mm.three_rooms()
grown_map = mm.Map(gu.Grid(np.pad(full.grid.occ, ((40, 0), (20, 0)),
                                  constant_values=gu.UNKNOWN), mm.RES,
                           (-40 * mm.RES, -20 * mm.RES)), full.pose)
_, snap2 = run(grown_map)
tracker.assign(snap2)
check("the map grew and its origin moved: A and B keep their keys",
      room_at(snap2, 2.5, 3).key == ka and room_at(snap2, 6.2, 3).key == kb
      and snap2.coarse.origin != snap1.coarse.origin)
check("...and the newly explored room C gets a new one",
      room_at(snap2, 9.8, 3).key not in (ka, kb, None))
tracker.assign(snap2)
check("the same tick again changes nothing",
      room_at(snap2, 2.5, 3).key == ka and room_at(snap2, 9.8, 3).key
      not in (ka, kb))

areas = [{"id": "area-A", "polygon": room_at(snap1, 2.5, 3).polygon, "area_m2": 12.6},
         {"id": "far", "polygon": [[30, 30], [33, 30], [33, 33], [30, 33]],
          "area_m2": 9.0}]
sg.match_areas(snap2, areas, 0.5)
check("a room that IS a saved area is matched to it on the grown map (IoU >= 0.5)",
      room_at(snap2, 2.5, 3).area_id == "area-A"
      and room_at(snap2, 2.5, 3).area_iou >= 0.9)
check("...and the others are not", room_at(snap2, 6.2, 3).area_id is None
      and room_at(snap2, 9.8, 3).area_id is None)


def partial_b(fraction: float) -> mm.Map:
    """Rooms A and B, with only the first `fraction` of B seen so far."""
    m = mm.three_rooms((True, True, False))
    cut = 4.6 + 3.3 * fraction
    mm.rect(m.grid, cut, 0.5, 8.2, 5.5, gu.UNKNOWN)
    return m


places = ad.FakePlaces()
store = st.AreaStore(db_path=TMP / "grow.db", data_dir=TMP / "grow", places=places)
store.ensure_schema()
_, early = run(partial_b(0.7))
eb = room_at(early, 5.2, 3)
saved = store.add_area("home", "map-1", polygon=eb.polygon, center=eb.center,
                       area_m2=eb.area_m2, radius_m=eb.radius_m, name="bedroom",
                       complete=False)
_, later = run(mm.three_rooms())                        # the whole house, walls and all
sg.match_areas(later, store.list_areas("home"), 0.5)
lb = room_at(later, 6.2, 3)
ups = sg.growth_updates(later, store.list_areas("home"), 1.2)
check("a saved room seen 70% -> now whole: same place (IoU >= 0.5), >= 20% bigger",
      lb.area_id == saved["id"] and len(ups) == 1 and ups[0][0] == saved["id"],
      f"iou {lb.area_iou:.2f}, {eb.area_m2:.1f} -> {lb.area_m2:.1f} m2")
aid, seg_new = ups[0]
store.update_geometry(aid, polygon=seg_new.polygon, center=seg_new.center,
                      area_m2=seg_new.area_m2, radius_m=seg_new.radius_m)
now_saved = store.get(aid)
check("...updated silently: new shape and centre, the NAME is kept",
      now_saved["name"] == "bedroom" and abs(now_saved["area_m2"] - lb.area_m2) < 1e-6
      and (places.find_exact("bedroom")["x"], places.find_exact("bedroom")["y"])
      == seg_new.center)
_, almost = run(partial_b(0.9))
ab = room_at(almost, 5.2, 3)
small_saved = [{"id": "x", "polygon": ab.polygon, "area_m2": ab.area_m2}]
sg.match_areas(later, small_saved, 0.5)
check("seen 90% -> whole is only ~11% bigger: no update",
      sg.growth_updates(later, small_saved, 1.2) == [])
_, third = run(partial_b(0.35))
tb = room_at(third, 5.0, 3)
third_saved = [{"id": "y", "polygon": tb.polygon, "area_m2": tb.area_m2}]
sg.match_areas(later, third_saved, 0.5)
check("seen a third -> whole overlaps under 0.5: not the same area by the rule",
      room_at(later, 6.2, 3).area_id is None)
_, open_c = run(mm.three_rooms((True, True, False)))
sg.match_areas(open_c, [{"id": "b", "polygon": eb.polygon, "area_m2": eb.area_m2}], 0.5)
check("a room that is NOT complete (a frontier inside) never 'grows' a saved area: "
      "mid-exploration shapes are transient",
      room_at(open_c, 6.2, 3).area_id == "b" and not room_at(open_c, 6.2, 3).complete
      and sg.growth_updates(open_c, [{"id": "b", "polygon": eb.polygon,
                                      "area_m2": eb.area_m2}], 1.2) == [])
whole_b = room_at(later, 6.2, 3)
_, half = run(partial_b(0.5))
hb = room_at(half, 5.2, 3)
sg.match_areas(half, [{"id": "kitchen", "polygon": whole_b.polygon,
                       "area_m2": whole_b.area_m2}], 0.5)
check("the reverse: a saved WHOLE room seen only in half now (IoU ~0.5 or less) "
      "is still that room -- it lies inside the saved outline",
      hb.area_id == "kitchen" and sg.iou(hb.mask, sg.rasterize(whole_b.polygon,
                                                             half.coarse)) < 0.6,
      f"iou {hb.area_iou:.2f}")
check("...so she does not ask about it again, and it does not shrink the saved one",
      sg.growth_updates(half, [{"id": "kitchen", "polygon": whole_b.polygon,
                                "area_m2": whole_b.area_m2}], 1.2) == [])
check("a room mostly OUTSIDE a saved area (30% in) is not it",
      sg.inside_fraction(np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool),
                         np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)) == 1.0
      and sg.inside_fraction(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], bool),
                             np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], bool)) == 0.3)


# =============================================================================
print("--- when to ask (spec 4.6): each condition alone blocks ---")

P = sg.LabelParams()


def scene(area=12.0, complete=True, area_id=None, inside=20.0, current=True,
          exited_ago=None, now=1000.0):
    """One room (key 7) and a Dwell that says what the robot did there."""
    coarse = gu.Grid(np.zeros((10, 10), np.int8), 0.1, (0, 0))
    seg = sg.Segment(label=1, mask=np.ones((10, 10), bool), area_m2=area,
                     polygon=[[0, 0], [1, 0], [1, 1], [0, 1]], center=(0.5, 0.5),
                     radius_m=1.0, complete=complete, key=7, area_id=area_id)
    d = sg.Dwell()
    d.inside_s[7] = inside
    d.current = 7 if current else None
    if exited_ago is not None:
        d.exited_at[7] = now - exited_ago
    return sg.Segmentation(coarse, np.ones((10, 10), np.uint16), [seg]), d, now


def ask(seg, d, now, last=None, exploring=True, params=P, **kw):
    return sg.choose_label_target(seg, d, now, last, exploring, params, **kw)


seg, d, now = scene()
got = ask(seg, d, now, last=now - 100)
check("all conditions met: she asks about the room she is in",
      got is not None and got.segment.key == 7 and got.rule == "segmentation"
      and got.complete)
check("too small (< 2.5 m2) -> no question", ask(*scene(area=2.0)) is None)
check("not complete -> no question", ask(*scene(complete=False)) is None)
check("already saved -> no question (it IS that area)",
      ask(*scene(area_id="kitchen-id")) is None)
check("inside only 5 s -> no question yet", ask(*scene(inside=5.0)) is None)
check("...inside 15 s -> asked", ask(*scene(inside=15.0)) is not None)
check("drove through in 5 s but left 10 s ago -> asked (he knows which room)",
      ask(*scene(inside=5.0, current=False, exited_ago=10.0)) is not None)
check("...left 25 s ago -> too late, no question",
      ask(*scene(inside=5.0, current=False, exited_ago=25.0)) is None)
seg, d, now = scene()
check("last question 10 s ago -> cooldown, no question",
      ask(seg, d, now, last=now - 10) is None)
check("...30 s ago -> asked", ask(seg, d, now, last=now - 30) is not None)
check("...at completion the cooldown is ignored once",
      ask(seg, d, now, last=now - 10, ignore_cooldown=True) is not None)
check("not exploring (paused / go-to / returning) -> no question",
      ask(seg, d, now, exploring=False) is None)
check("why_not names every failing condition",
      sg.why_not(scene(area=1, complete=False, area_id="x", inside=0)[0].segments[0],
                 sg.Dwell(), 0.0, P)
      == ["too small", "not complete", "already saved",
          "not dwelt in, not just left"])

coarse = gu.Grid(np.zeros((10, 10), np.int8), 0.1, (0, 0))


def mk(key, **kw):
    base = dict(label=key, mask=np.ones((10, 10), bool), area_m2=9.0,
                polygon=[[0, 0], [1, 0], [1, 1]], center=(0.5, 0.5), radius_m=1.0,
                complete=True, key=key)
    base.update(kw)
    return sg.Segment(**base)


multi = sg.Segmentation(coarse, np.ones((10, 10), np.uint16), [mk(1), mk(2), mk(3)])
d = sg.Dwell()
d.inside_s = {1: 20.0, 2: 20.0, 3: 20.0}
d.exited_at = {1: 900.0, 2: 950.0}
d.current = 3
check("several qualify -> the one she is IN", ask(multi, d, 1000.0).segment.key == 3)
d.current = None
d.exited_at[3] = 800.0
check("...in none of them -> the most recently exited",
      ask(multi, d, 1000.0).segment.key == 2)

dist = sg.LabelParams(trigger="distance")
seg, d, now = scene(complete=False)
got = ask(seg, d, now, params=dist, distance_from_saved_m=5.0)
check("distance trigger: 5 m from the nearest saved area, in an unsaved room -> "
      "asked, recorded as not complete",
      got is not None and got.rule == "distance" and got.complete is False)
check("...3 m away -> no question", ask(seg, d, now, params=dist,
                                       distance_from_saved_m=3.0) is None)
check("...the segmentation trigger never reads distance",
      ask(seg, d, now, params=P, distance_from_saved_m=50.0) is None)
both = sg.LabelParams(trigger="both")
check("both: the distance rule alone is enough",
      ask(seg, d, now, params=both, distance_from_saved_m=5.0) is not None)
seg2, d2, now2 = scene()
check("both: the segmentation rule alone is enough",
      ask(seg2, d2, now2, params=both, distance_from_saved_m=0.0).rule
      == "segmentation")
seg3, d3, now3 = scene(complete=False, area_id="saved")
check("distance trigger never asks about a room already saved",
      ask(seg3, d3, now3, params=dist, distance_from_saved_m=9.0) is None)

d = sg.Dwell()
d.update(5, 0.0)
d.update(5, 10.0)
d.update(None, 12.0)
d.update(6, 20.0)
d.update(6, 23.0)
check("Dwell adds up time inside each room and notes when she left",
      abs(d.inside_s[5] - 12.0) < 1e-9 and d.exited_at[5] == 12.0
      and abs(d.inside_s[6] - 3.0) < 1e-9 and d.current == 6
      and isinstance(json.dumps(d.to_dict()), str))
ne = sg.Segmentation(coarse, np.ones((10, 10), np.uint16),
                     [mk(5), mk(8), mk(9, complete=False), mk(10, area_id="s"),
                      mk(11, area_m2=1.0)])
check("at completion: complete, unsaved, big enough rooms she never entered "
      "are listed for the dashboard queue",
      [x.key for x in sg.never_entered(ne, d)] == [8])


# =============================================================================
print("--- the recorder (reads files, moves nothing) ---")

clock = ad.FakeClock(2_000_000.0)
frun = ad.FakeRun(clock)
frun.state_dir = TMP / "rec"
frun.status_data = {"tick": 1, "frontiers": 3}
script = [None, None,
          {"t": 1.0, "pose": [0, 0, 0], "reason": "matched", "scan": {0: 1500}},
          {"t": 1.0, "pose": [0, 0, 0], "reason": "matched", "scan": {0: 1500}},
          {"t": 2.0, "pose": [0.2, 0, 0], "reason": "matched", "scan": {0: 1300}},
          {"t": 3.0, "pose": [0.4, 0, 0.1], "reason": "score too low",
           "scan": {0: 1100, 90: 2000}}]
step = {"i": 0}


def fake_sleep(seconds: float) -> None:
    clock.advance(seconds)
    step["i"] += 1
    nxt = script[step["i"]] if step["i"] < len(script) else script[-1]
    frun.live_fix = None if nxt is None else dict(nxt)


frun.live = lambda max_age_s=None: (None if frun.live_fix is None
                                    else dict(frun.live_fix, age_s=0.0))
said: list[str] = []
path = recorder.record(frun, poll_s=0.25, idle_stop_s=3.0, sleep=fake_sleep,
                       say=said.append)
rows = recorder.load(path)
check("it waits for a run, then records each NEW fix once",
      path is not None and [r["t"] for r in rows] == [1.0, 2.0, 3.0], str(rows))
check("each line carries the pose, the sweep and the run's own status",
      rows[2]["pose"] == [0.4, 0, 0.1] and rows[2]["scan"] == {0: 1100, 90: 2000}
      and rows[2]["status"] == {"tick": 1, "frontiers": 3}
      and rows[2]["reason"] == "score too low")
check("it stops by itself once the run has gone quiet", any("quiet" in m for m in said))
with open(path, "a") as fh:
    fh.write('{"t": 4.0, "pose": [0.5')                # a torn last line
check("a torn last line (power cut) does not spoil the recording",
      len(recorder.load(path)) == 3)
clock2 = ad.FakeClock()
idle = ad.FakeRun(clock2)
idle.state_dir = TMP / "rec2"
check("no run within --wait: nothing recorded, no file",
      recorder.record(idle, wait_s=2.0, sleep=clock2.advance, say=lambda m: None)
      is None and not (TMP / "rec2" / "recordings").exists())


# =============================================================================
print("--- config, the real saved map, timing ---")

cfg = mcfg.load()
p, lp = sg.Params.from_config(cfg), sg.LabelParams.from_config(cfg)
check("Params come from config.EXPLORATION_MISSION",
      (p.door_half_width_m, p.min_room_m2, p.max_polygon_vertices,
       lp.min_area_m2, lp.cooldown_s, lp.trigger) == (0.40, 1.5, 40, 2.5, 30.0,
                                                      "segmentation"))
real = SLAM.planner_grid()
if real is not None:
    rr = fr.find(real, (0.0, 0.0, 0.0), INFLATE)
    rs = sg.segment(rr.coarse, p)
    sg.mark_complete(rs, rr)
    print("        real map: " + ", ".join(
        f"{x.area_m2:.1f} m2 {'complete' if x.complete else 'open'}"
        for x in rs.segments) + f" ({rs.ms['total']:.1f} ms)")
    check("runs on the real saved map (read only)", len(rs.segments) >= 1)

big = mm.big_house()
g = SLAM.planner_grid(mm.to_occupancy_grid(big.grid))
rb = fr.find(g, big.pose, INFLATE)
times, tr = [], sg.Tracker()
saved_big = None
for i in range(9):
    s_big = sg.segment(rb.coarse, p)
    sg.mark_complete(s_big, rb)
    tr.assign(s_big)
    if saved_big is None:
        saved_big = [{"id": f"a{k}", "polygon": x.polygon, "area_m2": x.area_m2}
                     for k, x in enumerate(s_big.segments[:6])]
    sg.match_areas(s_big, saved_big)
    times.append(s_big.ms["total"])
import time as _time                                     # noqa: E402
t0 = _time.perf_counter()
s_big = sg.segment(rb.coarse, p)
sg.mark_complete(s_big, rb)
tr.assign(s_big)
sg.match_areas(s_big, saved_big)
whole = (_time.perf_counter() - t0) * 1e3
print(f"        20 m house, {len(s_big.segments)} rooms: segment() median "
      f"{statistics.median(times):.1f} ms, worst {max(times):.1f}; whole job "
      f"(segment + complete + identity + 6 saved areas) {whole:.1f} ms; "
      f"load {os.getloadavg()[0]:.1f}")
check("20 m house: 12 explored rooms found", len(s_big.segments) == 12)
check("segmentation under 300 ms on this Pi (spec budget)",
      statistics.median(times) < 300.0)
check("the whole segmentation job under 300 ms", whole < 300.0)
check("segments serialise for the dashboard",
      isinstance(json.dumps([x.as_dict() for x in s_big.segments]), str))
check("the network guard saw no attempt", not netguard.attempts)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
