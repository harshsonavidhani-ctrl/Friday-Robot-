"""The persistent world map: it must remember, and it must not lie about what it knows.

Two failure modes this guards, both of which would be invisible at runtime:

  1. The map silently not persisting -- a robot that reports "restored from disk" while
     starting blank is worse than one that admits it forgot.
  2. The map corrupting the OBSERVATION. `LocalExplorationMap` feeds 72 of the policy's
     146 inputs with training geometry. `WorldMap` must run beside it and never touch
     it; if adding memory changed the observation, the network would be fed
     out-of-distribution input and nobody would see it until the robot hit something.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner, RobotConfig,
                       SensorSlot)
from world import MapperThread, WorldMap

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


print("--- world map / persistent memory ---")

# 1. Integration actually marks free and occupied space.
w = WorldMap()
room = np.full(cfg.n_beams, 3.0, np.float32)
for i in range(6):
    w.integrate(room, np.array([12.0, 12.0, 0.0]))
s = w.stats()
check("a sweep produces free space", s["free"] > 0, f"{s['free']} cells")
check("a sweep produces occupied space", s["occupied"] > 0, f"{s['occupied']} cells")
check("unknown space is still the majority early on", s["unknown"] > s["free"])
check("explored area is reported in m^2", s["explored_m2"] > 0, f"{s['explored_m2']:.1f}")

# 2. MAX-RANGE BEAMS MUST NOT MARK A WALL. The LD19 publishes max_range for an empty
#    bin, which is indistinguishable from a real 5 m return; marking it paints a
#    phantom arc around the robot on every single sweep.
w2 = WorldMap(max_range=5.0)
empty = np.full(cfg.n_beams, 5.0, np.float32)
for _ in range(6):
    w2.integrate(empty, np.array([12.0, 12.0, 0.0]))
s2 = w2.stats()
check("max-range beams carve free space", s2["free"] > 0, f"{s2['free']} cells")
check("max-range beams mark NO obstacle", s2["occupied"] == 0, f"{s2['occupied']} cells")

# 3. Blind (chassis) beams must contribute nothing at all -- not free, not occupied.
#    Carving through the robot's own body is a bug this project has already had once.
w3 = WorldMap()
close = np.full(cfg.n_beams, 0.25, np.float32)     # as if the chassis were all around
blind = np.arange(cfg.n_beams, dtype=np.int32)     # mask EVERY beam
for _ in range(4):
    w3.integrate(close, np.array([12.0, 12.0, 0.0]), blind=blind)
s3 = w3.stats()
check("a fully blind sweep changes nothing", s3["free"] == 0 and s3["occupied"] == 0,
      f"free={s3['free']} occ={s3['occupied']}")

# 4. The visit trace is the robot's FOOTPRINT, not a one-cell thread.
w4 = WorldMap()
w4.set_pose(12.0, 12.0, 0.0)
check("the robot's own cell is visited", w4.has_visited(12.0, 12.0))
check("a cell within the footprint is visited", w4.has_visited(12.15, 12.0))
check("a distant cell is NOT visited", not w4.has_visited(15.0, 15.0))
check("visited area is non-zero", w4.visited_m2 > 0, f"{w4.visited_m2:.2f} m^2")

# 5. PERSISTENCE. Save, reload, and confirm the knowledge survived.
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "m.npz"
    a = WorldMap(map_file=path)
    for i in range(8):
        a.integrate(room, np.array([12.0 + 0.05 * i, 12.0, 0.0]))
    before = a.stats()
    check("save writes a file", a.save() is not None and path.exists())

    b = WorldMap(map_file=path)
    check("load reports success", b.load() is True)
    after = b.stats()
    check("explored area survived the round trip",
          abs(after["explored_m2"] - before["explored_m2"]) < 1e-6,
          f"{before['explored_m2']:.2f} -> {after['explored_m2']:.2f}")
    check("lifetime sweep count survived", after["sweeps_total"] == before["sweeps_total"],
          f"{before['sweeps_total']} -> {after['sweeps_total']}")
    check("this-run sweep count resets", after["sweeps"] == 0)
    check("a restored map is flagged as restored", b.loaded_from_disk is True)

# 6. A missing map must be a fresh start, never a crash.
missing = WorldMap(map_file=Path(tempfile.gettempdir()) / "definitely-not-here-9137.npz")
check("a missing map file loads as False, not an exception", missing.load() is False)
check("...and leaves the map usable", missing.stats()["explored_m2"] == 0.0)

# 7. An empty map must NOT be saved over a good one.
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "keep.npz"
    good = WorldMap(map_file=path)
    for _ in range(6):
        good.integrate(room, np.array([12.0, 12.0, 0.0]))
    good.save()
    size = path.stat().st_size
    blank = WorldMap(map_file=path)
    check("an unobserved map refuses to save", blank.save() is None)
    check("...so the good map on disk is untouched", path.stat().st_size == size)

# 8. THE CRITICAL ONE: adding the world map must not change the observation.
def obs_after(n_ticks, map_enabled):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg)
    loop = ControlLoop(cfg, policy, NullMotorDriver(cfg), slots)
    loop.map_enabled = map_enabled
    if map_enabled:
        loop.mapper = MapperThread(loop.world, hz=100.0)
        loop.mapper.start()
    out = []
    for i in range(n_ticks):
        slots["lidar"].publish(np.full(cfg.n_beams, 2.0 + 0.01 * i, np.float32))
        loop.tick()
        out.append(loop._obs.copy())
    if loop.mapper is not None:
        loop.mapper.stop()
        loop.mapper.join(timeout=2.0)
    return np.array(out)

off, on = obs_after(12, False), obs_after(12, True)
check("the 146-dim observation is IDENTICAL with mapping on and off",
      np.array_equal(off, on),
      f"max delta {np.abs(off - on).max():.3e}" if off.shape == on.shape else "shape differs")

# 9. The mapper thread must never block the caller, and must survive bad input.
w5 = WorldMap()
m = MapperThread(w5, hz=50.0)
m.start()
import time as _t
# Spaced, not in a tight loop: submit() is drop-latest, so 50 back-to-back calls leave
# ONE snapshot to integrate -- and one sweep marks no free space at all, because a
# single log-odds miss (-0.40) lands at p=0.40, above the 0.35 free threshold. The
# grid is deliberately slow to believe things. Testing it needs real sweeps over real
# time, which is also how the control loop actually feeds it.
for i in range(50):
    m.submit(np.full(cfg.n_beams, 2.0, np.float32), np.array([12.0, 12.0, 0.0]))
    _t.sleep(0.004)
m.submit(np.full(3, np.nan), np.array([12.0, 12.0, 0.0]))   # malformed: must not kill it
_t.sleep(0.6)
alive = m.is_alive()
m.stop()
m.join(timeout=2.0)
check("the mapper survives a malformed sweep", alive)
check("the mapper integrated something", w5.stats()["free"] > 0)
check("superseded snapshots are counted, not queued", w5.dropped > 0, f"{w5.dropped}")

# 10. VISIT COUNTS. "Have I been here" and "have I been here five times" are
#     different questions, and only the second detects a robot going in circles.
w6 = WorldMap()
for i in range(10):
    w6.set_pose(12.0 + 0.15 * i, 12.0, 0.0)
one_pass = w6.stats()
check("a single pass produces no revisits", one_pass["revisited_m2"] == 0.0,
      f"{one_pass['revisited_m2']:.2f} m^2")
check("...and a revisit ratio of zero", one_pass["revisit_ratio"] == 0.0)

for i in range(10):                                    # drive back over the same line
    w6.set_pose(12.0 + 0.15 * (9 - i), 12.0, 0.0)
back = w6.stats()
check("driving back over ground registers revisits", back["revisited_m2"] > 0,
      f"{back['revisited_m2']:.2f} m^2, ratio {back['revisit_ratio']:.2f}")
check("re-treading does NOT inflate covered area",
      abs(back["visited_m2"] - one_pass["visited_m2"]) < 1e-9,
      f"{one_pass['visited_m2']:.2f} -> {back['visited_m2']:.2f} m^2")

# A PARKED robot must not invent revisits. The pose updates 10x a second whether or
# not the wheels turn; counting per update rather than per ENTRY would have a
# stationary robot reporting thousands of revisits and every circling alarm firing.
before_parked = w6.stats()["revisited_m2"]
for _ in range(300):
    w6.set_pose(12.0, 12.0, 0.0)
check("300 updates while parked add no revisits",
      w6.stats()["revisited_m2"] == before_parked)
check("visit_count reports a real number", w6.visit_count(12.0, 12.0) >= 1,
      f"{w6.visit_count(12.0, 12.0)}")
check("visit_count is 0 for somewhere never visited", w6.visit_count(40.0, 40.0) == 0)

# 11. THE VISIT TRACE MUST PERSIST. Phase 1 saved the occupancy grid but not this,
#     so the robot remembered what it had SEEN while forgetting where it had DRIVEN.
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "m.npz"
    a = WorldMap(map_file=path)
    for i in range(10):
        a.set_pose(12.0 + 0.15 * i, 12.0, 0.0)
        a.integrate(room, np.array([12.0 + 0.15 * i, 12.0, 0.0]))
    for i in range(10):
        a.set_pose(12.0 + 0.15 * (9 - i), 12.0, 0.0)
    src_stats = a.stats()
    a.save()
    check("saving writes a companion visits file", a.visits_file.exists(),
          a.visits_file.name)

    b = WorldMap(map_file=path)
    b.load()
    dst = b.stats()
    check("the visit trace is flagged as restored", b.visits_loaded is True)
    check("driven area survived the round trip",
          abs(dst["visited_m2"] - src_stats["visited_m2"]) < 1e-9,
          f"{src_stats['visited_m2']:.2f} -> {dst['visited_m2']:.2f} m^2")
    check("revisit counts survived the round trip",
          abs(dst["revisited_m2"] - src_stats["revisited_m2"]) < 1e-9,
          f"{src_stats['revisited_m2']:.2f} -> {dst['revisited_m2']:.2f} m^2")
    check("a restored cell keeps its exact count",
          b.visit_count(12.0, 12.0) == a.visit_count(12.0, 12.0),
          f"{a.visit_count(12.0, 12.0)} -> {b.visit_count(12.0, 12.0)}")

# 12. A map with no visits file must still load. Losing the trace is a degradation,
#     not a failure -- an occupancy map without one is still a useful map.
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "m.npz"
    a = WorldMap(map_file=path)
    for _ in range(6):
        a.integrate(room, np.array([12.0, 12.0, 0.0]))
    a.save()
    if a.visits_file.exists():
        a.visits_file.unlink()
    b = WorldMap(map_file=path)
    check("a map with no visits file still loads", b.load() is True)
    check("...and says the trace was not restored", b.visits_loaded is False)
    check("...and the occupancy survived", b.stats()["explored_m2"] > 0)

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
