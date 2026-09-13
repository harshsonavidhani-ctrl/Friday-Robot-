"""Short-term obstacle memory: remembered, moved with her, forgotten, and USED.

Harsh, 2026-09-13: "the lidar obstacles misses near it" and "the obstacle memory
should be used not just kept". Every case is one way that memory could be wrong:

    remembered after she turns        -> an obstacle seen ahead is behind after 180 deg
    forgotten on time                 -> dead reckoning drifts, so old points go
    cleared when the LiDAR sees past  -> a moved obstacle must not haunt her
    NOT cleared by a beam with nothing -> a dark surface returns nothing
    kept below the LiDAR floor        -> the blind band is the reason it exists
    ToF points never reach the rear   -> a pivot that did not happen must not block reverse
    the planner steers round it       -> it is USED, not just kept
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import RobotConfig                                  # noqa: E402
from near_memory import NearObstacleMemory, SRC_LIDAR, SRC_TOF     # noqa: E402
from dwa import DynamicWindowPlanner                               # noqa: E402

cfg = RobotConfig()
N = cfg.n_beams
MAX = cfg.lidar_max_range
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def empty_scan():
    return np.full(N, MAX, np.float32)


def scan_with(beam, r):
    s = empty_scan()
    s[beam % N] = r
    return s


print("near memory")
origin = (5.0, 5.0, 0.0)

# 1. Remembered, and moves with her: an object 0.5 m dead ahead, then she turns 180.
m = NearObstacleMemory(cfg, ttl_s=4.0)
m.update(origin, scan_with(0, 0.5), now=0.0)
pts = m.points(origin)
check("an object seen ahead is remembered", len(pts) >= 1 and abs(pts[0][0] - 0.5) < 0.02,
      f"{pts[:1].tolist()}")
turned = (5.0, 5.0, math.pi)
m.update(turned, empty_scan(), now=0.5)            # sensors now see nothing
pts = m.points(turned)
check("after a 180 deg pivot it is BEHIND her", len(pts) >= 1 and pts[0][0] < -0.45,
      f"{pts[:1].tolist()}")

# 2. Forgotten on time.
m.update(turned, empty_scan(), now=5.0)
check("forgotten after its TTL", len(m) == 0, f"left={len(m)}")

# 3. Cleared when a live beam returns from beyond it.
m = NearObstacleMemory(cfg, ttl_s=4.0)
m.update(origin, scan_with(0, 0.5), now=0.0)
m.update(origin, scan_with(0, 1.5), now=0.2)       # the same beam now sees 1.5 m
check("cleared when the LiDAR sees past it", len(m.points(origin, (SRC_LIDAR,))) == 0,
      f"left={len(m)} cleared={m.cleared}")

# 4. NOT cleared by a beam that returned nothing (dark, absorbing surface).
m = NearObstacleMemory(cfg, ttl_s=4.0)
m.update(origin, scan_with(0, 0.5), now=0.0)
m.update(origin, empty_scan(), now=0.2)
check("not cleared by a beam with no return", len(m.points(origin)) >= 1)

# 5. Kept when she drives up to it and it drops below the LiDAR floor.
m = NearObstacleMemory(cfg, ttl_s=4.0)
m.update(origin, scan_with(0, 0.40), now=0.0)
closer = (5.30, 5.0, 0.0)                          # drove 0.30 m: now 0.10 m away
m.update(closer, empty_scan(), now=1.0)            # the LiDAR discards it now
vs = m.virtual_scan(closer, N)
check("kept below the LiDAR floor, in the virtual scan",
      vs[0] < cfg.lidar_floor and abs(vs[0] - 0.10) < 0.03, f"beam0={vs[0]:.3f}")

# 6. ToF points steer, but never reach the rear-gate query.
m = NearObstacleMemory(cfg, ttl_s=4.0)
tof_wall = np.array([[0.25, y] for y in np.linspace(-0.2, 0.2, 5)], np.float32)
m.update(origin, empty_scan(), tof_points=tof_wall, now=0.0)
m.update(turned, empty_scan(), now=0.3)
behind_all = m.virtual_scan(turned, N)
behind_lidar = m.virtual_scan(turned, N, sources=(SRC_LIDAR,))
check("a ToF wall is remembered for steering", float(np.min(behind_all)) < 0.35,
      f"min={float(np.min(behind_all)):.2f}")
check("...but never appears in the LiDAR-only (rear gate) view",
      float(np.min(behind_lidar)) >= MAX - 1e-6)

# 7. Blind (chassis) beams are never remembered.
m = NearObstacleMemory(cfg, ttl_s=4.0)
m.update(origin, scan_with(18, 0.3), blind_beams=np.array([18]), now=0.0)
check("a masked chassis beam is not remembered", len(m) == 0, f"n={len(m)}")

# 8. Bounded: a dense, noisy room does not grow memory without limit.
m = NearObstacleMemory(cfg, ttl_s=100.0)
rng = np.random.default_rng(1)
for k in range(300):
    pose = (5.0 + rng.uniform(-1, 1), 5.0 + rng.uniform(-1, 1), rng.uniform(-3, 3))
    m.update(pose, rng.uniform(0.25, 0.95, N).astype(np.float32), now=k * 0.1)
check("memory stays bounded", len(m) <= 800, f"n={len(m)}")

# 9. USED: the arc planner steers round a remembered obstacle it can no longer see.
planner = DynamicWindowPlanner(cfg)
m = NearObstacleMemory(cfg, ttl_s=4.0)
wall = np.array([[0.45, y] for y in np.linspace(-0.15, 0.15, 7)], np.float32)
m.update(origin, empty_scan(), tof_points=wall, now=0.0)
blind_now = empty_scan()                           # nothing live sees it any more
without = planner.choose(blind_now, 0.3, 0.0, v_floor=0.15)
with_mem = planner.choose(blind_now, 0.3, 0.0, v_floor=0.15,
                          extra_obstacles=m.points(origin))
check("without memory she drives straight at it",
      without is not None and abs(without.omega) < 0.2,
      f"omega={getattr(without, 'omega', None)}")
check("with memory she steers away (or declines the arc)",
      with_mem is None or abs(with_mem.omega) >= 0.2,
      f"omega={getattr(with_mem, 'omega', None)}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
