"""Arc selection and situational navigation.

Every case here is one of Harsh's complaints turned into an assertion, so that a
regression fails the suite instead of showing up on the floor:

    "traversing in a zig-zag way in an open straight hallway"  -> straight-line cases
    "it hessitates a lot"                                      -> goes-around cases
    "lidar misses obstacles"                                   -> collision rejection
    "navigate with confidence"                                 -> commitment cases

The scan convention is pi_deploy's: n_beams beams, beam i at i*(360/n) degrees, beam 0
down the nose, ranges in metres, `lidar_max_range` meaning "this beam saw nothing".
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import RobotConfig                                  # noqa: E402
from dwa import DynamicWindowPlanner, SIM_TIME_S, SIM_STEPS        # noqa: E402
from navigator import (Navigator, OPEN, CORRIDOR, CLUTTERED,       # noqa: E402
                       BLOCKED, COMMIT_MARGIN)

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def clear_scan() -> np.ndarray:
    """Nothing anywhere: every beam reports max_range, i.e. 'saw nothing'."""
    return np.full(cfg.n_beams, cfg.lidar_max_range, dtype=np.float64)


def bearing_of(i: int) -> float:
    """Signed bearing of beam i, wrapped to +-pi."""
    b = i * (2.0 * math.pi / cfg.n_beams)
    return math.atan2(math.sin(b), math.cos(b))


def put(scan: np.ndarray, lo_deg: float, hi_deg: float, dist: float) -> np.ndarray:
    """Place returns at `dist` for every beam whose bearing is in [lo, hi] degrees."""
    for i in range(cfg.n_beams):
        d = math.degrees(bearing_of(i))
        if lo_deg <= d <= hi_deg:
            scan[i] = dist
    return scan


print("--- geometry and safety invariants ---")

# The property that makes point-sampling along an arc sound. If a step could ever be
# longer than the body clearance, a thin obstacle could fall between two samples and be
# driven straight through.
step_m = cfg.v_max * SIM_TIME_S / SIM_STEPS
clear_r = cfg.robot_radius + cfg.collision_margin
check("arc sampling cannot tunnel through an obstacle",
      step_m < clear_r, f"step {step_m:.3f} m vs clearance {clear_r:.3f} m")

dwa = DynamicWindowPlanner(cfg)

# 1. THE ZIG-ZAG. An open room with the goal dead ahead must produce a fast, straight
#    arc. This is the case that measured 515 deg/m on hardware.
res = dwa.choose(clear_scan(), v_allow=cfg.v_max, omega_want=0.0, desired_bearing=0.0)
check("open room + goal ahead -> drives", res is not None)
if res:
    check("...at full allowed speed", abs(res.v - cfg.v_max) < 1e-6, f"v={res.v:.2f}")
    check("...and essentially straight", abs(res.omega) < 0.30, f"omega={res.omega:+.2f}")

# 2. NEVER RAISES A CAP. The whole chain's invariant is that a stage may only reduce
#    authority; this stage is inserted into it and must obey it.
res = dwa.choose(clear_scan(), v_allow=0.20, omega_want=0.0, desired_bearing=0.0)
check("never exceeds the caller's speed allowance",
      res is not None and res.v <= 0.20 + 1e-9, f"v={res.v:.3f} vs allow 0.20")

# 3. A max_range beam is NOT an obstacle at 5 m -- it is a beam that saw nothing. The
#    converse mistake once carved free space straight through the chassis.
check("a scan of pure max_range yields no obstacles",
      dwa._obstacle_points(clear_scan(), None).shape[0] == 0)

# 4. Masked (chassis) beams must not block anything. Her own body is not an obstacle.
s = clear_scan()
s[0] = 0.25                                     # inside the swept corridor, dead ahead
blocked_with = dwa.choose(s, cfg.v_max, 0.0, desired_bearing=0.0)
blocked_without = dwa.choose(s, cfg.v_max, 0.0, desired_bearing=0.0,
                             blind_beams=np.array([0], dtype=np.int32))
check("a close return dead ahead is respected",
      blocked_with is None or abs(blocked_with.omega) > 0.2,
      "straight arc must not survive it")
check("...but the same return on a MASKED beam is ignored",
      blocked_without is not None and abs(blocked_without.omega) < 0.30,
      f"omega={blocked_without.omega:+.2f}" if blocked_without else "None")

# 5. Sub-footprint returns are geometrically impossible and already rejected upstream;
#    they must not be treated as obstacles here either.
s = clear_scan()
s[0] = 0.05
check("a sub-footprint return is not an obstacle",
      dwa._obstacle_points(s, None).shape[0] == 0)

print("--- hesitation: a wall ahead must produce a way round, not a stop ---")

# 6. THE HESITATION. A wall across the nose, open to the sides. She must choose an arc
#    around it -- turning while still moving -- not return None (which would hand the
#    tick to the escape machinery and, on hardware, spin in place).
s = put(clear_scan(), -25.0, 25.0, 0.50)
res = dwa.choose(s, cfg.v_max, 0.0, desired_bearing=0.0)
check("wall ahead, sides open -> still drives", res is not None)
if res:
    check("...by turning away from it", abs(res.omega) > 0.30, f"omega={res.omega:+.2f}")
    check("...while still moving forward", res.v > 0.0, f"v={res.v:.2f}")
    check("...on an arc that clears her body",
          res.clearance > clear_r, f"clearance {res.clearance:.2f} m")

# 7. Genuinely enclosed: every direction blocked at contact range. Nothing forward is
#    drivable and the honest answer is None -- NOT a command. The caller then falls
#    through to the escape/recovery path, which owns reverse.
s = np.full(cfg.n_beams, 0.30, dtype=np.float64)
check("boxed in -> declines, so the escape machinery keeps the tick",
      dwa.choose(s, cfg.v_max, 0.0, desired_bearing=0.0) is None)

print("--- situational assessment ---")

nav = Navigator(cfg, DynamicWindowPlanner(cfg))

a = nav.assess(clear_scan())
check("an empty room reads as OPEN", a.situation == OPEN, a.why)

s = clear_scan()
put(s, 60.0, 120.0, 0.60)                       # wall close on the left
put(s, -120.0, -60.0, 0.60)                     # wall close on the right
a = nav.assess(s)
check("walls both sides, clear ahead reads as CORRIDOR", a.situation == CORRIDOR, a.why)

s = np.full(cfg.n_beams, 0.30, dtype=np.float64)
a = nav.assess(s)
check("contact range all round reads as BLOCKED", a.situation == BLOCKED, a.why)

# 8. THE WIDEST GAP, NOT THE CENTRE OF MASS. A pillar dead ahead with clear air either
#    side has its centroid ON the pillar; steering by centroid drives into it. This is
#    the same assertion friday_freespace.py already makes for the camera.
s = put(clear_scan(), -12.0, 12.0, 0.70)
a = nav.assess(s)
check("a pillar ahead does not become the chosen gap",
      a.widest_gap_bearing is None or abs(a.widest_gap_bearing) > math.radians(15.0),
      f"gap bearing {math.degrees(a.widest_gap_bearing):+.0f} deg"
      if a.widest_gap_bearing is not None else "no gap")

print("--- confidence: commitment, and what may break it ---")

nav2 = Navigator(cfg, DynamicWindowPlanner(cfg))
nav2._commit_sign, nav2._commit_since = +1, time.monotonic()
nav2._commit_situation = CLUTTERED

# A rival that is only marginally better must NOT steal the decision -- that flip, ten
# times a second, is what a zig-zag is made of.
_, held, why = nav2._apply_commitment(-1.0, COMMIT_MARGIN * 0.4, CLUTTERED)
check("a marginally better rival does not break a commitment", held, why)

# ...but a clearly better one must, or commitment becomes stubbornness.
nav2._commit_sign, nav2._commit_since = +1, time.monotonic()
_, held, why = nav2._apply_commitment(-1.0, COMMIT_MARGIN * 2.0, CLUTTERED)
check("a clearly better rival does break it", not held, why)

# A changed situation invalidates the commitment: the reason for it is gone.
nav2._commit_sign, nav2._commit_since = +1, time.monotonic()
nav2._commit_situation = CORRIDOR
_, held, _ = nav2._apply_commitment(-1.0, 0.0, OPEN)
check("changing situation releases the commitment", not held)

print("--- end to end through Navigator.decide ---")

nav3 = Navigator(cfg, DynamicWindowPlanner(cfg))
d = nav3.decide(clear_scan(), v_allow=cfg.v_max, omega_want=0.0, desired_bearing=0.0)
check("open room -> a decision, not a handoff", d is not None)
if d:
    check("...labelled OPEN", d.situation == OPEN, d.why)
    check("...driving straight and fast",
          d.v > 0.3 and abs(d.omega) < 0.30, f"v={d.v:.2f} omega={d.omega:+.2f}")

d = nav3.decide(np.full(cfg.n_beams, 0.30), v_allow=cfg.v_max, omega_want=0.0)
check("boxed in -> hands the tick back (None)", d is None)
check("...and the handoff is counted", nav3.handoffs == 1, f"handoffs={nav3.handoffs}")

# A corridor must be driven SLOWER than an open room, from the same allowance. This is
# the "don't find out about slew in a doorway" rule.
nav4 = Navigator(cfg, DynamicWindowPlanner(cfg))
s = clear_scan()
put(s, 60.0, 120.0, 0.60)
put(s, -120.0, -60.0, 0.60)
d_corr = nav4.decide(s, v_allow=cfg.v_max, omega_want=0.0, desired_bearing=0.0)
d_open = Navigator(cfg, DynamicWindowPlanner(cfg)).decide(
    clear_scan(), v_allow=cfg.v_max, omega_want=0.0, desired_bearing=0.0)
check("a corridor is driven slower than an open room",
      d_corr is not None and d_open is not None and d_corr.v < d_open.v,
      f"corridor {d_corr.v:.2f} vs open {d_open.v:.2f}"
      if (d_corr and d_open) else "None")

print("--- speed authority is off by default, and bounded when on ---")

# The default must preserve the chain's stated invariant exactly: with authority 0 the
# navigator may only ever REDUCE. A default that quietly let it drive faster than the
# policy asked would be an architectural change smuggled in as a feature.
check("nav_speed_authority defaults to OFF", RobotConfig().nav_speed_authority == 0.0,
      f"default={RobotConfig().nav_speed_authority}")

# ...and when it IS on, it is still a cap, not a licence: nothing may exceed v_max.
cfg_auth = RobotConfig()
cfg_auth.nav_speed_authority = 0.6
allow = max(0.05, cfg_auth.v_max * cfg_auth.nav_speed_authority)
res = DynamicWindowPlanner(cfg_auth).choose(clear_scan(), allow, 0.0,
                                            desired_bearing=0.0)
check("authority raises the ceiling but never past v_max",
      res is not None and res.v <= cfg_auth.v_max + 1e-9
      and res.v <= allow + 1e-9, f"v={res.v:.2f} allow={allow:.2f}" if res else "None")

print("--- cost: this runs inside a 100 ms control budget ---")

# The Pi is CPU-bound, and tonight proved what that costs: the loop hit 172 ms and the
# reflex fired on 83% of ticks. A planner that is right but too slow is not right.
s = put(clear_scan(), -25.0, 25.0, 0.90)
dwa_t = DynamicWindowPlanner(cfg)
dwa_t.choose(s, cfg.v_max, 0.0, desired_bearing=0.0)          # warm numpy
t0 = time.perf_counter()
N = 50
for _ in range(N):
    dwa_t.choose(s, cfg.v_max, 0.0, desired_bearing=0.0)
per_ms = (time.perf_counter() - t0) * 1000.0 / N
check("choose() costs a small fraction of the control budget",
      per_ms < 25.0, f"{per_ms:.2f} ms/call (budget 100 ms, loop already ~62 ms)")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
