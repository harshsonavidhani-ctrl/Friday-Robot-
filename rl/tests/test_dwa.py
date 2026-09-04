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
                       BLOCKED, COMMIT_MARGIN, COMMIT_MAX_S, SPEED_OPEN)

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

# REGRESSION, and the one the unit tests above MISSED for two hardware runs. They fed
# `_apply_commitment` hand-picked margins (0.07, 0.36) and passed, while production
# passed it the winner's ABSOLUTE score -- about 1.5 against a 0.18 threshold -- so the
# commitment broke on every reversal and the feature was inert on the robot. The fix is
# only real if it is checked END TO END, through decide(), on values the robot actually
# produces. A unit test on a number nothing generates proves nothing.
# THE SCENE MUST BE A NEAR-TIE WITH NOISE, or the test cannot fail. A static scan makes
# DWA deterministic -- it returns the same argmax every tick whether commitment works or
# not -- and a first version of this check PASSED against a deliberately broken copy for
# exactly that reason. A zig-zag is noise breaking a tie between two near-equal options,
# so that is what has to be reproduced: a symmetric obstacle dead ahead (left and right
# are equally good) plus per-tick jitter of a centimetre or two.
# THE TWO PRODUCTION BUGS, tested at the level where they actually existed.
#
# An earlier attempt here drove `decide()` over an alternating scene and asserted she
# did not flip. It PASSED AGAINST A DELIBERATELY BROKEN COPY, three times, for three
# different reasons -- and the last one is worth recording: with commitment stubbed out
# entirely, `decide()` was ALREADY stable, because the widest-gap blending fixes the
# steering target in CLUTTERED regardless. So that test never measured commitment at
# all. These do, and they are honest about being unit-level.

# BUG 1: the margin was the winner's ABSOLUTE score (~1.5 in production) compared
# against COMMIT_MARGIN (0.18), so every reversal "earned it" and nothing ever held.
nav_m = Navigator(cfg, DynamicWindowPlanner(cfg))
nav_m._commit_sign, nav_m._commit_since = +1, time.monotonic()
nav_m._commit_situation = CLUTTERED
_, held, _ = nav_m._apply_commitment(-1.0, 1.5, CLUTTERED)     # a production-scale score
check("a production-scale SCORE passed as a margin would break every commitment",
      not held, "so the margin must be a difference, not a score -- see BUG 1")
nav_m._commit_sign, nav_m._commit_since = +1, time.monotonic()
_, held, _ = nav_m._apply_commitment(-1.0, 0.05, CLUTTERED)    # a real margin
check("...and a real margin under the threshold holds", held)

# BUG 2: the "held" branch returned the RIVAL's omega, so holding and switching
# produced identical commands and the mechanism was inert. decide() must return the
# committed direction's own arc, which it can only do from `alternatives`.
s_two = put(clear_scan(), -14.0, 14.0, 0.85)
res_two = DynamicWindowPlanner(cfg).choose(s_two, cfg.v_max, 0.0, desired_bearing=0.0)
check("choose() reports the best arc for EACH turn direction",
      res_two is not None and {-1, 1} <= set(res_two.alternatives),
      f"alternatives: {sorted(res_two.alternatives)}" if res_two else "None")
if res_two and {-1, 1} <= set(res_two.alternatives):
    winner_sign = 1 if res_two.omega > 0 else -1
    other = res_two.alternatives[-winner_sign]
    check("...and the losing direction's arc points the other way",
          (other[2] > 0) != (res_two.omega > 0),
          f"winner {res_two.omega:+.2f} vs held-direction {other[2]:+.2f}")

# WHAT IS *NOT* COVERED HERE, stated rather than papered over.
#
# There is no end-to-end test proving the margin fix changes what the robot does, and
# four attempts to write one all failed for instructive reasons. The last and most
# important: through `decide()`, the widest-gap blending sets the steering target and
# breaks the tie between the two sides BEFORE commitment is ever consulted -- measured
# margin 0.32 against a 0.18 threshold on a geometrically symmetric obstacle, where
# switching is then the correct answer. Stubbing commitment out entirely left decide()
# just as stable.
#
# So the honest conclusion is that commitment matters less in practice than it looks,
# because something else is already doing that job. The two bugs below were real and are
# fixed; their behavioural effect on hardware is UNPROVEN, and the run summary now
# separates `switched` from `expired` so the next run can actually measure it.

# A commitment to a direction with no surviving arc must YIELD, not steer her into the
# side the collision check just rejected.
nav_y = Navigator(cfg, DynamicWindowPlanner(cfg))
nav_y._commit_sign, nav_y._commit_since = +1, time.monotonic()
s_left_blocked = put(clear_scan(), 5.0, 90.0, 0.30)     # everything left is unusable
d_y = nav_y.decide(s_left_blocked, v_allow=cfg.v_max, omega_want=0.0,
                   desired_bearing=0.0)
check("a commitment to a blocked side does not survive",
      d_y is None or d_y.omega <= 0.0,
      f"omega={d_y.omega:+.2f}" if d_y else "handed back (also correct)")

# ...but a clearly better one must, or commitment becomes stubbornness.
nav2._commit_sign, nav2._commit_since = +1, time.monotonic()
_, held, why = nav2._apply_commitment(-1.0, COMMIT_MARGIN * 2.0, CLUTTERED)
check("a clearly better rival does break it", not held, why)

# REGRESSION, measured on hardware 2026-09-04: commitment used to be released whenever
# the situation LABEL changed, and over a 6-minute run that broke 121 commitments
# against 56 held. The label flickers between CORRIDOR and CLUTTERED as she moves --
# it is a threshold on a noisy range -- so the mechanism meant to provide steadiness
# was itself chasing the noise. A label change is not evidence about direction.
nav2._commit_sign, nav2._commit_since = +1, time.monotonic()
nav2._commit_situation = CORRIDOR
_, held, why = nav2._apply_commitment(+1.0, 0.0, CLUTTERED)
check("a flickering situation label does NOT break a commitment", held, why)

# It must still expire on its own, or commitment becomes stubbornness.
nav2._commit_sign = +1
nav2._commit_since = time.monotonic() - (COMMIT_MAX_S + 1.0)
_, held, why = nav2._apply_commitment(-1.0, 0.0, CLUTTERED)
check("...but it still expires with age", not held, why)

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

print("--- the near field: what the LiDAR structurally cannot see ---")

# THE WEDGE. The LiDAR discards everything below `lidar_floor` (0.22 m, her own body)
# and is masked across the chassis; the ToF reflex stops at 0.30 m. So an object in that
# band is invisible to the planner and fatal to the reflex, and she drives into it and
# is slammed to a halt -- measured as navigator engaged 94% of ticks while the reflex
# fired on 59% of them, and every run ending wedged.
nav_tof = Navigator(cfg, DynamicWindowPlanner(cfg))
clear_tof = np.full((8, 8), cfg.tof_max_range, dtype=np.float32)

# A totally clear ToF must add nothing at all.
check("a clear ToF contributes no obstacles",
      nav_tof._tof_obstacle_points(clear_tof).shape[0] == 0)
check("a missing ToF contributes no obstacles",
      nav_tof._tof_obstacle_points(None).shape[0] == 0)

# One hot zone is speckle, not an object -- the SAME quorum the reflex uses, so the two
# cannot disagree about what counts.
speckle = clear_tof.copy()
speckle[3, 4] = 0.25
check("a single hot ToF zone is speckle, not an obstacle",
      nav_tof._tof_obstacle_points(speckle).shape[0] == 0)

# ...and a real cluster is an obstacle laid across the corridor she sweeps.
real = clear_tof.copy()
real[4:7, 3:6] = 0.28
pts = nav_tof._tof_obstacle_points(real)
check("a real ToF cluster becomes obstacle points", pts.shape[0] > 0,
      f"{pts.shape[0]} points at x={pts[0][0]:.2f} m" if pts.shape[0] else "none")
check("...spanning the corridor her body sweeps, not a single ray",
      pts.shape[0] and abs(pts[:, 1].max() - pts[:, 1].min() - 2 * cfg.corridor_half) < 1e-6,
      f"span {pts[:, 1].max() - pts[:, 1].min():.3f} m vs corridor {2*cfg.corridor_half:.3f} m")

# THE POINT OF ALL OF IT: with the LiDAR reporting a wide open room, an obstacle only
# the ToF can see must still change what she does.
nav_a = Navigator(cfg, DynamicWindowPlanner(cfg))
blind = nav_a.decide(clear_scan(), v_allow=cfg.v_max, omega_want=0.0,
                     desired_bearing=0.0, tof_depth=clear_tof)
nav_b = Navigator(cfg, DynamicWindowPlanner(cfg))
seeing = nav_b.decide(clear_scan(), v_allow=cfg.v_max, omega_want=0.0,
                      desired_bearing=0.0, tof_depth=real)
check("an obstacle ONLY the ToF can see still changes the command",
      blind is not None and (seeing is None or abs(seeing.omega) > abs(blind.omega)
                             or seeing.v < blind.v),
      f"lidar-only v={blind.v:.2f} w={blind.omega:+.2f} -> with ToF "
      f"v={seeing.v:.2f} w={seeing.omega:+.2f}" if (blind and seeing) else "handed back")

# THE BARRIER IS RAISED AT EVERY RANGE THE ToF REPORTS, and the speed law runs on top
# of it. Both, always.
#
# THIS IS THE ASSERTION THAT WAS BRIEFLY INVERTED AND IT COST THE WORST REGRESSION OF
# THE PROJECT. The argument for inverting it sounded right -- "a planner must not be
# stricter than the emergency layer it feeds", since ToFReflex at 0.45 m merely scales
# speed while the planner was deleting arcs. Measured back to back on hardware:
#
#     |                     | barrier at every range | only inside stop |
#     | ground driven       | 16.1 m^2               | 5.6 m^2          |
#     | ToF reflex firing   | 30%                    | 64%              |
#     | navigator handbacks | 49%                    | 75%              |
#
# The argument is wrong because THE REFLEX ONLY BRAKES BECAUSE IT CANNOT STEER: it
# holds one k-th-smallest range with no bearing, so slowing is the only move available
# to it. The planner holds the whole arc set, and the barrier is the only channel by
# which the near field reaches that search. Deleting arcs that run INTO something while
# keeping every arc AROUND it is not a stricter version of capping speed; it is a
# different operation, and it is the entire reason this layer exists.
mid = np.full((8, 8), cfg.tof_max_range, dtype=np.float32)
mid[4:7, 3:6] = 0.45                          # inside slow, well outside stop
pts_mid, scale_mid = Navigator(cfg, DynamicWindowPlanner(cfg))._tof_constraint(mid)
check("a return the reflex would only slow for STILL raises a barrier to steer around",
      pts_mid.shape[0] > 0, f"{pts_mid.shape[0]} barrier points at 0.45 m")
check("...and scales speed by the reflex's own law as well",
      abs(scale_mid - (0.45 - cfg.tof_stop_distance)
          / (cfg.tof_slow_distance - cfg.tof_stop_distance)) < 1e-6,
      f"scale={scale_mid:.3f}")     # 1e-6, not 1e-9: the frame is float32

# The two must agree at the threshold: inside the stop distance, both refuse.
inside = np.full((8, 8), cfg.tof_max_range, dtype=np.float32)
inside[4:7, 3:6] = cfg.tof_stop_distance - 0.02
pts_in, scale_in = Navigator(cfg, DynamicWindowPlanner(cfg))._tof_constraint(inside)
check("inside the reflex's stop distance it raises a barrier and refuses speed",
      pts_in.shape[0] > 0 and scale_in == 0.0, f"{pts_in.shape[0]} points")

# Beyond the reflex's interest the barrier is still there -- that is the band where it
# does its most valuable work, steering around something while there is still room to
# steer -- but it costs her NO speed, because the reflex would not have slowed her.
far_tof = np.full((8, 8), cfg.tof_max_range, dtype=np.float32)
far_tof[4:7, 3:6] = cfg.tof_slow_distance + 0.10
pts_far, scale_far = Navigator(cfg, DynamicWindowPlanner(cfg))._tof_constraint(far_tof)
check("beyond the reflex's slow distance the barrier stands but costs no speed",
      pts_far.shape[0] > 0 and scale_far == 1.0,
      f"{pts_far.shape[0]} points, scale={scale_far:.2f}")

# AND IT MUST ACT BEFORE THE REFLEX DOES, or it has changed nothing. The reflex brakes
# from tof_slow_distance (0.60 m) and stops at tof_stop_distance (0.30 m). Steering away
# only once the brakes are already on is the old behaviour with extra steps; the whole
# point is to have turned before then.
def nav_at(range_m: float):
    t = np.full((8, 8), cfg.tof_max_range, dtype=np.float32)
    t[4:7, 3:6] = range_m
    return Navigator(cfg, DynamicWindowPlanner(cfg)).decide(
        clear_scan(), v_allow=cfg.v_max, omega_want=0.0, desired_bearing=0.0,
        tof_depth=t)

# SHE MUST HAVE TURNED BEFORE THE BRAKES COME ON. The reflex brakes from 0.60 m and
# stops at 0.30 m; steering away only once the brakes are on is the old behaviour with
# extra steps. At 0.80 m nothing downstream cares yet, so any turn here is the barrier's
# doing and nothing else's -- which is precisely the run-4 behaviour ("she turns away at
# 0.80 m, before the 0.60 m brake and the 0.30 m stop") that measured 16.1 m^2.
early = nav_at(cfg.tof_slow_distance - 0.05)     # 0.55 m: reflex brakes, does not stop
full = nav_at(cfg.tof_slow_distance + 0.20)      # 0.80 m: outside the reflex's interest
check("she steers away while the way ahead is still wide open (0.80 m)",
      full is not None and abs(full.omega) > 1e-6,
      f"0.80 m -> w={full.omega:+.2f}" if full else "handed back")
check("...at no cost in speed out there",
      full is not None and full.v >= cfg.v_max * SPEED_OPEN - 1e-6,
      f"v={full.v:.2f}" if full else "None")
check("the near field also slows her before the reflex would stop her",
      early is not None and full is not None and early.v < full.v,
      f"0.80 m -> v={full.v:.2f}; 0.55 m -> v={early.v:.2f}"
      if (early and full) else "None")
check("...without deleting her options, which is what handing back would mean",
      early is not None, "still steering, not handed back")

# ...and the response is graduated, not a cliff: closer means slower.
far, near_ = nav_at(0.55), nav_at(0.45)
check("...and slows further as it closes",
      far is not None and near_ is not None and near_.v < far.v,
      f"0.55 m -> v={far.v:.2f}; 0.45 m -> v={near_.v:.2f}"
      if (far and near_) else "None")

# Inside the stop distance there is no honest forward answer, and it must say so rather
# than invent one -- escape and recovery own reverse, and they are already tested.
check("inside the reflex stop distance it hands the tick back",
      nav_at(cfg.tof_stop_distance - 0.02) is None)

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
#
# THE BENCHMARK MUST BE THE WORST CASE, NOT A CONVENIENT ONE. The first version of this
# timed a scan that was almost all max_range -- and max_range beams are DROPPED as "saw
# nothing", so it was timing an empty obstacle set and reported 2.4 ms. The real room is
# a cluttered one where every beam returns: that is the (M, K, N) array at full size,
# and it is the only number worth defending. Measured on hardware, the navigator took
# the loop mean from 62 ms to 81 ms, so this is not free.
dense = np.linspace(0.8, 2.5, cfg.n_beams).astype(np.float64)   # every beam returns
dwa_t = DynamicWindowPlanner(cfg)
dwa_t.choose(dense, cfg.v_max, 0.0, desired_bearing=0.0)        # warm numpy
t0 = time.perf_counter()
N = 50
for _ in range(N):
    dwa_t.choose(dense, cfg.v_max, 0.0, desired_bearing=0.0)
per_ms = (time.perf_counter() - t0) * 1000.0 / N
check("choose() on a FULLY POPULATED scan fits the control budget",
      per_ms < 25.0, f"{per_ms:.2f} ms/call (budget 100 ms, loop already ~62 ms)")

# ...and the optimisation must not have changed the answer. Squared-distance comparison
# is exactly equivalent to comparing distances, and float32 has far more precision than
# a centimetre-accurate LiDAR needs -- but "should be equivalent" is not evidence.
s_ref = put(clear_scan(), -25.0, 25.0, 0.50)
ref = DynamicWindowPlanner(cfg).choose(s_ref, cfg.v_max, 0.0, desired_bearing=0.0)
check("clearance is still reported in metres, not squared metres",
      ref is not None and 0.0 < ref.clearance < cfg.lidar_max_range,
      f"clearance={ref.clearance:.3f} m" if ref else "None")

# ======================================================================================
# THE L-SHAPED CORNER. Harsh, 2026-09-04, after the best run yet: "it was still
# stuttering in a L-SHAPED corner/turns it hesitates a lot there".
#
# Turning INTO a corner blocks the inside of the turn -- that is what a corner is -- so
# the committed direction loses every collision-free arc for a tick or two mid-turn.
# That was read as "the other way is infinitely better" (decide() sets margin=inf when
# the committed side has no arc), so she switched; the mirror image happened next tick
# and she switched back. Measured over the 20-minute run: omega changed sign 103 times
# per 1000 driving ticks in the 0.30-0.60 m band against 22 in open space.
# ======================================================================================
print("--- the L-shaped corner ---")

from navigator import COMMIT_STARVE_TICKS                          # noqa: E402

nav = Navigator(cfg, DynamicWindowPlanner(cfg))
nav._commit_sign = 1                       # committed to turning LEFT, mid-corner
nav._commit_since = time.monotonic()

# The inside of the turn has no surviving arc; the planner's winner is on the right.
held, switches = [], []
for tick in range(COMMIT_STARVE_TICKS - 1):
    _, committed, why = nav._apply_commitment(-0.8, float("inf"), CLUTTERED,
                                              starved=True)
    held.append(nav._commit_sign)
check("a starved tick does not flip the commitment",
      all(s == 1 for s in held) and len(held) == COMMIT_STARVE_TICKS - 1,
      f"commit_sign after {len(held)} starved ticks: {held}")

# ...and the moment the inside of the turn opens again, she resumes it rather than
# having thrown the manoeuvre away.
_, committed, why = nav._apply_commitment(+0.8, 0.05, CLUTTERED, starved=False)
check("...and she resumes the turn when the inside opens up",
      nav._commit_sign == 1 and committed, f"sign={nav._commit_sign} why={why}")

# A DEAD END IS STILL A DEAD END. Held intent must not become a robot driving at a
# wall: once the direction is genuinely gone it has to be given up, and quickly.
# An ABSOLUTE bound, not `range(COMMIT_STARVE_TICKS)`. The first version of this
# looped exactly COMMIT_STARVE_TICKS times, so it scaled with the constant and passed
# happily when the constant was mutated to 9999 -- a robot that would hold a dead-end
# commitment for a quarter of an hour. What matters is that a dead end is recognised
# FAST, so the deadline is written here in ticks and does not move.
DEAD_END_TICKS = 5                         # 0.5 s at 10 Hz
nav2 = Navigator(cfg, DynamicWindowPlanner(cfg))
nav2._commit_sign = 1
nav2._commit_since = time.monotonic()
for _ in range(DEAD_END_TICKS):
    nav2._apply_commitment(-0.8, float("inf"), CLUTTERED, starved=True)
check(f"a direction that is genuinely gone is given up within {DEAD_END_TICKS} ticks",
      nav2._commit_sign == -1 and nav2.commit_starve_switches == 1,
      f"sign={nav2._commit_sign} starve_switches={nav2.commit_starve_switches} "
      f"(COMMIT_STARVE_TICKS={COMMIT_STARVE_TICKS})")

# THE SAFETY PROPERTY. Holding the INTENT must never mean driving the rejected arc.
# While starved, _apply_commitment reports committed=False precisely so decide() uses
# the planner's collision-free winner and never the committed side's.
nav3 = Navigator(cfg, DynamicWindowPlanner(cfg))
nav3._commit_sign = 1
nav3._commit_since = time.monotonic()
om, committed, _ = nav3._apply_commitment(-0.8, float("inf"), CLUTTERED, starved=True)
check("holding the intent still drives the SAFE arc, not the committed one",
      committed is False and abs(om + 0.8) < 1e-9,
      f"committed={committed} omega={om:+.2f}")

# And a rival that genuinely scores better on an AVAILABLE arc still wins, so this
# has not turned the commitment into a lock.
nav4 = Navigator(cfg, DynamicWindowPlanner(cfg))
nav4._commit_sign = 1
nav4._commit_since = time.monotonic()
nav4._apply_commitment(-0.8, COMMIT_MARGIN + 0.5, CLUTTERED, starved=False)
check("a genuinely better rival still takes the decision",
      nav4._commit_sign == -1, f"sign={nav4._commit_sign}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
