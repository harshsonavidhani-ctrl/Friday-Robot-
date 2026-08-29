"""Frontier target selection, and the arbitration that must never overstep.

The planner is the one component here that can steer the robot without being the
policy and without being a safety veto. Two things therefore have to hold absolutely:

  1. It may influence HEADING ONLY. If a planner could raise forward speed, a stale
     map could drive the robot into something the policy had already slowed for.
  2. Its influence is CAPPED. The policy sees local geometry at 10 Hz that a 5 cm
     occupancy grid updated at 0.5 Hz cannot; it has to keep enough authority to dodge.

Plus the ordinary target-lifecycle behaviour: choose, hold, retire on arrival,
abandon when stale, and hysteresis so two near-equal frontiers cannot make the robot
oscillate between them and drive nowhere.
"""
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from explorer import (ALIGN_TOL_RAD, MAX_BIAS_FRACTION, ExplorationPlanner,
                      steering_bias)
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner, RobotConfig,
                       SensorSlot)
from world import WorldMap

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
OMEGA = cfg.omega_max
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def mapped_world(n=40, seed=0):
    w = WorldMap()
    rng = np.random.default_rng(seed)
    for i in range(n):
        pose = np.array([12.0 + 0.05 * i, 12.0, 0.05 * i])
        w.integrate(rng.uniform(1.5, 5.0, cfg.n_beams).astype(np.float32), pose)
        w.set_pose(*pose)
    return w


print("--- frontier planner ---")

# 1. STEERING BIAS BOUNDS. These are the safety-relevant properties of the arbitration.
check("no target -> no bias", steering_bias(None, 0.0, OMEGA)[0] == 0.0)
check("aligned -> no bias", steering_bias(0.1, 0.0, OMEGA)[0] == 0.0)
check("at the tolerance edge -> no bias",
      steering_bias(ALIGN_TOL_RAD * 0.999, 0.0, OMEGA)[0] == 0.0)

big_left, _ = steering_bias(math.pi * 0.95, 0.0, OMEGA)
big_right, _ = steering_bias(-math.pi * 0.95, 0.0, OMEGA)
check("a target behind produces a large LEFT bias for a left error", big_left > 0)
check("...and a large RIGHT bias for a right error", big_right < 0)
check("bias never exceeds the cap",
      abs(big_left) <= MAX_BIAS_FRACTION * OMEGA + 1e-9,
      f"{abs(big_left):.3f} <= {MAX_BIAS_FRACTION * OMEGA:.3f}")

# Proportional: the correction must grow with the error, not switch on at a threshold.
mags = [abs(steering_bias(e, 0.0, OMEGA)[0])
        for e in (0.7, 1.2, 1.8, 2.4, 3.0)]
check("bias grows monotonically with bearing error",
      all(b <= a + 1e-9 for a, b in zip(mags, mags[1:])) is False
      and all(a <= b + 1e-9 for a, b in zip(mags, mags[1:])),
      f"{[round(m, 3) for m in mags]}")

# Already committed to the right turn? Stay out of it -- adding to a hard turn the
# policy has already chosen is how a robot overshoots and starts weaving.
quiet, why = steering_bias(1.5, 0.9 * OMEGA, OMEGA)
check("no bias when the policy is already turning hard the right way",
      quiet == 0.0, why)
loud, _ = steering_bias(1.5, -0.9 * OMEGA, OMEGA)
check("...but a bias WHEN the policy is turning the wrong way", loud > 0)

# 2. TARGET SELECTION on a real map.
w = mapped_world()
w.set_pose(13.0, 12.0, 0.0)
p = ExplorationPlanner(w, period_s=0.2)
p._plan_once()
s = p.snapshot()
check("a target is chosen from a real map", s["has_target"], str(s["target_xy"]))
check("the target has expected information gain", s["target_gain"] > 0, f"{s['target_gain']}")
check("the target has a real driving distance", (s["target_distance"] or 0) > 0,
      f"{s['target_distance']:.2f} m")
check("frontiers were actually found", s["frontiers"] > 0, f"{s['frontiers']}")
check("planning reported no error", s["error"] == "", repr(s["error"]))

# 3. BEARING. The sign must match omega's: positive = turn left.
w2 = WorldMap()
p2 = ExplorationPlanner(w2)


class _FakeTarget:
    def __init__(self, xy):
        self.approach = xy
        self.gain, self.distance, self.utility = 1, 1.0, 1.0

    def key(self, cell_m=0.25):
        return (int(self.approach[0] / cell_m), int(self.approach[1] / cell_m))


p2.target = _FakeTarget((13.0, 12.0))          # due east of a robot at (12,12)
p2.target_since = time.monotonic()
check("a target dead ahead has ~zero bearing error",
      abs(p2.bearing_error(np.array([12.0, 12.0, 0.0]))) < 1e-6)
err_left = p2.bearing_error(np.array([12.0, 12.0, -math.pi / 2]))
check("a target to the LEFT gives a positive error",
      err_left > 0, f"{math.degrees(err_left):.0f} deg")
err_right = p2.bearing_error(np.array([12.0, 12.0, math.pi / 2]))
check("a target to the RIGHT gives a negative error",
      err_right < 0, f"{math.degrees(err_right):.0f} deg")
check("bearing error is wrapped to +-pi",
      abs(p2.bearing_error(np.array([12.0, 12.0, 3.0]))) <= math.pi + 1e-9)

# 4. TARGET LIFECYCLE.
check("distance to target is real", abs(p2.distance_to_target(np.array([12.0, 12.0, 0.0])) - 1.0) < 1e-6)
p2.note_progress(np.array([12.0, 12.0, 0.0]))
check("a distant target is NOT retired", p2.target is not None)
p2.note_progress(np.array([12.9, 12.0, 0.0]))       # within REACHED_M
check("arriving retires the target", p2.target is None)
check("...and counts as an ARRIVAL, not merely revealed",
      p2.targets_arrived == 1 and p2.targets_revealed == 0)

# 5. NO MAP, NO CRASH. A planner on an empty world must simply have no target.
p3 = ExplorationPlanner(WorldMap(), period_s=0.1)
p3._plan_once()
check("an empty world yields no target and no exception", p3.snapshot()["has_target"] is False)

# 6. THE CRITICAL ONE: the planner may not touch forward speed.
#    Run identical tick sequences with the planner on and off and compare v.
def run_ticks(with_planner, n=25):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg)
    loop = ControlLoop(cfg, policy, NullMotorDriver(cfg), slots)
    loop.map_enabled = False
    if with_planner:
        pl = ExplorationPlanner(loop.world, period_s=999.0)   # never replans
        pl.target = _FakeTarget((99.0, 99.0))                 # far off to one side
        pl.target_since = time.monotonic()
        pl.reached_m = 0.0
        loop.planner = pl
    vs, ws = [], []
    for i in range(n):
        slots["lidar"].publish(np.full(cfg.n_beams, 3.0 + 0.01 * i, np.float32))
        row = loop.tick()
        vs.append(row["v"])
        ws.append(row["omega"])
    return np.array(vs), np.array(ws)


v_off, w_off = run_ticks(False)
v_on, w_on = run_ticks(True)
check("the planner does NOT change forward speed",
      np.allclose(v_off, v_on, atol=1e-9),
      f"max delta {np.abs(v_off - v_on).max():.3e} m/s")
check("...but it DOES change heading", not np.allclose(w_off, w_on, atol=1e-6),
      f"max delta {np.abs(w_off - w_on).max():.3f} rad/s")
check("biased omega stays within omega_max", np.abs(w_on).max() <= OMEGA + 1e-9,
      f"max |omega| {np.abs(w_on).max():.3f}")

# 7. PROVENANCE. A biased tick must be labelled PLANNER, never POLICY.
slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
         "tof": SensorSlot("tof", cfg.tof_deadline, True)}
loop = ControlLoop(cfg, PolicyRunner(HERE / "exploration_model.onnx", cfg),
                   NullMotorDriver(cfg), slots)
loop.map_enabled = False
pl = ExplorationPlanner(loop.world, period_s=999.0)
pl.target = _FakeTarget((99.0, -99.0))
pl.target_since = time.monotonic()
pl.reached_m = 0.0
loop.planner = pl
auths = []
for i in range(20):
    slots["lidar"].publish(np.full(cfg.n_beams, 3.0, np.float32))
    loop.tick()
    auths.append(loop.last_record.authority)
import telemetry as _tlm
check("a biased tick reports EXPLORATION_PLANNER", _tlm.SRC_PLANNER in auths,
      f"seen: {sorted(set(auths))}")
check("the record carries the planner's own context",
      loop.last_record.planner is not None
      and "bearing_error" in loop.last_record.planner)
check("raw policy output is still untouched by the bias",
      loop.last_record.stages[0].omega == loop.last_record.omega_raw)

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
