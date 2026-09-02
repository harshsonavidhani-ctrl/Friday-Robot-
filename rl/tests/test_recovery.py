"""Stuck detection and recovery.

The hardest requirement here is the NEGATIVE one. A stuck detector that fires on a
healthy robot is worse than none at all: it interrupts good driving, the interruption
looks like a fault, and the natural response is to switch the whole thing off. So the
false-positive tests matter more than the true-positive ones and are given the longest
runs.

The second requirement is that detection reads EXECUTED motion. On this drivetrain the
creep cutoff turns a sub-floor request into a full stop and the stiction floor lifts a
small one, so commanded and executed differ structurally rather than as noise. A
detector fed commands would report healthy travel while the robot sat buzzing against
a wall -- which is the exact failure that made --motor-min-duty 72 a default.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import telemetry as tlm
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner, RobotConfig,
                       SensorSlot)
from recovery import (LEVELS, OSC_WINDOW, PROGRESS_WINDOW, RecoveryController,
                      SPIN_WINDOW, VERIFY_TICKS)

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
SCAN = np.full(cfg.n_beams, 2.0, np.float32)
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def run(rc, n, ev, ew, cv, boxed=False, open_side=1.0):
    """Feed n ticks; return the first tick index at which recovery fired."""
    first = None
    for i in range(n):
        r = rc.update(SCAN, ev(i) if callable(ev) else ev,
                      ew(i) if callable(ew) else ew,
                      cv, boxed, np.zeros(3), i, open_side)
        if r is not None and first is None:
            first = i
    return first


print("--- stuck detection and recovery ---")

# 1. FALSE POSITIVES. These must never fire. Long runs, on purpose.
rc = RecoveryController(cfg)
check("healthy driving never triggers recovery",
      run(rc, 400, 0.30, lambda i: 0.15 * math.sin(i / 9.0), 0.30) is None,
      f"{rc.recoveries} recoveries")

rc = RecoveryController(cfg)
check("driving dead straight never triggers recovery",
      run(rc, 400, 0.35, 0.0, 0.35) is None, f"{rc.recoveries} recoveries")

# A deliberate, sustained pivot is a legitimate manoeuvre when the robot is also
# travelling; only rotation WITHOUT travel is spinning.
rc = RecoveryController(cfg)
check("turning while travelling is not 'spinning'",
      run(rc, 200, 0.25, 1.2, 0.25) is None, f"{rc.recoveries} recoveries")

# A robot legitimately stopped (commanding nothing) is not "making no progress".
rc = RecoveryController(cfg)
check("a robot commanding zero is not flagged for no progress",
      run(rc, 200, 0.0, 0.0, 0.0) is None, f"{rc.recoveries} recoveries")

# 2. TRUE POSITIVES.
rc = RecoveryController(cfg)
t = run(rc, 40, 0.0, lambda i: 0.9 * (1 if i % 2 else -1), 0.20)
check("left-right oscillation is detected", t is not None, f"at tick {t}")
check("...and attributed to oscillation", rc.triggers["oscillation"] == 1)
check("...no sooner than its window allows", t is None or t >= OSC_WINDOW - 2, f"{t}")

rc = RecoveryController(cfg)
t = run(rc, 60, 0.001, 1.4, 0.20)
check("spinning in place is detected", t is not None, f"at tick {t}")
check("...and attributed to spinning", rc.triggers["spinning"] == 1)

rc = RecoveryController(cfg)
t = run(rc, 70, 0.0, 0.0, 0.30)
check("commanding motion while not moving is detected", t is not None, f"at tick {t}")
check("...and attributed to no_progress", rc.triggers["no_progress"] == 1)

rc = RecoveryController(cfg)
t = run(rc, 10, 0.0, 0.0, 0.0, boxed=True)
check("boxed-in is detected immediately", t == 0, f"at tick {t}")
check("...and attributed to boxed_in", rc.triggers["boxed_in"] == 1)

# 3. THE RECOVERY ACTION ITSELF. It must NEVER command forward motion -- that is the
# invariant. What it may do instead depends on the trap.
#
# boxed_in means "forward blocked", and a pivot cannot escape nose-against-a-wall:
# inside the pivot sweep the ToF reflex freezes rotation too, correctly. So recovery
# BACKS OFF there. It used to pivot unconditionally, which measured 84 pivots and zero
# translation while she sat 5.5 cm from a wall.
rc = RecoveryController(cfg)
out = rc.update(SCAN, 0.0, 0.0, 0.0, True, np.zeros(3), 0, 1.0)
check("recovery returns a command", out is not None)
v, omega, why = out
check("recovery NEVER commands forward motion", v <= 0.0, f"v={v}")
check("boxed-in backs off instead of pivoting", v == cfg.v_min, f"v={v}")
check("...while still turning toward open space", 0 < abs(omega) <= cfg.omega_max,
      f"omega={omega}")
check("...toward the open side given", omega > 0)
out2 = RecoveryController(cfg).update(SCAN, 0.0, 0.0, 0.0, True, np.zeros(3), 0, -1.0)
check("...and the other way when the other side is open", out2[1] < 0)
check("the reason names the trigger", "boxed_in" in why, why)

# 4. STATE MACHINE: recover, verify, stand down.
rc = RecoveryController(cfg)
rc.update(SCAN, 0.0, 0.0, 0.0, True, np.zeros(3), 0, 1.0)
check("state is RECOVERING after a trigger", rc.state == "RECOVERING", rc.state)
for i in range(LEVELS[0]["ticks"] + 2):                   # ride out the manoeuvre
    rc.update(SCAN, 0.0, 0.0, 0.0, False, np.zeros(3), i, 1.0)
check("state becomes VERIFY when the manoeuvre ends", rc.state == "VERIFY", rc.state)
for i in range(VERIFY_TICKS + 2):                          # now drive healthily
    rc.update(SCAN, 0.30, 0.05, 0.30, False, np.zeros(3), i, 1.0)
check("state returns to MONITOR once clear", rc.state == "MONITOR", rc.state)
check("the recovery is recorded as a success", rc.successes == 1, f"{rc.successes}")
check("the event is marked resolved", rc.events[0].resolved is True)
check("...with the tick count it took", rc.events[0].resolved_after_ticks is not None)

# 5. ESCALATION. Staying stuck must turn further, not repeat the same thing forever.
rc = RecoveryController(cfg)
for i in range(400):
    rc.update(SCAN, 0.0, 0.0, 0.30, True, np.zeros(3), i, 1.0)
check("a robot that stays stuck escalates", rc.escalations > 0, f"{rc.escalations}")
check("escalation stops at the last level", rc.level <= len(LEVELS) - 1, f"level {rc.level}")
check("the last level asks for the target to be blacklisted",
      getattr(rc, "blacklist_requested", False) is True)

# 6. EVENT LOG. Every recovery must be recorded with grounded detail.
rc = RecoveryController(cfg)
run(rc, 60, 0.001, 1.4, 0.20)
check("an event was logged", len(rc.events) == 1)
e = rc.events[0]
check("the event names the trigger", e.trigger == "spinning", e.trigger)
check("the event carries measured detail", "deg of rotation" in e.detail, e.detail)
check("the event records the pose", len(e.pose) == 3)
check("the snapshot exposes the counts", rc.snapshot()["recoveries"] == 1)

# 7. INTEGRATION: a recovery tick must be labelled RECOVERY_CONTROLLER, and the raw
#    policy output must survive it untouched.
slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
         "tof": SensorSlot("tof", cfg.tof_deadline, True)}
loop = ControlLoop(cfg, PolicyRunner(HERE / "exploration_model.onnx", cfg),
                   NullMotorDriver(cfg), slots)
loop.map_enabled = False
auths, raw_ok = [], True
for i in range(60):
    slots["lidar"].publish(np.full(cfg.n_beams, 0.30, np.float32))   # boxed in
    loop.tick()
    r = loop.last_record
    auths.append(r.authority)
    if r.stages[0].omega != r.omega_raw:
        raw_ok = False
check("a recovery tick reports RECOVERY_CONTROLLER",
      tlm.SRC_RECOVERY in auths, f"seen: {sorted(set(auths))}")
check("the raw policy output survives recovery untouched", raw_ok)
check("recovery ticks are counted", loop.stats["recovery_ticks"] > 0,
      f"{loop.stats['recovery_ticks']}")

# 8. --no-recovery must genuinely disable it.
loop2 = ControlLoop(cfg, PolicyRunner(HERE / "exploration_model.onnx", cfg),
                    NullMotorDriver(cfg), slots)
loop2.map_enabled = False
loop2.recovery_enabled = False
for i in range(60):
    slots["lidar"].publish(np.full(cfg.n_beams, 0.30, np.float32))
    loop2.tick()
check("--no-recovery leaves the controller inert", loop2.stats["recovery_ticks"] == 0)

# 9. A maneuver must be RELEASED as soon as the path it was opening is actually open.
#
# It used to run its whole tick budget regardless. Measured over a 90 s exploration
# run: RECOVERY_CONTROLLER held authority for 112 of 316 ticks, and `_maneuver` only
# ever returns a pivot or a reverse -- so forward motion was impossible for all of
# them whether or not the way had opened.
#
# The release test is FORWARD CLEARANCE, deliberately not `not boxed_in`. That flag is
# a stuck counter, so it clears the moment the pivot starts moving her; releasing on it
# measured WORSE than never releasing (recovery 112 -> 12 ticks, forward ticks 10 -> 0)
# because she was handed back into a path still blocked.
BLOCKED = np.full(cfg.n_beams, 0.30, np.float32)     # inside hard_stop
OPEN = np.full(cfg.n_beams, 2.00, np.float32)

rc9 = RecoveryController(cfg)
rc9.update(BLOCKED, 0.0, 0.0, 0.30, True, np.zeros(3), 0, 1.0)
check("boxed_in starts a maneuver", rc9.state == "RECOVERING", rc9.state)
held = 0
for i in range(LEVELS[0]["ticks"]):                   # path is open now
    if rc9.update(OPEN, 0.0, 0.0, 0.30, True, np.zeros(3), 100 + i, 1.0) is None:
        break
    held += 1
check("...and releases early once the way ahead is genuinely open",
      rc9.state == "VERIFY" and held < LEVELS[0]["ticks"] - 1,
      f"held {held} of {LEVELS[0]['ticks']}, state {rc9.state}")
check("the early release is counted", rc9.early_exits == 1, f"{rc9.early_exits}")
check("snapshot exposes early_exits", rc9.snapshot().get("early_exits") == 1)

# The negative, which is the one that matters: a robot whose path is STILL blocked must
# get its whole budget. A release test that cannot tell these apart is the regression.
rc10 = RecoveryController(cfg)
rc10.update(BLOCKED, 0.0, 0.0, 0.30, True, np.zeros(3), 0, 1.0)
held10 = 0
for i in range(LEVELS[0]["ticks"] + 2):
    if rc10.update(BLOCKED, 0.0, 0.0, 0.30, True, np.zeros(3), 200 + i, 1.0) is None:
        break
    held10 += 1
check("a still-blocked robot keeps its full budget",
      rc10.early_exits == 0 and held10 >= LEVELS[0]["ticks"] - 1,
      f"held {held10}, early_exits {rc10.early_exits}")

# ...and clearance alone must not release a maneuver that a MOTION trigger started,
# because a spin is not fixed by there being room to spin in.
rc11 = RecoveryController(cfg)
run(rc11, SPIN_WINDOW + 2, 0.0, 2.0, 0.30)
if rc11.state == "RECOVERING":
    held11 = 0
    for i in range(LEVELS[0]["ticks"]):
        if rc11.update(OPEN, 0.0, 2.0, 0.30, False, np.zeros(3), 300 + i, 1.0) is None:
            break
        held11 += 1
    check("a spin recovery is not released by open space alone",
          rc11.early_exits == 0, f"early_exits {rc11.early_exits}, held {held11}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
