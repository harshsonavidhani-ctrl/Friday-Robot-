"""Stage 2b-bis: back out when the front is blocked and the rear is clear.

MEASURED on the robot, run-20260906-015904, 783 REFLEX_STOP ticks of 1462 logged:

    frozen   (v=0, w=0)      25%
    pivoting (v=0, w!=0)     50%   <- Harsh: "it sticks and confuses"
    REVERSING                24%

while the rear gate refused a reverse ELEVEN times in the whole twenty minutes.
The space behind her was there and almost never used, and the longest single
stuck stretch was ~665 ticks -- about a minute in one spot.

Every pre-existing escape pivots, deliberately: a pivot cannot approach anything
so it needs no proof about the rear. Reversing needs that proof, and stage 2d
already supplies it by re-applying the rear gate to whatever any stage authored.
So this stage only ASKS. These tests pin both halves: that it asks when it
should, and that asking is not the same as being allowed.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,
                       RobotConfig, SensorSlot)

cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class CanReverseDriver(NullMotorDriver):
    """The real drivetrain: a stiction floor low enough that v_min executes.

    0.1487 m/s is the measured floor at motor-max-duty 0.85, and |v_min| is 0.15
    -- the 0.001 m/s of margin that confident_explore.env warns must not be spent.
    """

    def min_speed_mps(self) -> float:
        return 0.1487


def make_loop(driver=None):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg, 1)
    loop = ControlLoop(cfg, policy, driver or CanReverseDriver(cfg), slots)
    # These tests pin STAGE 2b-bis itself. Since 2026-09-13 the decisive escape
    # (escape.py, test_decisive_escape.py) replaces it by default; switch that off
    # here so the legacy stage is still tested exactly as it was.
    loop.decisive_escape = False
    return loop, slots


def front_blocked(rear_m=3.0):
    """Something inside hard_stop dead ahead, and `rear_m` behind her."""
    s = np.full(cfg.n_beams, 3.0, np.float32)
    n = cfg.n_beams
    s[:n // 8] = 0.30                       # a wedge either side of straight ahead
    s[-(n // 8):] = 0.30
    s[n // 2 - n // 8: n // 2 + n // 8] = rear_m
    return s


print("--- reverse escape (front blocked, reverse available) ---")

# 1. The thing he asked for: back out instead of spinning.
loop, slots = make_loop()
outs = []
for _ in range(20):
    slots["lidar"].publish(front_blocked())
    outs.append(loop.tick())
check("a reverse escape fires when the front is blocked and the rear is clear",
      loop.stats["reverse_escapes"] > 0, f"{loop.stats['reverse_escapes']} ticks")
check("it actually reverses, it does not pivot",
      any(o["v"] < -0.01 for o in outs),
      f"min v {min(o['v'] for o in outs):+.3f}")
# THE SLEW LIMITER RAMPS, so these assert on the SETTLED command, not on every
# tick. Measured trace: ticks 1-3 climb -0.040, -0.080, -0.120 under the policy's
# own lifted reverse, the stage takes over at tick 4, and omega bleeds to zero by
# tick 6. Asserting tick-by-tick would be asserting that the limiter does not
# exist -- which is the thing that stops a reverse being a lurch.
settled = [o for o in outs if o["gate"] == "reverse_escape"]
check("it reverses at an EXECUTABLE speed, not under the stiction floor",
      settled and min(o["v"] for o in settled)
      <= -CanReverseDriver(cfg).min_speed_mps() + 1e-6,
      f"settled v {min(o['v'] for o in settled):+.3f} vs floor -0.1487")
check("it backs straight out rather than swinging the front through the obstacle",
      settled and all(abs(o["omega"]) < 1e-6 for o in settled[3:]),
      f"|omega| settles to {abs(settled[-1]['omega']):.3f}")
check("the gate reports it", any(o["gate"] == "reverse_escape" for o in outs))

# 2. ASKING IS NOT BEING ALLOWED. Stage 2d must still veto a blocked rear.
loop, slots = make_loop()
outs = []
for _ in range(20):
    slots["lidar"].publish(front_blocked(rear_m=0.20))   # inside rear_stop 0.36
    outs.append(loop.tick())
check("a blocked rear is still refused, whoever authored the reverse",
      all(o["v"] >= -1e-6 for o in outs),
      f"min v {min(o['v'] for o in outs):+.3f}")

# 3. It must not fire on a passing brake.
loop, slots = make_loop()
clear = np.full(cfg.n_beams, 3.0, np.float32)
for _ in range(30):
    slots["lidar"].publish(clear)
    loop.tick()
check("no reverse escape in open space", loop.stats["reverse_escapes"] == 0,
      f"{loop.stats['reverse_escapes']} ticks")

# 4. Not on the first blocked tick -- that is a brake, not being stuck.
loop, slots = make_loop()
slots["lidar"].publish(front_blocked())
first = loop.tick()
check("does not fire on the very first blocked tick",
      first["gate"] != "reverse_escape", f"gate {first['gate']}")

# 5. Bounded. Unbounded reversing is how a robot backs across a room blind.
loop, slots = make_loop()
for _ in range(120):
    slots["lidar"].publish(front_blocked())
    loop.tick()
check("backing out is bounded, not unlimited",
      loop.stats["reverse_escapes"] <= loop.reverse_escape_max_ticks,
      f"{loop.stats['reverse_escapes']} <= {loop.reverse_escape_max_ticks}")

# 6. Switchable off from the mode file, with no code change.
loop, slots = make_loop()
loop.reverse_escape = False
for _ in range(20):
    slots["lidar"].publish(front_blocked())
    loop.tick()
check("--no-reverse-escape restores the pivot-only behaviour",
      loop.stats["reverse_escapes"] == 0,
      f"{loop.stats['reverse_escapes']} ticks")

print(f"\n{'FAILED' if fails else 'OK'}: {fails} failure(s)")
sys.exit(1 if fails else 0)
