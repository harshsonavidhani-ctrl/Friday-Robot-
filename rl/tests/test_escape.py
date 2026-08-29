"""Boxed in: three individually correct decisions that together mean "do nothing".

MEASURED on the robot: parked with an obstacle at 0.33 m (inside hard_stop 0.36),
the governor zeroed forward on 268 of 300 ticks while the policy asked for +0.13.
Forward forbidden, reverse impossible on this drivetrain, and the turn the policy
wanted was too gentle to execute. She sat against the wall until a human moved her
-- which is precisely what "without human intervention" excludes.
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


class NoReverseDriver(NullMotorDriver):
    """A driver with a real stiction floor, like the L298N stack.

    The escape is deliberately inert without one: hardware that CAN reverse should
    reverse out, and the policy already knows how.
    """

    def min_speed_mps(self) -> float:
        return 0.179            # measured: duty 72 of ceiling 191 at 0.476 m/s


def make_loop(driver=None):
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg, 1)
    loop = ControlLoop(cfg, policy, driver or NoReverseDriver(cfg), slots)
    return loop, slots


def boxed_scan(open_left=True):
    """Something inside hard_stop all round, with one side clearly more open."""
    s = np.full(cfg.n_beams, 0.30, np.float32)
    n = cfg.n_beams
    if open_left:
        s[n // 4 - 4: n // 4 + 5] = 3.0          # left is open
    else:
        s[3 * n // 4 - 4: 3 * n // 4 + 5] = 3.0  # right is open
    return s


print("--- escape turn (boxed in) ---")

# 1. She must not sit there forever.
loop, slots = make_loop()
outs = []
for _ in range(20):
    slots["lidar"].publish(boxed_scan(open_left=True))
    outs.append(loop.tick())
check("an escape turn fires when boxed in",
      loop.stats["escape_turns"] > 0, f"{loop.stats['escape_turns']} turns")
check("the escape is a real turn, not a nudge",
      any(abs(o["omega"]) >= cfg.omega_max - 1e-6 for o in outs),
      f"max |omega| {max(abs(o['omega']) for o in outs):.2f}")
check("the escape never drives forward into what blocked her",
      all(o["v"] <= 1e-6 for o in outs), f"max v {max(o['v'] for o in outs):.3f}")
check("the gate reports it", any(o["gate"] == "escape_turn" for o in outs))

# 2. Direction must come from the LIVE scan. Turning a fixed way into the wall you
#    are already against is how a robot gets permanently stuck.
loop, slots = make_loop()
for _ in range(20):
    slots["lidar"].publish(boxed_scan(open_left=True))
    o = loop.tick()
check("boxed in with LEFT open -> turns left", o["omega"] > 0, f"omega {o['omega']:+.2f}")

loop, slots = make_loop()
for _ in range(20):
    slots["lidar"].publish(boxed_scan(open_left=False))
    o = loop.tick()
check("boxed in with RIGHT open -> turns right", o["omega"] < 0, f"omega {o['omega']:+.2f}")

# 3. It must NOT fire on a passing brake. A robot that pivots every time the
#    governor trims speed would never drive anywhere.
loop, slots = make_loop()
clear = np.full(cfg.n_beams, 3.0, np.float32)
for _ in range(30):
    slots["lidar"].publish(clear)
    loop.tick()
check("no escape turn in open space", loop.stats["escape_turns"] == 0,
      f"{loop.stats['escape_turns']} turns")

# 4. It must wait before deciding she is stuck, not trigger on the first tick.
loop, slots = make_loop()
slots["lidar"].publish(boxed_scan())
first = loop.tick()
check("does not fire on the very first blocked tick",
      first["gate"] != "escape_turn", f"gate {first['gate']}")

# 5. It must be switchable off, for seeing raw policy behaviour.
loop, slots = make_loop()
loop.escape_turns = False
for _ in range(20):
    slots["lidar"].publish(boxed_scan())
    loop.tick()
check("--no-escape-turns disables it", loop.stats["escape_turns"] == 0,
      f"{loop.stats['escape_turns']} turns")

# 6. On hardware that CAN reverse, the escape must stay out of the way.
loop, slots = make_loop(driver=NullMotorDriver(cfg))   # min_speed_mps() == 0
for _ in range(20):
    slots["lidar"].publish(boxed_scan())
    loop.tick()
check("no escape turn when the drivetrain CAN reverse",
      loop.stats["escape_turns"] == 0, f"{loop.stats['escape_turns']} turns")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
