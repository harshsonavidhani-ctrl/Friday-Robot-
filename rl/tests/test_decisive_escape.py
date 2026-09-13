"""Decisive escape: decide once, carry it through, hand back.

Harsh, 2026-09-13: "behave and explore like professional rovers do". The chain it
replaces flipped between four stages tick by tick (09-13 09:50 run: reverse_escape,
planner, reverse_escape, governor, planner, ...) and on 2026-09-12 she pivoted in
place on 26-77% of ticks. Every case below is one way a committed manoeuvre can
still go wrong:

    it backs out, then turns          -> the manoeuvre, in order
    the turn never flips side         -> the dithering it exists to end
    a blocked rear skips BACK         -> it does not argue with the rear gate
    asking is not being allowed       -> the rear gate still vetoes its reverse
    it lets go when the way clears    -> it does not hold the robot a tick longer
    it is bounded, and cools down     -> it cannot hold the robot forever
    not on a passing brake / open room
    --no-decisive-escape               -> the legacy stages come back
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot)
import escape as esc                                                    # noqa: E402

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
    """The real drivetrain's floor at motor-max-duty 0.85: v_min executes."""

    def min_speed_mps(self) -> float:
        return 0.1487


def make_loop():
    slots = {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
             "tof": SensorSlot("tof", cfg.tof_deadline, True)}
    policy = PolicyRunner(HERE / "exploration_model.onnx", cfg, 1)
    loop = ControlLoop(cfg, policy, CanReverseDriver(cfg), slots)
    loop.memory_enabled = False          # isolate the manoeuvre from memory here
    return loop, slots


def boxed(rear_m=3.0, open_left=True):
    """Front blocked inside hard_stop; one side much more open than the other."""
    n = cfg.n_beams
    s = np.full(n, 0.9, np.float32)
    s[:n // 8] = 0.30
    s[-(n // 8):] = 0.30
    left = slice(n // 8, n // 2 - n // 8)          # bearings 45..135 deg
    right = slice(n // 2 + n // 8, n - n // 8)     # bearings 225..315 deg
    s[left] = 3.0 if open_left else 0.45
    s[right] = 0.45 if open_left else 3.0
    s[n // 2 - n // 8: n // 2 + n // 8] = rear_m
    return s


def run(loop, slots, scans):
    outs = []
    for s in scans:
        slots["lidar"].publish(s)
        outs.append(loop.tick())
    return outs


print("--- decisive escape ---")

# 1. The manoeuvre, in order: BACK first, then TURN.
loop, slots = make_loop()
outs = run(loop, slots, [boxed()] * 45)
gates = [o["gate"] for o in outs]
first_back = next((i for i, g in enumerate(gates) if g == "escape_back"), None)
first_turn = next((i for i, g in enumerate(gates) if g == "escape_turn"), None)
check("an escape starts when boxed in", loop.escape.episodes >= 1,
      f"episodes={loop.escape.episodes}")
check("it backs out BEFORE it turns",
      first_back is not None and first_turn is not None and first_back < first_turn,
      f"back@{first_back} turn@{first_turn}")
backs = [o for o in outs if o["gate"] == "escape_back"]
check("BACK is a straight reverse at an executable speed",
      backs and min(o["v"] for o in backs) <= -0.1487 + 1e-6
      and all(abs(o["omega"]) < 1e-6 for o in backs[3:]),
      f"min v {min((o['v'] for o in backs), default=0):+.3f}")

# 2. The turn is decided ONCE: every TURN tick has the same sign, toward open space.
turns = [o for o in outs if o["gate"] == "escape_turn"]
signs = {np.sign(o["omega"]) for o in turns if abs(o["omega"]) > 0.05}
check("the turn never flips side mid-manoeuvre", len(signs) == 1, f"signs={signs}")
check("with the LEFT open it turns left", signs == {1.0}, f"signs={signs}")
loop, slots = make_loop()
outs = run(loop, slots, [boxed(open_left=False)] * 45)
rsigns = {np.sign(o["omega"]) for o in outs
          if o["gate"] == "escape_turn" and abs(o["omega"]) > 0.05}
check("with the RIGHT open it turns right", rsigns == {-1.0}, f"signs={rsigns}")

# 3. A blocked rear: it goes straight to TURN, and the gate still vetoes reverse.
loop, slots = make_loop()
outs = run(loop, slots, [boxed(rear_m=0.20)] * 40)
check("a blocked rear skips BACK entirely",
      not any(o["gate"] == "escape_back" for o in outs) and loop.escape.turns >= 1,
      f"backs={loop.escape.backs} turns={loop.escape.turns}")
check("...and nothing reverses into it", all(o["v"] >= -1e-6 for o in outs),
      f"min v {min(o['v'] for o in outs):+.3f}")

# 4. It lets go the moment the way ahead is clear.
loop, slots = make_loop()
outs = run(loop, slots, [boxed()] * 22)           # into BACK / TURN
in_control = loop.escape.active
clear = np.full(cfg.n_beams, 3.0, np.float32)
after = run(loop, slots, [clear] * 6)
check("it was in control while boxed in", in_control, f"phase={loop.escape.phase}")
check("it hands back once the way ahead is clear",
      not loop.escape.active and loop.escape.exits_clear >= 1
      and not any(o["gate"].startswith("escape_") for o in after[3:]),
      f"phase={loop.escape.phase} exits_clear={loop.escape.exits_clear} "
      f"gates={[o['gate'] for o in after]}")

# 5. Bounded, with a cooldown before it may start again.
loop, slots = make_loop()
outs = run(loop, slots, [boxed()] * 200)
longest = cur = 0
for o in outs:
    cur = cur + 1 if o["gate"].startswith("escape_") else 0
    longest = max(longest, cur)
bound = esc.BACK_MAX_TICKS + esc.TURN_MAX_TICKS + 2
check("one manoeuvre is bounded", longest <= bound, f"longest {longest} <= {bound}")
check("it re-arms after its cooldown rather than never again",
      loop.escape.episodes >= 2, f"episodes={loop.escape.episodes}")

# 6. Not a passing brake; not in the open.
loop, slots = make_loop()
first = run(loop, slots, [boxed()])[0]
check("not on the first blocked tick", not first["gate"].startswith("escape_"),
      f"gate {first['gate']}")
loop, slots = make_loop()
run(loop, slots, [clear] * 40)
check("never in open space", loop.escape.episodes == 0,
      f"episodes={loop.escape.episodes}")

# 7. The switch brings the legacy stages back.
loop, slots = make_loop()
loop.decisive_escape = False
outs = run(loop, slots, [boxed()] * 30)
check("--no-decisive-escape: no manoeuvre, legacy reverse escape instead",
      loop.escape.episodes == 0 and loop.stats["reverse_escapes"] > 0,
      f"episodes={loop.escape.episodes} reverse_escapes={loop.stats['reverse_escapes']}")

print(f"\n{'FAILED' if fails else 'OK'}: {fails} failure(s)")
sys.exit(1 if fails else 0)
