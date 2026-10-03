"""Turn commitment (Harsh, 2026-10-04: "noise more than movement ... it should be
smooth", then "DO whats still left").

MEASURED in the 490 Hz black boxes: blocked, the policy flips its turn between
+1.6 and -1.6 rad/s 10-38 times a minute, 35-45% of them back within a second,
each flip a one-tick (0, 0) at the slew limiter. ControlLoop._commit_turn keeps
a turn taken while blocked for turn_commit_s. These check that it only ever
keeps the DIRECTION of a turn the chain already asked for: never invents one,
never passes through a stop, never turns faster, and leaves the deliberate
turns (escape, recovery, undo, the mission's path, the open-side pivot) alone.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy                                        # noqa: E402
from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot, build_arg_parser)

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class Rec:
    def __init__(self):
        self.stages = []

    def add(self, *a):
        self.stages.append(a)


class Clock:
    t = 1000.0


cfg = RobotConfig()
cfg.tof_enabled = False
cfg.cam_enabled = False
loop = ControlLoop(cfg, PolicyRunner(HERE / "exploration_model.onnx", cfg, 1),
                   NullMotorDriver(cfg), {"lidar": SensorSlot("lidar", cfg.lidar_deadline, True),
                                          "tof": SensorSlot("tof", cfg.tof_deadline, True)})
real_monotonic = pi_deploy.time.monotonic
pi_deploy.time.monotonic = lambda: Clock.t


def step(v, w, gate="governor", dt=0.1):
    Clock.t += dt
    return loop._commit_turn(v, w, gate, Rec())


print("--- turn commitment ---")
check("off unless asked for (parser default 0, a new loop 0)",
      build_arg_parser().parse_args([]).turn_commit == 0.0 and loop.turn_commit_s == 0.0)
loop.turn_commit_s = 1.0
w1 = step(0.0, 1.6)                                     # blocked: turn left
w2 = step(0.0, -1.6)                                    # the policy flips right
w3 = step(0.0, -1.6)
check("blocked, a flip inside the window keeps the committed way at the same rate",
      w1 == 1.6 and w2 == 1.6 and w3 == 1.6, f"{w1} {w2} {w3}")
check("...and is counted", loop.stats.get("turn_commits") == 2, str(loop.stats.get("turn_commits")))
for _ in range(8):
    step(0.0, -1.6)
w4 = step(0.0, -1.6)                                    # > 1 s since the turn was taken
check("past turn_commit_s the new direction is taken", w4 == -1.6, str(w4))
check("a stop passes as it is, inside the window too (never a turn out of nothing)",
      step(0.0, 0.0) == 0.0 and step(0.0, 0.05) == 0.05)
w5 = step(0.0, 1.6)
check("...and a stop does not reset the window (no flip by way of a zero)",
      w5 == -1.6, str(w5))
check("never faster than asked", abs(step(0.0, 0.4)) == 0.4)
for gate in ("escape_turn", "recovery", "undo", "mission_drive", "no_reverse_turn"):
    loop._turn_sign, loop._turn_since = 1, Clock.t
    got = step(0.0, -1.2, gate=gate)
    check(f"exempt: {gate} turns the way it chose", got == -1.2 and loop._turn_sign == 0,
          str(got))
loop._turn_sign, loop._turn_since = 1, Clock.t
check("driving (|v| >= 0.15): normal steering, nothing kept",
      step(0.2, -0.8) == -0.8 and loop._turn_sign == 0)
loop._turn_sign, loop._turn_since = 1, Clock.t
check("reverse while turning (|v| >= 0.15 backwards) is driving too",
      step(-0.15, -0.8) == -0.8)
src = Path(pi_deploy.__file__).read_text()
check("the stage runs AFTER the ToF reflex and unstick, BEFORE the slew limit",
      src.index("# ---- 4. ToF reflex (hard veto)") < src.index("# ---- 4b. UNDO THE LAST MOVE")
      < src.index("# ---- 4c. TURN COMMITMENT") < src.index("omega = self._commit_turn(")
      < src.index("# ---- 5. slew limit + actuate"))
check("...and never while unstick's undo is driving",
      "if self.turn_commit_s > 0.0 and not self._undo_active:" in src)

pi_deploy.time.monotonic = real_monotonic
print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
