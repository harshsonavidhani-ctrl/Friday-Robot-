"""Dead reckoning must use what the WHEELS DID, not what the policy asked for.

The two differ structurally on this drivetrain, not by noise: the creep cutoff turns
a sub-floor request into a full STOP, the stiction floor lifts a small request UP to
floor speed, and the duty cap holds a large request DOWN to the ceiling. Reckoning
from the command puts the exploration map's pose somewhere the robot is not, which
corrupts the 72 frontier rays that depend on it.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import FridayMotorDriver, NullMotorDriver, RobotConfig

cfg = RobotConfig()
MIN_DUTY, MAX_DUTY = 72, 0.5
CEILING = int(255 * MAX_DUTY)
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class MockArduino:
    def __init__(self):
        self.ident, self.sent, self._out = "FRIDAY-MOTORS v1.2", [], b""
        self.timeout, self.is_open = 1.0, True

    def write(self, data):
        line = data.decode().strip()
        self.sent.append(line)
        if line == "P":
            self._out = (self.ident + "\n").encode()
        elif line and line[0] in "WMS":
            self._out = b"OK\n"
        else:
            self._out = b""
        return len(data)

    def flush(self):
        pass

    def readline(self):
        out, self._out = self._out, b""
        return out

    @property
    def in_waiting(self):
        return len(self._out)

    def read(self, n):
        out, self._out = self._out[:n], self._out[n:]
        return out

    def reset_input_buffer(self):
        self._out = b""

    def close(self):
        self.is_open = False


def make():
    mock = MockArduino()
    sys.modules["serial"] = types.SimpleNamespace(Serial=lambda *a, **k: mock)
    real = pi_deploy.time.sleep
    pi_deploy.time.sleep = lambda s: None
    try:
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=MAX_DUTY, min_duty=MIN_DUTY)
    finally:
        pi_deploy.time.sleep = real
    d._moving = True          # skip the kickstart so we read steady-state duty
    d._kick_left = 0
    return d


def speed_for_raw(target):
    return target / (255.0 * MAX_DUTY) * cfg.wheel_speed_max


print("--- executed vs commanded ---")

# 1. A driver that does not move must not claim it did. This is what dead-reckoned a
#    phantom pose across the room on a bench run and marked 81 m^2 explored from a
#    parked robot, pinning every frontier ray at max.
n = NullMotorDriver(cfg)
n.drive(0.30, 0.5)
check("null driver executes NOTHING", n.executed() == (0.0, 0.0), f"{n.executed()}")

# 2. The creep cutoff turns a sub-floor request into a stop, so nothing moves.
d = make()
d.drive(speed_for_raw(20), 0.0)          # under the 36 cutoff
check("a creep-cutoff request executes as a stop",
      d.executed() == (0.0, 0.0), f"{d.executed()}")

# 3. The stiction floor lifts a small request UP: the robot goes FASTER than asked.
d = make()
asked = speed_for_raw(60)                # below the 72 floor, above the cutoff
d.drive(asked, 0.0)
ex_v = d.executed()[0]
check("a floor-lifted request executes FASTER than commanded",
      ex_v > asked * 1.05, f"asked={asked:.3f} executed={ex_v:.3f} m/s")

# 4. Above the floor and below the cap, executed tracks commanded.
d = make()
asked = speed_for_raw(100)
d.drive(asked, 0.0)
ex_v = d.executed()[0]
check("a normal request executes as commanded",
      abs(ex_v - asked) < 0.01, f"asked={asked:.3f} executed={ex_v:.3f} m/s")

# 5. The duty cap holds a large request DOWN: the robot goes SLOWER than asked.
d = make()
asked = cfg.v_max                        # 0.5 m/s, far above this drivetrain's ceiling
d.drive(asked, 0.0)
ex_v = d.executed()[0]
check("a capped request executes SLOWER than commanded",
      ex_v < asked * 0.95, f"asked={asked:.3f} executed={ex_v:.3f} m/s")
check("the capped speed is the ceiling speed",
      abs(ex_v - cfg.wheel_speed_max) < 0.01, f"executed={ex_v:.3f} m/s")

# 6. An emergency stop is executed, whatever was asked for.
d = make()
d.drive(speed_for_raw(100), 0.0)
d.emergency_stop()
check("emergency stop executes as a stop", d.executed() == (0.0, 0.0), f"{d.executed()}")

# 7. Turn sign survives the round trip -- a mirrored omega would reckon the map
#    backwards through every corner.
d = make()
d.drive(0.0, 0.8)
check("a left turn executes as a left turn", d.executed()[1] > 0, f"{d.executed()}")
d = make()
d.drive(0.0, -0.8)
check("a right turn executes as a right turn", d.executed()[1] < 0, f"{d.executed()}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
