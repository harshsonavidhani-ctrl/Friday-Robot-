"""Stiction-floor semantics: a request too slow to execute must STOP, not lurch."""
import sys
import types

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import FridayMotorDriver, RobotConfig


class MockArduino:
    """Local copy -- importing it from test_motors would execute that file's tests."""

    def __init__(self, ident="FRIDAY-MOTORS v1.2"):
        self.ident, self.sent, self._out = ident, [], b""
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

cfg = RobotConfig()
MIN_DUTY = 72          # measured on the real robot
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def make(**kw):
    mock = MockArduino()
    sys.modules["serial"] = types.SimpleNamespace(Serial=lambda *a, **k: mock)
    real_sleep = pi_deploy.time.sleep
    pi_deploy.time.sleep = lambda s: None
    try:
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=0.8, min_duty=MIN_DUTY, **kw)
    finally:
        pi_deploy.time.sleep = real_sleep
    return d, mock


print("--- stiction floor / creep cutoff (min_duty=72, cutoff=0.5 -> 36) ---")
d, _ = make()

ceiling = int(255 * 0.8)


def duty_for(mps):
    return d._mps_to_duty(mps)


# Raw (uncapped) duty for a given speed, to pick test points precisely.
def raw(mps):
    return round(255.0 * 0.8 * (mps / cfg.wheel_speed_max))


def speed_for_raw(target):
    """The m/s whose uncapped duty is `target`, for THIS cfg.wheel_speed_max.

    These points used to be absolute m/s (0.02 / 0.28 / 0.5) chosen when
    wheel_speed_max was the simulator's 0.85. It has since been MEASURED at
    0.185, and the old constants then all exceeded the duty ceiling, so every
    point collapsed onto the cap and three assertions failed. What is under
    test here is the floor and the creep cutoff -- not the calibration -- so
    the points are derived from the config instead of frozen against one value.
    """
    return target / (255.0 * 0.8) * cfg.wheel_speed_max


# A speed well below the floor -> stop, NOT floor speed.
slow = speed_for_raw(20)              # under the 36 cutoff
check("very slow request stops instead of lurching",
      duty_for(slow) == 0, f"raw={raw(slow)} -> {duty_for(slow)}")

# A speed just under the floor -> lifted to the floor (it meant "go", roughly this fast).
near = speed_for_raw(67)   # just below the 72 floor, above the 36 cutoff
check("near-floor request lifts to the floor",
      duty_for(near) == MIN_DUTY, f"raw={raw(near)} -> {duty_for(near)}")

# Above the floor -> untouched.
fast = speed_for_raw(150)  # above the floor, below the 204 ceiling
check("above-floor request passes through",
      duty_for(fast) == raw(fast), f"raw={raw(fast)} -> {duty_for(fast)}")

# Zero stays zero.
check("zero stays zero", duty_for(0.0) == 0)

# Symmetric for reverse.
check("reverse: very slow stops", duty_for(-slow) == 0, f"{duty_for(-slow)}")
check("reverse: near-floor lifts", duty_for(-near) == -MIN_DUTY, f"{duty_for(-near)}")

# Cap still respected with the floor active.
worst = max(abs(duty_for(v / 100.0)) for v in range(-100, 101))
check("cap still respected", worst <= ceiling, f"peak={worst} cap={ceiling}")

# cutoff=0 restores the old always-lift behaviour, for anyone who wants it.
d2, _ = make(creep_cutoff=0.0)
check("cutoff=0 lifts everything to the floor",
      d2._mps_to_duty(slow) == MIN_DUTY, f"{d2._mps_to_duty(slow)}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

