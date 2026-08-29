"""Pivots must use a higher stiction floor than straight driving."""
import sys
import types

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import FridayMotorDriver, RobotConfig

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class MockArduino:
    def __init__(self):
        self.sent, self._out = [], b""
        self.timeout, self.is_open = 1.0, True

    def write(self, data):
        line = data.decode().strip()
        self.sent.append(line)
        self._out = b"FRIDAY-MOTORS v1\n" if line == "P" else b"OK\n"
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


def make(**kw):
    mock = MockArduino()
    sys.modules["serial"] = types.SimpleNamespace(Serial=lambda *a, **k: mock)
    real = pi_deploy.time.sleep
    pi_deploy.time.sleep = lambda s: None
    try:
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=0.7, min_duty=72, **kw)
    finally:
        pi_deploy.time.sleep = real
    d._moving = True          # skip kickstart so we read the steady-state duty
    d._kick_left = 0
    return d, mock


def duties(d, mock, v, omega):
    """Read a STEADY-STATE tick.

    A pivot's FIRST tick is deliberately boosted past the steady cap to break the
    sideways-scrub stiction (approved 2026-08-28; see test_wiring.py). That boost is
    correct and is tested there. What this suite is about is the floor that applies
    once the robot is already turning, so `_was_pivot` is pre-set to skip the
    break-away tick -- otherwise every reading here is the kick, not the floor.
    """
    mock.sent.clear()
    d._kick_left = 0
    d._moving = True
    d._was_pivot = True
    d.drive(v, omega)
    line = mock.sent[-1]
    if line == "S":
        return 0, 0
    return tuple(int(x) for x in line[1:].split())


def speed_for_raw(target):
    """The m/s whose uncapped duty is `target`, for THIS cfg.wheel_speed_max.

    The points below used to be absolute m/s chosen when wheel_speed_max was
    the simulator's 0.85. It has since been MEASURED at 0.185, and the old
    constants then exceeded the 178 duty ceiling, so the "lifted to the floor"
    assertions read the cap instead. The floor semantics are what is under
    test, not the calibration -- so derive the points from the config.
    """
    return target / (255.0 * 0.7) * cfg.wheel_speed_max


def omega_for_raw(target):
    """The pure-pivot omega whose uncapped wheel duty is `target`."""
    return speed_for_raw(target) / (cfg.wheel_base / 2.0)


print("--- pivot stiction floor (min=72, turn floor defaults to 1.4x = 100) ---")

d, mock = make()
check("turn floor defaults to 1.4x straight floor", d.turn_min_duty == 100,
      f"turn_min_duty={d.turn_min_duty}")

# Straight ahead uses the straight floor.
l, r = duties(d, mock, speed_for_raw(60), 0.0)   # under the 72 floor
check("straight uses straight floor", l == r == 72, f"L={l} R={r}")

# A pure pivot drives the wheels opposite and must use the higher floor.
l, r = duties(d, mock, 0.0, 1.2)
check("pivot wheels are opposed", l == -r and l != 0, f"L={l} R={r}")
check("pivot uses the higher floor", max(abs(l), abs(r)) >= 100, f"L={l} R={r}")

# Explicit override is honoured.
d2, mock2 = make(turn_min_duty=130)
l, r = duties(d2, mock2, 0.0, omega_for_raw(110))   # under the 130 turn floor
check("explicit turn floor honoured", max(abs(l), abs(r)) == 130, f"L={l} R={r}")

# A gentle arc (both wheels same direction) is NOT a pivot -- straight floor applies.
l, r = duties(d, mock, 0.4, 0.3)
same_sign = (l > 0) == (r > 0)
check("gentle arc is not treated as a pivot", same_sign and min(abs(l), abs(r)) >= 72,
      f"L={l} R={r}")

# The cap still holds for pivots.
ceiling = int(255 * 0.7)
worst = 0
for w in [x / 10 for x in range(-30, 31)]:
    l, r = duties(d, mock, 0.0, w)
    worst = max(worst, abs(l), abs(r))
check("pivot never exceeds the cap", worst <= ceiling, f"peak={worst} cap={ceiling}")

# The break-away tick itself: entering a pivot must be boosted ABOVE the steady
# floor, or she hums instead of turning. The bound on that boost lives in
# test_wiring.py; here we only assert it actually happens.
d3, mock3 = make(turn_min_duty=130)
mock3.sent.clear()
d3._moving = True
d3._kick_left = 0
d3._was_pivot = False               # entering a pivot
d3._last_turn_kick = -1e9           # cooldown expired
d3.drive(0.0, omega_for_raw(110))
l, r = (0, 0) if mock3.sent[-1] == "S" else tuple(int(x) for x in mock3.sent[-1][1:].split())
check("entering a pivot is boosted above the steady floor",
      max(abs(l), abs(r)) > 130, f"L={l} R={r} vs floor 130")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

