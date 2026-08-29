"""Test FridayMotorDriver against a mock Arduino speaking friday_motors.ino's protocol."""
import sys
import types

import numpy as np

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
    """Implements P / W / M / S exactly as the firmware does."""

    def __init__(self, ident="FRIDAY-MOTORS v1.2"):
        self.ident = ident
        self.sent = []            # every line received
        self._out = b""
        self.timeout = 1.0
        self.is_open = True

    def write(self, data):
        line = data.decode().strip()
        self.sent.append(line)
        if line == "P":
            self._out = (self.ident + "\n").encode()
        elif line.startswith("W") or line.startswith("M") or line == "S":
            self._out = b"OK\n"
        elif line == "":
            self._out = b""       # keepalive: no reply
        else:
            self._out = b"ERR\n"
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


def make_driver(mock=None, **kw):
    """Build the driver with serial/sleep patched out."""
    mock = mock or MockArduino()
    fake_serial = types.SimpleNamespace(Serial=lambda *a, **k: mock)
    real_sleep = pi_deploy.time.sleep
    sys.modules["serial"] = fake_serial
    pi_deploy.time.sleep = lambda s: None
    try:
        d = FridayMotorDriver(cfg, "/dev/mock", **kw)
    finally:
        pi_deploy.time.sleep = real_sleep
    return d, mock


print("--- FridayMotorDriver tests ---")

# 1. handshake + watchdog on connect
d, mock = make_driver()
check("sends identity handshake", mock.sent[0] == "P")
check("arms watchdog", any(s.startswith("W") for s in mock.sent),
      [s for s in mock.sent if s.startswith("W")])

# 2. refuses an unidentified board -- the safety gate
try:
    make_driver(MockArduino(ident="SOME-OTHER-BOARD"))
    check("refuses unknown board", False, "it connected anyway")
except RuntimeError as exc:
    check("refuses unknown board", "Refusing to drive" in str(exc))

# 3. stop uses the firmware's dedicated S, not M0 0
d, mock = make_driver()
mock.sent.clear()
d.emergency_stop()
check("emergency stop sends S", mock.sent[-1] == "S", mock.sent)

# 4. duty never exceeds the cap, at any commanded speed
d, mock = make_driver(max_duty=0.5)
ceiling = int(255 * 0.5)
worst = 0
for v in np.linspace(-1.0, 1.0, 21):
    for w in np.linspace(-3.0, 3.0, 21):
        mock.sent.clear()
        d._moving = True                      # skip kickstart boost
        d._kick_left = 0
        d.drive(float(v), float(w))
        line = mock.sent[-1]
        if line.startswith("M"):
            l, r = (int(x) for x in line[1:].split())
            worst = max(worst, abs(l), abs(r))
check("duty never exceeds max_duty cap", worst <= ceiling, f"peak |duty|={worst}, cap={ceiling}")

# 5. zero command stops rather than sending M0 0
d, mock = make_driver()
mock.sent.clear()
d.drive(0.0, 0.0)
check("zero speed sends S", mock.sent[-1] == "S", mock.sent)

# 6. stiction floor lifts a NEAR-floor request up to the floor.
#    0.28 m/s maps to raw duty ~50: under the 70 floor but above the creep cutoff (35),
#    so it means "go, roughly this fast" and is lifted. A much slower request is a
#    different case entirely and is covered in test_creep.py.
d, mock = make_driver(min_duty=70)
d._moving = True
d._kick_left = 0
mock.sent.clear()
d.drive(0.28, 0.0)
line = mock.sent[-1]
check("near-floor request reaches the wire", line.startswith("M"), line)
l, r = (int(x) for x in line[1:].split())
check("stiction floor applied", min(abs(l), abs(r)) >= 70, f"L={l} R={r}")

# 7. inversion flags
d, mock = make_driver(invert_left=True)
d._moving = True
d._kick_left = 0
mock.sent.clear()
d.drive(0.3, 0.0)
l_inv, r_inv = (int(x) for x in mock.sent[-1][1:].split())
d2, mock2 = make_driver()
d2._moving = True
d2._kick_left = 0
mock2.sent.clear()
d2.drive(0.3, 0.0)
l_norm, r_norm = (int(x) for x in mock2.sent[-1][1:].split())
check("invert-left flips only the left duty", l_inv == -l_norm and r_inv == r_norm,
      f"normal=({l_norm},{r_norm}) inverted=({l_inv},{r_inv})")

# 8. kickstart boosts out of standstill, then relaxes
d, mock = make_driver(kick_ticks=2, kick_scale=1.6)
d._moving = False
duties = []
for _ in range(4):
    mock.sent.clear()
    d.drive(0.15, 0.0)
    duties.append(int(mock.sent[-1][1:].split()[0]))
check("kickstart boosts then settles", duties[0] > duties[-1],
      f"duties={duties}")

# 9. straight-ahead command drives both wheels equally
d, mock = make_driver()
d._moving = True
d._kick_left = 0
mock.sent.clear()
d.drive(0.4, 0.0)
l, r = (int(x) for x in mock.sent[-1][1:].split())
check("straight command is symmetric", l == r, f"L={l} R={r}")

# 10. pure rotation drives wheels opposite
mock.sent.clear()
d._kick_left = 0
d.drive(0.0, 1.0)
l, r = (int(x) for x in mock.sent[-1][1:].split())
check("pure rotation is antisymmetric", l == -r and l != 0, f"L={l} R={r}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

