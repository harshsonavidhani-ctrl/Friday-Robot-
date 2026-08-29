"""Creep pulsing: how a robot with no creep executes a creep command.

MODIFIES HANDOFF section 4 invariant 5, with explicit approval. The invariant says
a request far below the stiction floor becomes STOP rather than floor speed, so
proportional braking cannot become a lurch. These assertions are the replacement
guarantee: the AVERAGE speed still tracks the request, and braking still slides to
a stop rather than latching at floor speed.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import FridayMotorDriver, RobotConfig, build_arg_parser

cfg = RobotConfig()
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
        self._out = ((self.ident + "\n").encode() if line == "P"
                     else b"OK\n" if line and line[0] in "WMS" else b"")
        return len(data)

    def flush(self):
        pass

    def readline(self):
        o, self._out = self._out, b""
        return o

    @property
    def in_waiting(self):
        return len(self._out)

    def read(self, n):
        o, self._out = self._out[:n], self._out[n:]
        return o

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
        kw.setdefault("pulse_creep_duty", 130)
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=0.75, min_duty=72,
                              turn_min_duty=190, **kw)
    finally:
        pi_deploy.time.sleep = real
    d._moving = True
    d._kick_left = 0
    d._was_pivot = True
    mock.sent.clear()
    return d, mock


def average_speed(d, mock, v, omega, ticks=200):
    """Mean executed forward speed over many ticks."""
    total = 0.0
    for _ in range(ticks):
        d._kick_left = 0
        d._moving = True
        d._was_pivot = True
        d.drive(v, omega)
        total += d.executed()[0]
    return total / ticks


print("--- creep pulsing (modifies HANDOFF invariant 5, approved) ---")

d, mock = make()
floor = d.min_speed_mps()
pulse = d.pulse_speed_mps()
check("the floor and pulse speeds are sane", 0 < floor < pulse,
      f"floor {floor:.3f} < pulse {pulse:.3f} m/s")

# 1. THE POINT: a request far below the floor must still average out correctly.
#    Measured on hardware, the policy asks for 0.05-0.09 m/s and the floor is 0.179.
for want in (0.06, 0.09, 0.12):
    got = average_speed(*make(), want, 0.0)
    check(f"average speed tracks a {want:.2f} m/s request",
          abs(got - want) < 0.02, f"got {got:.3f} m/s")

# 2. A pulse must have real AUTHORITY. Pulsing at the bare floor is what hummed:
#    one wheel at 72 dragging a dead one.
d, mock = make()
peaks = []
for _ in range(40):
    d._kick_left = 0; d._moving = True; d._was_pivot = True
    d.drive(0.08, 0.0)
    line = mock.sent[-1]
    if line != "S":
        peaks.append(max(abs(int(x)) for x in line[1:].split()))
check("driving pulses use the authoritative duty, not the bare floor",
      peaks and min(peaks) >= 130, f"min pulse duty {min(peaks) if peaks else 0}")

# 3. Turn geometry survives: both wheels scale by the SAME factor, so a gentle arc
#    keeps both wheels driven and keeps its shape.
d, mock = make()
both = []
for _ in range(60):
    d._kick_left = 0; d._moving = True; d._was_pivot = True
    d.drive(0.08, 0.10)                     # gentle arc
    line = mock.sent[-1]
    if line != "S":
        l, r = (int(x) for x in line[1:].split())
        if l and r:
            both.append((abs(l), abs(r)))
check("a gentle pulsed arc drives BOTH wheels", len(both) > 0,
      f"{len(both)} two-wheel pulses, e.g. {both[0] if both else '-'}")
check("the outer wheel leads on a pulsed arc",
      all(max(p) > min(p) for p in both) if both else False,
      f"{both[0] if both else '-'}")

# A SHARP arc legitimately asks the inner wheel for almost nothing, so driving one
# wheel is correct geometry -- what matters is the driven wheel having AUTHORITY.
# duty 72 beside a dead wheel is what hummed; 130 beside a dead wheel does not.
d, mock = make()
driven = []
for _ in range(60):
    d._kick_left = 0; d._moving = True; d._was_pivot = True
    d.drive(0.08, 0.4)                      # sharp arc, inner wheel ~0.01 m/s
    line = mock.sent[-1]
    if line != "S":
        driven.append(max(abs(int(x)) for x in line[1:].split()))
check("a sharp pulsed arc still drives its outer wheel with authority",
      driven and min(driven) >= 130, f"min driven duty {min(driven) if driven else 0}")

# 4. Proportional braking must stay proportional -- the thing invariant 5 protects.
#    As the request falls, the average must fall with it, not latch at floor speed.
speeds = [average_speed(*make(), v, 0.0) for v in (0.15, 0.10, 0.05, 0.02)]
check("braking slides down instead of latching at floor speed",
      all(speeds[i] > speeds[i + 1] for i in range(len(speeds) - 1)),
      " > ".join(f"{x:.3f}" for x in speeds))
check("a braking request never averages ABOVE the floor",
      max(speeds) < floor, f"peak average {max(speeds):.3f} vs floor {floor:.3f}")

# 5. Fast enough to execute directly -> untouched by pulsing.
d, mock = make()
d.drive(0.40, 0.0)
check("a request above the floor is not pulsed",
      abs(d.executed()[0] - 0.40) < 0.03, f"{d.executed()[0]:.3f} m/s")

# 6. Zero stays zero. A stop must never become a pulse.
d, mock = make()
for _ in range(30):
    d.drive(0.0, 0.0)
check("zero stays zero", d.executed() == (0.0, 0.0), f"{d.executed()}")

# 7. Disabling pulsing restores the original stop-only behaviour.
d, mock = make(pulse_creep_duty=0)
got = average_speed(*make(pulse_creep_duty=0), 0.06, 0.0)
check("pulse duty 0 restores stop-only creep", got == 0.0, f"{got:.3f} m/s")

args = build_arg_parser().parse_args([])
check("--motor-pulse-duty defaults to the chosen 130",
      args.motor_pulse_duty == 130, f"{args.motor_pulse_duty}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
