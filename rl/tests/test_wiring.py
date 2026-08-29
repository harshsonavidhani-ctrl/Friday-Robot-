"""The RIGHT motor's leads are physically reversed on this robot.

friday/config.py has compensated for it since the chassis was glued shut:
    MOTOR_INVERT_LEFT = False
    MOTOR_INVERT_RIGHT = True
`pi_deploy.py` was written on the dev machine and defaulted BOTH to False, so every
command it sent had the right side backwards:

    commanded STRAIGHT  M d  d   -> wheels opposed  -> she SPINS
    commanded PIVOT     M d -d   -> both forward    -> she DRIVES OFF

The second one turned a pivot test into a robot heading for a wall, and a human had
to physically catch her. This suite exists so that never silently regresses.
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
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=0.75, min_duty=72, **kw)
    finally:
        pi_deploy.time.sleep = real
    d._moving = True
    d._kick_left = 0
    mock.sent.clear()
    return d, mock


def last_duties(mock):
    line = mock.sent[-1]
    if line == "S":
        return 0, 0
    return tuple(int(x) for x in line[1:].split())


print("--- motor lead inversion (right side is reversed IN HARDWARE) ---")

# 1. The default must compensate. A default of False is not a preference, it is a
#    robot that drives off when told to turn on the spot.
args = build_arg_parser().parse_args([])
check("--motor-invert-right defaults to True", args.motor_invert_right is True,
      f"{args.motor_invert_right}")
check("--motor-invert-left defaults to False", args.motor_invert_left is False,
      f"{args.motor_invert_left}")

# 2. It must still be possible to turn off, for when the leads get corrected.
off = build_arg_parser().parse_args(["--no-motor-invert-right"])
check("--no-motor-invert-right turns it off", off.motor_invert_right is False)

# 3. On the wire: a commanded PIVOT must leave the firmware duties with the SAME
#    sign, because the reversed right leads turn that into opposed rotation.
d, mock = make(invert_right=True)
d.drive_duty(150, -150)
l, r = last_duties(mock)
check("a commanded pivot sends same-sign duties", l * r > 0, f"M{l} {r}")

# 4. ...and a commanded STRAIGHT must leave them OPPOSED, for the same reason.
d, mock = make(invert_right=True)
d.drive_duty(150, 150)
l, r = last_duties(mock)
check("a commanded straight sends opposed duties", l * r < 0, f"M{l} {r}")

# 5. The same must hold through the normal m/s path, not just the bench path.
d, mock = make(invert_right=True)
d.drive(0.0, 1.2)                      # pure pivot
l, r = last_duties(mock)
check("drive(): a pivot sends same-sign duties", l * r > 0, f"M{l} {r}")

d, mock = make(invert_right=True)
d.drive(cfg.wheel_speed_max, 0.0)      # pure straight
l, r = last_duties(mock)
check("drive(): straight sends opposed duties", l * r < 0, f"M{l} {r}")

# 6. Inversion is a wiring correction and must NOT change the motion the loop
#    dead-reckons from -- otherwise the map turns the wrong way at every corner.
a, _ = make(invert_right=True)
b, _ = make(invert_right=False)
a.drive(0.0, 1.0)
b.drive(0.0, 1.0)
check("inversion does not change the executed twist",
      abs(a.executed()[1] - b.executed()[1]) < 1e-6,
      f"{a.executed()[1]:.3f} vs {b.executed()[1]:.3f}")

# 7. The duty cap is the approved one for this L298N.
check("--motor-max-duty defaults to the approved 0.75",
      abs(args.motor_max_duty - 0.75) < 1e-9, f"{args.motor_max_duty}")


print("--- pivot stiction boost (APPROVED exemption from the max_duty cap) ---")

# HANDOFF section 4 invariant 4 caps duty at max_duty "including the kickstart
# boost". Pivots are exempt for their first few ticks ONLY, by explicit approval,
# because at the 0.75 ceiling she hummed instead of turning. These assertions are
# the bound on that exemption.
CEIL = int(255 * 0.75)          # 191, the normal cap
KICK = int(255 * 0.90)          # 229, the pivot-only kick cap


def fresh(**kw):
    # turn_min_duty 190 is the measured break-away floor -- the realistic case.
    kw.setdefault("turn_min_duty", 190)
    d, mock = make(invert_right=True, **kw)
    d._moving = False           # standstill, so the kick arms
    d._kick_left = 0
    d._was_pivot = False
    d._last_turn_kick = -1e9
    return d, mock


# 1. A pivot may exceed the normal cap on its first tick -- that is the point.
d, mock = fresh(turn_kick_duty=0.90)
d.drive(0.0, 1.6)
l, r = last_duties(mock)
check("a pivot kick may exceed the steady cap", max(abs(l), abs(r)) > CEIL,
      f"M{l} {r} vs cap {CEIL}")

# 2. ...but never beyond the approved kick ceiling.
check("a pivot kick never exceeds the approved kick cap",
      max(abs(l), abs(r)) <= KICK, f"M{l} {r} vs kick cap {KICK}")

# 3. STRAIGHT driving keeps the original cap, kickstart included. The exemption is
#    for sideways scrub, not a general speed limit raise.
d, mock = fresh(turn_kick_duty=0.90)
d.drive(10.0, 0.0)              # absurd request -> must still clamp to CEIL
l, r = last_duties(mock)
check("straight driving is still capped at max_duty",
      max(abs(l), abs(r)) <= CEIL, f"M{l} {r} vs cap {CEIL}")

# 4. The boost is transient: once the kick ticks are spent, a sustained pivot
#    falls back to the steady cap. Sustained current is the brownout risk.
d, mock = fresh(turn_kick_duty=0.90)
for _ in range(6):
    d.drive(0.0, 1.6)
l, r = last_duties(mock)
check("a sustained pivot falls back to the steady cap",
      max(abs(l), abs(r)) <= CEIL, f"M{l} {r} after 6 ticks")

# 5. The cooldown stops an oscillating policy re-triggering the boost every tick,
#    which would turn a transient into exactly the sustained draw we are avoiding.
d, mock = fresh(turn_kick_duty=0.90, turn_kick_cooldown_s=60.0)
peak = 0
for i in range(12):
    d.drive(0.0, 1.6 if i % 2 == 0 else 0.0)   # pivot / straight / pivot / ...
    li, ri = last_duties(mock)
    peak = max(peak, abs(li), abs(ri))
check("cooldown prevents repeated boosting", peak <= KICK, f"peak {peak}")
d, mock = fresh(turn_kick_duty=0.90, turn_kick_cooldown_s=60.0)
d.drive(0.0, 1.6)                       # first pivot: boosted
d.drive(0.0, 0.0)
d._moving = True
over = 0
for _ in range(8):
    d.drive(0.0, 1.6)
    li, ri = last_duties(mock)
    if max(abs(li), abs(ri)) > CEIL:
        over += 1
check("a second pivot inside the cooldown is NOT boosted", over == 0,
      f"{over} boosted ticks")

# 6. Disabling the exemption must actually disable it.
d, mock = fresh(turn_kick_duty=0.75)
d.drive(0.0, 1.6)
l, r = last_duties(mock)
check("turn_kick_duty == max_duty disables the exemption",
      max(abs(l), abs(r)) <= CEIL, f"M{l} {r}")

# 7. The CLI default is the approved value, not something higher.
check("--motor-turn-kick-duty defaults to the approved 0.90",
      abs(args.motor_turn_kick_duty - 0.90) < 1e-9, f"{args.motor_turn_kick_duty}")
check("--motor-turn-min-duty defaults to the measured 190",
      args.motor_turn_min_duty == 190, f"{args.motor_turn_min_duty}")

# A default of 0 here is not "no opinion": it disables the stiction floor AND the
# creep cutoff, so small commands go out as small duties and the motors hum for the
# whole run while the robot sits still. It happened.
check("--motor-min-duty defaults to the measured 72",
      args.motor_min_duty == 72, f"{args.motor_min_duty}")

# The floor must be reportable in m/s, because the no-reverse fallback is gated on
# it -- a floor of 0 silently disabled that too.
d, _ = make(invert_right=True)
check("min_speed_mps() reports a real floor",
      d.min_speed_mps() > 0.0, f"{d.min_speed_mps():.3f} m/s")
check("reverse is correctly reported as unreachable",
      abs(cfg.v_min) < d.min_speed_mps(),
      f"|v_min| {abs(cfg.v_min)} vs floor {d.min_speed_mps():.3f}")

# The floor is what makes a pivot happen at all: at omega_max a pivot only maps to
# duty 112, so without the floor she is never commanded enough torque to rotate.
d, mock = fresh(turn_min_duty=190)
d._moving = True
d._kick_left = 0
d._was_pivot = True                      # already pivoting: no kick, steady state
d.drive(0.0, 1.6)
l, r = last_duties(mock)
check("a steady pivot is lifted to the break-away floor",
      min(abs(l), abs(r)) >= 190, f"M{l} {r}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
