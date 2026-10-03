"""Quieter motors (Harsh, 2026-09-28: "the motors are like making too noise").

MEASURED in the 23:24 run (190 s): 252 motor starts, 159 after a single stopped
tick; ~300 ticks at kick duty because the standstill kick fired after EVERY
stop; 51 powered direction reversals (plugging). These check the three fixes
in FridayMotorDriver and, above all, that none of them delays a stop or adds
energy to the wheels.
"""
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy                                        # noqa: E402
from pi_deploy import FridayMotorDriver, RobotConfig, build_arg_parser  # noqa: E402


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


class Clock:
    t = 1000.0


def make(**kw):
    mock = MockArduino()
    sys.modules["serial"] = types.SimpleNamespace(Serial=lambda *a, **k: mock)
    real = pi_deploy.time.sleep
    pi_deploy.time.sleep = lambda s: None
    try:
        kw.setdefault("pulse_creep_duty", 130)
        d = FridayMotorDriver(cfg, "/dev/mock", max_duty=0.85, min_duty=80,
                              turn_min_duty=190, turn_kick_duty=0.90, **kw)
    finally:
        pi_deploy.time.sleep = real
    pi_deploy.time.monotonic = lambda: Clock.t
    mock.sent.clear()
    return d, mock


def tick(d, v, w):
    Clock.t += 0.1
    d.drive(v, w)


def wire(mock):
    line = mock.sent[-1]
    if line == "S":
        return (0, 0)
    m = re.match(r"M(-?\d+) (-?\d+)", line)
    return (int(m.group(1)), int(m.group(2)))


NEW = dict(pulse_burst_ticks=3, kick_rest_s=0.3, plug_guard=True)
real_monotonic = pi_deploy.time.monotonic
print("--- quieter motors ---")

# 1. Creep pulses come in bursts: same average speed, a third of the starts.
# (kick_ticks=0 on both: the old driver kicked every pulse, which inflated its
# average to 0.080 m/s for a 0.05 request -- the kick is tested separately below)
for label, kw in (("old", {}), ("new", NEW)):
    d, mock = make(kick_ticks=0, **kw)
    starts, total, prev_on = 0, 0.0, False
    for _ in range(300):
        tick(d, 0.05, 0.0)                       # well below the floor
        on = mock.sent[-1] != "S"
        starts += on and not prev_on
        prev_on = on
        total += d.executed()[0]
    if label == "old":
        old_starts, old_avg = starts, total / 300
    else:
        new_starts, new_avg = starts, total / 300
check("creep bursts keep the average speed",
      abs(new_avg - old_avg) < 0.004 and abs(new_avg - 0.05) < 0.005, f"old {old_avg:.3f} new {new_avg:.3f} m/s")
check("...with at most half the motor starts", new_starts * 2 <= old_starts,
      f"{old_starts} -> {new_starts}")

# 2. A stop request mid-burst stops on THAT tick.
d, mock = make(**NEW)
for _ in range(30):
    tick(d, 0.05, 0.0)
    if mock.sent[-1] != "S" and d._burst_left > 0:
        break
tick(d, 0.0, 0.0)
check("a stop mid-burst is a stop at once", mock.sent[-1] == "S", mock.sent[-3:])
d.emergency_stop()
check("emergency_stop mid-burst sends S", mock.sent[-1] == "S")

# 3. The standstill kick needs a real standstill.
d, mock = make(**NEW)
tick(d, 0.3, 0.0)
first = max(map(abs, wire(mock)))
tick(d, 0.3, 0.0)
tick(d, 0.3, 0.0)
steady = max(map(abs, wire(mock)))
check("a start from rest is still kicked", first > steady, f"{first} vs {steady}")
tick(d, 0.0, 0.0)                                # one-tick coast
tick(d, 0.3, 0.0)
after_blip = max(map(abs, wire(mock)))
check("no kick after a one-tick coast", after_blip == steady,
      f"{after_blip} vs steady {steady}")
for _ in range(4):
    tick(d, 0.0, 0.0)                            # 0.4 s at rest
tick(d, 0.3, 0.0)
check("kicked again after a real rest", max(map(abs, wire(mock))) > steady)

# 3b. The old behaviour is still there at kick_rest_s=0.
d, mock = make()
tick(d, 0.3, 0.0); tick(d, 0.3, 0.0); tick(d, 0.3, 0.0)
steady_old = max(map(abs, wire(mock)))
tick(d, 0.0, 0.0); tick(d, 0.3, 0.0)
check("kick_rest_s=0 is the old kick-after-every-stop",
      max(map(abs, wire(mock))) > steady_old)

# 4. No plugging: forward -> reverse coasts a tick first.
d, mock = make(**NEW)
for _ in range(3):
    tick(d, 0.3, 0.0)
tick(d, -0.15, 0.0)
check("forward straight to reverse coasts one tick", mock.sent[-1] == "S",
      mock.sent[-2:])
tick(d, -0.15, 0.0)
l, r = wire(mock)
check("...then reverses", l < 0 or r < 0, f"{l} {r}")
check("the coast is counted", d.noise_stats["plug_coasts"] == 1)

# 4b. A pivot swapping direction coasts too.
d, mock = make(**NEW)
for _ in range(3):
    tick(d, 0.0, 1.5)
tick(d, 0.0, -1.5)
check("pivot left straight to pivot right coasts one tick", mock.sent[-1] == "S")

# 4c. Same direction, different speed: no coast.
d, mock = make(**NEW)
tick(d, 0.3, 0.0)
tick(d, 0.2, 0.0)
check("slowing in the same direction is not a coast", mock.sent[-1] != "S")

# 5. NOTHING HERE ADDS ENERGY: for random command streams, every new-driver tick
#    is either a stop or no stronger than the old driver would send at most.
import random                                         # noqa: E402
rng = random.Random(3)
ok, worst = True, 0
cap = int(255 * 0.90)
for trial in range(100):
    d, mock = make(**NEW)
    for _ in range(60):
        v = rng.choice([0.0, 0.0, 0.03, 0.08, 0.2, 0.4, -0.15])
        w = rng.choice([0.0, 0.0, 0.3, 1.0, -1.0, 1.6])
        tick(d, v, w)
        m = max(map(abs, wire(mock)))
        worst = max(worst, m)
        if m > cap:
            ok = False
check("never above the kick ceiling", ok, f"peak {worst} vs cap {cap}")

d, mock = make(**NEW)
ok = True
for _ in range(500):
    v = rng.choice([0.0, 0.05, 0.3, -0.15])
    w = rng.choice([0.0, 1.2, -1.2])
    tick(d, v, w)
    if v == 0.0 and w == 0.0 and mock.sent[-1] != "S":
        ok = False
check("a zero request is always a stop on the same tick (500 random ticks)", ok)

# 6. The CLI defaults are the quiet ones, and both constructions pass them.
args = build_arg_parser().parse_args([])
check("CLI defaults: burst 3, kick rest 0.3 s, plug guard on",
      args.motor_pulse_burst == 3 and abs(args.motor_kick_rest - 0.3) < 1e-9
      and args.motor_plug_guard is True)
src = Path(pi_deploy.__file__).read_text(encoding="utf-8-sig")
check("both driver constructions pass the quiet flags",
      src.count("plug_guard=args.motor_plug_guard") == 2)

# --- steady turns (Harsh, 2026-10-04: "it should be smooth") -------------------
print("--- steady turns ---")
STEADY = dict(NEW, steady_pivots=True, pivot_hold_ticks=3)
w_turn = 0.6        # a turn in place well above turn_min_omega, slow at each wheel
for label, kw in (("quiet", NEW), ("steady", STEADY)):
    d, mock = make(**kw)
    seen = []
    for _ in range(20):
        tick(d, 0.0, w_turn)
        seen.append(mock.sent[-1])
    stops = sum(1 for s in seen[3:] if s == "S")
    if label == "quiet":
        quiet_stops = stops
    else:
        steady_stops, steady_seen = stops, seen
check("a slow turn in place is pulsed on/off by the creep pulse without steady turns",
      quiet_stops > 0, f"{quiet_stops} stops in 17 ticks")
check("...and runs STEADILY (no stop ticks) with them", steady_stops == 0,
      str(steady_seen[3:8]))
d, mock = make(**STEADY)
for _ in range(5):
    tick(d, 0.0, w_turn)
tick(d, 0.0, 0.0)
check("a stop request (0, 0) mid-turn is honoured on the tick it arrives",
      mock.sent[-1] == "S")
d, mock = make(**STEADY)
for _ in range(5):
    tick(d, 0.0, w_turn)
tick(d, 0.0, 0.05)                              # same direction, under turn_min_omega
held = mock.sent[-1] != "S"
for _ in range(4):
    tick(d, 0.0, 0.05)
check("a same-direction dip under the floor is held for the hold ticks, then dropped",
      held and mock.sent[-1] == "S" and d.noise_stats["pivot_holds"] == 3,
      f"held={held} last={mock.sent[-1]} holds={d.noise_stats['pivot_holds']}")
d, mock = make(**STEADY)
for _ in range(5):
    tick(d, 0.0, w_turn)
tick(d, 0.0, -0.05)                             # a tiny turn the OTHER way
check("...never into the other direction (that is a stop, then a new turn)",
      mock.sent[-1] == "S")
d, mock = make(**STEADY)
starts, prev_on = 0, False
for _ in range(60):
    tick(d, 0.05, 0.0)                          # straight creep: braking near a wall
    on = mock.sent[-1] != "S"
    starts += on and not prev_on
    prev_on = on
check("forward creep is still pulsed (the governor's braking never becomes a lurch)",
      starts >= 3, f"{starts} starts")
d, mock = make(**NEW)
check("off unless asked for (the parser default)",
      build_arg_parser().parse_args([]).motor_steady_pivots is False
      and d.steady_pivots is False)
src = Path(pi_deploy.__file__).read_text()
check("both FridayMotorDriver constructions pass the switch through",
      src.count("steady_pivots=args.motor_steady_pivots") == 2)

print("--- start debounce ---")
DEB = dict(STEADY, start_debounce_ticks=2)
d, mock = make(**DEB)
tick(d, 0.3, 0.0)
first = mock.sent[-1]
tick(d, 0.3, 0.0)
check("from a standstill: the first tick of 'go' waits, the second goes",
      first == "S" and mock.sent[-1] != "S", f"{first} then {mock.sent[-1]}")
tick(d, 0.0, 0.0)
check("a stop is never delayed", mock.sent[-1] == "S")
tick(d, 0.3, 0.0)
d.emergency_stop()
check("...nor the emergency stop", mock.sent[-1] == "S")
d, mock = make(**DEB)
tick(d, 0.3, 0.0)
tick(d, 0.3, 0.0)
tick(d, 0.3, 0.0)
check("once moving, motion is never held back", mock.sent[-1] != "S"
      and d.noise_stats["starts_debounced"] == 1)
check("off unless asked for (the parser default 1)",
      build_arg_parser().parse_args([]).motor_start_debounce == 1)

pi_deploy.time.monotonic = real_monotonic
print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
