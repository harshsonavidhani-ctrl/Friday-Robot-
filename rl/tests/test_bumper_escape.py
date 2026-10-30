"""One committed way out of a tight spot, and a gyro that lies.

Harsh, 2026-10-30: "just the tight space it confused". Pure logic: no motors.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import config                                               # noqa: E402
import body_sense                                           # noqa: E402
import escape as esc                                        # noqa: E402
from pi_deploy import RobotConfig                           # noqa: E402

cfg = RobotConfig()
fails = 0
DT = 0.1


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    fails += 0 if ok else 1


def tick(e, blocked=True, ahead=0.05, rear=1.0, can=True, side=1.0, ev=0.0, ew=0.0):
    return e.step(blocked=blocked, ahead_m=ahead, rear_min_m=rear, can_reverse=can,
                  open_side=side, exec_v=ev, exec_omega=ew, dt=DT)


# Not on a passing brake: this is what keeps the open floor untouched.
e = esc.EscapeManeuver(cfg)
out = [tick(e) for _ in range(esc.TRIGGER_TICKS - 1)]
check("a short brake does nothing", all(o is None for o in out) and not e.active)
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS - 1):
    tick(e)
tick(e, blocked=False)
check("a block that clears resets the count",
      tick(e) is None and e._blocked == 1)

# The flags must reach the loop: cfg is set AFTER the loop is built.
src = (ROOT / "friday_localnav_deploy.py").read_text()
check("the run really builds the escape from its flag",
      src.index("cfg.bumper_escape = bool") < src.index("loop.bumper_escape = _escape"))

# Back, then one turn.
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS - 1):
    tick(e)
o = tick(e, side=-1.0)
check("blocked long enough: she backs straight out",
      o is not None and o[0] < 0 and o[1] == 0.0, str(o))
seen = []
for i in range(60):
    o = tick(e, side=1.0 if i % 2 else -1.0, ev=-0.2, ew=-1.0)
    if o is None:
        break
    seen.append(o)
turns = [o for o in seen if o[1] != 0.0]
check("then a turn", bool(turns))
check("the side is chosen ONCE, whatever the LiDAR says later",
      len({math.copysign(1, o[1]) for o in turns}) == 1 and turns[0][1] < 0)
check("she never drives forward in it", all(o[0] <= 0.0 for o in seen))
check("backing comes before turning, never after",
      all(o[0] == 0.0 for o in seen[seen.index(turns[0]):]))
check("it is bounded", len(seen) <= esc.BACK_MAX_TICKS + esc.TURN_MAX_TICKS + 2)
check("it hands back", not e.active)
check("and cools down, so the navigator gets a turn",
      all(tick(e) is None for _ in range(esc.COOLDOWN_TICKS)))

# Reverse refused.
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS - 1):
    tick(e, can=False)
o = tick(e, can=False)
check("reverse refused: straight to the turn", o is not None and o[0] == 0.0 and o[1] != 0.0)
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS - 1):
    tick(e)
tick(e)
o = tick(e, can=False)
check("refused part way back: it turns, it does not sit there",
      o is not None and o[0] == 0.0 and o[1] != 0.0)
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS - 1):
    tick(e, rear=cfg.rear_stop)
o = tick(e, rear=cfg.rear_stop)
check("rear too close: no reverse", o is not None and o[0] == 0.0)

# The way clears.
e = esc.EscapeManeuver(cfg)
for _ in range(esc.TRIGGER_TICKS):
    tick(e, can=False)
tick(e, can=False, ahead=cfg.safe_distance + 0.1)
check("a clear way ends the turn at once",
      tick(e, can=False, ahead=cfg.safe_distance + 0.1) is None and not e.active)

# A gyro that lies.
config_ok = config.BODY_FAULT_S > 0


def imu(tilt, shake, tilted=False, rate=0.0):
    return {"status": "ok", "ok": True, "tilt_deg": tilt, "shake_dps": shake,
            "tilted": tilted, "rate_dps": rate, "bumps": 0}


d = body_sense.Detector(shake_lift=False, tilt_lift=False, bumps=False)
events, t = [], 100.0
for i in range(120):
    events += d.update(imu(80.0, 52.0 + (i % 3) * 0.2, tilted=True), t)
    t += 0.1
check("tilt 80, shake 52 steady (the 16:36 reading) is a fault", d.fault)
check("...so she is not lifted", not d.lifted)
check("...and not dizzy", events.count("shaken") == 0, str(events))

d = body_sense.Detector()
events, t = [], 100.0
for i in range(60):
    events += d.update(imu(10.0 + (i % 7) * 3, 20.0 + 25.0 * (i % 5)), t)
    t += 0.1
check("a real shake is not a fault", not d.fault)
check("...and she still feels it", "shaken" in events and d.lifted, str(events))

d = body_sense.Detector()
t = 100.0
for i in range(60):
    d.update(imu(80.0, 52.0, tilted=True), t)
    t += 0.1
check("still a fault", d.fault)
for i in range(30):
    d.update(imu(5.0, 0.2), t)
    t += 0.1
check("a chip that comes back is believed again", not d.fault)

d = body_sense.Detector()
t = 100.0
for i in range(50):
    d.update(imu(5.0, 0.1), t)
    t += 0.1
check("at rest is no fault", not d.fault and not d.lifted)

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
