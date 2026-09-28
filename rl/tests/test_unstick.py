"""Undo the last move (rl/unstick.py): the way out of the triple lock.

The lock (2026-09-27 22:00:44): ToF 0.09 m froze turning AND forward, the rear
gate refused reverse at 0.33 m, and she sat at (0, 0) for 2 min 40 s. These
check that the way out only ever retraces space she has just been in, never
goes forward, respects the rear LiDAR, and asks for help when it cannot.
"""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import unstick as U                                   # noqa: E402
from pi_deploy import RobotConfig                     # noqa: E402

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def make():
    return U.Unstick(robot_radius=cfg.robot_radius, v_min=cfg.v_min,
                     omega_max=cfg.omega_max,
                     tof_freeze_m=cfg.tof_pivot_min_clearance)


DT = 0.1
POSE = (1.0, 2.0, 0.0)


def lock(u, ticks, *, tof=0.09, rear=0.33, pose=POSE, t0=100.0):
    """Feed `ticks` frozen ticks; return the list of overrides and the end time."""
    out, t = [], t0
    for _ in range(ticks):
        u.record(0.0, 0.0, DT)
        out.append(u.update(frozen=True, tof_min=tof, rear_min=rear, pose=pose, now=t))
        t += DT
    return out, t


# 1. She pivoted INTO it (the 09-27 lock): the undo turns back the other way.
u = make()
for _ in range(8):                        # 0.8 s of turning right at -1.54 rad/s
    u.record(0.0, -1.54, DT)
outs, t = lock(u, U.LOCK_TICKS)
first = next((o for o in outs if o is not None), None)
check("the 09-27 lock is undone by turning back", first is not None
      and first[0] == 0.0 and first[1] > 0.0, f"{first}")
check("the undo turn lasts no longer than the turn it undoes",
      u.undo is not None and u.undo.duration_s <= 0.8 + 1e-9,
      f"{u.undo and u.undo.duration_s:.2f} s")
check("nothing happens before the lock has held LOCK_TICKS", all(
    o is None for o in outs[:-1]), f"{[o is not None for o in outs]}")

# 1b. ...and it stops as soon as the bumper has room to turn.
o = u.update(frozen=True, tof_min=cfg.tof_pivot_min_clearance + 0.05,
             rear_min=0.33, pose=POSE, now=t)
check("the undo ends once the bumper has room to turn",
      o is None and u.state == "idle", f"{u.state}")
check("an undo is never retraced (history cleared)", len(u.history) == 0)

# 2. She drove INTO it: reverse along the path, capped at 8 cm, arc negated.
u = make()
for _ in range(15):                       # 1.5 s at 0.2 m/s, gentle left arc
    u.record(0.20, 0.30, DT)
outs, t = lock(u, U.LOCK_TICKS, rear=0.40)
first = next((o for o in outs if o is not None), None)
travel = abs(cfg.v_min) * u.undo.duration_s if u.undo else 99
check("a drive is undone by reversing along it", first is not None
      and first[0] < 0 and first[1] < 0, f"{first}")
check("the reverse undo is at most 8 cm", travel <= U.UNDO_MAX_M + 1e-6,
      f"{travel * 100:.1f} cm")

# 2b. The rear LiDAR still guards it, every tick.
o = u.update(frozen=True, tof_min=0.09,
             rear_min=cfg.robot_radius + U.REAR_GUARD_EXTRA_M - 0.01,
             pose=POSE, now=t)
check("the rear LiDAR stops a reverse undo mid-way", o is None and u.state == "idle"
      and u.stats["undo_rear_stop"] == 1)

# 3. Not enough known-free space behind for the stop: no reverse, ask for help.
u = make()
for _ in range(3):                        # 3 cm forward only
    u.record(0.10, 0.0, DT)
outs, t = lock(u, U.LOCK_TICKS, rear=0.30)   # 8 cm past the body
check("no reverse when drive + rear clearance cannot hold the coast",
      all(o is None for o in outs) and u.state == "asking",
      f"state={u.state} stats={u.stats['no_move_to_undo']}")
check("asking says the words", ("unstuck_help",
                                 "I'm stuck. Please move me a little.") in u.events)
check("the stuck spot goes on the avoid list", u.avoid == [POSE[:2]])

# 4. She was reversing into it: undoing that is forward -- never.
u = make()
for _ in range(10):
    u.record(-0.15, 0.0, DT)
outs, _ = lock(u, U.LOCK_TICKS)
check("a reverse is never undone (that would be forward)",
      all(o is None for o in outs) and u.state == "asking")

# 5. A too-short turn is not retraced as a turn (it would not break stiction
#    without sweeping more than she just swept).
u = make()
u.record(0.0, 1.5, DT)
outs, _ = lock(u, U.LOCK_TICKS)
check("a 0.1 s turn is not retraced", all(o is None for o in outs)
      and u.state == "asking")

# 6. Aborts when the undo makes the bumper reading WORSE.
u = make()
for _ in range(8):
    u.record(0.0, -1.5, DT)
outs, t = lock(u, U.LOCK_TICKS, tof=0.10)
o = u.update(frozen=True, tof_min=0.10 - U.WORSE_M - 0.01, rear_min=0.33,
             pose=POSE, now=t)
check("an undo that brings the obstacle closer is stopped",
      o is None and u.stats["undo_worse"] == 1)

# 7. Two undos at the same spot, then she asks; a new spot starts again.
#    Between locks the chain moves her again (another turn into the same thing);
#    straight after an undo that did not free her there is nothing new to retrace.
u = make()
t = 100.0
for attempt in range(3):
    for _ in range(8):
        u.record(0.0, -1.5, DT)
    outs, t = lock(u, U.LOCK_TICKS, t0=t)
    while u.state == "undoing":                       # runs out, not freed
        u.record(0.0, 1.5, DT)
        u.update(frozen=True, tof_min=0.09, rear_min=0.33, pose=POSE, now=t)
        t += DT
    for _ in range(3):                                # she moves again
        u.record(0.0, -1.5, DT)
        u.update(frozen=False, tof_min=0.5, rear_min=0.33, pose=POSE, now=t)
        t += DT
check(f"after {U.MAX_TRIES} undos at one spot she asks for help",
      u.stats["undos"] == U.MAX_TRIES and u.state == "asking",
      f"undos={u.stats['undos']} state={u.state}")
said = [k for k, _ in u.events]
check("asking is said once, not every tick", said.count("unstuck_help") == 1,
      f"{said}")

# 7b. Freed (someone moved her): thanks, and the tries reset.
for _ in range(U.FREED_TICKS):
    u.update(frozen=False, tof_min=1.0, rear_min=1.0, pose=POSE, now=t)
    t += DT
check("freed while asking: thanks and carry on", u.state == "idle"
      and u.tries == 0 and said != [k for k, _ in u.events]
      and u.events[-1][0] == "unstuck_thanks")

# 7c. The request is repeated while nobody helps, but not more than every 30 s.
u = make()
outs, t = lock(u, U.LOCK_TICKS)                       # nothing to undo -> asks
outs, t = lock(u, int(10 / DT), t0=t)                 # 10 s later
n10 = [k for k, _ in u.events].count("unstuck_help")
outs, t = lock(u, int(25 / DT), t0=t)                 # 35 s after asking
n35 = [k for k, _ in u.events].count("unstuck_help")
check("the request repeats every 30 s, not every tick", n10 == 1 and n35 == 2,
      f"{n10} then {n35}")

# 8. A lock somewhere NEW is a fresh place with fresh tries.
u = make()
for _ in range(8):
    u.record(0.0, -1.5, DT)
lock(u, U.LOCK_TICKS)
u.tries = U.MAX_TRIES
for _ in range(8):
    u.record(0.0, -1.5, DT)
u.state = "idle"
outs, _ = lock(u, U.LOCK_TICKS, pose=(3.0, 2.0, 0.0))
check("a new place gets its own tries", any(o is not None for o in outs)
      and u.state == "undoing")

# 9. Avoid keys cover the spot at the planner's frontier-key resolution.
u = make()
u.avoid.append((1.0, 2.0))
keys = u.avoid_keys()
check("the stuck spot's own frontier key is avoided", (4, 8) in keys)
check("...within AVOID_RADIUS_M and no further",
      all(math.hypot(k[0] * 0.25 - 1.0, k[1] * 0.25 - 2.0) <= U.AVOID_RADIUS_M
          + 0.25 * 0.71 for k in keys), f"{len(keys)} keys")

# 10. NEVER FORWARD. Random histories, random locks, random sensor readings.
rng = random.Random(7)
worst = -1.0
for trial in range(400):
    u = make()
    t = 0.0
    for _ in range(rng.randint(0, 40)):
        u.record(rng.uniform(-0.2, 0.5), rng.uniform(-1.6, 1.6), DT)
    for _ in range(rng.randint(1, 80)):
        frozen = rng.random() < 0.85
        o = u.update(frozen=frozen, tof_min=rng.uniform(0.0, 0.4),
                     rear_min=rng.uniform(0.2, 1.0), pose=(rng.uniform(-1, 1),
                                                          rng.uniform(-1, 1), 0),
                     now=t)
        if o is not None:
            worst = max(worst, o[0])
        u.record(0.0 if o is None else o[0], 0.0 if o is None else o[1], DT)
        t += DT
check("an undo never commands forward (400 random runs)", worst <= 0.0,
      f"max v={worst:+.3f}")

# 11. Undo turns never exceed omega_max; undo reverses are exactly v_min.
ok = True
for trial in range(200):
    u = make()
    for _ in range(rng.randint(3, 40)):
        u.record(rng.choice([0.0, 0.2]), rng.uniform(-2.0, 2.0), DT)
    outs, _ = lock(u, U.LOCK_TICKS, rear=rng.uniform(0.3, 1.0))
    for o in outs:
        if o is None:
            continue
        if abs(o[1]) > cfg.omega_max + 1e-9 or (o[0] < 0 and
                                                abs(o[0] - cfg.v_min) > 1e-9):
            ok = False
check("undo speeds stay inside the robot's limits", ok)

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
