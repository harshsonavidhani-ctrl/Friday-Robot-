"""The bumper ToF looks DOWN. Its bottom rows are floor, and reading them as
obstacles is why Friday once could not drive forward at all, in any room, forever.

Every case here is a failure that actually happened on the robot. The measured
floor profile is (295, 223, 161, 120) mm for rows 4-7.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import (RobotConfig, ToFReflex, tof_obstacles_mm,
                       TOF_FLOOR_PROFILE_MM, TOF_NO_RETURN_MM)

cfg = RobotConfig()
FLOOR = TOF_FLOOR_PROFILE_MM          # rows 4..7
AHEAD_CLEAR = 1700                    # the room ahead, from the measured profile
# The ahead rows correctly report the room as a thing -- a wall at 1.7 m IS a thing,
# and the reflex only acts below tof_slow_distance. So "clear" here means nothing
# NEAR, not nothing at all.
NEAR_MM = 600                         # cfg.tof_slow_distance
fails = 0


def nearest(obs):
    return float(obs.min())


def near_cells(obs):
    return int((obs < NEAR_MM).sum())


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def frame(ahead=AHEAD_CLEAR, pitch=1.0):
    """A clear-floor frame: four rows of room, four rows of floor."""
    f = [[int(ahead)] * 8 for _ in range(4)]
    f += [[int(round(FLOOR[r] * pitch))] * 8 for r in range(4)]
    return f


print("--- ToF floor semantics (rows 4-7 are the ground) ---")

# 1. THE regression: a clear floor must not read as an obstacle.
obs, intact = tof_obstacles_mm(frame())
check("clear floor reports nothing NEAR",
      near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")
check("clear floor is intact", intact)

# 2. Pitch: the chassis noses down as the motors take up and EVERY ground row reads
#    short together. Un-normalised this read as an object at 246 mm and braked the
#    robot 0.2 s into its first move -- braking for its own acceleration.
for p in (0.88, 0.95, 1.05):
    obs, intact = tof_obstacles_mm(frame(pitch=p))
    check(f"pitched floor x{p} is still floor",
          near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")

# 3. An object standing on the floor OCCLUDES the ground rows: they read SHORT.
#    Deleting that evidence is what let her drive into something until the reflex
#    caught it at SIXTEEN MILLIMETRES.
f = frame()
for r in (4, 5):
    for c in (3, 4, 5):
        f[r][c] = 150
obs, intact = tof_obstacles_mm(f)
check("occluded ground rows ARE an obstacle",
      abs(float(obs.min()) - 150) < 1, f"nearest={float(obs.min()):.0f} mm")
check("only the occluded cells read near",
      near_cells(obs) == 6, f"{near_cells(obs)} near cells")

# 4. SPAD crosstalk: the row just above the horizon echoes the strong floor return
#    below it at ~295 mm. About 1 frame in 7. One of them braked her mid-move.
f = frame()
for c in (3, 4):
    f[3][c] = FLOOR[0]                    # 295 mm, exactly the floor's first row
obs, intact = tof_obstacles_mm(f)
check("horizon-row floor echo is rejected",
      near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")

# 5. ...but a REAL object at that range also occludes the floor, so it survives.
f = frame()
for c in (3, 4):
    f[3][c] = 290
    f[4][c] = 150                         # the same object, occluding the ground row
obs, intact = tof_obstacles_mm(f)
check("a real near object is NOT rejected as an echo",
      float(obs.min()) < 200, f"nearest={float(obs.min()):.0f} mm")

# 6. A floor that stopped reading is what a CLIFF looks like: cannot-see, never clear.
f = frame()
f[7] = [0] * 8
_, intact = tof_obstacles_mm(f)
check("a floor that stopped reading is NOT intact", not intact)

f = frame()
f[6] = [FLOOR[2] + 400] * 8               # reads much further than floor: a drop
_, intact = tof_obstacles_mm(f)
check("a drop-off is NOT intact", not intact)

# 7. End to end through the reflex -- the layer that actually stops the wheels.
reflex = ToFReflex(cfg)
obs, _ = tof_obstacles_mm(frame())
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("reflex does NOT fire on clear floor", not stop and scale == 1.0,
      f"scale={scale:.2f} stop={stop}")

f = frame()
for r in (4, 5):
    for c in (2, 3, 4, 5):
        f[r][c] = 200                     # an object at 0.20 m, under the 0.30 stop
obs, _ = tof_obstacles_mm(f)
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("reflex FIRES on a real close object", stop and scale == 0.0,
      f"scale={scale:.2f} stop={stop}")

# 8. Speckle immunity must survive the floor logic: one hot cell is not a stop.
f = frame()
f[2][4] = 120
obs, _ = tof_obstacles_mm(f)
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("a single hot cell does not stop the robot", not stop, f"stop={stop}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
